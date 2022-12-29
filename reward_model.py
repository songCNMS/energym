import sys
sys.path.insert(0, "./")

import numpy as np
import random
import torch
from torch import nn
from nn_builder.pytorch.NN import NN
from preference_data import sample_preferences, preference_per_round, len_traj, num_workers
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pickle
from buildings_factory import *
import multiprocessing as mp


class RewardNet(nn.Module):
    def __init__(self, input_dim):
        super(RewardNet, self).__init__()
        self.linear_relu_stack = NN(input_dim=input_dim, 
                      layers_info= [256, 256, 256, 1],
                      output_activation="sigmoid",
                      batch_norm=False, dropout=0.2,
                      hidden_activations=['LeakyReLU', 'LeakyReLU', 'LeakyReLU', 'LeakyReLU'], initialiser="Xavier")
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight.data, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                module.bias.data.zero_()
        
    def get_reward(self, x):
        x_out = self.linear_relu_stack(x)
        return 10.0*x_out

    def forward(self, x):
        num_sample = x.size(0)
        x_in = x.reshape(num_sample, 2, -1)
        logits1 = self.get_reward(x_in[:, 0, :])
        logits2 = self.get_reward(x_in[:, 1, :])
        logits = torch.cat((logits1, logits2), axis=1)
        return logits


def preference_loss(outputs, labels):
    # out_logsumexp = torch.logsumexp(outputs, axis=1, keepdim=True)
    # out_logsumexp = torch.cat((out_logsumexp, out_logsumexp), axis=1)
    # out = - outputs + out_logsumexp
    out = -torch.log(torch.nn.Softmax(dim=1)(outputs))
    return torch.mul(out, labels).sum(axis=1).mean()


class PreferencDataset(Dataset):
    def __init__(self, round, traj_idx, building_name, parent_loc):
        self.round = round
        self.traj_idx = traj_idx
        with open(f'{parent_loc}/data/offline_data/{building_name}/preferences_data/{len_traj}/preference_data_{round}_{traj_idx}.pkl', 'rb') as f:
            _raw_data = np.load(f, allow_pickle=True)
            self.raw_data = _raw_data[_raw_data[:, -1] != 0.5, :]
        self.data = torch.from_numpy(self.raw_data[:, :-2]).to(torch.float)
        self.labels = torch.from_numpy(self.raw_data[:, -2:]).to(torch.float)
        
    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        feature = self.data[idx, :]
        label = self.labels[idx, :]
        return feature, label
    
def train_loop(building_name, model, loss_fn, optimizer, round_list, parent_loc):
    total_loss = 0.0
    total_size = 0
    for round in round_list:
        for traj_idx in range(preference_per_round):
            if not os.path.exists(f'{parent_loc}/data/offline_data/{building_name}/preferences_data/{len_traj}/preference_data_{round}_{traj_idx}.pkl'): continue
            training_dataset = PreferencDataset(round, traj_idx, building_name, parent_loc)
            dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
            size = len(dataloader.dataset)
            # total_size += size
            for batch, (X, y) in enumerate(dataloader):
                # Compute prediction and loss
                num_samples = X.size(0)
                X = X.reshape(num_samples, 2, -1)
                X_left = X[:, 0, :].reshape(num_samples, len_traj, -1)
                X_right = X[:, 1, :].reshape(num_samples, len_traj, -1)
                pred = torch.zeros(X.size(0), 2, requires_grad=True).to(device)
                for i in range(len_traj):
                    model_in = torch.cat((X_left[:, i, :], X_right[:, i, :]), axis=1).to(device)
                    model_out = model(model_in)
                    pred = pred + model_out
                pred /= len_traj
                loss = loss_fn(pred, y.to(device))
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()
                total_size += 1
                if (batch % 10 == 0):
                    loss, current = loss.cpu().item(), batch * batch_size
                    print(f"round: {round}, traj_idx: {traj_idx}, loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return total_loss / total_size

def test_loop(building_name, model, loss_fn, round_list, parent_loc):
    total_num_batches = 0
    test_loss, correct = 0, 0
    model.eval()
    for round in round_list:
        for traj_idx in range(preference_per_round):
            if not os.path.exists(f'{parent_loc}/data/offline_data/{building_name}/preferences_data/{len_traj}/preference_data_{round}_{traj_idx}.pkl'): continue
            testing_dataset = PreferencDataset(round, traj_idx, building_name, parent_loc)
            dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                for X, y in dataloader:
                    y = y.to(device)
                    num_samples = X.size(0)
                    X = X.reshape(num_samples, 2, -1)
                    X_left = X[:, 0, :].reshape(num_samples, len_traj, -1)
                    X_right = X[:, 1, :].reshape(num_samples, len_traj, -1)
                    pred = torch.zeros(X.size(0), 2, requires_grad=True).to(device)
                    for i in range(len_traj):
                        model_in = torch.cat((X_left[:, i, :], X_right[:, i, :]), axis=1).to(device)
                        model_out = model(model_in)
                        pred = pred + model_out
                    test_loss += loss_fn(pred, y).cpu().item()
                    correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().cpu().item()
                    total_num_batches += 1
    model.train()
    correct /= (total_num_batches*batch_size)
    test_loss /= total_num_batches
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct
    
    
import matplotlib.pyplot as plt
from buildings_factory import *
from energym.wrappers.rl_wrapper import StableBaselinesRLWrapper


ensemble_num = 8
batch_size = 1024
# device = ("cuda" if torch.cuda.is_available() else "cpu")
# print(device)


# buildings_list = ["ApartmentsThermal-v0", "ApartmentsGrid-v0", "Apartments2Thermal-v0",
#                   "Apartments2Grid-v0", "OfficesThermostat-v0", "MixedUseFanFCU-v0",
#                   "SeminarcenterThermostat-v0", "SeminarcenterFull-v0", "SimpleHouseRad-v0",
#                   "SimpleHouseRSla-v0", "SwissHouseRSlaW2W-v0", "SwissHouseRSlaTank-v0"] 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--amlt', action='store_true', help="remote execution on amlt")
parser.add_argument('--building', type=str, help='building name', required=True)
parser.add_argument('--device', type=str, help='device', default="cuda:0")

def run_train(i, input_dim, parent_loc, building_name):
    train_round_list = list(range(num_workers))
    train_round_list.remove(i)        
    eval_round_list = [i]
    epochs = 50
    learning_rate = 0.001
    loss_fn = preference_loss
    model = RewardNet(input_dim).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_list = []
    test_loss_list = []
    correct_list = []
    model_loc = f"{parent_loc}/data/models/{building_name}/reward_model/"
    os.makedirs(model_loc, exist_ok=True)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        total_loss = train_loop(building_name, model, loss_fn, optimizer, train_round_list, parent_loc)
        loss_list.append(total_loss)
        fig, axs = plt.subplots(3, 1)
        axs[0].plot(loss_list)
        test_loss, correct = test_loop(building_name, model, loss_fn, eval_round_list, parent_loc)
        test_loss_list.append(test_loss)
        correct_list.append(correct)
        axs[1].plot(test_loss_list)
        axs[2].plot(correct_list)
        plt.savefig(f"{model_loc}/reward_model_cost_{i}.png")
        if np.min(test_loss_list) == test_loss: torch.save(model.state_dict(), f"{model_loc}/reward_model_best_{i}.pkl")
    print(f"Round {i} done!")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    args = parser.parse_args()
    is_remote = args.amlt
    parent_loc = (os.environ['AMLT_DATA_DIR'] if is_remote else "./")
    building_name = args.building
    min_kpis, max_kpis, min_outputs, max_outputs = collect_baseline_kpi(building_name)
    # building_idx = buildings_list.index(building_name)
    # env = get_env(building_name)
    # inputs = get_inputs(building_name, env)
    # default_control = default_controls[building_idx]
    env_rl = StableBaselinesRLWrapper(building_name, min_kpis, max_kpis, min_outputs, max_outputs, reward_func)
    input_dim = env_rl.observation_space.shape[0]
    
    # run_train(7, input_dim, parent_loc, building_name)
    # for i in range(ensemble_num): run_train(i, input_dim, parent_loc, building_name)
    jobs = []
    for i in range(ensemble_num):
        p = mp.Process(target=run_train, args=(i, input_dim, parent_loc, building_name))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()


