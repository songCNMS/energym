import sys
sys.path.insert(0, "./")

import numpy as np
import random
import torch
from torch import nn
from nn_builder.pytorch.NN import NN
from preference_data import sample_preferences, preference_per_round, num_workers, perference_pairs_per_sample
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import pickle
from buildings_factory import *
import multiprocessing as mp


class RewardNet(nn.Module):
    def __init__(self, input_dim, seed=0):
        super(RewardNet, self).__init__()
        self.linear_relu_stack = NN(input_dim=input_dim, 
                      layers_info= [256, 256, 256, 1],
                      output_activation="Sigmoid",
                      batch_norm=False, dropout=0.2,
                      random_seed=seed,
                      hidden_activations=['LeakyReLU', 'LeakyReLU', 'LeakyReLU', 'LeakyReLU'], 
                      initialiser="Xavier")
        
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
    def __init__(self, round, traj_idx, building_name, min_kpis, max_kpis, parent_loc, traj):
        self.round = round
        self.traj_idx = traj_idx
        if os.path.exists(f'{parent_loc}/data/offline_data/{building_name}/preferences_data/{traj}/preference_data_{round}_{traj_idx}.pkl'):
            print("loading ", f'{parent_loc}/data/offline_data/{building_name}/preferences_data/{traj}/preference_data_{round}_{traj_idx}.pkl')
            with open(f'{parent_loc}/data/offline_data/{building_name}/preferences_data/{traj}/preference_data_{round}_{traj_idx}.pkl', 'rb') as f:
                self.raw_data = np.load(f, allow_pickle=True)
        else:
            with open(f'{parent_loc}/data/offline_data/{building_name}/traj_data/{round}_{traj_idx}.pkl', 'rb') as f:
                trajectories = pickle.load(f)
                trajectory1, trajectory2 = trajectories[0], trajectories[1]
            preference_pairs1 = sample_preferences(trajectory1, trajectory2, min_kpis, max_kpis, num_preferences=perference_pairs_per_sample, traj_list=[traj])
            preference_pairs2 = sample_preferences(trajectory1, trajectory1, min_kpis, max_kpis, num_preferences=perference_pairs_per_sample, traj_list=[traj])
            preference_pairs3 = sample_preferences(trajectory2, trajectory2, min_kpis, max_kpis, num_preferences=perference_pairs_per_sample, traj_list=[traj])
            self.raw_data = np.array(preference_pairs1[0] + preference_pairs2[0] + preference_pairs3[0])
        self.data = torch.from_numpy(self.raw_data[:, :-2]).to(torch.float)
        self.labels = torch.from_numpy(self.raw_data[:, -2:]).to(torch.float)
        
    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        feature = self.data[idx, :]
        label = self.labels[idx, :]
        return feature, label
    
def train_loop(building_name, min_kpis, max_kpis, models, loss_fn, optimizers, round_list, parent_loc, traj, device):
    total_size = 0
    if not isinstance(models, list): models = [models]
    if not isinstance(optimizers, list): optimizers = [optimizers]
    num_models = len(models)
    total_losses = [0.0]*num_models
    
    for round in round_list:
        for traj_idx in range(preference_per_round):
            if os.path.exists(f'{parent_loc}/data/offline_data/{building_name}/preferences_data/{traj}/preference_data_{round}_{traj_idx}.pkl'):
                if not os.path.exists(f'{parent_loc}/data/offline_data/{building_name}/traj_data/{round}_{traj_idx}.pkl'): continue
            training_dataset = PreferencDataset(round, traj_idx, building_name, min_kpis, max_kpis, parent_loc, traj)
            # try: training_dataset = PreferencDataset(round, traj_idx, building_name, min_kpis, max_kpis, parent_loc, traj)
            # except: continue
            dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
            size = len(dataloader.dataset)
            # total_size += size
            for batch, (X, y) in enumerate(dataloader):
                # Compute prediction and loss
                num_samples = X.size(0)
                X = X.reshape(num_samples, 2, -1)
                X_left = X[:, 0, :].reshape(num_samples, traj, -1)
                X_right = X[:, 1, :].reshape(num_samples, traj, -1)
                preds = [torch.zeros(X.size(0), 2, requires_grad=True).to(device) for _ in range(num_models)]
                for i in range(traj):
                    model_in = torch.cat((X_left[:, i, :], X_right[:, i, :]), axis=1).to(device)
                    model_outs = [model(model_in) for model in models]
                    preds = [pred + model_out for pred, model_out in zip(preds, model_outs)]
                preds = [pred / traj for pred in preds]
                y = y.to(device)
                losses = [loss_fn(pred, y) for pred in preds]
                # Backpropagation
                for i in range(num_models): 
                    optimizers[i].zero_grad()
                    losses[i].backward()
                    optimizers[i].step()
                losses = [loss.cpu().item() for loss in losses]
                total_losses = [total_losses[i]+losses[i] for i in range(num_models)]
                total_size += 1
                if (batch % 100 == 0):
                    current = batch * batch_size
                    print(f"round: {round}, traj_idx: {traj_idx}, loss: {losses}  [{current:>5d}/{size:>5d}]")
                    # print("pred: ", preds, "y: ", y)
    return [total_loss / total_size for total_loss in total_losses]

def test_loop(building_name, min_kpis, max_kpis, models, loss_fn, round_list, parent_loc, traj, device):
    total_num_batches = 0
    total_samples = 0
    if not isinstance(models, list): models = [models]
    num_models = len(models)
    test_losses, corrects = [0]*num_models, [0]*num_models
    for model in models: model.eval()
    for round in round_list:
        for traj_idx in range(preference_per_round):
            if os.path.exists(f'{parent_loc}/data/offline_data/{building_name}/preferences_data/{traj}/preference_data_{round}_{traj_idx}.pkl'):
                if not os.path.exists(f'{parent_loc}/data/offline_data/{building_name}/traj_data/{round}_{traj_idx}.pkl'): continue
            try: testing_dataset = PreferencDataset(round, traj_idx, building_name, min_kpis, max_kpis, parent_loc, traj)
            except: continue
            dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=False)
            with torch.no_grad():
                for X, y in dataloader:
                    y = y.to(device)
                    num_samples = X.size(0)
                    X = X.reshape(num_samples, 2, -1)
                    X_left = X[:, 0, :].reshape(num_samples, traj, -1)
                    X_right = X[:, 1, :].reshape(num_samples, traj, -1)
                    preds = [torch.zeros(X.size(0), 2, requires_grad=True).to(device) for _ in range(num_models)]
                    for i in range(traj):
                        model_in = torch.cat((X_left[:, i, :], X_right[:, i, :]), axis=1).to(device)
                        model_outs = [model(model_in) for model in models]
                        preds = [pred + model_out for pred, model_out in zip(preds, model_outs)]
                    test_losses = [loss_fn(pred, y).cpu().item()+loss for loss,pred in zip(test_losses, preds)]
                    corrects = [(pred.argmax(1) == y.argmax(1)).type(torch.float).sum().cpu().item()+correct for correct,pred in zip(corrects, preds)]
                    total_num_batches += 1
                    total_samples += num_samples
    for model in models: model.train()
    corrects = [correct*100/total_samples for correct in corrects]
    test_losses = [test_loss/total_num_batches for test_loss in test_losses]
    print(f"Test Error: \n Accuracy: {corrects}, Avg loss: {test_losses} \n")
    return test_losses, corrects
    


def run_train(i, input_dim, parent_loc, building_name, min_kpis, max_kpis, traj):
    device = "cpu"
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device = "cuda:%i"%(i%device_count)
    train_round_list = list(range(num_workers))
    train_round_list.remove(i)        
    eval_round_list = [i]
    epochs = 50
    learning_rate = 0.001
    loss_fn = preference_loss
    model = RewardNet(input_dim, seed=i).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_list = []
    test_loss_list = []
    correct_list = []
    model_loc = f"{parent_loc}/data/models/{building_name}/reward_model/{traj}/"
    os.makedirs(model_loc, exist_ok=True)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        total_loss = train_loop(building_name, min_kpis, max_kpis, model, loss_fn, optimizer, train_round_list, parent_loc, traj, device)
        loss_list.append(total_loss[0])
        fig, axs = plt.subplots(3, 1)
        axs[0].plot(loss_list)
        test_loss, correct = test_loop(building_name, min_kpis, max_kpis, model, loss_fn, eval_round_list, parent_loc, traj, device)
        test_loss, correct = test_loss[0], correct[0]
        test_loss_list.append(test_loss)
        correct_list.append(correct)
        axs[1].plot(test_loss_list)
        axs[2].plot(correct_list)
        plt.savefig(f"{model_loc}/reward_model_cost_{i}.png")
        if np.min(test_loss_list) == test_loss: 
            torch.save({'epoch': t,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': np.mean(loss_list),
                        "eval_loss": np.mean(test_loss_list)
                        }, f"{model_loc}/reward_model_best_{i}.pkl")
    print(f"Round {i} done!")
    
    
import matplotlib.pyplot as plt
from buildings_factory import *
from energym.wrappers.rl_wrapper import StableBaselinesRLWrapper

ensemble_num = 4
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
parser.add_argument('--traj', type=str, help='traj. len', default="1")

input_dim_dict={"OfficesThermostat-v0": 53,
"MixedUseFanFCU-v0": 38,
"SeminarcenterThermostat-v0": 66,
"SeminarcenterFull-v0": 67,
"SimpleHouseRad-v0": 21,
"SimpleHouseRSla-v0": 22,
"SwissHouseRSlaW2W-v0": 22,
"SwissHouseRSlaTank-v0": 18}


if __name__ == "__main__":
    mp.set_start_method("spawn")
    args = parser.parse_args()
    is_remote = args.amlt
    parent_loc = (os.environ['AMLT_DATA_DIR'] if is_remote else "./")
    building_name = args.building
    min_kpis, max_kpis, min_outputs, max_outputs = collect_baseline_kpi(building_name, args.amlt)
    # env_rl = StableBaselinesRLWrapper(building_name, min_kpis, max_kpis, min_outputs, max_outputs, reward_func)
    # input_dim = env_rl.observation_space.shape[0]
    input_dim = input_dim_dict[building_name]
    
    # run_train(7, input_dim, parent_loc, building_name)
    # for i in range(ensemble_num): run_train(i, input_dim, parent_loc, building_name)
    jobs = []
    for traj in [int(c) for c in args.traj.split(",")]:
        for i in range(ensemble_num):
            p = mp.Process(target=run_train, args=(i, input_dim, parent_loc, building_name, min_kpis, max_kpis, traj))
            jobs.append(p)
            p.start()
    for proc in jobs:
        proc.join()


