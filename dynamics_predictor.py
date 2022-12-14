import sys
sys.path.insert(0, "./")

import pickle
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


class DynamicsPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DynamicsPredictor, self).__init__()
        self.linear_relu_stack = NN(input_dim=input_dim, 
                      layers_info= [512, 512, 512, output_dim],
                      output_activation="none",
                      batch_norm=False, dropout=0.0,
                      hidden_activations=['tanh', 'relu', 'relu', 'relu'], initialiser="Xavier", random_seed=43)
        
    def forward(self, x):
        x_out = self.linear_relu_stack(x)
        return x_out

def predictor_loss(outputs, labels):
    return nn.L1Loss()(outputs, labels)


class DynamicsDataset(Dataset):
    def __init__(self, round, traj_idx, building_name):
        self.round = round
        self.traj_idx = traj_idx
        # state, actions, reward, kpis, next_state
        with open(f'{parent_loc}/data/offline_data/{building_name}/traj_data/{round}_{traj_idx}.pkl', 'rb') as f:
            raw_data = pickle.load(f)
        num_samples = (len(raw_data)*len(raw_data[0])) // 4
        state_dim = len(raw_data[0][0])
        action_dim = len(raw_data[0][1])
        input_data = np.zeros((num_samples, state_dim+action_dim))
        output_data = np.zeros((num_samples, state_dim))
        k = 0
        for traj in raw_data:
            for i in range(len(traj) // 4):
                state, actions, next_state = traj[4*i], traj[4*i+1], traj[4*i+4]
                input_data[k, :] = np.concatenate((state, actions))
                output_data[k, :] = next_state
                k += 1
        self.data = torch.from_numpy(input_data).to(torch.float)
        self.labels = torch.from_numpy(output_data).to(torch.float)
        
    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        feature = self.data[idx, :]
        label = self.labels[idx, :]
        return feature, label
    
    
def train_loop(building_name, model, loss_fn, optimizer, round_list):
    total_loss = 0.0
    total_size = 0
    for round in round_list:
        for traj_idx in range(preference_per_round):
            if not os.path.exists(f'{parent_loc}/data/offline_data/{building_name}/traj_data/{round}_{traj_idx}.pkl'): continue
            training_dataset = DynamicsDataset(round, traj_idx, building_name)
            dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
            size = len(dataloader.dataset)
            for batch, (X, y) in enumerate(dataloader):
                # Compute prediction and loss
                pred = model(X.to(device))
                loss = loss_fn(pred, y.to(device))
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()
                total_size += 1
                if (not is_remote) and (batch % 10 == 0):
                    print(pred[:5, :], y[:5, :])
                    loss, current = loss.cpu().item(), batch * len(X)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return total_loss / total_size

def test_loop(building_name, model, loss_fn, round_list):
    total_num_batches = 0
    test_loss, correct = 0, 0
    total_size = 0
    model.eval()
    for round in round_list:
        for traj_idx in range(preference_per_round):
            if not os.path.exists(f'{parent_loc}/data/offline_data/{building_name}/traj_data/{round}_{traj_idx}.pkl'): continue
            training_dataset = DynamicsDataset(round, traj_idx, building_name)
            dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)
            total_size += len(dataloader)
            with torch.no_grad():
                for X, y in dataloader:
                    y = y.to(device)
                    pred = model(X.to(device))
                    test_loss += loss_fn(pred, y).cpu().item()
                    total_num_batches += 1
    model.train()
    test_loss /= total_num_batches
    print(f"Test Avg loss: {test_loss:>8f} \n")
    return test_loss
    
    
import matplotlib.pyplot as plt
from buildings_factory import *
from energym.wrappers.rl_wrapper import StableBaselinesRLWrapper

batch_size = 256
device = ("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# buildings_list = ["ApartmentsThermal-v0", "ApartmentsGrid-v0", "Apartments2Thermal-v0",
#                   "Apartments2Grid-v0", "OfficesThermostat-v0", "MixedUseFanFCU-v0",
#                   "SeminarcenterThermostat-v0", "SeminarcenterFull-v0", "SimpleHouseRad-v0",
#                   "SimpleHouseRSla-v0", "SwissHouseRSlaW2W-v0", "SwissHouseRSlaTank-v0"] 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--amlt', action='store_true', help="remote execution on amlt")
parser.add_argument('--building', type=str, help='building name', required=True)


if __name__ == "__main__":
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
    input_dim = env_rl.observation_space.shape[0] + env_rl.action_space.shape[0]
    output_dim = env_rl.observation_space.shape[0]
    model = DynamicsPredictor(input_dim, output_dim).to(device)
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = predictor_loss


    train_round_list = list(range(num_workers-1))
    eval_round_list = list(range(num_workers-1, num_workers))
    # train_round_list = [1, 7]
    # eval_round_list = [0]

    epochs = 100
    loss_list = []
    test_loss_list = []
    model_loc = f"{parent_loc}/data/models/{building_name}/dynamics_model/"
    os.makedirs(model_loc, exist_ok=True)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        total_loss = train_loop(building_name, model, loss_fn, optimizer, train_round_list)
        loss_list.append(total_loss)
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(loss_list)
        test_loss = test_loop(building_name, model, loss_fn, eval_round_list)
        test_loss_list.append(test_loss)
        axs[1].plot(test_loss_list)
        plt.savefig(f"{model_loc}/dynamics_model_cost.png")
        if test_loss == np.min(test_loss_list): torch.save(model.state_dict(), f"{model_loc}/dynamics_model_best.pkl")
    print("Done!")



