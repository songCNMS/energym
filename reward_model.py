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


class RewardNet(nn.Module):
    def __init__(self, input_dim):
        super(RewardNet, self).__init__()
        self.linear_relu_stack = NN(input_dim=input_dim, 
                      layers_info= [256, 256, 1],
                      output_activation="sigmoid",
                      batch_norm=False, dropout=0.3,
                      hidden_activations=["tanh", 'relu', 'relu'], initialiser="Xavier", random_seed=43)
        
    def get_reward(self, x):
        x_out = self.linear_relu_stack(x)
        return 10.0*x_out
        # return 5.0*torch.nn.Sigmoid()(x_out)
        # return torch.clamp(x_out, 0.0, 10.0)

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
    def __init__(self, round, building_name):
        self.round = round
        with open(f'offline_data/preferences_data_{building_name}/{len_traj}/preference_data_{round*preference_per_round}_{(round+1)*preference_per_round}.pkl', 'rb') as f:
            self.raw_data = np.load(f, allow_pickle=True)
        self.data = torch.from_numpy(self.raw_data[:, :-2]).to(torch.float)
        self.labels = torch.from_numpy(self.raw_data[:, -2:]).to(torch.float)
        
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
        training_dataset = PreferencDataset(round, building_name)
        dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
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
            if batch % 10 == 0:
                print(pred[:5, :], y[:5, :])
                loss, current = loss.cpu().item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return total_loss / total_size

def test_loop(building_name, model, loss_fn, round_list):
    total_num_batches = 0
    test_loss, correct = 0, 0
    total_size = 0
    for round in round_list:
        training_dataset = PreferencDataset(round, building_name)
        dataloader = DataLoader(training_dataset, batch_size=1, shuffle=False)
        total_size = len(dataloader.dataset)
        num_batches = len(dataloader)
        total_num_batches += num_batches
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
    correct /= total_size
    test_loss /= total_size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct
    
    
import matplotlib.pyplot as plt
from buildings_factory import *
from energym.wrappers.rl_wrapper import StableBaselinesRLWrapper

batch_size = 1024
device = ("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


# buildings_list = ["ApartmentsThermal-v0", "ApartmentsGrid-v0", "Apartments2Thermal-v0",
#                   "Apartments2Grid-v0", "OfficesThermostat-v0", "MixedUseFanFCU-v0",
#                   "SeminarcenterThermostat-v0", "SeminarcenterFull-v0", "SimpleHouseRad-v0",
#                   "SimpleHouseRSla-v0", "SwissHouseRSlaW2W-v0", "SwissHouseRSlaTank-v0"] 


if __name__ == "__main__":
    building_name = "SimpleHouseRad-v0"
    building_idx = buildings_list.index(building_name)
    env = get_env(building_name)
    inputs = get_inputs(building_name, env)
    default_control = default_controls[building_idx]
    env_rl = StableBaselinesRLWrapper(env, reward_func, inputs, default_control)
    # input_dim = env_rl.action_space.shape[0] + env_rl.observation_space.shape[0]
    input_dim = env_rl.observation_space.shape[0]
    model = RewardNet(input_dim).to(device)
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = preference_loss


    train_round_list = list(range(num_workers-1))
    eval_round_list = list(range(num_workers-1, num_workers))
    # train_round_list = [0]
    # eval_round_list = [0]

    epochs = 1000
    loss_list = []
    test_loss_list = []
    correct_list = []
    model_loc = f"models/{building_name}/"
    os.makedirs(model_loc, exist_ok=True)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        total_loss = train_loop(building_name, model, loss_fn, optimizer, train_round_list)
        torch.save(model.state_dict(), f"{model_loc}/reward_model_best.pkl")
        loss_list.append(total_loss)
        fig, axs = plt.subplots(3, 1)
        axs[0].plot(loss_list)
        test_loss, correct = test_loop(building_name, model, loss_fn, eval_round_list)
        test_loss_list.append(test_loss)
        correct_list.append(correct)
        axs[1].plot(test_loss_list)
        axs[2].plot(correct_list)
        plt.savefig(f"{model_loc}/reward_model_cost.png")
    print("Done!")



