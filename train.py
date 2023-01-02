import sys
sys.path.insert(0, "./")

from typing import Any, Callable, Dict, List, Optional, Union
import time
from energym.examples.Controller import LabController
from energym.factory import make
from energym.wrappers.downsample_outputs import DownsampleOutputs
from energym.wrappers.rescale_outputs import RescaleOutputs
from energym.wrappers.rl_wrapper import StableBaselinesRLWrapper
# from stable_baselines3.ppo2 import PPO2
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, BaseCallback
from buildings_factory import *
from reward_model import RewardNet, ensemble_num, train_loop, test_loop, preference_loss
from preference_data import num_workers, sample_preferences, perference_pairs_per_sample, len_traj, preference_per_round
import gym
import pandas as pd
import matplotlib.pyplot as plt
import torch
from dynamics_predictor import DynamicsPredictor
from stable_baselines3.common.noise import NormalActionNoise
import pickle

import wandb
import os


bs_total_reward_list = []
eval_total_reward_list = []

ori_bs_total_reward_list = []
ori_eval_total_reward_list = []

class EnergymEvalCallback(BaseCallback):
    def __init__(
        self,
        model,
        building_name,
        log_loc,
        min_kpis,
        max_kpis,
        min_outputs,
        max_outputs,
        env,
        is_remote,
        reward_model_retrain,
        simulation_days: int = 28,
        verbose: int = 0,
        is_wandb: bool = False,
        is_d3rl: bool = False,
    ):
        super().__init__(verbose=verbose)
        self.building_name = building_name
        self.simulation_days = simulation_days
        self.log_loc = log_loc
        self.min_kpis = min_kpis
        self.max_kpis = max_kpis
        self.min_outputs = min_outputs
        self.max_outputs = max_outputs
        self.is_wandb = is_wandb
        self.reward_function = env.reward_function
        self.env = env
        self.is_d3rl = is_d3rl
        self.is_remote = is_remote
        self.is_reward_model_retrain = reward_model_retrain
        self.reward_training_round = 1
        if not self.is_d3rl:
            super().init_callback(model)
        else: 
            self.model = model
            self.num_timesteps = 0
            
    def reward_model_trainig(self):
        if self.is_remote:
            online_data_loc = f"{os.environ['AMLT_DATA_DIR']}/data/offline_data/{self.building_name}/traj_data/online_traj.pkl"
            offline_data_loc = "{os.environ['AMLT_DATA_DIR']}/data/offline_data/{}/traj_data/0_{}.pkl"
            online_preference_data_loc = os.environ['AMLT_DATA_DIR'] + "/data/offline_data/{}/preferences_data/{}/"
        else:
            online_data_loc = f"data/offline_data/{self.building_name}/traj_data/online_traj.pkl"
            offline_data_loc = "data/offline_data/{}/traj_data/0_{}.pkl"
            online_preference_data_loc = "data/offline_data/{}/preferences_data/{}/"
        with open(online_data_loc, "rb") as f:
            online_trajectories = pickle.load(f)
        for i in range(preference_per_round):
            if not os.path.exists(offline_data_loc.format(self.building_name, i)): continue
            with open(offline_data_loc.format(self.building_name, i), "rb") as f:
                offline_trajectories = pickle.load(f)
            idx1 = np.random.randint(len(online_trajectories))
            idx2 = np.random.randint(len(offline_trajectories))
            trajectory1, trajectory2 = online_trajectories[idx1], offline_trajectories[idx2]
            preference_pairs1 = sample_preferences(trajectory1, trajectory2, min_kpis, max_kpis, num_preferences=perference_pairs_per_sample)
            preference_pairs2 = sample_preferences(trajectory1, trajectory1, min_kpis, max_kpis, num_preferences=perference_pairs_per_sample)
            preference_pairs3 = sample_preferences(trajectory2, trajectory2, min_kpis, max_kpis, num_preferences=perference_pairs_per_sample)
            _data_loc = online_preference_data_loc.format(self.building_name, len_traj)
            if i < int(0.8*preference_per_round): file_loc = f'{_data_loc}/preference_data_{num_workers+1}_{i}.pkl'
            else: file_loc = f'{_data_loc}/preference_data_{num_workers+2}_{i}.pkl'
            with open(file_loc, 'wb') as f:
                np.save(f, preference_pairs1[len_traj-1]+preference_pairs2[len_traj-1]+preference_pairs3[len_traj-1])

        parent_loc = (os.environ['AMLT_DATA_DIR'] if self.is_remote else "./")
        train_round_list = [num_workers+1]
        eval_round_list = [num_workers+2]
        loss_fn = preference_loss
        loss_list, test_loss_list, correct_list = [[]]*ensemble_num, [[]]*ensemble_num, [[]]*ensemble_num
        for reward_model in reward_models: reward_model.train()
        for t in range(30):
            print(f"Epoch {t+1}\n-------------------------------")
            total_losses = train_loop(self.building_name, reward_models, loss_fn, optimizers, train_round_list, parent_loc, args.device)
            for ridx, total_loss in enumerate(total_losses): loss_list[ridx].append(total_loss)
            
            test_losses, corrects = test_loop(self.building_name, reward_models, loss_fn, eval_round_list, parent_loc, args.device)
            for ridx, test_loss in enumerate(test_losses): test_loss_list[ridx].append(test_loss)
            for ridx, correct in enumerate(corrects): correct_list[ridx].append(correct)
            for ridx in range(ensemble_num):
                fig, axs = plt.subplots(3, 1)
                axs[0].plot(loss_list[ridx])
                axs[1].plot(test_loss_list[ridx])
                axs[2].plot(correct_list[ridx])
                plt.savefig(f"{model_loc}/reward_model_cost_{self.reward_training_round}_{ridx}.png")
        for reward_model in reward_models: reward_model.eval()
        os.remove(online_data_loc)
        self.reward_training_round += 1
        self.env.reward_function = lambda min_kip, max_kpi, kpi, state: learnt_reward_func(reward_models, min_kip, max_kpi, kpi, state)
    
    
    def _on_step(self) -> bool:
        building_idx = buildings_list.index(self.building_name)
        controller = controller_list[building_idx]
        weather = weather_list[building_idx]
        default_control = default_controls[building_idx]
        exp_name = f"{datetime.today().time().strftime('%m%d-%H%M%S')}"
        
        if self.is_wandb: wandb.init(project="Energym", config={}, group=self.building_name, name=f"{self.num_timesteps}_{exp_name}")

        bs_eval_env = make(self.building_name, weather=weather, 
                           simulation_days=self.simulation_days, 
                           eval_mode=True)
        eval_env_RL = StableBaselinesRLWrapper(self.building_name, 
                                               self.min_kpis, self.max_kpis, 
                                               self.min_outputs, self.max_outputs, 
                                               self.reward_function, eval=True)
        inputs = get_inputs(self.building_name, bs_eval_env)
        

        out_list = []
        controls = []
        reward_list = []
        ori_reward_list = []

        bs_out_list = []
        bs_controls = []
        bs_reward_list = []
        ori_bs_reward_list = []
            
        bs_outputs = bs_eval_env.get_output()
        done = False
        state = eval_env_RL.state
        step = 0
        hour = control_values[building_idx]
        res = {}
        while not done and step < eval_env_RL.max_episode_len:
            bs_control = controller(inputs, step)(bs_outputs, control_values[building_idx], hour)
            bs_control.update(default_control)
            bs_controls +=[ {p:bs_control[p][0] for p in bs_control} ]
            bs_outputs = bs_eval_env.step(bs_control)
            _,hour,_,_ = bs_eval_env.get_date()
            bs_out_list.append(bs_outputs)
            bs_state = eval_env_RL.transform_state(bs_outputs)
            ori_bs_reward = reward_func(self.min_kpis, self.max_kpis, bs_eval_env.get_kpi(start_ind=step, end_ind=step+1), bs_state)
            bs_reward = self.reward_function(self.min_kpis, self.max_kpis, bs_eval_env.get_kpi(start_ind=step, end_ind=step+1), bs_state)
            bs_reward_list.append(bs_reward)
            ori_bs_reward_list.append(ori_bs_reward)
            done = (done | (bs_eval_env.time >= bs_eval_env.stop_time))
            
            if self.is_d3rl: actions = self.model.predict([state])
            else: actions, _ = self.model.predict(state, deterministic=True)
            if len(actions.shape) > 1: actions = actions[0]
            state, reward, _done, info = eval_env_RL.step(actions)
            done = (done | _done)
            outputs = eval_env_RL.inverse_transform_state(state)
            ori_reward = reward_func(self.min_kpis, self.max_kpis, eval_env_RL.env.get_kpi(start_ind=step, end_ind=step+1), state)
            control = eval_env_RL.inverse_transform_action(actions)
            controls +=[ {p:control[p][0] for p in control} ]
            out_list.append(outputs)
            reward_list.append(reward)
            ori_reward_list.append(ori_reward)
            
            res["baseline_reward"] = bs_reward
            res["reward"] = reward
            for key in control:
                res[f"baseline_{key}"] = bs_control[key][0]
                res[key] = control[key][0]
            if self.is_wandb: wandb.log(res)
            step += 1

            if self.verbose:
                print("RL KPIs") 
                print(eval_env_RL.env.get_kpi())
                print("BS KPIs")
                print(bs_eval_env.get_kpi()) 

        eval_total_reward_list.append(np.sum(reward_list))
        bs_total_reward_list.append(np.sum(bs_reward_list))
        ori_eval_total_reward_list.append(np.sum(ori_reward_list))
        ori_bs_total_reward_list.append(np.sum(ori_bs_reward_list))
        best_result = (np.max(eval_total_reward_list) == np.sum(reward_list))

        out_df = pd.DataFrame(out_list)
        cmd_df = pd.DataFrame(controls)

        bs_out_df = pd.DataFrame(bs_out_list)
        bs_cmd_df = pd.DataFrame(bs_controls)
        
        result_data_dir = f"{self.log_loc}/{self.num_timesteps}/"
        best_result_data_dir = f"{self.log_loc}/best/"
        os.makedirs(result_data_dir, exist_ok=True)
        os.makedirs(best_result_data_dir, exist_ok=True)
        out_dirs = [result_data_dir]
        if best_result: out_dirs.append(best_result_data_dir)
        for _data_dir in out_dirs:
            out_df.to_csv(f"{_data_dir}/out.csv", index=False)
            bs_out_df.to_csv(f"{_data_dir}/bs_out.csv", index=False)
            cmd_df.to_csv(f"{_data_dir}/control.csv", index=False)
            bs_cmd_df.to_csv(f"{_data_dir}/bs_control.csv", index=False)
        
        
        all_cols_plot = []
        for cols in cols_plot[building_idx]: all_cols_plot.extend(cols)
        
        kpi_targets = {}
        for key, val in eval_env_RL.env.kpis.kpi_options.items():
            if "target" in val: kpi_targets[val["name"]] = val["target"]
        
        # plot key values
        f, axs = plt.subplots(len(all_cols_plot)+1,figsize=(10,15))#
        for i, col in enumerate(all_cols_plot):
            if (col not in out_df.columns) or (col not in bs_out_df.columns): continue
            axs[i].plot(out_df[col], 'r', bs_out_df[col], 'b')
            axs[i].set_ylabel(col)
            axs[i].set_xlabel('Steps')
            if col in kpi_targets: intervals = (kpi_targets[col] if isinstance(kpi_targets[col], list) else [kpi_targets[col], kpi_targets[col]])
            elif eval_env_RL.env.output_specs[col]["type"] == "scalar": intervals = [eval_env_RL.env.output_specs[col]['lower_bound'], eval_env_RL.env.output_specs[col]['upper_bound']]
            else: intervals = [0, 0]
            axs[i].plot([0, out_df.shape[0]], [intervals[0], intervals[0]], color='g', linestyle='--', linewidth=2)
            axs[i].plot([0, out_df.shape[0]], [intervals[1], intervals[1]], color='g', linestyle='--', linewidth=2)
            
            vals = out_df[col].tolist()
            bs_vals = bs_out_df[col].tolist()
            if self.is_wandb:
                for j in range(min(len(vals), len(bs_vals))):
                    wandb.log({f"{col}_out_lower_bound": intervals[0], 
                            f"{col}_out_upper_bound": intervals[1],
                            f"baseline_out_{col}": bs_vals[j],
                            f"out_{col}": vals[j]})
            
        axs[-1].plot(np.cumsum(reward_list), 'r--', np.cumsum(bs_reward_list), 'b--')
        axs[-1].set_ylabel('Reward')
        axs[-1].set_xlabel('Steps')
        plt.subplots_adjust(hspace=0.4)
        for _data_dir in out_dirs: plt.savefig(f"{_data_dir}/RL.png")
        
        # plot controls
        max_records_plot = 100
        if cmd_df.shape[0] > max_records_plot:
            cmd_df = cmd_df.iloc[-max_records_plot:, :].reset_index()
            bs_cmd_df = bs_cmd_df.iloc[-max_records_plot:, :].reset_index()
        f, axs = plt.subplots(len(inputs)+1,figsize=(10,15))#
        for i, col in enumerate(inputs):
            if (col not in cmd_df.columns) or (col not in bs_cmd_df.columns): continue
            axs[i].plot(cmd_df[col], 'r', bs_cmd_df[col], 'b')
            axs[i].set_ylabel(col)
            axs[i].set_xlabel('Steps')
            if eval_env_RL.env.input_specs[col]['type'] == 'scalar':
                intervals = [eval_env_RL.env.input_specs[col]['lower_bound'], eval_env_RL.env.input_specs[col]['upper_bound']]
            else: intervals = [0, 0]
            axs[i].plot([0, cmd_df.shape[0]], [intervals[0], intervals[0]], color='g', linestyle='--', linewidth=2)
            axs[i].plot([0, cmd_df.shape[0]], [intervals[1], intervals[1]], color='g', linestyle='--', linewidth=2)
            
            vals = cmd_df[col].tolist()
            bs_vals = bs_cmd_df[col].tolist()
            if self.is_wandb:
                for j in range(min(len(vals), len(bs_vals))):
                    wandb.log({f"{col}_cmd_lower_bound": intervals[0], 
                            f"{col}_cmd_upper_bound": intervals[1],
                            f"baseline_cmd_{col}": bs_vals[j],
                            f"cmd_{col}": vals[j]})
        
        reward_df = pd.DataFrame(data={"eval_episode_reward": eval_total_reward_list,
                                       "baseline_eval_episode_reward": bs_total_reward_list,
                                       "manual_eval_episode_reward": ori_eval_total_reward_list,
                                       "manual_baseline_eval_episode_reward": ori_bs_total_reward_list})
        for _data_dir in out_dirs: reward_df.to_csv(f"{_data_dir}/rewards.csv", index=False)
        
        if self.is_wandb:
            for i in range(len(eval_total_reward_list)):
                wandb.log({"eval_episode_reward": eval_total_reward_list[i],
                        "baseline_eval_episode_reward": bs_total_reward_list[i],
                        "manual_eval_episode_reward": ori_eval_total_reward_list[i],
                        "manual_baseline_eval_episode_reward": ori_bs_total_reward_list[i]})
        
        plt.subplots_adjust(hspace=0.4)
        for _data_dir in out_dirs: plt.savefig(f"{_data_dir}/Control_RL.png")
            
        f, axs = plt.subplots(2,figsize=(10,15))#
        axs[0].plot(eval_total_reward_list, 'r')
        ax01 = axs[0].twinx()
        ax01.plot(bs_total_reward_list, 'b')
        axs[0].set_ylabel("Rewards")
        axs[0].set_xlabel('Steps')
        axs[1].plot(ori_eval_total_reward_list, 'r--')
        ax11 = axs[1].twinx()
        ax11.plot(ori_bs_total_reward_list, 'b--')
        axs[1].set_ylabel("Manual Rewards")
        axs[1].set_xlabel('Steps')
        plt.subplots_adjust(hspace=0.4)
        for _data_dir in out_dirs: plt.savefig(f"{_data_dir}/reward.png")
        eval_env_RL.close()
        bs_eval_env.close()
        if self.is_wandb: wandb.finish()
        if self.is_reward_model_retrain: self.reward_model_trainig()
        
        
def load_reward_model(input_dim):
    reward_models = []
    optimizers = []
    for i in range(ensemble_num):
        reward_model = RewardNet(input_dim).to(args.device)
        optimizer = torch.optim.Adam(reward_model.parameters(), lr=0.001)
        _reward_model_loc = reward_model_loc.format(building_name, i)
        checkpoint = torch.load(_reward_model_loc)
        reward_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        reward_model.eval()
        reward_models.append(reward_model)
        optimizers.append(optimizer)
    return reward_models, optimizers

# buildings_list = ["ApartmentsThermal-v0", "ApartmentsGrid-v0", "Apartments2Thermal-v0",
#                   "Apartments2Grid-v0", "OfficesThermostat-v0", "MixedUseFanFCU-v0",
#                   "SeminarcenterThermostat-v0", "SeminarcenterFull-v0", "SimpleHouseRad-v0",
#                   "SimpleHouseRSla-v0", "SwissHouseRSlaW2W-v0", "SwissHouseRSlaTank-v0"]

import argparse
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--amlt', action='store_true', help="remote execution on amlt")
parser.add_argument('--building', type=str, help='building name', required=True)
parser.add_argument('--iter', type=int, help='learning steps', default=1000)
parser.add_argument("--exp_name", default=f"{datetime.today().date().strftime('%m%d-%H%M')}")
parser.add_argument('--logdir', type=str, help='dir of results', default="models")
parser.add_argument('--rm', action='store_true', help="whether using learnt reward model")
parser.add_argument('--dm', action='store_true', help="whether using learnt dynamics model")
parser.add_argument('--seed', type=int, help='seed', default=7)
parser.add_argument('--wandb', action='store_true', help="whether using wandb")
parser.add_argument('--algo', type=str, help='algorithm', default="SAC")
parser.add_argument('--device', type=str, help='device', default="cuda:0")
parser.add_argument('--inc', action='store_true', help="interleaving training")


if __name__ == "__main__":
    args = parser.parse_args()
    is_wandb = args.wandb
    if is_wandb:
        os.environ["WANDB_API_KEY"] = "116a4f287fd4fbaa6f790a50d2dd7f97ceae4a03"
        wandb.login()

    building_name = args.building
    min_kpis, max_kpis, min_outputs, max_outputs = collect_baseline_kpi(building_name)
    policy_name = args.algo
    reward_path_suffix = f"{policy_name}"
    reward_path_suffix += ("_inc" if args.inc else "")
    reward_path_suffix += ("_rewards" if args.rm else "_manual")
    reward_path_suffix += ("_predictor" if args.dm else "_simulator")
    reward_path_suffix += f"_seed{args.seed}"
    if args.amlt:
        model_loc = f"{os.environ['AMLT_DATA_DIR']}/data/{args.logdir}/{building_name}/{reward_path_suffix}/"
        reward_model_loc = os.environ['AMLT_DATA_DIR'] + "/data/models/{}/reward_model/reward_model_best_{}.pkl"
        dynamics_model_loc = f"{os.environ['AMLT_DATA_DIR']}/data/models/{building_name}/dynamics_model/dynamics_model_best.pkl"
        online_data_loc = f"{os.environ['AMLT_DATA_DIR']}/data/offline_data/{building_name}/traj_data/online_traj.pkl"
    else:
        model_loc = f"data/{args.logdir}/{building_name}/{reward_path_suffix}/"
        reward_model_loc = "data/models/{}/reward_model/reward_model_best_{}.pkl"
        dynamics_model_loc = f"data/models/{building_name}/dynamics_model/dynamics_model_best.pkl"
        online_data_loc = f"data/offline_data/{building_name}/traj_data/online_traj.pkl"
    
    os.remove(online_data_loc)
    log_loc = f"{model_loc}/logs/"
    os.makedirs(log_loc, exist_ok=True)
    if args.inc: env_RL = StableBaselinesRLWrapper(building_name, min_kpis, max_kpis, min_outputs, max_outputs, reward_func, save_data=True, data_loc=online_data_loc)
    else: env_RL = StableBaselinesRLWrapper(building_name, min_kpis, max_kpis, min_outputs, max_outputs, reward_func)
    
    episode_len = env_RL.max_episode_len
    map_location=torch.device(args.device)
    
    if args.rm:
        input_dim = env_RL.observation_space.shape[0]
        reward_models, optimizers = load_reward_model(input_dim)
        env_RL.reward_function = lambda min_kip, max_kpi, kpi, state: learnt_reward_func(reward_models, min_kip, max_kpi, kpi, state)
    
    if args.dm:
        input_dim = env_RL.observation_space.shape[0]
        action_dim=env_RL.action_space.shape[0]
        dynamics_predictor = DynamicsPredictor(input_dim+action_dim, input_dim).to(args.device)
        dynamics_predictor.load_state_dict(torch.load(dynamics_model_loc, map_location=map_location))
        dynamics_predictor.eval()
        env_RL.dynamics_predictor = dynamics_predictor
        
    batch_size = 1024
    total_num_steps = args.iter*episode_len
    print("total time steps: ", total_num_steps)
    n_actions = env_RL.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    if policy_name == "SAC":
        model = SAC('MlpPolicy', env_RL, verbose=1, device=args.device, 
                    train_freq=8, buffer_size=episode_len*max(1, args.iter//4), 
                    gamma=0.99, tau=0.01,
                    # action_noise=action_noise,
                    learning_starts=episode_len, 
                    batch_size=batch_size,
                    gradient_steps=8,
                    target_update_interval=16,
                    seed=args.seed,
                    policy_kwargs=dict(net_arch=[512, 512, 512], 
                                       activation_fn=torch.nn.ReLU))
    else:
        model = PPO('MlpPolicy', env_RL, verbose=1, 
                    device=args.device, batch_size=batch_size, 
                    seed=args.seed,
                    policy_kwargs=dict(net_arch=[512, 512, 512], 
                                       activation_fn=torch.nn.ReLU))
    checkpoint_callback = CheckpointCallback(save_freq=episode_len*max(1, args.iter//20), save_path=model_loc)
    post_eval_callback = EnergymEvalCallback(model, building_name, log_loc, 
                                             min_kpis, max_kpis, 
                                             min_outputs, max_outputs, 
                                             env_RL, args.amlt, args.inc, verbose=0)
    eval_callback = EvalCallback(env_RL, best_model_save_path=model_loc + "/best_model/",
                                 log_path=log_loc, eval_freq=episode_len*max(1, args.iter//20), 
                                 callback_after_eval=post_eval_callback)
    
    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])
    model.learn(total_num_steps, callback=callback)
    env_RL.close()
