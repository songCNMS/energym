import sys
sys.path.insert(0, "./")

from typing import Any, Callable, Dict, List, Optional, Union
import time
from energym.examples.Controller import LabController
from energym.factory import make
from energym.wrappers.downsample_outputs import DownsampleOutputs
from energym.wrappers.rescale_outputs import RescaleOutputs
from energym.wrappers.rl_wrapper import StableBaselinesRLWrapper
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback, BaseCallback
from buildings_factory import *
from reward_model import RewardNet, ensemble_num
import gym
import pandas as pd
import matplotlib.pyplot as plt
import torch
from dynamics_predictor import DynamicsPredictor


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
        reward_function,
        simulation_days: int = 28,
        verbose: int = 0
    ):
        super().__init__(verbose=verbose)
        self.building_name = building_name
        self.simulation_days = simulation_days
        self.log_loc = log_loc
        self.min_kpis = min_kpis
        self.max_kpis = max_kpis
        self.min_outputs = min_outputs
        self.max_outputs = max_outputs
        self.reward_function = reward_function
        super().init_callback(model)
    
    def _on_step(self) -> bool:
        building_idx = buildings_list.index(self.building_name)
        controller = controller_list[building_idx]
        weather = weather_list[building_idx]
        default_control = default_controls[building_idx]
        exp_name = f"{datetime.today().time().strftime('%m%d-%H%M%S')}"
        
        if is_wandb: wandb.init(project="Energym", config={}, group=building_name, name=f"{self.num_timesteps}_{exp_name}")

        bs_eval_env = make(self.building_name, weather=weather, simulation_days=self.simulation_days, eval_mode=True)
        eval_env_down_RL = StableBaselinesRLWrapper(self.building_name, self.min_kpis, self.max_kpis, self.min_outputs, self.max_outputs, self.reward_function, eval=True)
        inputs = get_inputs(building_name, bs_eval_env)
        

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
        state = eval_env_down_RL.state
        step = 0
        hour = control_values[building_idx]
        res = {}
        while not done:
            bs_control = controller(inputs, step)(bs_outputs, control_values[building_idx], hour)
            bs_control.update(default_control)
            bs_controls +=[ {p:bs_control[p][0] for p in bs_control} ]
            bs_outputs = bs_eval_env.step(bs_control)
            _,hour,_,_ = bs_eval_env.get_date()
            bs_out_list.append(bs_outputs)
            bs_state = eval_env_down_RL.transform_state(bs_outputs)
            ori_bs_reward = reward_func(self.min_kpis, self.max_kpis, bs_eval_env.get_kpi(start_ind=step, end_ind=step+1), bs_state)
            bs_reward = self.reward_function(self.min_kpis, self.max_kpis, bs_eval_env.get_kpi(start_ind=step, end_ind=step+1), bs_state)
            bs_reward_list.append(bs_reward)
            ori_bs_reward_list.append(ori_bs_reward)
            done = (done | (bs_eval_env.time >= bs_eval_env.stop_time))
            
            actions, _ = self.model.predict(state, deterministic=True)
            state, reward, _done, info = eval_env_down_RL.step(actions)
            done = (done | _done)
            outputs = eval_env_down_RL.inverse_transform_state(state)
            ori_reward = reward_func(self.min_kpis, self.max_kpis, eval_env_down_RL.env.get_kpi(start_ind=step, end_ind=step+1), state)
            control = eval_env_down_RL.inverse_transform_action(actions)
            controls +=[ {p:control[p][0] for p in control} ]
            out_list.append(outputs)
            reward_list.append(reward)
            ori_reward_list.append(ori_reward)
            
            res["baseline_reward"] = bs_reward
            res["reward"] = reward
            for key in control:
                res[f"baseline_{key}"] = bs_control[key][0]
                res[key] = control[key][0]
            if is_wandb: wandb.log(res)
            step += 1

            if self.verbose:
                print("RL KPIs") 
                print(eval_env_down_RL.env.get_kpi())
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
        for key, val in eval_env_down_RL.env.kpis.kpi_options.items():
            if "target" in val: kpi_targets[val["name"]] = val["target"]
        
        # plot key values
        f, axs = plt.subplots(len(all_cols_plot)+1,figsize=(10,15))#
        for i, col in enumerate(all_cols_plot):
            if (col not in out_df.columns) or (col not in bs_out_df.columns): continue
            axs[i].plot(out_df[col], 'r', bs_out_df[col], 'b')
            axs[i].set_ylabel(col)
            axs[i].set_xlabel('Steps')
            if col in kpi_targets: intervals = (kpi_targets[col] if isinstance(kpi_targets[col], list) else [kpi_targets[col], kpi_targets[col]])
            elif eval_env_down_RL.env.output_specs[col]["type"] == "scalar": intervals = [eval_env_down_RL.env.output_specs[col]['lower_bound'], eval_env_down_RL.env.output_specs[col]['upper_bound']]
            else: intervals = [0, 0]
            axs[i].plot([0, out_df.shape[0]], [intervals[0], intervals[0]], color='g', linestyle='--', linewidth=2)
            axs[i].plot([0, out_df.shape[0]], [intervals[1], intervals[1]], color='g', linestyle='--', linewidth=2)
            
            vals = out_df[col].tolist()
            bs_vals = bs_out_df[col].tolist()
            if is_wandb:
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
            if eval_env_down_RL.env.input_specs[col]['type'] == 'scalar':
                intervals = [eval_env_down_RL.env.input_specs[col]['lower_bound'], eval_env_down_RL.env.input_specs[col]['upper_bound']]
            else: intervals = [0, 0]
            axs[i].plot([0, cmd_df.shape[0]], [intervals[0], intervals[0]], color='g', linestyle='--', linewidth=2)
            axs[i].plot([0, cmd_df.shape[0]], [intervals[1], intervals[1]], color='g', linestyle='--', linewidth=2)
            
            vals = cmd_df[col].tolist()
            bs_vals = bs_cmd_df[col].tolist()
            if is_wandb:
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
        
        if is_wandb:
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
        eval_env_down_RL.close()
        bs_eval_env.close()
        if is_wandb: wandb.finish()

# buildings_list = ["ApartmentsThermal-v0", "ApartmentsGrid-v0", "Apartments2Thermal-v0",
#                   "Apartments2Grid-v0", "OfficesThermostat-v0", "MixedUseFanFCU-v0",
#                   "SeminarcenterThermostat-v0", "SeminarcenterFull-v0", "SimpleHouseRad-v0",
#                   "SimpleHouseRSla-v0", "SwissHouseRSlaW2W-v0", "SwissHouseRSlaTank-v0"]

import argparse
from datetime import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--amlt', action='store_true', help="remote execution on amlt")
parser.add_argument('--building', type=str, help='building name', required=True)
parser.add_argument('--iter', type=int, help='learning steps', default=204800)
parser.add_argument("--exp_name", default=f"{datetime.today().date().strftime('%m%d-%H%M')}")
parser.add_argument('--logdir', type=str, help='dir of results', default="models")
parser.add_argument('--rm', action='store_true', help="whether using learnt reward model")
parser.add_argument('--dm', action='store_true', help="whether using learnt dynamics model")
parser.add_argument('--seed', type=int, help='seed', default=7)
parser.add_argument('--wandb', action='store_true', help="whether using wandb")


if __name__ == "__main__":
    args = parser.parse_args()
    is_wandb = args.wandb
    if is_wandb:
        os.environ["WANDB_API_KEY"] = "116a4f287fd4fbaa6f790a50d2dd7f97ceae4a03"
        wandb.login()

    building_name = args.building
    min_kpis, max_kpis, min_outputs, max_outputs = collect_baseline_kpi(building_name)

    reward_path_suffix = ("rewards" if args.rm else "manual")
    reward_path_suffix += ("_predictor" if args.dm else "_simulator")
    reward_path_suffix += f"_seed{args.seed}"
    if args.amlt:
        model_loc = f"{os.environ['AMLT_DATA_DIR']}/data/{args.logdir}/{building_name}/{reward_path_suffix}/"
        reward_model_loc = os.environ['AMLT_DATA_DIR'] + "/data/models/{}/reward_model/reward_model_best_{}.pkl"
        dynamics_model_loc = f"{os.environ['AMLT_DATA_DIR']}/data/models/{building_name}/dynamics_model/dynamics_model_best.pkl"
    else:
        model_loc = f"data/{args.logdir}/{building_name}/{reward_path_suffix}/"
        reward_model_loc = "data/models/{}/reward_model/reward_model_best_{}.pkl"
        dynamics_model_loc = f"data/models/{building_name}/dynamics_model/dynamics_model_best.pkl"
    
    log_loc = f"{model_loc}/logs/"
    os.makedirs(log_loc, exist_ok=True)
    env_down_RL = StableBaselinesRLWrapper(building_name, min_kpis, max_kpis, min_outputs, max_outputs, reward_func)
    
    if args.rm:
        input_dim = env_down_RL.observation_space.shape[0]
        reward_models = []
        for i in range(4):
            reward_model = RewardNet(input_dim)
            _reward_model_loc = reward_model_loc.format(building_name, i)
            reward_model.load_state_dict(torch.load(_reward_model_loc))
            reward_model.eval()
            reward_models.append(reward_model)
        env_down_RL.reward_function = lambda min_kip, max_kpi, kpi, state: learnt_reward_func(reward_models, min_kip, max_kpi, kpi, state)
    
    if args.dm:
        input_dim = env_down_RL.observation_space.shape[0]
        action_dim=env_down_RL.action_space.shape[0]
        dynamics_predictor = DynamicsPredictor(input_dim+action_dim, input_dim)
        dynamics_predictor.load_state_dict(torch.load(dynamics_model_loc))
        dynamics_predictor.eval()
        env_down_RL.dynamics_predictor = dynamics_predictor
        

    model = SAC('MlpPolicy', env_down_RL, verbose=1, device='auto', train_freq=256, 
                learning_starts=5120, batch_size=512, gradient_steps=8, seed=args.seed)
    # model = PPO('MlpPolicy', env_down_RL, verbose=1, device='auto', batch_size=64, seed=43)
    checkpoint_callback = CheckpointCallback(save_freq=5120, save_path=model_loc)
    post_eval_callback = EnergymEvalCallback(model, building_name, log_loc, min_kpis, max_kpis, min_outputs, max_outputs, env_down_RL.reward_function, verbose=0)
    eval_callback = EvalCallback(env_down_RL, best_model_save_path=model_loc + "/best_model/",
                                 log_path=log_loc, eval_freq=5120, callback_after_eval=post_eval_callback)
    
    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])
    model.learn(args.iter, callback=callback)
    # model.load("models/SimpleHouseRad-v0/1669143145.6766512.pkl")
    env_down_RL.close()
    