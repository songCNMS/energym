import sys
sys.path.insert(0, "/home/lesong/energym/")

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
import gym
import pandas as pd
import matplotlib.pyplot as plt


class EnergymEvalCallback(BaseCallback):
    def __init__(
        self,
        model,
        building_name,
        log_loc,
        min_kpis,
        max_kpis,
        simulation_days: int = 14,
        verbose: int = 0
    ):
        super().__init__(verbose=verbose)
        self.building_name = building_name
        self.simulation_days = simulation_days
        self.log_loc = log_loc
        self.min_kpis = min_kpis
        self.max_kpis = max_kpis
        super().init_callback(model)
    
    def _on_step(self) -> bool:
        building_idx = buildings_list.index(self.building_name)
        controller = controller_list[building_idx]
        weather = weather_list[building_idx]
        default_control = default_controls[building_idx]

        bs_eval_env = make(self.building_name, weather=weather, simulation_days=self.simulation_days, eval_mode=True)
        eval_env_down_RL = StableBaselinesRLWrapper(self.building_name, self.min_kpis, self.max_kpis, reward_func, eval=True)
        inputs = get_inputs(building_name, bs_eval_env)
        

        out_list = []
        controls = []
        reward_list = []

        bs_out_list = []
        bs_controls = []
        bs_reward_list = []
            
        bs_outputs = bs_eval_env.get_output()
        done = False
        outputs = eval_env_down_RL.env.get_output()
        state = eval_env_down_RL.transform_state(outputs)
        step = 0
        hour = control_values[building_idx]
        while not done:
            control = controller(inputs, step)(bs_outputs, control_values[building_idx], hour)
            control.update(default_control)
            bs_controls +=[ {p:control[p][0] for p in control} ]
            bs_outputs = bs_eval_env.step(control)
            _,hour,_,_ = bs_eval_env.get_date()
            bs_out_list.append(bs_outputs)
            bs_reward = reward_func(self.min_kpis, self.max_kpis, bs_eval_env.get_kpi(start_ind=step, end_ind=step+1))
            bs_reward_list.append(bs_reward)
            done = (done | (bs_eval_env.time >= bs_eval_env.stop_time))
            
            actions, _ = self.model.predict(state)
            state, reward, _done, info = eval_env_down_RL.step(actions)
            done = (done | _done)
            outputs = eval_env_down_RL.inverse_transform_state(state)
            control = eval_env_down_RL.inverse_transform_action(actions)
            controls +=[ {p:control[p][0] for p in control} ]
            out_list.append(outputs)
            reward_list.append(reward)
            step += 1

            if self.verbose:
                print("RL KPIs") 
                print(eval_env_down_RL.env.get_kpi())
                print("BS KPIs")
                print(bs_eval_env.get_kpi())
        eval_env_down_RL.close()
        bs_eval_env.close()   

        out_df = pd.DataFrame(out_list)
        cmd_df = pd.DataFrame(controls)

        bs_out_df = pd.DataFrame(bs_out_list)
        bs_cmd_df = pd.DataFrame(bs_controls)
        
        result_data_dir = self.log_loc
        out_df.to_csv(f"{result_data_dir}/out.csv", index=False)
        bs_out_df.to_csv(f"{result_data_dir}/bs_out.csv", index=False)
        cmd_df.to_csv(f"{result_data_dir}/control.csv", index=False)
        bs_cmd_df.to_csv(f"{result_data_dir}/bs_control.csv", index=False)
        
        all_cols_plot = []
        for cols in cols_plot[building_idx]: all_cols_plot.extend(cols)
        
        kpi_targets = {}
        for key, val in eval_env_down_RL.env.kpis.kpi_options.items():
            if "target" in val: kpi_targets[val["name"]] = val["target"]
        
        # plot key values
        f, axs = plt.subplots(len(all_cols_plot)+1,figsize=(10,15))#
        for i, col in enumerate(all_cols_plot):
            axs[i].plot(out_df[col], 'r', bs_out_df[col], 'b')
            axs[i].set_ylabel(col)
            axs[i].set_xlabel('Steps')
            if col in kpi_targets: intervals = (kpi_targets[col] if isinstance(kpi_targets[col], list) else [kpi_targets[col], kpi_targets[col]])
            else: intervals = [eval_env_down_RL.env.output_specs[col]['lower_bound'], eval_env_down_RL.env.output_specs[col]['upper_bound']]
            axs[i].plot([0, out_df.shape[0]], [intervals[0], intervals[0]], color='g', linestyle='--', linewidth=2)
            axs[i].plot([0, out_df.shape[0]], [intervals[1], intervals[1]], color='g', linestyle='--', linewidth=2)
            
        axs[-1].plot(np.cumsum(reward_list), 'r--', np.cumsum(bs_reward_list), 'b--')
        axs[-1].set_ylabel('Reward')
        axs[-1].set_xlabel('Steps')
        plt.subplots_adjust(hspace=0.4)
        plt.savefig(f"{result_data_dir}/RL.png")
        
        # plot controls
        max_records_plot = 100
        if cmd_df.shape[0] > max_records_plot:
            cmd_df = cmd_df.iloc[-max_records_plot:, :].reset_index()
            bs_cmd_df = bs_cmd_df.iloc[-max_records_plot:, :].reset_index()
        f, axs = plt.subplots(len(inputs)+1,figsize=(10,15))#
        for i, col in enumerate(inputs):
            axs[i].plot(cmd_df[col], 'r', bs_cmd_df[col], 'b')
            axs[i].set_ylabel(col)
            axs[i].set_xlabel('Steps')
            intervals = [eval_env_down_RL.env.input_specs[col]['lower_bound'], eval_env_down_RL.env.input_specs[col]['upper_bound']]
            axs[i].plot([0, cmd_df.shape[0]], [intervals[0], intervals[0]], color='g', linestyle='--', linewidth=2)
            axs[i].plot([0, cmd_df.shape[0]], [intervals[1], intervals[1]], color='g', linestyle='--', linewidth=2)
            
        plt.subplots_adjust(hspace=0.4)
        plt.savefig(f"{result_data_dir}/Control_RL.png")
            
        


# buildings_list = ["ApartmentsThermal-v0", "ApartmentsGrid-v0", "Apartments2Thermal-v0",
#                   "Apartments2Grid-v0", "OfficesThermostat-v0", "MixedUseFanFCU-v0",
#                   "SeminarcenterThermostat-v0", "SeminarcenterFull-v0", "SimpleHouseRad-v0",
#                   "SimpleHouseRSla-v0", "SwissHouseRSlaW2W-v0", "SwissHouseRSlaTank-v0"]

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--amlt', action='store_true', help="remote execution on amlt")
parser.add_argument('--building', type=str, help='building name', required=True)

args = parser.parse_args()


def collect_baseline_kpi(building_name):
    max_kpis, min_kpis = {}, {}
    _env = get_env(building_name)
    building_idx = buildings_list.index(building_name)
    controller = controller_list[building_idx]
    default_control = default_controls[building_idx]
    hour = control_values[building_idx]
    step = 0
    inputs = get_inputs(building_name, _env)
    outputs = _env.get_output()
    done = False
    while not done:
        control = controller(inputs, step)(outputs, control_values[building_idx], hour)
        control.update(default_control)
        outputs = _env.step(control)
        _,hour,_,_ = _env.get_date()
        kpis = _env.get_kpi(start_ind=step, end_ind=step+1)
        done = (_env.time >= _env.stop_time)
        step += 1
        for key, val in kpis.items():
            if key not in max_kpis: 
                max_kpis[key] = {}
                max_kpis[key].update(val)
            if key not in min_kpis:
                min_kpis[key] = {}
                min_kpis[key].update(val)
            max_kpis[key]['kpi'] = max(max_kpis[key]['kpi'], val['kpi'])
            min_kpis[key]['kpi'] = min(min_kpis[key]['kpi'], val['kpi'])
    return min_kpis, max_kpis


if __name__ == "__main__":
    building_name = args.building
    min_kpis, max_kpis = collect_baseline_kpi(building_name)

    env_down_RL = StableBaselinesRLWrapper(building_name, min_kpis, max_kpis, reward_func)
    
    if args.amlt:
        model_loc = f"{os.environ['AMLT_OUTPUT_DIR']}/models/{building_name}/"
    else:
        model_loc = f"models/{building_name}/"
    
    log_loc = f"{model_loc}/logs/"
    os.makedirs(log_loc, exist_ok=True)

    # model = SAC('MlpPolicy', env_down_RL, verbose=1, device='auto', train_freq=256, learning_starts=5120, batch_size=512, gradient_steps=8, seed=43)
    model = PPO('MlpPolicy', env_down_RL, verbose=1, device='auto', batch_size=64, seed=43)
    checkpoint_callback = CheckpointCallback(save_freq=5120, save_path=model_loc)
    post_eval_callback = EnergymEvalCallback(model, building_name, log_loc, min_kpis, max_kpis)
    eval_callback = EvalCallback(env_down_RL, best_model_save_path=model_loc + "/best_model/",
                                 log_path=log_loc, eval_freq=5120, callback_after_eval=post_eval_callback)
    
    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])
    model.learn(1000000, callback=callback)
    # model.load("models/SimpleHouseRad-v0/1669143145.6766512.pkl")
    env_down_RL.close()
    