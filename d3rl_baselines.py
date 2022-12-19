import sys
sys.path.insert(0, "./")
import os
from preference_data import num_workers, preference_per_round
import numpy as np
import d3rlpy
import pickle
from buildings_factory import *
from energym.wrappers.rl_wrapper import StableBaselinesRLWrapper
from reward_model import RewardNet, ensemble_num
from d3rlpy.metrics.scorer import evaluate_on_environment
from train import EnergymEvalCallback


def get_d3rlpy_dataset(building_name, round_list, preference_list, reward_function):
    observation_list = []
    actions_list = []
    rewards_list = []
    terminals_list = []
    for round in round_list:
        for traj_idx in preference_list:
            data_loc = f'{parent_loc}/data/offline_data/{building_name}/traj_data/{round}_{traj_idx}.pkl'
            if not os.path.exists(data_loc): continue
            with open(data_loc, 'rb') as f:
                raw_data = pickle.load(f)
            num_samples = (len(raw_data)*len(raw_data[0])) // 4
            state_dim = len(raw_data[0][0])
            action_dim = len(raw_data[0][1])
            observations = np.zeros((num_samples, state_dim))
            actions = np.zeros((num_samples, action_dim))
            rewards = np.zeros(num_samples)
            terminals = np.zeros(num_samples)
            k = 0
            for traj in raw_data:
                for i in range(len(traj) // 4):
                    _, action, reward, kpi, next_state = traj[4*i], traj[4*i+1], traj[4*i+2], traj[4*i+3], traj[4*i+4]
                    observations[k, :] = next_state
                    actions[k, :] = action
                    rewards[k] = reward_function(kpi, next_state)
                    k += 1
                terminals[k-1] = 1
            observation_list.append(observations)
            actions_list.append(actions)
            rewards_list.append(rewards)
            terminals_list.append(terminals)
    observations = np.concatenate(observation_list, axis=0)
    actions = np.concatenate(actions_list, axis=0)
    rewards = np.concatenate(rewards_list)
    terminals = np.concatenate(terminals_list)
    dataset = d3rlpy.dataset.MDPDataset(
    observations=observations,
    actions=actions,
    rewards=rewards,
    terminals=terminals,
    )
    return dataset


# buildings_list = ["ApartmentsThermal-v0", "ApartmentsGrid-v0", "Apartments2Thermal-v0",
#                   "Apartments2Grid-v0", "OfficesThermostat-v0", "MixedUseFanFCU-v0",
#                   "SeminarcenterThermostat-v0", "SeminarcenterFull-v0", "SimpleHouseRad-v0",
#                   "SimpleHouseRSla-v0", "SwissHouseRSlaW2W-v0", "SwissHouseRSlaTank-v0"] 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--amlt', action='store_true', help="remote execution on amlt")
parser.add_argument('--building', type=str, help='building name', required=True)
parser.add_argument('--round', type=str, help='round', default="")
parser.add_argument('--algo', type=str, help='algorithm to use', default="TD3PlusBC")
parser.add_argument('--rm', action='store_true', help="use learnt reward function")


if __name__ == "__main__":
    args = parser.parse_args()
    building_name = args.building
    is_remote = args.amlt
    parent_loc = (os.environ['AMLT_DATA_DIR'] if is_remote else "./")
    min_kpis, max_kpis, min_outputs, max_outputs = collect_baseline_kpi(building_name)
    reward_path_suffix = f"baseline_{args.algo}"
    if args.amlt:
        reward_model_loc = os.environ['AMLT_DATA_DIR'] + "/data/models/{}/reward_model/reward_model_best_{}.pkl"
        model_loc = f"{os.environ['AMLT_DATA_DIR']}/data/models/{building_name}/{reward_path_suffix}/"
    else:
        reward_model_loc = "data/models/{}/reward_model/reward_model_best_{}.pkl"
        model_loc = f"data/models/{building_name}/{reward_path_suffix}/"
    
    reward_function = lambda kpi, state: reward_func(min_kpis, max_kpis, kpi, state)
    env_RL = StableBaselinesRLWrapper(building_name, min_kpis, max_kpis, min_outputs, max_outputs, reward_func)
    eval_env_RL = StableBaselinesRLWrapper(building_name, min_kpis, max_kpis, min_outputs, max_outputs, reward_func, eval=True)
    
    if args.rm:
        input_dim = env_RL.observation_space.shape[0]
        reward_models = []
        for i in range(ensemble_num):
            reward_model = RewardNet(input_dim)
            _reward_model_loc = reward_model_loc.format(building_name, i)
            reward_model.load_state_dict(torch.load(_reward_model_loc))
            reward_model.eval()
            reward_models.append(reward_model)
        reward_function = lambda kpi, state: learnt_reward_func(reward_models, min_kpis, max_kpis, kpi, state)
    algo = d3rlpy.algos.TD3PlusBC()
    dataset = get_d3rlpy_dataset(building_name, [1, 2], list(range(10)), reward_function)
    
    log_loc = f"{model_loc}/logs/"
    os.makedirs(log_loc, exist_ok=True)
    post_eval_callback = EnergymEvalCallback(algo, building_name, 
                                             log_loc, min_kpis, max_kpis, 
                                             min_outputs, max_outputs, 
                                             env_RL.reward_function, verbose=0, is_d3rl=True)
    def eval_callback(algo, epoch, total_step):
        print("eval on epoch", epoch)
        post_eval_callback._on_step()
        
    # start offline training
    algo.fit(
    dataset,
    eval_episodes=dataset.episodes,
    n_steps=1024000,
    n_steps_per_epoch=32,
    callback=eval_callback,
    scorers={'environment': evaluate_on_environment(eval_env_RL),})
    