import sys
sys.path.insert(0, "./")
import os
from preference_data import num_workers, preference_per_round
import numpy as np
import d3rlpy
import pickle
from buildings_factory import *
from train import load_reward_model
from energym.wrappers.rl_wrapper import StableBaselinesRLWrapper
from reward_model import RewardNet, ensemble_num
from d3rlpy.metrics.scorer import evaluate_on_environment
from train import EnergymEvalCallback
from dynamics_predictor import DynamicsPredictor
from d3rlpy.dynamics import ProbabilisticEnsembleDynamics
from d3rlpy.metrics.scorer import dynamics_observation_prediction_error_scorer
from d3rlpy.metrics.scorer import dynamics_reward_prediction_error_scorer
from d3rlpy.metrics.scorer import dynamics_prediction_variance_scorer
from sklearn.model_selection import train_test_split


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
                    rewards[k], _ = reward_function(kpi, next_state)
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
parser.add_argument('--seed', type=int, help='seed', default=7)
parser.add_argument('--iter', type=int, help='iter', default=100)
parser.add_argument('--algo', type=str, help='algorithm to use', default="TD3PlusBC")
parser.add_argument('--logdir', type=str, help='log location', default="models")
parser.add_argument('--rm', type=str, help='reward models', default="ori")
parser.add_argument('--dm', action='store_true', help="whether using learnt dynamics model")
parser.add_argument('--device', type=str, help='device', default="cuda:0")



if __name__ == "__main__":
    args = parser.parse_args()
    building_name = args.building
    is_remote = args.amlt
    parent_loc = (os.environ['AMLT_DATA_DIR'] if is_remote else "./")
    reward_path_suffix = f"{args.algo}"
    reward_path_suffix += f"_{args.rm}"
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
    
    min_kpis, max_kpis, min_outputs, max_outputs = collect_baseline_kpi(building_name, args.amlt)
    reward_function = lambda kpi, state: reward_func(min_kpis, max_kpis, kpi, state)
    env_RL = StableBaselinesRLWrapper(building_name, min_kpis, max_kpis, min_outputs, max_outputs, reward_func)
    eval_env_RL = StableBaselinesRLWrapper(building_name, min_kpis, max_kpis, min_outputs, max_outputs, reward_func, eval=True)
    
    episode_len = env_RL.max_episode_len
    is_gpu_on = torch.cuda.is_available()
    

    if args.rm == "dnn":
        input_dim = env_RL.observation_space.shape[0]
        reward_models, optimizers = load_reward_model(input_dim, reward_model_loc, building_name, args.device)
        reward_function = lambda min_kip, max_kpi, kpi, state: learnt_reward_func(reward_models, min_kip, max_kpi, kpi, state)
    elif args.rm == 'bs': reward_function = baseline_reward_func
    env_RL.reward_function = reward_function
    
    if args.dm:
        input_dim = env_RL.observation_space.shape[0]
        action_dim=env_RL.action_space.shape[0]
        dynamics_predictor = DynamicsPredictor(input_dim+action_dim, input_dim)
        dynamics_predictor.load_state_dict(torch.load(dynamics_model_loc, map_location=args.device))
        dynamics_predictor.eval()
        env_RL.dynamics_predictor = dynamics_predictor
    
    dataset = get_d3rlpy_dataset(building_name, list(range(num_workers)),
                                 list(range(preference_per_round)), reward_function)
    
    if args.device.find(":") >= 0:
        device_id = int(args.device.split(":")[-1])
    else: device_id = False
    
    d3rlpy.seed(args.seed)
    assert args.algo in ["TD3PlusBC", "SAC", "CQL", "MOPO"], "wrong algorithm"
    if args.algo == "TD3PlusBC":
        algo = d3rlpy.algos.TD3PlusBC(action_scaler="min_max", use_gpu=device_id)
    elif args.algo == "CQL":
        algo = d3rlpy.algos.CQL(action_scaler="min_max", use_gpu=device_id)
    elif args.algo == "MOPO":
        train_episodes, test_episodes = train_test_split(dataset)
        mopo = ProbabilisticEnsembleDynamics(learning_rate=1e-4, use_gpu=device_id)
        mopo.fit(train_episodes,
                 n_steps=episode_len,
                 eval_episodes=test_episodes,
                 scorers={
                    'observation_error': dynamics_observation_prediction_error_scorer,
                    'reward_error': dynamics_reward_prediction_error_scorer,
                    'variance': dynamics_prediction_variance_scorer,
                 })
        algo = d3rlpy.algos.MOPO(dynamics=mopo, action_scaler="min_max", use_gpu=device_id)
    else:
        algo = d3rlpy.algos.SAC(action_scaler="min_max", use_gpu=device_id)
    
    log_loc = f"{model_loc}/logs/"
    os.makedirs(log_loc, exist_ok=True)
    post_eval_callback = EnergymEvalCallback(algo, building_name, 
                                             log_loc, min_kpis, max_kpis, 
                                             min_outputs, max_outputs, 
                                             eval_env_RL,
                                             args.amlt,
                                             False, 
                                             verbose=0, 
                                             is_d3rl=True)
    def eval_callback(algo, epoch, total_step):
        if total_step % (max(1, args.iter//10)*episode_len) == 0:
            print("eval on epoch", epoch, "step: ", total_step)
            post_eval_callback.num_timesteps = total_step
            post_eval_callback._on_step()
    
    # start offline training
    algo.fit(
    dataset,
    eval_episodes=dataset.episodes,
    n_steps=episode_len*args.iter,
    n_steps_per_epoch=episode_len,
    callback=eval_callback)
    
    scorers={'environment': evaluate_on_environment(eval_env_RL)}
    print(scorers)