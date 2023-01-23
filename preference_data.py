import sys
sys.path.insert(0, "./")

import pickle
import random
import multiprocessing as mp
import pickle
import numpy as np
import os
from buildings_factory import *
from energym.wrappers.rl_wrapper import StableBaselinesRLWrapper
from stable_baselines3 import PPO, SAC


def add_kpi(cur_kpi, kpi, min_kpis, max_kpis):
    # {'kpi1': {'name': 'Fa_Pw_All', 'type': 'avg', 'kpi': 1921.4823123126669}, 'kpi2': {'name': 'Z01_T', 'type': 'avg_dev', 'target': [19, 24], 'kpi': 0.1573853083208828}, 'kpi3': {'name': 'Z02_T', 'type': 'avg_dev', 'target': [19, 24], 'kpi': 0.37227112901825216}, 'kpi4': {'name': 'Z03_T', 'type': 'avg_dev', 'target': [19, 24], 'kpi': 0.4261302830511886}, 'kpi5': {'name': 'Z04_T', 'type': 'avg_dev', 'target': [19, 24], 'kpi': 0.4957972201715515}, 'kpi6': {'name': 'Z05_T', 'type': 'avg_dev', 'target': [19, 24], 'kpi': 0.26159124363016045}, 'kpi7': {'name': 'Z06_T', 'type': 'avg_dev', 'target': [19, 24], 'kpi': 0.660215604516257}, 'kpi8': {'name': 'Z07_T', 'type': 'avg_dev', 'target': [19, 24], 'kpi': 0.7200224910693038}, 'kpi16': {'name': 'Z15_T', 'type': 'avg_dev', 'target': [19, 24], 'kpi': 2.297474883555688}, 'kpi17': {'name': 'Z16_T', 'type': 'avg_dev', 'target': [19, 24], 'kpi': 1.40532108250742}, 'kpi18': {'name': 'Z17_T', 'type': 'avg_dev', 'target': [19, 24], 'kpi': 0.0675676410001173}, 'kpi19': {'name': 'Z18_T', 'type': 'avg_dev', 'target': [19, 24], 'kpi': 0.4480624623851421}, 'kpi20': {'name': 'Z19_T', 'type': 'avg_dev', 'target': [19, 24], 'kpi': 0.030272357503866}, 'kpi21': {'name': 'Z20_T', 'type': 'avg_dev', 'target': [19, 24], 'kpi': 0.6901404546664289}, 'kpi26': {'name': 'Z25_T', 'type': 'avg_dev', 'target': [19, 24], 'kpi': 0.006139037759710159}, 'kpi27': {'name': 'Z01_T', 'type': 'tot_viol', 'target': [19, 24], 'kpi': 141}, 'kpi28': {'name': 'Z02_T', 'type': 'tot_viol', 'target': [19, 24], 'kpi': 89}, 'kpi29': {'name': 'Z03_T', 'type': 'tot_viol', 'target': [19, 24], 'kpi': 84}, 'kpi30': {'name': 'Z04_T', 'type': 'tot_viol', 'target': [19, 24], 'kpi': 97}, 'kpi31': {'name': 'Z05_T', 'type': 'tot_viol', 'target': [19, 24], 'kpi': 113}, 'kpi32': {'name': 'Z06_T', 'type': 'tot_viol', 'target': [19, 24], 'kpi': 121}, 'kpi33': {'name': 'Z07_T', 'type': 'tot_viol', 'target': [19, 24], 'kpi': 122}, 'kpi41': {'name': 'Z15_T', 'type': 'tot_viol', 'target': [19, 24], 'kpi': 200}, 'kpi42': {'name': 'Z16_T', 'type': 'tot_viol', 'target': [19, 24], 'kpi': 200}, 'kpi43': {'name': 'Z17_T', 'type': 'tot_viol', 'target': [19, 24], 'kpi': 29}, 'kpi44': {'name': 'Z18_T', 'type': 'tot_viol', 'target': [19, 24], 'kpi': 193}, 'kpi45': {'name': 'Z19_T', 'type': 'tot_viol', 'target': [19, 24], 'kpi': 16}, 'kpi46': {'name': 'Z20_T', 'type': 'tot_viol', 'target': [19, 24], 'kpi': 196}, 'kpi51': {'name': 'Z25_T', 'type': 'tot_viol', 'target': [19, 24], 'kpi': 8}}
    for key, val in kpi.items():
        min_v, max_v = min_kpis[key]['kpi'], max_kpis[key]['kpi']
        if val["type"] not in cur_kpi: cur_kpi[val["type"]] = {}
        if val["name"] not in cur_kpi[val["type"]]: cur_kpi[val["type"]][val["name"]] = 0
        cur_kpi[val["type"]][val["name"]] += val["kpi"]/max(1.0, max_v-min_v)
    return cur_kpi

def objective_compare(kpi1, kpi2):
    obj1 = np.sum([v for v in kpi1['avg'].values()]) 
    obj2 = np.sum([v for v in kpi2['avg'].values()]) 
    if obj1 > obj2: return -1.0
    elif obj1 < obj2: return 1.0
    else: return 0.0

def constraint_violate(kpis):
    if "avg_dev" in kpis: return np.sum([abs(v) for v in kpis['avg_dev'].values()]) > 0.0
    return False

def constraint_violate_compare(kpi1, kpi2):
    # preference = 0
    # for key in kpi1['avg_dev'].keys():
    #     val1 = kpi1['avg_dev'][key]
    #     val2 = kpi2['avg_dev'][key]
    #     if val1 > val2 and preference <= 0: preference = -1.0
    #     elif val1 > val2 and preference == 1.0:
    #         preference = 0.0
    #         break
    #     elif val1 < val2 and preference >= 0: preference = 1.0
    #     elif val1 < val2 and preference == -1.0:
    #         preference = 0.0
    #         break
    val1 = np.sum([v for v in kpi1['avg_dev'].values()])
    val2 = np.sum([v for v in kpi2['avg_dev'].values()])
    preference = (1.0 if val1 < val2 else (0.0 if val1 == val2 else -1.0))
    return preference

def get_info_from_trajectory(trajectory, _len_traj, min_kpis, max_kpis):
    traj_state = []
    cur_kpis = {}
    for i in range(_len_traj):
        state, action, reward, kpis, next_state = trajectory[4*i], trajectory[4*i+1], trajectory[4*i+2], trajectory[4*i+3], trajectory[4*i+4]
        # reward_state = np.concatenate((state, action, next_state))
        # reward_state = np.concatenate((state)
        traj_state.append(next_state)
        cur_kpis = add_kpi(cur_kpis, kpis, min_kpis, max_kpis)
    state = np.concatenate(traj_state)
    return state, cur_kpis

def compare_trajectory(trajectory1, trajectory2, _len_traj, min_kpis, max_kpis):
    state1, kpis1 = get_info_from_trajectory(trajectory1, _len_traj, min_kpis, max_kpis)
    state2, kpis2 = get_info_from_trajectory(trajectory2, _len_traj, min_kpis, max_kpis)
    preference = 0
    if (not constraint_violate(kpis1)) and (not constraint_violate(kpis2)): preference = objective_compare(kpis1, kpis2)
    elif constraint_violate(kpis1) and (not constraint_violate(kpis2)): preference = -1.0
    elif (not constraint_violate(kpis1)) and constraint_violate(kpis2): preference = 1.0
    else: preference = constraint_violate_compare(kpis1, kpis2)
    # obj1 = np.sum([v for v in kpis1['avg'].values()]) 
    # obj2 = np.sum([v for v in kpis2['avg'].values()]) 
    # vio1 = np.sum([v for v in kpis1["avg_dev"].values()])
    # vio2 = np.sum([v for v in kpis2["avg_dev"].values()])
    # r1 = -obj1-vio1
    # r2 = -obj2-vio2
    # preference = (1.0 if r1 > r2 else (-1.0 if r1 < r2 else 0.0))
    mu = ([1.0, 0.0] if preference==1.0 else ([0.5, 0.5] if preference == 0.0 else [0.0, 1.0]))
    return state1, state2, mu
        

def sample_trajectory(env, building_name, controller=None):
    building_idx = buildings_list.index(building_name)
    done = False
    state = env.reset()
    step = 0
    trajectory = [state]
    rule_controller = controller_list[building_idx]
    noisy_delta = np.random.choice([1.0, 0.7, 0.5, 0.3, 0.0])
    # trajectory.append(state)
    while not done:
        # if controller is None: 
        control = env.unwrapped.sample_random_action()
        random_actions = env.transform_action(control)
        if controller is not None:
            actions, _ = controller.predict(state, deterministic=True)
            # _,hour,_,_ = env.unwrapped.get_date()
            # outputs = env.inverse_transform_state(state)
            # control = rule_controller(env.action_keys, step)(outputs, control_values[building_idx], hour)
            # rule_actions = env.transform_action(control)
            actions = [noisy_delta*ra+(1.0-noisy_delta)*a for ra,a in zip(random_actions, actions)]
        else: actions = random_actions
            
        state, reward, done, info = env.step(actions)
        kpis = env.unwrapped.get_kpi(start_ind=step, end_ind=step+1)
        trajectory.extend([actions, reward, kpis, state])
        # print("control: ", control, "outputs: ", env.outputs)
        # print("state: ", state, " ori:", env.inverse_transform_state(state))
        step += 1
    return trajectory


def sample_preferences(trajectory1, trajectory2, min_kpis, max_kpis, num_preferences=8, traj_list=None):
    trajectory_len1 = len(trajectory1) // 4
    trajectory_len2 = len(trajectory2) // 4
    if traj_list is None: traj_list = len_traj_list
    preference_pairs = [[] for _ in traj_list]
    for i, _len_traj in enumerate(traj_list):
        for _ in range(num_preferences):
            start_idx1 = np.random.randint(trajectory_len1-_len_traj)
            start_idx2 = np.random.randint(trajectory_len2-_len_traj)
            traj1, traj2 = trajectory1[start_idx1*4:(start_idx1+_len_traj)*4+1], trajectory2[start_idx2*4:(start_idx2+_len_traj)*4+1]
            state1, state2, mu = compare_trajectory(traj1, traj2, _len_traj, min_kpis, max_kpis)
            preference_pairs[i].append(np.concatenate((state1, state2, np.array(mu))))
    return preference_pairs


def generate_offline_data_worker(is_remote, building_name, min_kpis, max_kpis, min_outputs, max_outputs, round, preference_idx_list, device):
    if is_remote: 
        data_loc = os.environ['AMLT_DATA_DIR'] + "/data/offline_data/{}/preferences_data/{}/"
        traj_data_loc = f"{os.environ['AMLT_DATA_DIR']}/data/offline_data/{building_name}/traj_data/"
        model_loc = os.environ['AMLT_DATA_DIR'] + "/data/models/{}/SAC_bs_simulator_seed{}/best_model/best_model.zip"
    else: 
        data_loc = "data/offline_data/{}/preferences_data/{}/"
        traj_data_loc = f"data/offline_data/{building_name}/traj_data/"
        model_loc = "data/models/{}/SAC_bs_simulator_seed{}/best_model/best_model.zip"
    env_rl = StableBaselinesRLWrapper(building_name, min_kpis, max_kpis, min_outputs, max_outputs, reward_func)
    for _len_traj in len_traj_list: os.makedirs(data_loc.format(building_name, _len_traj), exist_ok=True)
    os.makedirs(traj_data_loc, exist_ok=True)
    for i in preference_idx_list: 
        traj_file_loc = f"{traj_data_loc}/{round}_{i}.pkl"
        if not os.path.exists(traj_file_loc):
            sample_seed1 = np.random.choice([7, 13, 17, 19])
            sample_seed2 = np.random.choice([7, 13, 17, 19])
            controller1 = SAC.load(model_loc.format(building_name, sample_seed1), device=device)
            controller2 = SAC.load(model_loc.format(building_name, sample_seed2), device=device)
            trajectory1 = sample_trajectory(env_rl, building_name, controller=controller1)
            trajectory2 = sample_trajectory(env_rl, building_name, controller=controller2)
            trajectories = [trajectory1, trajectory2]
            with open(traj_file_loc, "wb") as f:
                pickle.dump(trajectories, f)
        # else:
        #     with open(traj_file_loc, "rb") as f:
        #         trajectories = pickle.load(f)
        #     trajectory1, trajectory2 = trajectories[0], trajectories[1]
        # preference_pairs1 = sample_preferences(trajectory1, trajectory2, min_kpis, max_kpis, num_preferences=perference_pairs_per_sample)
        # preference_pairs2 = sample_preferences(trajectory1, trajectory1, min_kpis, max_kpis, num_preferences=perference_pairs_per_sample)
        # preference_pairs3 = sample_preferences(trajectory2, trajectory2, min_kpis, max_kpis, num_preferences=perference_pairs_per_sample)
        
        # for j, _len_traj in enumerate(len_traj_list):
        #     _data_loc = data_loc.format(building_name, _len_traj)
        #     file_loc = f'{_data_loc}/preference_data_{round}_{i}.pkl'
        #     with open(file_loc, 'wb') as f:
        #         np.save(f, preference_pairs1[j]+preference_pairs2[j]+preference_pairs3[j])
        print(f"round {round}, preference {i+1} done!")
    print(f"round {round} done!")
    env_rl.close()


len_traj = 1
len_traj_list = [4, 8]
# len_traj_list = list(range(1, 9))
num_workers = 8
preference_per_round = 20
perference_pairs_per_sample = 102400

# buildings_list = ["ApartmentsThermal-v0", "ApartmentsGrid-v0", "Apartments2Thermal-v0",
#                   "Apartments2Grid-v0", "OfficesThermostat-v0", "MixedUseFanFCU-v0",
#                   "SeminarcenterThermostat-v0", "SeminarcenterFull-v0", "SimpleHouseRad-v0",
#                   "SimpleHouseRSla-v0", "SwissHouseRSlaW2W-v0", "SwissHouseRSlaTank-v0"] 

import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--amlt', action='store_true', help="remote execution on amlt")
parser.add_argument('--building', type=str, help='building name', required=True)
parser.add_argument('--round', type=str, help='round', default="")
parser.add_argument('--idx', type=str, help='idx', default="")
parser.add_argument('--device', type=str, help='device', default="cpu")



if __name__ == "__main__":
    args = parser.parse_args()
    building_name = args.building
    min_kpis, max_kpis, min_outputs, max_outputs = collect_baseline_kpi(building_name, args.amlt)
    if args.round == "": rounds_list = list(range(num_workers))
    else: rounds_list = [int(c) for c in args.round.split(",")]
    
    if args.idx == "": idx_list = list(range(preference_per_round))
    else: idx_list = [int(c) for c in args.idx.split(",")]
    
    if (not building_name.startswith("Simple")) and (not building_name.startswith("Swiss")):
        for i in rounds_list: 
            generate_offline_data_worker(args.amlt, building_name, min_kpis, max_kpis, min_outputs, max_outputs, i, idx_list, args.device)
    else:
        jobs = []
        for i in rounds_list:
            p = mp.Process(target=generate_offline_data_worker, args=(args.amlt, building_name, min_kpis, max_kpis, min_outputs, max_outputs, i, idx_list, args.device))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()