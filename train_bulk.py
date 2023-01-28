import argparse
import multiprocessing as mp
import os
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--amlt', action='store_true', help="remote execution on amlt")
parser.add_argument('--building', type=str, help='building name', required=True)
parser.add_argument('--seed', type=str, help='seed', default="7")
parser.add_argument('--alpha', type=str, help='alpha', default="1")
parser.add_argument('--traj', type=str, help='traj', default="1")

def run(cmd):
    print(cmd)
    os.system(cmd)
    return

if __name__ == "__main__":
    args = parser.parse_args()
    building_name_list = args.building.split(",")
    seed_list = [int(s) for s in args.seed.split(",")]
    cmds = []
    for building_name in building_name_list:
        for seed in seed_list:
            cmd_prefix = f"python train.py --building {building_name} --seed {seed} --iter 200 "
            if args.amlt: cmd_prefix += "--amlt "
            for alpha in args.alpha.split(","):
                cmds.extend([cmd_prefix+f"--rm dnn --dm --alpha {alpha} --traj {traj}" for traj in [int(c) for c in args.traj.split(",")]])
    device_count = torch.cuda.device_count()
    # if building_name.startswith("Swiss") or building_name.startswith("Simple"):
    # jobs = []
    # for i, cmd in enumerate(cmds):
    #     device_idx = i % device_count
    #     cmd += " --device cuda:%i"%device_idx
    #     p = mp.Process(target=run, args=(cmd,))
    #     jobs.append(p)
    #     p.start()
    # for proc in jobs:
    #     proc.join()
    # else:
    for i, cmd in enumerate(cmds): 
        device_idx = np.random.randint(device_count)
        cmd += " --device cuda:%i"%device_idx
        run(cmd)