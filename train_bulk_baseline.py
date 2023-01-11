import argparse
import multiprocessing as mp
import os
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--amlt', action='store_true', help="remote execution on amlt")
parser.add_argument('--building', type=str, help='building name', required=True)
parser.add_argument('--seed', type=str, help='seed', default="7")

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
            cmd_prefix = f"python d3rl_baselines.py --building {building_name} --seed {seed} --iter 100 "
            if args.amlt: cmd_prefix += "--amlt "
            # cmds.extend([cmd_prefix+"--algo TD3PlusBC", cmd_prefix+"--algo CQL", cmd_prefix+"--algo MOPO"])
            cmds.extend([cmd_prefix+"--algo MOPO"])
        
    device_count = torch.cuda.device_count()
    # for cmd in cmds: run(cmd)
    if building_name.startswith("Swiss") or building_name.startswith("Simple"):
        jobs = []
        for i, cmd in enumerate(cmds):
            device_idx = i % device_count
            cmd += " --device cuda:%i"%device_idx
            p = mp.Process(target=run, args=(cmd,))
            jobs.append(p)
            p.start()
        for proc in jobs:
            proc.join()
    else:
        for cmd in cmds: run(cmd + " --device cuda:0")