import argparse
import multiprocessing as mp
import os


parser = argparse.ArgumentParser()
parser.add_argument('--amlt', action='store_true', help="remote execution on amlt")
parser.add_argument('--building', type=str, help='building name', required=True)
parser.add_argument('--seed', type=int, help='seed', default=7)

def run(cmd):
    print(cmd)
    os.system(cmd)
    return

if __name__ == "__main__":
    args = parser.parse_args()
    building_name = args.building
    seed = args.seed
    cmd_prefix = f"python train.py --building {building_name} --seed {seed} "
    if args.amlt: cmd_prefix += "--amlt "
    cmds = [cmd_prefix, cmd_prefix+" --rm", cmd_prefix+" --dm", cmd_prefix+" --dm --rm"]
    jobs = []
    for cmd in cmds:
        p = mp.Process(target=run, args=(cmd,))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()