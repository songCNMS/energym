import os

# buildings_list = ["ApartmentsThermal-v0", "ApartmentsGrid-v0", "Apartments2Thermal-v0",
#                   "Apartments2Grid-v0", "OfficesThermostat-v0", "MixedUseFanFCU-v0",
#                   "SeminarcenterThermostat-v0", "SeminarcenterFull-v0", "SimpleHouseRad-v0",
#                   "SimpleHouseRSla-v0", "SwissHouseRSlaW2W-v0", "SwissHouseRSlaTank-v0"] 

description = "Energym Train with Learnt Dynamics and Reward Models"
suffix = "--rm --dm --iter 500"
seeds_list = [17]
buildings = ["SimpleHouseRSla-v0", "SwissHouseRSlaW2W-v0", "SwissHouseRSlaTank-v0"]

cmd="""
description: Energym Train using Learnt Reward and Dynamics Model

target:
  service: sing
  name: msrresrchvc
  vc: gcr-singularity-resrch

environment:
  image: energym_gcr:sing
  registry: resrchvc4cr.azurecr.io # any public registry can be specified here
  username: resrchvc4cr
  setup:
    - apt-get update

code:
  local_dir: $CONFIG_DIR/

data:
  local_dir: $CONFIG_DIR/data/
  remote_dir: energym/

jobs:
"""
suffix_in_name = suffix.replace(" ", "").replace("-", "")
for building in buildings:
    for seed in seeds_list:
        cmd += f"""
- name: {building}-{suffix_in_name}-seed{seed}
  sku: G2-V100
  command:
  - python train.py --building {building} --amlt {suffix} --seed {seed}
"""

with open("amult_config.yaml", 'w') as f:
    f.write(cmd)

