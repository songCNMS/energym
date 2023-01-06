import os

# buildings_list = ["ApartmentsThermal-v0", "ApartmentsGrid-v0", "Apartments2Thermal-v0",
#                   "Apartments2Grid-v0", "OfficesThermostat-v0", "MixedUseFanFCU-v0",
#                   "SeminarcenterThermostat-v0", "SeminarcenterFull-v0", "SimpleHouseRad-v0",
#                   "SimpleHouseRSla-v0", "SwissHouseRSlaW2W-v0", "SwissHouseRSlaTank-v0"] 

description = "Energym Train Baseline Reward NO Dynamics"
suffix = ""
seeds_list = ["7,13,17,19"]
buildings = ["ApartmentsThermal-v0", "ApartmentsGrid-v0", "Apartments2Thermal-v0",
             "Apartments2Grid-v0", "OfficesThermostat-v0", "MixedUseFanFCU-v0",
             "SeminarcenterThermostat-v0", "SeminarcenterFull-v0", "SimpleHouseRad-v0",
             "SimpleHouseRSla-v0", "SwissHouseRSlaW2W-v0", "SwissHouseRSlaTank-v0"]
file_name = "train_bulk"
res_file_name = "amulet_rmbs_simulator_config"

cmd=f"""
description: {description}

target:
  service: sing
  name: msrresrchvc
  vc: gcr-singularity-resrch

environment:
  image: energym_gcr:sing2
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
- name: {building}-{file_name}-{suffix_in_name}-seed{seed}
  sku: G1-V100
  command:
  - python {file_name}.py --building {building} --amlt {suffix} --seed {seed}
"""

with open(f"{res_file_name}.yaml", 'w') as f:
    f.write(cmd)

