import os

# buildings_list = ["ApartmentsThermal-v0", "ApartmentsGrid-v0", "Apartments2Thermal-v0",
#                   "Apartments2Grid-v0", "OfficesThermostat-v0", "MixedUseFanFCU-v0",
#                   "SeminarcenterThermostat-v0", "SeminarcenterFull-v0", "SimpleHouseRad-v0",
#                   "SimpleHouseRSla-v0", "SwissHouseRSlaW2W-v0", "SwissHouseRSlaTank-v0"] 

description = "Energym Preference Data"
suffix = ""
# seeds_list = ["7,13,17,19"]
buildings = ["ApartmentsThermal-v0", "ApartmentsGrid-v0", "Apartments2Thermal-v0",
             "Apartments2Grid-v0", "OfficesThermostat-v0", "MixedUseFanFCU-v0",
             "SeminarcenterThermostat-v0", "SeminarcenterFull-v0", "SimpleHouseRad-v0",
             "SimpleHouseRSla-v0", "SwissHouseRSlaW2W-v0", "SwissHouseRSlaTank-v0"]
file_name = "preference_data"
res_file_name = "amulet_preference_data"

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
  remote_dir: energym_v2/

jobs:
"""
suffix_in_name = suffix.replace(" ", "").replace("-", "")
for building in buildings:
  if not building.startswith("Apartments"): 
    cmd += f"""
- name: {building}-{file_name}-{suffix_in_name}-round_all-idx_all
  sku: G1
  command:
  - python {file_name}.py --building {building} --amlt {suffix} --device cuda:0
"""
#   else:
#     for round in range(8):
#         for idx in range(0, 50, 10):
#             idx_list = ','.join([str(idx+i) for i in range(10)])
#             cmd += f"""
# - name: {building}-{file_name}-{suffix_in_name}-round_{round}-idx_{idx_list}
#   sku: C1
#   command:
#   - python {file_name}.py --building {building} --amlt {suffix} --round {round} --idx {idx_list} --device cuda:0
# """

with open(f"{res_file_name}.yaml", 'w') as f:
  f.write(cmd)

