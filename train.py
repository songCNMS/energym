import sys
sys.path.insert(0, "/home/energym/energym/")

from energym.examples.Controller import LabController
from energym.factory import make
from energym.wrappers.downsample_outputs import DownsampleOutputs
from energym.wrappers.rescale_outputs import RescaleOutputs
from energym.wrappers.rl_wrapper import StableBaselinesRLWrapper
from stable_baselines3 import PPO


weather = "ESP_CT_Barcelona"
building_name = "Apartments2Grid-v0"

env = make(building_name, weather=weather, simulation_days=28)

inputs = env.get_inputs_names()
print(inputs)
controller = LabController(control_list=inputs, lower_tol=0.3, upper_tol=0.8, nighttime_setback=True, nighttime_start=18, nighttime_end=6, nighttime_temp=18)

downsampling_dic = {}
lower_bound =  {}
upper_bound = {}
def reward_func(kpi):
    reward = 0.0
    constraint = 0.0
    for key, val in kpi.items():
        if val['type'] != 'avg': constraint -= val["kpi"]
        else: reward -= abs(val['kpi'] / 1000.0)  
    return reward + constraint

outputs = env.get_output()
env.print_kpis()
# env_down = DownsampleOutputs(env, steps, downsampling_dic)
# env_down_res = RescaleOutputs(env_down, lower_bound, upper_bound)
env_down_RL = StableBaselinesRLWrapper(env, reward_func)
hour = 0
# for i in range(10):
#     control = controller.get_control(outputs, 21, hour)
#     control['Bd_Ch_EV1Bat_sp'] = [0.0]
#     control['Bd_Ch_EV2Bat_sp'] = [0.0]
#     state, reward, done, info = env_down_RL.step(control)
#     outputs = env_down_RL.inverse_transform_state(state)
#     print(reward, control, state)


model = PPO('MlpPolicy', env_down_RL, verbose=1, device='auto')

model.learn(1000000)
model.save(building_name)

# model.load(building_name)

bs_eval_env = make(building_name, weather=weather, simulation_days=14, eval_mode=False)
eval_env = make(building_name, weather=weather, simulation_days=14, eval_mode=False)
eval_env_down_RL = StableBaselinesRLWrapper(eval_env, reward_func)


out_list = []
controls = []
reward_list = []

bs_out_list = []
bs_controls = []
bs_reward_list = []
    
bs_outputs = bs_eval_env.get_output()
done = False
outputs = eval_env.get_output()
state = eval_env_down_RL.transform_state(outputs)
step = 0
while not done:
    control = controller.get_control(bs_outputs, 21, hour)
    control['Bd_Ch_EV1Bat_sp'] = [0.0]
    control['Bd_Ch_EV2Bat_sp'] = [0.0]
    bs_controls +=[ {p:control[p][0] for p in control} ]
    bs_outputs = bs_eval_env.step(control)
    _,hour,_,_ = bs_eval_env.get_date()
    bs_out_list.append(bs_outputs)
    bs_reward = reward_func(bs_eval_env.get_kpi(start_ind=step, end_ind=step+1))
    bs_reward_list.append(bs_reward)
    
    actions, _ = model.predict(state)
    state, reward, done, info = eval_env_down_RL.step(actions)
    # eval_env_down_RL.render()
    outputs = eval_env_down_RL.inverse_transform_state(state)
    control = eval_env_down_RL.inverse_transform_action(actions)
    controls +=[ {p:control[p][0] for p in control} ]
    out_list.append(outputs)
    reward_list.append(reward)
    step += 1
   
print("RL KPIs") 
print(eval_env.get_kpi())
print("BS KPIs")
print(bs_eval_env.get_kpi())
env.close()
eval_env.close()
bs_eval_env.close()   


import pandas as pd
out_df = pd.DataFrame(out_list)
cmd_df = pd.DataFrame(controls)

bs_out_df = pd.DataFrame(bs_out_list)
bs_cmd_df = pd.DataFrame(bs_controls)


import matplotlib.pyplot as plt

f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,figsize=(10,15))#

ax1.plot(out_df['Z01_T'], 'r', bs_out_df['Z01_T'], 'b')
ax1.set_ylabel('Temp')
ax1.set_xlabel('Steps')

ax2.plot(out_df['P1_T_Thermostat_sp_out'], 'r--', bs_out_df['P1_T_Thermostat_sp_out'], 'b--')
ax2.set_ylabel('Temp_SP')
ax2.set_xlabel('Steps')

ax3.plot(out_df['Ext_T'], 'r', bs_out_df['Ext_T'], 'b')
ax3.set_ylabel('Temp')
ax3.set_xlabel('Steps')

ax4.plot(out_df['Fa_Pw_All'], 'r--', bs_out_df['Fa_Pw_All'], 'b--')
ax4.set_ylabel('Power')
ax4.set_xlabel('Steps')

ax5.plot(reward_list, 'r--', bs_reward_list, 'b--')
ax5.set_ylabel('Reward')
ax5.set_xlabel('Steps')

plt.subplots_adjust(hspace=0.4)

plt.show()
plt.savefig(f"{building_name}_RL.png")