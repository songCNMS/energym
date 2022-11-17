import sys
sys.path.insert(0, "/home/energym/energym/")

from energym.examples.Controller import LabController
import energym
from energym.wrappers.downsample_outputs import DownsampleOutputs
from energym.wrappers.rescale_outputs import RescaleOutputs
from energym.wrappers.rl_wrapper import StableBaselinesRLWrapper
from stable_baselines3 import PPO


weather = "ESP_CT_Barcelona"
building_name = "Apartments2Grid-v0"

env = energym.make(building_name, weather=weather, simulation_days=30)

inputs = env.get_inputs_names()
print(inputs)
controller = LabController(control_list=inputs, lower_tol=0.3, upper_tol=0.8, nighttime_setback=True, nighttime_start=18, nighttime_end=6, nighttime_temp=18)

downsampling_dic = {}
lower_bound =  {}
upper_bound = {}
def reward(outputs):
    return outputs['Fa_E_self']

outputs = env.get_output()
# env_down = DownsampleOutputs(env, steps, downsampling_dic)
# env_down_res = RescaleOutputs(env_down, lower_bound, upper_bound)
env_down_RL = StableBaselinesRLWrapper(env, reward)
hour = 0
# for i in range(10):
#     control = controller.get_control(outputs, 21, hour)
#     control['Bd_Ch_EV1Bat_sp'] = [0.0]
#     control['Bd_Ch_EV2Bat_sp'] = [0.0]
#     state, reward, done, info = env_down_RL.step(control)
#     outputs = env_down_RL.inverse_transform_state(state)
#     print(reward, control, state)


model = PPO('MlpPolicy', env_down_RL, verbose=1, device='gpu:0').learn(100000)
model.save(building_name)


eval_env = energym.make(building_name, weather=weather, simulation_days=7, eval_mode=True)
eval_env_down_RL = StableBaselinesRLWrapper(eval_env, reward)

done = False
outputs = eval_env.get_output()
state = eval_env_down_RL.transform_state(outputs)
while not done:
    actions, _ = model.predict(state)
    print(actions, eval_env.time, eval_env.stop_time)
    eval_env_down_RL.render()
    state, reward, done, info = eval_env_down_RL.step(actions)
    
env.close()
eval_env.close()   
