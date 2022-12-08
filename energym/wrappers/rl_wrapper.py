import gym
from energym.envs.env import StepWrapper
from gym import spaces
import torch
import numpy as np
from energym.envs.utils.kpi import KPI
from collections import OrderedDict
from buildings_factory import *


def transform(val, l, u):
    return (val - l) / max((u - l), 1.0)

def inverse_transform(val, l, u):
    return min(u, max(l, val*max((u - l), 1.0) + l))


def state_distance(state1, state2):
    return np.abs(np.array(state1)-np.array(state2)).sum()

class RLWrapper(StepWrapper):
    r"""Transform steps outputs to have rl (gym like) outputs timesteps, i.e. add rewards, done, and info
    Changes the outputs structure, from outputs to outputs, reward, done, info, as in gym.
    **To be put at the end of all other wrappers.**
    Example::

        >>> TBD


    Args:
        env (Env): environment
        reward_function: reward function

    """

    def __init__(self, env, reward_function):
        super(RLWrapper, self).__init__(env)

        assert callable(reward_function)
        self.reward_function = reward_function

    def step(self, inputs):
        outputs = self.env.step(inputs)
        reward = self.reward_function(outputs)
        done = False
        info = {}
        return outputs, reward, done, info


class StableBaselinesRLWrapper(RLWrapper):
    r"""Transform steps outputs to have rl (gym like) outputs timesteps, i.e. add rewards, done, and info
    Changes the outputs structure, from outputs to outputs, reward, done, info, as in gym.
    **To be put at the end of all other wrappers.**
    Example::

        >>> TBD


    Args:
        env (Env): environment
        reward_function: reward function

    """
    metadata = {'render.modes': ['console']}
    def __init__(self, building_name, min_kpis, max_kpis, reward_function, dynamics_predictor=None, eval=False):
        self.building_name = building_name
        self.min_kpis = min_kpis
        self.max_kpis = max_kpis
        self.eval_mode = eval
        self.dynamics_predictor = dynamics_predictor
        building_idx = buildings_list.index(building_name)
        env = get_env(building_name, eval=eval)
        default_control = default_controls[building_idx]
        inputs = get_inputs(building_name, env)
        env.step(env.sample_random_action())
        self.outputs = env.get_output()
        super(StableBaselinesRLWrapper, self).__init__(env, reward_function)
        self.action_keys = [a_name for a_name in inputs if env.input_specs[a_name]["type"] == "scalar"]
        n_actions = len(self.action_keys)
        n_states = len(env.output_keys)
        self.action_space = spaces.Box(low=0.0, high=1.0,
                                        shape=(n_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                            shape=(n_states,), dtype=np.float32)
        assert callable(reward_function)
        self.reward_function = reward_function
        self.cur_step = 0
        self.default_control = default_control
        self.env = env
        self.controller = controller_list[building_idx]
        self.building_idx = building_idx
        self.hour = control_values[building_idx]
        self.num_steps = int(self.env.stop_time - self.env.time) // control_frequency[building_idx] 
        self.state = self.transform_state(self.outputs)
        self.max_episode_len = 100000
        self.kpis = KPI(self.env.kpi_options)
        
        
    def inverse_transform_action(self, actions):
        control =  {a_name: [inverse_transform(a, self.env.input_specs[a_name]['lower_bound'], self.env.input_specs[a_name]['upper_bound'])]
                        for a, a_name in zip(actions, self.action_keys)}
        for key in self.baseline_control.keys():
            if key not in self.action_keys: control[key] = self.baseline_control[key]
        control.update(self.default_control)
        return control
    
    def transform_action(self, actions):
        actions.update(self.default_control)
        return [transform(actions[a_name][0], self.env.input_specs[a_name]['lower_bound'], self.env.input_specs[a_name]['upper_bound'])
                        for a_name in self.action_keys]
    
    def inverse_transform_state(self, state):
        return OrderedDict({a_name: inverse_transform(a, self.env.output_specs[a_name]['lower_bound'], self.env.output_specs[a_name]['upper_bound'])
                        for a, a_name in zip(state, self.env.output_keys)})
    
    def transform_state(self, state):
        return [transform(state[a_name], self.env.output_specs[a_name]['lower_bound'], self.env.output_specs[a_name]['upper_bound'])
                        for a_name in self.env.output_keys]

    def seed(self, s):
        pass

    def reset(self):
        if self.building_name.startswith("Simple") or self.building_name.startswith("Swiss"):
            self.env = get_env(self.building_name, eval=self.eval_mode)
        else: self.env.reset()
        self.env.step(self.env.sample_random_action())
        self.outputs = self.env.get_output()
        self.cur_step = 0
        self.hour = control_values[self.building_idx]
        self.state = self.transform_state(self.outputs)
        return self.state
        
    def render(self, mode='console'):
        if mode != 'console':
           raise NotImplementedError()
        self.env.print_kpis()

    def close(self):
        if not(self.building_name.startswith("Simple") or self.building_name.startswith("Swiss")):
            self.env.close()

    def step(self, inputs):
        _,self.hour,_,_ = self.unwrapped.get_date()
        self.baseline_control = self.controller(self.action_keys, self.cur_step)(self.outputs, control_values[self.building_idx], self.hour)
        ori_inputs = self.inverse_transform_action(inputs)
        if self.dynamics_predictor is not None:
            with torch.no_grad():
                model_in = torch.from_numpy(np.concatenate((self.state, inputs))).reshape(1, -1).to(torch.float)
                next_state = self.dynamics_predictor(model_in.to(next(self.dynamics_predictor.parameters()).device))[0, :].cpu().detach().numpy()
            self.outputs = self.inverse_transform_state(next_state)
            self.kpis.add_observation(self.outputs)
            kpi = self.kpis.get_kpi(start_ind=self.cur_step, end_ind=self.cur_step+1)
            env_outputs = self.env.step(ori_inputs)
            env_state = self.transform_state(env_outputs)
            # state_gap = state_distance(next_state, env_state)
            state_gap = 0.0
        else:             
            self.outputs = self.env.step(ori_inputs)
            kpi = self.env.get_kpi(start_ind=self.cur_step, end_ind=self.cur_step+1)
            state_gap = 0.0

        self.state = self.transform_state(self.outputs)
        reward = self.reward_function(self.min_kpis, self.max_kpis, kpi, self.state) - state_gap
        done = ((self.unwrapped.time >= self.unwrapped.stop_time) | (self.cur_step >= self.max_episode_len))
        info = {}
        self.cur_step += 1
        return self.state, reward, done, info