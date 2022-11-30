from energym.envs.env import StepWrapper
from gym import spaces
import numpy as np
from collections import OrderedDict
from buildings_factory import *


def transform(val, l, u):
    return (val - l) / max((u - l), 1.0)

def inverse_transform(val, l, u):
    return min(u, max(l, val*max((u - l), 1.0) + l))

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
    def __init__(self, building_name, reward_function, eval=False):
        self.building_name = building_name
        building_idx = buildings_list.index(building_name)
        env = get_env(building_name, eval=eval)
        default_control = default_controls[building_idx]
        inputs = get_inputs(building_name, env)
        env.step(env.sample_random_action())
        super(StableBaselinesRLWrapper, self).__init__(env, reward_function)
        self.action_keys = [a_name for a_name in inputs]
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
        self.num_steps = int(self.env.stop_time - self.env.time) // control_frequency[building_idx] 
        
    def inverse_transform_action(self, actions):
        control =  {a_name: [inverse_transform(a, self.env.input_specs[a_name]['lower_bound'], self.env.input_specs[a_name]['upper_bound'])]
                        for a, a_name in zip(actions, self.action_keys)}
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

    def reset(self):
        if self.building_name.startswith("Simple") or self.building_name.startswith("Swiss"):
            self.env = get_env(self.building_name)
        else: self.env.reset()
        self.env.step(self.env.sample_random_action())
        outputs = self.env.get_output()
        self.cur_step = 0
        self.state = self.transform_state(outputs)
        return self.state
        
        
    def render(self, mode='console'):
        if mode != 'console':
           raise NotImplementedError()
        self.env.print_kpis()


    def close(self):
        self.env.close()

    def step(self, inputs):
        # print(inputs, type(inputs))
        ori_inputs = self.inverse_transform_action(inputs)
        outputs = self.env.step(ori_inputs)
        kpi = self.env.get_kpi(start_ind=self.cur_step, end_ind=self.cur_step+1)
        reward = self.reward_function(kpi)
        done = (self.env.time >= self.env.stop_time)
        info = {}
        self.state = self.transform_state(outputs)
        self.cur_step += 1
        return self.state, reward, done, info
