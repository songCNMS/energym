from energym.envs.env import StepWrapper
from gym import spaces
import numpy as np
from collections import OrderedDict


def transform(val, l, u):
    return (val - l) / max((u - l), 1.0)

def inverse_transform(val, l, u):
    return val*max((u - l), 1.0) + l

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
    def __init__(self, env, reward_function):
        super(StableBaselinesRLWrapper, self).__init__(env, reward_function)
        self.action_keys = [a_name for a_name in env.input_keys if a_name.find("Thermostat") >= 0]
        n_actions = len(self.action_keys)
        n_states = len(env.output_keys)
        self.action_space = spaces.Box(low=0.0, high=1.0,
                                        shape=(n_actions,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=0.0,
                                            shape=(n_states,), dtype=np.float32)
        assert callable(reward_function)
        self.reward_function = reward_function
        
    def inverse_transform_action(self, actions):
        control =  {a_name: [inverse_transform(a, self.env.input_specs[a_name]['lower_bound'], self.env.input_specs[a_name]['upper_bound'])]
                        for a, a_name in zip(actions, self.action_keys)}
        control['Bd_Ch_EV1Bat_sp'] = [0.0]
        control['Bd_Ch_EV2Bat_sp'] = [0.0]
        return control
    
    def transform_action(self, actions):
        return [transform(actions[a_name][0], self.env.input_specs[a_name]['lower_bound'], self.env.input_specs[a_name]['upper_bound'])
                        for a_name in self.action_keys]
    
    def inverse_transform_state(self, state):
        return OrderedDict({a_name: inverse_transform(a, self.env.output_specs[a_name]['lower_bound'], self.env.output_specs[a_name]['upper_bound'])
                        for a, a_name in zip(state, self.env.output_keys)})
    
    def transform_state(self, state):
        return [transform(state[a_name], self.env.output_specs[a_name]['lower_bound'], self.env.output_specs[a_name]['upper_bound'])
                        for a_name in self.env.output_keys]

    def reset(self):
        outputs = self.env.get_output()
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
        reward = self.reward_function(outputs)
        done = (self.env.time >= self.env.stop_time)
        info = {}
        self.state = self.transform_state(outputs)
        return self.state, reward, done, info
