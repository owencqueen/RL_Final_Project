import torch 
import random
import numpy as np

from collections import deque
from tqdm import trange
from typing import Union

from Blazer_Model import Model

from reward_funcs import speed_match_reward, raw_speed_reward

mydevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Step the blazer model simulation. Must call reset first. 

action: np.ndarray (3,) (1-dimensional)
    0: Engine power (W)
    1: Motor Power (W)
    2: Foundation Brakes Power (W)

ret: state (np.ndarray)
    0 : ego_speed (m/s)
    1 : driver_set_speed (m/s)
    2 : time_gap (s)
    3 : target_veh_dist (m)
    4 : target_veh_speed (m/s) (relative speed)
    5 : real engine power (W)
    6 : real motor power (W)
    7 : ego-vehicle jerk (m/s^3)
    8 : road grade (gradians)
    9 : SOC (% 0-100) 
    10: Electric motor speed (RPM)
    11: Engine speed (RPM)
    12: Distance to next intersection (m)
    13: Next intersection's current phase (0=red,1=green,2=amber)
    14: Next intersection's time to next phase (deca-seconds)
    15: Next intersection's next phase (0=red,1=green,2=amber)
    16: Next intersection is_valid (1=valid, 0=invalid)
"""


reward_func_options = {
    1: raw_speed_reward,
    2: speed_match_reward
}

class TDBufferEntry:
    '''
    One entry in the replay buffer for TD learning framework (DDPG)
        - Can hold more variables for advanced replay buffer methods
    '''
    def __init__(self, s1, a1, r1, s2):
        self.s1 = s1
        self.a1 = a1
        self.r1 = r1
        self.s2 = s2

class MCBufferEntry:
    '''
    One entry in the replay buffer for MC learning framework (REINFORCE)
        - Can hold more variables for advanced replay buffer methods
    '''
    def __init__(self, s1, a1, r1, s2):
        self.s1 = s1
        self.a1 = a1
        self.r1 = r1
        self.s2 = s2

class ReplayBuffer:
    '''
    Replay buffer class to manage sampling, entries, etc.
    '''

    def __init__(self, max_size, device = None):

        # Use deque because O(1) efficiency in pop front and pop back
        self.buffer = deque()
        self.max_size = max_size
        self.device = device

    def TD_entry(self, s1, a1, r1, s2):

        entry = TDBufferEntry(s1, a1, r1, s2)
        self.buffer.appendleft(entry)

        # Adjust buffer automatically
        if len(self.buffer) > self.max_size:
            self.buffer.pop()

    def MC_entry(self, s1, a1, r1):

        entry = MCBufferEntry(s1, a1, r1)
        self.buffer.appendleft(entry)

        # Adjust buffer automatically
        if len(self.buffer) > self.max_size:
            self.buffer.pop()

    def TD_sample(self, batch_size): 
        '''
        Args:
            batch_size (int): Size of batch to be sampled

        :rtype: (Tensor, Tensor, Tensor, Tensor)
        '''

        true_batch_size = min(batch_size, len(self.buffer))

        samp = random.sample(self.buffer, k = true_batch_size)

        # Decompose into tensors:
        s1_tensors = []
        a1_tensors = []
        r1_tensors = []
        s2_tensors = []
        # Use one loop instead of 4 list comprehensions
        for i in range(true_batch_size):
            s1_tensors.append(samp[i].s1)
            a1_tensors.append(samp[i].a1)
            r1_tensors.append(samp[i].r1)
            s2_tensors.append(samp[i].s2)


        s1 = torch.autograd.Variable(torch.stack(s1_tensors)).to(self.device)
        a1 = torch.autograd.Variable(torch.stack(a1_tensors)).to(self.device)
        r1 = torch.autograd.Variable(torch.stack(r1_tensors)).to(self.device)
        s2 = torch.autograd.Variable(torch.stack(s2_tensors)).to(self.device)
        
        return s1, a1, r1, s2

class Environment:
    '''
    Wrapper on the Blazer Model environment that performs preparation/
        sampling for RL models
        - Performs batching, with potential for online learning

    Args:

    '''
    def __init__(self, 
            drive_trace = 'US06',
            reward_func_option = 1,
            max_episodes_replay_buffer = 50,
            device = None
            ):

        self.env = Model(automatic_control=False)
        self.drive_trace = drive_trace
        self.device = device

        # Initialize Replay buffer:
        self.replay_buffer = ReplayBuffer(max_size = max_episodes_replay_buffer, device = self.device)

        self.max_ep_buffer = max_episodes_replay_buffer

        self.reward_weights = reward_weights
        self.optimize_mask = optimize_mask


    def TD_run_episode(self, 
            trainer,
            gather_buffer = True, 
            cutoff = None,
            SOC = 95,
            update_freq = 1,
            explore_noise_weight: Union[float, torch.Tensor] = None):
        s1 = torch.autograd.Variable(torch.from_numpy(self.env.reset(drive_trace = self.drive_trace, SOC = SOC))).float()
        reward_sum = 0

        num_steps = len(self.env)

        for i in range(num_steps):
            if explore_noise_weight is None: 
                a1 = trainer.explore_action(s1)
            else:
                a1 = trainer.explore_action(s1, explore_noise_weight = explore_noise_weight)
            #print(a1)
            action = a1.detach().clone().numpy()
            #print(action)
            s2 = self.env.step(action.astype(np.double))

            r1 = self.reward(s2) # Reward calculated via next-state variables
            s2 = torch.autograd.Variable(torch.from_numpy(s2)).float()
            reward_sum += r1

            if gather_buffer:
                # Add to buffer
                self.replay_buffer.TD_entry(
                    s1, 
                    a1, 
                    torch.tensor(r1), 
                    s2)

            # Optimize each step (as following with DDPG algorithm)
            if (i + 1) % update_freq == 0:
                trainer.optimize(self.replay_buffer)
            

            if cutoff is not None:
                if cutoff < i: # Cuts the episode early
                    #print(action)
                    break

            s1 = s2.detach().clone()

        return reward_sum

    def run_step(self, agent):
        pass

    def reward(self, state):
        r = raw_speed_reward(state)
        return r

        