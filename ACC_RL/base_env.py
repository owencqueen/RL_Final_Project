import torch
import numpy as np

from Blazer_Model import Model

"""
Step the blazer model simulation. Must call reset first. 

action: np.ndarray (3,) (1-dimensional)
    0: Engine power (W)
    1: Motor Power(W)
    2: Foundation Brakes Power (W)

ret: state (np.ndarray)
    0 : ego_speed (m/s)
    1 : driver_set_speed (m/s)
    2 : time_gap (s)
    3 : target_veh_dist (m)
    4 : target_veh_speed (m/s) (relative speed)
    5 : real engine power (W)
    6 : real motor power (W)
    7 : electric fuel flow (L/hr) [TODO: verify unit]
    8 : e10 fuel flow (L/hr) [same unit as electric fuel flow]
    9 : ego-vehicle jerk (m/s^3)
    10: road grade (gradians)
    11: Distance to next intersection (m)
    12: Next intersection's current phase (0=red,1=green,2=amber)
    13: Next intersection's time to next phase (deca-seconds)
    14: Next intersection's next phase (0=red,1=green,2=amber)
    15: Next intersection is_valid (1=valid, 0=invalid)
"""

basic_opt_mask = torch.zeros(16, dtype=torch.bool)
basic_opt_mask[3] = 1 # Basic optimization mask only tries to optimize target vehicle distance

def testing_reward(state_vec):
    return (state_vec[0] - state_vec[1]) ** 2


class Environment:
    '''
    Wrapper on the Blazer Model environment that performs preparation/
        sampling for RL models
        - Performs batching, with potential for online learning

    Args:

    '''
    def __init__(self, 
            drive_trace = 'US06',
            feature_mask = None,
            optimize_mask = None,
            reward_weights = None,
            loss_p = 2,
            max_episodes_replay_buffer = 50,
            ):

        self.env = Model(automatic_control=False)
        self.drive_trace = drive_trace

        if isinstance(feature_mask, str): # String options for feature mask
            if feature_mask == 'safety':
                self.feature_mask = torch.tensor(([1] * 12 + [0] * 4), dtype=bool) 

        # Establish replay buffer:
        self.replay_buffer_X = []
        self.replay_buffer_y = []
        self.replay_buffer_actions = []

        self.reward_weights = reward_weights
        self.optimize_mask = optimize_mask


    def run_episode(self, agent, gather_buffer = True, cutoff = None):
        state = torch.autograd.Variable(torch.from_numpy(self.env.reset(drive_trace = self.drive_trace))).float()
        discount = 0.9
        #state = torch.zeros([0] * 16, dtype=float)
        reward_sum = 0
        for i in range(len(self.env)):
            #action = np.zeros((3,), dtype=np.double)
            action = agent(state)
            state = torch.autograd.Variable(torch.from_numpy(self.env.step(action.detach().numpy().astype(np.double)))).float()

            y = self.reward(state.detach().numpy())

            reward_sum += y

            if gather_buffer:
                self.replay_buffer_X.append(torch.tensor(state, dtype = torch.double))
                self.replay_buffer_y.append(reward_sum) # Get returns as we go
                self.replay_buffer_actions.append(action)

            if cutoff is not None:
                if cutoff < i: # Cuts the episode early
                    print(action)
                    break

        return reward_sum

    def retrieve_buffer(self):
        pass

    def reward(self, state):
        return testing_reward(state)

        