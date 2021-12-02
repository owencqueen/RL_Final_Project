import torch
import numpy as np

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

def testing_reward(state_vec):
    return (state_vec[0] - state_vec[1]) ** 2

def headway_reward(state_vec):
    desired_headway = state_vec[2] * state_vec[0] + 7
    actual_headway = state_vec[3]

    if (actual_headway) < 0:
        return -1.0 * 1e6 # Very large negative reward for collision

    else:
        # Trying to keep the exact distance:
        return -1.0 * np.abs(actual_headway - desired_headway)

# def speed_reward(state_vec):
#     pass