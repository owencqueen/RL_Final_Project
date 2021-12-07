import sys, pickle; sys.path.append('..')
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from torch.autograd import Variable
import torch.autograd as autograd
import numpy as np

from REINFORCE.REINFORCE_PG import REINFORCE_trainer
from base_env import Environment
from Blazer_Model import Model
from reward_funcs import raw_speed_reward
from reward_funcs import speed_match_reward

def parse_argv():
    args_dict = {}
    args_dict['gamma'] = float(sys.argv[1])
    args_dict['lr'] = float(sys.argv[2])
    args_dict['drive_trace'] = sys.argv[3]
    args_dict['SOC'] = float(sys.argv[4])
    args_dict['epochs'] = int(sys.argv[5])
    args_dict['cutoff'] = int(sys.argv[6])
    args_dict['reward'] = int(sys.argv[7])
    args_dict['results_path'] = sys.argv[8]
    args_dict['model_path'] = sys.argv[9]

    return args_dict


# def evaluate_policy(policy, env, eval_episodes = 10):
#     reward_sum = 0.0
#     state = env.reset()
#     for _ in range(eval_episodes):
        
#         done = False
#         #while not done:
#         for _ in trange(len(env)):
#             action, log_prob = policy.select_action(np.array(state))
#             action = action.astype(np.double)
    
#             next_state = env.step(action)
#             reward = get_reward(next_state)
#             next_state = torch.autograd.Variable(torch.from_numpy(next_state)).float()
#             reward_sum += reward
       
#         print("avg reward is: {0}".fomat(reward_sum))

# def render_policy(policy):
#     state = env.reset()
#     done = False
#     while not done:
#         env.render()
#         action, log_prob = policy.select_action(np.array(state))
#         action = action.astype(np.double)
#         next_state = env.step(action)
#         reward = get_reward(next_state)
#         next_state = torch.autograd.Variable(torch.from_numpy(next_state)).float()
#     env.close()

def get_reward(state, opt):
    if opt == 1:
        r = raw_speed_reward(state)
    else:
        r = speed_match_reward(state)
    return r




def main():
    env = Model(automatic_control=False)

    args_dict = parse_argv()

    SOC = args_dict['SOC']
    drive_trace = args_dict['drive_trace']

    # Set:
    state_dim = 11
    action_dim = 3
    hidden_dims = 32

    policy = REINFORCE_trainer(
        state_dim, 
        hidden_dims, 
        action_dim,
        lr_pi = args_dict['lr'],
        gamma = args_dict['gamma']
        )

    max_episodes = args_dict['epochs']
    max_steps = args_dict['cutoff']

    total_episodes = 0
    save_rewards = []

    policy.load_model(args_dict['model_path'])
    

    #while total_episodes < max_episodes:
    for total_episodes in trange(max_episodes):
        #state = torch.autograd.Variable(torch.from_numpy(env.reset())).float()
        state = env.reset(drive_trace = drive_trace, SOC = SOC)
        total_steps = 0
        trajectory = []
        episode_reward = 0

        while total_steps < max_steps:
            action, log_prob = policy.select_action(np.array(state))
            #action = action.astype(np.double)
            #action = torch.squeeze(action)
            action = np.squeeze(action.astype(np.double))
            #action = np.squeeze(action)
            next_state = env.step(action)
            #print('num nans', next_state)
            reward = get_reward(next_state, opt = args_dict['reward'])
            next_state = torch.autograd.Variable(torch.from_numpy(next_state)).float()
            trajectory.append([np.array(state), action, log_prob, reward, next_state])
            state = next_state
            episode_reward += reward
            total_steps += 1
            #print(total_steps)

        policy_loss = policy.train(trajectory)
        save_rewards.append(episode_reward)

    policy.dump_model(args_dict['model_path'])

    pickle.dump(save_rewards, open(args_dict['results_path'], 'wb'))


if __name__ == '__main__':
    main()
