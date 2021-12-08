import sys; sys.path.append('..')
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

torch.autograd.set_detect_anomaly(True)

SOC = 10
drive_trace = 'IM240'
reward_func_option = 1

def evaluate_policy(policy, env, eval_episodes = 10):
    reward_sum = 0.0
    state = env.reset()
    for _ in range(eval_episodes):
        
        done = False
        
        for _ in trange(len(env)):
            action, log_prob = policy.select_action(np.array(state))
            action = action.astype(np.double)
    
            next_state = env.step(action)
            reward = get_reward(next_state)
            next_state = torch.autograd.Variable(torch.from_numpy(next_state)).float()
            reward_sum += reward
       
        print("avg reward is: {0}".fomat(reward_sum))



def get_reward(state):
        r = raw_speed_reward(state)
        return r

def main():
    env = Model(automatic_control=False)
    state_dim = 11
    action_dim = 3
    hidden_dims = 32

    policy = REINFORCE_trainer(state_dim, hidden_dims, action_dim)

    max_episodes = 20
    max_steps = 500

    total_episodes = 0
    save_rewards = []
    

    
    for total_episodes in trange(max_episodes):
        
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
            reward = get_reward(next_state)
            next_state = torch.autograd.Variable(torch.from_numpy(next_state)).float()
            trajectory.append([np.array(state), action, log_prob, reward, next_state])
            state = next_state
            episode_reward += reward
            total_steps += 1


        total_episodes += 1
        
        policy_loss = policy.train(trajectory, batch_size = None)
        save_rewards.append(episode_reward)


    plt.plot(save_rewards)
    plt.show()

if __name__ == '__main__':
    main()
