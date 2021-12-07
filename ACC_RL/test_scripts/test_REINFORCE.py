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

SOC = 10
drive_trace = 'IM240'
reward_func_option = 1

def evaluate_policy(policy, env, eval_episodes = 10):
    reward_sum = 0.0
    state = env.reset()
    for _ in range(eval_episodes):
        
        done = False
        while not done:
            action, log_prob = policy.select_action(np.array(state))
            action = action.astype(np.double)
    
            next_state = env.step(action)
            reward = get_reward(next_state)
            next_state = torch.autograd.Variable(torch.from_numpy(next_state)).float()
            reward_sum += reward
       
        print("avg reward is: {0}".fomat(reward_sum))

def render_policy(policy):
    state = env.reset()
    done = False
    while not done:
        env.render()
        action, log_prob = policy.select_action(np.array(state))
        action = action.astype(np.double)
        next_state = env.step(action)
        reward = get_reward(next_state)
        next_state = torch.autograd.Variable(torch.from_numpy(next_state)).float()
    env.close()

def get_reward(state):
        r = raw_speed_reward(state)
        return r




def main():
    env = Model(automatic_control=False)
    state_dim = 11
    action_dim = 3
    hidden_dims = 32

    policy = REINFORCE_trainer(state_dim, hidden_dims, action_dim)

    max_episodes = 500
    total_episodes = 0
    save_rewards = []
    

    while total_episodes < max_episodes:
        #state = torch.autograd.Variable(torch.from_numpy(env.reset())).float()
        state = env.reset()
        max_steps = 500
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
            reward = get_reward(next_state)
            next_state = torch.autograd.Variable(torch.from_numpy(next_state)).float()
            trajectory.append([np.array(state), action, log_prob, reward, next_state, done])
            state = next_state
            episode_reward += reward
            total_steps += 1
            print(total_steps)

        total_episodes += 1
        print(total_episodes)
        policy_loss = policy.train(trajectory)
        save_rewards.append(reward)

        if total_episodes % 10 == 0:
            evaluate_policy(policy,env)
        env.close()

    plt.plot(save_rewards)
    plt.show()

if __name__ == '__main__':
    main()
