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

def transform_state(state_vec):

    state_vec = state_vec.T
    my_vec = state_vec[:13][torch.tensor([1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1], dtype = bool)]
    
    return my_vec.T

def evaluate_policy(policy, env, eval_episodes = 10):
    avg_reward = 0.0
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, log_prob = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
        avg_reward /= eval_episodes
        print("avg reward is: {0}".fomat(avg_reward))

def render_policy(policy):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action,_,_,_ = policy.select_action(np.array(obs))
        obs, reward, done, _ = env.step(action)
    env.close()




def main():
    env = Environment(
        drive_trace = 'IM240',
        max_episodes_replay_buffer = 1e3
    )
    env = Model(automatic_control=False)
    state_dim = 11
    action_dim = 3
    hidden_dims = 32

    policy = REINFORCE_trainer(state_dim, hidden_dims, action_dim)

    max_episodes = 500
    total_episodes = 0
    save_rewards = []
    

    while total_episodes < max_episodes:
        obs = env.reset()
        done = False
        trajectory = []
        episode_reward = 0

        while not done:
            action, ln_prob = policy.select_action(np.array(obs))
            next_state, reward, done, _ = env.step(action)
            trajectory.append([np.array(obs), action, ln_prob, reward, next_state, done])
            obs = next_state
            episode_reward += reward
        total_episodes += 1
        policy_loss = policy.train(trajectory)
        save_rewards.append(reward)

        if total_episodes % 10 == 0:
            evaluate_policy(policy,env)
        env.close()

    plt.plot(save_rewards)
    plt.show()

if __name__ == '__main__':
    main()
