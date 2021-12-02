import sys; sys.path.append('..')
import torch
import matplotlib.pyplot as plt

from DDPG.train import DDPGTrainer
from base_env import Environment

def main():
    # Initialize trainer:
    trainer = DDPGTrainer(
        state_dim = 17,
        action_dim = 3,
        gamma = 0.9, # Discounting
    )

    # Initialize environment
    env = Environment(
        drive_trace = 'IM240',
        max_episodes_replay_buffer = 1e5
    )

    epochs = 100
    rewards = []

    for e in range(epochs):
        r = env.TD_run_episode(trainer = trainer, cutoff = 1e4, SOC = 95)
        rewards.append(r)

    plt.plot(rewards)
    plt.show()

if __name__ == '__main__':
    main()