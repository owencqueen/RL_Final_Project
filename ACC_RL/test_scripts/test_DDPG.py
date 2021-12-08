import sys, gc; sys.path.append('ACC_RL')
import torch
import matplotlib.pyplot as plt
from tqdm import trange

from DDPG.train import DDPGTrainer
from base_env import Environment

def transform_state(state_vec):

    state_vec = state_vec.T

    my_vec = state_vec[:13][torch.tensor([1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1], dtype = bool)]
    return my_vec.T

def main():
    # Initialize trainer:
    trainer = DDPGTrainer(
        state_dim = 11,
        action_dim = 3,
        gamma = 1, 
        exploration_noise = torch.tensor([1000, 1000, 10]),
        actor_lr = 1,
        critic_lr = 1,
        state_transform = transform_state,
        actor_layers=[32, 32],
        batch_size = 32
    )

    # Initialize environment
    env = Environment(
        drive_trace = 'IM240',
        max_episodes_replay_buffer = 1000
    )

    epochs = 100
    rewards = []

    for e in trange(epochs):
        down_weight = (1 / (e + 1)) ** (0.5)
        r = env.TD_run_episode(
            trainer = trainer, 
            cutoff = 5000, 
            SOC = 10, 
            update_freq = 500, 
            explore_noise_weight= torch.tensor([down_weight, down_weight, down_weight])
        )
        rewards.append(r * 1e2)
        gc.collect() # Garbage collection for memory efficiency

    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Sum of Rewards per Episode * 100')
    plt.title('DDPG Training')
    plt.show()

if __name__ == '__main__':
    main()