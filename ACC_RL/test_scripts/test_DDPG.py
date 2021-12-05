import sys; sys.path.append('..')
import torch
import matplotlib.pyplot as plt
from tqdm import trange

from DDPG.train import DDPGTrainer
from base_env import Environment

def transform_state(state_vec):
    #state_vec = torch.nan_to_num(state_vec)

    state_vec = state_vec.T

    my_vec = state_vec[:13][torch.tensor([1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1], dtype = bool)]
    #print('LEN', my_vec.shape)
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
        max_episodes_replay_buffer = 1e3
    )

    epochs = 500
    rewards = []

    for e in trange(epochs):
        down_weight = (1 / (e + 1)) ** (0.5)
        r = env.TD_run_episode(
            trainer = trainer, 
            cutoff = 3000, 
            SOC = 10, 
            update_freq=500,
            explore_noise_weight= torch.tensor([down_weight, down_weight, down_weight])
        )
        #print('OPTIMIZING')
        #trainer.optimize(env.replay_buffer)
        rewards.append(r * 1e2)

    plt.plot(rewards)
    plt.show()

if __name__ == '__main__':
    main()