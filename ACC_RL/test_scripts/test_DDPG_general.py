import sys, pickle, os; sys.path.append('..')
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

def parse_input_args():
    args_dict = {}
    # Gamma
    args_dict['gamma'] = float(sys.argv[1])
    # Actor learning rate
    args_dict['actor_lr'] = float(sys.argv[2])
    # Critic learning rate
    args_dict['critic_lr'] = float(sys.argv[3])
    # Actor layer variant (number)
    args_dict['actor_layer'] = int(sys.argv[4])
    # Exploration noise variant (number)
    args_dict['explore_noise'] = int(sys.argv[5])
    # Batch size
    args_dict['batch_size'] = int(sys.argv[6])
    # Drive trace
    args_dict['drive_trace'] = sys.argv[7]
    # Max episodes in replay buffer
    args_dict['max_eps'] = int(sys.argv[8])
    # Downweight factor (exponent)
    args_dict['dweight_factor'] = float(sys.argv[9])
    # SOC
    args_dict['SOC'] = float(sys.argv[10])
    # Epochs
    args_dict['epochs'] = int(sys.argv[11])
    # Cutoff
    args_dict['cutoff'] = int(sys.argv[12])
    # Output file
    args_dict['output_f'] = sys.argv[13]

    # Starting epoch (optional)
    if len(sys.argv) > 14:
        args_dict['running_ep_count'] = int(sys.argv[14])

    if len(sys.argv) > 15:
        args_dict['model_prefix'] = sys.argv[15]

    return args_dict

    # return (
    #     gamma,
    #     actor_lr,
    #     critic_lr,
    #     actor_layer,
    #     explore_noise,
    #     batch_size,
    #     drive_trace,
    #     max_eps,
    #     dweight_factor,
    #     epochs,
    #     cutoff,
    #     output_f
    # )

def main():

    args_dict = parse_input_args()

    # Initialize trainer:
    trainer = DDPGTrainer(
        state_dim = 11,
        action_dim = 3,
        gamma = args_dict['gamma'], 
        exploration_noise = torch.tensor([1000, 1000, 10]),
        actor_lr = args_dict['actor_lr'],
        critic_lr = args_dict['critic_lr'],
        state_transform = transform_state,
        actor_layers=[32, 32],
        batch_size = args_dict['batch_size']
    )

    # Initialize environment
    env = Environment(
        drive_trace = args_dict['drive_trace'],
        max_episodes_replay_buffer = args_dict['max_eps']
    )

    epochs = args_dict['epochs']
    rewards = []

    if 'model_prefix' in args_dict.keys():
        # Check if file exists:
        if os.path.exists(os.path.join('Models', '{}actor.pt'.format(args_dict['model_prefix']))):
            trainer.load_models(prefix = args_dict['model_prefix'])

    if 'running_ep_count' in args_dict.keys():
        evec = list(range(args_dict['running_ep_count'], args_dict['running_ep_count'] + epochs))
    else:
        evec = list(range(epochs))

    for e in trange(epochs):
        down_weight = (1 / (evec[e] + 1)) ** (1 / args_dict['dweight_factor'])
        r = env.TD_run_episode(
            trainer = trainer, 
            cutoff = args_dict['cutoff'], 
            SOC = 10, 
            update_freq = 500, 
            explore_noise_weight= torch.tensor([down_weight, down_weight, down_weight])
        )
        #print('OPTIMIZING')
        #trainer.optimize(env.replay_buffer)
        rewards.append(r)

    if 'model_prefix' in args_dict.keys():
        trainer.save_model(prefix = args_dict['model_prefix'])

    #plt.plot(rewards)
    #plt.show()
    pickle.dump(rewards, open(os.path.join('DDPG_outputs', args_dict['output_f'] + '.pickle'), 'wb'))

if __name__ == '__main__':
    main()
