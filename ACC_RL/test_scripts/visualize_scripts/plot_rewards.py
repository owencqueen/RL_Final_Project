import sys, glob
import os, pickle
import numpy as np

import matplotlib.pyplot as plt

def gather_from_dir(dirname, basename = None, show = False):
    '''
    Gathers files from directory
    '''
    
    rewards = []
    for i in range(1, len(os.listdir(dirname)) + 1):
        r = pickle.load(open(dirname + '/' + basename + '{}.pickle'.format(i), 'rb'))
        rewards += r

    if show:
        plt.plot(rewards)
        plt.show()

    return rewards


def plot_DDPG_noise_decay(r_opt=1):
    '''
    Plot noise decay experiments

    Args:
        r_opt (int): Reward option (1 is speed, 2 is control)
    '''

    main_dir = '../DDPG_outputs'

    reward_lists = []

    first = '1' if r_opt == 1 else '3'

    task = 'Speeding up Car' if r_opt == 1 else 'Speed Control'

    for i in range(5):

        r = gather_from_dir(os.path.join(main_dir, 'DDPG_d={}_cutoff={}0000_r={}'.format(i, first, r_opt)), 'DDPG_')

        if i == 0:
            x = range(25,len(r))
            plt.plot(x, r[25:], label = 'd={}'.format(i))   

        elif i == 1:
            plt.plot(r, label = 'd={}'.format(i))
        else:
            plt.plot(r, label = 'd=1/{}'.format(i))
    
    plt.title('DDPG Exploration Noise Decay - {}'.format(task))
    plt.xlabel('Number of Episodes')
    plt.ylabel('Sum of rewards per episode')
    plt.legend()
    plt.show()

def plot_DDPG_gamma(r_opt = 1):
    '''
    Plot gamma experiments

    Args:
        r_opt (int): Reward option (1 is speed, 2 is control)
    '''

    main_dir = '../DDPG_outputs'

    reward_lists = []

    first = '1' if r_opt == 1 else '3'

    task = 'Speeding up Car' if r_opt == 1 else 'Speed Control'

    for gamma in [0.9, 0.95, 1]:

        if gamma == 1:
            r = gather_from_dir(os.path.join(main_dir, 'DDPG_d=1_cutoff={}0000_r={}'.format(first, r_opt)), 'DDPG_')
        else:
            r = gather_from_dir(os.path.join(main_dir, 'DDPG_d=1_cutoff={}0000_r={}_gam={}'.format(first, r_opt, gamma)), 'DDPG_')

        if (gamma == 0.9 and r_opt == 1) or (r_opt == 2):
            x = range(25,len(r))
            plt.plot(x, r[25:], label = '$\gamma={}$'.format(gamma))
        else:
            plt.plot(r, label = '$\gamma={}$'.format(gamma))
    
    plt.title('DDPG Gamma - {}'.format(task))
    plt.xlabel('Number of Episodes')
    plt.ylabel('Sum of rewards per episode')
    plt.legend()
    plt.show()


def plot_REINFORCE_gamma():
    pass


if __name__ == '__main__':
    plot_DDPG_noise_decay(r_opt = 1)    
    plot_DDPG_noise_decay(r_opt = 2)

    plot_DDPG_gamma(r_opt = 1)
    plot_DDPG_gamma(r_opt = 2)
