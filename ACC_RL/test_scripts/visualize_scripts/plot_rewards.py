import sys, glob
import os, pickle
import numpy as np

import matplotlib.pyplot as plt

def gather_from_dir(dirname, basename = None, show = False):
    
    rewards = []
    for i in range(1, len(os.listdir(dirname)) + 1):
        r = pickle.load(open(dirname + '/' + basename + '{}.pickle'.format(i), 'rb'))
        rewards += r

    if show:
        plt.plot(rewards)
        plt.show()

    return rewards


def plot_DDPG_noise_decay(r_opt=1):

    main_dir = '../DDPG_outputs'

    reward_lists = []

    first = '1' if r_opt == 1 else '3'

    for i in range(5):

        r = gather_from_dir(os.path.join(main_dir, 'DDPG_d={}_cutoff={}0000_r={}'.format(i, first, r_opt)), 'DDPG_')

        if i == 0:
            x = range(25,len(r))
            plt.plot(x, r[25:], label = 'd={}'.format(i))    
        else:
            plt.plot(r, label = 'd={}'.format(i))
    
    plt.title('DDPG Exploration Noise Decay - Speeding up Car')
    plt.xlabel('Number of Episodes')
    plt.ylabel('Sum of rewards per episode')
    plt.legend()
    plt.show()




if __name__ == '__main__':
    #gather_from_dir('../DDPG_outputs/DDPG_d=2_cutoff=30000_r=2', 'DDPG_', show = True)
    plot_DDPG_noise_decay()
