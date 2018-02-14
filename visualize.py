from matplotlib import pyplot as plt
import numpy as np


def rewards(sum_rewards, window_size=10):
    '''
    Displays a graph which contains the sum-of-rewards for each episode, smoothed with a window.
    sum_rewards 
    
    Parameters
    ----------
    sum_rewards : list
        For each episode, contains a number representing the sum of rewards.
    window_size: int
        Size of window for smoothing the rewards.
    '''
    sum_rewards = np.convolve(sum_rewards, np.ones(
        (window_size,)) / window_size, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.title(
        'Smoothed Episode Reward over time (smoothing window size = {})'.format(window_size), fontsize=14)
    plt.xlabel('Episodes', fontsize=11)
    plt.ylabel('Sum of rewards during episode', fontsize=11)
    plt.grid()
    plt.plot(sum_rewards, 'darkorange')
