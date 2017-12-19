from matplotlib import pyplot as plt
import numpy as np


def rewards(sum_rewards, window_size=10):
    sum_rewards = np.convolve(sum_rewards, np.ones(
        (window_size,)) / window_size, mode='valid')
    plt.figure(figsize=(10, 5))
    plt.title(
        'Smoothed Episode Reward over time (smoothing window size = {})'.format(window_size), fontsize=14)
    plt.xlabel('Episodes', fontsize=11)
    plt.ylabel('Sum of rewards during episode', fontsize=11)
    plt.grid()
    plt.plot(sum_rewards, 'darkorange')
