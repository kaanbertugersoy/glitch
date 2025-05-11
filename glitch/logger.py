import datetime
import numpy as np
import matplotlib.pyplot as plt
from glitch.utils import smooth

import logging


class Logger:
    def __init__(self, num_episodes, verbose=False):
        self.verbose = verbose
        self.init_time = datetime.now()
        self.replay_buffer_fill_time = None
        # (episode, duration, steps, reward, time_per_step, epsilon)
        self.train_logs = np.zeros((num_episodes, 6), dtype=np.float32)
        self.average_time_per_step = None
        self.average_time_per_episode = None
        self.current_episode = 0
        self.average_reward = 0
        self.expected_remaining_time = None

    def _calculate_expected_finish_time(self):
        time_elapsed = datetime.now() - self.init_time
        time_per_episode = time_elapsed / self.current_episode
        remaining_episodes = len(self.train_logs) - self.current_episode
        self.expected_remaining_time = time_per_episode * remaining_episodes

    def write(self, message):
        print(message)
        self.log(message)

    def log(self, message):
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def plot_rewards(self):
        # Plot the smoothed rewards
        y = smooth(self.train_logs[:, 3])
        plt.plot(self.train_logs[:, 3], label='orig')
        plt.plot(y, label='smoothed')
        plt.legend()
        plt.show()
