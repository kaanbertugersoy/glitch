
import numpy as np


class ReplayBuffer:
    def __init__(self, frame_width, frame_height, size):
        self.states = np.zeros(
            [size, frame_width, frame_height, 1], dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.states2 = np.zeros(
            [size, frame_width, frame_height, 1], dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.pointer, self.size, self.max_size = 0, 0, size

    def store(self, s, a, r, s2, done):
        self.states[self.pointer] = s
        self.actions[self.pointer] = a
        self.rewards[self.pointer] = r
        self.states2[self.pointer] = s2
        self.dones[self.pointer] = done
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return self.states[idxs], self.actions[idxs], self.rewards[idxs], self.states2[idxs], self.dones[idxs]
