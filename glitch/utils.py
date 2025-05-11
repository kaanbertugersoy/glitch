import numpy as np


def smooth(x):
    # last 100
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - 99)
        y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)
    return y


def share_weights(src, dst):
    dst.set_weights(src.get_weights())


def update_state(state, obs_small):  # Not used since we use one frame as input rather than 4
    return np.append(state[:, :, 1:], np.expand_dims(obs_small, 2), axis=2)
