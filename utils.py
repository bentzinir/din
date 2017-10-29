import numpy as np


def sample(n, p, mode='discrete'):
    if mode == 'discrete':
        return np. random.multinomial(n, p)
    else:  # mode = 'continuous
        return p + np.random.normal(p, 1, n)
