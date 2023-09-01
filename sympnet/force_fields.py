import numpy as np


def f_pendulum(t, x):
    p, q = x
    return np.array([-np.sin(q), p])
