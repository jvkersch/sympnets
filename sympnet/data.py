import torch

import numpy as np
from scipy.integrate import solve_ivp


def prepare_training_test_data(
        f, x0, h=0.1, N_train=40, N_test=100, solver_kwargs=None):

    solver_kwargs = solver_kwargs or {}

    t_eval = np.arange(N_train + N_test) * h
    t_span = [0, t_eval.max()]
    result = solve_ivp(f, t_span, x0, t_eval=t_eval, **solver_kwargs)

    trajectory = torch.Tensor(result.y.T)
    train = trajectory[:N_train]
    test = trajectory[N_train:]

    return train[:-1], train[1:], test[:-1], test[1:]
