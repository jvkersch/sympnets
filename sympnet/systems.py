from abc import ABC, abstractmethod

import numpy as np
from scipy.integrate import solve_ivp


class AbstractSystem(ABC):

    @abstractmethod
    def get_trajectory(self, x0, ts):
        pass

    @abstractmethod
    def energy(self, x):
        pass


class Pendulum(AbstractSystem):

    def get_trajectory(self, x0, ts):
        return _run_solver(self._f, x0, ts)

    def energy(self, x):
        p, q = x.T
        return p**2/2 - np.cos(q)

    def _f(self, t, x):
        p, q = x
        return np.array([-np.sin(q), p])


class CoupledPendulum(AbstractSystem):
    # 2 uncoupled pendulums

    def __init__(self):
        self._pendulum = Pendulum()

    def get_trajectory(self, x0, ts):
        traj1 = self._pendulum.get_trajectory(x0[(0, 2)], ts)
        traj2 = self._pendulum.get_trajectory(x0[(1, 2)], ts)
        return np.column_stack([traj1[:, 0], traj2[:, 0],
                                traj1[:, 1], traj2[:, 1]])

    def energy(self, x):
        return (self._pendulum.energy(x[:, (0, 2)]) +
                self._pendulum.energy(x[:, (1, 3)]))


SYSTEMS = {
    "pendulum": Pendulum,
    "coupled_pendulum": CoupledPendulum,
}


def _run_solver(f, x0, t_eval):
    t_span = [min(t_eval), max(t_eval)]
    result = solve_ivp(f, t_span, x0, t_eval=t_eval)
    return result.y.T
