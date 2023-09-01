from itertools import cycle

import torch
import torch.nn as nn


def take(iterator, howmany):
    return [el for (el, _) in zip(cycle(iterator), range(howmany))]


class Linear(nn.Module):

    def __init__(self, dim, kind="upper", device=None):
        super().__init__()

        self._kind = kind
        self._dim = dim

        kwargs = {"device": device}
        self._A = nn.Parameter(torch.empty((dim, dim), **kwargs))
        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self._A)

    def forward(self, p, q):  # (B, dim)
        S = self._A + self._A.T
        if self._kind == "upper":
            p_new = p + torch.matmul(q, S)
            q_new = q
        else:
            p_new = p
            q_new = torch.matmul(p, S) + q
        return p_new, q_new


class LinearStack(nn.Module):

    def __init__(self, dim, sublayers, device=None):
        super().__init__()

        kinds = take(["upper", "lower"], sublayers)
        self._layers = nn.ModuleList([
            Linear(dim, kind, device) for kind in kinds
        ])

        # bias parameters
        self._b_p = nn.Parameter(torch.empty(dim, device=device))
        self._b_q = nn.Parameter(torch.empty(dim, device=device))
        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self._b_p)
        nn.init.normal_(self._b_q)

    def forward(self, p, q):
        for layer in self._layers:
            p, q = layer(p, q)
        return p + self._b_p, q + self._b_q


class Activation(nn.Module):

    def __init__(self, dim, activation, kind="upper", device=None):
        super().__init__()

        self._dim = dim
        self._kind = kind
        self._activation = activation

        kwargs = {"device": device}
        self._a = nn.Parameter(torch.empty(dim, **kwargs))

        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self._a)

    def forward(self, p, q):
        if self._kind == "lower":
            sigma = self._activation(p)
            q_new = q + self._a * sigma
            p_new = p
        else:
            sigma = self._activation(q)
            p_new = p + self._a * sigma
            q_new = q
        return p_new, q_new


class SympNet(nn.Module):

    def __init__(
            self, dim, n_layers, n_linear_sublayers, activation, device=None):

        super().__init__()

        kinds = take(["upper", "lower"], n_layers)
        layers = []
        for kind in kinds:
            layers.extend([
                LinearStack(dim, n_linear_sublayers, device),
                Activation(dim, activation, device=device, kind=kind)
            ])
        del layers[-1]  # no activation at the end

        self._layers = nn.ModuleList(layers)
        self._dim = dim
        self._args = {  # needed for serializing
            "dim": dim,
            "n_layers": n_layers,
            "n_linear_sublayers": n_linear_sublayers,
            "activation": activation
        }

    def forward(self, x):
        x = torch.atleast_2d(x)
        p, q = x[:, :self._dim], x[:, self._dim:]

        for layer in self._layers:
            p, q = layer(p, q)

        return torch.column_stack((p, q))
