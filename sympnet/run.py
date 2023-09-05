import click
import matplotlib.pyplot as plt
import numpy as np
import torch

from .io import load_model
from .systems import SYSTEMS


def run_model(model, x0, N):
    x0 = torch.as_tensor(x0)
    x = torch.empty(N, x0.shape[-1])
    x[0] = x0
    for i in range(1, N):
        x[i] = model(x[i - 1])
    return x.detach().numpy()


@click.command()
@click.argument("model_path")
@click.argument("system")  # TODO system is fixed once model is trained
def main(model_path, system):
    model = load_model(model_path)

    x = run_model(model, [1, 0], 1000)
    sys = SYSTEMS[system]()

    energy = sys.energy(x)

    fig, axes = plt.subplots(ncols=2)
    axes[0].plot(energy)
    axes[1].plot(x[:, 0], x[:, x.shape[-1] // 2])
    plt.show()


    # fig, axes = plt.subplots(ncols=2)
    # axes[0].plot(energy[::10])
    # axes[1].plot(q, p)

#     p = x[:, 0:2]
#     q = x[:, 2:4]
#     energy = (p[:, 0]**2 + p[:, 1]**2)/2 - q[:, 0]

#     fig, axes = plt.subplots(ncols=2)
#     axes[0].plot(energy)
#     axes[1].plot(q[:, 0], q[:, 1])

#     plt.show()


# if __name__ == "__main__":
#     main()
