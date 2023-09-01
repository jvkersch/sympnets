import click
import matplotlib.pyplot as plt
import numpy as np
import torch

from sympnet.io import load_model


def run_model(model, x0, N):
    x0 = torch.as_tensor(x0)
    x = torch.empty(N, x0.shape[-1])
    x[0] = x0
    for i in range(1, N):
        x[i] = model(x[i-1])
    return x.detach().numpy()


@click.command()
@click.argument("model_path")
def main(model_path):
    model = load_model(model_path)

    x = run_model(model, [1, 0], 1000)

    p, q = x.T
    energy = p**2/2 - np.cos(q)

    fig, axes = plt.subplots(ncols=2)
    axes[0].plot(energy[::10])
    axes[1].plot(q, p)
    plt.show()


if __name__ == "__main__":
    main()
