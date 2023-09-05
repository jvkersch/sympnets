import json
from pathlib import Path

import click

import matplotlib.pyplot as plt
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


def _get_system(run_name):
    with open(run_name / "metadata-data.json", "rt", encoding="utf-8") as fp:
        metadata = json.load(fp)
    return metadata["system"]


@click.command()
@click.option("--run_name", required=True)
@click.option("--x0", type=str, required=True)
@click.option("--n_iterations", type=int, default=1000)
def main(run_name, x0, n_iterations):
    run_name = Path(run_name)
    x0 = eval(x0)

    system = _get_system(run_name)
    model = load_model(run_name / "model.pth")

    x = run_model(model, x0, n_iterations)
    sys = SYSTEMS[system]()

    energy = sys.energy(x)

    fig, axes = plt.subplots(ncols=2)
    axes[0].plot(energy)
    axes[1].plot(x[:, 0], x[:, x.shape[-1] // 2])
    plt.show()
