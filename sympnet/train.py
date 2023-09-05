from pathlib import Path

import click

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .io import save_model
from .model import SympNet


def _load_data(data):
    d = np.load(data)
    train = torch.Tensor(d["train"])
    test = torch.Tensor(d["test"])
    return train[:-1], train[1:], test[:-1], test[1:]


@click.command()
@click.option("--n_epochs", default=100_000, type=int)
@click.option("--n_layers", default=3, type=int)
@click.option("--n_linear_sublayers", default=2, type=int)
@click.option("--lr", default=0.0001, type=float)
@click.argument("run_name")
def main(run_name, n_epochs, n_layers, n_linear_sublayers, lr):
    run_name = Path(run_name)
    data = run_name / "data.npz"
    train_x, train_y, test_x, test_y = _load_data(data)
    dim = train_x.shape[-1] // 2

    model = SympNet(
        dim=dim,
        n_layers=n_layers,
        n_linear_sublayers=n_linear_sublayers,
        activation=F.sigmoid,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    epochs = []
    train_losses = []
    test_losses = []
    for i in range(n_epochs):
        optimizer.zero_grad()

        predicted_train_y = model(train_x)
        loss = loss_fn(predicted_train_y, train_y)
        loss.backward()
        optimizer.step()

        if i % 500 == 0:
            with torch.no_grad():
                predicted_test_y = model(test_x)
                test_loss = loss_fn(predicted_test_y, test_y)

            print(f"{i}: training loss {loss.item()}, "
                  f"test loss {test_loss.item()}")

            epochs.append(i)
            train_losses.append(loss.item())
            test_losses.append(test_loss.item())

    model_fname = run_name / "model.pth"
    save_model(model, model_fname)
    click.echo(f"Model saved to {model_fname}")

    plt.semilogy(epochs, train_losses, label="Train")
    plt.semilogy(epochs, test_losses, label="Test")
    plt.legend()

    plot_fname = run_name / "training.png"
    plt.savefig(plot_fname)
    click.echo(f"Training plot saved to {plot_fname}")
