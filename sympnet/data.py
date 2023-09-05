import json
from pathlib import Path

import click
import numpy as np

from .systems import SYSTEMS


@click.command()
@click.argument("system")
@click.option("--run_name", required=True)
@click.option("--n_train", type=int, default=100)
@click.option("--n_test", type=int, default=40)
def main(system, run_name, n_train, n_test):
    system = SYSTEMS[system]()
    ts = 0.1 * np.arange(n_train + n_test)
    x0 = system.x0
    trajectory = system.get_trajectory(x0, ts)
    train = trajectory[:n_train]
    test = trajectory[:n_test]

    run_name = Path(run_name)
    run_name.mkdir(parents=True, exist_ok=True)

    data_fname = run_name / "data.npz"
    np.savez(data_fname, train=train, test=test)
    click.echo(f"Data saved as {data_fname}")
    click.echo(f"Train shape: {train.shape}")
    click.echo(f"Test shape: {test.shape}")

    # Save metadata
    metadata = {
        "system": system,
        "n_train": n_train,
        "n_test": n_test,
    }
    with open(run_name / "metadata-data.json", "wt", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)
