import json
from pathlib import Path

import click

import numpy as np
from scipy.integrate import solve_ivp

from .force_fields import f_pendulum
from .systems import SYSTEMS


def _run_solver(f, x0, N, h=0.1):
    t_eval = np.arange(N) * h
    t_span = [0, t_eval.max()]
    result = solve_ivp(f, t_span, x0, t_eval=t_eval)
    return result.y.T


def _prepare_data_generic(f, x0, h=0.1, N_train=40, N_test=100):
    traj = _run_solver(f, x0, N_train + N_test, h)
    return traj[:N_train], traj[N_train:]


def _prepare_data_pendulum_xy(x0, h=0.1, N_train=40, N_test=100):
    p, q = _run_solver(f_pendulum, x0, N_train + N_test, h).T
    cq = np.cos(q)
    sq = np.sin(q)
    xy = np.column_stack((-sq * p, cq * p, cq, sq))
    return xy[:N_train], xy[N_train:]


@click.command()
@click.argument("system")
@click.option("--run_name", required=True)
@click.option("--n_train", type=int, default=100)
@click.option("--n_test", type=int, default=40)
def main(system, run_name, n_train, n_test):
    if system == "pendulum_xy":
        train, test = _prepare_data_pendulum_xy([0, 1])
    elif system == "coupled_pendulum":
        x0 = [1, 1, 0, 0]
        ts = 0.1 * np.arange(n_train + n_test)
        trajectory = SYSTEMS[system]().get_trajectory(x0, ts)
        train = trajectory[:n_train]
        test = trajectory[:n_test]
    else:  # pendulum
        x0 = [1, 0]
        ts = 0.1 * np.arange(n_train + n_test)
        trajectory = SYSTEMS[system]().get_trajectory(x0, ts)
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
