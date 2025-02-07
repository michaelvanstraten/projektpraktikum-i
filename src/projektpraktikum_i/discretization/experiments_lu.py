"""Experiments for the Poisson problem."""

import numpy as np
import matplotlib.pyplot as plt
import click

from projektpraktikum_i import utils
from projektpraktikum_i.discretization import poisson_problem_2d
from projektpraktikum_i.discretization.poisson_problem_2d import example_u, example_f


@click.group()
def cli():
    """Experiments for the Poisson problem."""


save_to_option = click.option(
    "--save-to",
    type=click.Path(dir_okay=False, writable=True),
    help="Specify a file path to save the generated plot.",
)


@cli.command()
@click.option("-n", default=32, help="Number of intervals in each dimension.")
@utils.display_or_save
def plot_solutions(n):
    """Plot analytical and numerical solution of the Poisson problem in 2D."""
    evalutation_points = poisson_problem_2d.get_evaluation_points(128)
    analytical_solution = example_u(evalutation_points)
    numeric_solution = poisson_problem_2d.solve_via_lu_decomposition(
        n, example_f, fast=True
    ).reshape((n - 1, n - 1))

    # Plot solutions
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={"projection": "3d"})

    # Analytical solution
    ax1.plot_surface(
        *evalutation_points, analytical_solution, cmap="viridis", edgecolor="none"
    )
    ax1.set_xlabel("$X_1$")
    ax1.set_ylabel("$X_2$")
    ax1.set_zlabel("$u(X)$")
    ax1.set_title("Analytical Solution $u(X)$")

    # Numerical solution
    ax2.plot_surface(
        *poisson_problem_2d.get_evaluation_points(n),
        numeric_solution,
        cmap="coolwarm",
        edgecolor="none",
    )
    ax2.set_xlabel("$X_1$")
    ax2.set_ylabel("$X_2$")
    ax2.set_zlabel(r"$\^{u}(X)$")
    ax2.set_title(f"Numerical Solution $\\^{{u}}(X)$ with $n={n}$")

    plt.tight_layout()


@cli.command()
@click.option("-n", default=32, help="Number of intervals in each dimension.")
@utils.display_or_save
def plot_difference(n):
    """Plot the difference between analytical and numerical solutions of the
    Poisson problem in 2D."""
    evalutation_points = poisson_problem_2d.get_evaluation_points(n)
    analytical_solution = example_u(evalutation_points)
    numeric_solution = poisson_problem_2d.solve_via_lu_decomposition(
        n, example_f, fast=True
    ).reshape((n - 1, n - 1))

    # Compute the difference between numerical and analytical solutions
    difference = np.abs(analytical_solution - numeric_solution)

    # Plot solutions and error
    fig = plt.figure(figsize=(14, 6))

    # 3D plot of the error
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(*evalutation_points, difference, cmap="coolwarm", edgecolor="none")
    ax1.set_xlabel("$X_1$")
    ax1.set_ylabel("$X_2$")
    ax1.set_zlabel("Absolute Error")
    ax1.set_title(f"3D Plot of Absolute Error for $n = {n}$")

    # Heatmap of the error
    ax2 = fig.add_subplot(1, 2, 2)
    heatmap = ax2.imshow(
        difference, extent=(0, 1, 0, 1), origin="lower", cmap="coolwarm"
    )
    ax2.set_xlabel("$X_1$")
    ax2.set_ylabel("$X_2$")
    ax2.set_title(f"Heatmap of Absolute Error for $n = {n}$")
    fig.colorbar(heatmap, ax=ax2, shrink=0.8, aspect=20, label="Error")

    plt.tight_layout()


if __name__ == "__main__":
    cli()
