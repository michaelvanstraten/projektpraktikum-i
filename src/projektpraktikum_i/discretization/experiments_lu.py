"""Experiments for the Poisson problem."""

import numpy as np
import matplotlib.pyplot as plt
import click

from projektpraktikum_i.discretization import block_matrix_2d
from projektpraktikum_i.discretization import poisson_problem_2d
from projektpraktikum_i.discretization.poisson_problem_2d import example_u, example_f


@click.group()
def cli():
    """Experiments for the Poisson problem."""


def get_solutions(n):
    """Compute analytical and numerical solutions for the Poisson problem."""
    evalutation_points = poisson_problem_2d.get_evaluation_points(n)
    analytical_solution = example_u(evalutation_points)
    numeric_solution = poisson_problem_2d.solve_via_lu_decomposition(
        n, example_f
    ).reshape((n - 1, n - 1))
    return evalutation_points, analytical_solution, numeric_solution


@cli.command()
@click.option("-n", default=100, help="Number of intervals in each dimension.")
def plot_solutions(n):
    """Plot analytical and numerical solution of the Poisson problem in 2D."""
    evalutation_points, analytical_solution, numeric_solution = get_solutions(n)

    # Plot solutions
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6), subplot_kw={"projection": "3d"}
    )

    # Analytical solution
    surf1 = ax1.plot_surface(
        *evalutation_points, analytical_solution, cmap="viridis", edgecolor="none"
    )
    ax1.set_xlabel("$X_1$")
    ax1.set_ylabel("$X_2$")
    ax1.set_zlabel("$u(X)$")
    ax1.set_title("Analytical Solution $u(X)$")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    # Numerical solution
    surf2 = ax2.plot_surface(
        *evalutation_points, numeric_solution, cmap="coolwarm", edgecolor="none"
    )
    ax2.set_xlabel("$X_1$")
    ax2.set_ylabel("$X_2$")
    ax2.set_zlabel(r"$\^{u}(X)$")
    ax2.set_title(r"Numerical Solution $\^{u}(X)$")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    # Show plots
    plt.tight_layout()
    plt.show()


@cli.command()
@click.option("-n", default=100, help="Number of intervals in each dimension.")
def plot_difference(n):
    """Plot the difference between analytical and numerical solutions of the
    Poisson problem in 2D."""
    evalutation_points, analytical_solution, numeric_solution = get_solutions(n)

    # Compute the difference between numerical and analytical solutions
    difference = np.abs(analytical_solution - numeric_solution)

    # Plot solutions and error
    fig = plt.figure(figsize=(14, 6))

    # 3D plot of the error
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    surf1 = ax1.plot_surface(
        *evalutation_points, difference, cmap="coolwarm", edgecolor="none"
    )
    ax1.set_xlabel("$X_1$")
    ax1.set_ylabel("$X_2$")
    ax1.set_zlabel("Absolute Error")
    ax1.set_title("3D Plot of Absolute Error")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    # Heatmap of the error
    ax2 = fig.add_subplot(1, 2, 2)
    heatmap = ax2.imshow(
        difference, extent=(0, 1, 0, 1), origin="lower", cmap="coolwarm"
    )
    ax2.set_xlabel("$X$")
    ax2.set_ylabel("$Y$")
    ax2.set_title("Heatmap of Absolute Error")
    fig.colorbar(heatmap, ax=ax2, shrink=0.8, aspect=20, label="Error")

    # Show plots
    plt.tight_layout()
    plt.show()


start_option = click.option("--start", default=2, help="Start value for n.")
end_option = click.option("--end", default=20, help="End value for n.")
num_points_option = click.option(
    "--num_points", default=10, help="Number of points in the interval."
)
save_to_option = click.option(
    "--save-to",
    type=click.Path(dir_okay=False, writable=True),
    help="Specify a file path to save the generated plot.",
)


@cli.command()
@start_option
@end_option
@num_points_option
@save_to_option
def plot_error(start, end, num_points, save_to):
    """Plots the error of the numerical solution for different values of n."""
    poisson_problem_2d.plot_error(
        example_f,
        example_u,
        poisson_problem_2d.solve_via_lu_decomposition,
        (start, end, num_points),
        save_to,
    )


@cli.command()
@start_option
@end_option
@num_points_option
@save_to_option
def plot_compare_theoretical_memory_usage(start, end, num_points, save_to):
    """Plots the theoretical memory usage comparison between raw format and CRS format."""
    block_matrix_2d.plot_compare_theoretical_memory_usage(
        (start, end, num_points), save_to
    )


@cli.command()
@start_option
@end_option
@num_points_option
@save_to_option
def plot_non_zero_entries(start, end, num_points, save_to):
    """Plots the number of non-zero entries in $A$ as a function of $n$ and $N$,
    and compares it with the number of entries in a fully populated matrix."""
    block_matrix_2d.plot_non_zero_entries((start, end, num_points), save_to)


@cli.command()
@start_option
@end_option
@num_points_option
@save_to_option
def plot_non_zero_entries_lu(start, end, num_points, save_to):
    """Plots the number of non-zero entries in the matrix $A$ and its LU decomposition
    as a function of $N$."""
    block_matrix_2d.plot_non_zero_entries_lu((start, end, num_points), save_to)


if __name__ == "__main__":
    cli()
