"""
Module for running experiments for the iterative solvers in the
discretization part of the project.
"""

import timeit
import os

import click
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from projektpraktikum_i import utils
from projektpraktikum_i.discretization.poisson_problem_2d import (
    example_u,
    example_f,
    solve_via_sor,
    compute_error,
    solve_via_lu_decomposition_fast,
)


def log_tick_formatter(val, _):
    return f"{10**val:.0e}"


def get_optimal_omega(n):
    return 2 / (1 + np.sin(np.pi / n))


eps_option = click.option(
    "--eps",
    default=1e-8,
    help="""Tolerance for the norm of the residual in the infinity norm.
            If set less or equal to 0 no constraint on the norm of the residual is imposed.""",
)
max_iter_option = click.option(
    "--max-iter",
    type=int,
    default=0,
    help="""Maximal number of iterations that the solver will perform.
            If set less or equal to 0 no constraint on the number of iterations is imposed.""",
)
var_x_option = click.option(
    "--var-x",
    default=0,
    help="""Minimal change of the iterate in every step in the infinity norm.
            If set less or equal to 0 no constraint on the change is imposed.""",
)
omega_option = click.option(
    "--omega",
    type=float,
    required=False,
    help="Relaxation parameter, if not set, the optimal value will be used.",
)


@click.group(context_settings={"show_default": True})
def cli():
    """Experiments for the Poisson problem."""


@cli.command()
@utils.interval_option("n", start_default=1, stop_default=7, num_default=20, log=True)
@eps_option
@max_iter_option
@var_x_option
@omega_option
@utils.display_or_save
# pylint: disable=too-many-arguments
def plot_error(n_interval, eps, max_iter, var_x, omega):
    """Plots the error of the numerical solution for different values of n."""

    errors = [
        compute_error(
            n,
            solve_via_sor(
                n, example_f, {"eps": eps, "var_x": var_x, "max_iter": max_iter}, omega
            ),
            example_u,
        )
        for n in n_interval
    ]
    number_of_discretization_points = (n_interval - 1) ** 2

    # Plotting the error vs n
    plt.figure(figsize=(10, 6))
    plt.loglog(
        number_of_discretization_points,
        errors,
        marker="o",
        linestyle="-",
        color="r",
        label="Error",
    )
    plt.loglog(
        number_of_discretization_points,
        1 / number_of_discretization_points,
        linestyle="--",
        color="gray",
        label=r"$\frac{1}{N}$",
    )
    plt.xlabel("Number of Discretization Points ($N$)")
    plt.ylabel("Error")
    plt.title("Error vs Number of Discretization Points (Log-Log Scale)")
    plt.grid(True, ls="--")
    plt.legend()


@cli.command()
@utils.interval_option("n", 2, 7, 20, log=True)
@click.option("-k", "k_values", type=int, multiple=True, default=[-2, 0, 2, 4, 6])
@omega_option
@utils.display_or_save
def plot_optimal_eps(n_interval, k_values, omega):
    """Plots the error of the numerical solution for different values of n and k."""

    N = (n_interval - 1) ** 2  # Discretization points pylint: disable=invalid-name
    plt.figure(figsize=(12, 8))

    markers = ["o", "s", "^", "v", "D"]
    for i, k in enumerate(k_values):
        errors = []
        for n in n_interval:
            h = 1 / n
            eps = h**k
            u = solve_via_sor(
                n, example_f, {"eps": eps, "max_iter": 0, "var_x": 0}, omega
            )
            errors.append(compute_error(n, u, example_u))

        plt.loglog(
            N, errors, f"{markers[i%len(markers)]}-", label=f"k={k}", markersize=8
        )

    plt.xlabel("Number of Discretization Points ($N$)")
    plt.ylabel("Error")
    plt.title(r"Error vs N for Different $\epsilon(k) = h^k$")
    plt.grid(True, linestyle="--")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.subplots_adjust(right=0.85)


@cli.command()
@utils.interval_option("n", 8, 64, 20)
@utils.interval_option("omega", 0.0, 2.0, 20, dtype=float, endpoint=False)
@click.option("--max-iter", default=64, help="Maximum SOR iterations")
@click.option("--azim", default=120, type=float, help="Azimuth angle for 3D plot.")
@click.option("--elev", default=30, type=float, help="Elevation angle for 3D plot.")
@utils.display_or_save
# pylint: disable=invalid-name, too-many-locals
def plot_optimal_omega(n_interval, omega_interval, max_iter, azim, elev):
    """Plots the optimal omega for the SOR method for different values of n."""

    X, Y = np.meshgrid(n_interval, omega_interval)
    errors = np.zeros_like(X, dtype=float)

    for i, omega in enumerate(omega_interval):
        for j, n in enumerate(n_interval):
            u = solve_via_sor(
                n, example_f, {"eps": 0, "max_iter": max_iter, "var_x": 0}, omega=omega
            )
            errors[i, j] = compute_error(n, u, example_u)

    optimal_omega_indices = np.argmin(errors, axis=0)
    optimal_omegas = omega_interval[optimal_omega_indices]

    fig = plt.figure(figsize=(15, 6))

    # 3D surface plot
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(X, Y, np.log10(errors), cmap="viridis")
    ax1.set_title("Error vs. Matrix Size and Omega")
    ax1.set_xlabel("Matrix Size (n)")
    ax1.set_ylabel("Omega")
    ax1.set_zlabel("Log10(Error)")
    ax1.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    # Rotate the 3D plot for better view
    ax1.view_init(elev=elev, azim=azim)

    # 2D plot for optimal omegas
    ax2 = fig.add_subplot(122)
    ax2.plot(n_interval, optimal_omegas, "ro-", label="Empirical Optimal")
    ax2.plot(
        n_interval, get_optimal_omega(n_interval), "bo--", label="Theoretical Formula"
    )
    ax2.set_title("Optimal Omega for Minimal Error")
    ax2.set_xlabel("Matrix Size (n)")
    ax2.set_ylabel("Optimal Omega")

    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()


@cli.command()
@utils.interval_option("n", 3, 6, 20, log=True)
@click.option(
    "--fixed-eps", default=1e-3, type=float, help="Fixed eps value for the SOR method."
)
@utils.display_or_save
def plot_compare(n_interval, fixed_eps):
    """Plots the error comparison between LU and SOR for different values of n."""

    N = (n_interval - 1) ** 2  # Discretization points pylint: disable=invalid-name
    plt.figure(figsize=(10, 6))

    # LU
    errors_lu = [
        compute_error(n, solve_via_lu_decomposition_fast(n, example_f), example_u)
        if n < 128
        else np.nan
        for n in n_interval
    ]
    plt.loglog(N, errors_lu, "s-", label="LU")

    # SOR fixed eps
    errors_sor_fixed = [
        compute_error(
            n,
            solve_via_sor(
                n,
                example_f,
                {"eps": fixed_eps, "max_iter": 0, "var_x": 0},
                omega=get_optimal_omega(n),
            ),
            example_u,
        )
        for n in n_interval
    ]
    plt.loglog(N, errors_sor_fixed, "^-", label=rf"SOR ($\epsilon={fixed_eps:.2e}$)")

    # SOR optimal eps (k=2)
    errors_sor_opt = [
        compute_error(
            n,
            solve_via_sor(
                n,
                example_f,
                {"eps": (1 / n) ** 4, "max_iter": 0, "var_x": 0},
                omega=get_optimal_omega(n),
            ),
            example_u,
        )
        for n in n_interval
    ]
    plt.loglog(N, errors_sor_opt, "v-", label=r"SOR ($\epsilon=h^4$)")

    plt.xlabel("N")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend()
    plt.title("Error Comparison: LU vs SOR")


@cli.command()
@utils.interval_option("n", 2, 6, 20, log=True)
@click.option("--num-runs", default=10, help="Number of runs to average timing.")
@utils.display_or_save
# pylint: disable=unnecessary-lambda-assignment
def plot_time_comparison(n_interval, num_runs):
    """Plots the runtime comparison between LU and SOR for different values of n."""

    sor_times = []
    lu_times = []

    os.environ["DISABLE_JOBLIB_CACHE"] = "1"

    for n in n_interval:
        h = 1 / n
        sor_func = lambda: solve_via_sor(
            n,
            example_f,
            {"eps": h**2, "max_iter": 0, "var_x": 0},
            omega=get_optimal_omega(n),
        )
        sor_time = timeit.timeit(sor_func, number=num_runs)
        sor_times.append(sor_time)

        lu_func = lambda: solve_via_lu_decomposition_fast(n, example_f)
        lu_time = timeit.timeit(lu_func, number=num_runs)
        lu_times.append(lu_time)

    plt.figure(figsize=(10, 6))
    plt.loglog(n_interval, sor_times, "o-", label="SOR (Optimal)")
    plt.loglog(n_interval, lu_times, "s-", label="LU")
    plt.xlabel("$n$")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.legend()
    plt.title("Runtime Comparison: SOR vs LU")


if __name__ == "__main__":
    cli()
