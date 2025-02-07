"""Module for solving the Poisson problem using finite difference methods and LU
decomposition.

This module provides functions to discretize the Poisson problem, compute the
right-hand side vector, solve the problem using LU decomposition, and compute
the error of the numerical solution. It also includes example functions for
the right-hand side and the analytic solution of the Poisson problem, as well as
a function to plot the error of the numerical solution for different values of n.

Usage:
To use this module, import the necessary functions and call them with appropriate
arguments. For example, to solve the Poisson problem and plot the error, run the
main function.
"""

import functools

import matplotlib.pyplot as plt
import numpy as np
import click

from projektpraktikum_i import utils
from projektpraktikum_i.discretization import linear_solvers
from projektpraktikum_i.discretization.block_matrix_2d import BlockMatrix

__all__ = [
    "get_evaluation_points",
    "idx",
    "inv_idx",
    "rhs",
    "compute_error",
    "solve_via_lu_decomposition",
    "example_f",
    "example_u",
    "plot_error",
]


@functools.cache
def get_evaluation_points(n):
    """Generates the evaluation points for the given number of intervals.

    Parameters
    ----------
    n : int
        Number of intervals in each dimension.

    Returns
    -------
    tuple
        A tuple containing the x and y ranges and the evaluation points.
    """
    x_range = np.arange(1, n) / n
    evalutation_points = np.meshgrid(x_range, x_range)
    return evalutation_points


def idx(nx, n):
    """Calculates the number of an equation in the Poisson problem for a given
    discretization point.

    Parameters
    ----------
    nx : list of int
        Coordinates of a discretization point, multiplied by n.
    n : int
        Number of intervals in each dimension.

    Returns
    -------
    int
        Number of the corresponding equation in the Poisson problem.
    """
    i, j = nx
    return (j - 1) * (n - 1) + i


def inv_idx(m, n):
    """Calculates the coordinates of a discretization point for a given equation
    number of the Poisson problem.

    Parameters
    ----------
    m : int
        Number of an equation in the Poisson problem.
    n : int
        Number of intervals in each dimension.

    Returns
    -------
    list of int
        Coordinates of the corresponding discretization point, multiplied by n.
    """
    i = (m - 1) % (n - 1) + 1
    j = (m - 1) // (n - 1) + 1
    return [i, j]


def rhs(n, f):
    """Computes the right-hand side vector `b` for a given function `f`.

    Parameters
    ----------
    n : int
        Number of intervals in each dimension.
    f : callable
        Function right-hand-side of Poisson problem. The calling signature is `f(x)`.
        Here `x` is an array_like of `numpy`. The return value is a scalar.

    Returns
    -------
    numpy.ndarray
        Vector to the right-hand-side f.

    Raises
    ------
    ValueError
        If n < 2.
    """
    if n < 2:
        raise ValueError("Number of intervals must be at least 2.")

    evalutation_points = get_evaluation_points(n)
    return f(evalutation_points).flatten() / n**2


def compute_error(n, hat_u, u):
    """Computes the error of the numerical solution of the Poisson problem with
    respect to the infinity-norm.

    Parameters
    ----------
    n : int
        Number of intersections in each dimension.
    hat_u : array_like of 'numpy'
        Finite difference approximation of the solution of the Poisson problem
        at the discretization points.
    u : callable
        Solution of the Poisson problem. The calling signature is 'u(x)'.
        Here 'x' is an array_like of 'numpy'. The return value is a scalar.

    Returns
    -------
    float
        Maximal absolute error at the discretization points.
    """
    return np.max(np.abs(u(get_evaluation_points(n)).flatten() - hat_u))


def solve_discrete(n, f, equation_solver):
    """
    Solves the discretized Poisson problem using a provided equation solver.

    Parameters
    ----------
    n : int
        Number of discretization points in each dimension.
    f : callable
        Function right-hand-side of Poisson problem. The calling signature is `f(x)`.
        Here `x` is an array_like of `numpy`. The return value is a scalar.
    equation_solver : callable
        Function to solve the discretized system of equations. The calling signature
        is `equation_solver(A, b)`, where `A` is the discretization matrix and `b` is
        the right-hand side vector.

    Returns
    -------
    numpy.ndarray
        Approximate solution of the Poisson problem.
    """
    discretization_matrix = BlockMatrix(n)
    b = rhs(n, f)

    return equation_solver(discretization_matrix, b)


@utils.cache
def solve_via_lu_decomposition(n, f):
    """Solves the Poisson problem using LU decomposition.

    Parameters
    ----------
    n : int
        Number of intervals in each dimension.
    f : callable
        Function right-hand-side of Poisson problem. The calling signature is `f(x)`.
        Here `x` is an array_like of `numpy`. The return value is a scalar.

    Returns
    -------
    numpy.ndarray
        Approximate solution of the Poisson problem.
    """

    def solver(system_matrix, right_hand_side):
        p, l, u = system_matrix.get_lu()
        return linear_solvers.solve_lu(p, l, u, right_hand_side)

    return solve_discrete(n, f, solver)


@utils.cache
def solve_via_lu_decomposition_fast(n, f):
    """Solves the Poisson problem using LU decomposition.

    Parameters
    ----------
    n : int
        Number of intervals in each dimension.
    f : callable
        Function right-hand-side of Poisson problem. The calling signature is `f(x)`.
        Here `x` is an array_like of `numpy`. The return value is a scalar.

    Returns
    -------
    numpy.ndarray
        Approximate solution of the Poisson problem.
    """

    def solver(system_matrix, right_hand_side):
        from scipy.linalg import lu_factor, lu_solve  # pylint: disable=import-outside-toplevel

        lu, piv = lu_factor(system_matrix.get_sparse().toarray())
        return lu_solve((lu, piv), right_hand_side)

    return solve_discrete(n, f, solver)


@utils.cache
def solve_via_sor(n, f, params, omega):
    """
    Solves the Poisson problem using the Successive Over-Relaxation (SOR) method.
    """
    def solver(system_matrix, right_hand_side):
        _, iterates, _ = linear_solvers.solve_sor(
            system_matrix.get_sparse(), right_hand_side, right_hand_side, params, omega
        )

        return iterates[-1]

    return solve_discrete(n, f, solver)


def example_f(x, k=3):
    """Example function for the right-hand side of the Poisson problem.

    Parameters
    ----------
    x : array_like of 'numpy'
        Evaluation points.
    k : int, optional
        Scalar parameter (default is 3).

    Returns
    -------
    numpy.ndarray
        Computed values for the function.
    """
    return (
        2
        * k
        * np.pi
        * (
            -x[0] * np.cos(k * np.pi * x[1]) * np.sin(k * np.pi * x[0])
            + x[1]
            * (-np.cos(k * np.pi * x[0]) + k * np.pi * x[0] * np.sin(k * np.pi * x[0]))
            * np.sin(k * np.pi * x[1])
        )
    )


def example_u(x, k=3):
    """Example solution for the Poisson problem.

    Parameters
    ----------
    x : array_like of 'numpy'
        Evaluation points.
    k : int, optional
        Scalar parameter (default is 3).

    Returns
    -------
    numpy.ndarray
        Computed values for the function.
    """
    return (x[0] * np.sin(k * np.pi * x[0])) * (x[1] * np.sin(k * np.pi * x[1]))


class Solver(click.Choice):
    def __init__(self, available_solvers) -> None:
        self.solvers = available_solvers

        super().__init__(available_solvers.keys(), case_sensitive=False)

    def convert(self, value, param, ctx):
        choice = super().convert(value, param, ctx)
        return self.solvers[choice]


@click.group()
def cli():
    pass


@cli.command()
@utils.interval_option("n", start_default=1, stop_default=6, num_default=20, log=True)
@click.option(
    "--solver",
    type=Solver(
        available_solvers={
            "sor": solve_via_sor,
            "lu": solve_via_lu_decomposition,
            "lu-fast": solve_via_lu_decomposition_fast,
        },
    ),
)
@utils.display_or_save
# pylint: disable=too-many-arguments,too-many-positional-arguments
def plot_error(solver, interval):
    """Plots the error of the numerical solution for different values of n."""
    errors = [compute_error(n, solver(n, example_f), example_u) for n in interval]
    number_of_discretization_points = (interval - 1) ** 2

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


if __name__ == "__main__":
    cli()
