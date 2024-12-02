"""
Module for solving the Poisson problem using finite difference methods and LU decomposition.

This module provides functions to discretize the Poisson problem, compute the right-hand side vector,
solve the problem using LU decomposition, and compute the error of the numerical solution. It also
includes example functions for the right-hand side and the analytic solution of the Poisson problem,
as well as a function to plot the error of the numerical solution for different values of n.

Usage:
To use this module, import the necessary functions and call them with appropriate arguments.
For example, to solve the Poisson problem and plot the error, run the main function.

Example:
    from projektpraktikum_i.discretization import poisson_problem_2d
    poisson_problem_2d.plot_error(example_f, example_u, solve_via_lu_decomposition, (2, 20, 10))
"""

import matplotlib.pyplot as plt
import numpy as np

# We need to import dill here first so we can hash lambda functions
import dill as pickle
from joblib import Memory


from projektpraktikum_i.discretization import linear_solvers
from projektpraktikum_i.discretization.block_matrix_2d import BlockMatrix

__all__ = [
    "idx",
    "inv_idx",
    "rhs",
    "compute_error",
    "solve_via_lu_decomposition",
    "example_f",
    "example_u",
    "plot_error",
]

memory = Memory(location=".cache")


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
    """Calculates the number of an equation in the Poisson problem for a given discretization point.

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
    """Calculates the coordinates of a discretization point for a given equation number of the Poisson problem.

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
    """Computes the error of the numerical solution of the Poisson problem with respect to the infinity-norm.

    Parameters
    ----------
    n : int
        Number of intersections in each dimension.
    hat_u : array_like of 'numpy'
        Finite difference approximation of the solution of the Poisson problem at the discretization points.
    u : callable
        Solution of the Poisson problem. The calling signature is 'u(x)'.
        Here 'x' is an array_like of 'numpy'. The return value is a scalar.

    Returns
    -------
    float
        Maximal absolute error at the discretization points.
    """
    return np.max(np.abs(u(get_evaluation_points(n)).flatten() - hat_u))


@memory.cache
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
    discretization_matrix = BlockMatrix(n)
    p, l, u = discretization_matrix.get_lu()
    b = rhs(n, f)

    return linear_solvers.solve_lu(p, l, u, b)


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


def plot_error(f, analytic_u, solver, interval):
    """Plots the error of the numerical solution for different values of n.

    Parameters
    ----------
    f : callable
        Function to be used as the right-hand side of the Poisson problem.
    solver : callable
        Solver function that takes `n` and `f` and returns the approximate solution `hat_u`.
    interval : tuple
        Interval of values for n, defined as (start, end, num_points).
    """
    values_for_n = np.linspace(*interval, dtype=int)
    errors = [compute_error(n, solver(n, f), analytic_u) for n in values_for_n]

    # Plotting the error vs n
    plt.figure(figsize=(10, 6))
    plt.plot(
        values_for_n**2, errors, marker="o", linestyle="-", color="r", label="Error"
    )
    plt.xlabel("Number of Discretization Points ($N$)")
    plt.title("Error vs Number of Discretization Points")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    plot_error(example_f, example_u, solve_via_lu_decomposition, (2, 20, 10))


if __name__ == "__main__":
    main()
