import numpy as np

from projektpraktikum_i.derivative_approximation.finite_difference import (
    is_vectorized,
    infinity_norm,
)

def idx(nx, n):
    """Calculates the number of an equation in the Poisson problem for
    a given discretization point.

    Parameters
    ----------
    nx : list of int
        Coordinates of a discretization point, multiplied by n.
    n : int
        Number of intervals in each dimension.

    Return
    ------
    int
        Number of the corresponding equation in the Poisson problem.
    """

    i, j = nx
    return (j - 1) * (n - 1) + i


def inv_idx(m, n):
    """Calculates the coordinates of a discretization point for a
    given equation number of the Poisson problem.

    Parameters
    ----------
    m : int
        Number of an equation in the Poisson Problem
    n : int
        Number of intervals in each dimension.

    Return
    ------
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
        Function right-hand-side of Poisson problem. The calling signature is
        `f(x)`. Here `x` is an array_like of `numpy`. The return value
        is a scalar.

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
        raise ValueError("TODO: better error message")

    x_1 = np.arange(1, n)
    x = np.dstack(np.meshgrid(x_1, x_1)).reshape(-1, 2)

    if not is_vectorized(f, x):
        f = np.vectorize(f, signature="(2)->()")

    return f(x / n)


def compute_error(n, hat_u, u):
    """Computes the error of the numerical solution of the Poisson problem
    with respect to the infinity-norm.

    Parameters
    ----------
    n : int
        Number of intersections in each dimension
    hat_u : array_like of 'numpy'
        Finite difference approximation of the solution of the Poisson problem
        at the discretization points
    u : callable
        Solution of the Poisson problem
        The calling signature is 'u(x)'. Here 'x' is an array_like of 'numpy'.
        The return value is a scalar.

    Returns
    -------
    float
        maximal absolute error at the discretization points
    """
    return np.max(np.abs(rhs(n, u), hat_u))
