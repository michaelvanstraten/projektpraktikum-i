import numpy as np
import matplotlib.pyplot as plt
import itertools

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

    j, k = nx
    return int((k - 1) * (n - 1) + j)


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

    j = (m - 1) % (n - 1) + 1
    k = (m - 1) // (n - 1) + 1
    return [j, k]


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
        raise ValueError

    b = np.empty((n - 1) ** 2)
    x = np.linspace(1/n, 1 - 1/n, n - 1)
    X = list(itertools.product(x, x))
    for y in X:
        t = np.array(y)*n
        b[int(idx(t, n)) - 1] = (1 / n) ** 2 * f(np.array(y))

    return b

#k = 3
#def f(x):
    #x_1, x_2 = x
    #f = 2 * k * np.pi * (x_2 * np.cos(k * np.pi * x_1) * np.sin(k * np.pi * x_2) + 
                                    #x_1 * np.sin(k * np.pi * x_1) * np.cos(k * np.pi * x_2) - x_1 * x_2 * np.sin(k* np.pi * x_1) * np.sin(k * np.pi * x_2))
    #return f
#print(rhs(5, f))


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
    u_exakt = np.empty((n - 1)**2)
    x = np.linspace(1/n, 1 - 1/n, n - 1)
    X = list(itertools.product(x, x))
    for y in X:
        t = np.array(y)*n
        u_exakt[int(idx(t, n)) - 1] = u(np.array(y))
    max_error = 0
    for i in np.arange((n - 1)**2):
        if np.abs(u_exakt[i] - hat_u[i]) > max_error:
            max_error = np.abs(u_exakt[i] - hat_u[i])
    return max_error

def plot_max_error(n, hat_u, u):
    n_werte = np.arange(2, n + 1)
    N_werte = n_werte**2
    error_werte = compute_error(N_werte, hat_u, u)

    plt.figure(figsize=(10, 6))
    plt.yscale("log")
    plt.xscale("log")
    plt.plot(N_werte, error_werte, label="Maximaler Fehler", color="blue")
    plt.xlabel("N")
    plt.ylabel("Maximaler Fehler")
    plt.legend()
    plt.title("")
    plt.grid(True)