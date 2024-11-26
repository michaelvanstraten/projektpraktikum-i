import numpy as np




def idx(nx, n):
    """ Calculates the number of an equation in the Poisson problem for
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
    return ((k - 1) * (n - 1) + j)



def inv_idx(m, n):
    """ Calculates the coordinates of a discretization point for a
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

    j = (m - 1) // (n - 1) + 1
    k = (m - 1) % (n - 1) + 1
    return [j, k]

def rhs(n, f):
    """ Computes the right-hand side vector `b` for a given function `f`.

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
    
    b = np.empty((n - 1)**2)
    for i in range(1, n):
        for j in range(1, n):
            b[idx((i, j), n) - 1] = (1/n)**2 * f(i / n, j / n)

    return b

f = lambda x,y: x*np.sin(np.pi*x)*y*np.sin(np.pi*y)
print(rhs(5, f))




def compute_error(n, hat_u, u):
    """ Computes the error of the numerical solution of the Poisson problem
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

    