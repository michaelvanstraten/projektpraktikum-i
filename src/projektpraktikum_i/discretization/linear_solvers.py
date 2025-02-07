"""
Module providing various linear solvers for solving linear systems of equations.
"""

import numpy as np
import scipy

from projektpraktikum_i import utils


@utils.cache
def solve_lu(p, l, u, b):
    """Solves the linear system Ax = b via forward and backward substitution
    given the decomposition A = p * l * u.

    Parameters
    ----------
    p : numpy.ndarray
        permutation matrix of LU-decomposition
    l : numpy.ndarray
        lower triangular unit diagonal matrix of LU-decomposition
    u : numpy.ndarray
        upper triangular matrix of LU-decomposition
    b : numpy.ndarray
        vector of the right-hand-side of the linear system

    Returns
    -------
    x : numpy.ndarray
        solution of the linear system
    """

    # Using p^T = p^(-1)
    b_hat = np.matmul(np.transpose(p), b)

    # Solving Lz = b_hat with forward substitution
    z = np.empty(len(b))
    z[0] = b_hat[0] / l[0, 0]
    for i in range(1, len(b)):
        partial_sum = 0
        for j in range(1, i + 1):
            partial_sum += l[i, j - 1] * z[j - 1]
        z[i] = (b_hat[i] - partial_sum) / l[i, i]

    # Solving Rx = z with backward substitution
    x = np.empty(len(b))
    x[len(b) - 1] = z[len(b) - 1] / u[len(b) - 1, len(b) - 1]
    for i in range(len(b) - 2, -1, -1):
        partial_sum = 0
        for j in range(i + 1, len(b)):
            partial_sum += u[i, j] * x[j]
        x[i] = (z[i] - partial_sum) / u[i, i]

    return x


# pylint: disable=invalid-name
# pylint: disable=dangerous-default-value
# pylint: disable=too-many-locals
def solve_sor(
    A, b, x0, params={"eps": 1e-8, "max_iter": 1000, "var_x": 1e-4}, omega=1.5
):
    """Solves the linear system Ax = b via the successive over relaxation
    method.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        system matrix of the linear system
    b : numpy.ndarray (of shape (N,) )
        right-hand-side of the linear system
    x0 : numpy.ndarray (of shape (N,) )
        initial guess of the solution

    params : dict, optional
        dictionary containing termination conditions

        eps : float
            tolerance for the norm of the residual in the infinity norm. If set
            less or equal to 0 no constraint on the norm of the residual is imposed.
        max_iter : int
            maximal number of iterations that the solver will perform. If set
            less or equal to 0 no constraint on the number of iterations is imposed.
        var_x : float
            minimal change of the iterate in every step in the infinity norm. If set
            less or equal to 0 no constraint on the change is imposed.

    omega : float, optional
        relaxation parameter

    Returns
    -------
    str
        reason of termination. Key of the respective termination parameter.
    list (of numpy.ndarray of shape (N,) )
        iterates of the algorithm. First entry is `x0`.
    list (of float)
        infinity norm of the residuals of the iterates

    Raises
    ------
    ValueError
        If no termination condition is active, i.e., `eps=0` and `max_iter=0`, etc.
    """

    eps = params.get("eps", 1e-8)
    max_iter = params.get("max_iter", 1000)
    var_x = params.get("var_x", 1e-4)

    if eps <= 0 and max_iter <= 0 and var_x <= 0:
        raise ValueError(
            "No termination condition is active: eps <= 0, max_iter <= 0, var_x <= 0."
        )

    d_inv = scipy.sparse.diags(1.0 / A.diagonal())  # D^{-1}
    lower_tri = scipy.sparse.tril(A, k=-1)  # Strictly lower triangular part of A
    upper_tri = scipy.sparse.triu(A, k=1)  # Strictly upper triangular part of A

    # Precompute scaled matrices/terms used in each iteration
    scaled_d_inv = omega * d_inv  # ω * D^{-1}
    mat_lower_factor = scaled_d_inv @ lower_tri  # ω * D^{-1} * L
    mat_upper_factor = scaled_d_inv @ upper_tri  # ω * D^{-1} * U
    const_b = scaled_d_inv @ b  # ω * D^{-1} * b

    def compute_residual(x):
        """Compute the infinity norm of the residual."""
        return np.max(np.abs(b - A @ x))

    iterates = [x0]
    residuals = [compute_residual(x0)]

    def sor_iteration_loop():
        """Run the SOR iteration loop until a termination condition is met."""
        iteration_count = 0

        while True:
            current_x = iterates[-1]
            current_res = residuals[-1]

            # Termination condition: residual below threshold
            if eps > 0 and current_res <= eps:
                return "eps"

            # Termination condition: maximum iteration reached
            if 0 < max_iter <= iteration_count:
                return "max_iter"

            # Solves: (I + ω * D^{-1} * L) x^(k+1)
            # = (1 - ω) * x^(k) + ω * D^{-1} * b - ω * D^{-1} * U * x^(k)
            next_x = scipy.sparse.linalg.spsolve_triangular(
                mat_lower_factor,
                (1 - omega) * current_x + const_b - mat_upper_factor @ current_x,
                lower=True,
                unit_diagonal=True,
            )

            next_res = compute_residual(next_x)

            # Termination condition: change in iterate below threshold
            if var_x > 0 and np.max(np.abs(next_x - current_x)) <= var_x:
                return "var_x"

            iterates.append(next_x)
            residuals.append(next_res)

            iteration_count += 1

    termination_reason = sor_iteration_loop()

    return termination_reason, iterates, residuals
