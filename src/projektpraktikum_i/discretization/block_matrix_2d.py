"""
Module providing classes and functions for analyzing block matrices arising
from finite difference approximations of the Laplace operator. The module
includes methods for creating sparse representations, evaluating sparsity
patterns, and analyzing LU decomposition.

Usage:
------
Run the module directly to see example plots for various functions and analyses.
"""

import os

from scipy import linalg
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import click

from projektpraktikum_i import utils


__all__ = [
    "BlockMatrix",
    "plot_theoretical_memory_usage",
    "plot_sparsity",
    "plot_sparsity_lu",
]


class BlockMatrix:
    """Represents block matrices arising from finite difference approximations
    of the Laplace operator.

    Parameters
    ----------
    n : int
        Number of intervals in each dimension.

    Attributes
    ----------
    n : int
        Number of intervals in each dimension.

    Raises
    ------
    ValueError
        If n < 2.
    """

    def __init__(self, n):
        if n < 2:
            raise ValueError("The parameter `n` must be at least 2.")
        self.n = n
        self.get_lu = utils.cache(self.get_lu)
        self.eval_sparsity_lu = utils.cache(self.eval_sparsity_lu)

    def get_sparse(self):
        """Returns the block matrix as a sparse matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            The block matrix in a sparse data format.
        """
        num_intervals = self.n
        central_diag = sp.diags(
            [4, -1, -1], [0, -1, 1], shape=(num_intervals - 1, num_intervals - 1)
        )
        block_diag = sp.block_diag([central_diag] * (num_intervals - 1))
        off_diag = sp.diags(
            [-1, -1], [num_intervals - 1, -(num_intervals - 1)], shape=block_diag.shape
        )

        return block_diag + off_diag

    def eval_sparsity(self):
        """Returns the absolute and relative numbers of non-zero elements of the
        matrix. The relative quantities are with respect to the total number of
        elements of the represented matrix.

        Returns
        -------
        int
            Number of non-zeros
        float
            Relative number of non-zeros
        """
        n = self.n
        nnz = 9 + n * (-14 + 5 * n)
        total_elements = (n - 1) ** 4
        return nnz, nnz / total_elements

    def get_lu(self):  # pylint: disable=method-hidden
        """Provides an LU-Decomposition of the represented matrix A of the form
        A = P * L * U.

        Returns
        -------
        P : numpy.ndarray
            Permutation matrix of LU-decomposition
        L : numpy.ndarray
            Lower triangular unit diagonal matrix of LU-decomposition
        U : numpy.ndarray
            Upper triangular matrix of LU-decomposition
        """
        matrix_a = self.get_sparse().toarray()
        matrix_p, matrix_l, matrix_u = linalg.lu(matrix_a)  # pylint: disable=unbalanced-tuple-unpacking
        return matrix_p, matrix_l, matrix_u

    def eval_sparsity_lu(self, epsilon=0):  # pylint: disable=method-hidden
        """Returns the absolute and relative numbers of non-zero elements of the
        LU-Decomposition. The relative quantities are with respect to the total
        number of elements of the represented matrix.

        Returns
        -------
        int
            Number of non-zeros
        float
            Relative number of non-zeros
        """
        matrix_p, matrix_l, matrix_u = self.get_lu()
        number_of_grid_point = (self.n - 1) ** 2
        nnz = np.sum(np.abs(matrix_p @ matrix_l + matrix_u) > epsilon)
        total_elements = number_of_grid_point**2
        return nnz, nnz / total_elements

    def get_cond(self):
        """Computes the condition number of the represented matrix.

        Returns
        -------
        float
            Condition number with respect to the infinity-norm
        """
        return np.linalg.cond(self.get_sparse().toarray(), np.inf)


def generate_label_for_polynomial(coeffs, var="n"):
    """Generates a label for a polynomial.

    Parameters
    ----------
    coeffs : array-like
        The coefficients of the polynomial fit.
    var : str, optional
        The variable name to use in the label. Default is "n".

    Returns
    -------
    str
        A label for the polynomial fit.
    """
    terms = [
        f"{coeff:.0f}{var}{'^'+str(exp) if exp > 1 else ''}"
        for exp, coeff in enumerate(coeffs[::-1])
    ][::-1]

    return " + ".join(terms).replace("+ -", "- ")


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        example()


@cli.command()
@utils.interval_option("n", start_default=1, stop_default=6, num_default=10, log=True)
@utils.display_or_save
def plot_theoretical_memory_usage(n_interval):
    """Plots the theoretical memory usage comparison between raw format and CRS
    format.
    """
    number_of_discretization_points = (n_interval - 1) ** 2

    # Calculate memory usage for raw and CRS formats
    raw_memory = (n_interval - 1) ** 4
    crs_memory = [BlockMatrix(n).eval_sparsity()[0] * 3 + 1 for n in n_interval]
    crs_ref_line = np.round(np.polyfit(n_interval, crs_memory, 2))

    plt.loglog(number_of_discretization_points, raw_memory, label="Raw format")
    plt.loglog(number_of_discretization_points, crs_memory, label="CRS format")
    plt.loglog(
        number_of_discretization_points,
        np.poly1d(crs_ref_line)(n_interval),
        label=f"${generate_label_for_polynomial(crs_ref_line)}$",
        linestyle="--",
        color="gray",
    )
    plt.title("Space Complexity vs Number of Discretization Points ($N$)")
    plt.xlabel("Number of Discretization Points ($N$)")
    plt.ylabel("Space Complexity")
    plt.grid(True, ls="--")
    plt.legend()


@cli.command()
@utils.interval_option("n", start_default=1, stop_default=6, num_default=10, log=True)
@utils.display_or_save
def plot_sparsity(n_interval):
    """Plots the number of non-zero entries in $A$ as a function of $n$ and $N$,
    and compares it with the number of entries in a fully populated matrix.
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    number_of_discretization_points = (n_interval - 1) ** 2

    nnz, rel_nnz = zip(*[BlockMatrix(n).eval_sparsity() for n in n_interval])
    nnz_ref_line = np.round(np.polyfit(n_interval, nnz, 2))
    nnz_ref_line_label = generate_label_for_polynomial(nnz_ref_line)

    # Plot for non-zero entries vs discretization points
    ax1.loglog(
        number_of_discretization_points,
        (n_interval - 1) ** 4,
        label="Number of entries in $A$",
        color="orange",
        linestyle="--",
    )
    ax1.loglog(
        number_of_discretization_points,
        nnz,
        alpha=0.7,
        label="Non-zero entries in $A$",
    )
    ax1.loglog(
        number_of_discretization_points,
        np.poly1d(nnz_ref_line)(n_interval),
        linestyle="--",
        color="gray",
        label=f"${nnz_ref_line_label}$",
    )
    ax1.set_title("Non-zero Entries vs Number of Discretization Points ($N$)")
    ax1.set_xlabel("Number of Discretization Points ($N$)")
    ax1.set_ylabel("Number of Non-zero Entries")
    ax1.grid(True, ls="--")
    ax1.legend()

    # Relative sparsity
    ax2.loglog(
        number_of_discretization_points,
        rel_nnz,
        alpha=0.7,
        label="Relative non-zero entries in $A$",
    )
    ax2.loglog(
        number_of_discretization_points,
        np.poly1d(nnz_ref_line)(n_interval) / number_of_discretization_points**2,
        linestyle="--",
        color="gray",
        label=f"$\\frac{{({nnz_ref_line_label})}}{{N^2}}$",
    )
    ax2.set_title("Relative Non-zero Entries vs Number of Discretization Points ($N$)")
    ax2.set_xlabel("Number of Discretization Points ($N$)")
    ax2.set_ylabel("Relative Number of Non-zero Entries")
    ax2.grid(True, ls="--")
    ax2.legend()


@cli.command()
@utils.interval_option("n", start_default=1, stop_default=6, num_default=10, log=True)
@click.option(
    "--epsilon",
    default=1e-3,
    type=click.FLOAT,
    help="Set the threshold epsilon for filtering entries in the LU decomposition.",
)
@utils.display_or_save
# pylint: disable=too-many-locals
def plot_sparsity_lu(n_interval, epsilon=1e-3):
    """Plots the number of non-zero entries in the matrix $A$ and its LU
    decomposition as a function of $N$.
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    number_of_discretization_points = (n_interval - 1) ** 2

    nnz_lu, rel_nnz_lu = zip(*[BlockMatrix(n).eval_sparsity_lu() for n in n_interval])
    nnz_ge_eps_lu, rel_nnz_ge_eps_lu = zip(
        *[BlockMatrix(n).eval_sparsity_lu(epsilon) for n in n_interval]
    )
    nnz_lu_ref_line = np.round(np.polyfit(n_interval, nnz_lu, 3))
    nnz_lu_ref_line_label = generate_label_for_polynomial(nnz_lu_ref_line)

    # Plot for non-zero entries vs discretization points
    ax1.loglog(
        number_of_discretization_points,
        (n_interval - 1) ** 4,
        label="Number of entries in LU-Decomposition",
        linestyle="--",
        color="orange",
    )
    ax1.loglog(
        number_of_discretization_points,
        nnz_lu,
        alpha=0.7,
        label="Non-zero entries in LU-Decomposition",
    )
    ax1.loglog(
        number_of_discretization_points,
        np.poly1d(nnz_lu_ref_line)(n_interval),
        linestyle="--",
        color="gray",
        label=f"${nnz_lu_ref_line_label}$",
    )
    ax1.loglog(
        number_of_discretization_points,
        nnz_ge_eps_lu,
        linestyle="--",
        label=f"Entries in LU-Decomposition > {epsilon} (by abs.)",
    )

    ax1.set_title("Sparsity analysis of the LU-Decomposition of $A$")
    ax1.set_xlabel("Number of Discretization Points ($N$)")
    ax1.set_ylabel("Number of Non-zero Entries")
    ax1.grid(True, ls="--")
    ax1.legend()

    # Relative sparsity
    ax2.loglog(
        number_of_discretization_points,
        rel_nnz_lu,
        alpha=0.7,
        label="Relative non-zero entries in LU-Decomposition",
    )
    ax2.loglog(
        number_of_discretization_points,
        np.poly1d(nnz_lu_ref_line)(n_interval) / number_of_discretization_points**2,
        linestyle="--",
        color="gray",
        label=f"$\\frac{{({nnz_lu_ref_line_label})}}{{N^2}}$",
    )
    ax2.loglog(
        number_of_discretization_points,
        rel_nnz_ge_eps_lu,
        linestyle="--",
        label=f"Relative entries in LU-Decomposition > {epsilon} (by abs.)",
    )
    ax2.set_title("Relative Sparsity analysis of the LU-Decomposition of $A$")
    ax2.set_xlabel("Number of Discretization Points ($N$)")
    ax2.set_ylabel("Relative Number of Non-zero Entries")
    ax2.grid(True, ls="--")
    ax2.legend()


# pylint: disable=no-value-for-parameter
def example():
    """Show example plots for various functions and analyses of this module"""
    print("Demonstrating block_matrix_2d.py ...")

    # Set numpy print options to fit the terminal width
    terminal_width = os.get_terminal_size().columns
    np.set_printoptions(linewidth=terminal_width)

    # Example usage of BlockMatrix class
    n = 4

    print(f"\nAnalyzing BlockMatrix with n = {n}:")
    bm = BlockMatrix(n)

    # Get the sparse matrix
    sparse_matrix = bm.get_sparse()
    print(f"Sparse Matrix (CSR format) for n = {n}:\n{sparse_matrix.toarray()}")

    # Evaluate sparsity
    nnz, rel_nnz = bm.eval_sparsity()
    print(f"Number of non-zero elements: {nnz}")
    print(f"Relative number of non-zero elements: {rel_nnz:.4f}")

    # Get LU decomposition
    matrix_p, matrix_l, matrix_u = bm.get_lu()
    print(f"Permutation matrix P for n = {n}:\n{matrix_p}")
    print(f"Lower triangular matrix L for n = {n}:\n{matrix_l}")
    print(f"Upper triangular matrix U for n = {n}:\n{matrix_u}")

    # Evaluate sparsity of LU decomposition
    nnz_lu, rel_nnz_lu = bm.eval_sparsity_lu()
    print(f"Number of non-zero elements in LU decomposition: {nnz_lu}")
    print(f"Relative number of non-zero elements in LU decomposition: {rel_nnz_lu:.4f}")

    # Get condition number
    cond = bm.get_cond()
    print(f"Condition number of the matrix: {cond}")

    interval_opts = {"start": 1, "end": 6, "num_points": 10}

    # Plotting functions
    print("\nPlotting theoretical memory usage comparison...")
    plot_theoretical_memory_usage(standalone_mode=False)

    print("\nPlotting non-zero entries in matrix A...")
    plot_sparsity(standalone_mode=False)

    print("\nPlotting non-zero entries in LU decomposition...")
    plot_sparsity_lu(standalone_mode=False)


if __name__ == "__main__":
    cli()
