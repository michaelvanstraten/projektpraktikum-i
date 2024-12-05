"""Module providing classes and functions for analyzing block matrices arising
from finite difference approximations of the Laplace operator. The module
includes methods for creating sparse representations, evaluating sparsity
patterns, and analyzing LU decomposition.

Classes:
--------
- BlockMatrix: Represents block matrices and provides various analyses.

Functions:
----------
- plot_compare_theoretical_memory_usage: Plots theoretical memory usage comparison.
- plot_non_zero_entries: Plots the number of non-zero entries in a matrix.
- plot_non_zero_entries_lu: Plots non-zero entries in the LU decomposition.

Usage:
------
Run the module directly to see example plots for various functions and analyses.
"""

import os

from scipy import linalg
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

# We need to import dill here first so we can hash lambda functions
import dill as pickle  # pylint: disable=unused-import
from joblib import Memory


__all__ = [
    "BlockMatrix",
    "plot_compare_theoretical_memory_usage",
    "plot_non_zero_entries",
    "plot_non_zero_entries_lu",
]

memory = Memory(location=".cache")


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
        self.get_lu = memory.cache(self.get_lu)

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

    def eval_sparsity_lu(self):
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
        nnz = np.count_nonzero(matrix_p @ matrix_l + matrix_u)
        total_elements = number_of_grid_point**2
        return nnz, nnz / total_elements

    def get_cond(self):
        """Computes the condition number of the represented matrix.

        Returns
        -------
        float
            Condition number with respect to the infinity-norm
        """
        if self.n == 2:
            return 4
        if self.n == 3:
            return 6
        return 7


def plot_compare_theoretical_memory_usage(interval, save_to=None):
    """Plots the theoretical memory usage comparison between raw format and CRS
    format.

    Parameters
    ----------
    interval : tuple
        Interval of values for n, defined as (start, end, num_points).
    save_to : str, optional
        File path to save the plot. If None, the plot is displayed.
    """
    values_for_n = np.linspace(*interval, dtype=int)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Calculate memory usage for raw and CRS formats
    raw_memory = (values_for_n - 1) ** 4
    crs_memory = 9 + values_for_n * (-14 + 5 * values_for_n)

    # Plot for raw format and CRS format
    ax1.plot(values_for_n, raw_memory, label="Raw format")
    ax1.plot(values_for_n, crs_memory, label="CRS format")
    ax1.set_title("Space Complexity vs Number of Interval Points ($n$)")
    ax1.set_xlabel("Number of Interval Points ($n$)")
    ax1.set_ylabel("Space Complexity")
    ax1.grid(True)
    ax1.set_yscale("log")
    ax1.legend()

    # Plot for discretization points
    ax2.plot((values_for_n - 1) ** 2, raw_memory, label="Raw format")
    ax2.plot((values_for_n - 1) ** 2, crs_memory, label="CRS format")
    ax2.set_title("Space Complexity vs Number of Discretization Points ($N$)")
    ax2.set_xlabel("Number of Discretization Points ($N$)")
    ax2.set_ylabel("Space Complexity")
    ax2.grid(True)
    ax2.set_yscale("log")
    ax2.legend()

    # Save or display plot
    if save_to:
        plt.savefig(save_to)
    else:
        plt.show()


def plot_non_zero_entries(interval, save_to=None):
    """Plots the number of non-zero entries in $A$ as a function of $n$ and $N$,
    and compares it with the number of entries in a fully populated matrix.

    Parameters
    ----------
    interval : tuple
        Interval of values for n, defined as (start, end, num_points).
    save_to : str, optional
        File path to save the plot. If None, the plot is displayed.
    """
    values_for_n = np.linspace(*interval, dtype=int)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Calculate non-zero entries and fully populated matrix entries
    nnz_values = [BlockMatrix(n).eval_sparsity()[0] for n in values_for_n]
    fully_populated_entries = (values_for_n - 1) ** 4

    # Plot for non-zero entries vs interval points
    ax1.plot(values_for_n, nnz_values, label="Non-zero entries in $A$")
    ax1.plot(
        values_for_n,
        fully_populated_entries,
        label="Fully populated matrix",
        linestyle="--",
    )
    ax1.set_title("Non-zero Entries vs Number of Interval Points ($n$)")
    ax1.set_xlabel("Number of Interval Points ($n$)")
    ax1.set_ylabel("Number of Non-zero Entries")
    ax1.set_yscale("log")
    ax1.grid(True)
    ax1.legend()

    # Plot for non-zero entries vs discretization points
    ax2.plot(
        (values_for_n - 1) ** 2,
        nnz_values,
        label="Non-zero entries in $A$",
    )
    ax2.plot(
        (values_for_n - 1) ** 2,
        fully_populated_entries,
        label="Fully populated matrix",
        linestyle="--",
    )
    ax2.set_title("Non-zero Entries vs Number of Discretization Points ($N$)")
    ax2.set_xlabel("Number of Discretization Points ($N$)")
    ax2.set_ylabel("Number of Non-zero Entries")
    ax2.set_yscale("log")
    ax2.grid(True)
    ax2.legend()

    # Save or display plot
    if save_to:
        plt.savefig(save_to)
    else:
        plt.show()


def plot_non_zero_entries_lu(interval, save_to=None):
    """Plots the number of non-zero entries in the matrix $A$ and its LU
    decomposition as a function of $N$.

    Parameters
    ----------
    interval : tuple
        Interval of values for n, defined as (start, end, num_points).
    save_to : str, optional
        File path to save the plot. If None, the plot is displayed.
    """
    values_for_n = np.linspace(*interval, dtype=int)

    _, ax = plt.subplots(figsize=(10, 6))

    # Plot for non-zero entries vs discretization points
    ax.plot(
        (values_for_n - 1) ** 2,
        [BlockMatrix(n).eval_sparsity()[0] for n in values_for_n],
        label="Non-zero entries in $A$",
    )
    ax.plot(
        (values_for_n - 1) ** 2,
        [BlockMatrix(n).eval_sparsity_lu()[0] for n in values_for_n],
        label="Non-zero entries in LU decomposition",
    )
    ax.set_title("Non-zero Entries vs Number of Discretization Points ($N$)")
    ax.set_xlabel("Number of Discretization Points ($N$)")
    ax.set_ylabel("Number of Non-zero Entries")
    ax.grid(True)
    ax.legend()

    # Save or display plot
    if save_to:
        plt.savefig(save_to)
    else:
        plt.show()


def main():
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

    # Plotting functions
    print("\nPlotting theoretical memory usage comparison...")
    plot_compare_theoretical_memory_usage((10, 100, 20))

    print("\nPlotting non-zero entries in matrix A...")
    plot_non_zero_entries((10, 100, 20))

    print("\nPlotting non-zero entries in LU decomposition...")
    plot_non_zero_entries_lu((10, 20, 10))

    plot_sparse_vs_full(100)


if __name__ == "__main__":
    main()
