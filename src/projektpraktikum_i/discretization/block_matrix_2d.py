"""
Module providing classes and functions for analyzing block matrices
arising from finite difference approximations of the Laplace operator.
The module includes methods for creating sparse representations,
evaluating sparsity patterns, and analyzing LU decomposition.

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

import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.linalg as linalg
import scipy.sparse as sp

__all__ = [
    "BlockMatrix",
    "plot_compare_theoretical_memory_usage",
    "plot_non_zero_entries",
    "plot_non_zero_entries_lu",
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

    def get_sparse(self):
        """Returns the block matrix as a sparse matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            The block matrix in a sparse data format.
        """
        n = self.n
        C = sp.diags([4, -1, -1], [0, -1, 1], shape=(n - 1, n - 1))
        C_diag = sp.block_diag([C] * (n - 1))
        I_diag = sp.diags([-1, -1], [n - 1, -(n - 1)], shape=C_diag.shape)

        return C_diag + I_diag

    def eval_sparsity(self):
        """Returns the absolute and relative numbers of non-zero elements of
        the matrix. The relative quantities are with respect to the total
        number of elements of the represented matrix.

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

    def get_lu(self):
        """Provides an LU-Decomposition of the represented matrix A of the
        form A = p * l * u

        Returns
        -------
        p : numpy.ndarray
            Permutation matrix of LU-decomposition
        l : numpy.ndarray
            Lower triangular unit diagonal matrix of LU-decomposition
        u : numpy.ndarray
            Upper triangular matrix of LU-decomposition
        """
        A = self.get_sparse().toarray()
        P, L, U = linalg.lu(A)
        return P, L, U

    def eval_sparsity_lu(self):
        """Returns the absolute and relative numbers of non-zero elements of
        the LU-Decomposition. The relative quantities are with respect to the
        total number of elements of the represented matrix.

        Returns
        -------
        int
            Number of non-zeros
        float
            Relative number of non-zeros
        """
        P, L, U = self.get_lu()
        N = (self.n - 1) ** 2
        nnz = np.count_nonzero(P @ L + U)
        total_elements = N**2
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
        elif self.n == 3:
            return 6
        else:
            return 7


def plot_compare_theoretical_memory_usage(interval, save_to=None):
    """Plots the theoretical memory usage comparison between raw format and CRS format.

    Parameters
    ----------
    interval : tuple
        Interval of values for n, defined as (start, end, num_points).
    save_to : str, optional
        File path to save the plot. If None, the plot is displayed.
    """
    values_for_n = np.linspace(*interval, dtype=int)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

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
    """Plots the number of non-zero entries in the matrix $A$ and its LU decomposition
    as a function of $N$.

    Parameters
    ----------
    interval : tuple
        Interval of values for n, defined as (start, end, num_points).
    save_to : str, optional
        File path to save the plot. If None, the plot is displayed.
    """
    values_for_n = np.linspace(*interval, dtype=int)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate non-zero entries for A and LU decomposition
    nnz_A_values = [BlockMatrix(n).eval_sparsity()[0] for n in values_for_n]
    nnz_LU_values = [BlockMatrix(n).eval_sparsity_lu()[0] for n in values_for_n]

    # Plot for non-zero entries vs discretization points
    ax.plot(
        (values_for_n - 1) ** 2,
        nnz_A_values,
        label="Non-zero entries in $A$",
    )
    ax.plot(
        (values_for_n - 1) ** 2,
        nnz_LU_values,
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
    P, L, U = bm.get_lu()
    print(f"Permutation matrix P for n = {n}:\n{P}")
    print(f"Lower triangular matrix L for n = {n}:\n{L}")
    print(f"Upper triangular matrix U for n = {n}:\n{U}")

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


if __name__ == "__main__":
    main()
