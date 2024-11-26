import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sc
import scipy.sparse as sp


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
            raise ValueError(
                "The parameter `n` must be at least 2. "
                "Please provide a value `n > 1` when initializing `BlockMatrix`."
            )
        self.n = n

    def get_sparse(self):
        """Returns the block matrix as sparse matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            The block_matrix in a sparse data format.
        """

        n = self.n
        C = sp.diags([4, -1, -1], [0, -1, 1], shape=(n - 1, n - 1))
        C_diag = sp.block_diag([C] * (n - 1))
        I_diag = sp.diags(
            [-1, -1], [n - 1, -(n - 1)], shape=((n - 1) ** 2, (n - 1) ** 2)
        )

        A_sparse = C_diag + I_diag
        return A_sparse

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
        A = self.get_sparse()
        nze = A.nnz
        sparsity = nze / (self.n - 1) ** 4

        return nze, sparsity

    def get_lu(self):
        """Provides an LU-Decomposition of the represented matrix A of the
        form A = p * l * u

        Returns
        -------
        p : numpy.ndarray
            permutation matrix of LU-decomposition
        l : numpy.ndarray
            lower triangular unit diagonal matrix of LU-decomposition
        u : numpy.ndarray
            upper triangular matrix of LU-decomposition
        """

        A = self.get_sparse().toarray()
        P, L, U = sc.lu(A)
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

    def get_cond(self):
        """Computes the condition number of the represented matrix.

        Returns
        -------
        float
            condition number with respect to the infinity-norm
        """


def plot_sparse_vs_full(n: int):
    if n < 2:
        raise ValueError
    n_werte = np.arange(2, n)

    sparse_einträge = 3 * (5 * n_werte**2 - 14 * n_werte + 9)
    vollbesetzt = (n_werte - 1) ** 4

    plt.figure(figsize=(10, 6))
    plt.yscale("log")
    plt.xscale("log")
    plt.plot(n_werte, sparse_einträge, label="Sparse Matrix", color="blue")
    plt.plot(
        n_werte, vollbesetzt, label="Vollbesetzte Matrix", color="red", linestyle="--"
    )
    plt.xlabel("n")
    plt.ylabel("Speicherplatz")
    plt.legend()
    plt.title("Vergleich des Speicherplatzbedarfs (Sparse vs Vollbesetzt) von A")
    plt.grid(True)
    plt.show()


def main():
    n = 5
    block_matrix = BlockMatrix(n)

    A = block_matrix.get_sparse()
    print("Matrix A:")
    print(A)

    nze, relative_sparsity = block_matrix.eval_sparsity()
    print(f"Absolute Anzahl der nicht-Null-Einträge: {nze}")
    print(f"Relative Anzahl der nicht-Null-Einträge: {relative_sparsity}")


if __name__ == "__main__":
    main()
