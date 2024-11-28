import pytest
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sc

from projektpraktikum_i.discretization.block_matrix_2d import BlockMatrix


@pytest.mark.parametrize("n", [n for n in range(-20, 2)])
def test_init_with_invalid_n(n):
    with pytest.raises(ValueError):
        BlockMatrix(n)


@pytest.mark.parametrize("n", [n for n in range(2, 20)])
def test_get_sparse(n):
    block_matrix = BlockMatrix(n)
    sparse = block_matrix.get_sparse()
    assert isinstance(sparse, sp.csr_matrix)
    assert sparse.shape == ((n - 1) ** 2, (n - 1) ** 2)


@pytest.mark.parametrize("n", [n for n in range(2, 50)])
def test_eval_sparsity(n):
    block_matrix = BlockMatrix(4)

    nnz, _ = block_matrix.eval_sparsity()
    assert nnz == block_matrix.get_sparse().nnz


@pytest.mark.parametrize("n", [n for n in range(2, 10)])
def test_get_lu(n):
    block_matrix = BlockMatrix(n)
    P, L, U = block_matrix.get_lu()
    A = block_matrix.get_sparse().toarray()
    assert np.allclose(P @ L @ U, A)
    assert np.allclose(L, np.tril(L, k=-1) + np.eye(L.shape[0]))
    assert np.allclose(U, np.triu(U))


@pytest.mark.skip(reason="Not implemented yet")
def test_get_cond():
    block_matrix = BlockMatrix(3)
    cond = block_matrix.get_cond()
    A = block_matrix.get_sparse().toarray()
    expected_cond = np.linalg.cond(A, np.inf)
    assert np.isclose(cond, expected_cond)


@pytest.mark.parametrize("n", [n for n in range(2, 10)])
@pytest.mark.skip(reason="Not implemented yet")
def test_eval_sparsity_lu(n):
    block_matrix = BlockMatrix(n)
    nnz_lu, sparsity_lu = block_matrix.eval_sparsity_lu()

    # Compute the expected number of non-zero elements in the LU decomposition
    P, L, U = block_matrix.get_lu()
    nnz_expected = np.count_nonzero(P) + np.count_nonzero(L) + np.count_nonzero(U)
    sparsity_expected = nnz_expected / (block_matrix.n - 1) ** 4

    assert nnz_lu == nnz_expected
    assert np.isclose(sparsity_lu, sparsity_expected)


if __name__ == "__main__":
    pytest.main()
