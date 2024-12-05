# pylint: disable=missing-function-docstring,missing-module-docstring

import pytest
import numpy as np
import scipy.sparse as sp

from projektpraktikum_i.discretization.block_matrix_2d import BlockMatrix


@pytest.mark.parametrize("n", range(-20, 2))
def test_init_with_invalid_n(n):
    with pytest.raises(ValueError):
        BlockMatrix(n)


@pytest.mark.parametrize("n", range(2, 20))
def test_get_sparse(n):
    block_matrix = BlockMatrix(n)
    sparse = block_matrix.get_sparse()
    assert isinstance(sparse, sp.csr_matrix)
    assert sparse.shape == ((n - 1) ** 2, (n - 1) ** 2)


@pytest.mark.parametrize("n", range(2, 50))
def test_eval_sparsity(n):
    block_matrix = BlockMatrix(n)
    nnz, _ = block_matrix.eval_sparsity()
    assert nnz == block_matrix.get_sparse().nnz


@pytest.mark.parametrize("n", range(2, 10))
def test_get_lu(n):
    block_matrix = BlockMatrix(n)
    matrix_p, matrix_l, matrix_u = block_matrix.get_lu()
    matrix_a = block_matrix.get_sparse().toarray()
    assert np.allclose(matrix_p @ matrix_l @ matrix_u, matrix_a)
    assert np.allclose(matrix_l, np.tril(matrix_l, k=-1) + np.eye(matrix_l.shape[0]))
    assert np.allclose(matrix_u, np.triu(matrix_u))


@pytest.mark.skip(reason="Not implemented yet")
def test_get_cond():
    block_matrix = BlockMatrix(3)
    cond = block_matrix.get_cond()
    matrix_a = block_matrix.get_sparse().toarray()
    expected_cond = np.linalg.cond(matrix_a, np.inf)
    assert np.isclose(cond, expected_cond)


@pytest.mark.parametrize("n", range(2, 10))
@pytest.mark.skip(reason="Not implemented yet")
def test_eval_sparsity_lu(n):
    block_matrix = BlockMatrix(n)
    nnz_lu, sparsity_lu = block_matrix.eval_sparsity_lu()

    # Compute the expected number of non-zero elements in the LU decomposition
    matrix_p, matrix_l, matrix_u = block_matrix.get_lu()
    nnz_expected = (
        np.count_nonzero(matrix_p)
        + np.count_nonzero(matrix_l)
        + np.count_nonzero(matrix_u)
    )
    sparsity_expected = nnz_expected / (block_matrix.n - 1) ** 4

    assert nnz_lu == nnz_expected
    assert np.isclose(sparsity_lu, sparsity_expected)


if __name__ == "__main__":
    pytest.main()
