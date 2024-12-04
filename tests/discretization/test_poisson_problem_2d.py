# pylint: disable=missing-function-docstring,missing-module-docstring

import pytest

import numpy as np

from projektpraktikum_i.discretization.poisson_problem_2d import (
    idx,
    inv_idx,
    rhs,
)


# Example function for rhs
def f(x):
    return x[0] * np.sin(np.pi * x[0]) * x[1] * np.sin(np.pi * x[1])


@pytest.mark.parametrize("n", range(2, 20))
def test_idx_inv_idx_roundtrip(n):
    """Test the roundtrip identity check for idx and inv_idx."""
    for i in range(1, n):
        for j in range(1, n):
            coordinates = [i, j]
            equation_number = idx(coordinates, n)
            recovered_coordinates = inv_idx(equation_number, n)
            assert recovered_coordinates == coordinates


@pytest.mark.parametrize("n", range(2, 20))
def test_rhs(n):
    """Test the rhs function."""
    result = rhs(n, f)
    assert len(result) == (n - 1) ** 2


def test_rhs_raises_error():
    """Test that rhs raises a ValueError for n < 2."""
    with pytest.raises(ValueError):
        rhs(1, f)


if __name__ == "__main__":
    pytest.main()
