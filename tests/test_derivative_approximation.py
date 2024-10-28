import pytest

from projektpraktikum_i.derivative_approximation import FiniteDifference

def sample_function(x):
    return x**2

def test_compute_errors_without_derivatives():
    # Initialize without d_f and dd_f
    finite_diff = FiniteDifference(h=0.01, f=sample_function)

    with pytest.raises(ValueError):
        finite_diff.compute_errors(0, 10, 50)
