from typing import Callable

from numpy.typing import NDArray
import numpy as np

def is_vectorized(func: Callable, input_data: NDArray) -> bool:
    try:
        # Attempt to apply the function to an array
        result = func(np.array(input_data))
        # Check if the result is an array of the same shape
        return (
            isinstance(result, np.ndarray)
            and result.shape == np.array(input_data).shape
        )
    except Exception:
        return False


def infinity_norm(start: float, end: float, num_intervals: int) -> Callable:
    points = np.linspace(start, end, num_intervals)

    def approximation(f, g):
        if not is_vectorized(f, points): 
            f = np.vectorize(f)
        if not is_vectorized(g, points):
            g = np.vectorize(g)

        return np.max(np.abs(f(points) - g(points)))

    return approximation


class FiniteDifference:
    """
    Represents the first and second order finite difference approximation
    of a function and allows for a computation of error to the exact
    derivatives.

    Parameters
    ----------
    h : float
        Step size of the approximation.
    f : callable
        Function to approximate the derivatives of. The calling signature is
        ‘f(x)‘. Here ‘x‘ is a scalar or array_like of ‘numpy‘. The return
        value is of the same type as ‘x‘.
    d_f : callable, optional
        The analytic first derivative of ‘f‘ with the same signature.
    dd_f : callable, optional
        The analytic second derivative of ‘f‘ with the same signature.

    Attributes
    ----------
    h : float
        Step size of the approximation.
    """

    def __init__(self, h, f, d_f=None, dd_f=None):
        self.__h = h
        self.__f = f
        self.__d_f = d_f
        self.__dd_f = dd_f

    def compute_dh_right_f(self):
        """
        Calculates the approximation for the first derivative of the function `f`
        with step size `h` using the first right finite difference.

        Returns
        -------
        callable
            Calculates the approximation of the first derivative for a given x.
        """
        return lambda x: (self.__f(x + self.__h) - self.__f(x)) / self.__h

    def compute_ddh_f(self):
        """
        Calculates the approximation for the second derivative of the function `f`
        with step size `h`.

        Returns
        -------
        callable
            Calculates the approximation of the second derivative for a given x.
        """
        return (
            lambda x: (
                self.__f(x + self.__h) - 2 * self.__f(x) + self.__f(x - self.__h)
            )
            / self.__h
        )

    def compute_errors(self, a, b, p):
        """
        Calculates an approximation of the errors between the numerical approximation
        and the exact derivative for first and second order derivatives in the
        infinity norm.

        Parameters
        ----------
        a : float
            Start point of the interval.
        b : float
            End point of the interval.
        p : int
            Number of intervals used in the approximation of the infinity norm.

        Returns
        -------
        float
            Errors of the approximation of the first derivative, using the first
            right finite difference.
        float
            Errors of the approximation of the second derivative.

        Raises
        ------
        ValueError
            If no analytic derivative was provided by the user.
        """

        if not self.__d_f or not self.__dd_f:
            raise ValueError(
                "Analytic derivatives `d_f` and `dd_f` are required.\n"
                "Provide callable functions for them when initializing `FiniteDifference`."
            )

        norm = infinity_norm(a, b, p)

        return norm(self.compute_dh_right_f(), self.__d_f), norm(
            self.compute_ddh_f(), self.__dd_f
        )
