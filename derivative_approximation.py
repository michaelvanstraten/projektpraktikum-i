from typing import Callable

from manim.camera.mapping_camera import math
from numpy.typing import NDArray
import numpy as np

from projektpraktikum_1.utils import is_vectorized


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
        if not is_vectorized(f, points) or not is_vectorized(g, points):
            raise ValueError(
                "Both functions must support vectorized input using NumPy arrays."
            )

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
        if not is_vectorized(f, np.linspace(0, 1)):
            self.__f = np.vectorize(f)
        else:
            self.__f = f
        if d_f and not is_vectorized(d_f, np.linspace(0, 1)):
            self.__d_f = np.vectorize(d_f)
        else:
            self.__d_f = d_f
        if dd_f and not is_vectorized(dd_f, np.linspace(0, 1)):
            self.__dd_f = np.vectorize(dd_f)
        else:
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

        norm = infinity_norm(a, b, p)

        return norm(self.compute_dh_right_f(), self.__d_f), norm(
            self.compute_ddh_f(), self.__dd_f
        )


from manim import *


class LogScalingExample(Scene):
    def construct(self):
        ax = Axes(
            x_range=[-24, 24],
            y_range=[-1.2, 1.2],
            axis_config={"include_numbers": True})

        h = ValueTracker(12)

        d_f_approximation = ax.plot(
            lambda x: x,
            use_vectorized=True,
        )

        f = lambda x: np.sin(x) / x
        d_f = lambda x: (x * np.cos(x) - np.sin(x)) / x**2
        dd_f = lambda x: -((x**2 - 2) * np.sin(x) + 2 * x * np.cos(x)) / x**3

        d_f_approximation.add_updater(
            lambda old_graph: old_graph.become(
                ax.plot(
                    FiniteDifference(h.get_value(), f, d_f, dd_f).compute_dh_right_f(),
                    use_vectorized=True,
                )
            )
        )


        d_f_h_pi =                 ax.plot(
                    FiniteDifference(math.pi/10, f, d_f, dd_f).compute_dh_right_f(),
                    use_vectorized=True,
                )


        d_f_plot = ax.plot(
            d_f,
            use_vectorized=True,
            color=RED
        )


        self.add(ax, d_f_approximation, d_f_plot, d_f_h_pi)

        self.play(h.animate.set_value(0.0001), run_time=5)
