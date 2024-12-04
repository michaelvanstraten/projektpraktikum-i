"""
Module providing finite difference approximations and error analysis for numerical differentiation.

This module defines functions and a class to approximate the first and second derivatives of a
function using finite difference methods. It also supports error analysis for these approximations.

Functions:
- plot_errors: Plot errors in finite difference approximations.

Classes:
- FiniteDifference: Calculate first and second order finite difference approximations.
"""

# pylint: disable=unnecessary-lambda-assignment

from math import log10
from typing import Callable

from numpy.typing import NDArray
import matplotlib.pyplot as plt
import numpy as np


def is_vectorized(func: Callable, input_data: NDArray) -> bool:
    """
    Check if a function is vectorized by applying it to an array and checking the output shape.
    """
    try:
        # Attempt to apply the function to an array
        result = func(np.array(input_data))
        # Check if the result is an array of the same shape
        return (
            isinstance(result, np.ndarray)
            and result.shape == np.array(input_data).shape
        )
    except (TypeError, IndexError):
        return False


def infinity_norm(start: float, end: float, num_intervals: int) -> Callable:
    """
    Return a function to compute the infinity norm of the difference between two functions.
    """
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
            / self.__h**2
        )

    def compute_dh_left_f(self):
        """
        Calculates the approximation for the first derivative of f with step size h
        using the left finite difference.

        Returns
        -------
        callable
            Calculates the approximation of the first derivative for a given x.
        """

        return lambda x: (self.__f(x) - self.__f(x - self.__h)) / self.__h

    def compute_dh_central_f(self):
        """
        Calculates the approximation for the first derivative of f with step size h
        using the central finite difference.

        Returns
        -------
        callable
            Calculates the approximation of the first derivative for a given x.
        """

        return lambda x: (self.__f(x + self.__h) - self.__f(x - self.__h)) / (
            2 * self.__h
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

        return (
            norm(self.compute_dh_right_f(), self.__d_f),
            norm(self.compute_dh_central_f(), self.__d_f),
            norm(self.compute_dh_left_f(), self.__d_f),
            norm(self.compute_ddh_f(), self.__dd_f),
        )

    def plot(self, interval, dtype=None):
        """
        Plot the function f, its analytical derivatives df, ddf, and finite difference derivatives.

        Parameters
        ----------
        interval : tuple
            The interval over which to plot.
        dtype : type, optional
            The data type of the linspace.
        """

        inputs = np.linspace(*interval, dtype=dtype)

        plt.figure(figsize=(10, 6))

        def plot(*args, **kwargs):
            return plt.plot(inputs, *args, **kwargs)
        plot(self.__f(inputs), label="$f(x)$", color="red")
        plot(self.__d_f(inputs), label="$f'(x)$", color="orange")
        plot(self.__dd_f(inputs), label="$f''(x)$", color="green")
        plot(
            self.compute_dh_right_f()(inputs),
            label="$D_h^+ f(x)$",
            color="purple",
            linestyle="dashed",
        )
        plot(
            self.compute_dh_central_f()(inputs),
            label="$D_h^c f(x)$",
            color="blue",
            linestyle="dashed",
        )
        plot(
            self.compute_dh_right_f()(inputs),
            label="$D_h^- f(x)$",
            color="green",
            linestyle="dashed",
        )
        plot(
            self.compute_ddh_f()(inputs),
            label="$D_h^2 f(x)$",
            color="cyan",
            linestyle="dashed",
        )

        plt.xlabel("x")
        plt.ylabel("Function values")
        plt.title("Function and its Derivatives")

        # Add text with h value
        plt.text(
            0.90,
            0.95,
            f"$h = {self.__h}$",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox={
                "facecolor": "black",
                "edgecolor": "white",
                "boxstyle": "round,pad=0.5",
            },
        )

        plt.legend()


def plot_errors(h_values, f, d_f, dd_f, interval):  # pylint: disable=too-many-arguments
    """
    Plot the errors in finite difference approximations.
    """

    errors_right, errors_central, errors_left, errors_second = zip(
        *[FiniteDifference(h, f, d_f, dd_f).compute_errors(*interval) for h in h_values]
    )

    plt.figure(figsize=(10, 6))

    def plot(*args, **kwargs):
        return plt.loglog(h_values, *args, **kwargs)

    plot(errors_right, label="Error in $D_h^+ f(x)$", color="magenta", linewidth=2)
    plot(errors_central, label="Error in $D_h^c f(x)$", color="cyan", linewidth=2)
    plot(
        errors_left,
        label="Error in $D_h^- f(x)$",
        color="lime",
        linewidth=2,
        linestyle="dotted",
    )
    plot(errors_second, label="Error in $D_h^2 f(x)$", color="orange", linewidth=2)

    plot(h_values, label="$O(h)$", color="yellow", linestyle="dashed", linewidth=1.5)
    plot(
        h_values**2,
        label="$O(h^2)$",
        color="deepskyblue",
        linestyle="dashed",
        linewidth=1.5,
    )

    # Adjust limits to have padding
    error_values = np.concatenate(
        [errors_right, errors_central, errors_left, errors_second]
    )
    min_error = np.min(error_values)
    max_error = np.max(error_values)
    padding = 0.1
    shift = 10 ** (padding * (log10(max_error) - log10(min_error)))
    plt.ylim(
        bottom=min_error / shift,
        top=max_error * shift,
    )

    plt.xlabel("$h$")
    plt.ylabel("Error")
    plt.title("Error vs Step Size")
    plt.legend()


def main():
    """
    Main function to execute the plotting of functions and errors.
    """
    # Set plot style and generate plots
    plt.style.use("dark_background")

    # Define the function and its analytical derivatives
    def f(x):
        return np.sinc(x / np.pi)
    def d_f(x):
        return (x * np.cos(x) - np.sin(x)) / x ** 2
    def dd_f(x):
        return -((x ** 2 - 2) * np.sin(x) + 2 * x * np.cos(x)) / x ** 3

    # Define interval and discretization parameters
    interval = (np.pi, 3 * np.pi, 1000)

    # Initialize a FiniteDifference instance for plotting functions
    h = 0.5
    FiniteDifference(h, f, d_f, dd_f).plot(interval)

    # Define step sizes and create finite difference objects
    h_values = np.logspace(-18, 2, 1000, base=10)
    plot_errors(h_values, f, d_f, dd_f, interval)

    plt.grid(True, alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()
