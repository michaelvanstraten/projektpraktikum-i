import matplotlib.pyplot as plt
import numpy as np

from math import log10
from numpy.typing import NDArray
from typing import Callable


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
                self.__f(x + self.__h)
                - 2 * self.__f(x)
                + self.__f(x - self.__h)
            )
            / self.__h
        )

    def compute_dh_left_f(self):
        """
        Calculates the approximation for the first derivative of f with step size h
        using the left finite difference.

        Returns
        -------
        callable
            Calculates the approximation of the second derivative for a given x.
        """

        return lambda x: (self.__f(x) - self.__f(x - self.__h)) / self.__h

    def compute_dh_central_f(self):
        """
        Calculates the approximation for the first derivative of f with step size h
        using the central finite difference.

        Returns
        -------
        callable
            Calculates the approximation of the second derivative for a given x.
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


def plot_functions(f, df, ddf, f_d, params):
    """Plot the function f, its analytical derivatives df, ddf, and finite difference derivatives."""
    inputs = np.linspace(*params)

    plt.figure(figsize=(10, 6))

    plot = lambda *args, **kwargs: plt.plot(inputs, *args, **kwargs)
    plot(f(inputs), label="$f(x)$", color="red")
    plot(df(inputs), label="$f'(x)$", color="orange")
    plot(
        f_d.compute_dh_right_f()(inputs),
        label="$D_h^+ f(x)$",
        color="purple",
        linestyle="dashed",
    )
    plot(
        f_d.compute_ddh_f()(inputs),
        label="$D_h^2 f(x)$",
        color="blue",
        linestyle="dashed",
    )

    plt.xlabel("x")
    plt.ylabel("Function values")
    plt.title("Function and its Derivatives")
    plt.legend()


def plot_errors(f_ds, h_values, params, padding=0.1):
    """Plot the errors in finite difference approximations."""
    a, b, p = params
    e_f_1_right, e_f_1_central, e_f_1_left, e_f_2 = zip(
        *[f_d.compute_errors(a, b, p) for f_d in f_ds]
    )

    plt.figure(figsize=(10, 6))

    plot = lambda *args, **kwargs: plt.loglog(h_values, *args, **kwargs)
    plot(e_f_1_right, label="Error in $D_h^+ f$", color="green", linewidth=4)
    plot(e_f_1_central, label="Error in $D_h^c f$", color="blue")
    plot(e_f_1_left, label="Error in $D_h^- f$", color="red")
    plot(h_values, label="$O(h)$", color="grey", linestyle="dashed")
    plot(h_values**2, label="$O(h^2)$", color="grey", linestyle="dashed")

    error_values = np.concatenate([e_f_1_right, e_f_1_central, e_f_1_left])
    min_error = np.min(error_values)
    max_error = np.max(error_values)
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
    """Main function to execute the plotting of functions and errors."""
    # Define the function and its analytical derivatives
    f = lambda x: np.sinc(x / np.pi)
    df = lambda x: (x * np.cos(x) - np.sin(x)) / x**2
    ddf = lambda x: -((x**2 - 2) * np.sin(x) + 2 * x * np.cos(x)) / (x**3)

    # Define interval and discretization parameters
    params = (np.pi, 3 * np.pi, 1000)

    # Define step sizes and create finite difference objects
    h_values = np.logspace(10, -25, 1000, base=10)
    f_ds = [FiniteDifference(h, f, df, ddf) for h in h_values]

    # Initialize a FiniteDifference instance for plotting functions
    h = 0.1
    f_d = FiniteDifference(h, f, df, ddf)

    # Set plot style and generate plots
    plt.style.use("dark_background")
    plot_functions(f, df, ddf, f_d, params)
    plot_errors(f_ds, h_values, params)

    plt.grid(True, alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()
