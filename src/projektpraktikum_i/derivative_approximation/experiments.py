"""
Finite Difference Derivative Approximation Experimentation Script
"""

# pylint: disable=unnecessary-lambda-assignment

import click
import matplotlib.pyplot as plt
import numpy as np
from projektpraktikum_i.derivative_approximation.finite_difference import (
    FiniteDifference,
    plot_errors,
)

# Define target function and its derivatives
f = lambda x: np.sin(x) / x
d_f = lambda x: (x * np.cos(x) - np.sin(x)) / (x**2)
dd_f = lambda x: -((x**2 - 2) * np.sin(x) + 2 * x * np.cos(x)) / (x**3)


@click.group()
@click.option(
    "--save-path",
    type=click.Path(dir_okay=False, writable=True),
    help="Specify a file path to save the generated plot.",
)
@click.pass_context
def cli(ctx, save_path):
    """
    Finite Difference Derivative Approximation Experimentation Script.
    """
    ctx.ensure_object(dict)
    ctx.obj["SAVE_PATH"] = save_path
    plt.style.use("dark_background")


# Input interval option decorator
input_interval_option = click.option(
    "--input-interval",
    type=click.Tuple([click.FLOAT, click.FLOAT, click.INT]),
    default=(np.pi, 3 * np.pi, 1000),
    help="Define the input interval as (start, end, number of points).",
)


@cli.command("plot-compare")
@click.option(
    "--approximation-type",
    type=click.Choice(["first-right", "first-left", "first-central", "second"]),
    required=True,
    help="Choose the type of derivative approximation to plot.",
)
@input_interval_option
@click.pass_context
def plot_compare_command(ctx, approximation_type, input_interval):
    """
    Plot and compare the exact and approximate derivatives.

    Plots the true derivative alongside finite difference approximations using
    specified step sizes for the given approximation type.
    """
    x_values = np.linspace(*input_interval)

    # Define step sizes and labels for plotting
    step_sizes = [
        (np.pi / 3, r"$h = \frac{\pi}{3}$"),
        (np.pi / 4, r"$h = \frac{\pi}{4}$"),
        (np.pi / 5, r"$h = \frac{\pi}{5}$"),
        (np.pi / 10, r"$h = \frac{\pi}{10}$"),
    ]

    plt.figure()

    # Mapping of approximation types to their corresponding methods
    approximation_methods = {
        "first-right": lambda fd: fd.compute_dh_right_f()(x_values),
        "first-left": lambda fd: fd.compute_dh_left_f()(x_values),
        "first-central": lambda fd: fd.compute_dh_central_f()(x_values),
        "second": lambda fd: fd.compute_ddh_f()(x_values),
    }

    # Plot true derivative based on the selected approximation type
    if approximation_type == "second":
        analytical_values = dd_f(x_values)
        analytical_label = "$g''(x)$"
    else:
        analytical_values = d_f(x_values)
        analytical_label = "$g'(x)$"
    plt.plot(x_values, analytical_values, label=analytical_label, color="grey")

    # Plot finite difference approximations for each step size
    for h, h_label in step_sizes:
        f_d = FiniteDifference(h, f, d_f, dd_f)
        approx_values = approximation_methods[approximation_type](f_d)
        plt.plot(
            x_values,
            approx_values,
            label=f"Finite Approximation for {h_label}",
            linestyle="dashed",
        )

    plt.grid()
    plt.legend()

    # Save or display plot
    if ctx.obj["SAVE_PATH"]:
        plt.savefig(ctx.obj["SAVE_PATH"])
    else:
        plt.show()


@cli.command("plot-errors")
@click.option(
    "--log-h-interval",
    type=click.Tuple([click.INT, click.INT, click.INT]),
    default=(-18, 2, 1000),
    help="Defines the interval for h values on a log scale as (start, stop, num).",
)
@input_interval_option
@click.pass_context
def plot_errors_command(ctx, log_h_interval, input_interval):
    """
    Plot finite difference approximation errors as a function of step size.

    Generates a plot of approximation errors over a range of h values in logarithmic
    scale for the defined input interval.
    """
    h_values = np.logspace(*log_h_interval)

    plot_errors(h_values, f, d_f, dd_f, input_interval)

    # Save or display plot
    if ctx.obj["SAVE_PATH"]:
        plt.savefig(ctx.obj["SAVE_PATH"])
    else:
        plt.show()


if __name__ == "__main__":
    cli(obj={})  # pylint: disable=no-value-for-parameter
