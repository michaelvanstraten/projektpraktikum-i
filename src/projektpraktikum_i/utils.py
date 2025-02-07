"""
Module containing utility functions and decorators.
"""

import functools
import os

# We need to import dill here first so we can hash lambda functions
import dill as pickle  # pylint: disable=unused-import
from joblib import Memory

import click
import matplotlib.pyplot as plt
import numpy as np

memory = Memory(location=".cache")

__all__ = [
    "cache",
    "display_or_save",
    "interval_option",
]

def cache(func):
    """
    Decorator to cache the result of a function using joblib's memory.
    """
    cached_func = memory.cache(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if os.environ.get("DISABLE_JOBLIB_CACHE"):
            return func(*args, **kwargs)

        return cached_func(*args, **kwargs)

    return wrapper


def display_or_save(func):
    """
    Decorator to add a `--save-to` option to a Click command.
    If `--save-to` is provided, the plot is saved to the specified file.
    Otherwise, the plot is shown interactively.
    """

    @click.option(
        "--save-to",
        type=click.Path(dir_okay=False, writable=True),
        help="Save the plot to the specified file instead of showing it interactively.",
    )
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        save_to = kwargs.pop("save_to", None)
        func(*args, **kwargs)
        if save_to:
            plt.savefig(save_to)
            click.echo(f"Plot saved to {save_to}")
        else:
            plt.show()

    return wrapper


def interval_option(
    prefix,
    start_default,
    stop_default,
    num_default,
    endpoint=True,
    log=False,
    dtype=int,
):
    """Decorator factory to add interval options with prefix."""

    def decorator(func):
        @click.option(
            f"--{prefix}-start",
            default=start_default,
            type=float if dtype == float else int,
            show_default=True,
            help=f"Start value for {prefix} interval{' (log base 2)' if log else ''}.",
        )
        @click.option(
            f"--{prefix}-stop",
            default=stop_default,
            type=float if dtype == float else int,
            show_default=True,
            help=f"Stop value for {prefix} interval{' (log base 2)' if log else ''}.",
        )
        @click.option(
            f"--{prefix}-num",
            default=num_default,
            type=int,
            show_default=True,
            help=f"Number of samples to generate.",
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract parameters and generate interval
            start = kwargs.pop(f"{prefix}_start")
            end = kwargs.pop(f"{prefix}_stop")
            num_points = kwargs.pop(f"{prefix}_num")

            params = {
                "start": start,
                "stop": end,
                "num": num_points,
                "endpoint": endpoint,
                "dtype": dtype,
            }

            if log:
                interval = np.logspace(**params, base=2)
            else:
                interval = np.linspace(**params)

            return func(*args, **kwargs, **{f"{prefix}_interval": interval})

        return wrapper

    return decorator
