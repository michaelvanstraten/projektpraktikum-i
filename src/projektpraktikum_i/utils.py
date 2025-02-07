import os

# We need to import dill here first so we can hash lambda functions
import dill as pickle  # pylint: disable=unused-import
from joblib import Memory

memory = Memory(location=".cache")


def cache(func):
    cached_func = memory.cache(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if os.environ.get("DISABLE_JOBLIB_CACHE"):
            return func(*args, **kwargs)

        return cached_func(*args, **kwargs)

    return wrapper
