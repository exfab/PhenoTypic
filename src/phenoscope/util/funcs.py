import numpy as np
import time


def is_binary_mask(arr: np.ndarray):
    return True if (arr.ndim == 2 or arr.ndim == 3) and np.all((arr == 0) | (arr == 1)) else False


def timed_execution(func):
    """
    Decorator to measure and print the execution time of a function.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Execute the wrapped function
        end_time = time.time()  # Record the end time
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result

    return wrapper
