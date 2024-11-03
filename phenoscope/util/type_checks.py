import numpy as np


def is_binary_mask(arr: np.ndarray):
    if (arr.ndim == 2 or arr.ndim == 3) and np.all((arr == 0) | (arr == 1)):
        return True
    else:
        return False
