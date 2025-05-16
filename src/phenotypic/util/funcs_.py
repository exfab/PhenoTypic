from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

import numpy as np
import time
import inspect

from phenotypic.util.exceptions_ import OperationIntegrityError


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


def is_static_method(owner_cls: type, method_name: str) -> bool:
    """
    Return True if *method_name* is defined on *owner_cls* (or an
    ancestor in its MRO) as a staticmethod.
    """
    # Retrieve attribute without invoking the descriptor protocol
    attr = inspect.getattr_static(owner_cls, method_name)  # Python ≥3.2
    return isinstance(attr, staticmethod)  # True ⇒ @staticmethod


def validate_array_integrity(pre_op_imcopy: Image, post_op_image: Image) -> None:
    if post_op_image.imformat.is_array():
        if not np.array_equal(pre_op_imcopy.array[:], post_op_image.array[:]):
            raise OperationIntegrityError('array', post_op_image.name)


def validate_matrix_integrity(pre_op_imcopy: Image, post_op_image: Image) -> None:
    if not np.array_equal(pre_op_imcopy.matrix[:], post_op_image.matrix[:]):
        raise OperationIntegrityError('matrix', post_op_image.name)


def validate_enh_matrix_integrity(pre_op_imcopy: Image, post_op_image: Image) -> None:
    if not np.array_equal(pre_op_imcopy.enh_matrix[:], post_op_image.enh_matrix[:]):
        raise OperationIntegrityError('enh_matrix', post_op_image.name)


def validate_objmask_integrity(pre_op_imcopy: Image, post_op_image: Image) -> None:
    if not np.array_equal(pre_op_imcopy.objmask[:], post_op_image.objmask[:]):
        raise OperationIntegrityError('objmask', post_op_image.name)
