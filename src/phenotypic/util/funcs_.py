from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING: from phenotypic import Image

# this is a dummy variable so annotation's in ImageOperation, MeasureFeatures classes don't cause integrity check to throw an exception
Image: Any

import numpy as np
import time
import inspect
import mmh3
from functools import wraps

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


def murmur3_array_signature(arr: np.ndarray) -> bytes:
    """
    Return a 128‑bit MurmurHash3 digest of *arr*.

    The array is converted to a C‑contiguous view so that ``memoryview`` can
    safely expose its buffer.  If the array is already contiguous this is a
    zero‑copy operation.
    """
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return mmh3.mmh3_x64_128_digest(memoryview(arr))


def validate_operation_integrity(*targets: str):
    """
    Decorator to ensure that key NumPy arrays on the 'image' argument
    remain unchanged by an ImageOperation.apply() call.
    If no targets are specified, defaults to checking:
        image.array, image.matrix, image.enh_matrix, image.objmap

    Example Usage:
        @validate_member_integrity('image.array', 'image.objmap')
        def func(image: Image,...):
            ...
    """

    def decorator(func):
        sig = inspect.signature(func)
        # wipe out all annotations in the signature since Image is missing, but importing it causes a circular import
        params = [p.replace(annotation=inspect._empty) for p in sig.parameters.values()]
        sig = sig.replace(parameters=params, return_annotation=inspect._empty)

        # pick which attributes to check
        if targets:
            eff_targets = list(targets)
        else:
            if 'image' not in sig.parameters:
                raise OperationIntegrityError(
                    f"{func.__name__}: no 'image' parameter and no targets given",
                )
            eff_targets = [
                'image.array',
                'image.matrix',
                'image.enh_matrix',
                'image.objmap'
            ]

        def _get_array(bound_args, target: str) -> np.ndarray:
            parts = target.split('.')
            obj = bound_args.arguments.get(parts[0])
            if obj is None:
                raise OperationIntegrityError(
                    f"{func.__name__}: parameter '{parts[0]}' not found",
                )
            for attr in parts[1:]:
                obj = getattr(obj, attr)[:]
            if not isinstance(obj, np.ndarray):
                raise OperationIntegrityError(
                    f"{func.__name__}: '{target}' is not a NumPy array",
                )
            return obj

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            # hash originals (from the passed-in image)
            pre_hashes = {tgt: murmur3_array_signature(_get_array(bound, tgt))
                for tgt in eff_targets
            }

            # call the method, get the returned Image instance
            result = func(*args, **kwargs)

            # now re-hash those same attributes **on the returned Image**
            for tgt, old_hash in pre_hashes.items():
                parts = tgt.split('.')
                obj = result  # start with the returned object
                # walk the same attribute chain
                for attr in parts[1:]:
                    obj = getattr(obj, attr)[:]
                if not isinstance(obj, np.ndarray):
                    raise OperationIntegrityError(
                        f"{func.__name__}: '{tgt}' is not a NumPy array on result",
                    )
                new_hash = murmur3_array_signature(obj)
                if new_hash != old_hash:
                    raise OperationIntegrityError(opname=f'{func.__name__}', component=f'{tgt}', )

            return result

        # preserve metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__signature__ = sig
        return wrapper

    return decorator


def validate_measure_integrity(*targets: str):
    """
    Decorator to ensure that key NumPy arrays on the 'image' argument
    are not mutated by an MeasureFeatures.measure() call.

    If you pass explicit targets, it will honor those—for example:
        @validate_member_integrity('image.array')
    Otherwise it defaults to checking:
        image.array, image.matrix, image.enh_matrix, image.objmap
    """

    def decorator(func):
        sig = inspect.signature(func)
        # wipe out all annotations in the signature
        params = [p.replace(annotation=inspect._empty) for p in sig.parameters.values()]
        sig = sig.replace(parameters=params, return_annotation=inspect._empty)

        # determine which attributes to check
        if targets:
            eff_targets = list(targets)
        else:
            # apply only to methods with an 'image' parameter
            if 'image' not in sig.parameters:
                raise OperationIntegrityError(
                    f"{func.__name__}: no 'image' parameter and no targets given",
                )
            eff_targets = [
                'image.array',
                'image.matrix',
                'image.enh_matrix',
                'image.objmap'
            ]

        def _get_array(bound_args, target: str) -> np.ndarray:
            # e.g. target = 'image.array'
            obj = bound_args.arguments.get(target.split('.')[0])
            if obj is None:
                raise OperationIntegrityError(
                    f"{func.__name__}: cannot find parameter '{target.split('.')[0]}'",
                )
            for attr in target.split('.')[1:]:
                obj = getattr(obj, attr)[:]
            if not isinstance(obj, np.ndarray):
                raise OperationIntegrityError(
                    f"{func.__name__}: '{target}' is not a NumPy array")
            return obj

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            # hash each target before the call
            pre_hashes = {tgt: murmur3_array_signature(_get_array(bound, tgt))
                for tgt in eff_targets
            }

            # execute the original method
            result = func(*args, **kwargs)

            # re-hash and compare
            for tgt, old in pre_hashes.items():
                new = murmur3_array_signature(_get_array(bound, tgt))
                if new != old:
                    raise OperationIntegrityError(opname=f'{func.__name__}', component=f'{tgt}')

            return result

        # preserve metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__signature__ = sig
        return wrapper

    return decorator
