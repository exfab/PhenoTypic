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
        # Step 1: Get the function signature to analyze parameters
        sig = inspect.signature(func)
        # Remove all annotations in the signature to avoid circular import issues with Image class
        params = [p.replace(annotation=inspect._empty) for p in sig.parameters.values()]
        sig = sig.replace(parameters=params, return_annotation=inspect._empty)

        # Step 2: Determine which attributes to check for integrity
        # If targets are explicitly provided, use those
        if targets:
            eff_targets = list(targets)
        else:
            # Otherwise use default targets, but ensure 'image' parameter exists
            if 'image' not in sig.parameters:
                raise OperationIntegrityError(
                    f"{func.__name__}: no 'image' parameter and no targets given",
                )
            # Default attributes to check on the image object
            eff_targets = [
                'image.array',
                'image.matrix',
                'image.enh_matrix',
                'image.objmap'
            ]

        # Helper function to retrieve a NumPy array from an object by attribute path
        def _get_array(bound_args, target: str) -> np.ndarray:
            # Split the target path (e.g., 'image.array' -> ['image', 'array'])
            parts = target.split('.')
            # Get the root object from function arguments
            obj = bound_args.arguments.get(parts[0])
            if obj is None:
                raise OperationIntegrityError(
                    f"{func.__name__}: parameter '{parts[0]}' not found",
                )

            # Navigate through the attribute chain to get the final array
            for attr in parts[1:]:
                obj = getattr(obj, attr)[:]  # Use [:] to get a view of the array
            # Ensure the result is a NumPy array
            if not isinstance(obj, np.ndarray):
                raise OperationIntegrityError(
                    f"{func.__name__}: '{target}' is not a NumPy array",
                )
            return obj

        # The actual wrapper function that will replace the decorated function
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Step 3: Bind the provided arguments to the function signature
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            # Step 4: Calculate hash values for all target arrays before function execution
            # This creates a dictionary mapping each target to its hash value
            pre_hashes = {tgt: murmur3_array_signature(_get_array(bound, tgt))
                for tgt in eff_targets
            }

            # Step 5: Execute the original function
            result = func(*args, **kwargs)

            # Step 6: Verify integrity by comparing hash values after function execution
            # For each target, calculate a new hash and compare with the original
            for tgt, old_hash in pre_hashes.items():
                parts = tgt.split('.')
                # Start with the result object returned by the function
                obj = result
                # Navigate through the attribute chain on the result object
                for attr in parts[1:]:
                    obj = getattr(obj, attr)[:]
                # Ensure the attribute is still a NumPy array
                if not isinstance(obj, np.ndarray):
                    raise OperationIntegrityError(
                        f"{func.__name__}: '{tgt}' is not a NumPy array on result",
                    )
                # Calculate new hash and compare with original
                new_hash = murmur3_array_signature(obj)
                # If hashes don't match, the array was modified - raise an error
                if new_hash != old_hash:
                    raise OperationIntegrityError(opname=f'{func.__name__}', component=f'{tgt}', )

            # Step 7: Return the original function's result if integrity check passes
            return result

        # Step 8: Preserve the original function's metadata on the wrapper
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
