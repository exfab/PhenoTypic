from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
import time
import inspect
import mmh3
from functools import wraps

from phenotypic.sdk_.exceptions_ import OperationIntegrityError
import phenotypic.settings as settings

# this is a dummy variable so annotation's in ImageOperation, MeasureFeatures classes don't cause integrity check to throw an exception
Image: Any

_ABSENT_ARRAY = object()


def is_binary_mask(arr: np.ndarray):
    return (
        True
        if (arr.ndim == 2 or arr.ndim == 3) and np.all((arr == 0) | (arr == 1))
        else False
    )


def timed_execution(func):
    """
    Decorator to measure and print the execution time of a function.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Execute the wrapped function
        end_time = time.time()  # Record the end time
        print(
                f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds"
        )
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

    The rgb is converted to a C‑contiguous view so that ``memoryview`` can
    safely expose its buffer.  If the rgb is already contiguous this is a
    zero‑copy operation.
    """
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return mmh3.mmh3_x64_128_digest(memoryview(arr))


def _resolve_integrity_target(root: Any, target: str, func_name: str) -> np.ndarray | object:
    """Resolve an integrity target to an array or an intentionally absent RGB marker."""
    obj = root
    for attr in target.split(".")[1:]:
        value = getattr(obj, attr)
        if attr == "rgb" and hasattr(value, "isempty") and value.isempty():
            return _ABSENT_ARRAY
        obj = value[:]
    if not isinstance(obj, np.ndarray):
        raise RuntimeError(
                f"{func_name}: '{target}' is not a NumPy array",
        )
    return obj


def _integrity_signature(value: np.ndarray | object) -> bytes | object:
    """Hash an integrity target while preserving intentionally absent RGB state."""
    if value is _ABSENT_ARRAY:
        return _ABSENT_ARRAY
    return murmur3_array_signature(value)


def _integrity_changed(old_signature: bytes | object, value: np.ndarray | object) -> bool:
    """Return True when an integrity target changed, including absent/present flips."""
    if old_signature is _ABSENT_ARRAY:
        return value is not _ABSENT_ARRAY
    if value is _ABSENT_ARRAY:
        return True
    return murmur3_array_signature(value) != old_signature


def validate_operation_integrity(*targets: str):
    """
    Decorator to ensure that key NumPy arrays on the 'image' argument
    remain unchanged by an ImageOperation.apply() call.
    If no targets are specified, defaults to checking:

        image.rgb, image.gray, image.detect_mat, image.objmap

    Example Usage:

        @validate_member_integrity('image.rgb', 'image.objmap')
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
            if "image" not in sig.parameters:
                raise AttributeError(
                        f"{func.__name__}: no 'image' parameter and no targets given",
                )
            # Default attributes to check on the image object
            eff_targets = ["image.rgb", "image.gray", "image.detect_mat", "image.objmap"]

        # Helper function to retrieve an integrity target from an object by attribute path
        def _get_target(bound_args, target: str) -> np.ndarray | object:
            # Split the target path (e.g., 'image.rgb' -> ['image', 'rgb'])
            parts = target.split(".")
            # Get the root object from function arguments
            obj: Image = bound_args.arguments.get(parts[0])
            if obj is None:
                raise AttributeError(
                        f"{func.__name__}: parameter '{parts[0]}' not found",
                )

            return _resolve_integrity_target(obj, target, func.__name__)

        # The actual wrapper function that will replace the decorated function
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Step 3: Bind the provided arguments to the function signature
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            # Step 4: Calculate hash values for all target arrays before function execution
            # This creates a dictionary mapping each target to its hash value
            if settings.VALIDATE_OPS:
                pre_hashes = {
                    tgt: _integrity_signature(_get_target(bound, tgt))
                    for tgt in eff_targets
                }

            # Step 5: Execute the original function
            result = func(*args, **kwargs)

            # Step 6: Verify integrity by comparing hash values after function execution
            # For each target, calculate a new hash and compare with the original
            if settings.VALIDATE_OPS:
                for tgt, old_hash in pre_hashes.items():
                    obj = result
                    value = _resolve_integrity_target(obj, tgt, func.__name__)
                    if _integrity_changed(old_hash, value):
                        raise OperationIntegrityError(
                                opname=f"{func.__name__}",
                                component=f"{tgt}",
                        )

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
        @validate_member_integrity('image.rgb')
    Otherwise it defaults to checking:
        image.rgb, image.gray, image.detect_mat, image.objmap
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
            if "image" not in sig.parameters:
                raise OperationIntegrityError(
                        f"{func.__name__}: no 'image' parameter and no targets given",
                )
            eff_targets = ["image.rgb", "image.gray", "image.detect_mat", "image.objmap"]

        def _get_target(bound_args, target: str) -> np.ndarray | object:
            # e.g. target = 'image.rgb'
            obj = bound_args.arguments.get(target.split(".")[0])
            if obj is None:
                raise OperationIntegrityError(
                        f"{func.__name__}: cannot find parameter '{target.split('.')[0]}'",
                )
            return _resolve_integrity_target(obj, target, func.__name__)

        @wraps(func)
        def wrapper(*args, **kwargs):
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            # hash each target before the call
            if settings.VALIDATE_OPS:
                pre_hashes = {
                    tgt: _integrity_signature(_get_target(bound, tgt))
                    for tgt in eff_targets
                }

            # execute the original method
            result = func(*args, **kwargs)

            # re-hash and compare
            if settings.VALIDATE_OPS:
                for tgt, old in pre_hashes.items():
                    value = _get_target(bound, tgt)
                    if _integrity_changed(old, value):
                        raise OperationIntegrityError(
                                opname=f"{func.__name__}", component=f"{tgt}"
                        )

            return result

        # preserve metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__signature__ = sig
        return wrapper

    return decorator


def normalize_rgb_bitdepth(image: np.ndarray) -> np.ndarray:
    """
    Normalize an RGB rgb to [0,1] using bit-depth inference.

    Rules:

    - If dtype is integer: use dtype max (e.g. 255 for uint8, 65535 for uint16).
    - If dtype is float:

        * If max <= 1 → assume already normalized, return as-is.
        * If 255 < max <= 65535 → assume 16-bit, divide by 65535.
        * Else → assume 8-bit, divide by 255.

    Args:
        image: RGB rgb.

    Returns:
        np.ndarray: float64 normalized image in [0,1].
    """
    img = np.asarray(image.copy())

    if np.issubdtype(img.dtype, np.integer):
        max_val = np.iinfo(img.dtype).max
        return img.astype(np.float64) / max_val

    elif np.issubdtype(img.dtype, np.floating):
        tol = 1e-7
        m = img.max()
        if m <= 1.0 + tol:
            return img.astype(np.float32, copy=False)
        elif 1 < m <= (255 + tol):
            return (img.astype(np.float32) / 255.0).clip(0, 1)
        elif 255 < m <= (65535.0 + tol):
            return (img.astype(np.float32) / 65535.0).clip(0, 1)
        else:
            raise ValueError(
                    f"Invalid range: min={img.min():.02f} max={img.max():.02f}"
            )
    else:
        raise TypeError(f"Unsupported dtype: {img.dtype}")
