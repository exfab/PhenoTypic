"""Shared threshold method registry for detector implementations."""
from __future__ import annotations

from typing import ClassVar, Collection, Literal

import numpy as np
from skimage import filters

ThresholdMethodName = Literal[
    "otsu",
    "mean",
    "local",
    "triangle",
    "minimum",
    "isodata",
    "li",
    "yen",
]


class ThresholdingRegistry:
    """Private registry for threshold method validation and dispatch."""

    METHOD_MAP: ClassVar[dict[str, object]] = {
        "otsu": filters.threshold_otsu,
        "mean": filters.threshold_mean,
        "local": filters.threshold_local,
        "triangle": filters.threshold_triangle,
        "minimum": filters.threshold_minimum,
        "isodata": filters.threshold_isodata,
        "li": filters.threshold_li,
        "yen": filters.threshold_yen,
    }
    GRID_METHODS: ClassVar[frozenset[str]] = frozenset(
        {"otsu", "mean", "local", "triangle", "minimum", "isodata", "li"}
    )
    SCALAR_METHODS: ClassVar[frozenset[str]] = frozenset(
        {"otsu", "mean", "triangle", "minimum", "isodata", "li", "yen"}
    )
    NBINS_METHODS: ClassVar[frozenset[str]] = frozenset(
        {"otsu", "isodata", "minimum", "triangle", "yen"}
    )

    @classmethod
    def validate_method(
        cls,
        method: str,
        *,
        allowed_methods: Collection[str] | None = None,
    ) -> str:
        """Normalize and validate a threshold method name.

        Args:
            method: Method name to validate.
            allowed_methods: Optional subset allowed by a detector path.

        Returns:
            Lowercase method name.

        Raises:
            ValueError: If the method is unknown or disallowed.
        """
        method_name = method.lower()
        allowed = set(allowed_methods or cls.METHOD_MAP.keys())
        if method_name not in cls.METHOD_MAP or method_name not in allowed:
            valid = sorted(allowed)
            raise ValueError(
                f"Unknown threshold method '{method}'. Valid methods: {valid}"
            )
        return method_name

    @classmethod
    def threshold_value(
        cls,
        threshold_spec: str | int | float,
        data: np.ndarray,
        *,
        bit_depth: int | None = None,
        allowed_methods: Collection[str] | None = None,
    ) -> float:
        """Compute a scalar threshold from a manual value or method name."""
        if isinstance(threshold_spec, (int, float)):
            return float(threshold_spec)

        if threshold_spec.lower() == "local":
            cls.validate_method(threshold_spec)
            raise ValueError(
                "Threshold method 'local' does not produce a scalar threshold"
            )

        method_name = cls.validate_method(
            threshold_spec,
            allowed_methods=allowed_methods or cls.SCALAR_METHODS,
        )

        threshold_func = cls.METHOD_MAP[method_name]
        if method_name in cls.NBINS_METHODS and bit_depth is not None:
            return float(threshold_func(data, nbins=2 ** int(bit_depth)))  # type: ignore[operator]
        return float(threshold_func(data))  # type: ignore[operator]

    @classmethod
    def threshold_mask(
        cls,
        matrix: np.ndarray,
        *,
        method: str,
        bit_depth: int | None = None,
        local_block_size: int | None = None,
        allowed_methods: Collection[str] | None = None,
        inclusive: bool = True,
    ) -> np.ndarray:
        """Compute a boolean mask from a threshold method."""
        method_name = cls.validate_method(
            method,
            allowed_methods=allowed_methods or cls.GRID_METHODS,
        )
        if method_name == "local":
            block_size = max(int(local_block_size or 3), 3)
            threshold = filters.threshold_local(matrix, block_size=block_size)
        else:
            threshold = cls.threshold_value(
                method_name,
                matrix,
                bit_depth=bit_depth,
                allowed_methods=allowed_methods or cls.GRID_METHODS,
            )

        return matrix >= threshold if inclusive else matrix > threshold
