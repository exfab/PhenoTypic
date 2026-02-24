"""Mixin for controlling output clipping behavior in composite operations."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from phenotypic.abc_ import ImageEnhancer
    from phenotypic import ImagePipeline


class ClipControlMixin:
    """Mixin for operations that need to control clipping behavior of inner operations.

    Provides a method to create copies of ImageEnhancer or ImagePipeline instances
    with clipping disabled. This is useful for composite operations where an inner
    enhancer operates on non-normalized data (e.g., variance-stabilized values from
    the Generalized Anscombe Transform, typically in the range ~1-32).

    The mixin uses duck typing to check for a `clip` attribute on operations. If an
    operation has `clip=True`, the `_disable_clipping` method will create a shallow
    copy with `clip=False`. This preserves the original operation unchanged while
    allowing the copy to operate without output clipping.

    Example:
        Creating a clip-disabled copy of an enhancer:

        >>> from phenotypic.tools_ import ClipControlMixin
        >>> from phenotypic.enhance import BilateralDenoise
        >>>
        >>> enh = BilateralDenoise(sigma_spatial=5, clip=True)
        >>> copied = ClipControlMixin._disable_clipping(enh)
        >>> # Original unchanged, copy has clip=False
        >>> enh.clip, copied.clip
        (True, False)

        Creating a clip-disabled copy of a pipeline:

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.enhance import GaussianBlur, BilateralDenoise
        >>>
        >>> pipeline = ImagePipeline([
        ...     GaussianBlur(sigma=1.0),
        ...     BilateralDenoise(sigma_spatial=5, clip=True)
        ... ])
        >>> copied_pipe = ClipControlMixin._disable_clipping(pipeline)
        >>> # Only BilateralDenoise has clip attribute, so only it is affected
        >>> # _ops is a dict with operation names as keys
        >>> list(copied_pipe._ops.values())[1].clip
        False
    """

    @staticmethod
    def _disable_clipping(
        operation: Union["ImageEnhancer", "ImagePipeline"]
    ) -> Union["ImageEnhancer", "ImagePipeline"]:
        """Create a copy of an operation with clipping disabled.

        Creates a shallow copy of the operation (or pipeline) with `clip=False`
        on all enhancers that support the `clip` parameter. This is useful for
        composite operations that need to process data in a non-normalized domain
        (e.g., variance-stabilized data from the Generalized Anscombe Transform).

        Args:
            operation: An ImageEnhancer or ImagePipeline instance. If the operation
                has a `clip` attribute, a copy with `clip=False` is returned.
                If it's a pipeline, all operations in the pipeline are processed
                recursively.

        Returns:
            A copy of the operation with `clip=False` on all enhancers that support
            the `clip` parameter. Operations without a `clip` attribute are returned
            unchanged (not copied).

        Note:
            This method uses shallow copying (`copy.copy`), so modifications to
            mutable attributes of the copied operation may affect the original.
            However, since we only modify the `clip` attribute (a bool), this is
            safe in practice.

        Example:
            >>> from phenotypic.tools_ import ClipControlMixin
            >>> from phenotypic.enhance import BilateralDenoise
            >>>
            >>> enh = BilateralDenoise(sigma_spatial=5, clip=True)
            >>> copied = ClipControlMixin._disable_clipping(enh)
            >>> assert enh.clip == True  # Original unchanged
            >>> assert copied.clip == False  # Copy has clipping disabled
        """
        # Handle ImagePipeline (check for _ops attribute - it's a Dict[str, ImageOperation])
        if hasattr(operation, "_ops"):
            copied = copy.copy(operation)
            # Recursively disable clipping on all operations in the pipeline
            # _ops is a dictionary with operation names as keys
            copied._ops = {
                key: ClipControlMixin._disable_clipping(op)
                for key, op in operation._ops.items()
            }
            return copied

        # Handle ImageEnhancer with clip parameter
        if hasattr(operation, "clip"):
            copied = copy.copy(operation)
            copied.clip = False
            return copied

        # Return original if no clip parameter (operation doesn't support clipping)
        return operation
