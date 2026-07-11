"""Mixin for disabling output normalization on nested operations."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from phenotypic.abc_ import ImageEnhancer
    from phenotypic._core._image_pipeline import ImagePipeline


class NormControlMixin:
    """Mixin for operations that need to disable normalization of inner operations.

    Provides a method to create copies of ImageEnhancer or ImagePipeline instances
    with output normalization disabled. This is useful for composite operations where
    an inner enhancer operates on non-normalized data (e.g., variance-stabilized
    values from the Generalized Anscombe Transform, typically in the range ~1-32),
    where clipping or rescaling to [0, 1] would destroy the inverse transform.

    The mixin uses duck typing to check for a `norm` attribute on operations. If an
    operation has one, the `_disable_normalization` method will create a shallow copy
    with `norm=None`. This preserves the original operation unchanged while allowing
    the copy to operate without output normalization. Operations that carry no `norm`
    field (e.g. `GaussianBlur`) are returned unchanged.

    Note:
        Renamed from ``ClipControlMixin`` in 0.18.0, when ``clip: bool`` became
        :data:`~phenotypic.sdk_.typing_.NormOut`. The old name is gone.

    Example:
        Creating a normalization-disabled copy of an enhancer:

        >>> from phenotypic.abc_ import ImageEnhancer
        >>> from phenotypic.sdk_ import NormalizedOutputMixin, NormControlMixin
        >>>
        >>> class Denoise(NormalizedOutputMixin, ImageEnhancer):
        ...     '''Denoise a colony plate.
        ...
        ...     Args:
        ...         sigma: Smoothing width in pixels.
        ...         norm: Output normalization policy.
        ...     '''
        ...
        ...     sigma: float = 1.0
        ...
        ...     def _operate(self, image):
        ...         return image
        >>>
        >>> enh = Denoise(sigma=5.0, norm="clip")
        >>> copied = NormControlMixin._disable_normalization(enh)
        >>> # Original unchanged, copy has norm=None
        >>> enh.norm, copied.norm
        ('clip', None)

        Creating a normalization-disabled copy of a pipeline:

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.enhance import GaussianBlur
        >>>
        >>> pipeline = ImagePipeline(pipe_cfgs=[
        ...     GaussianBlur(sigma=1.0),
        ...     Denoise(sigma=5.0, norm="clip")
        ... ])
        >>> copied_pipe = NormControlMixin._disable_normalization(pipeline)
        >>> # Only Denoise has a norm attribute, so only it is affected
        >>> # _ops is a dict with operation names as keys
        >>> [getattr(op, "norm", "absent") for op in copied_pipe._ops.values()]
        ['absent', None]
    """

    @staticmethod
    def _disable_normalization(
        operation: Union["ImageEnhancer", "ImagePipeline"]
    ) -> Union["ImageEnhancer", "ImagePipeline"]:
        """Create a copy of an operation with output normalization disabled.

        Creates a shallow copy of the operation (or pipeline) with `norm=None`
        on all enhancers that support the `norm` parameter. This is useful for
        composite operations that need to process data in a non-normalized domain
        (e.g., variance-stabilized data from the Generalized Anscombe Transform).

        Args:
            operation: An ImageEnhancer or ImagePipeline instance. If the operation
                has a `norm` attribute, a copy with `norm=None` is returned.
                If it's a pipeline, all operations in the pipeline are processed
                recursively.

        Returns:
            A copy of the operation with `norm=None` on all enhancers that support
            the `norm` parameter. Operations without a `norm` attribute are returned
            unchanged (not copied).

        Note:
            This method uses shallow copying (`copy.copy`), so modifications to
            mutable attributes of the copied operation may affect the original.
            However, since we only modify the `norm` attribute (a str or None), this
            is safe in practice.

        Example:
            An operation carrying no `norm` field passes straight through, unchanged
            and uncopied:

            >>> from phenotypic.enhance import GaussianBlur
            >>> from phenotypic.sdk_ import NormControlMixin
            >>>
            >>> blur = GaussianBlur(sigma=1.0)
            >>> NormControlMixin._disable_normalization(blur) is blur
            True
        """
        # Handle ImagePipeline (check for _ops attribute - it's a Dict[str, ImageOperation])
        if hasattr(operation, "_ops"):
            copied = copy.copy(operation)
            # Recursively disable normalization on all operations in the pipeline
            # _ops is a dictionary with operation names as keys
            copied._ops = {
                key: NormControlMixin._disable_normalization(op)
                for key, op in operation._ops.items()
            }
            return copied

        # Handle ImageEnhancer with norm parameter
        if hasattr(operation, "norm"):
            copied = copy.copy(operation)
            copied.norm = None
            return copied

        # Return original if no norm parameter (operation does not normalize its output)
        return operation
