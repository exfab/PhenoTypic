from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Literal

import numpy as np
from pydantic import Field, field_validator

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from ..abc_ import ImageEnhancer
from ..sdk_.typing_ import OperationField
from ._gaussian_blur import GaussianBlur
from ._median_filter import MedianFilter

CombineMode = Literal["max", "mean", "min", "median"]


class CompositeEnhance(ImageEnhancer):
    """Enhance ``detect_mat`` by combining several enhancers' response maps pixel-wise.

    Apply two or more enhancers (or preprocessing pipelines ending in an
    enhancer) to the same plate image and reduce their ``detect_mat`` outputs
    into a single response map. This ensembles complementary preprocessing
    strategies: where one enhancer responds strongly to faint colonies and
    another to sharp edges, ``'max'`` keeps the strongest signal from either.
    It is the enhancer-side analogue of :class:`CompositeDetector`, which
    combines binary masks rather than continuous response maps.

    Each branch reads the *same* input ``detect_mat`` independently (branches
    do not chain into one another), so the order of ``enhancers`` does not
    affect the result for the commutative reductions used here.

    Best For:
        - Fusing complementary focus maps (e.g. an edge response and a blob
          response) so colonies that only one branch emphasises survive.
        - Averaging several denoisers (``'mean'`` / ``'median'``) to suppress
          branch-specific artefacts before thresholding.
        - Consensus preprocessing (``'min'``), where a pixel stays bright only
          if every branch agrees it is signal.

    Consider Also:
        - A single :class:`ImageEnhancer` when one preprocessing step already
          isolates colonies reliably.
        - :class:`CompositeDetector` when you want to combine *detections*
          (binary masks) rather than continuous enhancement maps.
        - An :class:`ImagePipeline` of enhancers when the steps should be
          applied *in sequence* rather than combined in parallel.

    Args:
        ops: List of :class:`~phenotypic.abc_.ImageEnhancer` or
            :class:`~phenotypic.ImagePipeline` instances to combine. Pipelines
            allow per-branch preprocessing before the response is collected.
            Defaults to ``[GaussianBlur(), MedianFilter()]`` when not
            specified. ``None`` entries mark unfilled GUI-builder slots and
            are skipped.
        mode: Pixel-wise reduction across branch response maps. ``'max'``
            keeps the brightest response (union of complementary signals,
            the default). ``'min'`` keeps the darkest (consensus suppression).
            ``'mean'`` averages all branches. ``'median'`` takes the per-pixel
            median (robust to a single outlier branch; most useful with three
            or more enhancers). Default: ``'max'``.
        clip: When ``True``, clamp the combined result into ``[0.0, 1.0]``
            after reduction. Leave ``False`` (default) when ``detect_mat`` is
            not unit-normalised, since clipping an arbitrarily-scaled map would
            distort it.

    Returns:
        Image: Input image with ``detect_mat`` set to the combined response
        map. ``rgb`` and ``gray`` are unchanged.

    Raises:
        ValueError: If ``ops`` is empty or contains only ``None`` slots.

    Examples:
        Combine two enhancers, keeping the strongest response per pixel:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.enhance import (
        ...     CompositeEnhance,
        ...     GaussianBlur,
        ...     SubtractRollingBall,
        ... )
        >>> image = load_synth_yeast_plate()
        >>> combiner = CompositeEnhance(
        ...     ops=[GaussianBlur(sigma=1.5), SubtractRollingBall()],
        ...     mode="max",
        ... )
        >>> enhanced = combiner.apply(image)
        >>> bool((image.gray[:] == enhanced.gray[:]).all())  # rgb/gray untouched
        True

        Average three branches and clamp the result to ``[0, 1]``:

        >>> from phenotypic.enhance import MedianFilter, EnhanceLocalContrast
        >>> combiner = CompositeEnhance(
        ...     ops=[GaussianBlur(), MedianFilter(), EnhanceLocalContrast()],
        ...     mode="mean",
        ...     clip=True,
        ... )
        >>> enhanced = combiner.apply(load_synth_yeast_plate())
        >>> float(enhanced.detect_mat[:].max()) <= 1.0
        True
    """

    # Each branch may be an ``ImageEnhancer`` or a nested ``ImagePipeline``.
    # ``OperationField`` keeps the concrete class of each entry across a JSON
    # round-trip (a bare ``model_dump`` of the union would lose the subclass).
    # ``| None`` permits an empty list slot that the GUI builder uses to mark
    # an unfilled enhancer slot in an in-progress pipeline.
    ops: List[OperationField | None] = Field(
        default_factory=lambda: [GaussianBlur(), MedianFilter()]
    )
    mode: CombineMode = "max"
    clip: bool = False

    @field_validator("ops", mode="before")
    @classmethod
    def _default_ops(cls, value: Any) -> Any:
        """Map an explicit ``None`` onto the default enhancer pair.

        Mirrors :class:`CompositeDetector`: the ``default_factory`` covers the
        omitted-argument case, and this validator preserves an explicit
        ``ops=None`` call by substituting the default pair.
        """
        if value is None:
            return [GaussianBlur(), MedianFilter()]
        return value

    def _operate(self, image: Image) -> Image:
        """Apply each branch and reduce their ``detect_mat`` maps pixel-wise."""

        # Import here to avoid a circular dependency at module load.
        from phenotypic import ImagePipeline

        response_maps: list[np.ndarray] = []
        for enhancer in self.ops:
            if enhancer is None:
                # An unfilled GUI-builder slot; nothing to apply -- skip it.
                continue
            if isinstance(enhancer, ImagePipeline):
                enhanced = enhancer.apply(image, inplace=False, reset=False)
            else:
                enhanced = enhancer.apply(image, inplace=False)
            response_maps.append(np.asarray(enhanced.detect_mat[:], dtype=float))

        if not response_maps:
            raise ValueError(
                "At least one enhancer must be provided to CompositeEnhance"
            )

        combined = self._combine(response_maps)

        if self.clip:
            combined = np.clip(combined, 0.0, 1.0)

        original = image.detect_mat[:]
        image.detect_mat[:] = combined.astype(original.dtype, copy=False)
        return image

    def _combine(self, response_maps: list[np.ndarray]) -> np.ndarray:
        """Reduce stacked branch response maps according to ``self.mode``."""
        if self.mode == "max":
            return np.maximum.reduce(response_maps)
        if self.mode == "min":
            return np.minimum.reduce(response_maps)
        stack = np.stack(response_maps, axis=0)
        if self.mode == "mean":
            return np.mean(stack, axis=0)
        # "median"
        return np.median(stack, axis=0)
