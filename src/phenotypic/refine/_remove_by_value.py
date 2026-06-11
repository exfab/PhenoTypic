from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from pydantic import field_validator, model_validator
from typing_extensions import Self

from ..abc_ import ObjectRefiner
from phenotypic.schema import OBJECT


class RemoveByValue(ObjectRefiner):
    """Remove objects whose measured feature value falls outside a ``[min, max]`` band.

    A generic, measurer-driven cleanup step: name any
    :class:`~phenotypic.abc_.MeasureFeatures` subclass, pick one of the values
    it produces, and discard every object whose value is below ``min_value`` or
    above ``max_value``. This subsumes the purpose-built filters
    (:class:`SmallObjectRemover` for area, :class:`RemoveLowCircularity` for
    shape) into a single configurable operation that works for *any* feature the
    measurement layer exposes.

    The named measurer is instantiated, run against the image, and the resulting
    per-object table is filtered: objects inside the inclusive band survive,
    objects outside it (or with a ``NaN`` value) are zeroed from ``objmap``.

    For an overview of refinement strategies, see
    :doc:`/explanation/refinement_strategies`.

    Best For:
        - Pruning by a phenotype that has no dedicated refiner (intensity,
          colour, texture, eccentricity, …) without writing a new operation.
        - Two-sided gating, e.g. keep colonies whose area sits within an
          expected window and drop both specks and merged blobs in one step.
        - Reusing the exact measurement definitions the pipeline reports, so the
          filter and the measurement column stay in lock-step.

    Consider Also:
        - :class:`SmallObjectRemover` / :class:`RemoveLowCircularity` for the
          common size/shape cases where a self-contained, single-parameter
          operation is clearer.

    Args:
        feature: Name of a :class:`~phenotypic.abc_.MeasureFeatures` subclass to
            run, e.g. ``"MeasureSize"``. Resolved from the public ``phenotypic``
            namespace. ``None`` (the default) makes the operation a no-op.
        value: Name of the measured value to filter on. Accepts either the
            category-prefixed column emitted by the measurer (``"Size_Area"``) or
            the bare label (``"Area"``); the bare form is resolved against this
            measurer's columns only, so there is no cross-feature ambiguity.
            ``None`` (the default) makes the operation a no-op.
        min_value: Inclusive lower bound. Objects with ``value < min_value`` are
            removed. ``None`` (the default) leaves the lower side unbounded.
        max_value: Inclusive upper bound. Objects with ``value > max_value`` are
            removed. ``None`` (the default) leaves the upper side unbounded.
        measure_kwargs: Optional keyword arguments forwarded to the measurer's
            constructor (e.g. ``{"include_XYZ": False}`` for ``MeasureColor``).
            Carried as a dict because operations are keyword-only pydantic models
            that forbid loose extra fields. ``None`` (the default) constructs the
            measurer with its own defaults.

    Returns:
        Image: Input image with ``objmap`` and ``objmask`` updated to exclude
        objects whose ``value`` is outside ``[min_value, max_value]``. ``rgb``,
        ``gray``, and ``detect_mat`` are unchanged. When ``feature`` or ``value``
        is ``None``, or both bounds are ``None``, the image is returned unchanged.

    Raises:
        ValueError: If ``feature`` does not name a ``MeasureFeatures`` subclass,
            if ``value`` matches no (or ambiguously several) measured columns, or
            if ``min_value > max_value``.

    See Also:
        :doc:`/explanation/refinement_strategies` for choosing the right
        refinement sequence.

    Examples:
        Keep only colonies whose area is at least 50 px, dropping smaller debris:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.refine import RemoveByValue
        >>> plate = load_synth_yeast_plate()
        >>> before = plate.num_objects
        >>> remover = RemoveByValue(feature="MeasureSize", value="Size_Area", min_value=50)
        >>> result = remover.apply(plate)
        >>> result.num_objects <= before
        True

        Constructed with defaults it is an inert no-op, so it is always safe to
        place in a pipeline before its parameters are configured:

        >>> RemoveByValue().apply(load_synth_yeast_plate()).num_objects > 0
        True
    """

    feature: str | None = None
    value: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    measure_kwargs: dict[str, Any] | None = None

    @field_validator("feature")
    @classmethod
    def _validate_feature(cls, feature: str | None) -> str | None:
        """Require ``feature`` to name a concrete ``MeasureFeatures`` subclass."""
        if feature is None:
            return None
        if cls._resolve_measurer(feature) is None:
            raise ValueError(
                    f"'{feature}' does not name a MeasureFeatures subclass in the "
                    f"phenotypic namespace. Ensure it is exported from its "
                    f"submodule's __init__.py."
            )
        return feature

    @model_validator(mode="after")
    def _validate_bounds(self) -> Self:
        """Reject an inverted ``[min_value, max_value]`` band."""
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError(
                    f"min_value ({self.min_value}) must not exceed "
                    f"max_value ({self.max_value})."
            )
        return self

    @staticmethod
    def _resolve_measurer(feature: str) -> type | None:
        """Resolve ``feature`` to a concrete ``MeasureFeatures`` subclass, or None."""
        from phenotypic.abc_ import MeasureFeatures
        from phenotypic._core._pipeline_parts._serializable_pipeline import (
            SerializablePipeline,
        )

        op_class = SerializablePipeline._find_class_in_phenotypic(feature)
        if (
            isinstance(op_class, type)
            and issubclass(op_class, MeasureFeatures)
        ):
            return op_class
        return None

    def _resolve_column(self, columns: list[str]) -> str:
        """Match ``self.value`` to one measured column (exact, then bare-label)."""
        candidates = [col for col in columns if col != OBJECT.LABEL]
        if self.value in candidates:
            return self.value  # type: ignore[return-value]

        suffix_matches = [
            col for col in candidates if col.split("_", 1)[-1] == self.value
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        if len(suffix_matches) > 1:
            raise ValueError(
                    f"'{self.value}' ambiguously matches columns {suffix_matches}; "
                    f"pass the fully-prefixed name."
            )
        raise ValueError(
                f"'{self.value}' is not a measured value of {self.feature}. "
                f"Available: {candidates}."
        )

    def _operate(self, image: Image) -> Image:
        if (
            self.feature is None
            or self.value is None
            or (self.min_value is None and self.max_value is None)
        ):
            return image

        measurer_cls = self._resolve_measurer(self.feature)
        if measurer_cls is None:  # pragma: no cover - guarded by the validator
            raise ValueError(f"'{self.feature}' is not a MeasureFeatures subclass.")
        measurer = measurer_cls(**(self.measure_kwargs or {}))

        table = measurer.measure(image)
        column = self._resolve_column(list(table.columns))

        values = table[column].to_numpy()
        labels = table[OBJECT.LABEL].to_numpy()
        keep = np.ones(len(values), dtype=bool)
        if self.min_value is not None:
            keep &= values >= self.min_value
        if self.max_value is not None:
            keep &= values <= self.max_value

        failing_labels = labels[~keep]
        if failing_labels.size:
            image.objmap[np.isin(image.objmap[:], failing_labels)] = 0
        return image
