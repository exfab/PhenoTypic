from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from phenotypic._core._reference_context import ReferenceContextError
from phenotypic.abc_ import RefMetadata
from phenotypic.abc_._enhance_markers._background_subtraction import BackgroundSubtraction
from phenotypic.sdk_ import RefImageColumn

if TYPE_CHECKING:
    from phenotypic import Image


class StaleDetectMatError(ReferenceContextError):
    """SubtractBlank ran on a detect_mat or image that its raw blank does not match."""


def _corrector_class_names() -> set[str]:
    from phenotypic.abc_ import ImageCorrector

    seen: set[type] = set()
    stack: list[type] = [ImageCorrector]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    return {f"{c.__module__}.{c.__qualname__}" for c in seen}


class SubtractBlank(BackgroundSubtraction, RefMetadata):
    """Subtract a time series' media-blank frame from ``detect_mat``.

    Each image names its blank — typically the plate's frame 0, imaged before
    inoculation — in a metadata column. The blank is read raw, taken in the
    target's detection mode, and subtracted pixel by pixel, so the agar,
    lid glare and scanner vignetting shared by every frame of that plate
    cancel and only growth since frame 0 remains.

    The metadata table is supplied at run time, never stored on the
    operation: ``with phenotypic.ReferenceContext(table, image_root=...)`` in
    Python, ``--metadata`` on the CLI, or the GUI's reference-metadata picker.

    The blank is raw, so ``detect_mat`` must be raw too: place SubtractBlank
    before any enhancer, or directly after a ``SetDetectMode``, and after no
    ``ImageCorrector``. It refuses otherwise. A later ``SetDetectMode``
    discards the subtraction, like any enhancement.

    Best For:
        - Time-lapse plates with a media-only frame taken before growth.
        - Filamentous colonies whose faint mycelium is lost against uneven
          agar when the background is estimated from the image itself.

    Consider Also:
        - :class:`SubtractGaussian` or :class:`SubtractRollingBall` when no
          blank frame exists and the background must be estimated from the
          image alone.

    Args:
        blank_column: Metadata column holding each image's blank, as a file
            stem or file name in the same input directory. Default
            ``"Metadata_BlankImage"``.
        polarity: Which change from the blank counts as colony.
            ``"brighter"`` keeps pixels brighter than the blank (white
            mycelium on darker agar); ``"darker"`` keeps pixels darker than
            the blank and flips them bright (pigmented colonies);
            ``"both"`` keeps the absolute difference.

    Returns:
        Image: Input image with ``detect_mat`` replaced by the clipped
        difference in ``[0, 1]``. ``rgb`` and ``gray`` are unchanged.

    Raises:
        RefMetadataUnavailableError: No ReferenceContext is active.
        ReferenceLookupError: The image's blank is missing, empty, ambiguous,
            or the image itself.
        ReferenceImageError: The blank cannot be resolved or read, or its
            shape or bit depth differs from the target's.
        StaleDetectMatError: ``detect_mat`` was already enhanced, or an
            ``ImageCorrector`` appears anywhere in the image's recorded
            history.

        All of these are :class:`~phenotypic.sdk_.ReferenceContextError`
        subclasses and reach a bare ``op.apply`` caller unwrapped. Inside an
        ``ImagePipeline`` each enclosing pipeline (and a composite holding a
        branch pipeline) wraps them in a ``RuntimeError``, so walk
        ``__cause__`` until you reach a ``ReferenceContextError``.

    Examples:
        A frame identical to its blank cancels to zero:

        >>> import pandas as pd
        >>> from phenotypic import ReferenceContext
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.enhance import SubtractBlank
        >>> plate = load_synth_yeast_plate()
        >>> plate.name = "plate1_t04"
        >>> blank = load_synth_yeast_plate()   # stands in for the media-only frame
        >>> layout = pd.DataFrame({"Metadata_ImageName": ["plate1_t04"],
        ...                        "Metadata_BlankImage": ["plate1_t00"]})
        >>> with ReferenceContext(layout, images={"plate1_t00": blank}):
        ...     out = SubtractBlank().apply(plate)
        >>> float(out.detect_mat[:].max())
        0.0
    """

    blank_column: RefImageColumn = "Metadata_BlankImage"
    polarity: Literal["brighter", "darker", "both"] = "brighter"

    def _operate(self, image: "Image") -> "Image":
        from phenotypic._core._image_parts.detection_modes import get_detection_mode
        from phenotypic._core._reference_context import (
            ReferenceImageError,
            ReferenceLookupError,
        )

        from phenotypic.sdk_._io_constants import source_image_stem

        name = self._ref_values(image)[self.blank_column]
        # Compare by the resolved file too: a blank written with its extension
        # ("t04.tif" for image "t04") resolves to the image's own file, and
        # subtracting a frame from itself would zero it with no error.
        target_file = self._require_context().resolve_image(name)
        if name == image.name or (
            isinstance(target_file, Path) and source_image_stem(target_file) == image.name
        ):
            raise ReferenceLookupError(
                f"Image {image.name!r} names itself as its blank in {self.blank_column}; "
                f"leave blank frames out of the input",
                reason="self",
                image_name=image.name,
                column=self.blank_column,
            )
        mode = get_detection_mode(image.detect_mode)
        self._require_raw_target(image, mode)
        blank = self._ref_image(name)
        if blank.gray.shape != image.gray.shape:
            raise ReferenceImageError(
                f"Blank {name!r} has shape {blank.gray.shape}, image {image.name!r} "
                f"has {image.gray.shape}; frames must be pixel-aligned"
            )
        if blank.bit_depth != image.bit_depth:
            raise ReferenceImageError(
                f"Blank {name!r} is {blank.bit_depth}-bit, image {image.name!r} "
                f"is {image.bit_depth}-bit"
            )
        background = mode.compute(blank)
        target = image.detect_mat[:]
        if self.polarity == "brighter":
            result = np.clip(target - background, 0.0, 1.0)
        elif self.polarity == "darker":
            result = np.clip(background - target, 0.0, 1.0)
        else:
            result = np.abs(target - background)
        image.detect_mat[:] = result.astype(target.dtype, copy=False)
        return image

    # Limitation: the records of a corrector that ran inside the *same*
    # composite branch are appended only when the composite finishes, so that
    # case is caught by neither check unless the corrector also changed
    # detect_mat. Correctors do not live inside composites today.
    @staticmethod
    def _require_raw_target(image: "Image", mode) -> None:
        # Every application, not only the last: a staged run's Stage-2 probe
        # copy opens a fresh application, and a corrector recorded by Stage 1
        # must still be seen there (else GPU time is spent, then Stage 3 refuses).
        # _operations flattens both journal schemas (v1 top-level "operations",
        # v2 per-application). Correctors are matched against the ImageCorrector
        # subclasses imported in this process -- every class the running
        # pipeline uses is imported by deserializing it.
        from phenotypic._core._provenance import _operations

        correctors = _corrector_class_names()
        for record in _operations(image._metadata.provenance_journal):
            if record.get("operation_class") in correctors:
                raise StaleDetectMatError(
                    f"SubtractBlank follows {record.get('operation_name')}, an "
                    f"ImageCorrector; the raw blank does not share its correction. "
                    f"Place SubtractBlank before any corrector."
                )
        if not np.array_equal(image.detect_mat[:], mode.compute(image)):
            raise StaleDetectMatError(
                "detect_mat was already enhanced; the raw blank cannot be subtracted "
                "from it. Place SubtractBlank before any enhancer, or directly after "
                "SetDetectMode."
            )
