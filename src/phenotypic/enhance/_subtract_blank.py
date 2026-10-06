from __future__ import annotations

import importlib
import os
import sys
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


#: Slack for float rounding in a [0, 1] projection (rgb2gray of white, say).
_UNIT_RANGE_TOLERANCE = 1e-6


#: Classes already resolved from journal records. Misses are not cached: a
#: module the user imports later must then resolve.
_RECORDED_CLASSES: dict[str, type] = {}


def _trusted_module(module_name: str) -> object | None:
    """The module *module_name*, importing it only if it is PhenoTypic's own.

    A journal comes from an image store, which may be third-party; importing
    whatever module it names would run that module's code. Outside the
    ``phenotypic`` package only modules already loaded are used -- including
    those ``PHENOTYPIC_PRELOAD_MODULES`` lists, which are loaded first.
    """
    if module_name == "phenotypic" or module_name.startswith("phenotypic."):
        try:
            return importlib.import_module(module_name)
        except Exception:  # noqa: BLE001 -- any import failure means "not this prefix"
            return None
    return sys.modules.get(module_name)


def _recorded_class(operation_class: str) -> type | None:
    """Resolve a journal's ``module.qualname`` to its class, or ``None``.

    The split between module and qualname is not recorded, so the longest
    resolvable module prefix wins. A stored history names classes the
    running pipeline never imported, so PhenoTypic's own modules are imported
    here; any other module must already be loaded.
    """
    cached = _RECORDED_CLASSES.get(operation_class)
    if cached is not None:
        return cached
    from phenotypic.sdk_._preload import preload_custom_operation_modules_once

    preload_custom_operation_modules_once()
    parts = operation_class.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        found = _trusted_module(".".join(parts[:cut]))
        for attribute in parts[cut:]:
            if found is None:
                break
            found = getattr(found, attribute, None)
        if isinstance(found, type):
            _RECORDED_CLASSES[operation_class] = found
            return found
    return None


def _outside_unit_range(values: np.ndarray) -> bool:
    return bool(
        values.min() < -_UNIT_RANGE_TOLERANCE or values.max() > 1.0 + _UNIT_RANGE_TOLERANCE
    )


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
        ReferenceImageError: The blank cannot be resolved or read; its
            shape or bit depth differs from the target's; one of the pair is
            RGB and the other single-channel; or either projects outside
            ``[0, 1]``.
        StaleDetectMatError: ``detect_mat`` was already enhanced, or an
            ``ImageCorrector`` appears anywhere in the image's recorded
            history, or a class recorded there cannot be imported.

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
        # An in-memory entry is compared by its own name, since a copy of the
        # target defeats identity. normcase: on a case-insensitive filesystem
        # "T04.tif" is the file of image "t04".
        target_file = self._require_context().resolve_image(name)
        own = os.path.normcase(image.name)
        if (
            os.path.normcase(name) == own
            or (isinstance(target_file, Path)
                and os.path.normcase(source_image_stem(target_file)) == own)
            or (not isinstance(target_file, Path) and target_file.name == image.name)
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
        if blank.rgb.isempty() != image.rgb.isempty():
            single, rgb = (image.name, name) if image.rgb.isempty() else (name, image.name)
            raise ReferenceImageError(
                f"{single!r} is a single-channel image and {rgb!r} is RGB; a "
                f"single-channel gray and an RGB frame's projection are different "
                f"quantities, so the blank and the image must both be RGB or both not"
            )
        target = image.detect_mat[:]
        # The target's colour configuration, not the blank's: the blank was read
        # with imread's defaults, and an L*a*b* projection under another
        # illuminant or gamma would not cancel against identical pixels.
        background = (
            mode.compute_from_rgb(blank.rgb.normed(), image=image)
            if mode.requires_rgb
            else mode.compute(blank)
        )
        for label, values in ((f"image {image.name!r}", target), (f"blank {name!r}", background)):
            if _outside_unit_range(values):
                raise ReferenceImageError(
                    f"The {label} projects to [{float(values.min()):.4g}, "
                    f"{float(values.max()):.4g}] in detect mode {mode.name!r}, outside "
                    f"[0, 1]; a difference of such values is not a growth signal"
                )
        if self.polarity == "brighter":
            difference = target - background
        elif self.polarity == "darker":
            difference = background - target
        else:
            difference = np.abs(target - background)
        image.detect_mat[:] = np.clip(difference, 0.0, 1.0).astype(target.dtype, copy=False)
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
        # v2 per-application). Each recorded class is resolved by importing it:
        # a stored history comes from an earlier pipeline, whose classes this
        # process need not have imported. A class that cannot be resolved is
        # refused, since the guard cannot vouch for a history it cannot read.
        from phenotypic._core._provenance import _operations
        from phenotypic.abc_ import ImageCorrector

        for record in _operations(image._metadata.provenance_journal):
            recorded = str(record.get("operation_class"))
            cls = _recorded_class(recorded)
            if cls is None:
                raise StaleDetectMatError(
                    f"SubtractBlank cannot resolve {recorded!r} in the image's recorded "
                    f"history, so it cannot rule out an ImageCorrector there; import "
                    f"the module that defines it (on the CLI or SLURM, list it in "
                    f"PHENOTYPIC_PRELOAD_MODULES), or start from the raw image."
                )
            if issubclass(cls, ImageCorrector):
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
