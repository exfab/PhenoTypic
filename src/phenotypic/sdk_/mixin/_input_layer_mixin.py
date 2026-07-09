"""Mixin letting an operation read either ``detect_mat`` or the pristine ``rgb`` layer."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel
from skimage.exposure import rescale_intensity

from phenotypic.sdk_.typing_ import InputLayer

if TYPE_CHECKING:
    from phenotypic._core._image import Image


class InputLayerMixin(BaseModel):
    """Adds an ``input_layer`` field selecting the operation's source array.

    Pointwise intensity curves are non-linear, so applying one to the three RGB
    channels and *then* collapsing to a detection matrix gives a different — often
    better — colony/background separation than collapsing first. This mixin exposes
    that choice without changing the output contract: the only layer an enhancer
    ever writes is still ``detect_mat``.

    When ``input_layer="rgb"`` the 3-D result is collapsed back to 2-D by projecting
    it through the image's own ``detect_mode``, so an upstream
    ``SetDetectMode(mode="MinRGB")`` is honoured.

    The field is **appended** to the end of the subclass's field order. When stacked
    with :class:`NormalizedOutputMixin`, list this mixin first; the resulting order
    is ``[…op params…, norm, input_layer]``.

    Note:
        Reading ``rgb`` discards any enhancement a prior operation wrote to
        ``detect_mat`` — the same behaviour as ``SetDetectMode``. This is documented,
        not enforced.
    """

    input_layer: InputLayer = "detect_mat"

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Move ``input_layer`` to the end of the subclass's field order."""
        super().__pydantic_init_subclass__(**kwargs)
        fields = cls.__pydantic_fields__
        if "input_layer" in fields and list(fields)[-1] != "input_layer":
            fields["input_layer"] = fields.pop("input_layer")
            cls.model_rebuild(force=True)

    def _read_input_layer(self, image: "Image") -> np.ndarray:
        """Return the source array for this operation.

        Both branches return a **read-only** array. Callers must not mutate the
        result in place; build a new array instead. The two layers would otherwise
        disagree — ``detect_mat`` hands back a non-writeable view onto the image's
        own buffer, while the ``rgb`` branch builds a fresh array — and an operation
        doing in-place work would succeed under one ``input_layer`` and raise under
        the other.

        Returns:
            The 2-D read-only ``detect_mat`` view, or a 3-D read-only float32 RGB
            array normalized to [0, 1].

        Raises:
            NoArrayError: If ``input_layer="rgb"`` on a grayscale-only image.
        """
        if self.input_layer == "rgb":
            # ``image.rgb[:]`` raises NoArrayError on a grayscale-only image, where
            # ``rgb.normed()`` would silently hand back a degenerate ``(0, 3)`` array.
            # Normalizing straight into float32 also avoids the uint->float64->float32
            # chain in ``normed()``, which peaks at 3x the size of its own result.
            raw = image.rgb[:]
            arr = np.asarray(raw, dtype=np.float32) / np.float32(image.rgb.vmax())
            arr.flags.writeable = False
            return arr
        return image.detect_mat[:]

    def _project_to_detect_mat(self, image: "Image", arr: np.ndarray) -> np.ndarray:
        """Collapse a 3-D array to 2-D via the image's ``detect_mode``.

        A 2-D array is returned unchanged (identity, not a copy). It is read-only,
        so a caller cannot corrupt the image's buffer through the alias.

        Raises:
            ValueError: If *arr* is neither 2-D nor a 3-channel 3-D array.
        """
        if arr.ndim == 2:
            return arr
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(
                f"Expected a 2-D detect_mat or a 3-D (rows, cols, 3) RGB array, "
                f"got shape {arr.shape}."
            )
        from phenotypic._core._image_parts.detection_modes import get_detection_mode

        mode = get_detection_mode(image.detect_mode)
        return mode.compute_from_rgb(arr, image=image)

    def _guard_input_range(self, arr: np.ndarray) -> np.ndarray:
        """Rescale *arr* into [0, 1] when it strays outside, else return it unchanged.

        skimage's ``adjust_gamma`` / ``adjust_log`` / ``adjust_sigmoid`` raise
        ``ValueError`` on negative input, which a signed filter such as
        ``FocusEdgeLaplace`` produces.

        Skipped entirely when ``norm is None`` so a deliberately non-normalized
        (e.g. GAT-stabilized) signal is left alone. An operation that mixes in
        :class:`InputLayerMixin` *without* :class:`NormalizedOutputMixin` has no
        ``norm`` field; it is treated as ``"clip"``, i.e. the guard is active. That
        default matches ``NormalizedOutputMixin``'s own, so stacking the mixins does
        not change input-side behaviour.

        Raises:
            ValueError: If *arr* contains NaN or infinity. ``min()``/``max()``
                propagate NaN, and both ``nan < 0`` and ``nan > 1`` are ``False``,
                so a NaN would otherwise slip past this guard silently and surface
                as skimage's opaque non-negative-values error. ``rescale_intensity``
                would additionally smear a single NaN across the whole array.
        """
        if getattr(self, "norm", "clip") is None:
            return arr
        # Two O(n) reductions, no temporary: min/max propagate NaN, and +/-inf
        # surface in exactly one of them.
        low, high = float(arr.min()), float(arr.max())
        if not (math.isfinite(low) and math.isfinite(high)):
            raise ValueError(
                f"{type(self).__name__}: the {self.input_layer!r} layer contains "
                f"non-finite values (min={low}, max={high}). Range-guarding it is "
                f"undefined. Fix the upstream operation, or set norm=None to pass "
                f"the array through untouched."
            )
        if low < 0.0 or high > 1.0:
            return rescale_intensity(arr, out_range=(0.0, 1.0))
        return arr
