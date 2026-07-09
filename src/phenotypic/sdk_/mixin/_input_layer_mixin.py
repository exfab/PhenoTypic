"""Mixin letting an operation read either ``detect_mat`` or the pristine ``rgb`` layer."""

from __future__ import annotations

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

        Returns:
            The 2-D ``detect_mat``, or a 3-D float32 RGB copy normalized to [0, 1].

        Raises:
            EmptyImageError: If ``input_layer="rgb"`` on a grayscale-only image.
        """
        if self.input_layer == "rgb":
            # ``normed()`` returns float64; halve the intermediate on large plates.
            return image.rgb.normed().astype(np.float32)
        return image.detect_mat[:]

    def _project_to_detect_mat(self, image: "Image", arr: np.ndarray) -> np.ndarray:
        """Collapse a 3-D array to 2-D via the image's ``detect_mode``.

        A 2-D array is returned unchanged (identity, not a copy).
        """
        if arr.ndim == 2:
            return arr
        from phenotypic._core._image_parts.detection_modes import get_detection_mode

        mode = get_detection_mode(image.detect_mode)
        return mode.compute_from_rgb(arr, image=image)

    def _guard_input_range(self, arr: np.ndarray) -> np.ndarray:
        """Rescale *arr* into [0, 1] when it strays outside, else return it unchanged.

        skimage's ``adjust_gamma`` / ``adjust_log`` / ``adjust_sigmoid`` raise
        ``ValueError`` on negative input, which a signed filter such as
        ``FocusEdgeLaplace`` produces. Skipped entirely when ``norm is None`` so a
        deliberately non-normalized (e.g. GAT-stabilized) signal is left alone.
        """
        if getattr(self, "norm", "clip") is None:
            return arr
        if arr.min() < 0.0 or arr.max() > 1.0:
            return rescale_intensity(arr, out_range=(0.0, 1.0))
        return arr
