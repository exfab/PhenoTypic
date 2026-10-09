from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from pydantic import Field

from phenotypic.abc_ import BackgroundSubtraction
from phenotypic.sdk_.mixin import NormalizedOutputMixin
from phenotypic.sdk_.typing_ import LineAxis, SurfaceFit, SurfaceMethod, TuneSpec

from ._poly_surface_kernels import flatten_surface


class SubtractPolySurface(NormalizedOutputMixin, BackgroundSubtraction):
    """Level ``detect_mat`` by subtracting a fitted constant, plane, polynomial surface or per-line polynomial.

    Fits a smooth background to the whole image (or to each scan line) by least
    squares and subtracts it, removing agar thickness gradients, scanner shading
    and tilted plates. Defaults reproduce Gwyddion 2.71's leveling tools [1].
    Unlike a blur-based estimate such as :class:`SubtractGaussian`, the
    background is a low-order analytic surface, so a large colony cannot be
    absorbed into it -- provided the fit is not itself pulled toward the
    colonies (see the warning below and ``fit``).

    Leveling methods:
        - ``"offset"``: subtract a constant -- the mean under
          ``fit="lstsq"``, the sigma-clipped background level under
          ``fit="robust"``.
        - ``"plane"``: fit a tilted plane and subtract its tilt (Gwyddion's
          Plane Level). The fitted constant is not subtracted, so the
          background stays near the image mean (see Output level).
        - ``"polynomial"``: fit and subtract a 2-D polynomial surface of degree
          ``order`` (Gwyddion's Polynomial Background). With
          ``independent=True`` the terms are ``u^p v^q`` with
          ``p, q <= order``; with ``False`` they are those with
          ``p + q <= order``.
        - ``"line"``: fit a degree-``line_order`` polynomial to every scan line
          separately (Gwyddion's Align Rows, *Polynomial* method -- not its
          default *Median* method) and level each line to the image's mean.

        Fields a method does not read are ignored, not rejected.

    Output level:
        Switching ``method`` also changes where the background lands.

        - ``"offset"``: 0 (Gwyddion's Zero Mean Value).
        - ``"plane"``: the image mean plus ``(bx + by) / 2``, where ``bx``,
          ``by`` are the fitted per-pixel slopes, because the tilt is pivoted
          at ``(W/2, H/2)`` (Gwyddion's Plane Level).
        - ``"polynomial"``: 0 (Gwyddion's Polynomial Background).
        - ``"line"``: the image mean, on every line (Gwyddion's Align Rows).

        With the default ``norm="clip"``, methods that land at 0 lose the
        negative half of the background noise. Use ``norm="rescale"`` or
        ``norm=None`` when that matters.

    Best For:
        - Plates with a smooth tilt or bowl-shaped shading across the scan bed.
        - Banded scanner artefacts (``method="line"``) on an image cropped to
          the agar, where no scan line is a fifth or more colony and none ends
          in the plate rim or the scan border.
        - Cheap, deterministic flattening ahead of thresholding, with few
          parameters.

    Consider Also:
        - :class:`SubtractGaussian` for backgrounds that are not low-order smooth.
        - :class:`SubtractRollingBall` for morphological background estimation.
        - :class:`FlattenIllumination` for frequency-domain illumination
          correction.

    Warning:
        ``fit="lstsq"`` (the Gwyddion default) is biased by colonies: measured
        at 2.3--9.5 sigma of the background noise at 10--40% plate cover.
        ``fit="robust"`` (least squares sigma-clipped on a median absolute
        deviation scale [2]) is the recommended setting for plates, within
        these measured limits:

        - Surfaces (``offset``, ``plane``, ``polynomial``) recover the
          background to 0.055 sigma or less with colonies dispersed over up
          to about 40% of the plate.
        - A contiguous region along an image edge -- plate rim, out-of-plate
          scan border, meniscus -- that covers about 10% or more of a
          dimension defeats the robust surface fit: 3.7--18 sigma at 10%,
          2.2--11 sigma at 20% (0.26 sigma or less at 5%). Crop to the plate
          first.
        - For ``method="line"`` the limit applies **per line**: lines under
          about 20% colony (dispersed) recover, lines between 20% and 50% are
          unreliable, and lines of 50% or more fail. A contiguous defect at
          the end of a line covering about 15% of it or more fails even below
          20% (8--112 sigma, growing with its amplitude). On an arrayed
          plate, a scan line through a row of colony centres is mostly colony
          and cannot be leveled by either fit.

    Args:
        method: Leveling to apply: ``"offset"``, ``"plane"``,
            ``"polynomial"`` or ``"line"``. Default: ``"plane"``.
        order: Polynomial degree per axis (or total degree when
            ``independent=False``). Read only by ``method="polynomial"``; the
            image must be at least ``order + 1`` pixels on each side.
            Default: 3.
        independent: ``True`` fits ``(order + 1)^2`` terms (per-axis degree);
            ``False`` fits the ``(order + 1)(order + 2) / 2`` terms of total
            degree ``order``. Read only by ``method="polynomial"``.
            Default: ``True``.
        line_order: Polynomial degree fitted along each scan line. Read only by
            ``method="line"``; lines must be at least ``line_order + 1`` pixels
            long. Default: 1.
        line_axis: ``"row"`` levels horizontal scan lines, ``"column"`` vertical
            ones. Read only by ``method="line"``. Default: ``"row"``.
        fit: ``"lstsq"`` for plain least squares, ``"robust"`` for iterative
            sigma-clipped least squares that ignores colonies. Read by all
            methods. Default: ``"lstsq"``.
        clip_sigma: Clipping threshold in robust standard deviations. Read only
            when ``fit="robust"``. Default: 3.0.
        max_iter: Cap on clipping iterations. Read only when ``fit="robust"``.
            Default: 10.
        norm: Output range policy. ``"clip"`` (default) saturates values outside
            [0, 1]; ``"rescale"`` remaps the observed range onto [0, 1]; ``None``
            passes values through untouched.

    Returns:
        Image: Input image with ``detect_mat`` leveled. ``rgb`` and ``gray``
        are unchanged.

    Raises:
        ValueError: At apply time, if the image is smaller than 2 pixels on a
            side, too small for the requested ``order`` or ``line_order``, or
            the fit is rank deficient. ``apply`` wraps it, so it is the root
            cause of the raised exception chain.

    Examples:
        Remove a smooth agar gradient while ignoring the colonies. With
        ``norm=None`` the leveled agar sits at 0; plain least squares is
        pulled up by the colonies and leaves the agar below 0:

        >>> import numpy as np
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.enhance import SubtractPolySurface
        >>> plate = load_synth_yeast_plate()
        >>> agar = plate.objmap[:] == 0  # background of the detected plate
        >>> robust = SubtractPolySurface(
        ...     method="polynomial", fit="robust", norm=None)
        >>> leveled = robust.apply(plate).detect_mat[:]
        >>> bool(abs(np.median(leveled[agar])) < 0.005)
        True
        >>> plain = SubtractPolySurface(method="polynomial", norm=None)
        >>> biased = plain.apply(load_synth_yeast_plate()).detect_mat[:]
        >>> bool(np.median(biased[agar]) < -0.05)
        True

    References:
        [1] D. Nečas and P. Klapetek, "Gwyddion: an open-source software for
        SPM data analysis," *Cent. Eur. J. Phys.*, vol. 10, no. 1,
        pp. 181--188, 2012, doi: 10.2478/s11534-011-0096-2.

        [2] P. J. Rousseeuw and C. Croux, "Alternatives to the median absolute
        deviation," *J. Amer. Statist. Assoc.*, vol. 88, no. 424,
        pp. 1273--1283, 1993, doi: 10.1080/01621459.1993.10476408.

    See Also:
        :doc:`/explanation/what_enhancement_does` for background on
        illumination correction.
    """

    method: SurfaceMethod = "plane"
    # TODO: review bound (unverified vs literature)
    order: Annotated[int, TuneSpec(2, 5)] = Field(3, ge=2, le=11)
    independent: bool = True
    # TODO: review bound (unverified vs literature)
    line_order: Annotated[int, TuneSpec(0, 3)] = Field(1, ge=0, le=5)
    line_axis: LineAxis = "row"
    fit: SurfaceFit = "lstsq"
    # TODO: review bound (unverified vs literature)
    clip_sigma: Annotated[float, TuneSpec(2.0, 4.0)] = Field(3.0, gt=0.0)
    max_iter: Annotated[int, TuneSpec(tunable=False)] = Field(10, ge=1)

    def _operate(self, image: Image) -> Image:
        original = image.detect_mat[:]
        flat = flatten_surface(
            original.astype(np.float64),
            method=self.method,
            order=self.order,
            independent=self.independent,
            line_order=self.line_order,
            line_axis=self.line_axis,
            fit=self.fit,
            clip_sigma=self.clip_sigma,
            max_iter=self.max_iter,
        )
        image.detect_mat[:] = self._apply_norm(flat.astype(original.dtype))
        return image
