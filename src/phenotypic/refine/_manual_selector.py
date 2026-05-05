from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_.mixin._footprint_mixin import FootprintMixin
from phenotypic.tools_.mixin._point_picker_mixin import PointPickerMixin


class ManualSelector(ObjectRefiner, PointPickerMixin, FootprintMixin):
    """Keep only objects whose footprints overlap user-specified coordinates.

    Filter an existing ``objmap``/``objmask`` by stamping a morphological
    footprint at each user-provided ``(y, x)`` coordinate and retaining
    only the labelled objects whose pixels intersect any stamp. Non-selected
    objects are dropped; **selected objects keep their original label IDs**,
    which allows downstream measurements and analyses to reference the same
    identifiers that existed before refinement.

    Unlike :class:`ManualPointDetector`, which *produces* an ``objmap`` from
    scratch at picked coordinates, ``ManualSelector`` *filters* the output of
    an earlier detector. It is the manual-curation counterpart to automated
    refiners such as :class:`SmallObjectRemover` or
    :class:`LowCircularityRemover`, and is suitable for ground-truth
    curation, interactive review, and correcting systematic detector misses
    on a handful of colonies.

    Args:
        centers: An N x 2 array-like of ``(y, x)`` pixel coordinates
            specifying each colony to keep. Accepts any sequence that
            ``np.asarray`` can convert (list of tuples, nested list, or
            NumPy array). When *None* or empty, :meth:`apply` returns the
            image unchanged (no-op) rather than zeroing the map — safer
            when the selector is chained in a pipeline before points have
            been picked.

        shape: Morphological footprint shape stamped at each coordinate
            when locating candidate labels. ``"disk"`` (default) preserves
            round colony geometry. ``"square"`` covers rectangular regions.
            ``"diamond"`` offers a compromise between the two.

        width: Diameter of the footprint in pixels (default 15). Larger
            widths are more forgiving — a click near an object is still
            captured even if it lands slightly off the colony body — but
            risk selecting multiple touching colonies with a single pick.
            Typical range: 5--50, depending on image resolution and colony
            size.

    Returns:
        Image: Input image with ``objmap``/``objmask`` restricted to the
        objects whose pixels overlap any stamped footprint. Original label
        IDs for surviving objects are preserved (non-consecutive labels
        are allowed).

    Best For:
        * Manual curation of auto-detected objects before measurement —
          drop false positives (dust, plate artefacts, merged colonies)
          without re-running the detector.
        * Building curated ground-truth subsets for benchmarking
          detection or measurement algorithms.
        * Interactive review of sparse or irregular plates where
          auto-detection misfires on a handful of colonies; pick the
          subset to keep rather than enumerating those to remove.

    Consider Also:
        * :class:`ManualPointDetector` when you want to *produce* an
          ``objmap`` at user coordinates rather than filter an existing
          one.
        * :class:`BorderObjectRemover` for automated exclusion of objects
          touching the image border (no manual step required).
        * :class:`SmallObjectRemover` for size-based filtering when
          artefacts are systematically smaller than true colonies.

    Note:
        The bundled :class:`PointPickerWidget` used by :meth:`napari`
        displays only ``rgb``, ``gray``, and ``detect_mat`` layers — it
        does **not** overlay the existing ``objmap``. Before calling
        ``selector.napari(image)``, preview what is available to pick
        with ``image.objmap.show()`` or ``image.plot.show()`` so you can
        see which detections exist.

    See Also:
        :doc:`/tutorials/notebooks/02_detecting_colonies`
            Step-by-step tutorial for basic colony detection.
        :doc:`/how_to/notebooks/choose_detection_algorithm`
            Guide for selecting the right detector for your plate images.

    Examples:
        Drop all detections except one chosen colony:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.refine import ManualSelector
        >>> import numpy as np
        >>> image = load_synth_yeast_plate()
        >>> detected = OtsuDetector().apply(image)
        >>> # Pick any pixel known to lie on an object
        >>> ys, xs = np.where(detected.objmap[:] > 0)
        >>> cy, cx = int(ys[0]), int(xs[0])
        >>> selector = ManualSelector(centers=[(cy, cx)], width=15)
        >>> curated = selector.apply(detected)
        >>> # Only the target label survives; its original ID is preserved
        >>> surviving = set(np.unique(curated.objmap[:])) - {0}
        >>> len(surviving)
        1

        Use in a pipeline for manual curation after automatic detection:

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.enhance import GaussianBlur
        >>> pipeline = ImagePipeline([
        ...     GaussianBlur(sigma=1.0),
        ...     OtsuDetector(),
        ...     ManualSelector(centers=[(cy, cx)], width=20),
        ... ])
        >>> result = pipeline.apply(image)
        >>> result.objmap[:].max() > 0
        True
    """

    def __init__(
        self,
        centers: np.ndarray | list | None = None,
        shape: Literal["square", "diamond", "disk"] = "disk",
        width: int = 15,
    ):
        super().__init__()
        self.centers = centers
        self.shape = shape
        self.width = width

    def _operate(self, image: Image) -> Image:  # type: ignore[override]
        if self.centers is None or len(self.centers) == 0:
            return image

        objmap = image.objmap[:]
        if objmap.max() == 0:
            return image

        h, w = objmap.shape
        fp_mask = self._make_footprint(shape=self.shape, width=self.width).astype(bool)
        selection_mask = np.zeros((h, w), dtype=bool)
        # _stamp_footprint also writes to a labeled buffer; we only need the
        # boolean union, so allocate a scratch buffer and discard it.
        scratch_labels = np.zeros((h, w), dtype=objmap.dtype)

        for idx, (cy, cx) in enumerate(self.centers, start=1):
            self._stamp_footprint(
                selection_mask, scratch_labels, fp_mask,
                int(round(cy)), int(round(cx)), idx,
            )

        keep_labels = np.unique(objmap[selection_mask])
        keep_labels = keep_labels[keep_labels > 0]
        filtered = np.where(
            np.isin(objmap, keep_labels), objmap, 0
        ).astype(objmap.dtype, copy=False)

        # objmask is a view derived from the same sparse backend as objmap;
        # writing objmap updates both. Writing objmask separately would
        # trigger skimage.measure.label() and destroy original label IDs.
        image.objmap[:] = filtered
        return image


ManualSelector.apply.__doc__ = ManualSelector._operate.__doc__
