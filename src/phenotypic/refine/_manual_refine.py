from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
from pydantic import field_validator

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import ObjectRefiner
from phenotypic.sdk_.mixin._footprint_mixin import FootprintMixin
from phenotypic.sdk_.mixin._point_picker_mixin import PointPickerMixin
from phenotypic.sdk_.typing_ import TuneSpec


class ManualRefine(ObjectRefiner, PointPickerMixin, FootprintMixin):
    """Keep only objects whose footprints overlap user-specified coordinates.

    Filter an existing ``objmap``/``objmask`` by stamping a morphological
    footprint at each user-provided ``(y, x)`` coordinate and retaining
    only the labelled objects whose pixels intersect any stamp. Non-selected
    objects are dropped; **selected objects keep their original label IDs**,
    which allows downstream measurements and analyses to reference the same
    identifiers that existed before refinement.

    Unlike :class:`ManualPointDetector`, which *produces* an ``objmap`` from
    scratch at picked coordinates, this refiner *filters* the output of an
    earlier detector. It is the manual-curation counterpart to automated
    refiners such as :class:`SmallObjectRemover` or
    :class:`RemoveLowCircularity`, and is suitable for ground-truth
    curation, interactive review, and correcting systematic detector misses
    on a handful of colonies. For where refinement sits in a pipeline see
    :doc:`/explanation/refinement_strategies`.

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
        * :class:`RemoveBorderObjects` for automated exclusion of objects
          touching the image border (no manual step required).
        * :class:`SmallObjectRemover` for size-based filtering when
          artefacts are systematically smaller than true colonies.

    Args:
        centers: An N x 2 array-like of ``(y, x)`` pixel coordinates, one
            per colony to keep, supplied by the user (purely a manual
            choice). Accepts any sequence that ``np.asarray`` can convert
            (list of tuples, nested list, or NumPy array). When *None*
            (default) or empty, :meth:`apply` returns the image unchanged
            (no-op) rather than zeroing the map — safer when this refiner is
            chained in a pipeline before points have been picked.

        shape: Morphological footprint shape stamped at each coordinate
            when locating candidate labels. ``"disk"`` (default) preserves
            round colony geometry. ``"square"`` covers rectangular regions.
            ``"diamond"`` offers a compromise between the two.

        width: Diameter of the stamp footprint in pixels, governing how
            forgiving each pick is rather than the final object shape (the
            kept object retains its detected extent). Larger widths tolerate
            clicks that land slightly off the colony body but risk a single
            pick selecting two touching colonies; smaller widths demand more
            precise clicks. A reasonable starting point is a small fraction
            of the colony spacing so one click maps to one colony, then grow
            it if your picks routinely miss. Typical range: 5--50, depending
            on image resolution and colony size. Default: 15.

    Returns:
        Image: Input image with ``objmap``/``objmask`` restricted to the
        objects whose pixels overlap any stamped footprint. Original label
        IDs for surviving objects are preserved (non-consecutive labels
        are allowed).

    Note:
        The bundled :class:`PointPickerWidget` used by :meth:`napari`
        displays only ``rgb``, ``gray``, and ``detect_mat`` layers — it
        does **not** overlay the existing ``objmap``. Before calling
        ``ManualRefine.napari(image)``, preview what is available to pick
        with ``image.objmap.show()`` or ``image.show(overlay=True)`` so you can
        see which detections exist.

    See Also:
        :doc:`/tutorials/notebooks/02_detecting_colonies`
            Step-by-step tutorial for basic colony detection.
        :doc:`/how_to/notebooks/choose_detection_algorithm`
            Guide for selecting the right detector for your plate images.
        :doc:`/explanation/refinement_strategies`
            How refiners fit into the detection-to-measurement pipeline.
    """

    centers: list[tuple[int, int]] | None = None
    shape: Literal["square", "diamond", "disk"] = "disk"
    width: Annotated[int, TuneSpec(tunable=False)] = 15

    @field_validator("centers", mode="before")
    @classmethod
    def _coerce_centers(cls, centers: Any) -> Any:
        """Normalize picked-point input to a JSON-native list of ``(y, x)`` pairs.

        Accepts ``None``, an ``np.ndarray``, or any list/tuple of
        coordinate pairs. This replaces the coordinate coercion that the
        :class:`PointPickerMixin` previously performed in a
        ``__setattr__`` override (removed in the pydantic migration so it
        no longer clashes with pydantic's own ``__setattr__``).
        """
        if centers is None:
            return None
        if isinstance(centers, np.ndarray):
            return centers.tolist()
        return centers

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


ManualRefine.apply.__doc__ = ManualRefine._operate.__doc__
