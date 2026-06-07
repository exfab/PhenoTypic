from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage
    from phenotypic._core._image import Image

from typing import Any, Literal
import gc

import numpy as np
import numpy.ma as ma
from pydantic import field_validator
from scipy.ndimage import distance_transform_edt
from skimage import feature, filters, morphology, segmentation

from phenotypic.abc_ import ThresholdDetector
from phenotypic.tools_.typing_ import NdArrayField


class WatershedDetector(ThresholdDetector):
    """Detect and separate touching colonies by watershed segmentation on a distance-transform surface.

    Thresholds the plate image to a binary mask, computes a Euclidean distance
    transform to locate colony centres, seeds markers at local maxima of that
    surface, and propagates labelled regions via compact watershed on the Sobel
    gradient. This region-growing approach individually labels colonies that are
    in physical contact — a scenario where global thresholding merges them into
    a single object. For a full comparison of detection strategies, see
    :doc:`/explanation/detection_strategies_compared`.

    Best For:
        - Dense plates where yeast or bacterial colonies touch or overlap and
          must be counted individually.
        - Mutant-library plates with variable colony sizes where the distance
          transform naturally adapts seed placement to each colony's footprint.
        - Post-incubation plates where colony crowding is the primary
          segmentation challenge.
        - Round, compact colonies on rich-media agar where compactness
          regularisation reinforces the expected shape.

    Consider Also:
        - :class:`OtsuDetector` when colonies are well-separated and a simple
          global threshold suffices without region-growing.
        - :class:`RoundPeaksDetector` when colonies sit on a regular pinned
          grid and peak-based grid-cell assignment is more efficient.
        - :class:`FilamentousFungiDetector` when colonies exhibit spreading
          hyphal growth rather than compact morphology.
        - :class:`CannyDetector` when edge contrast is stronger than intensity
          contrast for delineating colony boundaries.

    Args:
        footprint: Structuring element for peak suppression on the
            distance-transform surface. ``'auto'`` infers the diamond radius
            from grid spacing (GridImage only, half the well pitch); an
            ``int`` is expanded to a diamond of that radius in pixels; an
            ``ndarray`` supplies a custom binary footprint; ``None`` (default)
            lets scikit-image use a 1-px minimum distance, which typically
            over-seeds dense images. Larger footprints merge nearby peaks into
            fewer seeds; smaller footprints allow finer segmentation. Typical
            diamond radius range: 5--50 px. Default: None. A reasonable
            starting point for pinned-array plates is ``'auto'``, which sets
            the radius to approximately half the well pitch.
        min_size: Minimum object area in pixels. Objects smaller than this
            are removed from the binary mask before distance-transform
            computation (to reduce noise) and from the final labelled map.
            Typical range: 20--200 px, scaling with image resolution and
            colony size. Default: 50.
        compactness: Compact-watershed shape-regularisation penalty. Higher
            values produce more geometrically regular, convex segments; lower
            values let region boundaries follow the Sobel gradient freely,
            fitting irregular colony morphologies. Typical range:
            0.0001--0.1. Increase toward 0.01--0.05 for round yeast colonies
            on rich agar; decrease toward 0.0001 for mucoid, sectored, or
            spreading morphologies. Default: 0.001.
        connectivity: Pixel connectivity for watershed flooding and final
            region labelling. ``1`` for 4-connectivity (default); ``2`` for
            8-connectivity. 4-connectivity avoids false merges at diagonal
            colony contact points. Default: 1.
        relabel: When ``True`` (default), relabels segments to consecutive
            integer IDs starting at 1 after watershed. Set to ``False`` only
            when downstream code depends on the raw marker indices. Default:
            True.
        ignore_zeros: When ``True``, Otsu threshold is computed only from
            non-zero pixels and zero-valued pixels are forced to background.
            Enable for images with black borders, scanner shadow, or
            pre-masked regions outside the plate area where structural zeros
            would otherwise bias the Otsu histogram. Default: False.

    Returns:
        Image: Input image with ``objmap`` set to a labelled colony map where
        each colony receives a unique integer label, and ``objmask`` derived
        from the non-zero entries of that map.

    References:
        [1] P. Neubert and P. Protzel, "Compact watershed and preemptive SLIC:
        On improving trade-offs of superpixel segmentation algorithms," in
        *Proc. 22nd Int. Conf. Pattern Recognit. (ICPR)*, Stockholm, Sweden,
        2014, pp. 996--1001.
        [2] N. Otsu, "A threshold selection method from gray-level
        histograms," *IEEE Trans. Syst., Man, Cybern.*, vol. 9, no. 1,
        pp. 62--66, Jan. 1979.

    See Also:
        :doc:`/tutorials/notebooks/02_detecting_colonies` for a step-by-step
        tutorial demonstrating colony detection on real plate images.
        :doc:`/how_to/notebooks/choose_detection_algorithm` for a guide to
        selecting the right detector for your plate images.
        :doc:`/explanation/detection_strategies_compared` for an in-depth
        comparison of all detection strategies and their failure modes.
    """

    footprint: Literal["auto"] | NdArrayField | int | None = None
    min_size: int = 50
    compactness: float = 0.001
    connectivity: int = 1
    relabel: bool = True
    ignore_zeros: bool = False

    @field_validator("footprint", mode="before")
    @classmethod
    def _expand_int_footprint(cls, value: Any) -> Any:
        """Expand an integer ``footprint`` into a diamond structuring element.

        Reproduces the ``match`` block of the pre-migration ``__init__``:
        an ``int`` is turned into ``morphology.diamond(value)``; ``'auto'``,
        an ``np.ndarray``, and ``None`` pass through untouched (the
        ``np.ndarray`` branch is then handled by ``NdArrayField``).
        """
        if isinstance(value, int) and not isinstance(value, bool):
            return morphology.diamond(value)
        return value

    def _operate(self, image: Image | GridImage) -> Image:
        from phenotypic import Image, GridImage

        enhanced_matrix = image.detect_mat[
            :
        ]  # direct access to reduce memory shape, but careful to not delete
        self._log_memory_usage("getting detection matrix")

        # Determine shape for peak detection
        if self.footprint == "auto":
            if isinstance(image, GridImage):
                est_footprint_diameter = max(
                        image.shape[0] // image.grid.nrows,
                        image.shape[1] // image.grid.ncols,
                )
                footprint = morphology.diamond(est_footprint_diameter // 2)
                del est_footprint_diameter
            elif isinstance(image, Image):
                # Not enough information with a normal image to infer
                footprint = None
        else:
            # Use the shape as defined in __init__ (None, ndarray, or processed int)
            footprint = self.footprint
        self._log_memory_usage("determining shape")

        # Prepare values for threshold calculation
        if self.ignore_zeros:
            # Use masked array to avoid copying non-zero values
            masked_enh = ma.masked_equal(enhanced_matrix, 0)
            # Safety check: if all values are zero, fall back to using all values
            if masked_enh.count() == 0:
                threshold = filters.threshold_otsu(enhanced_matrix)
            else:
                threshold = filters.threshold_otsu(masked_enh)

            # Create binary mask: zeros are always background, non-zeros compared to threshold
            binary = (enhanced_matrix >= threshold) & (enhanced_matrix != 0)
            del masked_enh
        else:
            threshold = filters.threshold_otsu(enhanced_matrix)
            binary = enhanced_matrix >= threshold

        del threshold  # don't need this after obtaining binary mask
        self._log_memory_usage("threshold calculation and binary mask creation")

        binary = morphology.remove_small_objects(
                binary, min_size=self.min_size
        )  # clean to reduce runtime

        # Ensure binary is contiguous for memory-efficient operations (only if needed)
        if not binary.flags["C_CONTIGUOUS"]:
            binary = np.ascontiguousarray(binary)

        # Memory-intensive distance transform operation
        self._log_memory_usage("before distance transform",
                               include_tracemalloc=True)
        # Allocate float32 output directly to avoid intermediate float64 array
        dist_matrix = np.empty(binary.shape, dtype=np.float64)
        distance_transform_edt(binary, distances=dist_matrix)
        self._log_memory_usage("after distance transform",
                               include_tracemalloc=True)

        max_peak_indices = feature.peak_local_max(
                image=dist_matrix, footprint=footprint, labels=binary
        )

        del footprint, dist_matrix
        gc.collect()  # Force garbage collection to free memory before watershed
        self._log_memory_usage("after peak detection", include_tracemalloc=True)

        # Create markers more efficiently: allocate once and label directly
        max_peaks = np.zeros(shape=enhanced_matrix.shape, dtype=np.int32)
        max_peaks[tuple(max_peak_indices.T)] = np.arange(1, len(max_peak_indices) + 1)

        del max_peak_indices
        self._log_memory_usage("creating max peaks array")

        # Sobel filter enhances edges which improve watershed to nearly the point of necessity in most cases
        gradient = filters.sobel(enhanced_matrix)
        # Convert to float32 and ensure contiguity in one step if needed
        if gradient.dtype != np.float32 or not gradient.flags["C_CONTIGUOUS"]:
            gradient = np.asarray(gradient, dtype=np.float32, order="C")
        self._log_memory_usage("Sobel filter for gradient",
                               include_tracemalloc=True)

        # Memory-intensive watershed operation - detailed tracking
        self._log_memory_usage(
                "before watershed segmentation",
                include_process=True,
                include_tracemalloc=True,
        )

        objmap = segmentation.watershed(
                image=gradient,
                markers=max_peaks,
                compactness=self.compactness,
                connectivity=self.connectivity,
                mask=binary,
        )

        self._log_memory_usage(
                "after watershed segmentation",
                include_process=True,
                include_tracemalloc=True,
        )
        if objmap.dtype != np.uint16:
            objmap = objmap.astype(image._OBJMAP_DTYPE)

        del max_peaks, gradient, binary
        gc.collect()  # Force garbage collection after watershed to free memory

        objmap = morphology.remove_small_objects(objmap, min_size=self.min_size)
        image.objmap[:] = objmap
        if self.relabel:
            image.objmap.relabel(connectivity=self.connectivity)

        # Final comprehensive memory report
        self._log_memory_usage(
                "final cleanup and optional relabeling",
                include_process=True,
                include_tracemalloc=True,
        )

        return image
