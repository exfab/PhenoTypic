from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._grid_image import GridImage
    from phenotypic._core._image import Image

from typing import Literal
import gc

import numpy as np
import numpy.ma as ma
from scipy.ndimage import distance_transform_edt
from skimage import feature, filters, morphology, segmentation

from phenotypic.abc_ import ThresholdDetector


class WatershedDetector(ThresholdDetector):
    """Detect and separate touching colonies by watershed segmentation on a distance-transform surface.

    Threshold the plate image to a binary mask, compute a Euclidean distance
    transform to locate colony centres, seed markers at local maxima, and
    propagate labelled regions via watershed on the Sobel gradient. This
    region-growing approach individually labels colonies that are in physical
    contact -- a scenario where global thresholding merges them into a single
    object. For a full comparison see
    :doc:`/explanation/detection_strategies_compared`.

    Best For:
        * Dense plates where colonies touch or overlap and must be counted
          individually.
        * Plates with variable colony sizes (e.g., mutant libraries) where
          the distance transform naturally adapts seed placement.
        * Irregular colony morphologies that follow local intensity
          gradients better than geometric assumptions.
        * Post-incubation plates where colony crowding is the primary
          segmentation challenge.

    Consider Also:
        * :class:`OtsuDetector` when colonies are well-separated and a
          simple binary mask suffices.
        * :class:`RoundPeaksDetector` when colonies sit on a regular
          pinned grid and peak-based assignment is more efficient.
        * :class:`FilamentousFungiDetector` when colonies exhibit spreading,
          filamentous growth rather than compact morphology.
        * :class:`CannyDetector` when edge contrast is stronger than
          intensity contrast for delineating colony boundaries.

    Args:
        footprint: Structuring element for peak detection. ``'auto'``
            infers size from grid spacing (GridImage only); an int creates a
            diamond of that radius; an ndarray supplies a custom footprint;
            None (default) lets scikit-image choose. Larger footprints merge
            nearby peaks into fewer seeds; smaller footprints yield finer
            segmentation. Typical range: 5--50 (diamond radius in pixels).

        min_size: Minimum object area in pixels (default 50). Objects
            smaller than this are removed as dust or debris. Typical range:
            20--200 depending on image resolution and colony size.

        compactness: Watershed compactness parameter (default 0.001). Higher
            values enforce more regularly shaped segments; lower values let
            regions follow intensity gradients freely. Typical range:
            0.0001--0.1. Increase if colonies are round and over-segmented;
            decrease for irregular morphologies.

        connectivity: Connectivity for region labelling (1 = 4-connected,
            2 = 8-connected; default 1). Higher connectivity merges
            diagonally adjacent pixels.

        relabel: If True (default), relabel segments to consecutive IDs
            after watershed.

        ignore_zeros: If True (default), exclude zero-intensity pixels from
            threshold computation. Enable for plates with black borders or
            masked regions.

    Returns:
        Image: Input image with ``objmap`` set to a labelled colony map
        where each colony receives a unique integer label. ``objmask`` is
        derived from the non-zero region of the label map.

    Raises:
        ValueError: If invalid parameters are provided or if the distance
            transform / watershed computation fails.

    References:
        [1] S. Beucher and C. Lantuejoul, "Use of watersheds in contour
        detection," in *Proc. Int. Workshop on Image Processing*, CCETT,
        Rennes, France, 1979.

    See Also:
        :doc:`/tutorials/notebooks/02_detecting_colonies`
            Step-by-step tutorial for basic colony detection.
        :doc:`/how_to/notebooks/choose_detection_algorithm`
            Guide for selecting the right detector for your plate images.
        :doc:`/explanation/detection_strategies_compared`
            In-depth comparison of all detection strategies.
    """

    def __init__(
            self,
            footprint: Literal["auto"] | np.ndarray | int | None = None,
            min_size: int = 50,
            compactness: float = 0.001,
            connectivity: int = 1,
            relabel: bool = True,
            ignore_zeros: bool = False,
    ):
        super().__init__()

        match footprint:
            case x if isinstance(x, int):
                self.footprint = morphology.diamond(footprint)
            case x if isinstance(x, np.ndarray):
                self.footprint = footprint
            case "auto":
                self.footprint = "auto"
            case None:
                # shape will be automatically determined by implementation
                self.footprint = None
        self.min_size = min_size
        self.compactness = compactness
        self.connectivity = connectivity
        self.relabel = relabel
        self.ignore_zeros = ignore_zeros

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
                binary, max_size=self.min_size
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

        objmap = morphology.remove_small_objects(objmap, max_size=self.min_size)
        image.objmap[:] = objmap
        image.objmap.relabel(connectivity=self.connectivity)

        # Final comprehensive memory report
        self._log_memory_usage(
                "final cleanup and relabeling",
                include_process=True,
                include_tracemalloc=True,
        )

        return image
