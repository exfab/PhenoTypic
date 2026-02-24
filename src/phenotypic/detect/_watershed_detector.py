from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image, GridImage

from typing import Literal
import gc

import numpy as np
import numpy.ma as ma
from scipy.ndimage import distance_transform_edt
from skimage import feature, filters, morphology, segmentation

from phenotypic.abc_ import ThresholdDetector


class WatershedDetector(ThresholdDetector):
    """Region-growing colony detector using watershed segmentation from distance transform.

    WatershedDetector segments colonies using the watershed algorithm: (1) threshold
    image to binary mask, (2) compute distance transform to locate colony centers,
    (3) find local maxima as seed markers, (4) propagate regions via watershed on
    Sobel gradient. This region-growing approach effectively separates touching
    colonies and handles variable colony sizes better than global thresholding.

    Args:
        footprint: Structure element for peak detection. Options: 'auto' (infer from
            grid if GridImage), ndarray (custom), int (diamond radius), None (default).
            Controls neighborhood size for local maxima detection.

        min_size: Minimum object area in pixels (default 50). Objects smaller than
            this are removed, filtering dust and debris.

        compactness: Watershed compactness parameter (default 0.001). Higher values
            enforce more regularly shaped segments but may over-segment irregular colonies.

        connectivity: Connectivity for region labeling (1=4-connected, 2=8-connected,
            default 1). Controls how adjacent pixels merge into regions.

        relabel: If True (default), relabel segments to ensure consecutive IDs.

        ignore_zeros: If True (default), exclude zero-intensity pixels from threshold
            computation. Essential for images with black borders or masks.

    Attributes:
        footprint, min_size, compactness, connectivity, relabel, ignore_zeros

    Returns:
        Image: Input image with objmap set to labeled colonies from watershed segmentation.

    Raises:
        ValueError: If invalid parameters or computation fails (e.g., out of memory).

    **Use cases**

    - **Touching/overlapping colonies:** Region-growing effectively separates colonies
      in close contact where threshold-based methods merge them.
    - **Variable colony sizes:** Distance transform-based seeding adapts to colony size
      variations better than fixed-threshold methods.
    - **Irregular colony shapes:** Watershed respects local intensity gradients,
      handling non-circular morphologies better than geometric methods.

    **Limitations**

    - Memory-intensive. Distance transform, gradient, and watershed on large images
      consume significant RAM. Not suitable for very large images on memory-constrained systems.
    - Compactness parameter tuning required. Incorrect values cause over/under-segmentation.
    - Assumes detectable local intensity maxima. Very faint or flat colonies may not
      seed properly, causing under-segmentation.
    - Sensitive to noise. Noisy backgrounds can create spurious peaks. Pre-blur with
      GaussianBlur recommended before detection.
    - Slower than simple thresholding. Distance transform and watershed operations
      are computationally expensive.

    **Parameter effects on colony detection**

    - **footprint:** Larger footprints merge nearby peaks, fewer seeds → larger regions.
      Smaller footprints detect more peaks, more seeds → finer segmentation.
    - **min_size:** Filters small noise but may remove genuine small colonies if set
      too high. Balance sensitivity vs robustness.
    - **compactness:** Controls segment regularity. Higher values enforce compact shapes
      but may violate true colony boundaries. Lower values follow intensity gradients.

    Examples:
        Basic watershed detection with preprocessing::

            from phenotypic import Image
            from phenotypic.detect import WatershedDetector

            plate = Image.imread("plate.jpg")
            detector = WatershedDetector(min_size=50, compactness=0.001)
            detected = detector.apply(plate)
            num_colonies = detected.objects.count
            print(f"Detected {num_colonies} colonies via watershed")

        Pipeline with Gaussian blur for noise reduction::

            from phenotypic import ImagePipeline
            from phenotypic.enhance import GaussianBlur
            from phenotypic.detect import WatershedDetector

            pipeline = ImagePipeline([
                GaussianBlur(sigma=1.5),
                WatershedDetector(min_size=50, compactness=0.001)
            ])

            image = Image.imread("plate.jpg")
            result = pipeline.apply(image)
    """

    def __init__(
            self,
            footprint: Literal["auto"] | np.ndarray | int | None = None,
            min_size: int = 50,
            compactness: float = 0.001,
            connectivity: int = 1,
            relabel: bool = True,
            ignore_zeros: bool = True,
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
        image.objmap.relabel(connectivity=self.connectivity)

        # Final comprehensive memory report
        self._log_memory_usage(
                "final cleanup and relabeling",
                include_process=True,
                include_tracemalloc=True,
        )

        return image
