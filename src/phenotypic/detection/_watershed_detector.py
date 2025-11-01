from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image, GridImage

from typing import Literal
import gc

import numpy as np
import numpy.ma as ma
import scipy.ndimage as ndimage
from scipy.ndimage import distance_transform_edt
from skimage import feature, filters, morphology, segmentation

from phenotypic.ABC_ import ThresholdDetector


class WatershedDetector(ThresholdDetector):
    """
    Class for detecting objects in an image using the Watershed algorithm.

    The WatershedDetector class processes images to detect and segment objects
    by applying the watershed algorithm. This class extends the capabilities
    of ThresholdDetector and includes customization for parameters such as footprint
    size, minimum object size, compactness, and connectivity. This is useful for
    image segmentation tasks, where proximity-based object identification is needed.

    Note:
        Its recommended to use `GaussianBlur` beforehand

    Attributes:
        footprint (Literal['auto'] | np.ndarray | int | None): Structure element to define
            the neighborhood for dilation and erosion operations. Can be specified directly
            as 'auto', an ndarray, an integer for diamond size, or None for implementation-based
            determination.
        min_size (int): Minimum size of objects to retain during segmentation.
            Objects smaller than this other_image are removed.
        compactness (float): Compactness parameter controlling segment shapes. Higher values
            enforce more regularly shaped objects.
        connectivity (int): The connectivity level used for determining connected components.
            Represents the number of dimensions neighbors need to share (1 for fully
            connected, higher values for less connectivity).
        relabel (bool): Whether to relabel segmented objects during processing to ensure
            consistent labeling.
        ignore_zeros (bool): Whether to exclude zero-valued pixels from threshold calculation.
            When True, Otsu threshold is calculated using only non-zero pixels, and zero pixels
            are automatically treated as background. When False, all pixels (including zeros)
            are used for threshold calculation. Default is True, which is useful for microscopy
            images where zero pixels represent true background or imaging artifacts.
    """

    def __init__(self,
                 footprint: Literal['auto'] | np.ndarray | int | None = None,
                 min_size: int = 50,
                 compactness: float = 0.001,
                 connectivity: int = 1,
                 relabel: bool = True,
                 ignore_zeros: bool = True):
        super().__init__()

        match footprint:
            case x if isinstance(x, int):
                self.footprint = morphology.diamond(footprint)
            case x if isinstance(x, np.ndarray):
                self.footprint = footprint
            case 'auto':
                self.footprint = 'auto'
            case None:
                # footprint will be automatically determined by implementation
                self.footprint = None
        self.min_size = min_size
        self.compactness = compactness
        self.connectivity = connectivity
        self.relabel = relabel
        self.ignore_zeros = ignore_zeros

    def _operate(self, image: Image | GridImage) -> Image:
        from phenotypic import Image, GridImage

        enhanced_matrix = image.enh_matrix[:]  # direct access to reduce memory footprint, but careful to not delete
        self._log_memory_usage("getting enhanced matrix")

        # Determine footprint for peak detection
        if self.footprint == 'auto':
            if isinstance(image, GridImage):
                est_footprint_diameter = max(image.shape[0]//image.grid.nrows, image.shape[1]//image.grid.ncols)
                footprint = morphology.diamond(est_footprint_diameter//2)
                del est_footprint_diameter
            elif isinstance(image, Image):
                # Not enough information with a normal image to infer
                footprint = None
        else:
            # Use the footprint as defined in __init__ (None, ndarray, or processed int)
            footprint = self.footprint
        self._log_memory_usage("determining footprint")

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

        binary = morphology.remove_small_objects(binary, min_size=self.min_size)  # clean to reduce runtime

        # Ensure binary is contiguous for memory-efficient operations (only if needed)
        if not binary.flags['C_CONTIGUOUS']:
            binary = np.ascontiguousarray(binary)

        # Memory-intensive distance transform operation
        self._log_memory_usage("before distance transform", include_tracemalloc=True)
        # Allocate float32 output directly to avoid intermediate float64 array
        dist_matrix = np.empty(binary.shape, dtype=np.float64)
        distance_transform_edt(binary, distances=dist_matrix)
        self._log_memory_usage("after distance transform", include_tracemalloc=True)

        max_peak_indices = feature.peak_local_max(
                image=dist_matrix,
                footprint=footprint,
                labels=binary)

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
        if gradient.dtype != np.float32 or not gradient.flags['C_CONTIGUOUS']:
            gradient = np.asarray(gradient, dtype=np.float32, order='C')
        self._log_memory_usage("Sobel filter for gradient", include_tracemalloc=True)

        # Memory-intensive watershed operation - detailed tracking
        self._log_memory_usage("before watershed segmentation",
                               include_process=True, include_tracemalloc=True)

        objmap = segmentation.watershed(
                image=gradient,
                markers=max_peaks,
                compactness=self.compactness,
                connectivity=self.connectivity,
                mask=binary,
        )

        self._log_memory_usage("after watershed segmentation",
                               include_process=True, include_tracemalloc=True)
        if objmap.dtype != np.uint16:
            objmap = objmap.astype(image._OBJMAP_DTYPE)

        del max_peaks, gradient, binary
        gc.collect()  # Force garbage collection after watershed to free memory

        objmap = morphology.remove_small_objects(objmap, min_size=self.min_size)
        image.objmap[:] = objmap
        image.objmap.relabel(connectivity=self.connectivity)

        # Final comprehensive memory report
        self._log_memory_usage("final cleanup and relabeling",
                               include_process=True, include_tracemalloc=True)

        return image
