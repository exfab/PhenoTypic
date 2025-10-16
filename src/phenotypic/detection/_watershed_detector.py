from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING: from phenotypic import Image

from typing import Literal

import numpy as np
import scipy.ndimage as ndimage
from scipy.ndimage import distance_transform_edt
from skimage import feature, filters, morphology, segmentation

from phenotypic.abstract import ThresholdDetector


class WatershedDetector(ThresholdDetector):
    """
    Class for detecting objects in an image using the Watershed algorithm.

    The WatershedDetector class processes images to detect and segment objects
    by applying the watershed algorithm. This class extends the capabilities
    of ThresholdDetector and includes customization for parameters such as footprint
    size, minimum object size, compactness, and connectivity. This is useful for
    image segmentation tasks, where proximity-based object identification is needed.

    Note:
        Its recommended to use `GaussianSmoother` beforehand

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

        enhanced_matrix = image._data.enh_matrix  # direct access to reduce memory footprint, but careful to not delete
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
            enh_vals = enhanced_matrix[enhanced_matrix != 0]
            # Safety check: if all values are zero, fall back to using all values
            if len(enh_vals) == 0:
                enh_vals = enhanced_matrix
                threshold = filters.threshold_otsu(enh_vals)
            else:
                threshold = filters.threshold_otsu(enh_vals)

            # Create binary mask: zeros are always background, non-zeros compared to threshold
            binary = (enhanced_matrix != 0) & (enhanced_matrix >= threshold)
        else:
            enh_vals = enhanced_matrix
            threshold = filters.threshold_otsu(enh_vals)
            binary = enhanced_matrix >= threshold

        del threshold, enh_vals  # don't need these after obtaining binary mask
        self._log_memory_usage("threshold calculation and binary mask creation")

        binary = morphology.remove_small_objects(binary, min_size=self.min_size)  # clean to reduce runtime

        # Memory-intensive distance transform operation
        self._log_memory_usage("before distance transform", include_tracemalloc=True)
        dist_matrix = distance_transform_edt(binary).astype(np.float32)
        self._log_memory_usage("after distance transform", include_tracemalloc=True)

        max_peak_indices = feature.peak_local_max(
                image=dist_matrix,
                footprint=footprint,
                labels=binary)

        del footprint, dist_matrix
        self._log_memory_usage("after peak detection", include_tracemalloc=True)

        max_peaks = np.zeros(shape=enhanced_matrix.shape)
        max_peaks[tuple(max_peak_indices.T)] = 1

        del max_peak_indices
        self._log_memory_usage("creating max peaks array")

        max_peaks, _ = ndimage.label(max_peaks)  # label peaks

        # Sobel filter enhances edges which improve watershed to nearly the point of necessity in most cases
        gradient = filters.sobel(enhanced_matrix)
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

        objmap = morphology.remove_small_objects(objmap, min_size=self.min_size)
        image.objmap[:] = objmap
        image.objmap.relabel(connectivity=self.connectivity)

        # Final comprehensive memory report
        self._log_memory_usage("final cleanup and relabeling",
                               include_process=True, include_tracemalloc=True)

        return image
