import scipy.ndimage as ndimage
from scipy.ndimage import distance_transform_edt
from skimage import filters, segmentation, morphology, measure, feature
import numpy as np
from typing import Literal

from phenotypic.abstract import ThresholdDetector
from phenotypic import Image, GridImage


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
                 ignore_zeros:bool=True):
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
        enhanced_matrix = image.enh_matrix[:]

        # Determine footprint for peak detection
        if self.footprint == 'auto':
            if isinstance(image, GridImage):
                est_footprint_diameter = max(image.shape[0] // image.grid.nrows, image.shape[1] // image.grid.ncols)
                footprint = morphology.diamond(est_footprint_diameter // 2)
            elif isinstance(image, Image):
                # Not enough information with a normal image to infer
                footprint = None
        else:
            # Use the footprint as defined in __init__ (None, ndarray, or processed int)
            footprint = self.footprint

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
        binary = morphology.remove_small_objects(binary, min_size=self.min_size)
        dist_matrix = distance_transform_edt(binary)
        max_peak_indices = feature.peak_local_max(
            image=dist_matrix,
            footprint=footprint,
            labels=binary)
        max_peaks = np.zeros(shape=enhanced_matrix.shape)
        max_peaks[tuple(max_peak_indices.T)] = 1
        max_peaks, _ = ndimage.label(max_peaks)  # label peaks

        # Sobel filter enhances edges which improve watershed to nearly the point of necessity in most cases
        gradient = filters.sobel(enhanced_matrix)
        objmap = segmentation.watershed(
            image=gradient,
            markers=max_peaks,
            compactness=self.compactness,
            connectivity=self.connectivity,
            mask=binary,
        )

        objmap = morphology.remove_small_objects(objmap, min_size=self.min_size)
        image.objmap[:] = objmap
        image.objmap.relabel(connectivity=self.connectivity)
        return image
