import scipy.ndimage as ndimage
from scipy.ndimage import distance_transform_edt
from skimage import filters, segmentation, morphology, measure, feature
import numpy as np
from typing import Literal

from phenotypic.abstract import ThresholdDetector
from phenotypic import Image, GridImage


class WatershedDetector(ThresholdDetector):

    def __init__(self, footprint: Literal['auto'] | np.ndarray | int | None = None,
                 min_size: int = 50,
                 compactness: float = 0.001,
                 connectivity: int = 1):
        match footprint:
            case x if isinstance(x, int):
                self.footprint = morphology.disk(footprint)
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

    @staticmethod
    def _operate(image: Image | GridImage, footprint: int | str | np.ndarray | None, min_size: int, compactness: float, connectivity: float) -> Image:
        enhanced_matrix = image.enh_matrix[:]

        if footprint == 'auto':
            if isinstance(image, GridImage):
                est_footprint_diameter = max(image.shape[0] // image.grid.nrows, image.shape[1] // image.grid.ncols)
                footprint = morphology.disk(est_footprint_diameter // 2)
            elif isinstance(image, Image):
                # Not enough information with a normal image to infer
                footprint = None

        binary = enhanced_matrix > filters.threshold_otsu(enhanced_matrix)  # TODO: add alternative to otsu eventually?
        dist_matrix = distance_transform_edt(binary)
        max_peak_indices = feature.peak_local_max(
            image=dist_matrix,
            footprint=footprint,
            labels=binary)
        max_peaks = np.zeros(shape=enhanced_matrix.shape)
        max_peaks[tuple(max_peak_indices.T)] = 1
        max_peaks, _ = ndimage.label(max_peaks)  # label peaks
        print(max_peaks.shape)

        # Sobel filter enhances edges which improve watershed to nearly the point of necessity in most cases
        gradient = filters.sobel(enhanced_matrix)
        objmap = segmentation.watershed(
            image=gradient,
            markers=max_peaks,
            compactness=compactness,
            connectivity=connectivity,
            mask=binary,
        )

        objmap = morphology.remove_small_objects(objmap, min_size=min_size)
        image.objmap[:] = objmap
        return image
