from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

import numpy as np
from skimage.morphology import white_tophat

from phenotypic.abc_ import ImageEnhancer


class WhiteTophatEnhance(ImageEnhancer):
    """White top-hat transform to isolate small bright structures.

    Computes the white top-hat (original minus morphological opening) and keeps
    the result, effectively extracting bright features smaller than the
    structuring element. In agar plate colony images, this highlights small
    bright colonies or specks while suppressing larger background structures.

    Args:
        shape (str): Footprint geometry controlling which bright features are
            extracted. 'diamond' or 'disk' provide isotropic behavior on plates;
            'square' can align with sensor grid artifacts.
        width (int | None): Maximum bright-object width (in pixels) targeted
            for extraction. Objects smaller than this width will be highlighted.
            None picks a small default based on image dimensions.

    Returns:
        Image: Modified image with `detect_mat` containing only the extracted
        small bright structures.

    Raises:
        ValueError: If an unsupported footprint shape is provided.

    Use cases (agar plates):
        - Isolate small bright colonies from larger background structures.
        - Extract tiny bright specks for detection or quantification.
        - Highlight faint small colonies against uneven illumination.
        - Pre-processing step before detecting small colony phenotypes.

    Limitations:
        - Large colonies will be suppressed or removed entirely.
        - If width is too small, only noise/artifacts will be extracted.
        - Output is typically lower intensity than input; may need rescaling.
        - Best suited for images where colonies are brighter than background.

    Parameter effects:
        - shape: 'disk' preserves rounded colony shapes best; 'diamond' is
          computationally efficient; 'square' may introduce grid artifacts.
        - width: Larger values extract larger bright features. Set slightly
          larger than the maximum size of colonies you want to isolate.

    Examples:
        Basic usage to isolate small bright colonies:

        >>> from phenotypic.enhance import WhiteTophatEnhance
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> image = load_synth_yeast_plate()
        >>> enhancer = WhiteTophatEnhance(shape='disk', width=15)
        >>> enhanced = enhancer.apply(image)
        >>> # detect_mat now contains only small bright structures
        >>> enhanced.detect_mat[:].max() <= image.detect_mat[:].max()
        True

        Using in a pipeline to detect small colonies:

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.enhance import WhiteTophatEnhance
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> image = load_synth_yeast_plate()
        >>> pipeline = ImagePipeline([
        ...     WhiteTophatEnhance(shape='disk', width=20),
        ...     OtsuDetector()
        ... ])
        >>> result = pipeline.apply(image)
    """

    def __init__(self, shape: str = "disk", width: int = None):
        super().__init__()
        self.shape = shape
        self.width = width

    def _operate(self, image: Image) -> Image:
        white_tophat_results = white_tophat(
            image.detect_mat[:],
            footprint=self._get_footprint(
                self._get_footprint_width(detection_matrix=image.detect_mat[:]),
            ),
        )
        image.detect_mat[:] = white_tophat_results

        return image

    def _get_footprint_width(self, detection_matrix: np.ndarray) -> int:
        if self.width is None:
            return int(np.min(detection_matrix.shape) * 0.004)
        else:
            return self.width

    def _get_footprint(self, width: int) -> np.ndarray:
        match self.shape:
            # Use shared ImageEnhancer utility for common 2D shapes
            case "disk" | "square" | "diamond":
                return self._make_footprint(shape=self.shape, width=width)
            case _:
                raise ValueError(f"Unsupported footprint shape: {self.shape}")
