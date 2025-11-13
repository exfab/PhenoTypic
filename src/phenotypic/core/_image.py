from __future__ import annotations

from typing import Literal, TYPE_CHECKING

import colour
import numpy as np

if TYPE_CHECKING: from phenotypic import Image

from ._image_parts._image_io_handler import ImageIOHandler


class Image(ImageIOHandler):
    """A comprehensive class for handling image processing, including manipulation, information sync, metadata management, and format conversion.

    The `Image` class is designed to load, process, and manage image data using different
    representation formats (e.g., arrays and matrices). This class allows for metadata editing,
    schema definition, and subcomponent handling to streamline image processing tasks.

    Note:
        - If the arr is 2-D, the ImageHandler leaves the rgb form as empty
        - If the arr is 3-D, the ImageHandler will automatically set the gray component to the grayscale representation.
        - Added in v0.5.0, HSV handling support
    """

    def __init__(self,
                 arr: np.ndarray | Image | None = None,
                 name: str | None = None,
                 bit_depth: Literal[8, 16] | None = None,
                 color_profile='sRGB',
                 gamma_encoding: str | None = 'sRGB',
                 illuminant: str | None = 'D65',
                 observer='CIE 1931 2 Degree Standard Observer'

                 ):
        """
        Initializes an instance of the class with optional attributes for array data,
        name, bit depth, illuminant, color profile, and observer. The class is constructed to handle
        data related to image processing and its various configurations.

        Args:
            arr (np.ndarray | Image | None): An optional array or image object. It
                represents the pixel data or image source.
            name (str | None): An optional name or identifier for the image instance.
            bit_depth (Literal[8, 16] | None): An optional bit depth of the image.
                Either 8 or 16 bits for pixel representation. If None is specified, the bit depth
                will be guessed from the arr dtype. If the arr is a float, the bit_depth will default to 8.
            illuminant (str | None): A string specifying the illuminant standard for
                the image, defaulting to 'D65'.
            color_profile (str): The color profile for the image, defaulting to 'sRGB'.
            observer (str): Observer type in CIE standards, defaulting to 'CIE 1931
                2 Degree Standard Observer'.

        """
        super().__init__(
                arr=arr,
                name=name,
                bit_depth=bit_depth,
                gamma_encoding=gamma_encoding,
                illuminant=illuminant,
                observer=observer,
        )
    
    def interactive_measure(
        self,
        port: int = 8050,
        height: int = 800,
        mode: Literal['inline', 'external', 'jupyterlab'] = 'external',
        detector_type: str = 'otsu'
    ) -> None:
        """Launch an interactive Dash application for measuring object areas.
        
        This method creates and runs an interactive web-based tool that displays
        the image with detected objects and provides:
        - Visual overlay of detected objects
        - Click-to-select individual objects
        - Real-time area measurements
        - Adjustable overlay transparency
        - Export measurements to CSV
        
        The tool is particularly useful for:
        - Validating detection results
        - Measuring colony areas in microbiology images
        - Interactive quality control of image analysis
        - Parameter refinement for detection algorithms
        
        Note:
            Requires optional dependencies: dash and jupyter-dash.
            Install with: pip install phenotypic[interactive]
        
        Args:
            port: Port number for the Dash server. Defaults to 8050.
            height: Height of the image display in pixels. Defaults to 800.
            mode: Display mode - 'inline' (embedded in notebook), 'external' 
                (opens in browser), or 'jupyterlab' (for JupyterLab). 
                Defaults to 'external'.
            detector_type: Type of detector for parameter tuning reference.
                Currently informational only. Defaults to 'otsu'.
        
        Raises:
            ImportError: If dash or jupyter-dash are not installed.
            ValueError: If the image has no grayscale data.
        
        Example:
            >>> import phenotypic as pht
            >>> from phenotypic.detect import OtsuDetector
            >>> 
            >>> # Load and process image
            >>> image = pht.Image.imread('colony_plate.jpg')
            >>> detector = OtsuDetector()
            >>> detector.apply(image)
            >>> 
            >>> # Launch interactive measurement tool
            >>> image.interactive_measure(mode='external')
            >>> 
            >>> # In Jupyter, use inline mode
            >>> image.interactive_measure(mode='inline', height=600)
        
        See Also:
            :class:`phenotypic.tools.InteractiveMeasurementAnalyzer`: The underlying analyzer class.
            :class:`phenotypic.measure.MeasureSize`: Non-interactive area measurement.
        """
        from phenotypic.tools import InteractiveMeasurementAnalyzer
        
        analyzer = InteractiveMeasurementAnalyzer(
            image=self,
            port=port,
            height=height,
            mode=mode,
            detector_type=detector_type
        )
        analyzer.run()