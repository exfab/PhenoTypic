from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from typing_extensions import Callable

if TYPE_CHECKING:
    from phenotypic import Image

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from pandas.api.types import is_scalar
import scipy
import warnings
from functools import partial, wraps

from ._base_operation import BaseOperation
from phenotypic.tools.exceptions_ import OperationFailedError
from phenotypic.tools.funcs_ import validate_measure_integrity
from phenotypic.tools.constants_ import OBJECT
from abc import ABC


def catch_warnings_decorator(func):
    """
    A decorator that catches warnings, prepends the method name to the warning message,
    and reraises the warning.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings(record=True) as recorded_warnings:
            # Call the original function
            warnings.simplefilter("ignore")
            result = func(*args, **kwargs)

            # If any warnings were raised, prepend the method name and reraise
        for warning in recorded_warnings:
            message = f"{func.__name__}: {warning.message}"
            warnings.warn(message, warning.category, stacklevel=2)

        return result

    return wrapper


# <<Interface>>
class MeasureFeatures(BaseOperation, ABC):
    """Extract quantitative measurements from detected colony objects in images.

    MeasureFeatures is the abstract base class for all feature extraction operations
    in PhenoTypic. Unlike ImageOperation classes that return modified images,
    MeasureFeatures subclasses extract numerical measurements from detected objects
    and return pandas DataFrames.

    **Design Principles:**

    This class follows a strict pattern where subclasses implement minimal code:

    1. **__init__:** Define parameters and configuration for your measurement
    2. **_operate(image: Image) -> pd.DataFrame:** Implement your measurement logic
    3. Everything else (type validation, metadata handling, exception handling) is
       handled by the `measure()` method

    This ensures consistent behavior, robust error handling, and automatic memory
    profiling across all measurement operations.

    **How It Works:**

    Users call the public API method `measure(image, include_meta=False)`, which:

    1. Validates input (Image object with detected objects)
    2. Extracts operation parameters using introspection
    3. Calls `_operate()` with matched parameters
    4. Optionally merges image metadata into results
    5. Returns a pandas DataFrame with object labels in the first column

    Subclasses **only** override `_operate()` and `__init__()`. The `measure()`
    method provides automatic validation, exception handling, and metadata merging.

    **Accessing Image Data in _operate():**

    Within your `_operate()` implementation, access image data through accessors
    (lazy-evaluated, cached):

    - **image.gray[:]** - Grayscale intensity values (weighted luminance)
    - **image.enh_gray[:]** - Enhanced grayscale (preprocessed for analysis)
    - **image.objmask[:]** - Binary mask of detected objects (1 = object, 0 = background)
    - **image.objmap[:]** - Labeled integer array (label ID per object, 0 = background)
    - **image.color.Lab[:]** - CIE Lab color space (perceptually uniform)
    - **image.color.XYZ[:]** - CIE XYZ color space
    - **image.color.HSV[:]** - HSV color space (hue, saturation, value)
    - **image.objects** - High-level object interface (iterate with for loop)
    - **image.num_objects** - Count of detected objects

    **DataFrame Output Format:**

    Your `_operate()` method must return a pandas DataFrame with:

    - First column: OBJECT.LABEL (integer object IDs matching image.objmap[:])
    - Subsequent columns: Measurement results (numeric values)
    - One row per detected object

    Example structure:

    .. code-block:: python

        OBJECT.LABEL | Area | MeanIntensity | StdDev
        -----------  |------|---------------|--------
        1            | 1024 | 128.5         | 12.3
        2            | 956  | 135.2         | 14.1
        3            | 1101 | 120.8         | 11.9

    **Static Helper Methods:**

    This class provides 20+ static helper methods for common measurements on
    labeled objects:

    - **Statistical:** _calculate_mean(), _calculate_median(), _calculate_stddev(),
      _calculate_variance(), _calculate_sum(), _calculate_center_of_mass()
    - **Extrema:** _calculate_minimum(), _calculate_maximum(), _calculate_extrema()
    - **Quantiles:** _calculate_q1(), _calculate_q3(), _calculate_iqr()
    - **Advanced:** _calculate_coeff_variation(), _calculate_min_extrema(),
      _calculate_max_extrema()
    - **Custom:** _funcmap2objects() (apply arbitrary functions to labeled regions)
    - **Utility:** _ensure_array() (normalize scalar/array inputs)

    All helpers accept an `objmap` parameter (labeled integer array). If None,
    the entire non-zero region is treated as a single object.

    **Example: Creating a Custom Measurer for Bacterial Colonies**

    .. dropdown:: Implementing a custom colony measurement class

        .. code-block:: python

            from phenotypic.abc_ import MeasureFeatures
            from phenotypic.tools.constants_ import OBJECT
            import pandas as pd
            import numpy as np

            class MeasureCustom(MeasureFeatures):
                \"\"\"Measure custom morphology metrics for microbial colonies.\"\"\"

                def __init__(self, intensity_threshold=100):
                    \"\"\"Initialize with intensity threshold for bright pixels.\"\"\"
                    self.intensity_threshold = intensity_threshold

                def _operate(self, image):
                    \"\"\"Extract bright region area and mean intensity.\"\"\"
                    gray = image.enh_gray[:]
                    objmap = image.objmap[:]

                    # Identify bright pixels within each object
                    bright_mask = gray > self.intensity_threshold

                    # Count bright pixels per object
                    bright_area = self._calculate_sum(
                        array=bright_mask.astype(int),
                        objmap=objmap
                    )

                    # Mean intensity of bright pixels
                    bright_intensity = self._funcmap2objects(
                        func=lambda arr: np.mean(arr[arr > self.intensity_threshold]),
                        out_dtype=float,
                        array=gray,
                        objmap=objmap,
                        default=np.nan
                    )

                    # Create results DataFrame
                    results = pd.DataFrame({
                        'BrightArea': bright_area,
                        'BrightMeanIntensity': bright_intensity,
                    })
                    results.insert(0, OBJECT.LABEL, image.objects.labels2series())
                    return results

            # Usage:
            from phenotypic import Image
            image = Image.from_image_path('colony_plate.jpg')
            # (After detection...)
            measurer = MeasureCustom(intensity_threshold=150)
            measurements = measurer.measure(image)  # Returns DataFrame

    **When to Use MeasureFeatures vs ImageOperation:**

    Use MeasureFeatures when you want to extract **numerical metrics** from objects
    (returns DataFrame). Use ImageOperation (ImageEnhancer, ImageCorrector,
    ObjectDetector) when you want to **modify the image** (returns Image).

    **Microbe Phenotyping Context:**

    In arrayed microbial growth assays, measurements extract colony phenotypes:
    morphology (size, shape, compactness), color (pigmentation, growth medium
    binding), texture (biofilm formation, colony surface roughness), and intensity
    distribution (density variation, heterogeneity). These measurements feed into
    genetic and environmental association studies.

    Attributes:
        No public attributes. Configuration is passed through __init__() parameters.

    Examples:
        .. dropdown:: Basic usage: measure colony area and intensity

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.measure import MeasureSize

                # Load and detect colonies
                image = Image.from_image_path('plate_image.jpg')
                from phenotypic.detect import OtsuDetector
                detector = OtsuDetector()
                image = detector.operate(image)

                # Extract size measurements
                measurer = MeasureSize()
                df = measurer.measure(image)
                print(df)
                # Output:
                #   OBJECT.LABEL  Area  IntegratedIntensity
                # 0             1  1024                 128512
                # 1             2   956                 121232
                # 2             3  1101                 134232

        .. dropdown:: Advanced: extract multiple measurement types with metadata

            .. code-block:: python

                from phenotypic.measure import (
                    MeasureSize,
                    MeasureShape,
                    MeasureColor
                )
                from phenotypic.core import ImagePipeline

                # Create pipeline combining detectors and measurers
                pipeline = ImagePipeline(
                    detector=OtsuDetector(),
                    measurers=[
                        MeasureSize(),
                        MeasureShape(),
                        MeasureColor(include_XYZ=False)
                    ]
                )

                # Measure a single image with metadata
                results = pipeline.operate(image)
                # Combine measurements: merge multiple DataFrames by OBJECT.LABEL
                combined = results[0]
                for df in results[1:]:
                    combined = combined.merge(df, on=OBJECT.LABEL)
    """

    @validate_measure_integrity()
    def measure(self, image: Image, include_meta: bool = False) -> pd.DataFrame:
        """Execute the measurement operation on a detected-object image.

        This is the main public API method for extracting measurements. It handles:
        input validation, parameter extraction via introspection, calling the
        subclass-specific `_operate()` method, optional metadata merging, and
        exception handling.

        **How it works (for users):**

        1. Pass your processed Image (with detected objects) to measure()
        2. The method calls your subclass's _operate() implementation
        3. Results are validated and returned as a pandas DataFrame
        4. If include_meta=True, image metadata (filename, grid info) is merged in

        **How it works (for developers):**

        When you subclass MeasureFeatures, you only implement _operate(). This
        measure() method automatically:

        - Extracts __init__ parameters from your instance (introspection)
        - Passes matched parameters to _operate() as keyword arguments
        - Validates the Image has detected objects (objmap)
        - Wraps exceptions in OperationFailedError with context
        - Merges grid/object metadata if requested

        Args:
            image (Image): A PhenoTypic Image object with detected objects (must
                have non-empty objmap from a prior detection operation).
            include_meta (bool, optional): If True, merge image metadata columns
                (filename, grid position, etc.) into the results DataFrame.
                Defaults to False.

        Returns:
            pd.DataFrame: Measurement results with structure:

                - First column: OBJECT.LABEL (integer IDs from image.objmap[:])
                - Remaining columns: Measurement values (float, int, or string)
                - One row per detected object

                If include_meta=True, additional metadata columns are prepended
                before OBJECT.LABEL (e.g., Filename, GridRow, GridCol).

        Raises:
            OperationFailedError: If _operate() raises any exception, it is caught
                and re-raised as OperationFailedError with details including the
                original exception type, message, image name, and operation class.
                This provides consistent error handling across all measurers.

        Notes:
            - This method is the main entry point; do not override in subclasses
            - Subclasses implement _operate() only, not this method
            - Automatic memory profiling is available via logging configuration
            - Image must have detected objects (image.objmap should be non-empty)

        Examples:
            .. dropdown:: Basic measurement extraction

                .. code-block:: python

                    from phenotypic import Image
                    from phenotypic.measure import MeasureSize
                    from phenotypic.detect import OtsuDetector

                    # Load and detect
                    image = Image.from_image_path('plate.jpg')
                    image = OtsuDetector().operate(image)

                    # Extract measurements
                    measurer = MeasureSize()
                    df = measurer.measure(image)
                    print(df.head())

            .. dropdown:: Include metadata in measurements

                .. code-block:: python

                    # With image metadata (filename, grid info)
                    df_with_meta = measurer.measure(image, include_meta=True)
                    print(df_with_meta.columns)
                    # Output: ['Filename', 'GridRow', 'GridCol', 'OBJECT.LABEL',
                    #          'Area', 'IntegratedIntensity', ...]
        """

        try:
            matched_args = self._get_matched_operation_args()

            meas = self._operate(image, **matched_args)
            if include_meta:
                meta = (
                    image.grid.info(include_metadata=True)
                    if hasattr(image, "grid")
                    else image.objects.info(include_metadata=True)
                )
                meas = meta.merge(meas, on=OBJECT.LABEL)

            return meas

        except Exception as e:
            raise OperationFailedError(
                operation=self.__class__.__name__,
                image_name=image.name,
                err_type=type(e),
                message=str(e),
            )

    @staticmethod
    @abc.abstractmethod
    def _operate(image: Image) -> pd.DataFrame:
        return pd.DataFrame()

    @staticmethod
    def _ensure_array(value) -> np.ndarray:
        """Ensure value is a numpy array (handle both scalars and arrays uniformly).

        This utility method normalizes different input types (scalars, lists, tuples,
        already-array values) into a consistent numpy array format. Useful internally
        by helper methods to ensure consistent return types and by custom measurers
        that need uniform handling of computation results.

        Args:
            value: Input value (scalar, list, tuple, array, or any array-like).
                Can be float, int, bool, or numpy-compatible type.

        Returns:
            np.ndarray: Converted numpy array. Preserves dtype of input when
                possible. Scalars become 0-d arrays.

        Notes:
            This is primarily used internally by _calculate_*() helper methods to
            ensure that both single-object results and multi-object arrays are
            returned in consistent numpy array format.

        Examples:
            .. dropdown:: Normalize various input types

                .. code-block:: python

                    # Scalar to array
                    result = MeasureFeatures._ensure_array(5)
                    print(type(result), result.shape)
                    # Output: <class 'numpy.ndarray'> ()

                    # List to array
                    result = MeasureFeatures._ensure_array([1, 2, 3])
                    print(result, result.shape)
                    # Output: [1 2 3] (3,)

                    # Already array
                    arr = np.array([1.0, 2.0, 3.0])
                    result = MeasureFeatures._ensure_array(arr)
                    print(np.array_equal(result, arr))
                    # Output: True
        """
        if is_scalar(value):
            return np.asarray(value)
        else:
            return np.asarray(value)

    @staticmethod
    @catch_warnings_decorator
    def _calculate_center_of_mass(array: np.ndarray, objmap: ArrayLike = None):
        """Calculate the weighted center of mass for each labeled object.

        Computes the coordinate position (row, column) of the geometric center of
        mass for each labeled object. The center of mass is the intensity-weighted
        average position - objects with higher intensity values contribute more to
        the center location. Useful for tracking colony center position or detecting
        eccentricity in colony growth patterns.

        Args:
            array (np.ndarray): 2D intensity array (e.g., grayscale image or
                intensity within object regions). Values are weights; higher
                intensities shift center toward those pixels.
            objmap (ArrayLike, optional): 2D labeled integer array with same shape
                as `array`. Each unique positive integer (1, 2, 3...) identifies an
                object; 0 is background. If None, all non-zero elements of `array`
                are treated as a single object. Defaults to None.

        Returns:
            np.ndarray: If objmap provided: array of shape (num_objects, 2) with
                (row, col) coordinates for each object center of mass. If objmap=None:
                single (row, col) tuple. Coordinates are floating-point (subpixel
                precision possible).

        Notes:
            - Input array values act as weights; zero/near-zero values have minimal
              influence on center position
            - Useful for detecting colonies with uneven intensity distribution
            - Center coordinates may fall outside object pixels (edge effect)
            - For equal-weighted center, use uniform-intensity array with same shape

        Examples:
            .. dropdown:: Find colony centroid positions

                .. code-block:: python

                    # Measure intensity-weighted centers for stained colonies
                    gray = image.enh_gray[:]  # Preprocessed intensity
                    objmap = image.objmap[:]

                    centers = MeasureFeatures._calculate_center_of_mass(
                        array=gray,
                        objmap=objmap
                    )
                    # Returns: [[42.3, 105.8], [156.2, 89.5], ...]
                    # (row, col) for each object

                    # Detect colonies with off-center intensity (asymmetric growth)
                    for i, (row, col) in enumerate(centers):
                        obj_label = i + 1
                        # Analyze intensity distribution around center

            .. dropdown:: Single colony mass center (no objmap)

                .. code-block:: python

                    # Find center for isolated colony region
                    colony_region = image.enh_gray[50:150, 50:150]
                    center = MeasureFeatures._calculate_center_of_mass(colony_region)
                    # Returns: (42.5, 52.3) - single (row, col) tuple
        """
        if objmap is not None:
            indexes = np.unique(objmap)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._ensure_array(
            scipy.ndimage.center_of_mass(array, objmap, index=indexes)
        )

    @staticmethod
    @catch_warnings_decorator
    def _calculate_maximum(array: np.ndarray, objmap: ArrayLike = None):
        """Calculate the maximum pixel value within each labeled object.

        Finds the brightest (highest intensity) pixel within each object region.
        Useful for detecting bright spots in colonies (spore centers, pigment
        accumulation) or evaluating maximum growth intensity.

        Args:
            array (np.ndarray): 2D intensity array (e.g., grayscale or color channel).
            objmap (ArrayLike, optional): 2D labeled integer array. Each positive
                integer (1, 2, 3...) identifies an object; 0 is background. If None,
                all non-zero elements of `array` are treated as a single object.
                Defaults to None.

        Returns:
            np.ndarray: 1D array of maximum values, one per object. If objmap=None,
                returns scalar (0-d array).

        Notes:
            - Returns NaN for objects with all-zero pixels
            - Sensitive to image noise; consider preprocessing with smoothing first
            - Useful with binary masks to count bright-region pixels

        Examples:
            .. dropdown:: Find brightest pixels in colonies

                .. code-block:: python

                    gray = image.enh_gray[:]
                    objmap = image.objmap[:]
                    max_intensity = MeasureFeatures._calculate_maximum(gray, objmap)
                    # Returns: [245, 238, 241, ...] brightness per object
        """
        if objmap is not None:
            indexes = np.unique(objmap)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._ensure_array(
            scipy.ndimage.maximum(array, objmap, index=indexes)
        )

    @staticmethod
    @catch_warnings_decorator
    def _calculate_mean(array: np.ndarray, objmap: ArrayLike = None):
        """Calculate the mean (average) intensity for each labeled object.

        Computes the arithmetic mean of all pixels within each object region.
        Primary measure of overall colony brightness/density. In microbial
        phenotyping, mean intensity correlates with growth density, pigmentation,
        and biofilm formation.

        Args:
            array (np.ndarray): 2D intensity array (grayscale, color channel, etc.).
            objmap (ArrayLike, optional): 2D labeled integer array (same shape as
                array). Positive integers identify objects; 0 is background. If None,
                all non-zero array elements are treated as single object.
                Defaults to None.

        Returns:
            np.ndarray: 1D array of mean values (float), one per object. If objmap=None,
                returns scalar (0-d array). Returns NaN for empty objects.

        Notes:
            - Most commonly used intensity statistic
            - Not robust to outliers; use median for outlier-resistant average
            - Zero/background pixels excluded from calculation

        Examples:
            .. dropdown:: Compare colony growth intensity

                .. code-block:: python

                    gray = image.enh_gray[:]
                    objmap = image.objmap[:]
                    mean_intensities = MeasureFeatures._calculate_mean(gray, objmap)
                    # Returns: [128.5, 142.3, 135.8, ...] avg brightness per colony
                    # High values = dense/pigmented growth
        """
        if objmap is not None:
            indexes = np.unique(objmap)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._ensure_array(
            scipy.ndimage.mean(array, objmap, index=indexes)
        )

    @staticmethod
    @catch_warnings_decorator
    def _calculate_median(array: np.ndarray, objmap: ArrayLike = None):
        """Calculate the median intensity for each labeled object.

        Finds the middle value (50th percentile) of all pixels in each object.
        More robust to outliers than mean - useful when colonies contain bright
        specks, debris, or uneven staining. Also useful for binary masks (median
        of 0/1 gives fraction threshold).

        Args:
            array (np.ndarray): 2D intensity array.
            objmap (ArrayLike, optional): 2D labeled integer array (same shape).
                Positive integers = objects, 0 = background. If None, all non-zero
                array elements treated as single object. Defaults to None.

        Returns:
            np.ndarray: 1D array of median values, one per object. If objmap=None,
                returns scalar. Returns NaN for empty objects.

        Notes:
            - Robust to outliers and extreme values (unlike mean)
            - Slower computation than mean (requires sorting)
            - Good for colonies with speckling or uneven stain distribution

        Examples:
            .. dropdown:: Robust growth intensity measurement

                .. code-block:: python

                    gray = image.enh_gray[:]
                    objmap = image.objmap[:]
                    # For colonies with bright debris or speckles
                    median_int = MeasureFeatures._calculate_median(gray, objmap)
                    # More stable than mean for noisy images
        """
        if objmap is not None:
            indexes = np.unique(objmap)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._ensure_array(
            scipy.ndimage.median(array, objmap, index=indexes)
        )

    @staticmethod
    @catch_warnings_decorator
    def _calculate_minimum(array: np.ndarray, objmap: ArrayLike = None):
        """Calculate the minimum pixel value within each labeled object.

        Finds the darkest (lowest intensity) pixel within each object region.
        Useful for detecting thin regions, detecting edge pixels, or checking
        for completely black (dead) regions within colonies.

        Args:
            array (np.ndarray): 2D intensity array (grayscale or channel).
            objmap (ArrayLike, optional): 2D labeled integer array (same shape).
                Positive integers identify objects; 0 is background. If None,
                all non-zero array elements treated as single object.
                Defaults to None.

        Returns:
            np.ndarray: 1D array of minimum values, one per object. If objmap=None,
                returns scalar (0-d array).

        Notes:
            - Returns NaN for objects with all-zero pixels
            - Sensitive to shadows, indentations, or dark contaminants
            - Combined with maximum, reveals intensity range within colony

        Examples:
            .. dropdown:: Detect dark regions in colonies

                .. code-block:: python

                    gray = image.enh_gray[:]
                    objmap = image.objmap[:]
                    min_int = MeasureFeatures._calculate_minimum(gray, objmap)
                    # Identifies colonies with dark cores (aging or death)
        """
        if objmap is not None:
            indexes = np.unique(objmap)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._ensure_array(
            scipy.ndimage.minimum(array, objmap, index=indexes)
        )

    @staticmethod
    @catch_warnings_decorator
    def _calculate_stddev(array: np.ndarray, objmap: ArrayLike = None):
        """Calculate the standard deviation of pixel values within each object.

        Measures intensity variation within each colony region. High stddev
        indicates uneven growth, texture, or biofilm formation. Low stddev
        indicates uniform, homogeneous colonies.

        Args:
            array (np.ndarray): 2D intensity array.
            objmap (ArrayLike, optional): 2D labeled integer array (same shape).
                Positive integers = objects, 0 = background. If None, all
                non-zero array elements treated as single object.
                Defaults to None.

        Returns:
            np.ndarray: 1D array of standard deviations, one per object. If
                objmap=None, returns scalar.

        Notes:
            - Zero for perfectly uniform colonies
            - High values suggest rough surfaces, biofilms, or sectoring
            - Useful for phenotyping mixed cultures or detecting morphological
              mutants with rougher colony surfaces

        Examples:
            .. dropdown:: Detect rough/textured colonies

                .. code-block:: python

                    gray = image.enh_gray[:]
                    objmap = image.objmap[:]
                    stddev = MeasureFeatures._calculate_stddev(gray, objmap)
                    # High stddev = rough/textured colonies
                    # Low stddev = smooth/uniform growth
                    rough_colonies = np.where(stddev > 20)[0]
        """
        if objmap is not None:
            indexes = np.unique(objmap)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._ensure_array(
            scipy.ndimage.standard_deviation(array, objmap, index=indexes)
        )

    @staticmethod
    @catch_warnings_decorator
    def _calculate_sum(array: np.ndarray, objmap: ArrayLike = None):
        """Calculate the sum of all pixel values within each labeled object.

        Integrates intensity across entire object region. For binary masks
        (array=objmask), this equals the number of pixels (area). For intensity
        arrays, gives total accumulated intensity (used for "integrated intensity"
        in colony phenotyping). Proportional to both colony size and density.

        Args:
            array (np.ndarray): 2D value array (intensity, binary mask, etc.).
            objmap (ArrayLike, optional): 2D labeled integer array (same shape).
                Positive integers identify objects; 0 is background. If None,
                all non-zero array elements treated as single object.
                Defaults to None.

        Returns:
            np.ndarray: 1D array of sums, one per object. If objmap=None,
                returns scalar (0-d array).

        Notes:
            - On binary mask (0/1): sum = area (pixel count)
            - On grayscale: sum = total brightness (size * mean intensity)
            - Sensitive to both object size and intensity
            - No division by area (unlike mean), so larger objects always sum higher

        Examples:
            .. dropdown:: Calculate colony area and integrated intensity

                .. code-block:: python

                    # Area = sum of binary mask
                    area = MeasureFeatures._calculate_sum(image.objmask[:],
                                                         image.objmap[:])
                    # Returns pixel count per colony

                    # Integrated intensity = sum of grayscale
                    intensity = MeasureFeatures._calculate_sum(image.gray[:],
                                                              image.objmap[:])
                    # Returns total brightness per colony
        """
        if objmap is not None:
            indexes = np.unique(objmap)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._ensure_array(
            scipy.ndimage.sum_labels(array, objmap, index=indexes)
        )

    @staticmethod
    @catch_warnings_decorator
    def _calculate_variance(array: np.ndarray, objmap: ArrayLike = None):
        """Calculate the variance (stddev squared) of pixel values in each object.

        Measures intensity spread within each colony. Variance = stddev^2. Used
        in statistical analysis and texture characterization. Higher variance
        indicates more heterogeneous growth or surface texture.

        Args:
            array (np.ndarray): 2D intensity array.
            objmap (ArrayLike, optional): 2D labeled integer array (same shape).
                Positive integers = objects, 0 = background. If None, all
                non-zero array elements treated as single object.
                Defaults to None.

        Returns:
            np.ndarray: 1D array of variances, one per object. If objmap=None,
                returns scalar.

        Notes:
            - Variance = stddev^2; variance is useful for statistical tests
            - Units are (intensity)^2, making interpretation less intuitive
            - Preferred over stddev in statistical/ML contexts
            - Zero for perfectly uniform colonies

        Examples:
            .. dropdown:: Analyze colony texture variability

                .. code-block:: python

                    gray = image.enh_gray[:]
                    objmap = image.objmap[:]
                    variance = MeasureFeatures._calculate_variance(gray, objmap)
                    stddev = np.sqrt(variance)  # Convert back to original units
        """
        if objmap is not None:
            indexes = np.unique(objmap)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        return MeasureFeatures._ensure_array(
            scipy.ndimage.variance(array, objmap, index=indexes)
        )

    @staticmethod
    @catch_warnings_decorator
    def _calculate_coeff_variation(array: np.ndarray, objmap: ArrayLike = None):
        """Calculate the unbiased coefficient of variation (CV) for each object.

        The coefficient of variation measures relative intensity variability,
        normalized by the mean. CV = stddev / mean * 100%. Unbiased estimator
        accounts for sample size. Useful for comparing texture across colonies
        of different sizes or intensities. Lower CV = more uniform/homogeneous;
        higher CV = more heterogeneous/textured (biofilm, sectoring).

        Args:
            array (np.ndarray): 2D intensity array (grayscale or channel).
            objmap (ArrayLike, optional): 2D labeled integer array (same shape).
                Positive integers identify objects; 0 is background. If None,
                calculation returns NaN (requires object counts for unbiasing).
                Defaults to None.

        Returns:
            np.ndarray: 1D array of unbiased CV values (dimensionless ratio),
                one per object. If objmap=None, returns NaN. Values typically
                range 0.0 to 1.0+ (can exceed 1 for highly variable colonies).

        Notes:
            - Unbiased estimator: CV_unbiased = (1 + 1/n) * CV_biased
            - Dimensionless (unlike stddev), allows cross-sample comparison
            - Requires labeled objects (objmap); cannot compute for single-object mode
            - Returns NaN if objmap=None (need object counts for unbiasing)
            - Colonies with CV < 0.1 are very uniform
            - Colonies with CV > 0.3 show significant texture/heterogeneity

        References:
            - https://en.wikipedia.org/wiki/Coefficient_of_variation
            - Unbiased estimator: (1 + 1/n) * biased_cv

        Examples:
            .. dropdown:: Compare colony growth uniformity

                .. code-block:: python

                    gray = image.enh_gray[:]
                    objmap = image.objmap[:]
                    cv = MeasureFeatures._calculate_coeff_variation(gray, objmap)
                    # Returns: [0.08, 0.15, 0.32, ...] relative variation
                    # CV < 0.1: smooth, uniform colonies
                    # CV > 0.2: rough, textured growth (biofilm)
                    textured = np.where(cv > 0.2)[0]  # Find biofilm-like colonies
        """
        if objmap is not None:
            unique_labels, unique_counts = np.unique(objmap, return_counts=True)
            unique_counts = unique_counts[unique_labels != 0]
            biased_cv = MeasureFeatures._calculate_stddev(
                array, objmap
            ) / MeasureFeatures._calculate_mean(array, objmap)
            result = (1 + (1 / unique_counts)) * biased_cv
        else:
            # For the case when objmap is None, we can't calculate the coefficient of variation
            # because we need the counts of each label
            result = np.nan
        return MeasureFeatures._ensure_array(result)

    @staticmethod
    def _calculate_extrema(array: np.ndarray, objmap: ArrayLike = None):
        """Internal helper: calculate min/max values and positions for labeled objects.

        Returns tuple of (min_values, max_values, min_positions, max_positions).
        Mostly used internally by _calculate_min_extrema() and
        _calculate_max_extrema(). Use those public methods instead.

        Args:
            array (np.ndarray): 2D intensity array.
            objmap (ArrayLike, optional): 2D labeled integer array (same shape).
                Positive integers identify objects; 0 is background. If None,
                all non-zero array elements treated as single object.
                Defaults to None.

        Returns:
            tuple: (min_vals, max_vals, min_pos, max_pos) where each is numpy array
                of shape (num_objects, 2) with (row, col) coordinates for extrema
                or (num_objects,) for values.
        """
        if objmap is not None:
            indexes = np.unique(objmap)
            indexes = indexes[indexes != 0]
        else:
            indexes = None
        min_extrema, max_extrema, min_pos, max_pos = MeasureFeatures._ensure_array(
            scipy.ndimage.extrema(array, objmap, index=indexes)
        )
        return (
            MeasureFeatures._ensure_array(min_extrema),
            MeasureFeatures._ensure_array(max_extrema),
            MeasureFeatures._ensure_array(min_pos),
            MeasureFeatures._ensure_array(max_pos),
        )

    @staticmethod
    @catch_warnings_decorator
    def _calculate_min_extrema(array: np.ndarray, objmap: ArrayLike = None):
        """Find the darkest pixel and its (row, col) location in each object.

        Identifies the minimum intensity value and the spatial coordinate where
        that minimum occurs. Useful for detecting dark spots, shadows, or low-
        intensity regions within colonies. Combined with max_extrema, reveals
        intensity range and spatial distribution of brightness.

        Args:
            array (np.ndarray): 2D intensity array.
            objmap (ArrayLike, optional): 2D labeled integer array (same shape).
                Positive integers = objects, 0 = background. If None, all
                non-zero array elements treated as single object.
                Defaults to None.

        Returns:
            tuple: (min_values, min_positions) where:

                - min_values: 1D array of minimum intensity per object (float)
                - min_positions: Array of (row, col) coordinates per object
                  (shape depends on objmap)

        Notes:
            - Positions are pixel coordinates (row, col)
            - Useful for detecting dark colonies or dead regions
            - May return multiple positions if minimum occurs multiple times

        Examples:
            .. dropdown:: Locate dark regions in colonies

                .. code-block:: python

                    gray = image.enh_gray[:]
                    objmap = image.objmap[:]
                    min_vals, min_pos = MeasureFeatures._calculate_min_extrema(
                        gray, objmap
                    )
                    # min_vals: [50, 45, 52, ...] darkest pixel per colony
                    # min_pos: [(45, 103), (156, 87), ...] locations
        """
        min_extrema, _, min_pos, _ = MeasureFeatures._calculate_extrema(array, objmap)
        return min_extrema, min_pos

    @staticmethod
    @catch_warnings_decorator
    def _calculate_max_extrema(array: np.ndarray, objmap: ArrayLike = None):
        """Find the brightest pixel and its (row, col) location in each object.

        Identifies the maximum intensity value and the spatial coordinate where
        that maximum occurs. Useful for detecting bright spores, pigment centers,
        or regions of highest colony density. Combined with min_extrema, reveals
        intensity range and spatial variation within colonies.

        Args:
            array (np.ndarray): 2D intensity array.
            objmap (ArrayLike, optional): 2D labeled integer array (same shape).
                Positive integers identify objects; 0 is background. If None,
                all non-zero array elements treated as single object.
                Defaults to None.

        Returns:
            tuple: (max_values, max_positions) where:

                - max_values: 1D array of maximum intensity per object (float)
                - max_positions: Array of (row, col) coordinates per object

        Notes:
            - Positions are pixel coordinates (row, col)
            - Useful for locating spore centers, pigmentation hotspots
            - Bright colonies have high max values; dim colonies have low max values
            - Multiple positions possible if maximum occurs multiple times

        Examples:
            .. dropdown:: Find brightest spots in colonies

                .. code-block:: python

                    gray = image.enh_gray[:]
                    objmap = image.objmap[:]
                    max_vals, max_pos = MeasureFeatures._calculate_max_extrema(
                        gray, objmap
                    )
                    # max_vals: [245, 238, 241, ...] brightest pixel per colony
                    # max_pos: [(42, 105), (155, 89), ...] center locations
        """
        _, max_extreme, _, max_pos = MeasureFeatures._calculate_extrema(array, objmap)
        return max_extreme, max_pos

    @staticmethod
    def _funcmap2objects(
        func: Callable,
        out_dtype: np.dtype,
        array: np.ndarray,
        objmap: ArrayLike = None,
        default: int | float | np.nan = np.nan,
        pass_positions: bool = False,
    ):
        """Apply a custom function to each labeled region in an array (advanced helper).

        Powerful utility for computing arbitrary measurements on labeled objects.
        Applies your custom function to the pixels within each object and returns
        aggregated results. Useful for custom statistics not covered by built-in
        helpers (percentiles, custom transformations, etc.).

        Internally uses scipy.ndimage.labeled_comprehension, which iterates over
        unique object labels and extracts the corresponding pixels, passes them
        to your function, and collects results.

        Args:
            func (Callable): Function to apply to each object's pixels. Signature:
                `func(pixel_array) -> scalar`. If pass_positions=True:
                `func(pixel_array, pixel_positions) -> scalar`. Examples:
                np.mean, np.percentile (use partial for arguments).
            out_dtype (np.dtype): Output array data type (float64, int32, etc.).
                Affects precision and memory usage.
            array (np.ndarray): 2D input array (intensity, counts, etc.).
            objmap (ArrayLike, optional): 2D labeled integer array (same shape as
                array). Positive integers identify objects; 0 is background. If None,
                all non-zero array elements are treated as a single object.
                Defaults to None.
            default (int | float | np.nan, optional): Value to assign to labels
                not found in objmap (e.g., skipped background). Defaults to np.nan.
            pass_positions (bool, optional): If True, pass (pixel_array, positions)
                to func instead of just pixel_array. Positions are (row, col) tuples
                of non-zero pixels. Defaults to False.

        Returns:
            np.ndarray: 1D array of results (one per object). Output dtype matches
                out_dtype parameter. Returns scalar if single object.

        Notes:
            - Powerful for custom measurements beyond standard statistics
            - Can be slow for large images; vectorize if possible
            - Use functools.partial() to fix function parameters
            - Positions only available if pass_positions=True (slower)
            - func() receives actual pixel values, not labeled indices

        Examples:
            .. dropdown:: Custom measurement: find 90th percentile intensity

                .. code-block:: python

                    from functools import partial

                    gray = image.enh_gray[:]
                    objmap = image.objmap[:]

                    # Define custom function
                    q90 = partial(np.percentile, q=90)
                    results = MeasureFeatures._funcmap2objects(
                        func=q90,
                        out_dtype=float,
                        array=gray,
                        objmap=objmap,
                        default=np.nan
                    )
                    # Returns: [200, 195, 210, ...] 90th percentile per colony

            .. dropdown:: Custom: count pixels above threshold

                .. code-block:: python

                    # Count bright pixels within each colony
                    def count_bright(pixels):
                        return np.sum(pixels > 200)

                    bright_pixels = MeasureFeatures._funcmap2objects(
                        func=count_bright,
                        out_dtype=int,
                        array=gray,
                        objmap=objmap,
                        default=0
                    )
                    # Returns: [45, 32, 58, ...] bright pixel count per colony

            .. dropdown:: Advanced: use positions for spatial analysis

                .. code-block:: python

                    # Find distance from top-left for brightest pixel
                    def distance_to_topleft(pixels, positions):
                        if len(pixels) == 0:
                            return np.nan
                        brightest_idx = np.argmax(pixels)
                        row, col = positions[brightest_idx]
                        return np.sqrt(row**2 + col**2)

                    distances = MeasureFeatures._funcmap2objects(
                        func=distance_to_topleft,
                        out_dtype=float,
                        array=gray,
                        objmap=objmap,
                        pass_positions=True,
                        default=np.nan
                    )
        """
        if objmap is not None:
            index = np.unique(objmap)
            index = index[index != 0]
        else:
            index = None

        return MeasureFeatures._ensure_array(
            scipy.ndimage.labeled_comprehension(
                input=array,
                labels=objmap,
                index=index,
                func=func,
                out_dtype=out_dtype,
                pass_positions=pass_positions,
                default=default,
            ),
        )

    @staticmethod
    @catch_warnings_decorator
    def _calculate_q1(array, objmap=None, method: str = "linear"):
        """Calculate the first quartile (Q1, 25th percentile) for each object.

        Finds the value below which 25% of pixels fall. Less sensitive to extremes
        than mean/median, useful for characterizing lower intensity tail of
        distribution (sparse/background regions within colonies).

        Args:
            array (np.ndarray): 2D intensity array.
            objmap (ArrayLike, optional): 2D labeled integer array. Defaults to None.
            method (str, optional): Quantile interpolation method ('linear',
                'lower', 'higher', 'nearest', 'midpoint'). See numpy.quantile docs.
                Defaults to 'linear'.

        Returns:
            np.ndarray: 1D array of Q1 values, one per object.

        Notes:
            - Q1, Q2 (median), Q3 together form quartile summary
            - Low Q1 suggests colonies with sparse pixel coverage
            - Used to compute IQR (Q3 - Q1)
        """
        find_q1 = partial(np.quantile, q=0.25, method=method)
        q1 = MeasureFeatures._funcmap2objects(
            func=find_q1,
            out_dtype=array.dtype,
            array=array,
            objmap=objmap,
            default=np.nan,
            pass_positions=False,
        )
        return MeasureFeatures._ensure_array(q1)

    @staticmethod
    @catch_warnings_decorator
    def _calculate_q3(array, objmap=None, method: str = "linear"):
        """Calculate the third quartile (Q3, 75th percentile) for each object.

        Finds the value below which 75% of pixels fall. Characterizes upper
        intensity distribution (bright regions, dense growth areas). Together
        with Q1, defines the interquartile range (IQR).

        Args:
            array (np.ndarray): 2D intensity array.
            objmap (ArrayLike, optional): 2D labeled integer array. Defaults to None.
            method (str, optional): Quantile interpolation method ('linear',
                'lower', 'higher', 'nearest', 'midpoint'). Defaults to 'linear'.

        Returns:
            np.ndarray: 1D array of Q3 values, one per object.

        Notes:
            - High Q3 indicates colonies with bright, densely-packed pixels
            - IQR = Q3 - Q1 (robust measure of spread)
            - Used with Q1 to identify outlier intensities
        """
        find_q3 = partial(np.quantile, q=0.75, method=method)
        q3 = MeasureFeatures._funcmap2objects(
            func=find_q3,
            out_dtype=array.dtype,
            array=array,
            objmap=objmap,
            default=np.nan,
            pass_positions=False,
        )
        return MeasureFeatures._ensure_array(q3)

    @staticmethod
    @catch_warnings_decorator
    def _calculate_iqr(
        array, objmap=None, method: str = "linear", nan_policy: str = "omit"
    ):
        """Calculate the interquartile range (Q3 - Q1) for each object.

        Measures the spread of the middle 50% of intensities. Robust to outliers
        (unlike range or variance). Used for detecting intensity outliers and
        characterizing colony texture uniformity. Low IQR = narrow intensity range
        (uniform); high IQR = broad range (variable).

        Args:
            array (np.ndarray): 2D intensity array.
            objmap (ArrayLike, optional): 2D labeled integer array. Defaults to None.
            method (str, optional): Quantile interpolation method. Defaults to 'linear'.
            nan_policy (str, optional): How to handle NaN values ('omit', 'propagate').
                Defaults to 'omit'.

        Returns:
            np.ndarray: 1D array of IQR values, one per object.

        Notes:
            - IQR = Q3 - Q1; always non-negative
            - Robust measure; unaffected by extreme outliers
            - Colonies with IQR < 30: uniform intensity distribution
            - Colonies with IQR > 100: highly variable (texture, biofilm)
            - Preferred over stddev/variance for non-normal distributions

        Examples:
            .. dropdown:: Compare colony texture via IQR

                .. code-block:: python

                    gray = image.enh_gray[:]
                    objmap = image.objmap[:]
                    iqr = MeasureFeatures._calculate_iqr(gray, objmap)
                    # IQR < 30: smooth, uniform colonies
                    # IQR > 100: rough, textured growth
        """
        find_iqr = partial(
            scipy.stats.iqr, axis=None, nan_policy=nan_policy, interpolation=method
        )
        return MeasureFeatures._ensure_array(
            MeasureFeatures._funcmap2objects(
                func=find_iqr,
                out_dtype=array.dtype,
                array=array,
                objmap=objmap,
                default=np.nan,
                pass_positions=False,
            ),
        )
