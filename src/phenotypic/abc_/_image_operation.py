from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from phenotypic import Image

import numpy as np
from ._base_operation import BaseOperation
from ._lazy_widget_mixin import LazyWidgetMixin
from ..tools.exceptions_ import InterfaceError, OperationIntegrityError

from abc import ABC, abstractmethod


class ImageOperation(BaseOperation, LazyWidgetMixin, ABC):
    """Core abstract base class for all single-image transformation operations in PhenoTypic.

    ImageOperation is the foundation of PhenoTypic's algorithm system. It defines the
    interface for algorithms that transform an Image object by modifying specific
    components. Unlike GridOperation (which handles grid-aligned operations on plate
    images), ImageOperation acts on a single image independently.

    **What is ImageOperation?**

    ImageOperation manages the distinction between:

    - **apply() method:** The user-facing interface that handles memory management
      (copy vs. in-place) and integrity validation
    - **_operate() method:** The abstract algorithm-specific method that subclasses
      implement with the actual processing logic

    This separation ensures consistent behavior, automatic memory tracking, and
    validation across all image operations.

    **The Operation Hierarchy**

    ImageOperation has four main subclass categories, each modifying different image
    components with different integrity guarantees:

    .. code-block:: text

        ImageOperation (this class)
        ├── ImageEnhancer
        │   └── Modifies ONLY image.enh_gray
        │       ├── GaussianBlur, CLAHE, RankMedianEnhancer, ...
        │       └── Use for: noise reduction, contrast, edge sharpening
        │
        ├── ObjectDetector
        │   └── Modifies ONLY image.objmask and image.objmap
        │       ├── OtsuDetector, CannyDetector, RoundPeaksDetector, ...
        │       └── Use for: discovering and labeling colonies/particles
        │
        ├── ObjectRefiner
        │   └── Modifies ONLY image.objmask and image.objmap
        │       ├── Size filtering, merging, removing objects
        │       └── Use for: cleaning up detection results
        │
        └── ImageCorrector
            └── Modifies ALL image components
                ├── GridAligner, rotation, color correction
                └── Use for: general-purpose transformations

    **When to inherit from each subclass:**

    - **ImageEnhancer:** You only modify ``image.enh_gray`` (enhanced grayscale).
      Original ``image.rgb`` and ``image.gray`` are protected by integrity checks.
      Typical use: preprocessing before detection.

    - **ObjectDetector:** You analyze image data and produce only ``image.objmask``
      (binary mask) and ``image.objmap`` (labeled object map). Input image data is
      protected. Typical use: colony detection, particle finding.

    - **ObjectRefiner:** You edit the mask and map (filtering, merging, removing).
      Input image data is protected. Typical use: post-detection cleanup.

    - **ImageCorrector:** You transform the entire Image (every component may change).
      No integrity checks are performed. Typical use: rotation, alignment, global color
      correction.

    **Never inherit directly from ImageOperation.** Always choose one of the four
    subclasses above, as each provides appropriate integrity validation and shared
    utilities (e.g., ``_make_footprint()`` for morphology operations).

    **How apply() and _operate() work together**

    The user-facing method ``apply(image, inplace=False)`` is the entry point:

    1. **Calls ``_apply_to_single_image()``** with the operation logic
    2. **Handles copy/inplace semantics:**
       - If ``inplace=False`` (default): Image is copied before modification,
         original unchanged
       - If ``inplace=True``: Image is modified in-place for memory efficiency
    3. **Extracts matched parameters** via ``_get_matched_operation_args()``
       - Matches operation instance attributes to ``_operate()`` method parameters
       - Enables parallel execution in pipelines
    4. **Calls your _operate() static method** with the image and matched parameters
    5. **Validates integrity** (subclass-specific via ``@validate_operation_integrity``)
       - Detects unexpected modifications to protected image components
       - Only enabled if ``VALIDATE_OPS=True`` in environment

    Your subclass only needs to implement ``_operate(image, **kwargs) -> Image``.

    **The _operate() method contract**

    ``_operate()`` is a **static method** (required for parallel execution):

    - **Signature:** ``@staticmethod def _operate(image: Image, param1, param2=default) -> Image``
    - **Parameters:** All parameters except ``image`` are automatically matched to
      instance attributes via the ``_get_matched_operation_args()`` system
    - **Behavior:** Modify only the allowed image components (determined by subclass)
    - **Returns:** The modified Image object
    - **Must be static:** This enables serialization and parallel execution

    Example parameter matching:

    .. code-block:: python

        class MyEnhancer(ImageEnhancer):
            def __init__(self, sigma: float):
                super().__init__()
                self.sigma = sigma  # Instance attribute

            @staticmethod
            def _operate(image: Image, sigma: float = 1.0) -> Image:
                # When apply() is called, 'sigma' is automatically passed from self.sigma
                image.enh_gray[:] = gaussian_filter(image.enh_gray[:], sigma=sigma)
                return image

    The ``_apply_to_single_image()`` static method retrieves ``sigma`` from the
    instance (via ``_get_matched_operation_args()``) and passes it to ``_operate()``.

    **Data access through accessors**

    Within ``_operate()``, always access image data through accessors (never direct
    attribute modification). This ensures lazy evaluation, caching, and consistency:

    Reading data:

    - ``image.enh_gray[:]`` - Enhanced grayscale (for enhancers)
    - ``image.rgb[:]`` - Original RGB data
    - ``image.gray[:]`` - Luminance grayscale
    - ``image.objmask[:]`` - Binary object mask
    - ``image.objmap[:]`` - Labeled object map
    - ``image.color.Lab[:]``, ``image.color.HSV[:]`` - Color spaces

    Modifying data:

    - ``image.enh_gray[:] = new_array`` - Set enhanced grayscale
    - ``image.objmask[:] = binary_array`` - Set object mask
    - ``image.objmap[:] = labeled_array`` - Set object map

    **Never do this:**

    .. code-block:: python

        # ✗ WRONG - direct attribute modification
        image.rgb = new_data
        image._enh_gray = new_data
        image.objects_handler.enh_gray = new_data

    **Do this instead:**

    .. code-block:: python

        # ✓ CORRECT - use accessors
        image.enh_gray[:] = new_data
        image.objmask[:] = new_mask

    **Integrity validation with @validate_operation_integrity**

    Intermediate subclasses use the ``@validate_operation_integrity`` decorator to
    enforce that modifications are limited to specific components. For example:

    .. code-block:: python

        class ImageEnhancer(ImageOperation, ABC):
            @validate_operation_integrity('image.rgb', 'image.gray')
            def apply(self, image: Image, inplace=False) -> Image:
                return super().apply(image=image, inplace=inplace)

    This decorator:

    1. Calculates MurmurHash3 signatures of protected arrays **before** ``apply()``
    2. Calls the parent ``apply()`` method
    3. Recalculates signatures **after** operation completes
    4. Raises ``OperationIntegrityError`` if any protected component changed

    Only enabled if ``VALIDATE_OPS=True`` in environment (for performance).

    **Operation chaining and pipelines**

    Operations are designed for method chaining:

    .. code-block:: python

        result = (GaussianBlur(sigma=2).apply(image)
                 .apply_operation(OtsuDetector()))

    Or use ``ImagePipeline`` for multi-step workflows with automatic benchmarking:

    .. code-block:: python

        pipeline = ImagePipeline()
        pipeline.add(GaussianBlur(sigma=2))
        pipeline.add(OtsuDetector())
        pipeline.add(GridFinder())

        results = pipeline.operate([image1, image2, image3])

    **Parallel execution support**

    ImageOperation's static method design enables parallel execution. When
    ``ImagePipeline`` runs with multiple images, it:

    1. Extracts operation parameters via ``_get_matched_operation_args()``
    2. Serializes the operation instance (attributes only)
    3. Sends to worker processes
    4. Workers call ``_apply_to_single_image()`` in parallel

    This is why ``_operate()`` must be static and all parameters must be instance
    attributes matching the method signature.

    Attributes:
        None (all operation state is stored in subclass instances as attributes).

    Methods:
        apply(image, inplace=False): User-facing method that applies the operation.
            Handles copy/inplace logic and parameter matching.
        _operate(image, **kwargs): Abstract static method implemented by subclasses
            with algorithm logic. Parameters are automatically extracted from instance
            attributes via _get_matched_operation_args().
        _apply_to_single_image(cls_name, image, operation, inplace, matched_args):
            Static method that performs the actual apply operation. Handles copy/inplace
            logic and error handling. Called internally by apply(). Also called directly
            by ImagePipeline for parallel execution.

    Notes:
        - **No direct Image attribute modification:** Never write to ``image.rgb``,
          ``image.gray``, or other attributes directly. Use the accessor pattern
          (``image.component[:] = new_data``).

        - **Immutability by default:** Operations return modified copies by default.
          Original image is unchanged unless ``inplace=True`` is explicitly passed.

        - **Static _operate() is required:** The method must be static (decorated
          with ``@staticmethod``) to support parallel execution in pipelines. This
          enables ImagePipeline to serialize operations and execute them in worker
          processes.

        - **Parameter matching for parallelization:** All ``_operate()`` parameters
          (except ``image``) must exist as instance attributes with the same name.
          When ``apply()`` is called, ``_get_matched_operation_args()`` extracts
          these values and passes them to ``_operate()``. This is why subclasses
          store operation parameters as ``self.param_name`` in ``__init__``.

        - **Automatic memory/performance tracking:** BaseOperation (parent class)
          automatically tracks memory usage and execution time when the logger is
          configured for INFO level or higher. Disable by setting logger to WARNING.

        - **Cross-platform compatibility:** Some dependencies (rawpy, pympler) are
          platform-specific. Code must gracefully handle missing optional dependencies.

        - **Integrity validation is optional:** The ``@validate_operation_integrity``
          decorator only runs if ``VALIDATE_OPS=True`` in environment. This provides
          development-time safety without production overhead.

    Examples:
        .. dropdown:: Implementing a custom ImageEnhancer with parameter matching

            .. code-block:: python

                from phenotypic.abc_ import ImageEnhancer
                from phenotypic import Image
                from scipy.ndimage import gaussian_filter

                class GaussianEnhancer(ImageEnhancer):
                    '''Custom enhancer applying Gaussian blur to enh_gray.'''

                    def __init__(self, sigma: float = 1.0):
                        super().__init__()
                        self.sigma = sigma  # Instance attribute matched to _operate()

                    @staticmethod
                    def _operate(image: Image, sigma: float = 1.0) -> Image:
                        '''Apply Gaussian blur to enh_gray.'''
                        # Read enhanced grayscale
                        enh = image.enh_gray[:]

                        # Apply Gaussian filter
                        blurred = gaussian_filter(enh.astype(float), sigma=sigma)

                        # Modify enh_gray through accessor
                        image.enh_gray[:] = blurred.astype(enh.dtype)

                        return image

                # Usage
                enhancer = GaussianEnhancer(sigma=2.5)
                enhanced = enhancer.apply(image)  # Original unchanged
                enhanced_inplace = enhancer.apply(image, inplace=True)  # Original modified

        .. dropdown:: Implementing a custom ObjectDetector

            .. code-block:: python

                from phenotypic.abc_ import ObjectDetector
                from phenotypic import Image
                from skimage.feature import peak_local_max
                from skimage.measure import label as measure_label
                import numpy as np

                class PeakDetector(ObjectDetector):
                    '''Detector using local peak finding to locate colonies.'''

                    def __init__(self, min_distance: int = 10, threshold_abs: int = 100):
                        super().__init__()
                        self.min_distance = min_distance
                        self.threshold_abs = threshold_abs

                    @staticmethod
                    def _operate(image: Image, min_distance: int = 10,
                                 threshold_abs: int = 100) -> Image:
                        '''Find peaks in enh_gray and create object mask/map.'''
                        # Find local maxima (colony peaks)
                        coords = peak_local_max(
                            image.enh_gray[:],
                            min_distance=min_distance,
                            threshold_abs=threshold_abs
                        )

                        # Create binary mask from peaks
                        mask = np.zeros(image.enh_gray.shape, dtype=bool)
                        for y, x in coords:
                            mask[y, x] = True

                        # Create labeled map from mask
                        labeled_map = measure_label(mask)

                        # Set detection results
                        image.objmask[:] = mask
                        image.objmap[:] = labeled_map

                        return image

                # Usage - automatic integrity validation in ImageDetector
                detector = PeakDetector(min_distance=15, threshold_abs=120)
                detected = detector.apply(image)
                colonies = detected.objects
                print(f"Detected {len(colonies)} colonies")

        .. dropdown:: Understanding inplace parameter and memory efficiency

            .. code-block:: python

                from phenotypic.enhance import GaussianBlur
                from phenotypic import Image

                image = Image.from_image_path('colony_plate.jpg')
                enhancer = GaussianBlur(sigma=2.0)

                # Default: inplace=False (safe, creates copy)
                enhanced = enhancer.apply(image)
                print(f"Same object? {id(image) == id(enhanced)}")  # False

                # For memory efficiency with large images
                result = enhancer.apply(image, inplace=True)
                print(f"Same object? {id(image) == id(result)}")  # True

                # inplace=True is useful in pipelines with many large images
                # to minimize memory overhead, but modifies the original

        .. dropdown:: Using operations in a processing pipeline

            .. code-block:: python

                from phenotypic import Image, ImagePipeline
                from phenotypic.enhance import GaussianBlur
                from phenotypic.detect import OtsuDetector
                from phenotypic.grid import GridFinder

                # Load image
                image = Image.from_image_path('colony_plate.jpg')

                # Sequential chaining
                enhanced = GaussianBlur(sigma=2).apply(image)
                detected = OtsuDetector().apply(enhanced)
                grid = GridFinder().apply(detected)

                # Or use ImagePipeline for batch processing
                pipeline = ImagePipeline()
                pipeline.add(GaussianBlur(sigma=2))
                pipeline.add(OtsuDetector())
                pipeline.add(GridFinder())

                # Process multiple images with automatic parallelization
                images = [Image.from_image_path(f) for f in image_files]
                results = pipeline.operate(images)
                # Results are fully processed images

        .. dropdown:: How parameter matching enables parallel execution

            .. code-block:: python

                from phenotypic.abc_ import ImageOperation
                from phenotypic import Image

                class CustomThreshold(ImageOperation):
                    def __init__(self, threshold: int, min_size: int = 5):
                        super().__init__()
                        self.threshold = threshold  # Matched to _operate
                        self.min_size = min_size    # Matched to _operate

                    @staticmethod
                    def _operate(image: Image, threshold: int,
                                 min_size: int = 5) -> Image:
                        # 'threshold' and 'min_size' automatically passed
                        binary = image.enh_gray[:] > threshold
                        image.objmask[:] = binary
                        return image

                # When apply() is called:
                op = CustomThreshold(threshold=100, min_size=10)

                # apply() internally:
                # 1. Calls _get_matched_operation_args()
                # 2. Extracts: {'threshold': 100, 'min_size': 10}
                # 3. Calls _apply_to_single_image(..., matched_args=...)
                # 4. _apply_to_single_image passes kwargs to _operate()

                result = op.apply(image)
    """

    def apply(self, image: Image, inplace=False) -> Image:
        """
        Applies the operation to an image, either in-place or on a copy.

        Args:
            image (Image): The arr image to apply the operation on.
            inplace (bool): If True, modifies the image in place; otherwise,
                operates on a copy of the image.

        Returns:
            Image: The modified image after applying the operation.
        """
        try:
            matched_args = self._get_matched_operation_args()
            image = self._apply_to_single_image(
                    cls_name=self.__class__.__name__,
                    image=image,
                    operation=self._operate,
                    inplace=inplace,
                    matched_args=matched_args,
            )
            return image
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            raise RuntimeError(
                    f"{self.__class__.__name__} failed on image {image.name}: {e}"
            ) from e

    @staticmethod
    @abstractmethod
    def _operate(image: Image) -> Image:
        """
        A placeholder for the main subfunction for an image operator for processing
        image objects.

        This method is called from ImageOperation.apply() and must be implemented in a
        subclass. This allows for checks for data integrity to be made.

        Args:
            image (Image): The image object to be processed by internal operations.

        Raises:
            InterfaceError: Raised if the method is not implemented in a subclass.
        """
        return image

    @staticmethod
    def _apply_to_single_image(cls_name, image, operation, inplace, matched_args):
        """Applies the operation to a single image. this intermediate function is needed for parallel execution."""
        try:
            return operation(image=image if inplace else image.copy(), **matched_args)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            raise Exception(f"{cls_name} failed on image {image.name}: {e}") from e
