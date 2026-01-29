from __future__ import annotations

from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from phenotypic import Image, GridImage

from ._base_operation import BaseOperation
from phenotypic.tools_.mixin import LazyWidgetMixin

from abc import ABC, abstractmethod
import traceback


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
    3. **Calls your _operate() instance method** with the image
    4. **Validates integrity** (subclass-specific via ``@validate_operation_integrity``)
       - Detects unexpected modifications to protected image components
       - Only enabled if ``VALIDATE_OPS=True`` in environment

    Your subclass only needs to implement ``_operate(self, image) -> Image``.

    **The _operate() method contract**

    ``_operate()`` is an **instance method** (no ``@staticmethod`` decorator):

    - **Signature:** ``def _operate(self, image: Image) -> Image:``
    - **Parameters:** Access operation parameters directly via ``self.param_name``
    - **Behavior:** Modify only the allowed image components (determined by subclass)
    - **Returns:** The modified Image object

    Example implementation:

    .. code-block:: python

        class MyEnhancer(ImageEnhancer):
            def __init__(self, sigma: float):
                super().__init__()
                self.sigma = sigma  # Instance attribute

            def _operate(self, image: Image) -> Image:
                # Access parameters via self
                image.enh_gray[:] = gaussian_filter(image.enh_gray[:], sigma=self.sigma)
                return image

    The instance method pattern is simpler and more Pythonic than the old static method approach.

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

    ImageOperation supports parallel execution through operation serialization. When
    ``ImagePipeline`` runs with multiple images, it:

    1. Serializes the operation instance with all attributes (``op.__dict__``)
    2. Sends the pickled operation to worker processes
    3. Workers unpickle the operation (restoring all ``self.param`` values)
    4. Workers call ``operation.apply(image)`` which invokes ``_operate(self, image)``

    Instance methods work perfectly with parallel execution because the entire
    operation object (with all parameters) is serialized together.

    Attributes:
        None (all operation state is stored in subclass instances as attributes).

    Methods:
        apply(image, inplace=False): User-facing method that applies the operation.
            Handles copy/inplace logic, calls ``_operate()``, and validates integrity.
        _operate(self, image): Abstract instance method implemented by subclasses
            with algorithm logic. Access parameters via ``self.param_name``.
        _apply_to_single_image(cls_name, image, operation, inplace):
            Static helper method that performs the actual apply operation. Handles
            copy/inplace logic and error handling. Called internally by apply().
            Also used by ImagePipeline for parallel execution.

    Notes:
        - **No direct Image attribute modification:** Never write to ``image.rgb``,
          ``image.gray``, or other attributes directly. Use the accessor pattern
          (``image.component[:] = new_data``).

        - **Immutability by default:** Operations return modified copies by default.
          Original image is unchanged unless ``inplace=True`` is explicitly passed.

        - **Instance method pattern:** The ``_operate()`` method should be an instance
          method (no ``@staticmethod`` decorator). Access operation parameters directly
          via ``self.param_name``. This is simpler and more Pythonic than the old
          static method approach.

        - **Parallel execution compatibility:** Instance methods work seamlessly with
          parallel execution. Operations are serialized with all instance attributes
          (``op.__dict__``) and unpickled in worker processes with full state restored.

        - **Automatic memory/performance tracking:** BaseOperation (parent class)
          automatically tracks memory usage and execution time when the logger is
          configured for INFO level or higher. Disable by setting logger to WARNING.

        - **Cross-platform compatibility:** Some dependencies (rawpy, pympler) are
          platform-specific. Code must gracefully handle missing optional dependencies.

        - **Integrity validation is optional:** The ``@validate_operation_integrity``
          decorator only runs if ``VALIDATE_OPS=True`` in environment. This provides
          development-time safety without production overhead.

    Examples:
        Implementing a custom ImageEnhancer:

        >>> from phenotypic.abc_ import ImageEnhancer
        >>> from phenotypic import Image
        >>> from scipy.ndimage import gaussian_filter
        >>> class GaussianEnhancer(ImageEnhancer):
        ...     '''Custom enhancer applying Gaussian blur to enh_gray.'''
        ...
        ...     def __init__(self, sigma: float = 1.0):
        ...         super().__init__()
        ...         self.sigma = sigma  # Instance attribute
        ...
        ...     def _operate(self, image: Image) -> Image:
        ...         '''Apply Gaussian blur to enh_gray.'''
        ...         # Read enhanced grayscale
        ...         enh = image.enh_gray[:]
        ...         # Apply Gaussian filter (access parameter via self)
        ...         blurred = gaussian_filter(enh.astype(float), sigma=self.sigma)
        ...         # Modify enh_gray through accessor
        ...         image.enh_gray[:] = blurred.astype(enh.dtype)
        ...         return image
        >>> # Usage
        >>> enhancer = GaussianEnhancer(sigma=2.5)
        >>> enhanced = enhancer.apply(image)  # Original unchanged
        >>> enhanced_inplace = enhancer.apply(image, inplace=True)  # Original modified

        Implementing a custom ObjectDetector:

        >>> from phenotypic.abc_ import ObjectDetector
        >>> from phenotypic import Image
        >>> from skimage.feature import peak_local_max
        >>> from skimage.measure import label as measure_label
        >>> import numpy as np
        >>> class PeakDetector(ObjectDetector):
        ...     '''Detector using local peak finding to locate colonies.'''
        ...
        ...     def __init__(self, min_distance: int = 10, threshold_abs: int = 100):
        ...         super().__init__()
        ...         self.min_distance = min_distance
        ...         self.threshold_abs = threshold_abs
        ...
        ...     def _operate(self, image: Image) -> Image:
        ...         '''Find peaks in enh_gray and create object mask/map.'''
        ...         # Find local maxima (colony peaks) - access parameters via self
        ...         coords = peak_local_max(
        ...             image.enh_gray[:],
        ...             min_distance=self.min_distance,
        ...             threshold_abs=self.threshold_abs
        ...         )
        ...         # Create binary mask from peaks
        ...         mask = np.zeros(image.enh_gray.shape, dtype=bool)
        ...         for y, x in coords:
        ...             mask[y, x] = True
        ...         # Create labeled map from mask
        ...         labeled_map = measure_label(mask)
        ...         # Set detection results
        ...         image.objmask[:] = mask
        ...         image.objmap[:] = labeled_map
        ...         return image
        >>> # Usage - automatic integrity validation in ImageDetector
        >>> detector = PeakDetector(min_distance=15, threshold_abs=120)
        >>> detected = detector.apply(image)
        >>> colonies = detected.objects
        >>> print(f"Detected {len(colonies)} colonies")

        Understanding inplace parameter and memory efficiency:

        >>> from phenotypic.enhance import GaussianBlur
        >>> from phenotypic import Image
        >>> image = Image.imread('colony_plate.jpg')
        >>> enhancer = GaussianBlur(sigma=2.0)
        >>> # Default: inplace=False (safe, creates copy)
        >>> enhanced = enhancer.apply(image)
        >>> print(f"Same object? {id(image) == id(enhanced)}")  # False
        >>> # For memory efficiency with large images
        >>> result = enhancer.apply(image, inplace=True)
        >>> print(f"Same object? {id(image) == id(result)}")  # True
        # inplace=True is useful in pipelines with many large images
        # to minimize memory overhead, but modifies the original

        Using operations in a processing pipeline:

        >>> from phenotypic import Image, ImagePipeline
        >>> from phenotypic.enhance import GaussianBlur
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.grid import GridFinder
        >>> # Load image
        >>> image = Image.imread('colony_plate.jpg')
        >>> # Sequential chaining
        >>> enhanced = GaussianBlur(sigma=2).apply(image)
        >>> detected = OtsuDetector().apply(enhanced)
        >>> grid = GridFinder().apply(detected)
        >>> # Or use ImagePipeline for batch processing
        >>> pipeline = ImagePipeline()
        >>> pipeline.add(GaussianBlur(sigma=2))
        >>> pipeline.add(OtsuDetector())
        >>> pipeline.add(GridFinder())
        >>> # Process multiple images with automatic parallelization
        >>> images = [Image.imread(f) for f in image_files]
        >>> results = pipeline.operate(images)
        # Results are fully processed images

        How instance methods work with parallel execution:

        >>> from phenotypic.abc_ import ImageOperation
        >>> from phenotypic import Image
        >>> class CustomThreshold(ImageOperation):
        ...     def __init__(self, threshold: int, min_size: int = 5):
        ...         super().__init__()
        ...         self.threshold = threshold
        ...         self.min_size = min_size
        ...
        ...     def _operate(self, image: Image) -> Image:
        ...         # Access parameters via self
        ...         binary = image.enh_gray[:] > self.threshold
        ...         image.objmask[:] = binary
        ...         return image
        >>> # When apply() is called:
        >>> op = CustomThreshold(threshold=100, min_size=10)
        # apply() internally:
        # 1. Calls _apply_to_single_image() with self._operate (bound method)
        # 2. _apply_to_single_image calls operation(image)
        # 3. The bound method includes self, so all parameters are available
        >>> result = op.apply(image)
        # For parallel execution, the entire operation object (with all
        # attributes) is pickled and sent to worker processes
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @overload
    def apply(self, image: Image, inplace: bool = False) -> Image:
        ...

    @overload
    def apply(self, image: GridImage, inplace: bool = False) -> GridImage:
        ...

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
            image = self._apply_to_single_image(
                    cls_name=self.__class__.__name__,
                    image=image,
                    operation=self._operate,
                    inplace=inplace,
            )
            return image
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            raise RuntimeError(
                    f"{self.__class__.__name__} failed on image {image.name}:\n"
                    f"{traceback.format_exc()}"
            ) from e

    @overload
    @abstractmethod
    def _operate(self, image: Image) -> Image:
        ...

    @overload
    @abstractmethod
    def _operate(self, image: GridImage) -> GridImage:
        ...

    @abstractmethod
    def _operate(self, image: Image) -> Image:
        """
        Abstract instance method for processing an image.

        Subclasses implement this to define their operation logic.
        Access operation parameters via self.param_name.

        Args:
            image: The Image object to process.

        Returns:
            The modified Image object.
        """
        return image

    @staticmethod
    def _apply_to_single_image(cls_name, image, operation, inplace):
        """
        Applies the operation to a single image.

        Supports both instance methods (recommended) and static methods (backward compatibility).
        For instance methods, operation is a bound method with self included.
        For static methods, operation is an unbound function.

        Args:
            cls_name: Name of the operation class for error messages.
            image: The Image to process.
            operation: The _operate method (bound instance method or static method).
            inplace: If True, modify image in-place; otherwise work on a copy.

        Returns:
            The processed Image.
        """
        try:
            if inplace:
                return operation(image)
            else:
                return operation(image.copy())
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            raise Exception(f"{cls_name} failed on image {image.name}: {e}") from e
