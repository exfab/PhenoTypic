from __future__ import annotations
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from phenotypic._core._image import Image
    from phenotypic._core._grid_image import GridImage

from ._image_operation import ImageOperation

from phenotypic.tools_.funcs_ import validate_operation_integrity
from abc import ABC, abstractmethod
from phenotypic.tools_.mixin import FootprintMixin


class ImageEnhancer(FootprintMixin, ImageOperation, ABC):
    """Abstract base class for preprocessing operations that improve colony detection through detection matrix.

    ImageEnhancer is the foundation for all preprocessing algorithms that modify only the enhanced
    grayscale channel (`image.detect_mat`) to improve colony visibility and detection quality. Unlike
    ImageCorrector (which transforms the entire Image), ImageEnhancer leaves the original RGB and
    grayscale data untouched, protecting image integrity while enabling targeted preprocessing.

    **Quick Decision Guide: Which Operation Type?**

    - **ImageEnhancer (this class):** Modify only ``image.detect_mat`` for preprocessing.
      Use for: noise reduction, contrast enhancement, illumination correction.
      Examples: [GaussianBlur](src/phenotypic/enhance/_gaussian_blur.py),
      [CLAHE](src/phenotypic/enhance/_clahe.py),
      [BilateralDenoise](src/phenotypic/enhance/_bilateral_denoise.py).
    - **ImageCorrector:** Transform entire image (rotation, cropping, perspective).
      Use for: geometric corrections, global color transformations.
    - **ObjectDetector:** Analyze image, produce only ``objmask`` and ``objmap``.
      Use for: colony/object detection and labeling.
    - **ObjectRefiner:** Edit mask and map (filtering, merging, removing objects).
      Use for: post-detection cleanup and refinement.

    **What is ImageEnhancer?**

    ImageEnhancer operates on the principle of **non-destructive preprocessing**: all modifications
    are applied to `image.detect_mat` (a working copy of grayscale), while original image components
    (`image.rgb`, `image.gray`, `image.objmask`, `image.objmap`) remain protected and unchanged.
    This allows you to experiment with multiple enhancement chains without affecting raw data or
    detection results.

    **Role in the Detection Pipeline**

    ImageEnhancer sits at the beginning of the processing chain:

    .. code-block:: text

        Raw Image (image.rgb, image.gray)
              ↓
        ImageEnhancer(s) → Improve visibility, reduce noise
              ↓
        ObjectDetector → Detect colonies/objects
              ↓
        ObjectRefiner → Clean up detections (optional)

    When you call `enhancer.apply(image)`, you get back an Image with improved `detect_mat` but
    identical RGB/gray data—ready for detection algorithms to operate on enhanced contrast.

    **Why Enhancement Matters for Colony Phenotyping**

    Real agar plate imaging introduces several challenges that enhancement operations address:

    - **Uneven illumination:** Vignetting, shadows, and scanner lighting gradients make colonies appear faint in dark regions.
    - **Noise and texture:** Scanner noise, agar granularity, dust, and condensation create artifacts confusing detection.
    - **Faint colonies:** Small or translucent colonies blend into background, reducing detectability.
    - **Poor contrast:** Low-contrast colonies on dense plates require local contrast enhancement.

    Enhancement operations preserve colony morphology while suppressing artifacts for robust detection.

    **Subclass References**

    The following are canonical examples of ImageEnhancer implementations:

    - [GaussianBlur](src/phenotypic/enhance/_gaussian_blur.py): Noise reduction via Gaussian filtering.
    - [CLAHE](src/phenotypic/enhance/_clahe.py): Contrast-limited adaptive histogram equalization for local contrast.
    - ``GrayOpening``: Morphological opening using ``FootprintMixin``.
    - [BilateralDenoise](src/phenotypic/enhance/_bilateral_denoise.py): Edge-preserving denoising.

    **Integrity Validation: Protection of Core Data**

    ImageEnhancer uses the ``@validate_operation_integrity`` decorator on the ``apply()`` method
    to guarantee that RGB and grayscale data are never modified:

    .. code-block:: python

        @validate_operation_integrity('image.rgb', 'image.gray')
        def apply(self, image: Image, inplace: bool = False) -> Image:
            return super().apply(image=image, inplace=inplace)

    This decorator:

    1. Calculates cryptographic signatures of `image.rgb` and `image.gray` **before** processing
    2. Calls the parent `apply()` method to execute your `_operate()` implementation
    3. Recalculates signatures **after** operation completes
    4. Raises ``OperationIntegrityError`` if any protected component was modified

    **Note:** Integrity validation only runs if the ``VALIDATE_OPS=True`` environment variable
    is set (development-time safety; disabled in production for performance).

    **Implementing a Custom ImageEnhancer**

    Subclass ImageEnhancer and implement a single method:

    .. code-block:: python

        from phenotypic.abc_ import ImageEnhancer
        from phenotypic import Image
        from scipy.ndimage import gaussian_filter

        class MyCustomEnhancer(ImageEnhancer):
            sigma: float = 1.0  # Annotated class-level field

            def _operate(self, image: Image) -> Image:
                # Modify ONLY detect_mat; read, process, write back
                enh = image.detect_mat[:]
                filtered = gaussian_filter(enh.astype(float), sigma=self.sigma)
                image.detect_mat[:] = filtered.astype(enh.dtype)
                return image

    **Key Rules for Implementation:**

    1. ``_operate()`` should be an **instance method** (no ``@staticmethod`` decorator).
    2. Access operation parameters directly via ``self.param_name``.
    3. **Only modify ``image.detect_mat[:]``**—all other components are protected.
    4. Always use the accessor pattern: ``image.detect_mat[:] = new_data`` (never direct attribute
       assignment like ``image._detect_mat = ...``).
    5. Return the modified Image object.

    **Accessing and Modifying detect_mat**

    Within your `_operate()` method, use the accessor interface:

    .. code-block:: python

        # Reading detection matrix data
        enh_data = image.detect_mat[:]        # Full array
        region = image.detect_mat[10:50, 20:80]  # Slicing with NumPy syntax

        # Modifying detection matrix
        image.detect_mat[:] = processed_array  # Full replacement
        image.detect_mat[10:50, 20:80] = new_region  # Partial update

    The accessor handles all consistency checks and automatic cache invalidation.

    **The _make_footprint() Static Utility**

    ImageEnhancer provides a static helper for generating morphological structuring elements
    (footprints) used in morphological operations like erosion, dilation, and median filtering:

    .. code-block:: python

        @staticmethod
        def _make_footprint(shape: Literal["square", "diamond", "disk"], width: int) -> np.ndarray:
            '''Creates a binary morphological shape for image processing.'''

    **Footprint Shapes and When to Use Each**

    - **"disk":** Circular/isotropic shape. Best for preserving rounded colony shapes and
      applying uniform processing in all directions. Use for: general-purpose smoothing, median
      filtering, dilations that expand colonies symmetrically.

    - **"square":** Square shape with 8-connectivity. Emphasizes horizontal/vertical edges
      and aligns with pixel grid. Use for: grid-aligned artifacts (imaging hardware stripe patterns),
      when processing speed matters (slightly faster than disk).

    - **"diamond":** Diamond-shaped (rotated square) shape with 4-connectivity. Creates a
      cross-like neighborhood pattern. Use for: specialized cases where diagonal connections should
      be de-emphasized; less common in practice.

    **The width parameter** controls the neighborhood size (in pixels). Larger radii affect more
    neighbors and produce broader effects (more noise suppression, but potential colony merging).
    Choose width smaller than the minimum colony diameter to avoid destroying fine details.

    **Common Morphological Patterns**

    Use `_make_footprint()` with morphological operations from `scipy.ndimage` or `skimage.morphology`:

    .. code-block:: python

        from skimage.morphology import erosion, dilation
        from phenotypic.abc_ import ImageEnhancer

        disk_fp = ImageEnhancer._make_footprint('disk', width=5)

        # Erosion: shrink bright regions (removes small colonies/noise)
        eroded = erosion(binary_image, footprint=disk_fp)

        # Dilation: expand bright regions (closes holes, merges nearby colonies)
        dilated = dilation(binary_image, footprint=disk_fp)

    **When and Why to Chain Multiple Enhancements**

    Enhancement operations are typically chained together to address multiple issues in sequence:

    .. code-block:: python

        # Example pipeline: handle uneven illumination + noise
        # Step 1: Remove background gradients
        result = SubtractRollingBall(width=50).apply(image)

        # Step 2: Boost local contrast for faint colonies
        result = CLAHE(kernel_size=50, clip_limit=0.02).apply(result)

        # Step 3: Smooth remaining noise
        result = GaussianBlur(sigma=2).apply(result)

        # Step 4: Detect colonies in detection matrix
        result = OtsuDetector().apply(result)

    **Rationale for chaining:**

    - **Order matters:** Background correction before contrast enhancement yields better results
      than vice versa.
    - **Divide and conquer:** One enhancer per problem (illumination, noise, contrast) is more
      maintainable and tunable than one monolithic algorithm.
    - **No data loss:** Each enhancer preserves the original RGB/gray, so intermediate results
      can be inspected and validated.
    - **Reproducibility:** Chained operations can be serialized to YAML for documentation and
      reuse across experiments.

    **Use ImagePipeline for convenient chaining:**

    .. code-block:: python

        from phenotypic import Image, ImagePipeline
        from phenotypic.enhance import SubtractRollingBall, CLAHE, GaussianBlur
        from phenotypic.detect import OtsuDetector

        pipeline = ImagePipeline()
        pipeline.add(SubtractRollingBall(width=50))
        pipeline.add(CLAHE(kernel_size=50, clip_limit=0.02))
        pipeline.add(GaussianBlur(sigma=2))
        pipeline.add(OtsuDetector())

        # Process a batch of images with automatic parallelization
        images = [Image.imread(f) for f in plate_scans]
        results = pipeline.operate(images)

    **Methods and Attributes**

    Attributes:
        None at the ImageEnhancer level; subclasses define enhancement parameters
        as instance attributes (e.g., sigma, kernel_size, clip_limit).

    Methods:
        apply(image, inplace=False): Applies the enhancement to an image. Returns a modified
            Image with enhanced `detect_mat` but unchanged RGB/gray/objects. Handles copy/inplace
            logic and validates data integrity.
        _operate(self, image): Abstract instance method implemented by subclasses.
            Performs the actual enhancement algorithm. Access parameters via ``self.param_name``.
        _make_footprint(shape, width): Static utility that creates a binary morphological
            shape (disk, square, or diamond) for use in morphological operations.

    Notes:
        - **Protected components:** The ``@validate_operation_integrity`` decorator ensures
          that ``image.rgb`` and ``image.gray`` cannot be modified. Only ``image.detect_mat``
          can be changed.

        - **Immutability by default:** ``apply(image)`` returns a modified copy by default.
          Set ``inplace=True`` for memory-efficient in-place modification.

        - **Instance method pattern:** The ``_operate()`` method should be an instance method
          (no ``@staticmethod`` decorator). Access operation parameters directly via
          ``self.param_name``. This is simpler and more Pythonic.

        - **Accessor pattern:** Always use ``image.detect_mat[:] = new_data`` to modify
          detection matrix. Never use direct attribute assignment.

        - **Automatic cache invalidation:** When you modify ``image.detect_mat[:]``, the
          Image's internal caches (e.g., color space conversions, object maps) are
          automatically invalidated to prevent stale results.

    Examples:
        Basic usage with noise reduction:

        >>> from phenotypic.abc_ import ImageEnhancer
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from scipy.ndimage import gaussian_filter
        >>>
        >>> class GaussianEnhancer(ImageEnhancer):
        ...     sigma: float = 1.5
        ...     def _operate(self, image):
        ...         enh = image.detect_mat[:]
        ...         filtered = gaussian_filter(enh.astype(float), sigma=self.sigma)
        ...         image.detect_mat[:] = filtered.astype(enh.dtype)
        ...         return image
        >>>
        >>> image = load_synth_yeast_plate()
        >>> enhancer = GaussianEnhancer(sigma=2.0)
        >>> enhanced = enhancer.apply(image)
        >>> # Original RGB and gray are unchanged
        >>> assert (image.gray[:] == enhanced.gray[:]).all()

        Morphological enhancement with FootprintMixin for colony hole-filling:

        >>> from phenotypic.abc_ import ImageEnhancer
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from skimage.morphology import closing
        >>>
        >>> class MorphologicalEnhancer(ImageEnhancer):
        ...     operation: str = 'closing'
        ...     width: int = 3
        ...     def _operate(self, image):
        ...         enh = image.detect_mat[:]
        ...         footprint = ImageEnhancer._make_footprint('disk', self.width)
        ...         binary = enh > enh.mean()
        ...         refined = closing(binary, footprint=footprint)
        ...         image.detect_mat[:] = (refined * 255).astype(enh.dtype)
        ...         return image
        >>>
        >>> image = load_synth_yeast_plate()
        >>> enhancer = MorphologicalEnhancer(operation='closing', width=5)
        >>> enhanced = enhancer.apply(image)

        Chaining multiple enhancements in pipeline:

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.enhance import GaussianBlur, CLAHE
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.data import load_synth_yeast_plate
        >>>
        >>> image = load_synth_yeast_plate()
        >>> pipeline = ImagePipeline(pipe_cfgs=[
        ...     GaussianBlur(sigma=1.5),
        ...     CLAHE(clip_limit=2.0),
        ...     OtsuDetector()
        ... ])
        >>> result = pipeline.apply(image)
        >>> colonies = result.objects
        >>> len(colonies) > 0
        True

    """

    @overload
    def apply(self, image: Image, inplace: bool = False) -> Image: ...

    @overload
    def apply(self, image: GridImage, inplace: bool = False) -> GridImage: ...

    @validate_operation_integrity("image.rgb", "image.gray")
    def apply(self, image: Image, inplace: bool = False) -> Image:
        return super().apply(image=image, inplace=inplace)

    @overload
    @abstractmethod
    def _operate(self, image: Image) -> Image: ...

    @overload
    @abstractmethod
    def _operate(self, image: GridImage) -> GridImage: ...

    @abstractmethod
    def _operate(self, image: Image) -> Image:
        return image
