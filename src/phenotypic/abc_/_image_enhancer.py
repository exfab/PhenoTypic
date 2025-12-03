from __future__ import annotations
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

import numpy as np
from ._image_operation import ImageOperation

from phenotypic.tools.funcs_ import validate_operation_integrity
from abc import ABC
from skimage.morphology import disk, square, diamond


class ImageEnhancer(ImageOperation, ABC):
    """Abstract base class for preprocessing operations that improve colony detection through enhanced grayscale.

    ImageEnhancer is the foundation for all preprocessing algorithms that modify only the enhanced
    grayscale channel (`image.enh_gray`) to improve colony visibility and detection quality. Unlike
    ImageCorrector (which transforms the entire Image), ImageEnhancer leaves the original RGB and
    grayscale data untouched, protecting image integrity while enabling targeted preprocessing.

    **What is ImageEnhancer?**

    ImageEnhancer operates on the principle of **non-destructive preprocessing**: all modifications
    are applied to `image.enh_gray` (a working copy of grayscale), while original image components
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

    When you call `enhancer.apply(image)`, you get back an Image with improved `enh_gray` but
    identical RGB/gray data—ready for detection algorithms to operate on enhanced contrast.

    **Why Enhancement Matters for Colony Phenotyping**

    Real agar plate imaging introduces several challenges:

    - **Uneven illumination:** Vignetting, shadows, and scanner lighting gradients make colonies
      appear faint in dark regions and over-exposed elsewhere.
    - **Noise and texture:** Scanner noise, agar granularity, condensation droplets, and dust create
      artifacts that confuse thresholding or edge detection.
    - **Faint colonies:** Small or translucent colonies blend into background, reducing detectability.
    - **Poor contrast:** Low-contrast colonies on dense plates require local contrast enhancement.

    Enhancement operations target these issues in a **domain-specific way**: they preserve colony
    morphology while suppressing artifacts, enabling robust detection in downstream algorithms.

    **When to Use ImageEnhancer vs Other Operations**

    - **ImageEnhancer:** You only modify `enh_gray` for preprocessing. Use for: noise reduction
      (Gaussian blur, median filtering), contrast enhancement (CLAHE), illumination compensation
      (background subtraction), edge enhancement (Sobel, Laplacian). Typical use: before detection.

    - **ImageCorrector:** You transform the entire Image (rotation, cropping, color correction).
      Typical use: geometric corrections or global color transformations.

    - **ObjectDetector:** You analyze image data and produce only `objmask` and `objmap`. Input
      image data is protected. Typical use: colony detection and labeling.

    - **ObjectRefiner:** You edit mask and map (filtering, merging, removing objects). Typical use:
      post-detection cleanup and validation.

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
            def __init__(self, sigma: float = 1.0):
                super().__init__()
                self.sigma = sigma  # Instance attribute matched to _operate()

            @staticmethod
            def _operate(image: Image, sigma: float = 1.0) -> Image:
                # Modify ONLY enh_gray; read, process, write back
                enh = image.enh_gray[:]
                filtered = gaussian_filter(enh.astype(float), sigma=sigma)
                image.enh_gray[:] = filtered.astype(enh.dtype)
                return image

    **Key Rules for Implementation:**

    1. ``_operate()`` must be **static** (required for parallel execution in pipelines).
    2. All parameters except `image` must exist as instance attributes with matching names
       (enables automatic parameter matching via `_get_matched_operation_args()`).
    3. **Only modify ``image.enh_gray[:]``**—all other components are protected.
    4. Always use the accessor pattern: ``image.enh_gray[:] = new_data`` (never direct attribute
       assignment like ``image._enh_gray = ...``).
    5. Return the modified Image object.

    **Accessing and Modifying enh_gray**

    Within your `_operate()` method, use the accessor interface:

    .. code-block:: python

        # Reading enhanced grayscale data
        enh_data = image.enh_gray[:]        # Full array
        region = image.enh_gray[10:50, 20:80]  # Slicing with NumPy syntax

        # Modifying enhanced grayscale
        image.enh_gray[:] = processed_array  # Full replacement
        image.enh_gray[10:50, 20:80] = new_region  # Partial update

    The accessor handles all consistency checks and automatic cache invalidation.

    **The _make_footprint() Static Utility**

    ImageEnhancer provides a static helper for generating morphological structuring elements
    (footprints) used in morphological operations like erosion, dilation, and median filtering:

    .. code-block:: python

        @staticmethod
        def _make_footprint(shape: Literal["square", "diamond", "disk"], radius: int) -> np.ndarray:
            '''Creates a binary morphological footprint for image processing.'''

    **Footprint Shapes and When to Use Each**

    - **"disk":** Circular/isotropic footprint. Best for preserving rounded colony shapes and
      applying uniform processing in all directions. Use for: general-purpose smoothing, median
      filtering, dilations that expand colonies symmetrically.

    - **"square":** Square footprint with 8-connectivity. Emphasizes horizontal/vertical edges
      and aligns with pixel grid. Use for: grid-aligned artifacts (imaging hardware stripe patterns),
      when processing speed matters (slightly faster than disk).

    - **"diamond":** Diamond-shaped (rotated square) footprint with 4-connectivity. Creates a
      cross-like neighborhood pattern. Use for: specialized cases where diagonal connections should
      be de-emphasized; less common in practice.

    **The radius parameter** controls the neighborhood size (in pixels). Larger radii affect more
    neighbors and produce broader effects (more noise suppression, but potential colony merging).
    Choose radius smaller than the minimum colony diameter to avoid destroying fine details.

    **Common Morphological Patterns**

    Use `_make_footprint()` with morphological operations from `scipy.ndimage` or `skimage.morphology`:

    .. code-block:: python

        from scipy.ndimage import binary_dilation, binary_erosion
        from phenotypic.abc_ import ImageEnhancer

        disk_fp = ImageEnhancer._make_footprint('disk', radius=5)

        # Erosion: shrink bright regions (removes small colonies/noise)
        eroded = binary_erosion(binary_image, structure=disk_fp)

        # Dilation: expand bright regions (closes holes, merges nearby colonies)
        dilated = binary_dilation(binary_image, structure=disk_fp)

    **When and Why to Chain Multiple Enhancements**

    Enhancement operations are typically chained together to address multiple issues in sequence:

    .. code-block:: python

        # Example pipeline: handle uneven illumination + noise
        # Step 1: Remove background gradients
        result = RollingBallRemoveBG(radius=50).apply(image)

        # Step 2: Boost local contrast for faint colonies
        result = CLAHE(kernel_size=50, clip_limit=0.02).apply(result)

        # Step 3: Smooth remaining noise
        result = GaussianBlur(sigma=2).apply(result)

        # Step 4: Detect colonies in enhanced grayscale
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
        from phenotypic.enhance import RollingBallRemoveBG, CLAHE, GaussianBlur
        from phenotypic.detect import OtsuDetector

        pipeline = ImagePipeline()
        pipeline.add(RollingBallRemoveBG(radius=50))
        pipeline.add(CLAHE(kernel_size=50, clip_limit=0.02))
        pipeline.add(GaussianBlur(sigma=2))
        pipeline.add(OtsuDetector())

        # Process a batch of images with automatic parallelization
        images = [Image.from_image_path(f) for f in plate_scans]
        results = pipeline.operate(images)

    **Methods and Attributes**

    Attributes:
        None at the ImageEnhancer level; subclasses define enhancement parameters
        as instance attributes (e.g., sigma, kernel_size, clip_limit).

    Methods:
        apply(image, inplace=False): Applies the enhancement to an image. Returns a modified
            Image with enhanced `enh_gray` but unchanged RGB/gray/objects. Handles copy/inplace
            logic and validates data integrity.
        _operate(image, **kwargs): Abstract static method implemented by subclasses.
            Performs the actual enhancement algorithm. Parameters are automatically matched
            to instance attributes.
        _make_footprint(shape, radius): Static utility that creates a binary morphological
            footprint (disk, square, or diamond) for use in morphological operations.

    Notes:
        - **Protected components:** The ``@validate_operation_integrity`` decorator ensures
          that ``image.rgb`` and ``image.gray`` cannot be modified. Only ``image.enh_gray``
          can be changed.

        - **Immutability by default:** ``apply(image)`` returns a modified copy by default.
          Set ``inplace=True`` for memory-efficient in-place modification.

        - **Static _operate() requirement:** The ``_operate()`` method must be static to
          support parallel execution in pipelines.

        - **Parameter matching for parallelization:** All ``_operate()`` parameters except
          ``image`` must exist as instance attributes. When ``apply()`` is called, these
          values are extracted and passed to ``_operate()``.

        - **Accessor pattern:** Always use ``image.enh_gray[:] = new_data`` to modify
          enhanced grayscale. Never use direct attribute assignment.

        - **Automatic cache invalidation:** When you modify ``image.enh_gray[:]``, the
          Image's internal caches (e.g., color space conversions, object maps) are
          automatically invalidated to prevent stale results.

    Examples:
        .. dropdown:: Implementing a custom noise-reduction enhancer with Gaussian blur

            .. code-block:: python

                from phenotypic.abc_ import ImageEnhancer
                from phenotypic import Image
                from scipy.ndimage import gaussian_filter
                import numpy as np

                class CustomGaussianEnhancer(ImageEnhancer):
                    '''Enhance by applying Gaussian blur to reduce noise.'''

                    def __init__(self, sigma: float = 1.5):
                        super().__init__()
                        self.sigma = sigma

                    @staticmethod
                    def _operate(image: Image, sigma: float = 1.5) -> Image:
                        enh = image.enh_gray[:]
                        # Convert to float for processing
                        filtered = gaussian_filter(enh.astype(float), sigma=sigma)
                        # Restore original dtype
                        image.enh_gray[:] = filtered.astype(enh.dtype)
                        return image

                # Usage
                from phenotypic import Image
                from phenotypic.detect import OtsuDetector

                image = Image.from_image_path('agar_plate.jpg')
                enhancer = CustomGaussianEnhancer(sigma=2.0)
                enhanced = enhancer.apply(image)  # Original unchanged
                detected = OtsuDetector().apply(enhanced)  # Detect in enhanced data
                colonies = detected.objects
                print(f"Detected {len(colonies)} colonies")

        .. dropdown:: Morphological operations using _make_footprint for colony refinement

            .. code-block:: python

                from phenotypic.abc_ import ImageEnhancer
                from phenotypic import Image
                from scipy.ndimage import binary_closing, binary_opening
                import numpy as np

                class MorphologicalEnhancer(ImageEnhancer):
                    '''Enhance by applying morphological closing/opening to fill holes and remove noise.'''

                    def __init__(self, operation: str = 'closing', radius: int = 3):
                        super().__init__()
                        self.operation = operation  # 'closing' or 'opening'
                        self.radius = radius

                    @staticmethod
                    def _operate(image: Image, operation: str = 'closing', radius: int = 3) -> Image:
                        enh = image.enh_gray[:]
                        # Create a disk footprint for isotropic processing
                        footprint = ImageEnhancer._make_footprint('disk', radius)

                        # Apply morphological operation to binary image
                        binary = enh > enh.mean()
                        if operation == 'closing':
                            # Close small holes within colonies
                            refined = binary_closing(binary, structure=footprint)
                        elif operation == 'opening':
                            # Remove small noise regions
                            refined = binary_opening(binary, structure=footprint)
                        else:
                            return image

                        # Convert back to grayscale (refined mask as 0/255)
                        image.enh_gray[:] = (refined * 255).astype(enh.dtype)
                        return image

                # Usage
                enhancer = MorphologicalEnhancer(operation='closing', radius=5)
                result = enhancer.apply(image)

        .. dropdown:: Chaining multiple enhancements to handle complex agar plate imaging conditions

            .. code-block:: python

                from phenotypic import Image, ImagePipeline
                from phenotypic.enhance import (
                    RollingBallRemoveBG, CLAHE, GaussianBlur
                )
                from phenotypic.detect import OtsuDetector

                # Scenario: Agar plate image with vignetting, dust, and low contrast

                # Build a processing pipeline
                pipeline = ImagePipeline()

                # Step 1: Remove illumination gradient (vignetting)
                pipeline.add(RollingBallRemoveBG(radius=80))

                # Step 2: Boost local contrast for faint colonies
                pipeline.add(CLAHE(kernel_size=50, clip_limit=0.02))

                # Step 3: Smooth dust and scanner noise
                pipeline.add(GaussianBlur(sigma=1.5))

                # Step 4: Detect colonies
                pipeline.add(OtsuDetector())

                # Process a batch of plate images
                image_paths = ['plate1.tif', 'plate2.tif', 'plate3.tif']
                images = [Image.from_image_path(p) for p in image_paths]
                results = pipeline.operate(images)

                # Each result has cleaned detection results
                for i, result in enumerate(results):
                    colonies = result.objects
                    print(f"Plate {i}: {len(colonies)} colonies detected")

        .. dropdown:: Using different footprint shapes for specialized morphological filtering

            .. code-block:: python

                from phenotypic.abc_ import ImageEnhancer
                from phenotypic import Image
                from skimage.filters.rank import median
                from skimage.util import img_as_ubyte, img_as_float
                import numpy as np

                class SelectiveMedianEnhancer(ImageEnhancer):
                    '''Enhance by applying median filtering with configurable footprint shape.'''

                    def __init__(self, shape: str = 'disk', radius: int = 3):
                        super().__init__()
                        self.shape = shape  # 'disk', 'square', or 'diamond'
                        self.radius = radius

                    @staticmethod
                    def _operate(image: Image, shape: str = 'disk', radius: int = 3) -> Image:
                        enh = image.enh_gray[:]

                        # Create footprint with specified shape
                        footprint = ImageEnhancer._make_footprint(shape, radius)

                        # Apply median filter (rank filter)
                        # Convert to uint8 for rank filter compatibility
                        as_uint8 = img_as_ubyte(enh)
                        filtered = median(as_uint8, footprint=footprint)

                        # Restore original dtype
                        image.enh_gray[:] = img_as_float(filtered) if enh.dtype == np.float64 else filtered
                        return image

                # Usage with different shapes
                image = Image.from_image_path('plate.jpg')

                # Isotropic smoothing (preserves round colony shapes)
                result1 = SelectiveMedianEnhancer(shape='disk', radius=3).apply(image)

                # Grid-aligned smoothing (for hardware artifacts)
                result2 = SelectiveMedianEnhancer(shape='square', radius=3).apply(image)

                # Both preserve original image.rgb and image.gray
                assert np.array_equal(image.gray[:], result1.gray[:])
                assert np.array_equal(image.rgb[:], result1.rgb[:])

    """

    @validate_operation_integrity("image.rgb", "image.gray")
    def apply(self, image: Image, inplace: bool = False) -> Image:
        return super().apply(image=image, inplace=inplace)

    @staticmethod
    def _make_footprint(
            shape: Literal["square", "diamond", "disk"], radius: int
    ) -> np.ndarray:
        """
        Creates a morphological footprint for image processing.

        This static utility method generates a structuring element (footprint) useful
        for morphological operations like dilation and erosion. It supports different
        shapes such as square, diamond, and disk, which are often used in image analysis
        tasks. These morphological tools are particularly helpful in analyzing colonies
        of microbes on solid media agar.

        Args:
            shape (Literal["square", "diamond", "disk"]): The shape of the footprint to create.
                Adjusting the shape changes the way the morphological operations interact
                with the image. For example:
                - "square" creates a square footprint, which may emphasize features with
                  sharp edges.
                - "diamond" creates a diamond-shaped footprint, which may enhance diagonal
                  connections while being less sensitive to orthogonal edges.
                - "disk" generates a circular footprint, which may better preserve rounded
                  microbial colony shapes.

            radius (int): The radius of the footprint. This defines the size of the
                structuring element. Larger radii will lead to broader morphological
                effects, which could impact the resolution of small colonies but can help
                to merge fragmented edges or clean noise.

        Returns:
            np.ndarray: A binary numpy array representing the generated footprint. The
            footprint will be used for convolutional operations over the microbial colony
            image. The specific shape and radius passed as arguments dictate the size
            and morphology of this array.

        Raises:
            ValueError: If an unsupported shape type is passed to the function.
        """
        radius = int(radius)
        match shape:
            case "square":
                return square(width=radius*2)
            case "diamond":
                return diamond(radius=radius)
            case "disk":
                return disk(radius=radius)
            case _:
                raise ValueError(f"Unknown shape: {shape}")
