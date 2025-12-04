from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

from typing import Union, Dict

from ._image_operation import ImageOperation
from phenotypic.tools.exceptions_ import InterfaceError, OperationFailedError
from abc import ABC


class ImageCorrector(ImageOperation, ABC):
    """Abstract base class for whole-image transformation operations affecting all components.

    ImageCorrector is a specialized subclass of ImageOperation for global image transformations
    that modify **every image component together** (rgb, gray, enh_gray, objmask, objmap).
    Unlike ImageEnhancer (modifies only enh_gray) or ObjectDetector/ObjectRefiner (modify only
    detection results), an ImageCorrector transforms the entire image geometry or structure,
    ensuring all components remain synchronized.

    **What is ImageCorrector?**

    ImageCorrector handles operations where it is impossible or meaningless to modify only a
    single component. When you rotate, warp, or apply perspective transforms to an image, the
    rgb and gray representations must change together, and any existing detection masks and
    maps must be rotated identically. ImageCorrector guarantees this synchronization without
    requiring manual alignment of separate components.

    **Key Design Principle: No Integrity Checks**

    Unlike ImageEnhancer and ObjectDetector, ImageCorrector uses **no @validate_operation_integrity
    decorator**. This is by design: since all components must change together in a coordinated way,
    there is nothing to "protect" or "validate". The entire image is intentionally modified as a
    unit. The absence of integrity checks reflects this design, not a security weakness.

    **When to use ImageCorrector vs other operation types**

    Use the operation type that matches your modification scope:

    - **ImageEnhancer:** You only modify ``image.enh_gray`` (preprocessing for better detection).
      Use when: blur, contrast, edge detection, background subtraction.
      Example: GaussianBlur, CLAHE, RankMedianEnhancer.

    - **ObjectDetector:** You analyze image data and produce ``image.objmask`` and ``image.objmap``.
      Use when: discovering and labeling colonies, particles, or features.
      Example: OtsuDetector, CannyDetector, RoundPeaksDetector.

    - **ObjectRefiner:** You edit the mask and map (filter by size, merge, remove objects).
      Use when: post-processing detection results to clean up false positives.
      Example: SizeFilter, ComponentMerger, MorphologyRefiner.

    - **ImageCorrector (this class):** You transform the entire image structure (geometry,
      orientation, coordinate system). All components change together.
      Use when: rotation, alignment, perspective correction, image resampling.
      Example: GridAligner (rotates image to align detected colonies with grid rows/columns).

    **Typical Use Cases**

    ImageCorrector is designed for operations that physically transform the image:

    - **Rotation:** Align a plate image with detected grid structure. Rotate to make
      colony rows parallel to image axes, improving grid-based analysis.
    - **Perspective transformation:** Correct camera angle or lens distortion.
    - **Image resampling:** Change resolution or interpolation method.
    - **Global color correction:** Apply white balance or color space mapping to entire image.
    - **Alignment:** Register image to a reference coordinate system.

    **Why ImageCorrector is Rare in Practice**

    Most image processing operations are **targeted to specific aspects** of the image:

    - Colony detection focuses on finding objects in image data.
    - Post-detection cleanup focuses on refining the mask/map.
    - Preprocessing focuses on making detection more robust.

    Operations that transform the **entire image structure** are comparatively rare because:

    1. Plate images are typically already well-oriented from the scanner/camera.
    2. Most analysis works directly with image data as acquired (no rotation needed).
    3. Grid-based alignment is a specialized step, not routine preprocessing.

    However, when needed, ImageCorrector provides the correct abstraction.

    **How to Implement a Custom ImageCorrector**

    Inherit from ImageCorrector and implement the ``_operate()`` static method:

    .. code-block:: python

        from phenotypic.abc_ import ImageCorrector
        from phenotypic import Image

        class MyRotator(ImageCorrector):
            def __init__(self, angle: float):
                super().__init__()
                self.angle = angle  # Instance attribute, matched to _operate()

            @staticmethod
            def _operate(image: Image, angle: float) -> Image:
                # Rotate ALL image components together
                image.rotate(angle_of_rotation=angle, mode='edge')
                return image

        # Usage
        rotator = MyRotator(angle=5.0)
        rotated_image = rotator.apply(image)

    **Critical Implementation Detail: Updating All Components**

    Your ``_operate()`` method must ensure **all image components are updated together**:

    .. code-block:: python

        @staticmethod
        def _operate(image: Image, **kwargs) -> Image:
            # Rotate rgb and gray (color representation)
            image.rotate(angle_of_rotation=angle, mode='edge')

            # Rotate enh_gray (enhanced version for detection)
            # This is automatically handled by image.rotate()

            # Rotate objmask and objmap (detection results)
            # This is also automatically handled by image.rotate()

            return image

    **Access image data through accessors** (never direct attributes):

    - Reading: ``image.rgb[:]``, ``image.gray[:]``, ``image.enh_gray[:]``, ``image.objmask[:]``
    - Modifying: ``image.rgb[:] = new_data``, ``image.objmap[:] = new_map``

    The Image class provides helper methods for common transformations:

    - ``image.rotate(angle_of_rotation, mode='constant')`` - Rotates all components identically
    - For custom transformations, apply the same operation to all components

    **Performance and Interpolation Considerations**

    When rotating or resampling:

    - **Color data (rgb, gray):** Use smooth interpolation (order=1 or higher) to preserve
      color gradients and colony boundaries.
    - **Detection data (objmask, objmap):** Use nearest-neighbor interpolation (order=0) to
      preserve discrete object identities. Object labels must remain integer-valued.
    - **Enhanced grayscale (enh_gray):** Use the same interpolation as color data for consistency.

    Example with explicit interpolation control:

    .. code-block:: python

        from scipy.ndimage import rotate as ndimage_rotate
        from skimage.transform import rotate as skimage_rotate

        # For rgb/gray: use higher-order interpolation
        rotated_rgb = skimage_rotate(image.rgb[:], angle=5.0, order=1, preserve_range=True)

        # For objmap: use nearest-neighbor (order=0)
        rotated_objmap = ndimage_rotate(image.objmap[:], angle=5.0, order=0, reshape=False)

    **Edge Handling During Transformation**

    Transformations may introduce edge artifacts:

    - **Rotation with 'edge' mode:** Replicas image border pixels (minimal artificiality).
    - **Rotation with 'constant' mode:** Fills with a constant value (usually 0 for dark edge).
    - **Rotation with 'reflect' mode:** Reflects image at boundary (avoids abrupt discontinuities).

    Choose the mode based on downstream analysis. For colony detection, 'edge' is often safest.

    **Attributes**

    ImageCorrector itself has no public attributes. Subclasses define operation-specific
    parameters as instance attributes. All subclass attributes must be matched to the
    ``_operate()`` method signature for parallelization support.

    **Methods**

    Inherits all methods from ImageOperation. Key methods:

    - ``apply(image, inplace=False)`` - Execute the correction (default: returns new image).
    - ``_operate(image, **kwargs)`` - Abstract method you implement with transformation logic.

    **Notes**

    - **Static method requirement:** ``_operate()`` must be static to enable parallel execution
      in ImagePipeline. This allows the operation to be serialized and sent to worker processes.

    - **Parameter matching:** All ``_operate()`` parameters (except ``image``) must exist as
      instance attributes with matching names. When ``apply()`` is called, these are
      automatically extracted and passed to ``_operate()``.

    - **No copy by default:** Operations return modified copies by default (inplace=False).
      Original image is unchanged unless ``inplace=True`` is explicitly passed.

    - **Coordinate system changes:** After an ImageCorrector, downstream operations may need
      to re-detect objects or re-measure features, since the spatial coordinate system has changed.

    - **Grid alignment workflow:** GridAligner is the canonical example—it rotates the entire
      GridImage to align detected colonies with the expected grid structure, then downstream
      operations proceed with the aligned image.

    **Examples**

    .. dropdown:: GridAligner: rotating a GridImage to align colonies with rows/columns

        .. code-block:: python

            from phenotypic import GridImage, Image
            from phenotypic.detect import RoundPeaksDetector
            from phenotypic.correction import GridAligner

            # Load plate image
            image = Image.from_image_path('colony_plate.jpg')

            # Detect colonies in original orientation
            detected = RoundPeaksDetector().apply(image)

            # Create GridImage for grid-aware operations
            grid_image = GridImage(detected)
            grid_image.detect_grid()  # Estimate grid structure

            # Rotate entire image to align colonies with grid axes
            aligner = GridAligner(axis=0)  # Align rows horizontally
            aligned = aligner.apply(grid_image)

            # Now all components (rgb, gray, enh_gray, masks, map) are rotated together
            # Downstream grid-based operations work with aligned coordinates
            print(f"Original shape: {grid_image.shape}")
            print(f"Aligned shape: {aligned.shape}")

    .. dropdown:: Custom perspective correction (conceptual example)

        .. code-block:: python

            from phenotypic.abc_ import ImageCorrector
            from phenotypic import Image
            from skimage.transform import warp, ProjectiveTransform
            import numpy as np

            class PerspectiveCorrector(ImageCorrector):
                \"\"\"Correct camera angle by applying perspective transform.\"\"\"

                def __init__(self, tilt_angle: float, direction: str = 'x'):
                    super().__init__()
                    self.tilt_angle = tilt_angle
                    self.direction = direction

                @staticmethod
                def _operate(image: Image, tilt_angle: float, direction: str) -> Image:
                    # Define perspective transform
                    h, w = image.shape
                    # (Implementation details omitted for brevity)
                    # Apply warp to all components with appropriate interpolation
                    return image

            # Usage
            corrector = PerspectiveCorrector(tilt_angle=10.0, direction='x')
            corrected = corrector.apply(image)
            # All image components are perspective-corrected together
    """

    pass
