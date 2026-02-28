from __future__ import annotations
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from phenotypic._core._image import Image
    from phenotypic._core._grid_image import GridImage

from ._image_operation import ImageOperation
from abc import ABC


class ImageCorrector(ImageOperation, ABC):
    """Abstract base class for whole-image transformation operations affecting all components.

    ImageCorrector is a specialized subclass of ImageOperation for global image transformations
    that modify **every image component together** (rgb, gray, detect_mat, objmask, objmap).
    Unlike ImageEnhancer (modifies only detect_mat) or ObjectDetector/ObjectRefiner (modify only
    detection results), an ImageCorrector transforms the entire image geometry or structure,
    ensuring all components remain synchronized.

    **Quick Decision Guide: Which Operation Type?**

    - **ImageEnhancer:** Modify only ``image.detect_mat`` for preprocessing.
      Use for: noise reduction, contrast enhancement, background subtraction.
    - **ObjectDetector:** Analyze image, produce only ``objmask`` and ``objmap``.
      Use for: colony/object detection and labeling.
    - **ObjectRefiner:** Edit mask and map (filtering, merging, removing objects).
      Use for: post-detection cleanup and refinement.
    - **ImageCorrector (this class):** Transform entire image (rotation, resampling, perspective).
      Use for: geometric corrections, coordinate system changes, alignment.
      Example: [GridAligner](src/phenotypic/correction/_grid_aligner.py).

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

    **Typical Use Cases**

    ImageCorrector is designed for operations that physically transform the image:

    - **Rotation:** Align plate image with detected grid structure to make colony rows parallel to axes.
    - **Perspective transformation:** Correct camera angle or lens distortion effects.
    - **Image resampling:** Change resolution or interpolation method for downstream processing.
    - **Global color correction:** Apply white balance or color space mapping to entire image.
    - **Alignment:** Register image to reference coordinate system for grid-based analysis.

    **When NOT to use ImageCorrector**

    Do NOT use ImageCorrector for operations that affect only specific image aspects:

    - **Don't use for:** Enhancing only ``detect_mat`` (preprocessing). Use ``ImageEnhancer`` instead.
    - **Don't use for:** Detecting colonies or objects. Use ``ObjectDetector`` instead.
    - **Don't use for:** Filtering or editing detection results. Use ``ObjectRefiner`` instead.
    - **Don't use for:** Local edits (e.g., removing a specific region). These typically require custom masking or ``ObjectRefiner``.

    The key question: Are you transforming **the entire image geometry**, or only a specific aspect?
    If only an aspect, use the specialized operation type (Enhancer, Detector, Refiner).

    **Why ImageCorrector is Rare in Practice**

    Most image processing operations are targeted to specific aspects of the image:

    - Colony detection focuses on finding objects in image data.
    - Post-detection cleanup focuses on refining the mask/map.
    - Preprocessing focuses on making detection more robust.

    Operations transforming the entire image structure are comparatively rare because:

    - Plate images are typically already well-oriented from the scanner/camera.
    - Most analysis works directly with image data as acquired (no rotation needed).
    - Grid-based alignment is a specialized step, not routine preprocessing.

    However, when needed, ImageCorrector provides the correct abstraction.

    **Subclass References**

    ImageCorrector implementations are rare. The canonical example is:

    - [GridAligner](src/phenotypic/correction/_grid_aligner.py): Rotates entire GridImage to align
      detected colonies with expected grid structure. Demonstrates synchronizing all components during transformation.

    **How to Implement a Custom ImageCorrector**

    Inherit from ImageCorrector and implement the ``_operate()`` static method. The static method
    requirement enables parallel execution in pipelines—operations can be serialized and sent to
    worker processes without instance state.

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

    **Key Implementation Rules**

    1. ``_operate()`` must be a **@staticmethod** (no instance access, enables serialization).
    2. All parameters (except ``image``) must match instance attributes by name exactly.
    3. The method must transform **all components equally** (rgb, gray, detect_mat, objmask, objmap).
    4. Never modify only one component—this breaks the "whole-image transformation" contract.
    5. Use the Image class's helper methods (``image.rotate()``) whenever possible for consistency.
    6. Always return the modified Image object after transformation.

    **Parameter Matching Example**

    When instance attributes don't match ``_operate()`` parameters, serialization and parallelization fail:

    .. code-block:: python

        # CORRECT: attribute names match parameter names
        class GoodRotator(ImageCorrector):
            def __init__(self, angle: float):
                super().__init__()
                self.angle = angle  # Matches _operate() parameter

            @staticmethod
            def _operate(image: Image, angle: float) -> Image:
                image.rotate(angle_of_rotation=angle, mode='edge')
                return image

        # WRONG: attribute name doesn't match parameter name
        class BadRotator(ImageCorrector):
            def __init__(self, angle: float):
                super().__init__()
                self.rotation_angle = angle  # Mismatch!

            @staticmethod
            def _operate(image: Image, angle: float) -> Image:  # Parameter is 'angle'
                # This will fail—no 'angle' attribute on self
                image.rotate(angle_of_rotation=angle, mode='edge')
                return image

    **Critical Implementation Detail: Updating All Components**

    Your ``_operate()`` method must ensure **all image components are updated together**. When
    any geometric transformation is applied, it must affect every component identically:

    .. code-block:: python

        @staticmethod
        def _operate(image: Image, angle: float) -> Image:
            # Rotate rgb and gray (color representation)
            image.rotate(angle_of_rotation=angle, mode='edge')

            # The following are automatically handled by image.rotate():
            # - Rotate detect_mat (enhanced version for detection)
            # - Rotate objmask and objmap (detection results)
            # - Synchronize all caches and metadata

            return image

    **What happens if components get out of sync?**

    If you accidentally rotate only ``image.rgb`` without rotating ``image.objmap``, downstream
    analysis breaks because pixel coordinates no longer match object labels. The Image class's
    helper methods protect against this by guaranteeing synchronized updates.

    **Access image data through accessors** (never direct attributes):

    When implementing custom transformations, always use the accessor interface:

    - **Reading:** ``image.rgb[:]``, ``image.gray[:]``, ``image.detect_mat[:]``, ``image.objmask[:]``, ``image.objmap[:]``
    - **Modifying:** ``image.rgb[:] = new_data``, ``image.objmap[:] = new_map``

    The accessor interface ensures that:

    - Caches are invalidated appropriately after modifications.
    - Color space conversions remain synchronized with RGB data.
    - Object detection results stay consistent with image geometry.

    The Image class provides helper methods for common transformations:

    - ``image.rotate(angle_of_rotation, mode='edge')`` - Rotates all components identically
    - For custom transformations, apply the same operation to all components
    - Always verify that helper methods exist before implementing custom transform code

    **Pipeline Integration and Serialization**

    ImageCorrector operations are fully serializable and can be included in ImagePipeline for
    batch processing. The static method design enables distributed execution:

    - **Automatic parameter passing:** Instance attributes are extracted when ``apply()`` is called.
    - **Serialization:** Operations can be saved to JSON/YAML and reconstructed on worker processes.
    - **Batch processing:** Use ``ImagePipeline.apply_and_measure()`` for automatic benchmarking.
    - **Reproducibility:** Serialized pipelines document the exact transformations applied.

    **Performance and Interpolation Considerations**

    When rotating or resampling, use appropriate interpolation for each component:

    - **Color data (rgb, gray):** Use smooth interpolation (order=1 bilinear or higher) to preserve color gradients and colony boundaries.
    - **Detection data (objmask, objmap):** Use nearest-neighbor interpolation (order=0) to preserve discrete object identities and integer labels.
    - **Detection matrix (detect_mat):** Use same interpolation as color data for consistency.

    Example with explicit interpolation control:

    >>> from scipy.ndimage import rotate as ndimage_rotate
    >>> from skimage.transform import rotate as skimage_rotate
    >>> # For rgb/gray: use bilinear interpolation
    >>> rotated_rgb = skimage_rotate(image.rgb[:], angle=5.0, order=1, preserve_range=True)
    >>> # For objmap: use nearest-neighbor to preserve integer labels
    >>> rotated_objmap = ndimage_rotate(image.objmap[:], angle=5.0, order=0, reshape=False)

    **Edge Handling During Transformation**

    Transformations introduce edge artifacts; choose mode based on downstream analysis:

    - **'edge' mode:** Replicas image border pixels (minimal artifacts, safest for colony detection).
    - **'constant' mode:** Fills with constant value (usually 0 for dark edge, may create false boundaries).
    - **'reflect' mode:** Reflects image at boundary (avoids abrupt discontinuities but changes image content).

    **Common Pitfalls and Best Practices**

    - **Pitfall:** Modifying only one component (e.g., rotating RGB but not objmap). Result: pixel coordinates become misaligned.
    - **Best practice:** Use Image class helper methods (``image.rotate()``) which synchronize all components automatically.
    - **Pitfall:** Using smooth interpolation for object maps. Result: object labels become non-integer, breaking downstream analysis.
    - **Best practice:** Use order=0 (nearest-neighbor) for masks and maps to preserve discrete identities.
    - **Pitfall:** Forgetting to handle edge artifacts. Result: false objects detected at image boundaries.
    - **Best practice:** Choose 'edge' mode for colony detection (minimal artifacts) or use image padding before transformation.

    **Attributes**

    - ImageCorrector has no public attributes; subclasses define operation-specific parameters as instance attributes.
    - All subclass attributes must match ``_operate()`` method signature for parallelization support.

    **Methods**

    - ``apply(image, inplace=False)`` - Execute the correction (default: returns new image).
    - ``_operate(image, **kwargs)`` - Abstract method you implement with transformation logic.

    **Notes**

    - **Static method requirement:** ``_operate()`` must be static to enable parallel execution in ImagePipeline.
    - **Parameter matching:** All ``_operate()`` parameters (except ``image``) must exist as instance attributes.
    - **No copy by default:** Operations return modified copies by default (inplace=False).
    - **Coordinate system changes:** Downstream operations may need re-detection after transformation.
    - **Grid alignment workflow:** [GridAligner](src/phenotypic/correction/_grid_aligner.py) is the canonical example.

    **Examples**

    Basic rotation operation:

    >>> from phenotypic.abc_ import ImageCorrector
    >>> from phenotypic.data import load_synth_yeast_plate
    >>>
    >>> class SimpleRotator(ImageCorrector):
    ...     def __init__(self, angle=5.0):
    ...         super().__init__()
    ...         self.angle = angle
    ...     @staticmethod
    ...     def _operate(image, angle):
    ...         image.rotate(angle_of_rotation=angle, mode='edge')
    ...         return image
    >>>
    >>> image = load_synth_yeast_plate()
    >>> rotator = SimpleRotator(angle=5.0)
    >>> rotated = rotator.apply(image)
    >>> # All components rotated together: rgb, gray, detect_mat, objmask, objmap
    >>> rotated.shape == image.shape
    False

    Custom perspective correction (preserving component synchronization):

    >>> from phenotypic.abc_ import ImageCorrector
    >>> from phenotypic.data import load_synth_yeast_plate
    >>> import numpy as np
    >>>
    >>> class PerspectiveCorrector(ImageCorrector):
    ...     def __init__(self, tilt_angle=10.0):
    ...         super().__init__()
    ...         self.tilt_angle = tilt_angle
    ...     @staticmethod
    ...     def _operate(image, tilt_angle):
    ...         # Apply perspective transform to all components
    ...         image.rotate(angle_of_rotation=tilt_angle, mode='edge')
    ...         return image
    >>>
    >>> image = load_synth_yeast_plate()
    >>> corrector = PerspectiveCorrector(tilt_angle=10.0)
    >>> corrected = corrector.apply(image)
    >>> # All components are transformed together, maintaining synchronization
    """

    @overload
    def apply(self, image: Image, inplace: bool = False) -> Image: ...

    @overload
    def apply(self, image: GridImage, inplace: bool = False) -> GridImage: ...

    def apply(self, image: Image, inplace: bool = False) -> Image:
        return super().apply(image=image, inplace=inplace)
