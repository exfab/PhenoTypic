from __future__ import annotations
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic import Image

import numpy as np

from ._image_operation import ImageOperation
from phenotypic.tools.exceptions_ import (
    OperationFailedError,
    InterfaceError,
    DataIntegrityError,
)
from phenotypic.tools.funcs_ import validate_operation_integrity
from abc import ABC
from skimage.morphology import disk, square, diamond


# <<Interface>>
class ObjectRefiner(ImageOperation, ABC):
    """Abstract base class for post-detection refinement operations that modify object masks and maps.

    ObjectRefiner is the foundation for all post-detection cleanup algorithms that refine colony
    detections through morphological operations, filtering, and merging. Unlike ObjectDetector
    (which analyzes image data to create initial detections), ObjectRefiner only modifies the
    object mask and labeled map, leaving preprocessing data untouched.

    **What is ObjectRefiner?**

    ObjectRefiner operates on the principle of **non-destructive post-processing**: all modifications
    are applied only to `image.objmask` (binary mask) and `image.objmap` (labeled map), while original
    image components (`image.rgb`, `image.gray`, `image.enh_gray`) remain protected and unchanged.
    This allows you to experiment with multiple refinement chains without affecting raw or enhanced
    image data, ensuring reproducibility and enabling comparison of different cleanup strategies.

    **Key Principle: ObjectRefiner Modifies Only Detection Results**

    ObjectRefiner operations:

    - **Read** `image.objmask[:]` (binary mask) and `image.objmap[:]` (labeled map) from prior detection.
    - **Write** only `image.objmask[:]` and `image.objmap[:]` with refined results.
    - **Protect** `image.rgb`, `image.gray`, and `image.enh_gray` via automatic integrity validation
      (`@validate_operation_integrity` decorator).

    Any attempt to modify protected image components raises `OperationIntegrityError` when
    `VALIDATE_OPS=True` in the environment (enabled during development/testing).

    **Role in the Detection-to-Measurement Pipeline**

    ObjectRefiner sits after detection but before measurement:

    .. code-block:: text

        Raw Image (rgb, gray, enh_gray)
              ↓
        ImageEnhancer(s) → Improve visibility, reduce noise
              ↓
        ObjectDetector → Detect colonies/objects (initial, often noisy)
              ↓
        ObjectRefiner(s) → Clean up detections (optional but recommended)
              ↓
        MeasureFeatures → Extract colony properties
              ↓
        Analysis → Statistical phenotyping, clustering, growth curves

    When you call `refiner.apply(image)`, you get back an Image with refined `objmask` and `objmap`
    but identical preprocessing and image data—ready for downstream measurement and analysis.

    **Why Refinement Matters for Colony Phenotyping**

    Raw detections from ObjectDetector often contain artifacts:

    - **Spurious small objects:** Dust, sensor noise, agar texture, or salt-and-pepper thresholding
      artifacts create false-positive detections that bias colony counts and statistics.
    - **Fragmented colonies:** Uneven lighting, pigment heterogeneity, or aggressive thresholding
      fragments a single colony into multiple disconnected regions, inflating counts and distorting
      area measurements.
    - **Merged colonies:** In dense plates or when colonies touch, thresholding may merge adjacent
      colonies into a single detection, losing individuality and requiring post-hoc separation.
    - **Holes in masks:** Internal voids within colony masks (from glare or non-uniform pigmentation)
      create discontinuous shapes that confuse morphological measurements or downstream analysis.
    - **Border artifacts:** Colonies touching plate or well boundaries may be incomplete, biasing
      per-well phenotyping in high-throughput formats.

    Refinement operations target these issues with **domain-specific strategies**: morphological
    operations (erosion, dilation, opening, closing), shape filtering (circularity, solidity),
    size thresholding, and boundary enforcement to produce clean, valid detection results.

    **Differences: ObjectDetector vs ObjectRefiner**

    - **ObjectDetector:** Analyzes image data (grayscale, RGB, color spaces) and produces initial
      `objmask` and `objmap`. Input: enhanced image. Output: detection results. Typical use:
      thresholding, edge detection, peak finding, watershed segmentation.

    - **ObjectRefiner:** Modifies existing `objmask` and `objmap` without analyzing image data.
      Input: detection results. Output: refined detection results. Typical use: size filtering,
      morphological cleanup, shape filtering, merging/splitting objects, border removal.

    **When to Use ObjectRefiner vs Building Better ObjectDetector**

    Should you refine or improve the detector? Consider:

    - **Use ObjectRefiner if:**
      - The detector produces mostly correct detections but with manageable noise/artifacts
      - You can characterize the artifacts (small, fragmented, low-circularity, etc.)
      - Chaining simple refinement operations is clearer than tuning detector parameters
      - You want to compare cleanup strategies or enable parameter sweeps

    - **Improve ObjectDetector if:**
      - The detector fundamentally fails (misses most colonies, detects at wrong threshold)
      - Raw detections are too noisy to salvage through simple refinement
      - The problem is best solved through domain-specific detection logic, not post-hoc cleanup
      - You have labeled ground truth for detector optimization

    **Typical Refinement Strategies**

    Common ObjectRefiner implementations address specific issues:

    - **Size filtering:** Remove objects below/above size thresholds (e.g., `SmallObjectRemover`).
      Targets: spurious noise, dust, agar artifacts, or unrealistically large regions.

    - **Shape filtering:** Remove objects with poor morphology (low circularity, low solidity).
      Targets: elongated artifacts, merged colonies, debris. Example: `LowCircularityRemover`.

    - **Hole filling:** Fill holes within colony masks for solid shape representation (e.g., `MaskFill`).
      Targets: voids from uneven illumination, pigment patterns. Improves area measurements.

    - **Morphological operations:** Erosion, dilation, opening, closing to refine mask edges.
      Targets: fragmented edges, small protrusions, internal gaps. Uses `_make_footprint()`.

    - **Border removal:** Remove or exclude objects touching image/well boundaries.
      Targets: incomplete colonies in arrayed formats. Example: clear_border operations.

    - **Merging/splitting:** Combine nearby objects (dilation + relabeling) or separate touching regions
      (watershed, distance transform). Targets: fragmented colonies, merged colonies.

    **Integrity Validation: Protection of Core Data**

    ObjectRefiner uses the ``@validate_operation_integrity`` decorator on the ``apply()`` method
    to guarantee that preprocessing data are never modified:

    .. code-block:: python

        @validate_operation_integrity('image.rgb', 'image.gray', 'image.enh_gray')
        def apply(self, image: Image, inplace: bool = False) -> Image:
            return super().apply(image=image, inplace=inplace)

    This decorator:

    1. Calculates cryptographic signatures of `image.rgb`, `image.gray`, and `image.enh_gray`
       **before** processing
    2. Calls the parent `apply()` method to execute your `_operate()` implementation
    3. Recalculates signatures **after** operation completes
    4. Raises ``OperationIntegrityError`` if any protected component was modified

    **Note:** Integrity validation only runs if the ``VALIDATE_OPS=True`` environment variable
    is set (development-time safety; disabled in production for performance).

    **Implementing a Custom ObjectRefiner**

    Subclass ObjectRefiner and implement a single method:

    .. code-block:: python

        from phenotypic.abc_ import ObjectRefiner
        from phenotypic import Image
        from skimage.morphology import remove_small_objects

        class MyCustomRefiner(ObjectRefiner):
            def __init__(self, min_size: int = 50):
                super().__init__()
                self.min_size = min_size  # Instance attribute matched to _operate()

            @staticmethod
            def _operate(image: Image, min_size: int = 50) -> Image:
                # Modify ONLY objmap; read, process, write back
                # objmask will be auto-updated from objmap via relabel()
                refined_map = remove_small_objects(image.objmap[:], min_size=min_size)
                image.objmap[:] = refined_map
                return image

    **Key Rules for Implementation:**

    1. ``_operate()`` must be **static** (required for parallel execution in pipelines).
    2. All parameters except `image` must exist as instance attributes with matching names
       (enables automatic parameter matching via `_get_matched_operation_args()`).
    3. **Only modify ``image.objmask[:]`` and ``image.objmap[:]``**—all other components are
       protected. Reading image data is allowed but modifications will trigger integrity errors.
    4. Always use the accessor pattern: ``image.objmap[:] = new_data`` (never direct attribute
       assignment).
    5. Return the modified Image object.

    **Modifying objmask and objmap**

    Within your `_operate()` method, use the accessor interface to read and write detection results:

    .. code-block:: python

        # Reading detection data
        mask = image.objmask[:]          # Binary mask (True = object)
        objmap = image.objmap[:]         # Labeled map (0 = background, 1+ = object label)
        objects = image.objects          # High-level ObjectCollection interface

        # Modifying detection data
        image.objmask[:] = refined_mask  # Full replacement of binary mask
        image.objmap[:] = refined_map    # Full replacement of labeled map

        # Partial updates (boolean indexing)
        # Mark certain labels as background (set to 0)
        keep_labels = [1, 3, 5]  # Labels to retain
        filtered_map = np.where(np.isin(objmap, keep_labels), objmap, 0)
        image.objmap[:] = filtered_map

    **Relationship Between objmask and objmap**

    - **objmap (labeled map):** Each pixel contains the object label (0 = background, 1+ = object ID).
      Authoritative source of truth; defines which pixels belong to which colony.

    - **objmask (binary mask):** Simple binary version of objmap; True where objmap > 0, False elsewhere.
      Derived from objmap via `image.objmap.relabel()`.

    When you modify objmap, objmask is automatically updated. When you modify objmask directly,
    call `image.objmap.relabel()` to ensure consistency (or reconstruct objmap from objmask via
    connected-component labeling).

    **The _make_footprint() Static Utility**

    ObjectRefiner provides a static helper for generating morphological structuring elements
    (footprints) used in erosion, dilation, and other morphological operations:

    .. code-block:: python

        @staticmethod
        def _make_footprint(shape: Literal["square", "diamond", "disk"], radius: int) -> np.ndarray:
            '''Creates a binary morphological footprint for image processing.'''

    **Footprint Shapes and When to Use Each**

    - **"disk":** Circular/isotropic footprint. Best for preserving rounded colony shapes and
      applying uniform processing in all directions. Use for: general-purpose morphology (dilation
      to merge fragments, erosion to remove noise), operations that respect colony roundness.

    - **"square":** Square footprint with 8-connectivity. Emphasizes horizontal/vertical edges
      and aligns with pixel grid. Use for: grid-aligned artifacts, operations aligned with imaging
      hardware, when processing speed matters (slightly faster than disk).

    - **"diamond":** Diamond-shaped (rotated square) footprint with 4-connectivity. Creates a
      cross-like neighborhood pattern. Use for: specialized cases where diagonal connections should
      be de-emphasized; less common in practice.

    **The radius parameter** controls the neighborhood size (in pixels). Larger radii affect more
    neighbors and produce broader morphological effects (merge more fragments, remove larger noise,
    but risk bridging adjacent colonies). Choose radius smaller than minimum inter-colony spacing
    to avoid creating false merges.

    **Common Morphological Refinement Patterns**

    Use `_make_footprint()` with morphological operations from `scipy.ndimage` or `skimage.morphology`:

    .. code-block:: python

        from scipy.ndimage import binary_dilation, binary_erosion
        from skimage.morphology import binary_closing, binary_opening
        from phenotypic.abc_ import ObjectRefiner

        disk_fp = ObjectRefiner._make_footprint('disk', radius=3)

        # Dilation: expand object regions (merge fragmented colonies)
        dilated_mask = binary_dilation(binary_mask, structure=disk_fp)

        # Erosion: shrink object regions (remove thin protrusions, small noise)
        eroded_mask = binary_erosion(binary_mask, structure=disk_fp)

        # Closing: dilation then erosion (fill small holes)
        closed_mask = binary_closing(binary_mask, structure=disk_fp)

        # Opening: erosion then dilation (remove small noise)
        opened_mask = binary_opening(binary_mask, structure=disk_fp)

    **Chaining Multiple Refinements**

    Refinement operations are typically chained to address multiple issues in sequence:

    .. code-block:: python

        from phenotypic import Image, ImagePipeline
        from phenotypic.refine import SmallObjectRemover, MaskFill, LowCircularityRemover

        # Build a refinement pipeline
        pipeline = ImagePipeline()
        pipeline.add(SmallObjectRemover(min_size=100))          # Remove dust/noise
        pipeline.add(MaskFill())                                 # Fill holes in colonies
        pipeline.add(LowCircularityRemover(cutoff=0.75))        # Remove elongated artifacts

        # Apply to detected image
        image = Image.from_image_path('plate.jpg')
        from phenotypic.detect import OtsuDetector
        detected = OtsuDetector().apply(image)

        # Refine
        refined = pipeline.operate([detected])[0]
        colonies = refined.objects
        print(f"After refinement: {len(colonies)} colonies")

    **Rationale for chaining:**

    - **Order matters:** Remove small noise before filling holes (no point filling tiny artifacts).
      Remove low-circularity objects before morphological operations (cleaner starting point).
    - **Divide and conquer:** One refiner per issue (size, shape, holes, borders) is clearer than
      monolithic operations.
    - **No data loss:** Original detection and image data are preserved, so intermediate steps can
      be inspected and validated.
    - **Reproducibility:** Chained operations can be serialized to YAML for documentation and reuse.

    **Methods and Attributes**

    Attributes:
        None at the ObjectRefiner level; subclasses define refinement parameters
        as instance attributes (e.g., min_size, cutoff, radius).

    Methods:
        apply(image, inplace=False): Applies the refinement to an image. Returns a modified
            Image with refined `objmask` and `objmap` but unchanged RGB/gray/enh_gray. Handles
            copy/inplace logic and validates data integrity.
        _operate(image, **kwargs): Abstract static method implemented by subclasses.
            Performs the actual refinement algorithm. Parameters are automatically matched
            to instance attributes.
        _make_footprint(shape, radius): Static utility that creates a binary morphological
            footprint (disk, square, or diamond) for use in morphological operations.

    Notes:
        - **Protected components:** The ``@validate_operation_integrity`` decorator ensures
          that ``image.rgb``, ``image.gray``, and ``image.enh_gray`` cannot be modified.
          Only ``image.objmask`` and ``image.objmap`` can be changed.

        - **Immutability by default:** ``apply(image)`` returns a modified copy by default.
          Set ``inplace=True`` for memory-efficient in-place modification.

        - **Static _operate() requirement:** The ``_operate()`` method must be static to
          support parallel execution in pipelines.

        - **Parameter matching for parallelization:** All ``_operate()`` parameters except
          ``image`` must exist as instance attributes. When ``apply()`` is called, these
          values are extracted and passed to ``_operate()``.

        - **Accessor pattern:** Always use ``image.objmap[:] = new_data`` to modify
          object maps. Never use direct attribute assignment.

        - **objmap/objmask consistency:** When modifying objmap, call `image.objmap.relabel()`
          to ensure objmask is updated. When modifying objmask directly, reconstruct objmap
          via connected-component labeling.

        - **Boolean indexing for filtering:** Use numpy boolean arrays to filter labels:
          ``mask = np.isin(objmap, keep_labels); filtered_map = objmap * mask``

    Examples:
        .. dropdown:: Removing small spurious objects below minimum size

            .. code-block:: python

                from phenotypic.abc_ import ObjectRefiner
                from phenotypic import Image
                from skimage.morphology import remove_small_objects
                from scipy import ndimage

                class SimpleSmallObjectRemover(ObjectRefiner):
                    '''Remove objects smaller than a minimum size threshold.'''

                    def __init__(self, min_size: int = 50):
                        super().__init__()
                        self.min_size = min_size

                    @staticmethod
                    def _operate(image: Image, min_size: int = 50) -> Image:
                        '''Remove small objects from labeled map.'''
                        # Get current labeled map
                        objmap = image.objmap[:]

                        # Remove small objects (automatically updates objmap)
                        refined = remove_small_objects(objmap, min_size=min_size)

                        # Set refined result
                        image.objmap[:] = refined
                        return image

                # Usage
                from phenotypic.detect import OtsuDetector

                image = Image.from_image_path('plate.jpg')
                detected = OtsuDetector().apply(image)

                # Remove noise below 100 pixels
                refiner = SimpleSmallObjectRemover(min_size=100)
                cleaned = refiner.apply(detected)

                print(f"Before: {detected.objmap[:].max()} objects")
                print(f"After: {cleaned.objmap[:].max()} objects")

        .. dropdown:: Removing low-circularity objects (merged colonies, artifacts)

            .. code-block:: python

                from phenotypic.abc_ import ObjectRefiner
                from phenotypic import Image
                from skimage.measure import regionprops_table
                import pandas as pd
                import numpy as np
                import math

                class CircularityFilter(ObjectRefiner):
                    '''Remove objects with low circularity (merged colonies, artifacts).'''

                    def __init__(self, min_circularity: float = 0.7):
                        super().__init__()
                        self.min_circularity = min_circularity

                    @staticmethod
                    def _operate(image: Image, min_circularity: float = 0.7) -> Image:
                        '''Filter objects by circularity using Polsby-Popper metric.'''
                        objmap = image.objmap[:]

                        # Measure shape properties
                        props = regionprops_table(
                            label_image=objmap,
                            properties=['label', 'area', 'perimeter']
                        )
                        df = pd.DataFrame(props)

                        # Calculate circularity (Polsby-Popper: 4*pi*area / perimeter^2)
                        df['circularity'] = (4 * math.pi * df['area']) / (df['perimeter'] ** 2)

                        # Keep only circular objects
                        keep_labels = df[df['circularity'] >= min_circularity]['label'].values

                        # Filter map: keep only selected labels
                        refined_map = np.where(np.isin(objmap, keep_labels), objmap, 0)
                        image.objmap[:] = refined_map
                        return image

                # Usage
                image = Image.from_image_path('plate.jpg')
                from phenotypic.detect import OtsuDetector
                detected = OtsuDetector().apply(image)

                # Keep only well-formed circular colonies
                refiner = CircularityFilter(min_circularity=0.75)
                refined = refiner.apply(detected)

                print(f"Removed elongated artifacts: {detected.objmap[:].max()} -> {refined.objmap[:].max()}")

        .. dropdown:: Filling holes in colony masks for solid shape representation

            .. code-block:: python

                from phenotypic.abc_ import ObjectRefiner
                from phenotypic import Image
                from scipy.ndimage import binary_fill_holes

                class HoleFiller(ObjectRefiner):
                    '''Fill holes within colony masks for solid shape representation.'''

                    def __init__(self):
                        super().__init__()

                    @staticmethod
                    def _operate(image: Image) -> Image:
                        '''Fill holes in binary mask.'''
                        mask = image.objmask[:]

                        # Fill holes (interior voids within objects)
                        filled = binary_fill_holes(mask)

                        # Update mask
                        image.objmask[:] = filled

                        # Reconstruct labeled map from filled mask
                        from scipy import ndimage
                        labeled, _ = ndimage.label(filled)
                        image.objmap[:] = labeled

                        return image

                # Usage
                image = Image.from_image_path('plate.jpg')
                from phenotypic.detect import OtsuDetector
                detected = OtsuDetector().apply(image)

                # Fill holes from uneven illumination or pigmentation
                refiner = HoleFiller()
                refined = refiner.apply(detected)

                # Result: solid, contiguous colony shapes better for area measurements
                print(f"Holes filled; colonies now solid")

        .. dropdown:: Morphological refinement with dilation to merge fragmented colonies

            .. code-block:: python

                from phenotypic.abc_ import ObjectRefiner
                from phenotypic import Image
                from scipy.ndimage import binary_dilation, label as ndi_label
                import numpy as np

                class FragmentMerger(ObjectRefiner):
                    '''Merge fragmented colonies via morphological dilation and relabeling.'''

                    def __init__(self, dilation_radius: int = 2):
                        super().__init__()
                        self.dilation_radius = dilation_radius

                    @staticmethod
                    def _operate(image: Image, dilation_radius: int = 2) -> Image:
                        '''Dilate mask and relabel to merge nearby fragments.'''
                        mask = image.objmask[:]

                        # Create disk footprint for isotropic dilation
                        fp = ObjectRefiner._make_footprint('disk', dilation_radius)

                        # Dilate to bridge fragmented regions
                        dilated = binary_dilation(mask, structure=fp)

                        # Relabel connected components
                        relabeled, _ = ndi_label(dilated)

                        # Set refined results
                        image.objmask[:] = dilated
                        image.objmap[:] = relabeled
                        return image

                # Usage
                image = Image.from_image_path('plate.jpg')
                from phenotypic.detect import OtsuDetector
                detected = OtsuDetector().apply(image)

                # Merge fragments from uneven lighting
                refiner = FragmentMerger(dilation_radius=3)
                merged = refiner.apply(detected)

                print(f"Merged fragments: {detected.objmap[:].max()} -> {merged.objmap[:].max()} objects")

        .. dropdown:: Chaining multiple refinements in a pipeline

            .. code-block:: python

                from phenotypic import Image, ImagePipeline
                from phenotypic.enhance import GaussianBlur
                from phenotypic.detect import OtsuDetector
                from phenotypic.refine import (
                    SmallObjectRemover, MaskFill, LowCircularityRemover
                )
                from phenotypic.measure import MeasureColor

                # Build complete processing pipeline with enhancement, detection, and refinement
                pipeline = ImagePipeline()

                # Preprocessing
                pipeline.add(GaussianBlur(sigma=1.5))

                # Detection
                pipeline.add(OtsuDetector())

                # Refinement (chain multiple cleanup operations)
                pipeline.add(SmallObjectRemover(min_size=100))          # Remove dust
                pipeline.add(MaskFill())                                 # Fill internal holes
                pipeline.add(LowCircularityRemover(cutoff=0.75))        # Remove merged/irregular

                # Measurement
                pipeline.add(MeasureColor())

                # Load images and process
                image = Image.from_image_path('plate.jpg')
                results = pipeline.operate([image])
                final = results[0]

                # Access final clean detection results
                colonies = final.objects
                measurements = final.measurements

                print(f"Detected and cleaned: {len(colonies)} colonies")
                print(f"Color measurements: {measurements.shape}")
    """

    @validate_operation_integrity("image.rgb", "image.gray", "image.enh_gray")
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
                return square(width=radius * 2)
            case "diamond":
                return diamond(radius=radius)
            case "disk":
                return disk(radius=radius)
            case _:
                raise ValueError(f"Unknown shape: {shape}")
