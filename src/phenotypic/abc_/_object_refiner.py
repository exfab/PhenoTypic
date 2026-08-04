from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar, overload

if TYPE_CHECKING:
    from phenotypic._core._image import Image
    from phenotypic._core._grid_image import GridImage

from ._image_operation import ImageOperation
from phenotypic.sdk_.funcs_ import validate_operation_integrity
from abc import ABC, abstractmethod


# <<Interface>>
class ObjectRefiner(ImageOperation, ABC):
    """Abstract base class for post-detection refinement operations that modify object masks and maps.

    ObjectRefiner is the foundation for all post-detection cleanup algorithms that refine colony
    detections through morphological operations, filtering, and merging. Unlike ObjectDetector
    (which analyzes image data to create initial detections), ObjectRefiner only modifies the
    object mask and labeled map, leaving preprocessing data untouched.

    **Quick Decision Guide: ObjectRefiner vs Alternatives**

    - **Use ObjectRefiner if:** Detector produces mostly correct detections with manageable noise/artifacts
      (small objects, fragmented regions, holes, low circularity) that can be characterized and filtered.
    - **Use ObjectDetector if:** Detector fundamentally fails to detect colonies or produces too much noise
      to salvage via post-hoc cleanup.
    - **Use ImageEnhancer if:** Problem is image quality (blur, contrast, noise) affecting detection;
      improve input before detection rather than refining output.
    - **ObjectRefiner vs ObjectDetector:** Refiners work on existing masks (objmask/objmap), detectors create
      masks from image data. Refiners are for cleanup, detectors are for initial analysis.
    - **Size filtering:** Use for removing dust, noise, agar artifacts (too small) or unrealistic regions
      (too large). Example: [SmallObjectRemover](src/phenotypic/refine/_small_object_remover.py).
    - **Morphological cleanup:** Use for fragmented edges, thin protrusions, internal gaps. Example:
      [MaskDilation](src/phenotypic/refine/_mask_dilation.py) (uses FootprintMixin).
    - **Hole filling:** Use for voids from uneven illumination or pigment patterns within colonies.
    - **Shape filtering:** Use for removing elongated artifacts, merged colonies, low-circularity debris.
    - **Merging operations:** Use for bridging fragmented colonies or combining nearby regions. Example:
      [NearestNeighborMerger](src/phenotypic/refine/_nearest_neighbor_merger.py).
    - **When to chain:** Combine multiple refiners in ImagePipeline (remove small noise before filling holes,
      filter shapes before morphological operations) for clearer, divide-and-conquer approach.

    **What is ObjectRefiner?**

    ObjectRefiner operates on the principle of **non-destructive post-processing**: all modifications
    are applied only to `image.objmask` (binary mask) and `image.objmap` (labeled map), while original
    image components (`image.rgb`, `image.gray`, `image.detect_mat`) remain protected and unchanged.
    This allows you to experiment with multiple refinement chains without affecting raw or enhanced
    image data, ensuring reproducibility and enabling comparison of different cleanup strategies.

    **Key Principle: ObjectRefiner Modifies Only Detection Results**

    ObjectRefiner operations:

    - **Read** `image.objmask[:]` (binary mask) and `image.objmap[:]` (labeled map) from prior detection.
    - **Write** only `image.objmask[:]` and `image.objmap[:]` with refined results.
    - **Protect** `image.rgb`, `image.gray`, and `image.detect_mat` via automatic integrity validation
      (`@validate_operation_integrity` decorator).

    Any attempt to modify protected image components raises `OperationIntegrityError` when
    ``phenotypic.settings.VALIDATE_OPS`` is true (enabled during development/testing).

    **Role in the Detection-to-Measurement Pipeline**

    ObjectRefiner sits after detection but before measurement:

    .. code-block:: text

        Raw Image (rgb, gray, detect_mat)
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

    - **Size filtering:** [SmallObjectRemover](src/phenotypic/refine/_small_object_remover.py) removes
      objects below/above thresholds. Targets: spurious noise, dust, agar artifacts, oversized regions.

    - **Shape filtering:** Remove objects with poor morphology (low circularity, low solidity, high
      aspect ratio). Targets: elongated artifacts, merged colonies, debris.

    - **Hole filling:** Fill interior voids within colony masks for solid shape representation.
      Targets: voids from uneven illumination, pigment heterogeneity. Improves area measurements.

    - **Morphological operations:** Erosion, dilation, opening, closing with [MaskDilation](src/phenotypic/refine/_mask_dilation.py),
      [MaskErosion](src/phenotypic/refine/_mask_erosion.py), [MaskOpening](src/phenotypic/refine/_mask_opening.py).
      Targets: fragmented edges, thin protrusions, internal gaps. Uses FootprintMixin for shape control.

    - **Border removal:** Remove or exclude objects touching image/well boundaries.
      Targets: incomplete colonies in arrayed formats.

    - **Merging/splitting:** [NearestNeighborMerger](src/phenotypic/refine/_nearest_neighbor_merger.py)
      combines nearby objects via dilation and relabeling. Targets: fragmented colonies, nearby regions.

    **Integrity Validation: Protection of Core Data**

    ObjectRefiner uses the ``@validate_operation_integrity`` decorator on the ``apply()`` method
    to guarantee that preprocessing data are never modified:

    .. code-block:: python

        @validate_operation_integrity('image.rgb', 'image.gray', 'image.detect_mat')
        def apply(self, image: Image, inplace: bool = False) -> Image:
            return super().apply(image=image, inplace=inplace)

    This decorator:

    1. Calculates cryptographic signatures of `image.rgb`, `image.gray`, and `image.detect_mat`
       **before** processing
    2. Calls the parent `apply()` method to execute your `_operate()` implementation
    3. Recalculates signatures **after** operation completes
    4. Raises ``OperationIntegrityError`` if any protected component was modified

    **Note:** Integrity validation only runs if ``phenotypic.settings.VALIDATE_OPS`` is true
    (development-time safety; disabled in production for performance).

    **Implementing a Custom ObjectRefiner**

    Subclass ObjectRefiner and implement a single method:

    .. code-block:: python

        from phenotypic.abc_ import ObjectRefiner
        from phenotypic import Image
        from skimage.morphology import remove_small_objects

        class MyCustomRefiner(ObjectRefiner):
            min_size: int = 50  # Annotated class-level field

            def _operate(self, image: Image) -> Image:
                # Modify ONLY objmap; read, process, write back
                # objmask will be auto-updated from objmap via relabel()
                refined_map = remove_small_objects(
                    image.objmap[:], min_size=self.min_size
                )
                image.objmap[:] = refined_map
                return image

    **Morphological Operations with FootprintMixin**

    For operations requiring morphological structuring elements (dilation, erosion, opening, closing),
    inherit from FootprintMixin. See [MaskDilation](src/phenotypic/refine/_mask_dilation.py) for example:

    .. code-block:: python

        from phenotypic.abc_ import ObjectRefiner
        from phenotypic.sdk_ import FootprintMixin
        from phenotypic import Image
        from skimage.morphology import dilation

        class MyMorphRefiner(ObjectRefiner, FootprintMixin):
            footprint_shape: str = 'disk'  # Annotated class-level fields
            footprint_width: int = 2

            def _operate(self, image: Image) -> Image:
                # Use _make_footprint from ObjectRefiner or FootprintMixin
                fp = ObjectRefiner._make_footprint(
                    self.footprint_shape, self.footprint_width
                )
                dilated = dilation(image.objmask[:], footprint=fp)
                image.objmask[:] = dilated
                # Reconstruct objmap from dilated mask
                from scipy.ndimage import label as ndi_label
                relabeled, _ = ndi_label(dilated)
                image.objmap[:] = relabeled
                return image

    **Key Rules for Implementation:**

    1. ``_operate()`` must be an **instance method** (access parameters via ``self``).
    2. All parameters except `image` must be declared as annotated class-level fields.
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
        def _make_footprint(shape: Literal["square", "diamond", "disk"], width: int) -> np.ndarray:
            '''Creates a binary morphological shape for image processing.'''

    **Footprint Shapes and When to Use Each**

    - **"disk":** Circular/isotropic shape. Best for preserving rounded colony shapes and
      applying uniform processing in all directions. Use for: general-purpose morphology (dilation
      to merge fragments, erosion to remove noise), operations that respect colony roundness.

    - **"square":** Square shape with 8-connectivity. Emphasizes horizontal/vertical edges
      and aligns with pixel grid. Use for: grid-aligned artifacts, operations aligned with imaging
      hardware, when processing speed matters (slightly faster than disk).

    - **"diamond":** Diamond-shaped (rotated square) shape with 4-connectivity. Creates a
      cross-like neighborhood pattern. Use for: specialized cases where diagonal connections should
      be de-emphasized; less common in practice.

    **The width parameter** controls the neighborhood size (in pixels). Larger radii affect more
    neighbors and produce broader morphological effects (merge more fragments, remove larger noise,
    but risk bridging adjacent colonies). Choose width smaller than minimum inter-colony spacing
    to avoid creating false merges.

    **Common Morphological Refinement Patterns**

    Use `_make_footprint()` with morphological operations from `skimage.morphology`:

    .. code-block:: python

        from skimage.morphology import dilation, erosion, closing, opening
        from phenotypic.abc_ import ObjectRefiner

        disk_fp = ObjectRefiner._make_footprint('disk', width=3)

        # Dilation: expand object regions (merge fragmented colonies)
        dilated_mask = dilation(binary_mask, footprint=disk_fp)

        # Erosion: shrink object regions (remove thin protrusions, small noise)
        eroded_mask = erosion(binary_mask, footprint=disk_fp)

        # Closing: dilation then erosion (fill small holes)
        closed_mask = closing(binary_mask, footprint=disk_fp)

        # Opening: erosion then dilation (remove small noise)
        opened_mask = opening(binary_mask, footprint=disk_fp)

    **Chaining Multiple Refinements**

    Refinement operations are typically chained to address multiple issues in sequence:

    .. code-block:: python

        from phenotypic import Image, ImagePipeline
        from phenotypic.refine import SmallObjectRemover, MaskFill, RemoveLowCircularity

        # Build a refinement pipeline
        pipeline = ImagePipeline()
        pipeline.add(SmallObjectRemover(min_size=100))          # Remove dust/noise
        pipeline.add(MaskFill())                                 # Fill holes in colonies
        pipeline.add(RemoveLowCircularity(cutoff=0.75))        # Remove elongated artifacts

        # Apply to detected image
        image = Image.imread('plate.jpg')
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
        as instance attributes (e.g., min_size, cutoff, width).

    Methods:
        apply(image, inplace=False): Applies the refinement to an image. Returns a modified
            Image with refined `objmask` and `objmap` but unchanged RGB/gray/detect_mat. Handles
            copy/inplace logic and validates data integrity.
        _operate(image, **kwargs): Abstract instance method implemented by subclasses.
            Performs the actual refinement algorithm. Access parameters via ``self``.
        _make_footprint(shape, width): Static utility that creates a binary morphological
            shape (disk, square, or diamond) for use in morphological operations.

    Notes:
        - **Protected components:** The ``@validate_operation_integrity`` decorator ensures
          that ``image.rgb``, ``image.gray``, and ``image.detect_mat`` cannot be modified.
          Only ``image.objmask`` and ``image.objmap`` can be changed.

        - **Immutability by default:** ``apply(image)`` returns a modified copy by default.
          Set ``inplace=True`` for memory-efficient in-place modification.

        - **Instance _operate() method:** The ``_operate()`` method is an instance method;
          access parameters via ``self``.

        - **Field-based parameters:** All ``_operate()`` parameters except ``image``
          are declared as annotated class-level fields. ``_operate()`` accesses them
          via ``self``; pydantic handles construction and serialization.

        - **Accessor pattern:** Always use ``image.objmap[:] = new_data`` to modify
          object maps. Never use direct attribute assignment.

        - **objmap/objmask consistency:** When modifying objmap, call `image.objmap.relabel()`
          to ensure objmask is updated. When modifying objmask directly, reconstruct objmap
          via connected-component labeling.

        - **Boolean indexing for filtering:** Use numpy boolean arrays to filter labels:
          ``mask = np.isin(objmap, keep_labels); filtered_map = objmap * mask``

    Examples:
        Removing small spurious objects below minimum size:

        >>> from phenotypic.abc_ import ObjectRefiner
        >>> from phenotypic import Image
        >>> from skimage.morphology import remove_small_objects
        >>> from scipy import ndimage
        >>> class SimpleSmallObjectRemover(ObjectRefiner):
        ...     '''Remove objects smaller than a minimum size threshold.'''
        ...
        ...     min_size: int = 50
        ...
        ...     def _operate(self, image: Image) -> Image:
        ...         '''Remove small objects from labeled map.'''
        ...         # Get current labeled map
        ...         objmap = image.objmap[:]
        ...         # Remove small objects (automatically updates objmap)
        ...         refined = remove_small_objects(objmap, min_size=self.min_size)
        ...         # Set refined result
        ...         image.objmap[:] = refined
        ...         return image
        >>> # Usage
        >>> from phenotypic.detect import OtsuDetector
        >>> image = Image.imread('plate.jpg')
        >>> detected = OtsuDetector().apply(image)
        >>> # Remove noise below 100 pixels
        >>> refiner = SimpleSmallObjectRemover(min_size=100)
        >>> cleaned = refiner.apply(detected)
        >>> print(f"Before: {detected.objmap[:].max()} objects")
        >>> print(f"After: {cleaned.objmap[:].max()} objects")

        Removing low-circularity objects (merged colonies, artifacts):

        >>> from phenotypic.abc_ import ObjectRefiner
        >>> from phenotypic import Image
        >>> from skimage.measure import regionprops_table
        >>> import pandas as pd
        >>> import numpy as np
        >>> import math
        >>> class CircularityFilter(ObjectRefiner):
        ...     '''Remove objects with low circularity (merged colonies, artifacts).'''
        ...
        ...     min_circularity: float = 0.7
        ...
        ...     def _operate(self, image: Image) -> Image:
        ...         '''Filter objects by circularity using Polsby-Popper metric.'''
        ...         objmap = image.objmap[:]
        ...         # Measure shape properties
        ...         props = regionprops_table(
        ...             label_image=objmap,
        ...             properties=['label', 'area', 'perimeter']
        ...         )
        ...         df = pd.DataFrame(props)
        ...         # Calculate circularity (Polsby-Popper: 4*pi*area / perimeter^2)
        ...         df['circularity'] = (4 * math.pi * df['area']) / (df['perimeter'] ** 2)
        ...         # Keep only circular objects
        ...         keep_labels = df[
        ...             df['circularity'] >= self.min_circularity
        ...         ]['label'].values
        ...         # Filter map: keep only selected labels
        ...         refined_map = np.where(np.isin(objmap, keep_labels), objmap, 0)
        ...         image.objmap[:] = refined_map
        ...         return image
        >>> # Usage
        >>> image = Image.imread('plate.jpg')
        >>> from phenotypic.detect import OtsuDetector
        >>> detected = OtsuDetector().apply(image)
        >>> # Keep only well-formed circular colonies
        >>> refiner = CircularityFilter(min_circularity=0.75)
        >>> refined = refiner.apply(detected)
        >>> print(f"Removed elongated artifacts: {detected.objmap[:].max()} -> {refined.objmap[:].max()}")

        Filling holes in colony masks for solid shape representation:

        >>> from phenotypic.abc_ import ObjectRefiner
        >>> from phenotypic import Image
        >>> from scipy.ndimage import binary_fill_holes
        >>> class HoleFiller(ObjectRefiner):
        ...     '''Fill holes within colony masks for solid shape representation.'''
        ...
        ...     def _operate(self, image: Image) -> Image:
        ...         '''Fill holes in binary mask.'''
        ...         mask = image.objmask[:]
        ...         # Fill holes (interior voids within objects)
        ...         filled = binary_fill_holes(mask)
        ...         # Update mask
        ...         image.objmask[:] = filled
        ...         # Reconstruct labeled map from filled mask
        ...         from scipy import ndimage
        ...         labeled, _ = ndimage.label(filled)
        ...         image.objmap[:] = labeled
        ...         return image
        >>> # Usage
        >>> image = Image.imread('plate.jpg')
        >>> from phenotypic.detect import OtsuDetector
        >>> detected = OtsuDetector().apply(image)
        >>> # Fill holes from uneven illumination or pigmentation
        >>> refiner = HoleFiller()
        >>> refined = refiner.apply(detected)
        >>> # Result: solid, contiguous colony shapes better for area measurements
        >>> print(f"Holes filled; colonies now solid")

        Morphological refinement with dilation to merge fragmented colonies:

        >>> from phenotypic.abc_ import ObjectRefiner
        >>> from phenotypic import Image
        >>> from scipy.ndimage import label as ndi_label
        >>> from skimage.morphology import dilation
        >>> import numpy as np
        >>> class FragmentMerger(ObjectRefiner):
        ...     '''Merge fragmented colonies via morphological dilation and relabeling.'''
        ...
        ...     dilation_radius: int = 2
        ...
        ...     def _operate(self, image: Image) -> Image:
        ...         '''Dilate mask and relabel to merge nearby fragments.'''
        ...         mask = image.objmask[:]
        ...         # Create disk shape for isotropic dilation
        ...         fp = ObjectRefiner._make_footprint('disk', self.dilation_radius)
        ...         # Dilate to bridge fragmented regions
        ...         dilated = dilation(mask, footprint=fp)
        ...         # Relabel connected components
        ...         relabeled, _ = ndi_label(dilated)
        ...         # Set refined results
        ...         image.objmask[:] = dilated
        ...         image.objmap[:] = relabeled
        ...         return image
        >>> # Usage
        >>> image = Image.imread('plate.jpg')
        >>> from phenotypic.detect import OtsuDetector
        >>> detected = OtsuDetector().apply(image)
        >>> # Merge fragments from uneven lighting
        >>> refiner = FragmentMerger(dilation_radius=3)
        >>> merged = refiner.apply(detected)
        >>> print(f"Merged fragments: {detected.objmap[:].max()} -> {merged.objmap[:].max()} objects")

        Merging nearby objects via nearest-neighbor distance:

        >>> from phenotypic.abc_ import ObjectRefiner
        >>> from phenotypic import Image
        >>> from scipy.ndimage import label as ndi_label, distance_transform_edt
        >>> from skimage.morphology import dilation
        >>> import numpy as np
        >>> class MergeFragmentChains(ObjectRefiner):
        ...     '''Merge objects within specified distance via distance transform.'''
        ...
        ...     merge_distance: int = 5
        ...
        ...     def _operate(self, image: Image) -> Image:
        ...         '''Merge objects closer than merge_distance via dilation of distance map.'''
        ...         mask = image.objmask[:]
        ...         # Compute distance transform from object interior
        ...         dist_map = distance_transform_edt(mask)
        ...         # Dilate distance map to bridge nearby objects
        ...         fp = ObjectRefiner._make_footprint('disk', self.merge_distance)
        ...         dilated_dist = dilation(dist_map > 0, footprint=fp)
        ...         # Relabel connected components in dilated region
        ...         relabeled, _ = ndi_label(dilated_dist)
        ...         # Set refined results
        ...         image.objmask[:] = dilated_dist
        ...         image.objmap[:] = relabeled
        ...         return image
        >>> # Usage: merge nearby fragments from partial colonies
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.detect import OtsuDetector
        >>> image = load_synth_yeast_plate()
        >>> detected = OtsuDetector().apply(image)
        >>> # Merge fragments within 10-pixel distance
        >>> merger = MergeFragmentChains(merge_distance=10)
        >>> merged = merger.apply(detected)
        >>> print(f"Merged nearby objects: {detected.objmap[:].max()} -> {merged.objmap[:].max()}")

        Chaining multiple refinements in a pipeline:

        >>> from phenotypic import Image, ImagePipeline
        >>> from phenotypic.enhance import BlurGauss
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.refine import (
        ...     SmallObjectRemover, MaskFill, RemoveLowCircularity
        ... )
        >>> from phenotypic.measure import MeasureColor
        >>> # Build complete processing pipeline with enhancement, detection, and refinement
        >>> pipeline = ImagePipeline()
        >>> # Preprocessing
        >>> pipeline.add(BlurGauss(sigma=1.5))
        >>> # Detection
        >>> pipeline.add(OtsuDetector())
        >>> # Refinement (chain multiple cleanup operations)
        >>> pipeline.add(SmallObjectRemover(min_size=100))          # Remove dust
        >>> pipeline.add(MaskFill())                                 # Fill internal holes
        >>> pipeline.add(RemoveLowCircularity(cutoff=0.75))        # Remove merged/irregular
        >>> # Measurement
        >>> pipeline.add(MeasureColor())
        >>> # Load images and process
        >>> image = Image.imread('plate.jpg')
        >>> results = pipeline.operate([image])
        >>> final = results[0]
        >>> # Access final clean detection results
        >>> colonies = final.objects
        >>> measurements = final.measurements
        >>> print(f"Detected and cleaned: {len(colonies)} colonies")
        >>> print(f"Color measurements: {measurements.shape}")
    """
    _footprint_shapes: ClassVar[set[str]] = {"square", "diamond", "disk"}

    @overload
    def apply(self, image: GridImage, inplace: bool = False) -> GridImage: ...

    @overload
    def apply(self, image: Image, inplace: bool = False) -> Image: ...

    @validate_operation_integrity("image.rgb", "image.gray", "image.detect_mat")
    def apply(self, image: Image, inplace: bool = False) -> Image:
        return super().apply(image=image, inplace=inplace)

    @abstractmethod
    def _operate(self, image: Image) -> Image:
        return image
