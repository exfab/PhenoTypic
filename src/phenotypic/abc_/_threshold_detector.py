from __future__ import annotations
from typing import TYPE_CHECKING, overload
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from phenotypic._core._image import Image
    from phenotypic._core._grid_image import GridImage

from ._object_detector import ObjectDetector


# <<Interface>>
class ThresholdDetector(ObjectDetector, ABC):
    """Marker ABC for threshold-based colony detection strategies.

    ThresholdDetector specializes ObjectDetector for algorithms that detect colonies
    by converting grayscale intensity to a binary mask via thresholding. Unlike edge-based
    (Canny) or peak-based (RoundPeaks) approaches, thresholding works by partitioning
    intensity space: pixels above a threshold value become foreground (colonies), pixels
    below become background.

    **Quick Decision Guide**

    Choose your detection strategy based on image characteristics:

    - **Threshold-based:** Clear intensity separation between colonies and background?
      Try global threshold (Otsu, Yen) or local adaptive (block-based) thresholding.
    - **Edge-based (CannyDetector):** Faint or merged colonies? Canny edge detection finds
      boundaries where gradient is high; invert edges to label regions.
    - **Peak-based (RoundPeaksDetector):** Well-separated round colonies? Peak detection
      assumes circular shapes and grows from intensity maxima.
    - **Subclass decision:** Is your algorithm threshold-based? Subclass ThresholdDetector.
      Otherwise subclass ObjectDetector directly.
    - **Local vs global threshold:** Uneven illumination? Local (adaptive) thresholding adjusts
      per neighborhood; global methods fail on gradient-heavy plates.
    - **Advanced strategy:** Need dual-threshold with edge tracking? See [HysteresisDetector](src/phenotypic/detect/_hysteresis_detector.py).

    **Why threshold-based detection?**

    Thresholding is ideal when:

    - **Clear intensity separation:** Colonies have distinctly different intensity than
      background (common on high-contrast agar plates or with good lighting).
    - **Simplicity and speed:** Single-pass algorithms (no iterative edge tracking or
      distance computation).
    - **Robustness to morphology:** Works equally well on round and irregular colonies
      (unlike peak-based approaches that assume circular shapes).
    - **Well-defined boundary:** Sharp transitions between foreground and background
      (less effective on blurry or faded colonies).

    **Thresholding strategies implemented in PhenoTypic**

    - [OtsuDetector](src/phenotypic/detect/_otsu_detector.py): Minimizes within-class variance.
      Automatic, global, works for balanced histograms.
    - [LiDetector](src/phenotypic/detect/_li_detector.py): Minimizes Kullback-Leibler divergence.
      Good for dark colonies on bright background.
    - [YenDetector](src/phenotypic/detect/_yen_detector.py): Maximizes object variance.
      Excellent for sharply defined colonies.
    - [TriangleDetector](src/phenotypic/detect/_triangle_detector.py): Connects histogram
      extrema. Works well for non-overlapping bimodal distributions.
    - [IsodataDetector](src/phenotypic/detect/_isodata_detector.py): Iteratively refines based
      on class means. Robust but slower.
    - [MeanDetector](src/phenotypic/detect/_mean_detector.py) / [MinimumDetector](src/phenotypic/detect/_minimum_detector.py):
      Simple heuristic thresholds. Fast, useful for baseline.
    - [HysteresisDetector](src/phenotypic/detect/_hysteresis_detector.py): Advanced dual-threshold
      with edge tracking. Handles variable colony intensity.

    **When to subclass ThresholdDetector vs ObjectDetector directly**

    Subclass ThresholdDetector if your algorithm uses threshold-based intensity partitioning:

    - Converts intensity to binary mask via threshold comparison (``mask = enh > threshold``).
    - Uses automatic threshold selection (Otsu, Li, Yen, Triangle, Isodata, etc.).
    - Uses simple heuristic thresholds (mean, minimum, percentile-based).
    - Signals intent to other developers: "this detector groups with thresholding methods."
    - May share utility methods in future (e.g., post-processing filters).

    Subclass ObjectDetector directly if your algorithm uses alternative strategies:

    - Edge detection (find gradients, not intensity levels).
    - Peak finding (assumes round shapes, grows from maxima).
    - Watershed segmentation or other region-based approaches.
    - Hybrid methods that don't fit threshold → binary mask → label pattern.

    **Typical workflow: enhance → threshold → label → refine**

    Most ThresholdDetector implementations follow this pipeline:

    1. **Read detection matrix:** ``enh = image.detect_mat[:]`` (preprocessed for
       contrast and noise suppression).
    2. **Compute threshold:** Use chosen strategy (Otsu, Li, Yen, etc.) to find
       optimal threshold value from histogram.
    3. **Create binary mask:** ``mask = enh > threshold`` or
       ``mask = enh >= threshold`` (test both if edge pixels ambiguous).
    4. **Post-process (optional):** Remove small noise, clear borders, morphological
       cleanup to improve mask quality.
    5. **Label connected components:** Use ``scipy.ndimage.label()`` to assign
       unique integer IDs to each colony (objmap).
    6. **Set both outputs:** ``image.objmask = mask``, ``image.objmap = labeled_map``.

    **Parameter tuning guidance**

    Threshold-based detectors expose parameters affecting detection quality:

    - **Threshold value (manual methods only):** Higher → fewer, larger colonies; lower → more,
      noisier. Test range on representative images to find balance.
    - **Block size (local/adaptive methods):** Larger blocks smooth mask but miss small colonies;
      smaller blocks add detail but amplify noise. Start with 1/8 to 1/4 of image width.
    - **ignore_zeros:** Skip pure black pixels in threshold computation. Useful when background
      has significant black regions (shadows, vignetting).
    - **ignore_borders:** Automatically remove objects touching image edges. Prevents partial
      colonies at plate edges from skewing analysis.
    - **min_size / max_size:** Post-processing filters. Remove objects below min (noise) or
      above max (artifacts). Measure typical colony size on your plates first.

    **Comparison with other detection strategies**

    - [CannyDetector](src/phenotypic/detect/_canny_detector.py) (edge-based): Finds intensity
      gradients to locate boundaries. Better for faint or merged colonies; requires tuning
      gradient thresholds.
    - [RoundPeaksDetector](src/phenotypic/detect/_round_peaks_detector.py) (peak-based): Assumes
      round shapes, grows from maxima. Excellent for well-separated round colonies; fails on
      irregular or merged shapes.
    - **Threshold-based (this class):** Direct intensity partitioning. Robust, fast, works for
      any shape; requires good intensity separation between colonies and background.

    **Common pitfalls and remedies**

    - **Over-segmentation (too many small objects):** Use ``ignore_zeros=True`` to skip dark
      pixels, apply morphological opening (remove_small_objects), or refine with ObjectRefiner.
    - **Under-segmentation (merged colonies):** Try local thresholding (adaptive block-based),
      morphological closing, or watershed post-processing in ObjectRefiner.
    - **False positives at edges:** Use ``ignore_borders=True`` parameter or ``clear_border()``
      in ObjectRefiner to remove edge-touching objects.
    - **Uneven illumination (vignetting, shadows):** Apply enhancement (EnhanceLocalContrast, illumination
      correction) before detection, or switch to local adaptive thresholding.
    - **Threshold too high/low:** Visualize objmask on sample images to diagnose. Adjust
      parameters and re-test on representative plates before batch processing.

    **Local thresholding pattern (adaptive to uneven illumination)**

    When images have uneven illumination or vignetting, local (adaptive) thresholding computes
    a threshold per neighborhood instead of globally. This handles gradual intensity changes:

    .. code-block:: python

        from skimage import filters
        from scipy import ndimage
        import numpy as np

        enh = image.detect_mat[:]
        # Compute local threshold for each pixel
        block_size = 31  # Neighborhood size (odd integer)
        threshold_map = filters.threshold_local(enh, block_size=block_size)
        # Create mask: pixel > its local threshold
        mask = enh > threshold_map
        # Label connected components
        labeled, _ = ndimage.label(mask)
        image.objmask[:] = mask
        image.objmap[:] = labeled
        return image

    **Implementation pattern: Global automatic threshold**

    For global automatic thresholding (Otsu, Li, Yen, Triangle, Isodata), follow this pattern:

    .. code-block:: python

        from skimage import filters
        from scipy import ndimage

        def _operate(self, image):
            enh = image.detect_mat[:]
            # Compute threshold value via automatic method
            threshold = filters.threshold_otsu(enh)  # or threshold_li, threshold_yen, etc.
            # Create binary mask: pixels above threshold
            mask = enh > threshold
            # Label connected components
            labeled, num_objects = ndimage.label(mask)
            # Set both outputs
            image.objmask[:] = mask
            image.objmap[:] = labeled
            return image

    Key points: Read preprocessed ``detect_mat``, compute single threshold, compare all pixels at
    once, label result. This is fast and deterministic (same image always produces same result).

    **Interface specification**

    Subclasses of ThresholdDetector must:

    1. Inherit from ThresholdDetector (which provides ObjectDetector's interface).
    2. Implement ``_operate(image: Image) -> Image`` as an instance method.
    3. Within ``_operate()``:

       - Read ``image.detect_mat[:]`` (and optionally ``image.rgb[:], image.gray[:]``).
       - Compute threshold (automatically or from parameter).
       - Generate binary mask via comparison: ``mask = enh > threshold``.
       - Label connected components: ``labeled, _ = ndimage.label(mask)``.
       - Set both outputs: ``image.objmask = mask``, ``image.objmap = labeled``.
       - Return modified image.

    4. Add to ``phenotypic.detect.__init__.py`` exports for public discovery.

    Notes:
        This is a marker ABC with no additional methods. It exists to categorize
        threshold-based detectors in the class hierarchy and enable flexible
        discovery and code organization.

    Examples:
        Detect colonies using Otsu's automatic threshold:

        >>> from phenotypic import Image
        >>> from phenotypic.detect import OtsuDetector
        >>> # Load a plate image
        >>> plate = Image.imread("agar_plate.jpg")
        >>> # Apply Otsu threshold detection
        >>> detector = OtsuDetector(ignore_zeros=True, ignore_borders=True)
        >>> detected = detector.apply(plate)
        >>> # Access results
        >>> mask = detected.objmask[:]  # Binary mask
        >>> objmap = detected.objmap[:]  # Labeled map
        >>> num_colonies = objmap.max()
        >>> print(f"Detected {num_colonies} colonies")
        >>> # Iterate over colonies
        >>> for colony in detected.objects:
        ...     print(f"Colony {colony.label}: area={colony.area} px")

        Compare different threshold strategies:

        >>> from phenotypic import Image
        >>> from phenotypic.detect import (
        ...     OtsuDetector, LiDetector, YenDetector, TriangleDetector
        ... )
        >>> plate = Image.imread("agar_plate.jpg")
        >>> # Test multiple threshold strategies
        >>> detectors = {
        ...     "Otsu": OtsuDetector(),
        ...     "Li": LiDetector(),
        ...     "Yen": YenDetector(),
        ...     "Triangle": TriangleDetector(),
        ... }
        >>> for name, detector in detectors.items():
        ...     result = detector.apply(plate)
        ...     num = result.objmap[:].max()
        ...     print(f"{name}: detected {num} colonies")

        Build a pipeline with thresholding and refinement:

        >>> from phenotypic import Image, ImagePipeline
        >>> from phenotypic.enhance import ContrastEnhancer
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.refine import RemoveSmallObjectsRefiner
        >>> # Create pipeline
        >>> pipeline = ImagePipeline()
        >>> pipeline.add(ContrastEnhancer(factor=1.5))  # Boost contrast
        >>> pipeline.add(OtsuDetector(ignore_zeros=True))  # Threshold
        >>> pipeline.add(RemoveSmallObjectsRefiner(min_size=50))  # Cleanup
        >>> # Process image
        >>> plate = Image.imread("agar_plate.jpg")
        >>> result = pipeline.operate([plate])[0]
        >>> print(f"Final colonies: {result.objmap[:].max()}")

        Tuning threshold detection on your plate images:

        >>> from phenotypic import Image
        >>> from phenotypic.detect import (
        ...     OtsuDetector, YenDetector, TriangleDetector
        ... )
        >>> # Load a sample plate image
        >>> plate = Image.imread("sample_plate.jpg")
        >>> # Test different threshold strategies
        >>> strategies = {
        ...     "Otsu": OtsuDetector(ignore_zeros=True),
        ...     "Yen": YenDetector(ignore_zeros=True),
        ...     "Triangle": TriangleDetector(ignore_zeros=True),
        ... }
        >>> best_detector = None
        >>> best_count = 0
        >>> for name, detector in strategies.items():
        ...     result = detector.apply(plate)
        ...     num_colonies = result.objmap[:].max()
        ...     print(f"{name}: {num_colonies} colonies detected")
        ...     # Choose detector that finds expected number of colonies
        ...     if best_detector is None:
        ...         best_detector = detector
        ...         best_count = num_colonies
        >>> # Use best detector for batch processing
        >>> print(f"Selected: {type(best_detector).__name__}")
    """

    @overload
    @abstractmethod
    def _operate(self, image: Image) -> Image: ...

    @overload
    @abstractmethod
    def _operate(self, image: GridImage) -> GridImage: ...

    @abstractmethod
    def _operate(self, image: Image) -> Image:
        return image
