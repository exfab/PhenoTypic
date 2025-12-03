from ._object_detector import ObjectDetector
from abc import ABC


# <<Interface>>
class ThresholdDetector(ObjectDetector, ABC):
    """Marker ABC for threshold-based colony detection strategies.

    ThresholdDetector specializes ObjectDetector for algorithms that detect colonies
    by converting grayscale intensity to a binary mask via thresholding. Unlike edge-based
    (Canny) or peak-based (RoundPeaks) approaches, thresholding works by partitioning
    intensity space: pixels above a threshold value become foreground (colonies), pixels
    below become background.

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

    - **Otsu's method:** Finds threshold that minimizes within-class variance. Automatic,
      global, works for most balanced foreground/background histograms.
    - **Li's method:** Minimizes Kullback-Leibler divergence. Good for dark foreground
      on bright background.
    - **Yen's method:** Maximizes Yen's object variance criterion. Good for sharply
      defined objects.
    - **Triangle method:** Connects histogram extrema. Works well for non-overlapping
      bimodal distributions.
    - **Isodata/Iterative selection:** Iteratively refines threshold based on class means.
      Robust but slower.
    - **Mean/Minimum methods:** Simple heuristic thresholds (average or minimum intensity).
      Fast, useful for baseline or preprocessing.
    - **Local/Adaptive thresholding:** Applies threshold per neighborhood instead of globally.
      Handles uneven illumination on agar.

    **When to subclass ThresholdDetector vs ObjectDetector directly**

    - **Subclass ThresholdDetector if:**

      - Your algorithm produces objmask and objmap via thresholding (any strategy).
      - You want to signal intent: "this detector groups with other thresholding methods."
      - You may add shared utility methods later (e.g., post-processing filters).
      - You value categorization for discovery and code organization.

    - **Subclass ObjectDetector directly if:**

      - Your algorithm uses edge detection (Canny), peak finding, watershed, or
        morphological operations (not thresholding).
      - Your approach doesn't fit the threshold → binary mask → label pattern.

    **Typical workflow: enhance → threshold → label → refine**

    Most ThresholdDetector implementations follow this pipeline:

    1. **Read enhanced grayscale:** ``enh = image.enh_gray[:]`` (preprocessed for
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

    Threshold-based detectors typically expose parameters that affect detection quality:

    - **Threshold value:** For manual methods (Mean, Minimum), directly controls the
      intensity cutoff. Higher values → fewer, larger colonies; lower → more, noisier.
    - **Block size (local methods):** Size of neighborhood for adaptive threshold.
      Larger blocks → smoother mask but may miss small colonies; smaller blocks →
      more detail but noise-prone.
    - **Post-processing parameters:** ``ignore_zeros`` (skip pure black pixels in
      threshold computation), ``ignore_borders`` (remove edge-touching objects),
      ``min_size`` (filter objects below pixel count).

    **Comparison with other detection strategies**

    - **Edge-based (CannyDetector):** Finds intensity gradients (colony boundaries).
      Better for faint or merged colonies; requires gradient-based preprocessing.
    - **Peak-based (RoundPeaksDetector):** Assumes round peaks; grows from maxima.
      Excellent for well-separated round colonies; fails on irregular shapes.
    - **Threshold-based (this class):** Direct intensity partitioning. Robust, fast,
      works for any shape; requires good intensity separation.

    **Common pitfalls and remedies**

    - **Over-segmentation (too many small objects):** Use ``ignore_zeros=True`` to
      skip dark pixels, apply morphological opening, or use ObjectRefiner with
      ``remove_small_objects(min_size=...)``.
    - **Under-segmentation (merged colonies):** Local thresholding, morphological
      closing, or watershed post-processing.
    - **False positives at edges:** Use ``ignore_borders=True`` or
      ``clear_border()`` in post-processing.
    - **Uneven illumination:** Apply enhancement (contrast stretching, illumination
      correction) before detection, or use local thresholding.

    **Example implementations**

    See concrete subclasses for reference patterns:

    - **OtsuDetector:** Global automatic thresholding via Otsu's variance minimization.
    - **LiDetector, YenDetector, TriangleDetector:** Alternative global strategies
      from scikit-image.filters.
    - **MeanDetector, MinimumDetector:** Simple heuristic thresholds.

    **Interface specification**

    Subclasses of ThresholdDetector must:

    1. Inherit from ThresholdDetector (which provides ObjectDetector's interface).
    2. Implement ``_operate(image: Image) -> Image`` as a static method.
    3. Within ``_operate()``:

       - Read ``image.enh_gray[:]`` (and optionally ``image.rgb[:], image.gray[:]``).
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
        .. dropdown:: Detect colonies using Otsu's automatic threshold

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.detect import OtsuDetector

                # Load a plate image
                plate = Image.from_image_path("agar_plate.jpg")

                # Apply Otsu threshold detection
                detector = OtsuDetector(ignore_zeros=True, ignore_borders=True)
                detected = detector.apply(plate)

                # Access results
                mask = detected.objmask[:]  # Binary mask
                objmap = detected.objmap[:]  # Labeled map
                num_colonies = objmap.max()
                print(f"Detected {num_colonies} colonies")

                # Iterate over colonies
                for colony in detected.objects:
                    print(f"Colony {colony.label}: area={colony.area} px")

        .. dropdown:: Compare different threshold strategies

            .. code-block:: python

                from phenotypic import Image
                from phenotypic.detect import (
                    OtsuDetector, LiDetector, YenDetector, TriangleDetector
                )

                plate = Image.from_image_path("agar_plate.jpg")

                # Test multiple threshold strategies
                detectors = {
                    "Otsu": OtsuDetector(),
                    "Li": LiDetector(),
                    "Yen": YenDetector(),
                    "Triangle": TriangleDetector(),
                }

                for name, detector in detectors.items():
                    result = detector.apply(plate)
                    num = result.objmap[:].max()
                    print(f"{name}: detected {num} colonies")

        .. dropdown:: Build a pipeline with thresholding and refinement

            .. code-block:: python

                from phenotypic import Image, ImagePipeline
                from phenotypic.enhance import ContrastEnhancer
                from phenotypic.detect import OtsuDetector
                from phenotypic.refine import RemoveSmallObjectsRefiner

                # Create pipeline
                pipeline = ImagePipeline()
                pipeline.add(ContrastEnhancer(factor=1.5))  # Boost contrast
                pipeline.add(OtsuDetector(ignore_zeros=True))  # Threshold
                pipeline.add(RemoveSmallObjectsRefiner(min_size=50))  # Cleanup

                # Process image
                plate = Image.from_image_path("agar_plate.jpg")
                result = pipeline.operate([plate])[0]

                print(f"Final colonies: {result.objmap[:].max()}")
    """

    pass
