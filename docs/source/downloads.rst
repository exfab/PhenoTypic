Downloads
=========

This page contains downloadable scripts, notebooks, and utilities for PhenoTypic.

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: phenotypic-cli.py
        :shadow: md

        Backwards compatibility wrapper for phenotypic-cli.py.

        +++
        :download:`Download script <_downloads/phenotypic-cli.py>`

    .. grid-item-card:: SLURM Job Chain Manager for Image Processing
        :shadow: md

        Processes multiple directories sequentially with dependencies USAGE With directory file

        +++
        :download:`Download script <_downloads/phenotypic-slurm.sh>`


Downloadable Notebooks
======================

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: Tutorial 1: Your First Plate Image
        :shadow: md

        Welcome to PhenoTypic! In this tutorial, you will load your first agar plate image, compare `Image` and `GridImage`, inspect its basic properties, and explore the **accessor pattern** that gives you different views of the same plate.

        +++
        :download:`Download notebook <tutorials/notebooks/01_your_first_plate_image.ipynb>`

    .. grid-item-card:: Tutorial 2: Detecting Colonies
        :shadow: md

        In [Tutorial 1](01_your_first_plate_image.ipynb), you loaded a plate image and explored its accessors. Now let's put that plate to work — we'll detect the individual *Rhodotorula* colonies using PhenoTypic's `OtsuDetector`.

        +++
        :download:`Download notebook <tutorials/notebooks/02_detecting_colonies.ipynb>`

    .. grid-item-card:: Tutorial 3: Enhancing Before Detection
        :shadow: md

        Detection operates on `detect_mat` -- a grayscale representation of your plate that the detector reads to separate colonies from background. If you improve `detect_mat` *before* running detection, you get cleaner, more complete results.

        +++
        :download:`Download notebook <tutorials/notebooks/03_enhancing_before_detection.ipynb>`

    .. grid-item-card:: Tutorial 4: Building a Pipeline
        :shadow: md

        **Learning goals:**

        +++
        :download:`Download notebook <tutorials/notebooks/04_building_a_pipeline.ipynb>`

    .. grid-item-card:: Tutorial 5: Working with Detected Grid Plates
        :shadow: md

        After colony detection, a `GridImage` can assign each colony to a well, extract individual wells as subimages, count colonies per grid section, and show detected objects together with the grid overlay.

        +++
        :download:`Download notebook <tutorials/notebooks/05_working_with_grid_plates.ipynb>`

    .. grid-item-card:: Tutorial 6: Batch Processing
        :shadow: md

        When you have dozens or hundreds of plates to process, running Python code one image at a time is not practical. PhenoTypic's **command-line interface** (CLI) lets you apply a saved pipeline to an entire directory of plate images with built-in parallelism, checkpointing, and resume.

        +++
        :download:`Download notebook <tutorials/notebooks/06_batch_processing.ipynb>`

    .. grid-item-card:: Tutorial 7: Measuring and Exporting
        :shadow: md

        Detection tells you *where* colonies are. Measurement tells you *what they are* — how big, how round, how bright. In this tutorial you will add measurements to a pipeline, extract a DataFrame of colony features, and export the results for downstream analysis.

        +++
        :download:`Download notebook <tutorials/notebooks/07_measuring_and_exporting.ipynb>`

    .. grid-item-card:: Tutorial 8: Using Prefab Pipelines
        :shadow: md

        Building a pipeline from scratch gives you full control, but PhenoTypic also ships **prefab pipelines** — pre-configured `ImagePipeline` subclasses tuned for common organisms and plate types. In this tutorial you will survey the available prefabs, apply one, and compare results.

        +++
        :download:`Download notebook <tutorials/notebooks/08_using_prefab_pipelines.ipynb>`

    .. grid-item-card:: Tutorial 9: Diagnosing Image Quality
        :shadow: md

        Before building a pipeline, it helps to assess the quality of your plate images. PhenoTypic's diagnostics plotter gives you objective metrics for noise, contrast, and structure — so you can make informed decisions about which enhancers and detectors to use.

        +++
        :download:`Download notebook <tutorials/notebooks/09_diagnosing_image_quality.ipynb>`

    .. grid-item-card:: Tutorial 10: Detecting Filamentous Fungi
        :shadow: md

        Filamentous fungi like *Neurospora* grow as branching networks of hyphae, not compact round colonies. Standard threshold detectors struggle with this morphology because hyphae are thin and have varying intensity. PhenoTypic includes a specialized pipeline for exactly this scenario.

        +++
        :download:`Download notebook <tutorials/notebooks/10_detecting_filamentous_fungi.ipynb>`

    .. grid-item-card:: How To: Assess Image Quality Before Pipeline Design
        :shadow: md

        Run diagnostics on a plate image to objectively assess noise, contrast, and structure before choosing enhancers and detectors.

        +++
        :download:`Download notebook <how_to/notebooks/assess_image_quality.ipynb>`

    .. grid-item-card:: How To: Choose a Detection Algorithm
        :shadow: md

        PhenoTypic offers multiple detectors, each suited to different plate conditions. This guide compares the most commonly used options on the same plate image so you can pick the right one.

        +++
        :download:`Download notebook <how_to/notebooks/choose_detection_algorithm.ipynb>`

    .. grid-item-card:: How To: Combine Detectors with CompositeDetector
        :shadow: md

        When no single detector captures all colonies reliably, combine multiple detectors using `CompositeDetector`. It merges their results via union, intersection, or overlap.

        +++
        :download:`Download notebook <how_to/notebooks/combine_detectors.ipynb>`

    .. grid-item-card:: How To: Correct Edge Effects in Plate Assays
        :shadow: md

        Colonies on the outer rows and columns of an agar plate often grow differently due to temperature gradients and humidity effects. Use `EdgeCorrector` to statistically identify and correct these biases.

        +++
        :download:`Download notebook <how_to/notebooks/correct_edge_effects.ipynb>`

    .. grid-item-card:: How To: Correct Grid Rotation
        :shadow: md

        Apply `GridAligner` to straighten plates that were scanned at a slight angle. This improves grid detection accuracy for downstream well-level analysis.

        +++
        :download:`Download notebook <how_to/notebooks/correct_grid_rotation.ipynb>`

    .. grid-item-card:: How To: Crop and Pad Images for Batch Consistency
        :shadow: md

        Use `ImageCropper` to remove scanner borders and `ImagePadder` to make images a uniform size for batch processing.

        +++
        :download:`Download notebook <how_to/notebooks/crop_and_pad.ipynb>`

    .. grid-item-card:: How To: Denoise Low-Light Images
        :shadow: md

        Low-light or high-ISO plate images contain noise that confuses detectors. PhenoTypic offers several denoising approaches, each with different tradeoffs between speed and quality.

        +++
        :download:`Download notebook <how_to/notebooks/denoise_low_light.ipynb>`

    .. grid-item-card:: How To: Enhance Low-Contrast Images
        :shadow: md

        When colonies are faint or the agar background is uneven, detection suffers. Use `CLAHE`, `ContrastStretching`, or `HomomorphicFilter` to boost local contrast in `detect_mat` before detection.

        +++
        :download:`Download notebook <how_to/notebooks/enhance_low_contrast.ipynb>`

    .. grid-item-card:: How To: Fit Logistic Growth Models
        :shadow: md

        Fit logistic growth curves to time-series colony measurements and extract kinetic parameters (growth rate, carrying capacity, lag time) using `LogGrowthModel`.

        +++
        :download:`Download notebook <how_to/notebooks/fit_logistic_growth.ipynb>`

    .. grid-item-card:: How To: Process Time-Series for Growth Curves
        :shadow: md

        Process a series of plate images taken at different time points, extract measurements from each, and combine them into a time-series DataFrame suitable for growth curve analysis.

        +++
        :download:`Download notebook <how_to/notebooks/growth_curves.ipynb>`

    .. grid-item-card:: How To: Manual Grid Detection When Automatic Fails
        :shadow: md

        When automatic grid detection does not find the correct grid layout, use `ManualGridDetector` to specify colony positions explicitly.

        +++
        :download:`Download notebook <how_to/notebooks/manual_grid_detection.ipynb>`

    .. grid-item-card:: How To: Measure Colony Size and Intensity
        :shadow: md

        Extract area, perimeter, circularity, and intensity statistics from detected colonies using `MeasureSize`, `MeasureShape`, and `MeasureIntensity`.

        +++
        :download:`Download notebook <how_to/notebooks/measure_colony_size_intensity.ipynb>`

    .. grid-item-card:: How To: Merge Fragmented Detections
        :shadow: md

        When a single colony is split into multiple objects (fragmentation), use refiners to merge the fragments back together.

        +++
        :download:`Download notebook <how_to/notebooks/merge_fragmented_detections.ipynb>`

    .. grid-item-card:: How To: Refine Noisy Detection Boundaries
        :shadow: md

        After detection, colony masks often have ragged edges, small holes, or spurious fragments. Use refiners to clean up the mask before measuring.

        +++
        :download:`Download notebook <how_to/notebooks/refine_noisy_boundaries.ipynb>`

    .. grid-item-card:: Write Your First Custom Operation
        :shadow: md

        PhenoTypic is designed to be extended. In this tutorial you will create a custom `ImageEnhancer` from scratch, implement its `_operate()` method, and use it inside an `ImagePipeline`.

        +++
        :download:`Download notebook <extending/notebooks/write_your_first_custom_operation.ipynb>`
