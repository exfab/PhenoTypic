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

    .. grid-item-card:: 2. Using `ImagePipeline`s to process batches
        :shadow: md

        Learn to tune and use prefabricated image processing pipelines, then deploy them at scale.

        +++
        :download:`Download notebook <user_guide/tutorial/notebooks/BatchProcessing.ipynb>`

    .. grid-item-card:: 1. Getting Started
        :shadow: md

        Get started with image processing in PhenoTypic. --- Getting started with image processing is straightforward. There's three classes of operations in `phenotypic`: `ImageOperation`, `MeasureFeature`, and `ImagePipeline`. - `ImageOperation`(s): processes that operate on the data of an image in preparation for feature extraction with `MeasureFeature`. - `MeasureFeature`(s) extract measurements from the objects within the image based on the pixel values. - `ImagePipeline`(s) are a collection of operations and measurements compiled into a single class for convenience. ## Understanding the Accessor Pattern PhenoTypic uses an **accessor pattern** to provide clean, consistent access to image data without exposing raw attributes: - `image.rgb[:]` - RGB color array - `image.gray[:]` - Grayscale representation (automatic luminance conversion) - `image.detect_mat[:]` - Enhanced grayscale for processing (used by detection operations) - `image.objects` - High-level interface for detected objects and their properties This design ensures lazy evaluation, caching, and transparent format conversion. ## Using Pre-built Pipelines To get started with `phenotypic`, it's fastest to start by using one of the pipelines in `phenotypic.prefab`. Below we use `phenotypic.prefab.HeavyRoundPeaksPipeline`, which is optimized for images of *Saccharomyces cerevisiae* and similar microorganisms. To load in an image you'll use `phenotypic.GridImage.imread()`. This method is also available for regular images, `phenotypic.Image.imread()`.

        +++
        :download:`Download notebook <user_guide/tutorial/notebooks/GettingStarted.ipynb>`

    .. grid-item-card:: 4. Working with GridImage: Grid-Specific Features
        :shadow: md

        This tutorial focuses on `GridImage` - a specialized class for analyzing arrayed microbe colonies on agar plates. GridImage extends the `Image` class with grid tracking and position-aware analysis.

        +++
        :download:`Download notebook <user_guide/tutorial/notebooks/GridImages.ipynb>`

    .. grid-item-card:: 6. Fitting Logistic Growth Curves
        :shadow: md

        ---

        +++
        :download:`Download notebook <user_guide/tutorial/notebooks/GrowthCurves.ipynb>`

    .. grid-item-card:: 5. Make your own `ImagePipeline`
        :shadow: md

        For a more detailed explanation on the components of `phenotypic.Image` and `phenotypic.GridImage`, see the {doc} `Images` and {doc} `GridImages` tutorial.

        +++
        :download:`Download notebook <user_guide/tutorial/notebooks/ImagePipelines.ipynb>`

    .. grid-item-card:: 3. Understanding the Image Class: Data Components and Accessors
        :shadow: md

        A comprehensive guide to the PhenoTypic `Image` class architecture, focusing on its main data components and how they interact with detection and analysis modules.

        +++
        :download:`Download notebook <user_guide/tutorial/notebooks/Images.ipynb>`
