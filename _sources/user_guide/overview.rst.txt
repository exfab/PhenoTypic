Overview
========

It's highly recommended to start with prototyping with a subset of images
in a jupyter notebook, then converting to a script for full scale analysis.
This is a brief overview of what the workflow of using PhenoTypic would look like.
For a more detailed explanation go to the tutorials.
A simple workflow for PhenoTypic using a pre-built pipline would look like:

1. Tune pipeline and parameters
-------------------------------

.. code-block:: python

    import phenotypic as pht
    from phenotypic.prefab import HeavyWatershedPipeline

    # Assuming 96-array format
    image = pht.GridImage.imread("path/to/my/image.jpg", nrows=8, ncols=12)
    fig, ax = image.show()
    fig.show()

    # This is the start of the parameter tuning step
    # Create pipeline object. Class params are the settings for the pipeline
    pipe = HeavyWatershedPipeline()
    new_image_one = pipe.apply(image)

    # GridImage.show_overlay() shows the objects detected in the image
    overlay_fig, overlay_ax = new_image_one.show_overlay()
    overlay_fig.show()

    # If the default settings don't work for your images continue tuning
    pipe_two = HeavyWatershedPipeline(border_remover_size=50)
    new_image_two = pipe_two.apply(image)
    overlay_two_fig, overlay_two_ax = new_image_two.show_overlay()
    overlay_two_fig.show()

    # Once the overlay detection looks like how you want, then you
    #   can measure the objects in the image
    meas = pipe_two.measure(new_image_two)

    # export
    overlay_two_fig.savefig("overlay.png")
    meas.to_csv("measurements.csv")



2. Deploy as .py script
-----------------------
The following is a basic script for using PhenoTypic on a set of images. This can take a long time however.
See the :doc:`Getting Started <tutorial/notebooks/GettingStarted>` tutorial for parallel processing.

.. code-block:: python

    from pathlib import Path
    import phenotypic as pht
    from phenotypic.prefab import HeavyWatershedPipeline

    DIRPATH_IMAGES = Path("path/to/images")
    IMAGE_SUFFIX = ".jpg" # This is the target files we're trying to import

    filepaths = [
        filepath for filepath in DIRPATH_IMAGES.iterdir()
        if (filepath.is_file())                     # Ensures only files are read
            and (filepath.suffix == IMAGE_SUFFIX)   # Ensures only jpegs are read
    ]

    # Create pipeline object.
    pipe = HeavyWatershedPipeline(border_remover_size=50)

    meas = []
    for impath in filepaths:

        # Create a GridImage for a 96-well format plate
        image = pht.GridImage.imread(impath, nrows=8, ncols=12)
        current_meas = pipe.apply_and_measure(image, inplace=True)
        meas.append(current_meas)

    meas = pd.concat(meas, axis=0)
