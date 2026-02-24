import logging

from phenotypic import Image, GridImage, ImagePipeline
from phenotypic._core._pipeline_parts import IntermediateResult
from phenotypic.correction import GridAligner
from phenotypic.data import load_plate_12hr
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import CLAHE, ContrastStretching, GaussianBlur, MedianFilter
from phenotypic.measure import (
    MeasureColor,
    MeasureIntensity,
    MeasureShape,
    MeasureTexture,
)
from phenotypic.refine import (
    BorderObjectRemover,
    LowCircularityRemover,
    SmallObjectRemover,
    ResidualOutlierRemover,
    ReduceMultipleGridObjects,
)
from phenotypic.grid import GridApply
from ..resources.TestHelper import timeit

# Configure logging to see all debug information
logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@timeit
def test_empty_pipeline():
    empty_pipeline = ImagePipeline({})
    assert empty_pipeline.apply(GridImage(load_plate_12hr())).num_objects == 0


@timeit
def test_pipeline_on_image(plate_grid_images):
    pipe = ImagePipeline(
            ops={
                "blur"     : GaussianBlur(sigma=5),
                "detection": OtsuDetector(),
                "remove"   : BorderObjectRemover(50),
            },
            meas={
                "MeasureColor"    : MeasureColor(),
                "MeasureShape"    : MeasureShape(),
                "MeasureIntensity": MeasureIntensity(),
                "MeasureTexture"  : MeasureTexture(scale=[3, 4], quant_lvl=8),
            },
    )
    output = pipe.apply(plate_grid_images)
    output = pipe.measure(output)
    assert output is not None

    compound_output = pipe.apply_and_measure(plate_grid_images, reset=True)

    # Compare with better NaN handling and allow for floating point differences
    import pandas as pd
    import numpy as np

    # Check same shape
    assert output.shape == compound_output.shape, (
        f"Different shapes: {output.shape} vs {compound_output.shape}"
    )

    # Check same columns
    assert set(output.columns) == set(compound_output.columns), "Different columns"

    # Exclude columns that are expected to differ (e.g., UUIDs that change between runs)
    cols_to_skip = {"Metadata_ImageName"}  # UUIDs change between pipeline runs

    # For each column, check if values are close (handling NaNs)
    for col in output.columns:
        if col in cols_to_skip:
            continue

        o_series = output[col]
        c_series = compound_output[col]

        # Handle categorical columns
        if isinstance(o_series.dtype, pd.CategoricalDtype):
            # Convert to underlying codes for comparison
            assert np.array_equal(o_series.cat.codes, c_series.cat.codes), (
                f"Column {col} has different categorical values"
            )
        # Check if both are numeric
        elif pd.api.types.is_numeric_dtype(o_series):
            # Use allclose with NaN handling
            assert np.allclose(
                    o_series.values, c_series.values, equal_nan=True, rtol=1e-10,
                    atol=1e-10
            ), f"Column {col} has different values"
        else:
            # For non-numeric, use equals
            assert o_series.equals(c_series), f"Column {col} has different values"


@timeit
def test_kmarx_pipeline_pickleable(plate_grid_images):
    import pickle

    pipe = ImagePipeline(
            {
                "blur"                            : GaussianBlur(sigma=2),
                "clahe"                           : CLAHE(),
                "median filter"                   : MedianFilter(),
                "detection"                       : OtsuDetector(),
                "border_removal"                  : BorderObjectRemover(50),
                "low circularity remover"         : LowCircularityRemover(0.6),
                "small object remover"            : SmallObjectRemover(100),
                "Reduce by section residual error": ReduceMultipleGridObjects(),
                "outlier removal"                 : ResidualOutlierRemover(),
                "align"                           : GridAligner(),
                "section-level detect"            : GridApply(
                        ImagePipeline(
                                {
                                    "blur"               : GaussianBlur(sigma=5),
                                    "median filter"      : MedianFilter(),
                                    "contrast stretching": ContrastStretching(),
                                    "detection"          : OtsuDetector(),
                                }
                        )
                ),
                "small object remover 2"          : SmallObjectRemover(100),
                "grid_reduction"                  : ReduceMultipleGridObjects(),
            }
    )
    pickle.dumps(pipe.apply_and_measure)


# ---------------------------------------------------------------------------
# apply_with_intermediates tests
# ---------------------------------------------------------------------------


def _make_three_op_pipeline():
    """Helper: build a 3-op pipeline used by several intermediate tests."""
    return ImagePipeline(
        ops={
            "blur": GaussianBlur(sigma=5),
            "detection": OtsuDetector(),
            "remove": BorderObjectRemover(50),
        },
    )


@timeit
def test_apply_with_intermediates_in_memory():
    pipe = _make_three_op_pipeline()
    image = GridImage(load_plate_12hr())

    result = pipe.apply_with_intermediates(image)

    # Returns the correct namedtuple type
    assert isinstance(result, IntermediateResult)

    # Three operations produce three intermediate snapshots
    assert len(result.intermediates) == 3
    assert set(result.intermediates.keys()) == {"blur", "detection", "remove"}

    # Each intermediate is an Image copy (not None)
    for key, intermediate_img in result.intermediates.items():
        assert isinstance(intermediate_img, Image), (
            f"Intermediate '{key}' should be an Image, got {type(intermediate_img)}"
        )

    # The final image is returned and is an Image instance
    assert isinstance(result.image, Image)


@timeit
def test_apply_with_intermediates_to_disk(tmp_path):
    pipe = _make_three_op_pipeline()
    image = GridImage(load_plate_12hr())

    out_dir = tmp_path / "intermediates"
    result = pipe.apply_with_intermediates(image, output_dir=out_dir)

    # Directory was created and contains the expected HDF5 files
    assert out_dir.is_dir()
    h5_files = sorted(out_dir.glob("*.h5"))
    assert len(h5_files) == 3

    # Filenames follow the 00_<name>.h5 pattern
    expected_prefixes = ["00_", "01_", "02_"]
    for h5_file, prefix in zip(h5_files, expected_prefixes):
        assert h5_file.name.startswith(prefix), (
            f"Expected filename starting with '{prefix}', got '{h5_file.name}'"
        )

    # Dict values are None (saved to disk, not kept in memory)
    for key, val in result.intermediates.items():
        assert val is None, (
            f"Intermediate '{key}' should be None when saved to disk"
        )

    # Each HDF5 file can be loaded back into an Image
    for h5_file in h5_files:
        loaded = Image.load_hdf5(h5_file)
        assert isinstance(loaded, Image)


@timeit
def test_apply_with_intermediates_preserves_gridimage():
    pipe = _make_three_op_pipeline()
    image = GridImage(load_plate_12hr())

    result = pipe.apply_with_intermediates(image)

    # The returned final image preserves the GridImage type
    assert isinstance(result.image, GridImage), (
        f"Expected GridImage, got {type(result.image)}"
    )


@timeit
def test_apply_with_intermediates_empty_pipeline():
    pipe = ImagePipeline({})
    image = GridImage(load_plate_12hr())

    result = pipe.apply_with_intermediates(image)

    # No operations means empty intermediates dict
    assert isinstance(result.intermediates, dict)
    assert len(result.intermediates) == 0

    # Final image is still returned
    assert isinstance(result.image, Image)


@timeit
def test_apply_with_intermediates_inplace_false():
    pipe = _make_three_op_pipeline()
    image = GridImage(load_plate_12hr())

    num_objects_before = image.num_objects

    result = pipe.apply_with_intermediates(image, inplace=False)

    # The original image is unchanged
    assert image.num_objects == num_objects_before, (
        f"Original image was modified: num_objects changed from "
        f"{num_objects_before} to {image.num_objects}"
    )

    # The result image has been processed (pipeline includes a detector)
    assert isinstance(result.image, Image)
