import logging

from phenotypic import Image, GridImage, ImagePipeline
from phenotypic._core._pipeline_parts import IntermediateResult
from phenotypic.correction import GridAligner
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
def test_empty_pipeline(plate_12hr_grid_image):
    empty_pipeline = ImagePipeline(pipe_cfgs={})
    assert empty_pipeline.apply(plate_12hr_grid_image.copy()).num_objects == 0


@timeit
def test_pipeline_on_image(plate_grid_images):
    pipe = ImagePipeline(
            ops={
                "blur"     : GaussianBlur(sigma=5),
                "detection": OtsuDetector(),
                "remove"   : BorderObjectRemover(border_size=50),
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
            pipe_cfgs={
                "blur"                            : GaussianBlur(sigma=2),
                "clahe"                           : CLAHE(),
                "median filter"                   : MedianFilter(),
                "detection"                       : OtsuDetector(),
                "border_removal"                  : BorderObjectRemover(border_size=50),
                "low circularity remover"         : LowCircularityRemover(cutoff=0.6),
                "small object remover"            : SmallObjectRemover(min_size=100),
                "Reduce by section residual error": ReduceMultipleGridObjects(),
                "outlier removal"                 : ResidualOutlierRemover(),
                "align"                           : GridAligner(),
                "section-level detect"            : GridApply(
                        image_op=ImagePipeline(
                                pipe_cfgs={
                                    "blur"               : GaussianBlur(sigma=5),
                                    "median filter"      : MedianFilter(),
                                    "contrast stretching": ContrastStretching(),
                                    "detection"          : OtsuDetector(),
                                }
                        )
                ),
                "small object remover 2"          : SmallObjectRemover(min_size=100),
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
            "remove": BorderObjectRemover(border_size=50),
        },
    )


@timeit
def test_apply_with_intermediates_in_memory(plate_12hr_grid_image):
    pipe = _make_three_op_pipeline()
    image = plate_12hr_grid_image.copy()

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
def test_apply_with_intermediates_to_disk(tmp_path, plate_12hr_grid_image):
    pipe = _make_three_op_pipeline()
    image = plate_12hr_grid_image.copy()

    out_dir = tmp_path / "intermediates"
    result = pipe.apply_with_intermediates(image, output_dir=out_dir)

    # Directory was created and contains the expected HDF5 files
    assert out_dir.is_dir()
    h5_files = sorted(out_dir.glob("*.h5"))
    assert len(h5_files) == 4  # base + 3 operations

    # Filenames follow the 00_<name>.h5 pattern
    expected_prefixes = ["00_", "01_", "02_", "base_"]
    for h5_file, prefix in zip(h5_files, expected_prefixes):
        assert h5_file.name.startswith(prefix), (
            f"Expected filename starting with '{prefix}', got '{h5_file.name}'"
        )

    # Dict values are None (saved to disk, not kept in memory)
    for key, val in result.intermediates.items():
        assert val is None, (
            f"Intermediate '{key}' should be None when saved to disk"
        )


@timeit
def test_apply_with_intermediates_preserves_gridimage(plate_12hr_grid_image):
    pipe = _make_three_op_pipeline()
    image = plate_12hr_grid_image.copy()

    result = pipe.apply_with_intermediates(image)

    # The returned final image preserves the GridImage type
    assert isinstance(result.image, GridImage), (
        f"Expected GridImage, got {type(result.image)}"
    )


@timeit
def test_apply_with_intermediates_empty_pipeline(plate_12hr_grid_image):
    pipe = ImagePipeline(pipe_cfgs={})
    image = plate_12hr_grid_image.copy()

    result = pipe.apply_with_intermediates(image)

    # No operations means empty intermediates dict
    assert isinstance(result.intermediates, dict)
    assert len(result.intermediates) == 0

    # Final image is still returned
    assert isinstance(result.image, Image)


@timeit
def test_apply_with_intermediates_inplace_false(plate_12hr_grid_image):
    pipe = _make_three_op_pipeline()
    image = plate_12hr_grid_image.copy()

    num_objects_before = image.num_objects

    result = pipe.apply_with_intermediates(image, inplace=False)

    # The original image is unchanged
    assert image.num_objects == num_objects_before, (
        f"Original image was modified: num_objects changed from "
        f"{num_objects_before} to {image.num_objects}"
    )

    # The result image has been processed (pipeline includes a detector)
    assert isinstance(result.image, Image)


# ---------------------------------------------------------------------------
# Benchmark memory tracking tests
# ---------------------------------------------------------------------------


@timeit
def test_benchmark_memory_columns(plate_12hr_grid_image):
    """benchmark_results() DataFrame contains the new memory columns."""
    pipe = ImagePipeline(
        ops=[GaussianBlur(sigma=5), OtsuDetector()],
        benchmark=True,
    )
    pipe.apply(plate_12hr_grid_image)
    df = pipe.benchmark_results()

    expected_cols = {
        "Process Type",
        "Process Name",
        "Execution Time (s)",
        "Memory Delta (MB)",
        "RSS After (MB)",
    }
    assert expected_cols == set(df.columns), (
        f"Expected columns {expected_cols}, got {set(df.columns)}"
    )


@timeit
def test_benchmark_memory_populated(plate_12hr_grid_image):
    """_operation_memory and _operation_rss are filled when benchmark=True."""
    pipe = ImagePipeline(
        ops={"blur": GaussianBlur(sigma=5), "detect": OtsuDetector()},
        benchmark=True,
    )
    pipe.apply(plate_12hr_grid_image)

    assert len(pipe._operation_memory) == 2
    assert len(pipe._operation_rss) == 2
    assert "blur" in pipe._operation_memory
    assert "detect" in pipe._operation_rss

    # RSS values should be positive (process always uses some memory)
    for rss in pipe._operation_rss.values():
        assert rss > 0, f"RSS should be positive, got {rss}"


@timeit
def test_benchmark_nested_pipeline_expansion(plate_12hr_grid_image):
    """Sub-rows appear for nested ImagePipeline operations."""
    inner = ImagePipeline(
        ops={"inner_blur": GaussianBlur(sigma=3), "inner_detect": OtsuDetector()},
    )
    outer = ImagePipeline(
        ops={"outer_blur": GaussianBlur(sigma=5), "nested": inner},
        benchmark=True,
    )
    outer.apply(plate_12hr_grid_image)
    df = outer.benchmark_results()

    names = df["Process Name"].tolist()

    # Top-level rows
    assert "outer_blur" in names
    assert "nested" in names

    # Expanded sub-rows from the nested pipeline
    assert "  nested > inner_blur" in names
    assert "  nested > inner_detect" in names

    # Sub-rows have memory values populated
    sub_rows = df[df["Process Name"].str.startswith("  ")]
    assert len(sub_rows) == 2
    for _, row in sub_rows.iterrows():
        assert row["RSS After (MB)"] > 0

    # Total row should not double-count sub-rows
    total_row = df[df["Process Type"] == "Total"]
    assert len(total_row) == 1
    top_level = df[
        (df["Process Type"] == "Operation") & ~df["Process Name"].str.startswith("  ")
    ]
    assert abs(
        total_row["Execution Time (s)"].iloc[0] - top_level["Execution Time (s)"].sum()
    ) < 1e-9


@timeit
def test_benchmark_no_memory_when_disabled(plate_12hr_grid_image):
    """Memory dicts stay empty when benchmark=False."""
    pipe = ImagePipeline(
        ops=[GaussianBlur(sigma=5), OtsuDetector()],
        benchmark=False,
    )
    pipe.apply(plate_12hr_grid_image)

    assert len(pipe._operation_memory) == 0
    assert len(pipe._operation_rss) == 0


# ---------------------------------------------------------------------------
# Soft grid-shape preset (nrows/ncols)
# ---------------------------------------------------------------------------


@timeit
def test_grid_preset_auto_injects_grid_finder(synth_plate_detected):
    """measure() auto-injects AutoGridFinder when preset set and none configured."""
    from phenotypic.grid import AutoGridFinder

    pipe = ImagePipeline(meas=[MeasureShape()], nrows=8, ncols=12)
    assert "AutoGridFinder" not in pipe._meas  # not persisted

    df = pipe.measure(synth_plate_detected.copy())

    # AutoGridFinder ran first, so the result has grid columns.
    assert "Grid_RowNum" in df.columns and "Grid_ColNum" in df.columns
    # _meas itself was not mutated.
    assert "AutoGridFinder" not in pipe._meas
    # Sanity: the preset is reachable on the pipeline instance.
    assert pipe.nrows == 8 and pipe.ncols == 12
    # Auto-injected step uses the preset values: build a fresh run order and
    # confirm the injected instance carries them.
    run_order = pipe._build_measurement_run_order()
    injected = run_order["AutoGridFinder"]
    assert isinstance(injected, AutoGridFinder)
    assert injected.nrows == 8 and injected.ncols == 12


@timeit
def test_grid_preset_does_not_override_existing_grid_finder(synth_plate_detected):
    """An existing GridFinder step wins; the preset does not auto-inject."""
    from phenotypic.grid import AutoGridFinder

    explicit = AutoGridFinder(nrows=4, ncols=6)
    # Preset says 16x24 but explicit says 4x6 — explicit must win.
    pipe = ImagePipeline(meas=[explicit], nrows=16, ncols=24)

    run_order = pipe._build_measurement_run_order()
    finders = [m for m in run_order.values() if isinstance(m, AutoGridFinder)]
    assert len(finders) == 1
    assert finders[0] is explicit
    assert finders[0].nrows == 4 and finders[0].ncols == 6


@timeit
def test_grid_preset_no_op_when_unset():
    """Without the preset, measure() does not auto-inject anything."""
    from phenotypic.grid import AutoGridFinder

    pipe = ImagePipeline(meas=[MeasureShape()])
    run_order = pipe._build_measurement_run_order()

    assert all(not isinstance(m, AutoGridFinder) for m in run_order.values())
    assert list(run_order.keys()) == list(pipe._meas.keys())


@timeit
def test_grid_preset_idempotent_repeated_measure(synth_plate_detected):
    """Repeat measure() calls neither accumulate nor mutate _meas."""
    pipe = ImagePipeline(meas=[MeasureShape()], nrows=8, ncols=12)

    pipe.measure(synth_plate_detected.copy())
    pipe.measure(synth_plate_detected.copy())

    # _meas was never touched, regardless of how many measure() calls ran.
    assert list(pipe._meas.keys()) == ["MeasureShape"]
