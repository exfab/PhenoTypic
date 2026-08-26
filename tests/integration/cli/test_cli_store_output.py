"""
Integration tests for the store-centric CLI output layout and measure-mode rerun.

These tests exercise the full CLI via :mod:`click.testing.CliRunner`:

* Forward runs write exactly one OME-Zarr **store directory** per image under
  ``results/<dataset>/zarr/<stem>.ome.zarr/``, with its Parquet table at
  ``tables/measurements/table.parquet`` inside that store — and nothing else
  (no per-layer ``rgb/`` / ``gray/`` / ``detect_mat/`` / ``objmap/`` folders,
  and no ``hdf/``).
* Forward runs always write a PNG overlay per image alongside the store.
* Schema-3 outputs reject ``--mode measure`` before it can invalidate their
  marker-authorized per-image evidence.
* ``--mode measure`` rejects incompatible flags with a clear error message.

The store is a directory, so every existence check here is ``is_dir()`` and
every path is resolved through ``zarr_store_path`` — the ``.ome.zarr`` double
suffix is never hand-joined.

All tests pass ``--force-local`` so SLURM is never dispatched.
"""

import json
import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner
from PIL import Image as PILImage

from phenotypic.data import load_synth_yeast_plate
from phenotypic._cli._cli_completion import (
    valid_aggregate_snapshot,
    valid_run_completion,
)
from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.prefab import RoundPeaksPipeline
from phenotypic.sdk_ import (
    MEASUREMENT_TABLE_RELATIVE_PATH,
    aggregate_publication_marker_path,
    image_completion_marker_path,
    manifest_json_path,
    run_completion_marker_path,
    zarr_store_path,
)
from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    """Provide a Click CliRunner for CLI invocation.

    Click 8.2+ always merges stderr into ``result.output``, so the
    validation tests can assert on click.UsageError messages via
    ``result.output`` without extra wiring.
    """
    return CliRunner()


@pytest.fixture
def temp_pipeline():
    """Write a RoundPeaksPipeline JSON to a temporary file."""
    pipeline = RoundPeaksPipeline(
        blur_sigma=3,
        detector_thresh_method="otsu",
        detector_subtract_background=True,
        detector_remove_noise=True,
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as handle:
        handle.write(pipeline.to_json())
        pipeline_path = Path(handle.name)

    try:
        yield pipeline_path
    finally:
        if pipeline_path.exists():
            pipeline_path.unlink()


def _write_synth_image(target_path: Path) -> None:
    """Render a synthetic yeast plate as RGB and save to ``target_path``."""
    grid_image = load_synth_yeast_plate()
    pil_img = PILImage.fromarray(grid_image.rgb[:].astype("uint8"))
    pil_img.save(target_path)


@pytest.fixture
def plates_input_dir(tmp_path):
    """Create a deterministically named input directory with one synth image.

    The directory is named ``plates`` so that the CLI's dataset discovery
    (which uses the input directory's basename for the dataset name) always
    produces a predictable ``results/plates/...`` layout.
    """
    input_dir = tmp_path / "plates"
    input_dir.mkdir()
    _write_synth_image(input_dir / "plate_001.png")
    return input_dir


# ---------------------------------------------------------------------------
# Forward run — store-only output layout
# ---------------------------------------------------------------------------


class TestForwardRunStoreLayout:
    """A forward run must embed its table in the store and nothing else."""

    def test_forward_run_produces_store_only(
        self, runner, temp_pipeline, plates_input_dir, tmp_path
    ):
        """Forward run writes only zarr/, with a readable embedded table."""
        output_dir = tmp_path / "out"

        result = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(temp_pipeline),
                "--input",
                str(plates_input_dir),
                "-o",
                str(output_dir),
                "--njobs",
                "1",
                "--skip-validation",
                "--force-local",
            ],
        )

        assert result.exit_code == 0, (
            f"CLI failed (exit_code={result.exit_code}):\n{result.output}"
        )

        dataset_dir = output_dir / "results" / "plates"
        store = zarr_store_path(output_dir, "plates", "plate_001")
        table_file = store / MEASUREMENT_TABLE_RELATIVE_PATH
        overlay_file = output_dir / "deliverables" / "overlays" / "plates" / "plate_001.png"

        # A store is a DIRECTORY with a root `zarr.json`. `.exists()` alone
        # would also pass for a stray file of the same name.
        assert store.is_dir(), (
            f"Expected an OME-Zarr store at {store} (got contents: "
            f"{list(dataset_dir.rglob('*')) if dataset_dir.exists() else 'no dataset dir'})"
        )
        assert (store / "zarr.json").is_file()
        assert not list(output_dir.rglob("*.h5")), (
            "A forward run must not write an HDF any more."
        )
        assert table_file.is_file(), (
            f"Expected embedded measurements at {table_file}"
        )
        measurements = pd.read_parquet(table_file)
        assert not measurements.empty
        assert any(column.startswith("Shape_") for column in measurements)
        assert not (dataset_dir / "measurements").exists()
        assert overlay_file.exists(), (
            f"Expected overlay PNG at {overlay_file} (overlays are always "
            f"written for forward runs)"
        )
        assert valid_aggregate_snapshot(output_dir) is not None
        assert valid_run_completion(output_dir) is not None

        # None of the legacy per-layer directories must be created.
        for legacy_folder in ("rgb", "gray", "detect_mat", "objmap"):
            legacy_path = dataset_dir / legacy_folder
            assert not legacy_path.exists(), (
                f"Legacy per-layer folder {legacy_path} should NOT exist "
                f"after forward run; CLI regressed to pre-HDF layout."
            )

        # Whitelist check: current runs create no sibling measurement or HDF
        # authority beside the store.
        actual_children = {p.name for p in dataset_dir.iterdir() if p.is_dir()}
        assert actual_children == {"zarr"}, (
            f"Unexpected dataset-level folders after forward run: "
            f"{sorted(actual_children)}"
        )

    def test_default_run_continues_and_processes_only_new_input(
        self, runner, temp_pipeline, plates_input_dir, tmp_path
    ):
        """A second invocation needs no lifecycle flag and admits appended images."""
        output_dir = tmp_path / "out"
        args = [
            "--pipeline",
            str(temp_pipeline),
            "--input",
            str(plates_input_dir),
            "-o",
            str(output_dir),
            "--njobs",
            "1",
            "--skip-validation",
            "--force-local",
        ]
        first = runner.invoke(phenotypic_cli, args)
        assert first.exit_code == 0, first.output
        old_marker = image_completion_marker_path(
            output_dir, "plates", "plate_001"
        )
        old_marker_bytes = old_marker.read_bytes()

        _write_synth_image(plates_input_dir / "plate_002.png")
        second = runner.invoke(phenotypic_cli, args)
        assert second.exit_code == 0, second.output
        assert "Continuing processing (1 images remaining)" in second.output
        assert old_marker.read_bytes() == old_marker_bytes
        assert zarr_store_path(output_dir, "plates", "plate_002").is_dir()
        manifest = json.loads(
            manifest_json_path(output_dir).read_text(encoding="utf-8")
        )
        assert manifest["successful"] == 2
        assert manifest["terminal_failed"] == 0
        assert manifest["pending"] == 0

        aggregate_bytes = aggregate_publication_marker_path(
            output_dir
        ).read_bytes()
        run_bytes = run_completion_marker_path(output_dir).read_bytes()
        third = runner.invoke(phenotypic_cli, args)
        assert third.exit_code == 0, third.output
        assert "All images already processed" in third.output
        assert aggregate_publication_marker_path(output_dir).read_bytes() == (
            aggregate_bytes
        )
        assert run_completion_marker_path(output_dir).read_bytes() == run_bytes

        # Marker-only image no-op still repairs a crashed finalizer. No image
        # success evidence is rewritten while aggregate and run publication
        # evidence are rebuilt from current marker-authorized sources.
        aggregate_publication_marker_path(output_dir).unlink()
        run_completion_marker_path(output_dir).unlink()
        repaired = runner.invoke(phenotypic_cli, args)
        assert repaired.exit_code == 0, repaired.output
        assert "Aggregating measurements" in repaired.output
        assert old_marker.read_bytes() == old_marker_bytes
        assert valid_aggregate_snapshot(output_dir) is not None
        assert valid_run_completion(output_dir) is not None


# ---------------------------------------------------------------------------
# Overlays — always-on for forward runs
# ---------------------------------------------------------------------------


class TestOverlayAlwaysOn:
    """Forward runs always write an overlay PNG per image."""

    def test_forward_run_writes_overlay(
        self, runner, temp_pipeline, plates_input_dir, tmp_path
    ):
        """A default forward run produces an overlay PNG without any flag."""
        output_dir = tmp_path / "out"
        result = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(temp_pipeline),
                "--input",
                str(plates_input_dir),
                "-o",
                str(output_dir),
                "--njobs",
                "1",
                "--skip-validation",
                "--force-local",
            ],
        )
        assert result.exit_code == 0, (
            f"Default forward run failed:\n{result.output}"
        )

        overlay_png = (
            output_dir / "deliverables" / "overlays" / "plates" / "plate_001.png"
        )
        assert overlay_png.exists(), (
            f"Expected overlay PNG at {overlay_png} on a default forward run."
        )


# ---------------------------------------------------------------------------
# measure-mode rerun semantics
# ---------------------------------------------------------------------------


class TestMeasureRerun:
    """Schema-3 marker-authorized outputs reject direct measure mutation."""

    def test_measure_rerun_rewrites_measurements(
        self, runner, temp_pipeline, plates_input_dir, tmp_path
    ):
        """Measure mode cannot invalidate a marker-authorized image outcome."""
        output_dir = tmp_path / "out"

        # Forward run.
        forward = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(temp_pipeline),
                "--input",
                str(plates_input_dir),
                "-o",
                str(output_dir),
                "--njobs",
                "1",
                "--skip-validation",
                "--force-local",
            ],
        )
        assert forward.exit_code == 0, (
            f"Initial forward run failed:\n{forward.output}"
        )

        store = zarr_store_path(output_dir, "plates", "plate_001")
        store_root = store / "zarr.json"
        table_path = store / MEASUREMENT_TABLE_RELATIVE_PATH
        overlay_path = output_dir / "deliverables" / "overlays" / "plates" / "plate_001.png"

        assert store_root.is_file()
        assert table_path.is_file()
        assert not (output_dir / "results" / "plates" / "measurements").exists()
        assert overlay_path.exists(), (
            "Forward run should always write overlay PNG."
        )

        # The root `zarr.json` is written LAST by promote_store, so its
        # mtime is the whole store's publication time.
        store_mtime_before = store_root.stat().st_mtime_ns
        table_bytes_before = table_path.read_bytes()
        overlay_mtime_before = overlay_path.stat().st_mtime_ns

        # Sleep so mtimes actually differ on filesystems that round to seconds.
        time.sleep(0.1)

        # Measure rerun discovers image stores under the existing output root.
        measure = runner.invoke(
            phenotypic_cli,
            [
                "--mode",
                "measure",
                "--pipeline",
                str(temp_pipeline),
                "-o",
                str(output_dir),
                "--njobs",
                "1",
                "--skip-validation",
                "--force-local",
            ],
        )
        assert measure.exit_code != 0
        assert "cannot mutate a marker-authorized incremental run" in measure.output
        assert table_path.read_bytes() == table_bytes_before
        assert store_root.stat().st_mtime_ns == store_mtime_before
        assert overlay_path.stat().st_mtime_ns == overlay_mtime_before

    def test_measure_rerun_grid_image_store(
        self, runner, temp_pipeline, plates_input_dir, tmp_path
    ):
        """A GridImage store carries the state measure mode dispatches on."""
        output_dir = tmp_path / "out_grid"

        forward = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(temp_pipeline),
                "--input",
                str(plates_input_dir),
                "-o",
                str(output_dir),
                "--njobs",
                "1",
                "--skip-validation",
                "--force-local",
                "--image-type",
                "GridImage",
                "--nrows",
                "8",
                "--ncols",
                "12",
            ],
        )
        assert forward.exit_code == 0, (
            f"GridImage forward run failed:\n{forward.output}"
        )

        store = zarr_store_path(output_dir, "plates", "plate_001")
        assert store.is_dir(), f"Expected an OME-Zarr store at {store}"

        # ``image_class`` is what ``load_image_from_store`` dispatches on, so
        # the measure worker rehydrates a GridImage rather than degrading to a
        # plain Image. It is deliberately NOT ``Metadata_ImageType``.
        block = read_phenotypic_attributes(store)
        assert block[PhenotypicAttr.IMAGE_CLASS] == "GridImage", (
            f"Expected image_class='GridImage' so the SLURM/local worker "
            f"auto-selects GridImage.load_zarr, got "
            f"{block.get(PhenotypicAttr.IMAGE_CLASS)!r}."
        )

        # The grid state itself must round-trip, or grid info is silently lost
        # on reload even though image_class is right.
        from phenotypic import GridImage

        reloaded = GridImage.load_zarr(store)
        assert isinstance(reloaded, GridImage)
        assert (reloaded.nrows, reloaded.ncols) == (8, 12)
        assert reloaded.grid_finder is not None, (
            "The grid finder was not rehydrated from the store."
        )
        assert (
            reloaded.grid_finder.nrows,
            reloaded.grid_finder.ncols,
        ) == (8, 12)

        time.sleep(0.1)

        measure = runner.invoke(
            phenotypic_cli,
            [
                "--mode",
                "measure",
                "--pipeline",
                str(temp_pipeline),
                "-o",
                str(output_dir),
                "--njobs",
                "1",
                "--skip-validation",
                "--force-local",
                "--image-type",
                "GridImage",
                "--nrows",
                "8",
                "--ncols",
                "12",
            ],
        )
        assert measure.exit_code != 0
        assert "cannot mutate a marker-authorized incremental run" in measure.output

        table_path = store / MEASUREMENT_TABLE_RELATIVE_PATH
        assert table_path.is_file(), (
            f"Expected embedded measurements at {table_path}"
        )
        assert not (output_dir / "results" / "plates" / "measurements").exists()

        df = pd.read_parquet(table_path)
        assert not df.empty

        # Sanity: the canonical category-prefixed columns from the
        # RoundPeaksPipeline measurement set (Shape + Intensity) must be
        # present — catches regressions where measure mode silently returns a
        # column-schema-stripped frame.
        shape_cols = {c for c in df.columns if c.startswith("Shape_")}
        intensity_cols = {c for c in df.columns if c.startswith("Intensity_")}
        assert shape_cols, (
            f"Measurements parquet is missing Shape_* columns after forward mode; "
            f"columns={list(df.columns)}"
        )
        assert intensity_cols, (
            f"Measurements parquet is missing Intensity_* columns after forward mode; "
            f"columns={list(df.columns)}"
        )

    def test_measure_rerun_overwrites_existing_parquet(
        self, runner, temp_pipeline, plates_input_dir, tmp_path
    ):
        """Forward run + --overwrite forward run leaves a readable parquet."""
        output_dir = tmp_path / "out_overwrite"

        first = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(temp_pipeline),
                "--input",
                str(plates_input_dir),
                "-o",
                str(output_dir),
                "--njobs",
                "1",
                "--skip-validation",
                "--force-local",
            ],
        )
        assert first.exit_code == 0, f"First forward run failed:\n{first.output}"

        store = zarr_store_path(output_dir, "plates", "plate_001")
        table_path = store / MEASUREMENT_TABLE_RELATIVE_PATH
        assert table_path.is_file()
        assert not (output_dir / "results" / "plates" / "measurements").exists()
        table_mtime_before = table_path.stat().st_mtime_ns

        time.sleep(0.1)

        second = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(temp_pipeline),
                "--input",
                str(plates_input_dir),
                "-o",
                str(output_dir),
                "--njobs",
                "1",
                "--skip-validation",
                "--force-local",
                "--overwrite",
            ],
        )
        assert second.exit_code == 0, (
            f"Second (--overwrite) forward run failed:\n{second.output}"
        )

        assert table_path.is_file(), (
            "Embedded measurements missing after --overwrite rerun."
        )
        assert not (output_dir / "results" / "plates" / "measurements").exists()
        table_mtime_after = table_path.stat().st_mtime_ns
        assert table_mtime_after > table_mtime_before, (
            f"Embedded table mtime did not advance after --overwrite rerun "
            f"(before={table_mtime_before}, after={table_mtime_after})."
        )

        # The replacement is readable, non-empty, and remains authorized.
        df = pd.read_parquet(table_path)
        assert not df.empty, (
            "Embedded table rewritten by --overwrite is unexpectedly empty."
        )
        assert valid_aggregate_snapshot(output_dir) is not None
        assert valid_run_completion(output_dir) is not None


# ---------------------------------------------------------------------------
# measure-mode validation
# ---------------------------------------------------------------------------


class TestMeasureFlagValidation:
    """Measure mode must reject incompatible flags with clear errors."""

    @pytest.fixture
    def prepared_output_dir(
        self, runner, temp_pipeline, plates_input_dir, tmp_path
    ):
        """Forward-run once so measure mode has stores to rediscover.

        The incompatible-flag checks fire *before* store discovery, so most of
        these tests would work against an empty output dir — but the
        non-existent-output-dir case needs an actual directory to contrast
        against, and we'd rather share setup than duplicate it.
        """
        output_dir = tmp_path / "prepared_out"
        forward = runner.invoke(
            phenotypic_cli,
            [
                "--pipeline",
                str(temp_pipeline),
                "--input",
                str(plates_input_dir),
                "-o",
                str(output_dir),
                "--njobs",
                "1",
                "--skip-validation",
                "--force-local",
            ],
        )
        assert forward.exit_code == 0, (
            f"Setup forward run failed:\n{forward.output}"
        )
        return output_dir

    def test_measure_rejects_sample(
        self, runner, temp_pipeline, prepared_output_dir
    ):
        """Measure mode with --sample N must exit non-zero with a clear message."""
        result = runner.invoke(
            phenotypic_cli,
            [
                "--mode",
                "measure",
                "--pipeline",
                str(temp_pipeline),
                "-o",
                str(prepared_output_dir),
                "--force-local",
                "--skip-validation",
                "--sample",
                "1",
            ],
        )
        assert result.exit_code != 0, (
            f"Measure mode with --sample should fail but got exit_code=0:\n"
            f"{result.output}"
        )
        assert "--mode measure cannot be combined with --sample" in result.output, (
            f"Error message missing expected substring:\n{result.output}"
        )

    def test_removed_resume_option_is_rejected(
        self, runner, temp_pipeline, prepared_output_dir
    ):
        """The removed public continuation option is rejected by Click."""
        result = runner.invoke(
            phenotypic_cli,
            [
                "--mode",
                "measure",
                "--pipeline",
                str(temp_pipeline),
                "-o",
                str(prepared_output_dir),
                "--force-local",
                "--skip-validation",
                "--resume",
            ],
        )
        assert result.exit_code != 0, (
            f"Removed option should fail but got exit_code=0:\n"
            f"{result.output}"
        )
        assert "No such option: --resume" in result.output, (
            f"Error message missing expected substring:\n{result.output}"
        )

    def test_measure_rejects_restart(
        self, runner, temp_pipeline, prepared_output_dir
    ):
        """Measure mode with --restart must exit non-zero with a clear message."""
        result = runner.invoke(
            phenotypic_cli,
            [
                "--mode",
                "measure",
                "--pipeline",
                str(temp_pipeline),
                "-o",
                str(prepared_output_dir),
                "--force-local",
                "--skip-validation",
                "--restart",
            ],
        )
        assert result.exit_code != 0, (
            f"Measure mode with --restart should fail but got exit_code=0:\n"
            f"{result.output}"
        )
        assert "--mode measure cannot be combined with --restart" in result.output, (
            f"Error message missing expected substring:\n{result.output}"
        )

    def test_measure_rejects_retry_failures(
        self, runner, temp_pipeline, prepared_output_dir
    ):
        """Measure mode with --retry-failures must exit non-zero."""
        result = runner.invoke(
            phenotypic_cli,
            [
                "--mode",
                "measure",
                "--pipeline",
                str(temp_pipeline),
                "-o",
                str(prepared_output_dir),
                "--force-local",
                "--skip-validation",
                "--retry-failures",
            ],
        )
        assert result.exit_code != 0, (
            f"Measure mode with --retry-failures should fail but got exit_code=0:\n"
            f"{result.output}"
        )
        assert (
            "--mode measure cannot be combined with --retry-failures" in result.output
        ), f"Error message missing expected substring:\n{result.output}"

    def test_measure_rejects_overwrite(
        self, runner, temp_pipeline, prepared_output_dir
    ):
        """Measure mode with --overwrite must exit non-zero with a clear message."""
        result = runner.invoke(
            phenotypic_cli,
            [
                "--mode",
                "measure",
                "--pipeline",
                str(temp_pipeline),
                "-o",
                str(prepared_output_dir),
                "--force-local",
                "--skip-validation",
                "--overwrite",
            ],
        )
        assert result.exit_code != 0, (
            f"Measure mode with --overwrite should fail but got exit_code=0:\n"
            f"{result.output}"
        )
        assert "--mode measure cannot be combined with --overwrite" in result.output, (
            f"Error message missing expected substring:\n{result.output}"
        )

    def test_measure_rejects_missing_output_dir(
        self, runner, temp_pipeline, tmp_path
    ):
        """Measure mode against a non-existent -o dir must exit non-zero."""
        missing_dir = tmp_path / "does_not_exist"
        assert not missing_dir.exists()

        result = runner.invoke(
            phenotypic_cli,
            [
                "--mode",
                "measure",
                "--pipeline",
                str(temp_pipeline),
                "-o",
                str(missing_dir),
                "--force-local",
                "--skip-validation",
            ],
        )
        assert result.exit_code != 0, (
            f"Measure mode against missing -o dir should fail but got "
            f"exit_code=0:\n{result.output}"
        )
        assert (
            "--mode measure output directory does not exist" in result.output
        ), f"Error message missing expected substring:\n{result.output}"
