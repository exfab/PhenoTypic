"""
Integration tests for the HDF-centric CLI output layout and ``--measure`` rerun.

These tests exercise the full CLI via :mod:`click.testing.CliRunner` to verify
the behaviour promised by Phase 7 of the HDF-centric CLI plan:

* Forward runs write exactly one HDF per image under
  ``results/<dataset>/hdf/<stem>.h5`` and a parquet measurements file under
  ``results/<dataset>/measurements/<stem>.parquet`` — and nothing else
  (no per-layer ``rgb/`` / ``gray/`` / ``detect_mat/`` / ``objmap/`` folders).
* Forward runs always write a PNG overlay per image alongside the HDF.
* ``--measure`` reruns :meth:`ImagePipeline.measure` on existing HDFs
  without touching the HDF files on disk and without regenerating overlays.
* ``--measure`` rejects incompatible flags with a clear error message.

All tests pass ``--force-local`` so SLURM is never dispatched.
"""

import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner
from PIL import Image as PILImage

from phenotypic.data import load_synth_yeast_plate
from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.prefab import RoundPeaksPipeline


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
# Forward run — HDF-only output layout
# ---------------------------------------------------------------------------


class TestForwardRunHdfLayout:
    """A forward run must write HDF + parquet and nothing else."""

    def test_forward_run_produces_hdf_only(
        self, runner, temp_pipeline, plates_input_dir, tmp_path
    ):
        """Forward run writes hdf/ + measurements/; no per-layer folders."""
        output_dir = tmp_path / "out"

        result = runner.invoke(
            phenotypic_cli,
            [
                str(temp_pipeline),
                str(plates_input_dir),
                "-o",
                str(output_dir),
                "--n-jobs",
                "1",
                "--skip-validation",
                "--force-local",
            ],
        )

        assert result.exit_code == 0, (
            f"CLI failed (exit_code={result.exit_code}):\n{result.output}"
        )

        dataset_dir = output_dir / "results" / "plates"
        hdf_file = dataset_dir / "hdf" / "plate_001.h5"
        parquet_file = dataset_dir / "measurements" / "plate_001.parquet"
        overlay_file = dataset_dir / "overlays" / "plate_001.png"

        assert hdf_file.exists(), (
            f"Expected HDF output at {hdf_file} (got contents: "
            f"{list(dataset_dir.rglob('*')) if dataset_dir.exists() else 'no dataset dir'})"
        )
        assert parquet_file.exists(), (
            f"Expected parquet measurements at {parquet_file}"
        )
        assert overlay_file.exists(), (
            f"Expected overlay PNG at {overlay_file} (overlays are always "
            f"written for forward runs)"
        )

        # None of the legacy per-layer directories must be created.
        for legacy_folder in ("rgb", "gray", "detect_mat", "objmap"):
            legacy_path = dataset_dir / legacy_folder
            assert not legacy_path.exists(), (
                f"Legacy per-layer folder {legacy_path} should NOT exist "
                f"after forward run; CLI regressed to pre-HDF layout."
            )

        # Whitelist check: after a default forward run the dataset directory
        # must contain exactly `hdf/`, `measurements/`, and `overlays/`. Any
        # other subdir (e.g. a renamed or new per-layer folder) would regress
        # the layout.
        actual_children = {p.name for p in dataset_dir.iterdir() if p.is_dir()}
        assert actual_children == {"hdf", "measurements", "overlays"}, (
            f"Unexpected dataset-level folders after forward run. "
            f"Got {sorted(actual_children)}; expected "
            f"{{'hdf', 'measurements', 'overlays'}}."
        )


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
                str(temp_pipeline),
                str(plates_input_dir),
                "-o",
                str(output_dir),
                "--n-jobs",
                "1",
                "--skip-validation",
                "--force-local",
            ],
        )
        assert result.exit_code == 0, (
            f"Default forward run failed:\n{result.output}"
        )

        overlay_png = (
            output_dir / "results" / "plates" / "overlays" / "plate_001.png"
        )
        assert overlay_png.exists(), (
            f"Expected overlay PNG at {overlay_png} on a default forward run."
        )


# ---------------------------------------------------------------------------
# --measure rerun semantics
# ---------------------------------------------------------------------------


class TestMeasureRerun:
    """``--measure`` reruns measurements without touching HDFs or overlays."""

    def test_measure_rerun_rewrites_measurements(
        self, runner, temp_pipeline, plates_input_dir, tmp_path
    ):
        """Forward run, then --measure: parquet changes, HDF does not."""
        output_dir = tmp_path / "out"

        # Forward run.
        forward = runner.invoke(
            phenotypic_cli,
            [
                str(temp_pipeline),
                str(plates_input_dir),
                "-o",
                str(output_dir),
                "--n-jobs",
                "1",
                "--skip-validation",
                "--force-local",
            ],
        )
        assert forward.exit_code == 0, (
            f"Initial forward run failed:\n{forward.output}"
        )

        dataset_dir = output_dir / "results" / "plates"
        hdf_path = dataset_dir / "hdf" / "plate_001.h5"
        parquet_path = dataset_dir / "measurements" / "plate_001.parquet"
        overlay_path = dataset_dir / "overlays" / "plate_001.png"

        assert hdf_path.exists()
        assert parquet_path.exists()
        assert overlay_path.exists(), (
            "Forward run should always write overlay PNG."
        )

        hdf_mtime_before = hdf_path.stat().st_mtime_ns
        parquet_mtime_before = parquet_path.stat().st_mtime_ns
        overlay_mtime_before = overlay_path.stat().st_mtime_ns

        # Sleep so mtimes actually differ on filesystems that round to seconds.
        time.sleep(0.1)

        # --measure rerun. INPUT_PATH is still accepted (but ignored with a
        # warning); we pass the same one for minimal drift from a real user.
        measure = runner.invoke(
            phenotypic_cli,
            [
                str(temp_pipeline),
                str(plates_input_dir),
                "-o",
                str(output_dir),
                "--n-jobs",
                "1",
                "--skip-validation",
                "--force-local",
                "--measure",
            ],
        )
        assert measure.exit_code == 0, (
            f"--measure rerun failed:\n{measure.output}"
        )

        # Parquet must be rewritten (mtime advanced).
        parquet_mtime_after = parquet_path.stat().st_mtime_ns
        assert parquet_mtime_after > parquet_mtime_before, (
            f"Parquet mtime did not advance after --measure rerun "
            f"(before={parquet_mtime_before}, after={parquet_mtime_after}). "
            f"--measure should rewrite measurements."
        )

        # HDF must be untouched.
        hdf_mtime_after = hdf_path.stat().st_mtime_ns
        assert hdf_mtime_after == hdf_mtime_before, (
            f"HDF mtime changed after --measure rerun "
            f"(before={hdf_mtime_before}, after={hdf_mtime_after}). "
            f"--measure must not rewrite HDF files."
        )

        # Existing overlay must NOT be rewritten by a measure rerun.
        overlay_mtime_after = overlay_path.stat().st_mtime_ns
        assert overlay_mtime_after == overlay_mtime_before, (
            f"Overlay mtime changed after --measure rerun "
            f"(before={overlay_mtime_before}, after={overlay_mtime_after}). "
            f"--measure must not regenerate overlays."
        )

    def test_measure_rerun_grid_image_hdf(
        self, runner, temp_pipeline, plates_input_dir, tmp_path
    ):
        """GridImage HDFs round-trip through --measure successfully."""
        output_dir = tmp_path / "out_grid"

        forward = runner.invoke(
            phenotypic_cli,
            [
                str(temp_pipeline),
                str(plates_input_dir),
                "-o",
                str(output_dir),
                "--n-jobs",
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

        hdf_path = output_dir / "results" / "plates" / "hdf" / "plate_001.h5"
        assert hdf_path.exists(), f"Expected HDF at {hdf_path}"

        # Confirm the HDF is tagged as a GridImage so the worker's
        # auto-dispatch routes to GridImage.load_hdf5.
        import h5py

        with h5py.File(hdf_path, "r") as f:
            phenotypic_class = f.attrs.get("phenotypic_class")
            # h5py returns bytes for string attrs on some platforms.
            if isinstance(phenotypic_class, bytes):
                phenotypic_class = phenotypic_class.decode("utf-8")
            assert phenotypic_class == "GridImage", (
                f"Expected phenotypic_class='GridImage' so the SLURM/local "
                f"worker auto-selects GridImage.load_hdf5, got "
                f"{phenotypic_class!r}."
            )

            # The /grid/ subgroup + grid_finder_json dataset are what
            # actually restore grid state on load. Their absence would
            # silently drop grid info even though phenotypic_class is set.
            assert "grid" in f, (
                "GridImage HDF is missing /grid/ subgroup; grid state "
                "would be lost on reload."
            )
            grid_group = f["grid"]
            assert "grid_finder_json" in grid_group, (
                "/grid/grid_finder_json dataset missing; the grid finder "
                "cannot be rehydrated on load."
            )
            assert "nrows" in grid_group.attrs, (
                "/grid/ missing `nrows` attr."
            )
            assert "ncols" in grid_group.attrs, (
                "/grid/ missing `ncols` attr."
            )

        time.sleep(0.1)

        measure = runner.invoke(
            phenotypic_cli,
            [
                str(temp_pipeline),
                "-o",
                str(output_dir),
                "--n-jobs",
                "1",
                "--skip-validation",
                "--force-local",
                "--measure",
                "--image-type",
                "GridImage",
                "--nrows",
                "8",
                "--ncols",
                "12",
            ],
        )
        assert measure.exit_code == 0, (
            f"GridImage --measure rerun failed:\n{measure.output}"
        )

        parquet_path = (
            output_dir / "results" / "plates" / "measurements" / "plate_001.parquet"
        )
        assert parquet_path.exists(), (
            f"Expected rewritten parquet at {parquet_path}"
        )

        df = pd.read_parquet(parquet_path)
        assert not df.empty, (
            "GridImage --measure rerun produced an empty measurements frame."
        )

        # Sanity: the canonical category-prefixed columns from the
        # RoundPeaksPipeline measurement set (Shape + Intensity) must be
        # present — catches regressions where --measure silently returns a
        # column-schema-stripped frame.
        shape_cols = {c for c in df.columns if c.startswith("Shape_")}
        intensity_cols = {c for c in df.columns if c.startswith("Intensity_")}
        assert shape_cols, (
            f"Measurements parquet is missing Shape_* columns after --measure; "
            f"columns={list(df.columns)}"
        )
        assert intensity_cols, (
            f"Measurements parquet is missing Intensity_* columns after --measure; "
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
                str(temp_pipeline),
                str(plates_input_dir),
                "-o",
                str(output_dir),
                "--n-jobs",
                "1",
                "--skip-validation",
                "--force-local",
            ],
        )
        assert first.exit_code == 0, f"First forward run failed:\n{first.output}"

        parquet_path = (
            output_dir / "results" / "plates" / "measurements" / "plate_001.parquet"
        )
        assert parquet_path.exists()
        parquet_mtime_before = parquet_path.stat().st_mtime_ns

        time.sleep(0.1)

        second = runner.invoke(
            phenotypic_cli,
            [
                str(temp_pipeline),
                str(plates_input_dir),
                "-o",
                str(output_dir),
                "--n-jobs",
                "1",
                "--skip-validation",
                "--force-local",
                "--overwrite",
            ],
        )
        assert second.exit_code == 0, (
            f"Second (--overwrite) forward run failed:\n{second.output}"
        )

        assert parquet_path.exists(), (
            "Parquet measurements file missing after --overwrite rerun."
        )
        parquet_mtime_after = parquet_path.stat().st_mtime_ns
        assert parquet_mtime_after > parquet_mtime_before, (
            f"Parquet mtime did not advance after --overwrite rerun "
            f"(before={parquet_mtime_before}, after={parquet_mtime_after})."
        )

        # Sanity check: the parquet is readable and non-empty.
        df = pd.read_parquet(parquet_path)
        assert not df.empty, (
            "Parquet rewritten by --overwrite forward run is unexpectedly empty."
        )


# ---------------------------------------------------------------------------
# --measure flag validation
# ---------------------------------------------------------------------------


class TestMeasureFlagValidation:
    """``--measure`` must reject incompatible flags with clear errors."""

    @pytest.fixture
    def prepared_output_dir(
        self, runner, temp_pipeline, plates_input_dir, tmp_path
    ):
        """Forward-run once so ``--measure`` has HDFs to rediscover.

        The incompatible-flag checks fire *before* HDF discovery, so most of
        these tests would work against an empty output dir — but the
        non-existent-output-dir case needs an actual directory to contrast
        against, and we'd rather share setup than duplicate it.
        """
        output_dir = tmp_path / "prepared_out"
        forward = runner.invoke(
            phenotypic_cli,
            [
                str(temp_pipeline),
                str(plates_input_dir),
                "-o",
                str(output_dir),
                "--n-jobs",
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
        """--measure --sample N must exit non-zero with a clear message."""
        result = runner.invoke(
            phenotypic_cli,
            [
                str(temp_pipeline),
                "-o",
                str(prepared_output_dir),
                "--force-local",
                "--skip-validation",
                "--measure",
                "--sample",
                "1",
            ],
        )
        assert result.exit_code != 0, (
            f"--measure --sample should fail but got exit_code=0:\n"
            f"{result.output}"
        )
        assert "--measure cannot be combined with --sample" in result.output, (
            f"Error message missing expected substring:\n{result.output}"
        )

    def test_measure_rejects_resume(
        self, runner, temp_pipeline, prepared_output_dir
    ):
        """--measure --resume must exit non-zero with a clear message."""
        result = runner.invoke(
            phenotypic_cli,
            [
                str(temp_pipeline),
                "-o",
                str(prepared_output_dir),
                "--force-local",
                "--skip-validation",
                "--measure",
                "--resume",
            ],
        )
        assert result.exit_code != 0, (
            f"--measure --resume should fail but got exit_code=0:\n"
            f"{result.output}"
        )
        assert "--measure cannot be combined with --resume" in result.output, (
            f"Error message missing expected substring:\n{result.output}"
        )

    def test_measure_rejects_restart(
        self, runner, temp_pipeline, prepared_output_dir
    ):
        """--measure --restart must exit non-zero with a clear message."""
        result = runner.invoke(
            phenotypic_cli,
            [
                str(temp_pipeline),
                "-o",
                str(prepared_output_dir),
                "--force-local",
                "--skip-validation",
                "--measure",
                "--restart",
            ],
        )
        assert result.exit_code != 0, (
            f"--measure --restart should fail but got exit_code=0:\n"
            f"{result.output}"
        )
        assert "--measure cannot be combined with --restart" in result.output, (
            f"Error message missing expected substring:\n{result.output}"
        )

    def test_measure_rejects_retry_failures(
        self, runner, temp_pipeline, prepared_output_dir
    ):
        """--measure --retry-failures must exit non-zero with a clear message."""
        result = runner.invoke(
            phenotypic_cli,
            [
                str(temp_pipeline),
                "-o",
                str(prepared_output_dir),
                "--force-local",
                "--skip-validation",
                "--measure",
                "--retry-failures",
            ],
        )
        assert result.exit_code != 0, (
            f"--measure --retry-failures should fail but got exit_code=0:\n"
            f"{result.output}"
        )
        assert (
            "--measure cannot be combined with --retry-failures" in result.output
        ), f"Error message missing expected substring:\n{result.output}"

    def test_measure_rejects_overwrite(
        self, runner, temp_pipeline, prepared_output_dir
    ):
        """--measure --overwrite must exit non-zero with a clear message."""
        result = runner.invoke(
            phenotypic_cli,
            [
                str(temp_pipeline),
                "-o",
                str(prepared_output_dir),
                "--force-local",
                "--skip-validation",
                "--measure",
                "--overwrite",
            ],
        )
        assert result.exit_code != 0, (
            f"--measure --overwrite should fail but got exit_code=0:\n"
            f"{result.output}"
        )
        assert "--measure cannot be combined with --overwrite" in result.output, (
            f"Error message missing expected substring:\n{result.output}"
        )

    def test_measure_rejects_missing_output_dir(
        self, runner, temp_pipeline, tmp_path
    ):
        """--measure against a non-existent -o dir must exit non-zero."""
        missing_dir = tmp_path / "does_not_exist"
        assert not missing_dir.exists()

        result = runner.invoke(
            phenotypic_cli,
            [
                str(temp_pipeline),
                "-o",
                str(missing_dir),
                "--force-local",
                "--skip-validation",
                "--measure",
            ],
        )
        assert result.exit_code != 0, (
            f"--measure against missing -o dir should fail but got "
            f"exit_code=0:\n{result.output}"
        )
        assert (
            "--measure output directory does not exist" in result.output
        ), f"Error message missing expected substring:\n{result.output}"
