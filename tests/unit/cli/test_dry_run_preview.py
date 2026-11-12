"""``--dry-run`` describes the run that would actually happen.

Plan ``docs/superpowers/plans/2026-10-31-docs-reorganization/03-small-code-fixes.md``
Task 6; design §10 item 9. The preview used to print "None rows × None
columns" (the default ``--image-type`` is ``GridImage`` and no ``--nrows`` was
given), the pre-``deliverables/`` output tree, and none of the pipeline's
steps. ``resolve_grid_shape`` returns no source, so the preview works out
where the shape came from itself (K0-26).

The rich configuration panel printed before the preview already says
"default 8 × 12", so the grid assertions here match the preview's own
``Grid shape:`` line, not the bare ``8 × 12`` the panel would satisfy.
"""

from __future__ import annotations

import os
from pathlib import Path

import click
import numpy as np
import pytest
import tifffile
from click.testing import CliRunner

from phenotypic import ImagePipeline
from phenotypic._cli._cli_interactive import _CHANGED_WIDTH, execute_dry_run
from phenotypic.analysis import LogGrowthModel
from phenotypic.detect import CompositeDetector, OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.measure import MeasureSize
from phenotypic.phenotypicCLI import _display_execution_config, phenotypic_cli

from ._preflight_support import make_config, make_datasets


def _plate_image() -> np.ndarray:
    """A small RGB plate: dark agar with two bright colonies."""
    image = np.full((32, 32, 3), 20, dtype=np.uint8)
    image[6:12, 6:12] = 220
    image[20:27, 18:25] = 200
    return image


def _save_pipeline(path: Path, ops=None, **kwargs) -> Path:
    pipeline = ImagePipeline(
        ops=ops if ops is not None else [BlurGauss(sigma=3.0), OtsuDetector()],
        meas=[MeasureSize()],
        **kwargs,
    )
    # write_text, not to_json(path): the latter appends ``.pht-pipe``.
    path.write_text(pipeline.to_json(), encoding="utf-8")
    return path


def _dry_run_cli(pipe: Path, inp: Path, tmp_path: Path):
    """Run the real CLI with ``--dry-run`` over *pipe* and *inp*."""
    args = ["--pipeline", str(pipe), "--input", str(inp), "--output", str(tmp_path / "out"), "--dry-run"]
    # A wide terminal, so the rich configuration panel does not wrap its cells.
    return CliRunner().invoke(phenotypic_cli, args, env={"COLUMNS": "200"})


def _section(output: str, heading: str) -> list[str]:
    """The lines under *heading*, up to the next blank line."""
    lines = output.splitlines()
    start = lines.index(heading) + 1
    end = next((i for i in range(start, len(lines)) if not lines[i].strip()), len(lines))
    return lines[start:end]


def _preview(config, tmp_path: Path) -> str:
    """Run ``execute_dry_run`` alone over a one-image dataset and return what it printed."""
    datasets = make_datasets(tmp_path / "plate1" / "a.tiff")

    @click.command()
    def _command() -> None:
        execute_dry_run(config, list(datasets), tmp_path / "out")

    result = CliRunner().invoke(_command, [])
    assert result.exit_code == 0, result.output
    return result.output


@pytest.fixture
def pipe(tmp_path: Path) -> Path:
    return _save_pipeline(tmp_path / "pipeline.json")


@pytest.fixture
def inp(tmp_path: Path) -> Path:
    image = tmp_path / "images" / "plate1" / "img001.tiff"
    image.parent.mkdir(parents=True)
    tifffile.imwrite(image, _plate_image())
    return tmp_path / "images"


def test_dry_run_preview_is_current(tmp_path, pipe, inp):
    result = _dry_run_cli(pipe, inp, tmp_path)
    assert result.exit_code == 0, result.output
    assert "None rows" not in result.output
    assert "8 × 12" in result.output
    assert "deliverables" in result.output
    for name in ("BlurGauss", "OtsuDetector", "MeasureSize"):
        assert name in result.output
    # The preview's own line, not the configuration panel's "default 8 × 12".
    assert "Grid shape: 8 × 12 (rows × columns), from the built-in default" in result.output
    # The summary table, with the one changed parameter.
    assert "sigma=3.0" in result.output
    # The current layout, by its real file names.
    for current in (
        "deliverables/measurements.csv",
        "deliverables/master_measurements.parquet",
        "results/<dataset>/zarr/<stem>.ome.zarr",
        "img001.ome.zarr",
    ):
        assert current.replace("/", os.sep) in result.output, current
    # The pre-deliverables tree is gone.
    for stale in ("measurements/ (*.csv)", "in measurements/", "rgb/ (*.", "Layers always saved"):
        assert stale not in result.output, stale
    assert not (tmp_path / "out").exists()


def test_flags_and_pipeline_each_name_their_axis(tmp_path):
    pipe = _save_pipeline(tmp_path / "pipeline.json", nrows=4, ncols=24)
    config = make_config(pipeline_json=pipe, nrows=16)
    output = _preview(config, tmp_path)
    assert "Grid shape: 16 × 24 (rows × columns), rows from --nrows, columns from the pipeline" in output


def test_a_pipeline_preset_is_reported_as_the_source(tmp_path):
    pipe = _save_pipeline(tmp_path / "pipeline.json", nrows=16, ncols=24)
    output = _preview(make_config(pipeline_json=pipe), tmp_path)
    assert "Grid shape: 16 × 24 (rows × columns), from the pipeline" in output


def test_measure_mode_lists_only_the_measurers_and_the_stored_grid(tmp_path, pipe):
    config = make_config(pipeline_json=pipe, measure_only=True)
    output = _preview(config, tmp_path)
    assert "MeasureSize" in output
    assert "BlurGauss" not in output
    assert "Grid shape: 8 × 12" not in output
    assert "as stored in each image store" in output


def test_an_unreadable_pipeline_does_not_fail_the_preview(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text("{ not json", encoding="utf-8")
    config = make_config(pipeline_json=broken, skip_validation=True)
    output = _preview(config, tmp_path)
    assert "could not be read" in output
    assert "Failed to load pipeline: ValueError: Invalid JSON data" in output  # read_pipeline_file's wording (W2 S5)
    assert "None rows" not in output


def test_the_configuration_panel_agrees_with_the_preview(tmp_path, pipe, inp):
    """The rich panel prints the preview's own grid text (review W2 S1)."""
    result = _dry_run_cli(pipe, inp, tmp_path)
    assert result.exit_code == 0, result.output
    assert "auto (from pipeline preset" not in result.output
    grid = "8 × 12 (rows × columns), from the built-in default"
    assert result.output.count(grid) == 2, result.output  # the panel and the preview


def test_the_panel_names_the_stored_grid_in_measure_mode(tmp_path, pipe, capsys, monkeypatch):
    monkeypatch.setenv("COLUMNS", "200")
    _display_execution_config(make_config(pipeline_json=pipe, measure_only=True), [])
    output = capsys.readouterr().out
    assert "as stored in each image store" in output
    assert "8 × 12" not in output


def test_the_step_table_is_left_aligned_and_wraps_long_parameters(tmp_path):
    wide = CompositeDetector(
        ops=[
            OtsuDetector(ignore_zeros=True),
            OtsuDetector(ignore_borders=False),
            OtsuDetector(ignore_zeros=True, ignore_borders=False),
        ]
    )
    pipe = _save_pipeline(tmp_path / "pipeline.json", ops=[BlurGauss(sigma=3.0), wide])
    output = _preview(make_config(pipeline_json=pipe), tmp_path)
    steps = _section(output, "Pipeline Steps:")
    assert steps[0].split() == ["step", "slot", "operation", "module", "writes", "changed"]
    assert steps[1].startswith("  BlurGauss ")  # left-aligned, not padded from the left
    changed_at = steps[0].index("changed")
    assert all(len(line) <= changed_at + _CHANGED_WIDTH for line in steps), steps
    assert len(steps) > 4  # the composite's cell wrapped onto continuation lines
    expected = ImagePipeline(ops=[BlurGauss(sigma=3.0), wide]).summary()["changed"].iloc[1]
    assert "".join("".join(steps).split()).count("".join(expected.split())) == 1


def test_the_layout_lists_every_deliverable_a_plain_run_writes(tmp_path, pipe):
    output = _preview(make_config(pipeline_json=pipe), tmp_path)
    layout = "\n".join(_section(output, "Output Layout:"))
    for written in (
        "measurements_by_feature",
        "measurements_by_category",
        "dashboard.html",
        "README.md",
        "rembi.yaml",
        "pipeline.json.pht-pipe",
    ):
        assert written in layout, written
    assert "processing_report.html" not in layout
    assert "every image's measurements, without your --metadata columns" in layout
    # Written only when configured: no --metadata, model, plots or QC here.
    for configured in ("metadata.csv", "analysis_manifest.json", "plots", "qc"):
        assert os.sep + configured not in layout, configured


def test_the_layout_adds_what_metadata_and_a_model_write(tmp_path):
    pipe = _save_pipeline(
        tmp_path / "pipeline.json", model=LogGrowthModel(on="Size_Area", groupby=["Metadata_Strain"])
    )
    config = make_config(pipeline_json=pipe, metadata_csv=tmp_path / "plates.csv")
    layout = "\n".join(_section(_preview(config, tmp_path), "Output Layout:"))
    for written in ("metadata.csv", "LogGrowthModel.csv", "analysis_manifest.json"):
        assert written in layout, written
