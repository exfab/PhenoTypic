"""One initial CLI call per run, through the real CLI (figures spec §1a).

A fresh run records its call -- UTC date, UTC timestamp, pid -- in
``state.config``; continuing the run on a later day keeps all three; ``--restart``
records a new call. The clock is pinned by patching ``_utc_now``, the one clock
read every run date and timestamp comes from.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from phenotypic.phenotypicCLI import phenotypic_cli


def _run(out: Path, pipeline: Path, inputs: Path, *extra: str):
    result = CliRunner().invoke(
        phenotypic_cli,
        [
            "--pipeline", str(pipeline),
            "--input", str(inputs),
            "--output", str(out),
            "--force-local", "--skip-validation",
            "--njobs", "1",
            *extra,
        ],
    )
    assert result.exit_code == 0, result.output
    return result


def _recorded(out: Path) -> tuple:
    from phenotypic._cli._cli_state_management import load_processing_state

    config = load_processing_state(out).config
    return (
        config.get("figures_run_date"),
        config.get("initiated_at_utc"),
        config.get("initiated_pid"),
    )


def _clock(monkeypatch: pytest.MonkeyPatch, *when: int) -> None:
    from phenotypic.sdk_ import _image_figures

    instant = datetime(*when, tzinfo=timezone.utc)
    monkeypatch.setattr(_image_figures, "_utc_now", lambda: instant)


def test_a_continued_run_keeps_its_call_and_a_restart_records_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synth_plate_dir: Path,
    simple_pipeline_json: Path,
) -> None:
    import os

    out = tmp_path / "out"
    _clock(monkeypatch, 2026, 9, 22, 23, 59, 58)
    _run(out, simple_pipeline_json, synth_plate_dir)
    first = ("2026-09-22", "2026-09-22T23:59:58.000Z", os.getpid())
    assert _recorded(out) == first

    # A new image makes the continuation do real work, so the state is
    # rebuilt by the resume branch rather than left as the first run wrote it.
    from phenotypic.sdk_ import zarr_store_path
    from tests.integration.cli.conftest import _write_synth_image

    _write_synth_image(synth_plate_dir / "plate_002.png")
    _clock(monkeypatch, 2026, 9, 30, 8, 0, 0)
    _run(out, simple_pipeline_json, synth_plate_dir)
    assert _recorded(out) == first
    assert zarr_store_path(out, "plates", "plate_002").is_dir()

    _run(out, simple_pipeline_json, synth_plate_dir, "--restart")
    assert _recorded(out) == ("2026-09-30", "2026-09-30T08:00:00.000Z", os.getpid())


def test_a_local_run_across_midnight_writes_one_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, synth_plate_dir: Path
) -> None:
    """MINOR-12: the clock moves to the next day between the two images, and
    both still land in the run's one folder -- the day the call was made."""
    from phenotypic import ImagePipeline
    from phenotypic._cli import _cli_execution_strategies
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureSize
    from phenotypic.plotting import PlotDiagnostics
    from phenotypic.sdk_ import _image_figures, zarr_store_path
    from phenotypic.sdk_._image_figures import read_image_figures_descriptor
    from tests.integration.cli.conftest import _write_synth_image

    _write_synth_image(synth_plate_dir / "plate_002.png")
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text(
        ImagePipeline(
            ops={"detect": OtsuDetector()},
            meas={"size": MeasureSize()},
            plots=[PlotDiagnostics()],
        ).to_json(),
        encoding="utf-8",
    )
    clock = {"now": datetime(2026, 9, 22, 23, 59, 58, tzinfo=timezone.utc)}
    monkeypatch.setattr(_image_figures, "_utc_now", lambda: clock["now"])
    real = _cli_execution_strategies.process_single_image_core
    processed: list[str] = []

    def _then_midnight(*args: object, **kwargs: object) -> object:
        result = real(*args, **kwargs)
        processed.append(str(kwargs["image_path"]))
        clock["now"] = datetime(2026, 9, 23, 0, 0, 5, tzinfo=timezone.utc)
        return result

    monkeypatch.setattr(
        _cli_execution_strategies, "process_single_image_core", _then_midnight
    )

    out = tmp_path / "out"
    _run(out, pipeline, synth_plate_dir)

    assert len(processed) == 2
    runs = {
        stem: set(read_image_figures_descriptor(zarr_store_path(out, "plates", stem))["runs"])
        for stem in ("plate_001", "plate_002")
    }
    [only] = runs["plate_001"]
    assert only.startswith("2026-09-22-")
    assert runs["plate_002"] == {only}
