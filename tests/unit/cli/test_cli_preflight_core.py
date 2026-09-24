"""The run preflight's report types, orchestrator, scoping and CLI wiring.

Spec: ``docs/superpowers/specs/2026-09-24-cli-preflight/design.md`` §0, §2.
Plan: Task 3. The write tripwire here is re-run by every later task that adds
a check; it is what enforces "the preflight writes nothing".
"""

from __future__ import annotations

import builtins
import os
import shutil
from pathlib import Path
from typing import get_args

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner

from phenotypic import ImagePipeline
from phenotypic._cli import _cli_preflight
from phenotypic._cli._cli_preflight import (
    HINTS,
    MODE_SLOTS,
    FindingCode,
    PreflightContext,
    PreflightFinding,
    PreflightReport,
    load_pipeline_for_validation,
    operations_in_scope,
    run_preflight,
)
from phenotypic._cli._cli_validation import validate_pipeline
from phenotypic.detect import CompositeDetector, OtsuDetector
from phenotypic.enhance import BlurGauss
from phenotypic.measure import MeasureSize
from phenotypic.phenotypicCLI import phenotypic_cli
from phenotypic.post import AppendString


def _context(pipeline: ImagePipeline, mode: str = "full") -> PreflightContext:
    return PreflightContext(config=None, pipeline=pipeline, datasets=(), mode=mode)  # type: ignore[arg-type]


def _finding(severity: str, code: str = "PF-PIPELINE-LOAD", n: int = 0) -> PreflightFinding:
    return PreflightFinding(
        code=code,
        severity=severity,  # type: ignore[arg-type]
        message=f"{severity} finding",
        subjects=tuple(f"img{i:03d}.tiff" for i in range(n)),
    )


# --- orchestrator ---------------------------------------------------------


def test_a_crashing_check_becomes_one_warning_and_others_still_run() -> None:
    def broken(context):
        raise RuntimeError("boom")

    def healthy(context):
        return [_finding("warning")]

    report = run_preflight(_context(ImagePipeline(ops={})), checks=(broken, healthy))

    crashed = [f for f in report.findings if f.code == "PF-CHECK-CRASHED"]
    assert len(crashed) == 1
    assert crashed[0].severity == "warning"
    assert "broken" in crashed[0].message and "boom" in crashed[0].message
    assert len(report.findings) == 2
    assert not report.errors


def test_errors_and_warnings_partition_the_findings() -> None:
    findings = (_finding("warning"), _finding("error"), _finding("warning"))
    report = PreflightReport(findings)

    assert report.errors == (findings[1],)
    assert report.warnings == (findings[0], findings[2])


def test_render_prints_errors_first_and_caps_subjects() -> None:
    report = PreflightReport((_finding("warning", n=25), _finding("error", n=1)))

    lines = report.render_lines()

    assert lines[0].startswith("✗ Error [PF-PIPELINE-LOAD]")
    warning_at = next(i for i, line in enumerate(lines) if line.startswith("! Warning"))
    subjects = [line for line in lines[warning_at:] if line.startswith("    - ")]
    assert len(subjects) == 20
    assert "    … and 5 more" in lines
    assert all(line.strip() for line in lines)


def test_every_finding_code_has_a_hint_and_no_hint_is_orphaned() -> None:
    codes = set(get_args(FindingCode))
    assert codes == set(HINTS)
    assert all(HINTS[code].strip() for code in codes)


# --- scoping ----------------------------------------------------------------


@pytest.fixture
def slotted_pipeline() -> ImagePipeline:
    """One operation in every slot, plus a composite and a nested pipeline."""
    nested = ImagePipeline(
        ops={"inner_blur": BlurGauss()}, meas={"inner_size": MeasureSize()}
    )
    return ImagePipeline(
        ops={
            "composite": CompositeDetector(ops=[OtsuDetector(), OtsuDetector()]),
            "nested": nested,
        },
        meas={"size": MeasureSize()},
        post={"tag": AppendString(column="Strain", value="_x")},
    )


def _paths(pipeline: ImagePipeline, mode: str) -> set[str]:
    return {"/".join(path) for path, _ in operations_in_scope(_context(pipeline, mode))}


def test_full_mode_scopes_root_slots_and_only_the_ops_of_a_nested_pipeline(
    slotted_pipeline: ImagePipeline,
) -> None:
    paths = _paths(slotted_pipeline, "full")

    assert {"composite", "composite/ops[0]", "composite/ops[1]"} <= paths
    assert {"nested", "nested/inner_blur", "meas:size", "post:tag"} <= paths
    # A nested pipeline is applied, never measured.
    assert not any(p.startswith("nested/meas:") for p in paths)


def test_process_mode_scopes_ops_only(slotted_pipeline: ImagePipeline) -> None:
    paths = _paths(slotted_pipeline, "process")

    assert "composite/ops[0]" in paths and "nested/inner_blur" in paths
    assert not any(":" in p.split("/")[0] for p in paths)


def test_measure_mode_scopes_measurement_slots_only(slotted_pipeline: ImagePipeline) -> None:
    paths = _paths(slotted_pipeline, "measure")

    assert paths == {"meas:size", "post:tag"}


def test_mode_slots_are_the_three_modes() -> None:
    assert set(MODE_SLOTS) == {"full", "measure", "process"}


# --- loading ------------------------------------------------------------------


def test_load_pipeline_for_validation_returns_the_pipeline_or_one_finding(
    tmp_path: Path,
) -> None:
    good = tmp_path / "good.json"
    good.write_text(ImagePipeline(ops={"det": OtsuDetector()}).to_json(), encoding="utf-8")
    bad = tmp_path / "bad.json"
    bad.write_text("{ not json", encoding="utf-8")

    pipeline, finding = load_pipeline_for_validation(good)
    assert isinstance(pipeline, ImagePipeline) and finding is None

    pipeline, finding = load_pipeline_for_validation(bad)
    assert pipeline is None
    assert finding is not None and finding.code == "PF-PIPELINE-LOAD"
    assert finding.severity == "error"
    assert finding.message.startswith("Failed to load pipeline")


def test_validate_pipeline_keeps_its_verdict_contract(tmp_path: Path) -> None:
    good = tmp_path / "good.json"
    good.write_text(ImagePipeline(ops={"det": OtsuDetector()}).to_json(), encoding="utf-8")
    empty = tmp_path / "empty.json"
    empty.write_text(ImagePipeline(ops={}).to_json(), encoding="utf-8")

    assert validate_pipeline(good) == (True, None)
    assert validate_pipeline(empty) == (False, "Pipeline has no operations or measurements")
    assert validate_pipeline(tmp_path / "missing.json", skip_validation=True) == (True, None)


# --- the write tripwire -------------------------------------------------------


@pytest.fixture
def write_tripwire(monkeypatch: pytest.MonkeyPatch):
    """Make every filesystem write raise, so a preflight that writes fails loudly."""

    def refuse(*args, **kwargs):
        raise AssertionError(f"preflight attempted a write: {args!r}")

    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wax+"):
            refuse(file, mode)
        return real_open(file, mode, *args, **kwargs)

    real_os_open = os.open
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def guarded_os_open(path, flags, *args, **kwargs):
        if flags & write_flags:
            refuse(path, flags)
        return real_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    for name in ("remove", "unlink", "rename", "replace", "mkdir", "makedirs", "rmdir"):
        monkeypatch.setattr(os, name, refuse)
    monkeypatch.setattr(shutil, "rmtree", refuse)
    for name in ("mkdir", "touch", "unlink", "rmdir", "write_text", "write_bytes", "rename", "replace"):
        monkeypatch.setattr(Path, name, refuse)
    yield


def test_the_preflight_writes_nothing(tmp_path: Path, write_tripwire) -> None:
    """Every registered check runs under a tripwire that refuses any write.

    Built before the tripwire arms: the fixture's inputs are written first by
    pytest's own ``tmp_path`` machinery, and this test then only reads.
    """
    pipeline = ImagePipeline(
        ops={"det": OtsuDetector()},
        meas={"size": MeasureSize()},
        post={"tag": AppendString(column="Strain", value="_x")},
    )
    for mode in ("full", "process", "measure"):
        report = run_preflight(_context(pipeline, mode))
        assert not [f for f in report.findings if f.code == "PF-CHECK-CRASHED"], (
            report.render_lines()
        )


def test_the_tripwire_catches_a_check_that_writes(tmp_path: Path, write_tripwire) -> None:
    """Self-test: without this, an empty CHECKS makes the test above vacuous."""

    def writes(context):
        (tmp_path / "leak.txt").write_text("x")
        return []

    report = run_preflight(_context(ImagePipeline(ops={})), checks=(writes,))

    assert [f.code for f in report.findings] == ["PF-CHECK-CRASHED"]
    assert "attempted a write" in report.findings[0].message


# --- CLI wiring ---------------------------------------------------------------


@pytest.fixture
def cli_inputs(tmp_path: Path) -> tuple[Path, Path]:
    image = np.full((32, 32, 3), 20, dtype=np.uint8)
    image[6:12, 6:12] = 220
    tree = tmp_path / "images"
    (tree / "plate1").mkdir(parents=True)
    tifffile.imwrite(tree / "plate1" / "img001.tiff", image)
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text(
        ImagePipeline(ops={"det": OtsuDetector()}, meas={"size": MeasureSize()}).to_json(),
        encoding="utf-8",
    )
    return tree, pipeline


def test_a_clean_dry_run_reports_the_preflight_passed(cli_inputs, tmp_path: Path) -> None:
    tree, pipeline = cli_inputs
    result = CliRunner().invoke(
        phenotypic_cli,
        ["--pipeline", str(pipeline), "--input", str(tree),
         "--output", str(tmp_path / "out"), "--dry-run"],
    )

    assert result.exit_code == 0, result.output
    assert "Preflight checks passed" in result.output


def test_a_preflight_error_refuses_before_anything_is_written(
    cli_inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse_everything(context):
        return [PreflightFinding(code="PF-PIPELINE-LOAD", severity="error", message="planted")]

    monkeypatch.setattr(_cli_preflight, "CHECKS", (refuse_everything,))
    tree, pipeline = cli_inputs
    output_dir = tmp_path / "out"

    result = CliRunner().invoke(
        phenotypic_cli,
        ["--pipeline", str(pipeline), "--input", str(tree), "--output", str(output_dir)],
    )

    assert result.exit_code == 1, result.output
    assert "planted" in result.output
    assert "nothing under --output was changed" in result.output
    assert not output_dir.exists()


def test_skip_validation_skips_the_preflight(
    cli_inputs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse_everything(context):
        raise AssertionError("the preflight ran under --skip-validation")

    monkeypatch.setattr(_cli_preflight, "CHECKS", (refuse_everything,))
    tree, pipeline = cli_inputs

    result = CliRunner().invoke(
        phenotypic_cli,
        ["--pipeline", str(pipeline), "--input", str(tree),
         "--output", str(tmp_path / "out"), "--dry-run", "--skip-validation"],
    )

    assert result.exit_code == 0, result.output
    assert "Preflight checks passed" not in result.output
