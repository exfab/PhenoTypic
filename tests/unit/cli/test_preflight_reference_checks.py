"""PF-REF-*: the run preflight for reference metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from phenotypic import ImagePipeline
from phenotypic._cli import _cli_preflight
from phenotypic._cli._cli_preflight import check_reference_metadata
from phenotypic.detect import OtsuDetector
from phenotypic.enhance import SubtractBlank
from phenotypic.measure import MeasureSymZones
from tests.unit.cli._preflight_support import make_context, make_datasets
from tests.unit.cli.test_cli_preflight_core import write_tripwire  # noqa: F401 -- fixture


def _images(root: Path, *stems: str) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for stem in stems:
        path = root / f"{stem}.tif"
        tifffile.imwrite(path, np.zeros((4, 4), dtype=np.uint8))
        paths.append(path)
    return paths


def _context(tmp_path, rows: dict | None, *, mode="full", stems=("t01", "t02"), pipeline=None):
    root = tmp_path / "in" / "plate1"
    _images(root, "blank", *stems)
    metadata = None
    if rows is not None:
        metadata = tmp_path / "layout.csv"
        pd.DataFrame(rows).to_csv(metadata, index=False)
    datasets = make_datasets(*[root / f"{s}.tif" for s in stems], name="plate1")
    # make_context builds the ExecutionConfig itself from keyword overrides
    # (and sets measure_only / process_only_layer from `mode`).
    return make_context(
        pipeline or ImagePipeline(ops={"sb": SubtractBlank(), "d": OtsuDetector()}),
        mode,
        datasets,
        metadata_csv=metadata,
        output_dir=tmp_path / "out",
    )


def _center_detector(enhancer) -> ImagePipeline:
    """A measurer's private detector that blank-subtracts first (runs in full mode only)."""
    return ImagePipeline(ops={"sb": enhancer, "det": OtsuDetector()})


def _codes(findings):
    return {(f.code, f.severity) for f in findings}


def test_ordinary_pipeline_has_no_findings(tmp_path):
    ctx = _context(tmp_path, None, pipeline=ImagePipeline(ops={"d": OtsuDetector()}))
    assert check_reference_metadata(ctx) == []


def test_measure_mode_is_skipped(tmp_path):
    assert check_reference_metadata(_context(tmp_path, None, mode="measure")) == []


def test_no_table_is_an_error(tmp_path):
    findings = check_reference_metadata(_context(tmp_path, None))
    assert _codes(findings) == {("PF-REF-NO-TABLE", "error")}
    assert findings[0].subjects == ("sb",)


def test_the_full_mode_snapshot_serves_a_continuation(tmp_path):
    """No --metadata on a re-run: the run's own deliverables/metadata.csv is the table."""
    from phenotypic.sdk_._io_constants import metadata_csv_deliverable_path

    snapshot = metadata_csv_deliverable_path(tmp_path / "out")
    snapshot.parent.mkdir(parents=True)
    pd.DataFrame(
        {"Metadata_ImageName": ["t01", "t02"], "Metadata_BlankImage": ["blank", "blank"]}
    ).to_csv(snapshot, index=False)
    assert check_reference_metadata(_context(tmp_path, None)) == []


def test_missing_column_is_an_error(tmp_path):
    findings = check_reference_metadata(_context(tmp_path, {"Metadata_ImageName": ["t01"]}))
    assert _codes(findings) == {("PF-REF-COLUMN", "error")}
    assert "Metadata_BlankImage" in findings[0].message


def test_table_without_image_name_is_a_table_error(tmp_path):
    findings = check_reference_metadata(_context(tmp_path, {"Metadata_BlankImage": ["blank"]}))
    assert _codes(findings) == {("PF-REF-TABLE", "error")}


def test_a_clean_table_has_no_findings(tmp_path):
    rows = {"Metadata_ImageName": ["t01", "t02"], "Metadata_BlankImage": ["blank", "blank.tif"]}
    assert check_reference_metadata(_context(tmp_path, rows)) == []


def test_the_check_never_hashes_a_reference_image(tmp_path, monkeypatch):
    """The module contract is headers only (S7): the check itself, not just the planner."""
    from phenotypic._core import _reference_context

    def refuse(path):
        raise AssertionError(f"the preflight hashed {path}")

    monkeypatch.setattr(_reference_context, "reference_file_digest", refuse)
    rows = {"Metadata_ImageName": ["t01", "t02"], "Metadata_BlankImage": ["blank", "blank"]}
    assert check_reference_metadata(_context(tmp_path, rows)) == []


def test_bare_headers_are_accepted(tmp_path):
    """ImageName/BlankImage behave like their Metadata_ spellings (Review Focus 3)."""
    rows = {"ImageName": ["t01", "t02"], "BlankImage": ["blank", "blank"]}
    assert check_reference_metadata(_context(tmp_path, rows)) == []


def test_some_unmatched_is_a_warning(tmp_path):
    rows = {"Metadata_ImageName": ["t01"], "Metadata_BlankImage": ["blank"]}
    findings = check_reference_metadata(_context(tmp_path, rows))
    assert _codes(findings) == {("PF-REF-UNMATCHED", "warning")}
    assert findings[0].subjects == ("plate1/t02",)


def test_every_image_unresolved_escalates_to_error(tmp_path):
    rows = {"Metadata_ImageName": ["t01", "t02"], "Metadata_BlankImage": ["nope", "nope"]}
    assert _codes(check_reference_metadata(_context(tmp_path, rows))) == {("PF-REF-UNRESOLVED", "error")}


def test_mixed_failures_covering_every_image_escalate_together(tmp_path):
    """Half unmatched + half unresolved: every image fails, so both are errors."""
    rows = {"Metadata_ImageName": ["t01"], "Metadata_BlankImage": ["nope"]}
    assert _codes(check_reference_metadata(_context(tmp_path, rows))) == {
        ("PF-REF-UNMATCHED", "error"),
        ("PF-REF-UNRESOLVED", "error"),
    }


def test_ambiguous_is_reported_and_the_reference_plate_is_not(tmp_path):
    """t02 names itself: the reference plate is an ordinary blank now. t03 is
    clean, so the failure (t01) does not cover every image: a warning."""
    rows = {
        "Metadata_ImageName": ["t01", "t01", "t02", "t03"],
        "Metadata_BlankImage": ["blank", "other", "t02", "blank"],
    }
    findings = check_reference_metadata(_context(tmp_path, rows, stems=("t01", "t02", "t03")))
    assert _codes(findings) == {("PF-REF-AMBIGUOUS", "warning")}
    assert findings[0].subjects == ("plate1/t01",)


def test_a_run_of_only_reference_plates_has_no_findings(tmp_path):
    rows = {"Metadata_ImageName": ["t01", "t02"], "Metadata_BlankImage": ["t01", "t02.tif"]}
    assert check_reference_metadata(_context(tmp_path, rows)) == []


def test_an_empty_blank_cell_is_still_ambiguous(tmp_path):
    rows = {"Metadata_ImageName": ["t01", "t02"], "Metadata_BlankImage": ["blank", None]}
    findings = check_reference_metadata(_context(tmp_path, rows))
    assert _codes(findings) == {("PF-REF-AMBIGUOUS", "warning")}
    assert findings[0].subjects == ("plate1/t02",)


def test_there_is_no_self_reference_code():
    from typing import get_args

    assert "PF-REF-SELF" not in get_args(_cli_preflight.FindingCode)
    assert "PF-REF-SELF" not in _cli_preflight.HINTS


def test_a_file_path_blank_that_is_not_an_image_is_unresolved(tmp_path):
    """t01's path does not exist; t02's is no image. Both are PF-REF-UNRESOLVED."""
    notes = tmp_path / "notes.json"
    notes.write_text("{}", encoding="utf-8")
    rows = {
        "Metadata_ImageName": ["t01", "t02"],
        "Metadata_BlankImage": [str(tmp_path / "missing.tif"), str(notes)],
    }
    findings = check_reference_metadata(_context(tmp_path, rows))
    assert _codes(findings) == {("PF-REF-UNRESOLVED", "error")}
    assert findings[0].subjects == ("plate1/t01", "plate1/t02")


def test_a_file_path_blank_that_exists_has_no_findings(tmp_path):
    blank = tmp_path / "blanks" / "b0.tif"
    _images(blank.parent, "b0")
    rows = {"Metadata_ImageName": ["t01", "t02"], "Metadata_BlankImage": [str(blank)] * 2}
    assert check_reference_metadata(_context(tmp_path, rows)) == []


def test_process_mode_is_checked(tmp_path):
    assert _codes(check_reference_metadata(_context(tmp_path, None, mode="process"))) == {
        ("PF-REF-NO-TABLE", "error")
    }


def test_an_operation_the_mode_never_runs_is_not_checked(tmp_path):
    """A reference op nested in a measurer runs in full mode only (spec D10)."""
    nested = ImagePipeline(
        ops={"d": OtsuDetector()},
        meas={"zones": MeasureSymZones(center_detector=_center_detector(SubtractBlank()))},
    )
    assert check_reference_metadata(_context(tmp_path, None, mode="process", pipeline=nested)) == []
    assert _codes(check_reference_metadata(_context(tmp_path, None, pipeline=nested))) == {
        ("PF-REF-NO-TABLE", "error")
    }


def test_columns_only_an_out_of_scope_operation_reads_are_not_required(tmp_path):
    """The column check, like the gate, reads only the operations the mode runs."""
    pipeline = ImagePipeline(
        ops={"sb": SubtractBlank(), "d": OtsuDetector()},
        meas={"zones": MeasureSymZones(
            center_detector=_center_detector(SubtractBlank(blank_column="Metadata_OtherBlank"))
        )},
    )
    rows = {"Metadata_ImageName": ["t01", "t02"], "Metadata_BlankImage": ["blank", "blank"]}
    assert check_reference_metadata(_context(tmp_path, rows, mode="process", pipeline=pipeline)) == []
    findings = check_reference_metadata(_context(tmp_path, rows, pipeline=pipeline))
    assert _codes(findings) == {("PF-REF-COLUMN", "error")}
    assert "Metadata_OtherBlank" in findings[0].message


def test_the_check_is_registered_after_the_metadata_join():
    checks = list(_cli_preflight.CHECKS)
    assert checks.index(check_reference_metadata) == checks.index(_cli_preflight.check_metadata_join) + 1


def test_the_reference_check_writes_nothing(tmp_path, request):
    """The table and directory listings are read under a write tripwire."""
    rows = {"Metadata_ImageName": ["t01"], "Metadata_BlankImage": ["nope"]}
    ctx = _context(tmp_path, rows)
    request.getfixturevalue("write_tripwire")
    assert _codes(check_reference_metadata(ctx)) == {
        ("PF-REF-UNMATCHED", "error"),
        ("PF-REF-UNRESOLVED", "error"),
    }
