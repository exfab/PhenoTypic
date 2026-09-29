"""Metadata-join preflight and the one metadata CSV reader.

Spec ``2026-09-24-cli-preflight`` §9, §10.5 (F21, F22; review R1, R22, R38).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from phenotypic import ImagePipeline
from phenotypic._cli._cli_preflight import check_metadata_join
from phenotypic._cli._embedded_measurement_tables import prepare_image_tables
from phenotypic._cli._metadata_join import read_metadata_csv
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSize
from tests.unit.cli._preflight_support import make_context, make_datasets


def _csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _outlier_csv(tmp_path: Path) -> Path:
    """150 integer strain ids, then one alphanumeric id at row 151."""
    rows = [f"img{i:03d},{i}" for i in range(150)] + ["img150,A7"]
    return _csv(tmp_path / "meta.csv", "ImageName,Strain\n" + "\n".join(rows) + "\n")


# --- one reader (F22, §10.5) -------------------------------------------------------


def test_the_shared_reader_survives_a_late_outlier(tmp_path: Path) -> None:
    frame = read_metadata_csv(_outlier_csv(tmp_path))

    assert frame.height == 151
    assert frame.get_column("Strain").to_list()[-1] == "A7"


def test_the_worker_path_survives_a_late_outlier(tmp_path: Path) -> None:
    """At 81d19ec the worker used pl.read_csv's 100-row inference and raised."""
    measurements = pd.DataFrame(
        {"Metadata_ImageName": ["img150"], "Object_Label": [1], "Size_Area": [10.0]}
    )

    tables = prepare_image_tables(measurements, _outlier_csv(tmp_path))

    assert tables.metadata is not None


# --- the check -------------------------------------------------------------------


def _pipeline() -> ImagePipeline:
    return ImagePipeline(ops={"d": OtsuDetector()}, meas={"s": MeasureSize()})


def _context(tmp_path: Path, csv: Path, *names: str, mode: str = "full"):
    datasets = make_datasets(*(tmp_path / "images" / "plate1" / n for n in names))
    return make_context(_pipeline(), mode, datasets, metadata_csv=csv)


def _by_code(findings) -> dict[str, object]:
    return {f.code: f for f in findings}


def test_a_matching_csv_produces_nothing(tmp_path: Path) -> None:
    csv = _csv(tmp_path / "m.csv", "ImageName,Strain\nimg001,WT\nimg002,mut\n")

    assert check_metadata_join(_context(tmp_path, csv, "img001.tiff", "img002.tiff")) == []


def test_unmatched_images_warn_and_are_listed(tmp_path: Path) -> None:
    csv = _csv(tmp_path / "m.csv", "ImageName,Strain\nimg001,WT\n")

    findings = _by_code(check_metadata_join(_context(tmp_path, csv, "img001.tiff", "img002.tiff")))

    unmatched = findings["PF-META-UNMATCHED"]
    assert unmatched.severity == "warning"
    assert len(unmatched.subjects) == 1 and unmatched.subjects[0].endswith("img002.tiff")


def test_orphan_rows_warn(tmp_path: Path) -> None:
    csv = _csv(tmp_path / "m.csv", "ImageName,Strain\nimg001,WT\nimg999,ghost\n")

    findings = _by_code(check_metadata_join(_context(tmp_path, csv, "img001.tiff")))

    assert findings["PF-META-ORPHANS"].severity == "warning"


def test_no_shared_key_and_no_measurement_key_is_an_error(tmp_path: Path) -> None:
    csv = _csv(tmp_path / "m.csv", "Strain,Condition\nWT,heat\n")

    findings = _by_code(check_metadata_join(_context(tmp_path, csv, "img001.tiff")))

    assert findings["PF-META-NO-KEYS"].severity == "error"


def test_duplicate_keys_without_a_measurement_key_are_an_error(tmp_path: Path) -> None:
    csv = _csv(tmp_path / "m.csv", "ImageName,Strain\nimg001,WT\nimg001,mut\n")

    findings = _by_code(check_metadata_join(_context(tmp_path, csv, "img001.tiff")))

    assert findings["PF-META-DUP-KEYS"].severity == "error"


def test_a_per_well_plate_map_is_not_refused(tmp_path: Path) -> None:
    """Review R1: keyed on ImageName + grid position, which only measurements carry."""
    rows = "\n".join(f"img001,{r},{c},S{r}{c}" for r in range(2) for c in range(2))
    csv = _csv(tmp_path / "m.csv", f"ImageName,Grid_RowNum,Grid_ColNum,Strain\n{rows}\n")

    findings = check_metadata_join(_context(tmp_path, csv, "img001.tiff"))

    assert not [f for f in findings if f.severity == "error"], findings
    assert "PF-META-UNVERIFIED" in _by_code(findings)


def test_a_grid_layout_without_image_names_is_not_refused(tmp_path: Path) -> None:
    """Review R1: one layout applied to every plate, keyed on grid position alone."""
    rows = "\n".join(f"{r},{c},S{r}{c}" for r in range(2) for c in range(2))
    csv = _csv(tmp_path / "m.csv", f"Grid_RowNum,Grid_ColNum,Strain\n{rows}\n")

    findings = _by_code(check_metadata_join(_context(tmp_path, csv, "img001.tiff")))

    assert findings["PF-META-NO-KEYS"].severity == "warning"
    assert "PF-META-UNVERIFIED" in findings
    assert "PF-META-UNMATCHED" not in findings


def test_conflicting_aliases_are_an_error(tmp_path: Path) -> None:
    csv = _csv(tmp_path / "m.csv", "ImageName,Strain,Metadata_Strain\nimg001,WT,mut\n")

    findings = _by_code(check_metadata_join(_context(tmp_path, csv, "img001.tiff")))

    assert findings["PF-META-ALIAS"].severity == "error"


def test_an_unparseable_csv_is_an_error(tmp_path: Path) -> None:
    csv = tmp_path / "m.csv"
    csv.write_bytes(b"ImageName,Strain\n\"unterminated,WT\n")

    findings = _by_code(check_metadata_join(_context(tmp_path, csv, "img001.tiff")))

    assert findings["PF-META-PARSE"].severity == "error"


@pytest.mark.parametrize("mode", ["process", "measure"])
def test_modes_that_do_not_join_are_out_of_scope(tmp_path: Path, mode: str) -> None:
    csv = _csv(tmp_path / "m.csv", "Strain\nWT\n")

    assert check_metadata_join(_context(tmp_path, csv, "img001.tiff", mode=mode)) == []


def test_skip_validation_still_refuses_an_unreadable_csv(tmp_path: Path) -> None:
    """Review R22: the startup parse stays outside --skip-validation."""
    from click.testing import CliRunner

    from phenotypic.phenotypicCLI import phenotypic_cli

    pipeline = tmp_path / "p.json"
    pipeline.write_text(_pipeline().to_json(), encoding="utf-8")
    (tmp_path / "in").mkdir()
    csv = tmp_path / "m.csv"
    csv.write_bytes(b"ImageName,Strain\n\"unterminated,WT\n")

    result = CliRunner().invoke(
        phenotypic_cli,
        ["--pipeline", str(pipeline), "--input", str(tmp_path / "in"),
         "--output", str(tmp_path / "out"), "--metadata", str(csv), "--skip-validation"],
    )

    assert result.exit_code != 0
    assert "Cannot read metadata CSV" in result.output


# --- the join is only partly knowable (review D2, D4) ----------------------------


def test_a_csv_keyed_on_metadata_the_images_carry_is_not_refused(tmp_path: Path) -> None:
    """Review D2: PhenoTypic exports restore ``Strain``, which the join keys on."""
    from phenotypic import Image
    import numpy as np

    paths = []
    for name, strain in (("p0", "WT"), ("p1", "mut")):
        image = Image(np.full((16, 16, 3), 40, dtype=np.uint8), name=name)
        image.metadata["Strain"] = strain
        path = tmp_path / "images" / "plate1" / f"{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.rgb.imsave(path)
        paths.append(path)
    csv = _csv(tmp_path / "m.csv", "Strain,Media\nWT,YPD\nmut,SC\n")

    findings = _by_code(check_metadata_join(
        make_context(_pipeline(), "full", make_datasets(*paths), metadata_csv=csv)
    ))

    assert findings["PF-META-NO-KEYS"].severity == "warning"
    assert not [f for f in findings.values() if f.severity == "error"]


def test_a_qualified_attribute_is_not_a_measurement_key(tmp_path: Path) -> None:
    """Review D4: ``Strain_ID`` is joined as an attribute, so nothing keys the join."""
    csv = _csv(tmp_path / "m.csv", "Strain_ID,Media\nS1,YPD\n")

    findings = _by_code(check_metadata_join(_context(tmp_path, csv, "img001.tiff")))

    assert findings["PF-META-NO-KEYS"].severity == "error"
    assert "PF-META-UNVERIFIED" not in findings


def test_a_custom_operation_may_emit_a_qualified_key(tmp_path: Path) -> None:
    """Review D4: with a custom op in scope, any qualified column may be its output."""
    from phenotypic.abc_ import ObjectDetector

    class _LabDetector(ObjectDetector):
        def _operate(self, image):
            return image

    _LabDetector.__module__ = "my_lab.detectors"
    csv = _csv(tmp_path / "m.csv", "Plate_Barcode,Media\nB1,YPD\n")
    datasets = make_datasets(tmp_path / "images" / "plate1" / "img001.tiff")
    pipeline = ImagePipeline(ops={"d": _LabDetector()}, meas={"s": MeasureSize()})

    findings = _by_code(check_metadata_join(
        make_context(pipeline, "full", datasets, metadata_csv=csv)
    ))

    assert findings["PF-META-NO-KEYS"].severity == "warning"
    assert "PF-META-UNVERIFIED" in findings


def test_the_startup_parse_uses_the_shared_reader(tmp_path: Path) -> None:
    """Review D6 (M3): a long FIRST data row, which pandas reads as an index
    column but the shared reader refuses (a longer later row fails both)."""
    from click.testing import CliRunner

    from phenotypic.phenotypicCLI import phenotypic_cli

    pipeline = tmp_path / "p.json"
    pipeline.write_text(_pipeline().to_json(), encoding="utf-8")
    (tmp_path / "in").mkdir()
    csv = tmp_path / "m.csv"
    csv.write_text("ImageName,Strain\nimg001,WT,extra\nimg002,mut\n", encoding="utf-8")

    result = CliRunner().invoke(
        phenotypic_cli,
        ["--pipeline", str(pipeline), "--input", str(tmp_path / "in"),
         "--output", str(tmp_path / "out"), "--metadata", str(csv), "--skip-validation"],
    )

    assert result.exit_code != 0
    assert "Cannot read metadata CSV" in result.output


def test_process_mode_does_not_parse_the_metadata_it_ignores(tmp_path: Path) -> None:
    """Review D8: process mode never reads --metadata, so it cannot refuse it."""
    from click.testing import CliRunner

    from phenotypic.phenotypicCLI import phenotypic_cli

    pipeline = tmp_path / "p.json"
    pipeline.write_text(_pipeline().to_json(), encoding="utf-8")
    (tmp_path / "in").mkdir()
    csv = tmp_path / "m.csv"
    csv.write_bytes(b"ImageName,Strain\n\"unterminated,WT\n")

    result = CliRunner().invoke(
        phenotypic_cli,
        ["--mode", "process", "--layer", "gray", "--pipeline", str(pipeline),
         "--input", str(tmp_path / "in"), "--output", str(tmp_path / "out"),
         "--metadata", str(csv), "--dry-run"],
    )

    assert "Cannot read metadata CSV" not in result.output
