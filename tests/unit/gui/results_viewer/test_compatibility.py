"""Pure preflight and explicit Results recipe migration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from phenotypic.gui.results_viewer._compatibility import (
    CompatibilityMigrationError,
    migrate_output_recipe,
    preflight_output_compatibility,
)
from phenotypic.sdk_ import file_fingerprint
from phenotypic.sdk_._qc_recipe import QcRecipe
from phenotypic.schema import METADATA

_FIXTURES = Path(__file__).parents[3] / "fixtures" / "output_compatibility"


def _historical_pipeline(tmp_path: Path) -> Path:
    """Write the exact historical GridOccupancy shape and current metadata."""
    metadata = tmp_path / "layout.csv"
    pd.DataFrame({
        str(METADATA.IMAGE_NAME): ["plate-a.png", "plate-a.png"],
        "Object_Label": [1, 2],
    }).to_csv(metadata, index=False)
    raw = (
        _FIXTURES / "grid_occupancy_metadata_source_v1.json"
    ).read_text(encoding="utf-8")
    pipeline = tmp_path / "pipeline.json.pht-pipe"
    pipeline.write_text(
        raw.replace("__METADATA_PATH__", str(metadata)),
        encoding="utf-8",
    )
    return pipeline


def test_preflight_maps_exact_historical_grid_shape_without_writes(
    tmp_path: Path,
) -> None:
    pipeline = _historical_pipeline(tmp_path)
    before = pipeline.read_bytes()

    report = preflight_output_compatibility(pipeline)

    assert report.status == "migratable"
    assert pipeline.read_bytes() == before
    assert report.migrated_pipeline_payload is not None
    entry = report.migrated_pipeline_payload["qc"][0]  # type: ignore[index]
    assert entry["historical_note"] == {"preserve": "verbatim"}  # type: ignore[index]
    params = entry["params"]  # type: ignore[index]
    assert params["metadata"].endswith("layout.csv")  # type: ignore[index]
    assert "metadata_source" not in params
    assert "cell_label" not in params
    assert params["groupby"] == [str(METADATA.IMAGE_NAME)]  # type: ignore[index]
    assert [issue.code for issue in report.issues] == [
        "qc.grid.metadata_source",
        "qc.grid.cell_label_null",
        "qc.grid.groupby_alias",
    ]


def test_preflight_blocks_ambiguous_metadata_without_guessing(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    pd.DataFrame({str(METADATA.IMAGE_NAME): ["a"]}).to_csv(first, index=False)
    pd.DataFrame({str(METADATA.IMAGE_NAME): ["b"]}).to_csv(second, index=False)
    pipeline = tmp_path / "pipeline.json.pht-pipe"
    payload = json.loads(_historical_pipeline(tmp_path).read_text(encoding="utf-8"))
    params = payload["qc"][0]["params"]
    params["metadata"] = str(first)
    params["metadata_source"] = str(second)
    pipeline.write_text(json.dumps(payload), encoding="utf-8")

    report = preflight_output_compatibility(pipeline)

    assert report.status == "blocked"
    assert report.migrated_pipeline_payload is None
    assert [issue.code for issue in report.issues] == [
        "qc.grid.metadata.ambiguous"
    ]


def test_preflight_reports_equal_retired_metadata_source_once(
    tmp_path: Path,
) -> None:
    """A redundant retired field is removed with one deterministic issue."""
    pipeline = _historical_pipeline(tmp_path)
    payload = json.loads(pipeline.read_text(encoding="utf-8"))
    params = payload["qc"][0]["params"]
    params["metadata"] = params["metadata_source"]
    pipeline.write_text(json.dumps(payload), encoding="utf-8")

    report = preflight_output_compatibility(pipeline)

    assert report.status == "migratable"
    assert sum(
        issue.code == "qc.grid.metadata_source" for issue in report.issues
    ) == 1


def test_preflight_blocks_missing_historical_metadata(
    tmp_path: Path,
) -> None:
    """A retired metadata path is never accepted without reading its target."""
    pipeline = _historical_pipeline(tmp_path)
    payload = json.loads(pipeline.read_text(encoding="utf-8"))
    payload["qc"][0]["params"]["metadata_source"] = str(
        tmp_path / "missing.csv"
    )
    pipeline.write_text(json.dumps(payload), encoding="utf-8")

    report = preflight_output_compatibility(pipeline)

    assert report.status == "blocked"
    assert [issue.code for issue in report.issues][-1] == (
        "qc.grid.metadata.unavailable"
    )


def test_preflight_blocks_unmapped_historical_groupby(
    tmp_path: Path,
) -> None:
    """Only the observed and tested metadata column alias is migrated."""
    pipeline = _historical_pipeline(tmp_path)
    payload = json.loads(pipeline.read_text(encoding="utf-8"))
    payload["qc"][0]["params"]["groupby"] = ["Metadata_ImageFile"]
    pipeline.write_text(json.dumps(payload), encoding="utf-8")

    report = preflight_output_compatibility(pipeline)

    assert report.status == "blocked"
    assert [issue.code for issue in report.issues][-1] == (
        "qc.grid_migration.invalid"
    )


def test_unknown_entry_is_blocked_and_raw_file_is_preserved(
    tmp_path: Path,
) -> None:
    pipeline = tmp_path / "pipeline.json.pht-pipe"
    fixture = _FIXTURES / "unknown_qc_entry.json"
    pipeline.write_bytes(fixture.read_bytes())
    before = pipeline.read_bytes()
    recipe = QcRecipe._load_from_paths(pipeline, pipeline)
    warnings_before = list(recipe.load_warnings)

    report = preflight_output_compatibility(pipeline)

    assert report.status == "blocked"
    assert report.migrated_pipeline_payload is None
    assert pipeline.read_bytes() == before
    assert recipe.load_warnings == warnings_before
    assert len(report.issues) == 1


def test_explicit_migration_is_backed_up_atomic_receipted_and_idempotent(
    tmp_path: Path,
) -> None:
    pipeline = _historical_pipeline(tmp_path)
    original = pipeline.read_bytes()
    report = preflight_output_compatibility(pipeline)

    result = migrate_output_recipe(
        pipeline,
        expected_source_fingerprint=report.source_fingerprint,
    )

    assert result.applied is True
    assert result.backup_path is not None
    assert result.backup_path.read_bytes() == original
    assert result.receipt_path is not None
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["old_fingerprint"] == report.source_fingerprint
    assert receipt["new_fingerprint"] == file_fingerprint(pipeline)
    assert receipt["backup_path"] == str(result.backup_path.resolve())
    assert preflight_output_compatibility(pipeline).status == "compatible"

    artifacts_before = sorted(result.backup_path.parent.iterdir())
    current = preflight_output_compatibility(pipeline)
    second = migrate_output_recipe(
        pipeline,
        expected_source_fingerprint=current.source_fingerprint,
    )
    assert second.applied is False
    assert sorted(result.backup_path.parent.iterdir()) == artifacts_before


def test_migration_refuses_changed_source_without_writing_backup(
    tmp_path: Path,
) -> None:
    pipeline = _historical_pipeline(tmp_path)
    report = preflight_output_compatibility(pipeline)
    pipeline.write_text(
        pipeline.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CompatibilityMigrationError, match="changed"):
        migrate_output_recipe(
            pipeline,
            expected_source_fingerprint=report.source_fingerprint,
        )

    assert not (pipeline.parent / ".migration_backups").exists()
