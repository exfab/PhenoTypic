"""Atomic explicit QC database rebuild tests."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import polars as pl
import pytest

from phenotypic._core._image_pipeline import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount, ReplicateAgreement
from phenotypic.gui.results_viewer._qc_tab import _rebuild
from phenotypic.gui.results_viewer._qc_tab._rebuild import (
    QcRebuildError,
    preflight_qc_rebuild,
    rebuild_qc_database,
)
from phenotypic.sdk_ import (
    BundleLayout,
    gui_launch_owner_path,
    qc_duckdb_path,
)
from phenotypic.sdk_._qc_recipe import QcRecipeEntry
from phenotypic.sdk_._qc_recipe._runner import run_qc
from phenotypic.schema import IMAGE
from tests._output_layout import (
    write_master,
    write_measurements_mirror,
    write_pipeline_json,
)

_INSTANCE_ID = "qc-SE-rebuild01"


def _seed_output(root: Path) -> BundleLayout:
    frame = pl.DataFrame(
        {
            "Metadata_Dataset": ["d1"] * 4,
            str(IMAGE.IMAGE_NAME): ["a", "a", "b", "b"],
            "Object_Label": [1, 2, 1, 2],
            "Size_Area": [10.0, 11.0, 20.0, 40.0],
        }
    )
    write_master(root, frame)
    write_measurements_mirror(root, frame)
    pipeline = ImagePipeline(name="qc-rebuild")
    pipeline.set_qc(
        [
            QcRecipeEntry(
                cls=ReplicateAgreement,
                params={
                    "on": "Size_Area",
                    "groupby": [str(IMAGE.IMAGE_NAME)],
                    "min_replicates": 2,
                },
                instance_id=_INSTANCE_ID,
                enabled=True,
            )
        ]
    )
    write_pipeline_json(root, pipeline)
    qc_duckdb_path(root).parent.mkdir(parents=True, exist_ok=True)
    return BundleLayout.detect(root)


def _seed_count_output(root: Path) -> tuple[BundleLayout, Path]:
    """Seed a file-backed count QC recipe and return its metadata path."""
    frame = pl.DataFrame(
        {
            "Metadata_Dataset": ["d1"] * 2,
            str(IMAGE.IMAGE_NAME): ["a", "a"],
            "Object_Label": [1, 2],
        }
    )
    metadata = root / "layout.csv"
    frame.select(str(IMAGE.IMAGE_NAME), "Object_Label").write_csv(metadata)
    write_master(root, frame)
    write_measurements_mirror(root, frame)
    pipeline = ImagePipeline(name="qc-count-rebuild")
    pipeline.set_qc(
        [
            QcRecipeEntry(
                cls=ExpectedVsDetectedCount,
                params={
                    "metadata": str(metadata),
                    "groupby": [str(IMAGE.IMAGE_NAME)],
                },
                instance_id="qc-Count-rebuild01",
                enabled=True,
            )
        ]
    )
    write_pipeline_json(root, pipeline)
    qc_duckdb_path(root).parent.mkdir(parents=True, exist_ok=True)
    return BundleLayout.detect(root), metadata


def test_rebuild_success_validates_catalog_and_is_idempotent(
    tmp_path: Path,
) -> None:
    layout = _seed_output(tmp_path)
    preflight = preflight_qc_rebuild(layout)
    assert preflight.ready

    first = rebuild_qc_database(
        layout,
        expected_source_fingerprint=preflight.source_fingerprint,
    )
    assert first.applied
    assert first.receipt_path.is_file()
    with duckdb.connect(str(first.target), read_only=True) as con:
        assert con.execute(
            "SELECT instance_id FROM qc_modules"
        ).fetchall() == [(_INSTANCE_ID,)]

    second = rebuild_qc_database(
        layout,
        expected_source_fingerprint=preflight.source_fingerprint,
    )
    assert not second.applied
    assert second.database_fingerprint == first.database_fingerprint


def test_rebuild_backs_up_existing_database(tmp_path: Path) -> None:
    layout = _seed_output(tmp_path)
    target = layout.qc_duckdb
    original = b"legacy database bytes"
    target.write_bytes(original)
    preflight = preflight_qc_rebuild(layout)

    result = rebuild_qc_database(
        layout,
        expected_source_fingerprint=preflight.source_fingerprint,
    )

    assert result.backup_path is not None
    assert result.backup_path.read_bytes() == original
    assert target.read_bytes() != original


def test_rebuild_rolls_back_existing_database_when_receipt_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _seed_output(tmp_path)
    target = layout.qc_duckdb
    original = b"existing generation"
    target.write_bytes(original)
    preflight = preflight_qc_rebuild(layout)
    monkeypatch.setattr(
        _rebuild,
        "atomic_write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("receipt")),
    )

    with pytest.raises(QcRebuildError, match="rolled back"):
        rebuild_qc_database(
            layout,
            expected_source_fingerprint=preflight.source_fingerprint,
        )

    assert target.read_bytes() == original
    assert not list(target.parent.glob("*.rolled-back"))
    assert not list(target.parent.glob(".*.generation"))
    assert not (target.parent / ".rebuild_backups").exists()


def test_rebuild_failure_restores_true_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _seed_output(tmp_path)
    target = layout.qc_duckdb
    preflight = preflight_qc_rebuild(layout)
    monkeypatch.setattr(
        _rebuild,
        "atomic_write_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("receipt")),
    )

    with pytest.raises(QcRebuildError, match="rolled back"):
        rebuild_qc_database(
            layout,
            expected_source_fingerprint=preflight.source_fingerprint,
        )

    assert not target.exists()
    assert not list(target.parent.glob("*.rolled-back"))
    assert not list(target.parent.glob(".*.generation"))


def test_rebuild_refuses_active_owner_and_changed_source(tmp_path: Path) -> None:
    layout = _seed_output(tmp_path)
    owner = gui_launch_owner_path(tmp_path)
    owner.parent.mkdir(parents=True, exist_ok=True)
    owner.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    blocked = preflight_qc_rebuild(layout)
    assert not blocked.ready
    assert "nonterminal owner" in " ".join(blocked.blockers)

    owner.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    ready = preflight_qc_rebuild(layout)
    layout.mirror_parquet.write_bytes(layout.mirror_parquet.read_bytes() + b"x")
    with pytest.raises(QcRebuildError, match="changed"):
        rebuild_qc_database(
            layout,
            expected_source_fingerprint=ready.source_fingerprint,
        )


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        ("{malformed", "unreadable"),
        (json.dumps({"status": "future-state"}), "missing or unknown"),
        (json.dumps({"not_status": "complete"}), "missing or unknown"),
    ],
)
def test_rebuild_preflight_fails_closed_on_malformed_owner(
    tmp_path: Path,
    payload: str,
    reason: str,
) -> None:
    layout = _seed_output(tmp_path)
    owner = gui_launch_owner_path(tmp_path)
    owner.parent.mkdir(parents=True, exist_ok=True)
    owner.write_text(payload, encoding="utf-8")

    preflight = preflight_qc_rebuild(layout)

    assert preflight.ready is False
    assert reason in " ".join(preflight.blockers)


def test_rebuild_discards_staging_when_runner_changes_source(
    tmp_path: Path,
) -> None:
    """A source change during run_qc fails the final publication CAS."""
    layout = _seed_output(tmp_path)
    preflight = preflight_qc_rebuild(layout)

    def _mutating_runner(*args, **kwargs):
        result = run_qc(*args, **kwargs)
        layout.mirror_parquet.write_bytes(
            layout.mirror_parquet.read_bytes() + b"changed"
        )
        return result

    with pytest.raises(QcRebuildError, match="changed"):
        rebuild_qc_database(
            layout,
            expected_source_fingerprint=preflight.source_fingerprint,
            runner=_mutating_runner,
        )

    assert not layout.qc_duckdb.exists()
    assert not (layout.qc_dir / ".rebuild_receipts").exists()
    assert not (layout.qc_dir / ".rebuild_backups").exists()


def test_rebuild_rolls_back_when_source_changes_at_receipt_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final receipt-boundary CAS cannot certify a stale database."""
    layout = _seed_output(tmp_path)
    original = b"prior generation"
    layout.qc_duckdb.write_bytes(original)
    preflight = preflight_qc_rebuild(layout)
    receipt_path = _rebuild._receipt_path(
        layout.qc_duckdb,
        preflight.source_fingerprint,
    )
    receipt_path.parent.mkdir(parents=True)
    original_receipt = b'{"legacy_receipt":"preserve"}\n'
    receipt_path.write_bytes(original_receipt)
    real_atomic_write_json = _rebuild.atomic_write_json

    def _write_receipt_then_mutate(path, payload, **kwargs):
        real_atomic_write_json(path, payload, **kwargs)
        layout.mirror_parquet.write_bytes(
            layout.mirror_parquet.read_bytes() + b"changed"
        )

    monkeypatch.setattr(
        _rebuild,
        "atomic_write_json",
        _write_receipt_then_mutate,
    )

    with pytest.raises(QcRebuildError, match="receipt boundary"):
        rebuild_qc_database(
            layout,
            expected_source_fingerprint=preflight.source_fingerprint,
        )

    assert layout.qc_duckdb.read_bytes() == original
    assert receipt_path.read_bytes() == original_receipt
    assert not (layout.qc_dir / ".rebuild_backups").exists()


def test_file_backed_qc_dependency_participates_in_idempotence(
    tmp_path: Path,
) -> None:
    """Changing metadata invalidates the rebuild receipt and updates QC."""
    layout, metadata = _seed_count_output(tmp_path)
    first_preflight = preflight_qc_rebuild(layout)
    first = rebuild_qc_database(
        layout,
        expected_source_fingerprint=first_preflight.source_fingerprint,
    )

    pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): ["a", "a", "a"],
            "Object_Label": [1, 2, 3],
        }
    ).write_csv(metadata)
    second_preflight = preflight_qc_rebuild(layout)
    assert second_preflight.source_fingerprint != first.source_fingerprint
    second = rebuild_qc_database(
        layout,
        expected_source_fingerprint=second_preflight.source_fingerprint,
    )

    assert second.applied
    with duckdb.connect(str(layout.qc_duckdb), read_only=True) as con:
        expected = con.execute(
            'SELECT "QC_Count_Expected" FROM "qc_count_rebuild01"'
        ).fetchone()
    assert expected == (3,)


def test_missing_file_backed_qc_dependency_blocks_rebuild(
    tmp_path: Path,
) -> None:
    """Preflight rejects a recipe whose enabled metadata source vanished."""
    layout, metadata = _seed_count_output(tmp_path)
    metadata.unlink()

    preflight = preflight_qc_rebuild(layout)

    assert not preflight.ready
    assert any(
        str(metadata) in blocker and "missing" in blocker
        for blocker in preflight.blockers
    )


def test_rebuild_preserves_legacy_root_qc_topology(tmp_path: Path) -> None:
    layout = _seed_output(tmp_path)
    canonical = layout.deliverables_base / "qc"
    canonical.rmdir()
    legacy = tmp_path / "qc"
    legacy.mkdir()
    legacy_state = legacy / "review_state.json"
    legacy_state.write_text('{"module":{"reviewed":[],"last":null}}')
    resolved = BundleLayout.detect(tmp_path)
    preflight = preflight_qc_rebuild(resolved)

    result = rebuild_qc_database(
        resolved,
        expected_source_fingerprint=preflight.source_fingerprint,
    )

    assert result.target.parent == legacy
    assert legacy_state.is_file()
    assert not canonical.exists()
