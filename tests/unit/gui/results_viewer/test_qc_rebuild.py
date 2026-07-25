"""Atomic explicit QC database rebuild tests."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import polars as pl
import pytest

from phenotypic._core._image_pipeline import ImagePipeline
from phenotypic.analysis import ReplicateAgreement
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
from phenotypic.schema import METADATA
from tests._output_layout import (
    write_master,
    write_measurements_mirror,
    write_pipeline_json,
)

_INSTANCE_ID = "qc-SE-rebuild01"


def _seed_output(root: Path) -> BundleLayout:
    frame = pl.DataFrame(
        {
            "MetadataExperiment_Dataset": ["d1"] * 4,
            str(METADATA.IMAGE_NAME): ["a", "a", "b", "b"],
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
                    "groupby": [str(METADATA.IMAGE_NAME)],
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
