"""Unit tests for the pipeline-backed :class:`phenotypic.tools_._qc_recipe._recipe.QcRecipe`.

Covers the Phase B recipe adapter contract:

* **scoped** read-modify-write: a mutator rewrites only the ``qc`` array of
  ``pipeline.json`` and preserves ``operations``/``post``/``model``;
* **mtime refusal**: a CLI write landing between the QC tab's read and write
  is refused (the in-memory edit is rolled back) until ``reload``;
* **idempotent + atomic** sidecar migration: a legacy
  ``.viewer_cache/qc_recipe.json`` is folded into ``pipeline.json`` exactly
  once, and never overwrites an existing non-empty ``qc`` array.
"""

from __future__ import annotations

import json
from pathlib import Path


from phenotypic import ImagePipeline
from phenotypic.analysis import ReplicateAgreement
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureShape
from phenotypic.tools_._qc_recipe import QcRecipe, QcRecipeEntry
from phenotypic.tools_ import pipeline_json_path


def _seed_pipeline_json(output_dir: Path) -> Path:
    """Write a pipeline.json with ops + meas (no qc) and return its path.

    ``pipeline.json`` is a user-facing deliverable, so it lives under
    ``<output_dir>/deliverables/`` (via :func:`pipeline_json_path`), which is
    exactly where ``QcRecipe.load`` reads it from.
    """
    pipe = ImagePipeline(ops=[OtsuDetector()], meas=[MeasureShape()])
    path = pipeline_json_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pipe.to_json() or "")
    return path


def _seed_legacy_pipeline_json(output_dir: Path) -> Path:
    """Write a legacy deliverables/pipeline.json and return its path."""
    pipe = ImagePipeline(ops=[OtsuDetector()], meas=[MeasureShape()])
    path = output_dir / "deliverables" / "pipeline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pipe.to_json() or "")
    return path


def _se_params() -> dict:
    return {"on": "Size_Area", "groupby": ["Metadata_ImageFile"]}


class TestScopedReadModifyWrite:
    """Mutators touch only the ``qc`` array."""

    def test_add_preserves_operations_and_meas(self, tmp_path: Path) -> None:
        pj = _seed_pipeline_json(tmp_path)
        before = json.loads(pj.read_text())

        recipe = QcRecipe.load(tmp_path)
        iid = recipe.add(ReplicateAgreement, _se_params())
        assert iid is not None

        after = json.loads(pj.read_text())
        # Untouched sections survive byte-for-byte.
        assert after["pipe_cfgs"] == before["pipe_cfgs"]
        assert after["meas"] == before["meas"]
        # qc array gained the entry.
        assert [e["instance_id"] for e in after["qc"]] == [iid]

    def test_remove_then_update_round_trip(self, tmp_path: Path) -> None:
        _seed_pipeline_json(tmp_path)
        recipe = QcRecipe.load(tmp_path)
        iid = recipe.add(ReplicateAgreement, _se_params())

        assert recipe.update(iid, enabled=False) is True
        reloaded = QcRecipe.load(tmp_path)
        assert reloaded.entries[0].enabled is False

        assert recipe.remove(iid) is True
        assert QcRecipe.load(tmp_path).entries == []

    def test_add_when_pipeline_json_absent_creates_minimal_doc(
        self, tmp_path: Path
    ) -> None:
        recipe = QcRecipe.load(tmp_path)  # no pipeline.json yet
        iid = recipe.add(ReplicateAgreement, _se_params())

        doc = json.loads(pipeline_json_path(tmp_path).read_text())
        assert [e["instance_id"] for e in doc["qc"]] == [iid]

    def test_add_from_legacy_pipeline_preserves_pipeline_body(
        self, tmp_path: Path
    ) -> None:
        legacy = _seed_legacy_pipeline_json(tmp_path)
        before = json.loads(legacy.read_text())

        recipe = QcRecipe.load(tmp_path)
        iid = recipe.add(ReplicateAgreement, _se_params())

        typed = pipeline_json_path(tmp_path)
        assert iid is not None
        assert typed.exists()
        after = json.loads(typed.read_text())
        assert after["pipe_cfgs"] == before["pipe_cfgs"]
        assert after["meas"] == before["meas"]
        assert [e["instance_id"] for e in after["qc"]] == [iid]


class TestMtimeRefusal:
    """A concurrent CLI write between read and write is refused."""

    def test_is_stale_after_external_write(self, tmp_path: Path) -> None:
        pj = _seed_pipeline_json(tmp_path)
        recipe = QcRecipe.load(tmp_path)
        assert recipe.is_stale() is False

        # Simulate a CLI --recompile rewriting pipeline.json mid-session.
        import os
        import time

        bumped = json.loads(pj.read_text())
        bumped["desc"] = "rewritten by CLI"
        pj.write_text(json.dumps(bumped))
        # Force a distinct mtime even on coarse clocks.
        future = time.time_ns() + 1_000_000_000
        os.utime(pj, ns=(future, future))

        assert recipe.is_stale() is True

    def test_mutation_refused_when_stale_and_rolled_back(
        self, tmp_path: Path
    ) -> None:
        pj = _seed_pipeline_json(tmp_path)
        recipe = QcRecipe.load(tmp_path)

        import os
        import time

        bumped = json.loads(pj.read_text())
        bumped["desc"] = "rewritten by CLI"
        pj.write_text(json.dumps(bumped))
        future = time.time_ns() + 1_000_000_000
        os.utime(pj, ns=(future, future))

        result = recipe.add(ReplicateAgreement, _se_params())

        assert result is None  # write refused
        assert recipe.entries == []  # in-memory append rolled back
        # On-disk file was NOT clobbered with a qc array.
        assert "qc" not in json.loads(pj.read_text())

    def test_reload_clears_staleness(self, tmp_path: Path) -> None:
        pj = _seed_pipeline_json(tmp_path)
        recipe = QcRecipe.load(tmp_path)

        import os
        import time

        pj.write_text(pj.read_text())
        future = time.time_ns() + 1_000_000_000
        os.utime(pj, ns=(future, future))
        assert recipe.is_stale() is True

        recipe.reload()
        assert recipe.is_stale() is False
        # After reload the mutation succeeds.
        assert recipe.add(ReplicateAgreement, _se_params()) is not None


class TestSidecarMigration:
    """Legacy ``.viewer_cache/qc_recipe.json`` folds in once, atomically."""

    def _write_sidecar(self, output_dir: Path, instance_id: str) -> Path:
        cache = output_dir / ".viewer_cache"
        cache.mkdir(parents=True, exist_ok=True)
        sidecar = cache / "qc_recipe.json"
        sidecar.write_text(json.dumps({
            "version": 1,
            "checks": [
                {
                    "instance_id": instance_id,
                    "class": "ReplicateAgreement",
                    "enabled": True,
                    "params": _se_params(),
                }
            ],
        }))
        return sidecar

    def test_migration_folds_entries_into_pipeline(
        self, tmp_path: Path
    ) -> None:
        _seed_pipeline_json(tmp_path)
        sidecar = self._write_sidecar(tmp_path, "qc-SE-legacy01")

        migrated = QcRecipe.migrate_from_sidecar(tmp_path)

        assert migrated is True
        doc = json.loads(pipeline_json_path(tmp_path).read_text())
        assert [e["instance_id"] for e in doc["qc"]] == ["qc-SE-legacy01"]
        # Sidecar retired so it is not folded again.
        assert not sidecar.exists()
        assert sidecar.with_suffix(".json.migrated").exists()

    def test_migration_is_idempotent(self, tmp_path: Path) -> None:
        _seed_pipeline_json(tmp_path)
        self._write_sidecar(tmp_path, "qc-SE-legacy01")

        assert QcRecipe.migrate_from_sidecar(tmp_path) is True
        # Second call: sidecar already retired -> no-op.
        assert QcRecipe.migrate_from_sidecar(tmp_path) is False
        doc = json.loads(pipeline_json_path(tmp_path).read_text())
        assert len(doc["qc"]) == 1

    def test_migration_never_overwrites_existing_qc(
        self, tmp_path: Path
    ) -> None:
        # pipeline.json already has a qc entry -> migration must not clobber it.
        pipe = ImagePipeline(
            ops=[OtsuDetector()],
            qc=[QcRecipeEntry(
                cls=ReplicateAgreement,
                params=_se_params(),
                instance_id="qc-SE-pipeline",
                enabled=True,
            )],
        )
        pj = pipeline_json_path(tmp_path)
        pj.parent.mkdir(parents=True, exist_ok=True)
        pj.write_text(pipe.to_json() or "")
        sidecar = self._write_sidecar(tmp_path, "qc-SE-legacy01")

        assert QcRecipe.migrate_from_sidecar(tmp_path) is False
        doc = json.loads(pipeline_json_path(tmp_path).read_text())
        # Pipeline's own qc entry wins; legacy entry is NOT merged in.
        assert [e["instance_id"] for e in doc["qc"]] == ["qc-SE-pipeline"]
        # The stale sidecar is retired so it stops being re-checked.
        assert not sidecar.exists()

    def test_no_sidecar_is_noop(self, tmp_path: Path) -> None:
        _seed_pipeline_json(tmp_path)
        assert QcRecipe.migrate_from_sidecar(tmp_path) is False
