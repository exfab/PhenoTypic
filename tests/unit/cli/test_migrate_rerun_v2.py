"""#281 P3 review fixes (C16r): re-running ``--mode migrate`` over a v2 tree.

``--mode migrate`` is the documented, re-runnable recovery step, so a second
pass over a tree it already converted -- or over a tree forward modes wrote
after P3 -- must change nothing. The review (``review-p3.md``) found three ways
it did:

- **C1.** It certified the store's *latest* run folder, not the one the
  image's record certifies, so a tree measured A, B, A was silently moved to
  B (spec D6a).
- **I1.** The republish after the ``tables_v2`` step wrote
  ``provenance: "forward"`` over a migrated record and dropped its
  ``source_provenance``.
- **I2.** The marker stage rebuilt identity from the legacy
  ``image_complete/`` marker only, so a record-era tree got default identity.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic.sdk_ import list_table_runs, resolve_run_state
from phenotypic.sdk_._image_record import read_image_record
from tests._tiny_run import one_store, run_cli, tree_digest

_TINY_STEMS = ("plate_1", "plate_2")


def _records(out: Path) -> dict[str, bytes]:
    return {
        stem: (out / ".phenotypic" / "progress" / "images" / "ds" / f"{stem}.json").read_bytes()
        for stem in _TINY_STEMS
    }


def _scientific_tree(out: Path) -> dict[str, str]:
    """Digests of everything a migrate pass could rewrite for an image."""
    return {
        "results": tree_digest(out / "results"),
        "deliverables": tree_digest(out / "deliverables"),
        "images": tree_digest(out / ".phenotypic" / "progress" / "images"),
    }


# ---------------------------------------------------------------------------
# C1: a record certifying a non-latest folder keeps it
# ---------------------------------------------------------------------------


def test_migrate_over_a_tree_certifying_an_older_folder_changes_nothing(
    certified_older_than_latest,
):
    out = certified_older_than_latest
    records = _records(out)
    before = _scientific_tree(out)
    result = run_cli(None, out, mode="migrate")
    assert result.exit_code == 0, result.output
    assert _records(out) == records
    assert _scientific_tree(out) == before
    assert resolve_run_state(out, depth="deep").completion == "complete"


def test_a_second_migrate_pass_is_a_no_op_too(certified_older_than_latest):
    out = certified_older_than_latest
    assert run_cli(None, out, mode="migrate").exit_code == 0
    before = _scientific_tree(out)
    assert run_cli(None, out, mode="migrate").exit_code == 0
    assert _scientific_tree(out) == before


def test_the_certified_folder_is_the_one_migrate_keeps(certified_older_than_latest):
    out = certified_older_than_latest
    assert run_cli(None, out, mode="migrate").exit_code == 0
    for stem in _TINY_STEMS:
        first, _latest = list_table_runs(one_store(out, stem=stem))
        record = read_image_record(out, "ds", stem)
        assert record is not None
        assert out / record["artifacts"]["measurements"]["path"] == first.measurements_path()


# ---------------------------------------------------------------------------
# I1: the republish keeps provenance and source_provenance
# ---------------------------------------------------------------------------

_PROVENANCES = [
    pytest.param("forward", None, id="forward"),
    pytest.param("migrated", {"note": "verbatim bookkeeping"}, id="migrated"),
]


def _set_record_fields(out: Path, stem: str, **fields) -> dict:
    """Rewrite fields of one image's record in place and return the new record."""
    path = out / ".phenotypic" / "progress" / "images" / "ds" / f"{stem}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    for key, value in fields.items():
        if value is None:
            record.pop(key, None)
        else:
            record[key] = value
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record


_IDENTITY = ("work_id", "relative_image_path", "attempt_id", "mode")


def _assert_identity_kept(before: dict, after: dict, *, epoch: bool) -> None:
    """The record's identity survived a republish.

    ``lifecycle_epoch`` is checked only where no lifecycle fence is live. A
    migrate republish stamps the live migration generation instead, by design:
    under SLURM ``publish_image_success`` refuses any other epoch.
    """
    for field in _IDENTITY + (("lifecycle_epoch",) if epoch else ()):
        assert after[field] == before[field], field
    assert after.get("provenance", "forward") == before.get("provenance", "forward")
    assert after.get("source_provenance") == before.get("source_provenance")


@pytest.mark.parametrize(("provenance", "source"), _PROVENANCES)
def test_the_tables_v2_republish_keeps_provenance(v1_tree, provenance, source):
    before = _set_record_fields(
        v1_tree, "plate", provenance=provenance, source_provenance=source
    )
    result = run_cli(None, v1_tree, mode="migrate")
    assert result.exit_code == 0, result.output
    after = read_image_record(v1_tree, "ds", "plate")
    assert after is not None
    _assert_identity_kept(before, after, epoch=False)
    assert resolve_run_state(v1_tree, depth="deep").completion == "complete"


def test_a_migrated_record_with_its_own_work_id_stays_valid(v1_tree):
    """U-10: a migrated record's work_id need not match the configured one."""
    before = _set_record_fields(
        v1_tree,
        "plate",
        provenance="migrated",
        source_provenance={"note": "verbatim bookkeeping"},
        work_id="0" * 64,
    )
    result = run_cli(None, v1_tree, mode="migrate")
    assert result.exit_code == 0, result.output
    after = read_image_record(v1_tree, "ds", "plate")
    assert after is not None
    _assert_identity_kept(before, after, epoch=False)
    assert resolve_run_state(v1_tree, depth="deep").completion == "complete"


@pytest.mark.parametrize(("provenance", "source"), _PROVENANCES)
def test_the_measure_republish_keeps_provenance(
    finished_tiny_run, pipeline_file_b, provenance, source
):
    out = finished_tiny_run
    before = {
        stem: _set_record_fields(
            out, stem, provenance=provenance, source_provenance=source
        )
        for stem in _TINY_STEMS
    }
    result = run_cli(None, out, mode="measure", pipeline=pipeline_file_b)
    assert result.exit_code == 0, result.output
    for stem in _TINY_STEMS:
        after = read_image_record(out, "ds", stem)
        assert after is not None
        _assert_identity_kept(before[stem], after, epoch=True)
