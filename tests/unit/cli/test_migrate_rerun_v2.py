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


def _deliverables_digest(out: Path) -> dict[str, bytes]:
    """Every deliverable's bytes, except ``rembi.yaml``.

    ``rembi.yaml`` lists the images in an order that varies between two
    finalizations of the same mirror (a pre-existing nondeterminism, reported
    separately), so a byte comparison of it says nothing about migrate.
    """
    root = out / "deliverables"
    return {
        file.relative_to(root).as_posix(): file.read_bytes()
        for file in sorted(root.rglob("*"))
        if file.is_file() and file.name != "rembi.yaml"
    }


def _scientific_tree(out: Path) -> dict[str, object]:
    """Digests of everything a migrate pass could rewrite for an image."""
    return {
        "results": tree_digest(out / "results"),
        "deliverables": _deliverables_digest(out),
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


# ---------------------------------------------------------------------------
# I2: the marker stage reads the record before the legacy marker
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("provenance", "source"), _PROVENANCES)
def test_a_crash_before_the_tables_v2_republish_keeps_identity(
    v1_tree, provenance, source
):
    """Convert, die before the republish, re-run migrate.

    The re-run finds a v2 store and a record naming the flat path that no
    longer exists, so the record no longer verifies and the marker stage
    republishes it. It must republish the record's identity, not defaults.
    """
    from phenotypic.sdk_ import convert_tables_v1_to_v2

    before = _set_record_fields(
        v1_tree, "plate", provenance=provenance, source_provenance=source
    )
    store = one_store(v1_tree)
    convert_tables_v1_to_v2(store, durable=False)  # the process died here
    assert resolve_run_state(v1_tree, depth="deep").completion == "incomplete"

    result = run_cli(None, v1_tree, mode="migrate")
    assert result.exit_code == 0, result.output
    after = read_image_record(v1_tree, "ds", "plate")
    assert after is not None
    _assert_identity_kept(before, after, epoch=False)
    (run,) = list_table_runs(store)
    assert v1_tree / after["artifacts"]["measurements"]["path"] == run.measurements_path()
    assert resolve_run_state(v1_tree, depth="deep").completion == "complete"


def test_the_crash_path_and_the_clean_path_publish_the_same_identity(tmp_path, v1_tree):
    """Everything but the timestamp and the migration's own generation agrees."""
    import shutil

    from phenotypic.sdk_ import convert_tables_v1_to_v2

    clean = tmp_path / "clean"
    shutil.copytree(v1_tree, clean, symlinks=True)
    assert run_cli(None, clean, mode="migrate").exit_code == 0
    convert_tables_v1_to_v2(one_store(v1_tree), durable=False)
    assert run_cli(None, v1_tree, mode="migrate").exit_code == 0

    def _identity(out: Path) -> dict:
        record = read_image_record(out, "ds", "plate")
        assert record is not None
        return {
            key: record.get(key)
            for key in _IDENTITY + ("provenance", "source_provenance")
        } | {"measurements": record["artifacts"]["measurements"]["path"]}

    assert _identity(v1_tree) == _identity(clean)


def test_a_record_for_another_work_id_is_not_borrowed(tmp_path):
    """A forward record written for another work_id gives default identity."""
    from phenotypic._cli._cli_image_record import publish_image_record
    from phenotypic._cli._cli_migrate_image import _existing_marker_identity

    out = tmp_path / "out"
    artifact = out / "artifact.bin"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"x")
    publish_image_record(
        out, work_id="a" * 64, dataset="ds", image_stem="img",
        relative_image_path="ds/img.tif", mode="full", stages={},
        artifacts={"x": artifact}, attempt_id="attempt", lifecycle_epoch="epoch",
    )
    identity = _existing_marker_identity(out, "ds", "img", "b" * 64)
    assert identity["work_id"] == "b" * 64
    assert identity["attempt_id"] == "migration"
    kept = _existing_marker_identity(out, "ds", "img", "a" * 64)
    assert kept["attempt_id"] == "attempt" and kept["relative_image_path"] == "ds/img.tif"


# ---------------------------------------------------------------------------
# OQ3: readers resolve a `legacy` last application to D10's folder
# ---------------------------------------------------------------------------

#: The digest the committed v1 fixture's journal records.
_V1_FIXTURE_SHA = "ae2757956880293d6534d21cebb9358ca7a309b95e92e704ac174e619750747f"


def _make_journal_legacy(store: Path) -> None:
    """Relabel the store's one journal application ``legacy``, digest kept.

    The shape ``--mode migrate`` leaves an HDF-era store in: pass 5 converts
    its v1 journal into one ``legacy`` application that still names the
    pipeline's digest.
    """
    root_path = store / "zarr.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    applications = root["attributes"]["phenotypic"]["provenance"]["applications"]
    applications[-1]["kind"] = "legacy"
    root_path.write_text(json.dumps(root), encoding="utf-8")


def test_a_legacy_journal_resolves_to_the_folder_migrate_named(v1_store):
    from phenotypic._core._provenance import journal_pipeline_sha256
    from phenotypic.sdk_ import (
        convert_tables_v1_to_v2,
        journal_measurement_table,
        journal_table_run,
        ngff_,
    )

    _make_journal_legacy(v1_store)
    run = convert_tables_v1_to_v2(v1_store, durable=False)
    assert run is not None and run.run_id == f"1-{_V1_FIXTURE_SHA[:12]}"
    resolved = journal_table_run(v1_store)
    assert resolved is not None and resolved.run_id == run.run_id
    assert journal_measurement_table(v1_store) == run.measurements_path()
    # D2's writer rule is unchanged: a save over this journal knows no digest.
    journal = ngff_.read_phenotypic_attributes(v1_store)["provenance"]
    assert journal_pipeline_sha256(journal) is None


def test_after_migrate_a_staged_continuation_reuses_the_migrated_image(v1_tree):
    """The staged resume classifier reads the migrated table as the image's own.

    With Stage-3 markers off, the embedded table is the whole completeness
    evidence; before OQ3 the probe asked for ``1-unknown`` and sent every
    migrated image back through Stage 2.
    """
    from phenotypic._cli._cli_staged_resume import classify_staged_image
    from phenotypic.sdk_._run_state import staged_image_is_complete

    _make_journal_legacy(one_store(v1_tree))
    assert run_cli(None, v1_tree, mode="migrate").exit_code == 0
    assert staged_image_is_complete(
        v1_tree, "ds", "plate", markers_required=False, resume=True
    )
    stage = classify_staged_image(
        output_dir=v1_tree,
        dataset="ds",
        image=Path("plate.tif"),
        slot="slot",
        input_root=v1_tree,
        process_only_layer=None,
        markers_required=False,
    )
    assert stage == "complete"
