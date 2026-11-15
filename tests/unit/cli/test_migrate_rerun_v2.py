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

from pathlib import Path


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
