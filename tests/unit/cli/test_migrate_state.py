"""P7 Task 2: the two legacy per-image trees become one record per image.

Spec §11.1. The load-bearing assertions here are the ones about what migrate
does **not** do -- leave the legacy trees standing until every record is
written, copy artifact descriptors rather than re-deriving them, and never
touch the Stage-2 token. Migrate "rewrites machine state across the whole tree
and, unlike the rest of the change, cannot be rolled back by reverting code"
(§15.1), so a wrong conversion here is wrong on a user's real run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic._cli._cli_migrate_state import (
    LEGACY_MARKER_SEGMENTS,
    apply_per_image_records,
    convert_per_image_markers,
    plan_per_image_records,
)
from phenotypic.sdk_ import (
    DIR_STAGE2_DONE,
    atomic_write_json,
    image_completion_marker_path,
    image_record_path,
    progress_dir,
)
from phenotypic.sdk_._image_record import PROVENANCE_MIGRATED

_STAGE3_SEGMENT = "stage3_complete"


def _legacy_payload(stem: str, *, artifacts: dict | None = None) -> dict:
    """The marker shape the pre-collapse writer produced.

    Mirrors ``tests/unit/cli/test_schema_gate.py::_legacy_marker_payload``,
    which is the shipped gate's own idea of a legacy marker -- so this suite
    and the detection suite cannot disagree about what migrate is converting.
    """
    return {
        "version": 2,
        "work_id": f"w-{stem}",
        "dataset": "plate",
        "relative_image_path": f"plate/{stem}.png",
        "image_stem": stem,
        "mode": "full",
        "attempt_id": "attempt",
        "lifecycle_epoch": "gen",
        "artifacts": artifacts if artifacts is not None else {},
        "completed_at": "2026-09-03T00:00:00.000+00:00",
    }


def _legacy_marker_path(root: Path, dataset: str, stem: str) -> Path:
    return image_completion_marker_path(root, dataset, stem)


def _stage3_marker_path(root: Path, dataset: str, stem: str) -> Path:
    return progress_dir(root) / _STAGE3_SEGMENT / dataset / f"{stem}.json"


def _plant_legacy_markers(
    root: Path,
    *,
    dataset: str,
    stem: str,
    image_complete: bool = False,
    stage3_complete: bool = False,
    artifacts: dict | None = None,
) -> None:
    """Plant either or both legacy markers for one image."""
    if image_complete:
        atomic_write_json(
            _legacy_marker_path(root, dataset, stem),
            _legacy_payload(stem, artifacts=artifacts),
        )
    if stage3_complete:
        atomic_write_json(
            _stage3_marker_path(root, dataset, stem),
            _legacy_payload(stem, artifacts=artifacts),
        )


def _plant_stage2_token(root: Path, dataset: str, stem: str) -> Path:
    token = progress_dir(root) / DIR_STAGE2_DONE / dataset / f"{stem}.json"
    atomic_write_json(token, {"completed_at": "2026-09-03T00:00:00.000+00:00"})
    return token


def _record(root: Path, dataset: str, stem: str) -> dict:
    return json.loads(
        image_record_path(root, dataset, stem).read_text(encoding="utf-8")
    )


# ---------------------------------------------------------------------------
# The conversion table's two rows, and nothing else
# ---------------------------------------------------------------------------


def test_two_markers_become_one_record(tmp_path: Path) -> None:
    """Rows 1 and 2. ``image_complete/`` -> ``measured``, ``stage3_complete/``
    -> ``stage3``, and no row producing ``stage2`` from a marker tree."""
    _plant_legacy_markers(
        tmp_path, dataset="plate", stem="a",
        image_complete=True, stage3_complete=True,
    )
    assert convert_per_image_markers(tmp_path) == 1

    record = _record(tmp_path, "plate", "a")
    assert set(record["stages"]) == {"stage3", "measured"}
    assert record["work_id"] == "w-a"
    assert record["provenance"] == PROVENANCE_MIGRATED


def test_a_stage3_marker_with_no_image_complete_still_converts(
    tmp_path: Path,
) -> None:
    """Stage 3 finished and the run died before publishing completion.

    A real interrupted state, and the one an ``image_complete/``-only walk
    drops on the floor. Only ``stages.stage3`` is reachable from a
    ``stage3_complete/`` marker -- an earlier draft of this test planted that
    marker, said "Stage 2 finished", and asserted ``{"stage2"}``.
    """
    _plant_legacy_markers(
        tmp_path, dataset="plate", stem="a",
        image_complete=False, stage3_complete=True,
    )
    convert_per_image_markers(tmp_path)
    assert set(_record(tmp_path, "plate", "a")["stages"]) == {"stage3"}


def test_the_segments_match_the_schema_gate(tmp_path: Path) -> None:
    """Detection and conversion must agree on what a legacy tree is.

    **Fires when** either side gains or loses a segment. The gate classifying
    a tree ``CONVERT`` that the converter then ignores is a tree stranded
    behind a refusal in every writing mode (INV-DISCHARGEABLE); the reverse is
    a converter renaming a tree nothing said was legacy.
    """
    from phenotypic.sdk_ import _schema_shape

    source = Path(_schema_shape.__file__).read_text(encoding="utf-8")
    assert "for segment in (DIR_IMAGE_COMPLETE, _DIR_STAGE3_COMPLETE):" in source
    assert tuple(seg for seg, _ in LEGACY_MARKER_SEGMENTS) == (
        "image_complete",
        "stage3_complete",
    )


# ---------------------------------------------------------------------------
# stage2_done/ -- read and leave
# ---------------------------------------------------------------------------


def test_the_stage2_token_is_read_into_the_record(tmp_path: Path) -> None:
    """The interrupted state the struck row cared about, preserved by reading.

    An image whose Stage 2 finished gets a ``stages.stage2`` entry, because
    that is a real fact about the image -- it is only the *renaming* of the
    token that was wrong.
    """
    _plant_legacy_markers(
        tmp_path, dataset="plate", stem="a", stage3_complete=True
    )
    _plant_stage2_token(tmp_path, "plate", "a")
    convert_per_image_markers(tmp_path)
    assert set(_record(tmp_path, "plate", "a")["stages"]) == {"stage3", "stage2"}


def test_the_stage2_token_survives_the_conversion_byte_for_byte(
    tmp_path: Path,
) -> None:
    """⛔ The destructive finding, pinned.

    ``stage2_done/`` holds a **consumable token** that Stage 3 ``unlink``s
    after replaying the raw array. Renaming it aside -- into the tree where
    rollback guarantees nothing reads it -- orphans every un-consumed Stage-2
    result for a staged run live across the migrate, silently, while migrate
    reports success.

    **Fires when** the token is renamed, moved, unlinked or rewritten. It is
    the only test here whose failure is data loss on a user's run rather than
    a wrong record.
    """
    _plant_legacy_markers(
        tmp_path, dataset="plate", stem="a", stage3_complete=True
    )
    token = _plant_stage2_token(tmp_path, "plate", "a")
    before = token.read_bytes()

    convert_per_image_markers(tmp_path)

    assert token.is_file(), "the Stage-2 token was moved or unlinked"
    assert token.read_bytes() == before, "the Stage-2 token was rewritten"
    assert not list(
        (progress_dir(tmp_path)).glob("legacy-v2/**/stage2_done/**")
    ), "the Stage-2 tree was renamed aside"


def test_planning_writes_nothing_at_all(tmp_path: Path) -> None:
    """The dry-run seam. ``--dry-run`` renders a plan by calling this alone.

    **Fires when** planning gains a write -- which would make the dry run a
    real conversion, on a tree the operator was still deciding about.
    """
    _plant_legacy_markers(
        tmp_path, dataset="plate", stem="a", image_complete=True
    )
    _plant_stage2_token(tmp_path, "plate", "a")
    before = {
        p.relative_to(tmp_path): p.read_bytes()
        for p in sorted(tmp_path.rglob("*"))
        if p.is_file()
    }

    planned = plan_per_image_records(tmp_path)

    assert len(planned) == 1
    assert {
        p.relative_to(tmp_path): p.read_bytes()
        for p in sorted(tmp_path.rglob("*"))
        if p.is_file()
    } == before


# ---------------------------------------------------------------------------
# The two load-bearing assertions (Step 5)
# ---------------------------------------------------------------------------


def test_artifact_descriptors_survive_conversion_byte_for_byte(
    tmp_path: Path,
) -> None:
    """The descriptors are the content proof.

    Re-deriving them during migration would certify whatever is on disk now,
    including a corrupted artifact -- which turns migrate from a format change
    into a laundering step.

    **Fires when** the converter recomputes descriptors from the store instead
    of copying them: the sha256 below is deliberately not the hash of
    anything, so any re-derivation produces a different value.
    """
    artifacts = {
        "store": {
            "kind": "store",
            "path": "results/plate/zarr/a.ome.zarr",
            "sha256": "0" * 64,
            "size": 123,
        }
    }
    _plant_legacy_markers(
        tmp_path, dataset="plate", stem="a",
        image_complete=True, artifacts=artifacts,
    )
    before = json.loads(
        _legacy_marker_path(tmp_path, "plate", "a").read_text(encoding="utf-8")
    )["artifacts"]

    convert_per_image_markers(tmp_path)

    assert _record(tmp_path, "plate", "a")["artifacts"] == before


def test_the_legacy_trees_are_removed_only_after_every_record_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Marker-last, applied to the migration itself.

    A conversion that deleted as it went and then died would leave a tree that
    is neither shape. This converter does not remove the legacy trees at all
    -- that is rollback's shared primitive -- so the property asserted is the
    stronger one: an interrupted conversion leaves every legacy marker intact
    and therefore re-runnable.

    **Fires when** a rename or unlink is moved into the per-image loop.
    """
    import phenotypic._cli._cli_migrate_state as mod

    _plant_legacy_markers(
        tmp_path, dataset="plate", stem="a", image_complete=True
    )
    _plant_legacy_markers(
        tmp_path, dataset="plate", stem="b", image_complete=True
    )
    planned = plan_per_image_records(tmp_path)
    assert len(planned) == 2

    calls = {"n": 0}
    real = mod.atomic_write_json

    def _fail_after_one(path, payload, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("injected failure partway through")
        return real(path, payload, **kwargs)

    monkeypatch.setattr(mod, "atomic_write_json", _fail_after_one)
    with pytest.raises(RuntimeError):
        apply_per_image_records(tmp_path, planned)

    for stem in ("a", "b"):
        assert _legacy_marker_path(tmp_path, "plate", stem).exists(), (
            f"legacy marker {stem} was removed before conversion completed"
        )


# ---------------------------------------------------------------------------
# Idempotence and the merge rule
# ---------------------------------------------------------------------------


def test_conversion_is_idempotent(tmp_path: Path) -> None:
    """Re-running after an interruption is the documented recovery procedure."""
    _plant_legacy_markers(
        tmp_path, dataset="plate", stem="a", image_complete=True
    )
    convert_per_image_markers(tmp_path)
    first = image_record_path(tmp_path, "plate", "a").read_bytes()
    convert_per_image_markers(tmp_path)
    assert image_record_path(tmp_path, "plate", "a").read_bytes() == first


def test_conversion_merges_into_an_existing_record(tmp_path: Path) -> None:
    """CAN-13. Merge, do not overwrite.

    The both-shapes-present case is real: an old-build SLURM array holds the
    old schema for its whole lifetime -- up to 30 days -- and keeps writing
    the legacy trees after a partial migrate. ``test_conversion_is_idempotent``
    cannot catch this, because it converts a tree with no forward record.

    **Fires when** the converter replaces the record instead of merging: the
    ``stage3`` entry written by the forward path below disappears.
    """
    from phenotypic._cli._cli_image_record import record_stage

    record_stage(tmp_path, "plate", "a", "stage3", {"at": "new"})
    _plant_legacy_markers(
        tmp_path, dataset="plate", stem="a", image_complete=True
    )
    convert_per_image_markers(tmp_path)

    stages = _record(tmp_path, "plate", "a")["stages"]
    assert {"stage3", "measured"} <= set(stages), (
        "the forward path's stage3 entry was lost"
    )
    assert stages["stage3"]["at"] == "new", (
        "the newer stage3 payload was overwritten by the legacy one"
    )


def test_a_stage_collision_keeps_the_later_entry(tmp_path: Path) -> None:
    """CAN-13's second half: the rule is the later ``completed_at``.

    ``test_conversion_merges_into_an_existing_record`` cannot catch this. It
    plants only ``image_complete/``, so the legacy record contributes
    ``measured`` and the forward ``stage3`` is never contested -- the union
    succeeds whichever direction the merge prefers.

    **Fires when** the merge is a blind ``update`` in either direction. It was:
    the first draft of this module did ``merged.update(converted)``, which
    replaces a forward ``stage3`` with the legacy one every time, and the
    suite stayed green because no test made the two collide.
    """
    from phenotypic._cli._cli_image_record import record_stage

    # Forward path records stage3 with a LATER timestamp than the legacy
    # marker's 2026-09-03.
    record_stage(tmp_path, "plate", "a", "stage3", {"at": "2026-12-25T00:00:00+00:00"})
    _plant_legacy_markers(
        tmp_path, dataset="plate", stem="a", stage3_complete=True
    )
    convert_per_image_markers(tmp_path)

    stages = _record(tmp_path, "plate", "a")["stages"]
    assert stages["stage3"]["at"] == "2026-12-25T00:00:00+00:00", (
        "the legacy stage3 replaced a newer forward entry"
    )


def test_a_stage_collision_takes_the_legacy_entry_when_it_is_later(
    tmp_path: Path,
) -> None:
    """The other direction, so the rule is a comparison and not a preference.

    **Fires when** the merge is hardcoded to prefer the existing record --
    which would pass the test above for the wrong reason and silently drop
    every legacy stage that is genuinely newer than a stale forward one.
    """
    from phenotypic._cli._cli_image_record import record_stage

    record_stage(tmp_path, "plate", "a", "stage3", {"at": "2020-01-01T00:00:00+00:00"})
    _plant_legacy_markers(
        tmp_path, dataset="plate", stem="a", stage3_complete=True
    )
    convert_per_image_markers(tmp_path)

    stages = _record(tmp_path, "plate", "a")["stages"]
    assert stages["stage3"]["at"] == "2026-09-03T00:00:00.000+00:00", (
        "a legacy stage3 newer than the forward one was discarded"
    )


def test_an_unreadable_marker_does_not_abort_the_whole_migration(
    tmp_path: Path,
) -> None:
    """One truncated marker in a tree of thousands must not strand the rest.

    **Fires when** the converter lets a JSON error escape: image ``b`` would
    have no record, and the operator would be told the migration failed with
    no indication of which of ten thousand files was at fault.
    """
    _plant_legacy_markers(
        tmp_path, dataset="plate", stem="a", image_complete=True
    )
    _legacy_marker_path(tmp_path, "plate", "a").write_text("{trunca", encoding="utf-8")
    _plant_legacy_markers(
        tmp_path, dataset="plate", stem="b", image_complete=True
    )

    assert convert_per_image_markers(tmp_path) == 1
    assert image_record_path(tmp_path, "plate", "b").is_file()
    assert not image_record_path(tmp_path, "plate", "a").exists()
