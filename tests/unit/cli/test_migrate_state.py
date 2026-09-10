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

from phenotypic._cli._cli_identity import derive_processing_generation
from phenotypic._cli._cli_migrate_state import (
    LEGACY_MARKER_SEGMENTS,
    apply_per_image_records,
    convert_per_image_markers,
    convert_processing_state,
    plan_per_image_records,
    plan_processing_state,
)
from phenotypic.sdk_ import (
    DIR_STAGE2_DONE,
    ProcessingStateKey,
    atomic_write_json,
    image_completion_marker_path,
    image_record_path,
    processing_state_path,
    progress_dir,
)
from phenotypic.sdk_._image_record import PROVENANCE_MIGRATED

_STAGE3_SEGMENT = "stage3_complete"

#: Distinguishes "the caller did not supply a restart epoch" from "the caller
#: supplied ``None``". Signal 4 of the schema gate fires on ``work_ids``
#: present with ``restart_epoch`` **absent**, so a fixture that cannot express
#: absence cannot build the shape the gate detects.
_ABSENT = object()


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


# ---------------------------------------------------------------------------
# Task 3 -- processing_state.json
# ---------------------------------------------------------------------------

def _plant_legacy_state(
    root: Path,
    *,
    processing_generation: str = "deadbeef" * 4,
    pipeline_sha256: str = "a" * 64,
    datasets: dict | None = None,
    restart_epoch: object = _ABSENT,
) -> Path:
    """Write a v2 ``processing_state.json`` by hand.

    Hand-written on purpose, unlike the per-image fixtures: the shape under
    test is one this build **no longer writes** (P5 stopped emitting
    ``datasets.{completed,failed,started}``), so `save_processing_state`
    cannot produce it and a fixture routed through the production writer
    would silently test the current schema instead of the legacy one.
    """
    payload: dict = {
        ProcessingStateKey.VERSION: "2.0.0",
        ProcessingStateKey.PIPELINE_PATH: str(root / "pipeline.json"),
        ProcessingStateKey.INPUT_PATH: str(root / "input"),
        ProcessingStateKey.OUTPUT_DIR: str(root),
        ProcessingStateKey.TIMESTAMP: "2026-01-01T00:00:00",
        ProcessingStateKey.LAST_UPDATED: "2026-01-01T00:00:00",
        ProcessingStateKey.EXECUTION_MODE: "local",
        ProcessingStateKey.DATASETS: datasets
        if datasets is not None
        else {
            "plate": {
                ProcessingStateKey.INITIAL_IMAGES: ["a.tif", "b.tif"],
                ProcessingStateKey.COMPLETED: ["a.tif"],
                ProcessingStateKey.STARTED: ["b.tif"],
                ProcessingStateKey.FAILED: [],
            }
        },
        ProcessingStateKey.CONFIG: {
            "processing_generation": processing_generation,
            "pipeline_sha256": pipeline_sha256,
            "work_ids": {"plate": {"a.tif": "work-a", "b.tif": "work-b"}},
            "success_markers_required": True,
        },
    }
    if restart_epoch is not _ABSENT:
        payload[ProcessingStateKey.CONFIG]["restart_epoch"] = restart_epoch
    path = processing_state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _state(root: Path) -> dict:
    return json.loads(processing_state_path(root).read_text(encoding="utf-8"))


def _state_config(root: Path) -> dict:
    return _state(root)[ProcessingStateKey.CONFIG]


def _assert_conversion_ran(root: Path) -> None:
    """Assert the conversion actually happened, for an invariance test to lean on.

    **An invariance assertion without this is vacuous.** "``work_ids`` is
    unchanged", "the state is stable", "planning writes nothing" are all
    satisfied *most strongly* by an implementation that does nothing at all,
    so on their own they cannot fail for the reason they exist and cannot
    drive an implementation. Pairing each with a witness that the operation
    occurred is what makes "unchanged **while** something changed" the claim.

    The witness is the generation: it is the one field whose new value cannot
    be present before the conversion, because it is derived rather than
    copied.
    """
    config = _state_config(root)
    assert config["processing_generation"] != "deadbeef" * 4, (
        "the conversion did not run, so any invariance below is vacuous"
    )
    assert config["restart_epoch"] == 0


def test_the_uuid_generation_becomes_content_derived(tmp_path: Path) -> None:
    """Spec §5.1 / D3: a generation two invocations can agree on without meeting.

    The legacy value is a ``uuid4()``, which no other process can re-derive --
    so a cold SLURM worker cannot fence itself against a run it has never
    seen. The replacement is a digest of the configuration, and this asserts
    it equals what ``derive_processing_generation`` produces from the tree's
    own recorded inputs rather than merely that it changed.
    """
    _plant_legacy_state(tmp_path)

    convert_processing_state(tmp_path)

    config = _state_config(tmp_path)
    assert config.get("restart_epoch") == 0
    assert config["processing_generation"] != "deadbeef" * 4
    assert config["processing_generation"] == derive_processing_generation(
        pipeline_sha256="a" * 64,
        per_image_config=None,
        restart_epoch=0,
    )


def test_the_generation_is_stable_across_two_conversions(
    tmp_path: Path,
) -> None:
    """What "content-derived" has to mean, and the only test that shows it.

    ``!= "deadbeef"...`` passes for a second ``uuid4()``. Deriving the same
    token twice from the same tree is the property, and it is also what makes
    re-running an interrupted migrate the documented recovery procedure
    instead of a way to mint a third identity.
    """
    _plant_legacy_state(tmp_path)

    convert_processing_state(tmp_path)
    _assert_conversion_ran(tmp_path)
    first = _state_config(tmp_path)["processing_generation"]
    convert_processing_state(tmp_path)

    _assert_conversion_ran(tmp_path)
    assert _state_config(tmp_path)["processing_generation"] == first


def test_the_consumed_dataset_counts_are_removed(tmp_path: Path) -> None:
    """§4.2, and the two keys go for **different** reasons.

    ``completed`` is a cache of a cache *when an event log exists*:
    ``load_processing_state`` re-aggregates it (``:148-167``), so the stored
    copy answers a question the reader has already answered. That is why its
    deletion is conditional -- without an event log the fallback at
    ``:183-187`` reads this key and has nowhere else to go.

    ``started`` is not a cache of anything. It has **no reader at all** and no
    writer since P5, so it goes unconditionally. Stating one reason for both
    would be wrong about one of them.
    """
    _plant_legacy_state(tmp_path)
    _plant_legacy_markers(
        tmp_path, dataset="plate", stem="a", image_complete=True
    )
    _plant_legacy_markers(
        tmp_path, dataset="plate", stem="b", image_complete=True
    )
    convert_per_image_markers(tmp_path)

    convert_processing_state(tmp_path)

    plate = _state(tmp_path)[ProcessingStateKey.DATASETS]["plate"]
    assert "completed" not in plate
    assert "started" not in plate
    assert plate[ProcessingStateKey.INITIAL_IMAGES] == ["a.tif", "b.tif"]


def test_an_unconsumed_dataset_keeps_its_counts(tmp_path: Path) -> None:
    """MIG-16, and the reason this deletion cannot be unconditional.

    On a **pre-markers** tree there is no ``image_complete/``, so Task 2's
    converter produces no records and ``datasets.completed`` is the *only*
    surviving record of what finished. Deleting it in the same pass that
    failed to consume it is not a schema change, it is losing the tree's own
    account of its work -- and the next ``--mode full`` reprocesses every
    image from source.

    Task 2b's ported promoter is what consumes this shape. Until it has run
    for a dataset, that dataset's counts stay.
    """
    _plant_legacy_state(tmp_path)

    convert_processing_state(tmp_path)

    _assert_conversion_ran(tmp_path)
    plate = _state(tmp_path)[ProcessingStateKey.DATASETS]["plate"]
    assert plate[ProcessingStateKey.COMPLETED] == ["a.tif"]
    # `started` goes anyway, and unconditionally: P5 stopped writing it and
    # `grep -rn "ProcessingStateKey.STARTED" src/` finds no reader, so there is
    # nothing for a condition to protect. It is not a weaker copy of
    # `completed`; it is a key with no writer and no reader.
    assert ProcessingStateKey.STARTED not in plate


def test_completed_survives_a_partially_consumed_dataset(
    tmp_path: Path,
) -> None:
    """The test a "does this dataset have any record" predicate fails.

    ``completed`` names ``a.tif``; the only record belongs to ``b.tif``. The
    weaker predicate sees *a* record and drops the key, deleting the tree's
    only statement that ``a`` finished on the strength of ``b`` having one --
    and on a tree with no event log there is nowhere else to recover it
    (``_cli_state_management.py:183-187``). The per-stem predicate keeps it.
    """
    _plant_legacy_state(tmp_path)
    _plant_legacy_markers(
        tmp_path, dataset="plate", stem="b", image_complete=True
    )
    convert_per_image_markers(tmp_path)

    convert_processing_state(tmp_path)

    _assert_conversion_ran(tmp_path)
    plate = _state(tmp_path)[ProcessingStateKey.DATASETS]["plate"]
    assert plate[ProcessingStateKey.COMPLETED] == ["a.tif"], (
        "the account of a.tif was deleted because b.tif had a record"
    )


def test_the_failure_set_is_retained_until_it_has_a_home(
    tmp_path: Path,
) -> None:
    """MIG-16 again, and a **deliberate departure from Task 3's Step 1**.

    That step's test asserts ``failed`` is deleted alongside ``completed`` and
    ``started``. It must not be, yet — though **not** for the reason an earlier
    draft of this docstring gave. That draft said ``failed`` has nothing to
    re-aggregate from; it is false, and it would send a reader to the wrong
    line. ``load_processing_state`` uses the event-derived ``DatasetState``
    wholesale (``_cli_state_management.py:176-179``), ``failed`` included.

    The case that carries the retention is the fallback at ``:181-187``: a
    tree with **no event log** reads ``failed`` and ``completed`` from
    ``processing_state.json`` and has nowhere else to get them. That is the
    pre-markers shape.

    Task 2b Step 3 gives it a destination -- each entry becomes an
    ``append_terminal_failure`` record -- and deletes it there, conditioned on
    that conversion having run. Task 2b has **not** run
    (``_migrate_legacy_success_evidence`` is still at
    ``phenotypicCLI.py:571``), so this task retains the set and says why.
    """
    _plant_legacy_state(
        tmp_path,
        datasets={
            "plate": {
                ProcessingStateKey.INITIAL_IMAGES: ["a.tif"],
                ProcessingStateKey.COMPLETED: [],
                ProcessingStateKey.STARTED: [],
                ProcessingStateKey.FAILED: ["a.tif"],
            }
        },
    )
    _plant_legacy_markers(
        tmp_path, dataset="plate", stem="a", image_complete=True
    )
    convert_per_image_markers(tmp_path)

    convert_processing_state(tmp_path)

    _assert_conversion_ran(tmp_path)
    plate = _state(tmp_path)[ProcessingStateKey.DATASETS]["plate"]
    assert ProcessingStateKey.COMPLETED not in plate, (
        "this dataset HAS records, so the re-derivable keys must be gone -- "
        "otherwise the retention below is proved by nothing being deleted at "
        "all, which is the vacuous reading"
    )
    assert plate[ProcessingStateKey.FAILED] == ["a.tif"], (
        "the failure set was deleted before anything consumed it"
    )


def test_work_ids_are_untouched(tmp_path: Path) -> None:
    """D-C keeps ``processing_configuration_digest`` unchanged, so every
    existing ``work_id`` stays valid. Re-minting them would invalidate every
    record migrate just converted, and a tree half-migrated across that
    boundary cannot be recovered without the original config.
    """
    _plant_legacy_state(tmp_path)
    before = _state_config(tmp_path)["work_ids"]

    convert_processing_state(tmp_path)

    _assert_conversion_ran(tmp_path)
    assert _state_config(tmp_path)["work_ids"] == before


def test_planning_the_state_writes_nothing_at_all(tmp_path: Path) -> None:
    """The ``--dry-run`` seam, pinned the same way Task 2's planner is.

    Task 5 renders a dry run by calling the planner alone, so "writes
    nothing" has to be a property of the function rather than of how the
    caller happens to use it. Compares the whole tree's bytes, not just the
    state file: a planner that wrote a lock, a temp file or a `.phenotypic/`
    directory would pass a narrower check.
    """
    _plant_legacy_state(tmp_path)
    before = {
        p: p.read_bytes() for p in sorted(tmp_path.rglob("*")) if p.is_file()
    }

    planned = plan_processing_state(tmp_path)

    assert planned is not None
    # The witness: the PLAN carries the conversion while the FILE does not.
    # Without this the test passes against a planner that returns an unchanged
    # copy, or one that returns the input verbatim.
    assert planned.config["processing_generation"] != "deadbeef" * 4
    assert planned.config["restart_epoch"] == 0
    assert _state_config(tmp_path)["processing_generation"] == "deadbeef" * 4

    after = {
        p: p.read_bytes() for p in sorted(tmp_path.rglob("*")) if p.is_file()
    }
    assert after == before


def test_converting_the_state_is_idempotent(tmp_path: Path) -> None:
    """Re-running after an interruption is the documented recovery procedure."""
    _plant_legacy_state(tmp_path)

    convert_processing_state(tmp_path)
    _assert_conversion_ran(tmp_path)
    first = processing_state_path(tmp_path).read_bytes()
    convert_processing_state(tmp_path)

    assert processing_state_path(tmp_path).read_bytes() == first


def test_a_tree_with_no_state_is_a_cheap_no_op(tmp_path: Path) -> None:
    """Same shape as the per-image converter: unconditional at the call site.

    A tree that never had `processing_state.json` -- a bundle, a directory
    this package never wrote -- must not acquire one, and must not raise.
    """
    assert plan_processing_state(tmp_path) is None
    assert convert_processing_state(tmp_path) is False
    assert not processing_state_path(tmp_path).exists()


def test_an_unreadable_state_is_left_alone(tmp_path: Path) -> None:
    """A truncated state file is *unreadable*, not *legacy*.

    ``requires_conversion`` gives that shape its own verdict rather than
    ``CONVERT``, precisely because migrate cannot repair it
    (``_schema_shape.ConversionVerdict.UNREADABLE_STATE``). Overwriting it
    here with a synthesised v3 block would destroy whatever a human could
    still have recovered by hand.
    """
    path = processing_state_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{truncated", encoding="utf-8")

    assert plan_processing_state(tmp_path) is None
    assert convert_processing_state(tmp_path) is False
    assert path.read_text(encoding="utf-8") == "{truncated"
