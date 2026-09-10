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


# ---------------------------------------------------------------------------
# Task 4: the pre-D8 master CSV, and stores the projection cannot read
# ---------------------------------------------------------------------------


def test_the_pre_d8_master_csv_is_deleted(tmp_path: Path) -> None:
    """D8 removed the constant, the helper and the reader; this is the on-disk
    half, for trees already written.

    **Fires when** the deletion is dropped: the file survives. The co-witness
    is the sibling parquet, which must NOT be deleted -- an implementation that
    cleared the whole deliverables directory would pass a bare "the csv is
    gone" assertion.
    """
    from phenotypic._cli._cli_migrate_state import convert_legacy_master_csv
    from phenotypic.sdk_ import deliverables_dir

    deliverables = deliverables_dir(tmp_path)
    deliverables.mkdir(parents=True, exist_ok=True)
    legacy = deliverables / "master_measurements.csv"
    legacy.write_text("a,b\n1,2\n", encoding="utf-8")
    sibling = deliverables / "master_measurements.parquet"
    sibling.write_bytes(b"PAR1-not-really")

    assert convert_legacy_master_csv(tmp_path) is True

    assert not legacy.exists(), "the pre-D8 master CSV survived the migration"
    assert sibling.is_file(), "the parquet master was deleted too"
    assert sibling.read_bytes() == b"PAR1-not-really"


def test_deleting_the_master_csv_is_a_no_op_when_absent(tmp_path: Path) -> None:
    """Every conversion in this module is unconditional and cheap on a tree
    that does not need it -- the shape the call site depends on."""
    from phenotypic._cli._cli_migrate_state import convert_legacy_master_csv
    from phenotypic.sdk_ import deliverables_dir

    deliverables_dir(tmp_path).mkdir(parents=True, exist_ok=True)
    assert convert_legacy_master_csv(tmp_path) is False


def test_planning_the_master_csv_deletion_writes_nothing(tmp_path: Path) -> None:
    """The dry-run seam, with a co-witness that planning found something.

    **Fires when** planning gains a write. The `is not None` assertion is the
    co-witness: without it, an implementation that returned `None`
    unconditionally would satisfy "the tree is unchanged" most strongly of all.
    """
    from phenotypic._cli._cli_migrate_state import plan_legacy_master_csv
    from phenotypic.sdk_ import deliverables_dir

    deliverables = deliverables_dir(tmp_path)
    deliverables.mkdir(parents=True, exist_ok=True)
    (deliverables / "master_measurements.csv").write_text("a\n1\n", encoding="utf-8")
    before = {
        p.relative_to(tmp_path): p.read_bytes()
        for p in sorted(tmp_path.rglob("*"))
        if p.is_file()
    }

    planned = plan_legacy_master_csv(tmp_path)

    assert planned is not None, "planning found nothing, so the check below is vacuous"
    assert {
        p.relative_to(tmp_path): p.read_bytes()
        for p in sorted(tmp_path.rglob("*"))
        if p.is_file()
    } == before


def test_a_store_with_no_measurement_descriptor_is_named_not_raised(
    tmp_path: Path,
) -> None:
    """CAN-32 / Step 0b: a reachable unhandled path in the projection.

    ``read_embedded_measurement_descriptor`` documents an absent descriptor as
    a **normal state** -- a ``--mode process`` run never measures -- while
    ``embedded_measurement_columns`` raises ``KeyError`` on it. So a tree this
    phase exists to convert can crash the projection.

    **Fires when** the enumeration lets the ``KeyError`` escape, or stops
    finding the store. The second store is the co-witness: a function that
    returned every store it saw, or none, would pass a one-store version.
    """
    from phenotypic._cli._cli_migrate_state import unprojectable_stores
    from phenotypic.sdk_ import zarr_store_path

    bare = zarr_store_path(tmp_path, "plate", "a")
    bare.mkdir(parents=True)
    (bare / "zarr.json").write_text("{}", encoding="utf-8")

    assert unprojectable_stores(tmp_path) == ("results/plate/zarr/a.ome.zarr",)


def test_enumerating_unprojectable_stores_writes_nothing(tmp_path: Path) -> None:
    """It feeds an advisory, so it must not touch the tree it reports on."""
    from phenotypic._cli._cli_migrate_state import unprojectable_stores
    from phenotypic.sdk_ import zarr_store_path

    bare = zarr_store_path(tmp_path, "plate", "a")
    bare.mkdir(parents=True)
    (bare / "zarr.json").write_text("{}", encoding="utf-8")
    before = {
        p.relative_to(tmp_path): p.read_bytes()
        for p in sorted(tmp_path.rglob("*"))
        if p.is_file()
    }

    found = unprojectable_stores(tmp_path)

    assert found, "nothing was enumerated, so the check below is vacuous"
    assert {
        p.relative_to(tmp_path): p.read_bytes()
        for p in sorted(tmp_path.rglob("*"))
        if p.is_file()
    } == before


# ---------------------------------------------------------------------------
# Task 5: retention and revert
# ---------------------------------------------------------------------------


def _retention(root: Path) -> Path:
    from phenotypic._cli._cli_migrate_state import legacy_retention_dir

    return legacy_retention_dir(root)


def test_the_converted_trees_are_renamed_not_deleted(tmp_path: Path) -> None:
    """CAN-12 / §15.1. A user who reverts the code after a migrate would
    otherwise face a tree the old build reads as entirely unprocessed.

    **Fires when** retention deletes instead of moving. The byte assertion is
    the co-witness: a `mkdir` that produced an empty `legacy-v2/` would satisfy
    "the directory exists" but not "the marker is in it, unchanged".
    """
    from phenotypic._cli._cli_migrate_state import retain_legacy_trees
    from phenotypic.sdk_ import progress_dir

    _plant_legacy_markers(tmp_path, dataset="plate", stem="a", image_complete=True)
    before = _legacy_marker_path(tmp_path, "plate", "a").read_bytes()

    assert retain_legacy_trees(tmp_path) == 1

    assert not (progress_dir(tmp_path) / "image_complete").exists()
    retained = _retention(tmp_path) / "image_complete" / "plate" / "a.json"
    assert retained.is_file(), "the legacy tree was deleted, not retained"
    assert retained.read_bytes() == before


def test_the_retained_tree_does_not_make_the_output_look_unconverted(
    tmp_path: Path,
) -> None:
    """It sits below `.phenotypic/`, not `progress/`, so the gate cannot see it.

    **Fires when** retention is moved under `progress/` -- which would make
    every migrated tree classify CONVERT forever, an INV-DISCHARGEABLE
    violation that no test of retention alone would catch.
    """
    from phenotypic._cli._cli_migrate_state import retain_legacy_trees
    from phenotypic.sdk_ import progress_dir
    from phenotypic.sdk_._schema_shape import requires_conversion

    _plant_legacy_markers(tmp_path, dataset="plate", stem="a", image_complete=True)
    assert requires_conversion(tmp_path) is not None, (
        "the tree did not classify CONVERT to begin with, so the check below "
        "cannot show retention made the difference"
    )

    retain_legacy_trees(tmp_path)

    assert _retention(tmp_path).is_dir()
    assert progress_dir(tmp_path) not in _retention(tmp_path).parents
    assert requires_conversion(tmp_path) is None


def test_a_second_migrate_does_not_collide_on_a_non_empty_retention(
    tmp_path: Path,
) -> None:
    """MIG-13, and this case is EXPECTED rather than exceptional.

    Step 1c: an old-build SLURM array holds the old schema for its whole
    lifetime and writes the legacy trees directly, so a tree migrated while
    one is live re-acquires the old shape and is migrated again.

    **Fires when** the rename uses either obvious primitive: `os.replace`
    raises on a non-empty target directory, and `shutil.move` NESTS the source
    inside it -- which would leave the second generation at
    `legacy-v2/image_complete/image_complete/`, findable by nothing.
    """
    from phenotypic._cli._cli_migrate_state import retain_legacy_trees

    _plant_legacy_markers(tmp_path, dataset="plate", stem="a", image_complete=True)
    retain_legacy_trees(tmp_path)

    # The late old-build worker recreates the tree.
    _plant_legacy_markers(tmp_path, dataset="plate", stem="b", image_complete=True)
    assert retain_legacy_trees(tmp_path) == 1

    retained = _retention(tmp_path) / "image_complete"
    assert (retained / "plate" / "b.json").is_file()
    assert not (retained / "image_complete").exists(), (
        "the second generation was nested inside the first"
    )
    assert not list(_retention(tmp_path).glob("*.trash")), "trash was left behind"


def test_revert_puts_the_tree_back(tmp_path: Path) -> None:
    """MIG-13 / §15.1: revert costs a rename back, not a reprocess."""
    from phenotypic._cli._cli_migrate_state import (
        retain_legacy_trees,
        revert_legacy_trees,
    )

    _plant_legacy_markers(tmp_path, dataset="plate", stem="a", image_complete=True)
    before = {
        p.relative_to(tmp_path): p.read_bytes()
        for p in sorted(tmp_path.rglob("*"))
        if p.is_file()
    }
    retain_legacy_trees(tmp_path)
    assert _retention(tmp_path).is_dir(), "nothing was retained, so revert is vacuous"

    assert revert_legacy_trees(tmp_path) == 1

    assert {
        p.relative_to(tmp_path): p.read_bytes()
        for p in sorted(tmp_path.rglob("*"))
        if p.is_file()
    } == before
    assert not _retention(tmp_path).exists()


def test_revert_refuses_when_a_record_the_legacy_tree_does_not_cover_exists(
    tmp_path: Path,
) -> None:
    """Reverting past work done after the migration would discard it.

    **Fires when** revert moves unconditionally. The error names the images,
    because a refusal the user cannot act on is the bug class this whole
    change exists to remove.
    """
    from phenotypic._cli._cli_image_record import record_stage
    from phenotypic._cli._cli_migrate_state import (
        retain_legacy_trees,
        revert_legacy_trees,
    )

    _plant_legacy_markers(tmp_path, dataset="plate", stem="a", image_complete=True)
    retain_legacy_trees(tmp_path)
    # A forward run finishes a NEW image after the migration.
    record_stage(tmp_path, "plate", "b", "measured", {"at": "later"})

    with pytest.raises(RuntimeError, match="plate/b"):
        revert_legacy_trees(tmp_path)

    assert _retention(tmp_path).is_dir(), "the refusal still moved the trees"


def test_revert_refuses_a_tree_with_nothing_retained(tmp_path: Path) -> None:
    """A tree migrated before retention shipped, or already reverted."""
    from phenotypic._cli._cli_migrate_state import revert_legacy_trees

    with pytest.raises(RuntimeError, match="nothing to revert"):
        revert_legacy_trees(tmp_path)


def test_planning_retention_writes_nothing(tmp_path: Path) -> None:
    """The dry-run seam, with a co-witness that planning found a tree."""
    from phenotypic._cli._cli_migrate_state import plan_legacy_tree_retention

    _plant_legacy_markers(tmp_path, dataset="plate", stem="a", image_complete=True)
    before = {
        p.relative_to(tmp_path): p.read_bytes()
        for p in sorted(tmp_path.rglob("*"))
        if p.is_file()
    }

    planned = plan_legacy_tree_retention(tmp_path)

    assert planned, "planning found no tree, so the check below is vacuous"
    assert {
        p.relative_to(tmp_path): p.read_bytes()
        for p in sorted(tmp_path.rglob("*"))
        if p.is_file()
    } == before


def test_a_restart_preserves_the_revert_path(tmp_path: Path) -> None:
    """MIG-12. `clear_machine_state` rmtrees every child of `.phenotypic/`
    except `_PRESERVED_ON_RESTART`, so without the addition `--restart` would
    silently destroy the revert path.

    **Fires when** `legacy-v2` is dropped from that set. The progress
    directory is the co-witness: it must be gone, or the test would pass
    against a `clear_machine_state` that cleared nothing.
    """
    from phenotypic._cli._cli_migrate_state import retain_legacy_trees
    from phenotypic.sdk_ import clear_machine_state, progress_dir

    _plant_legacy_markers(tmp_path, dataset="plate", stem="a", image_complete=True)
    retain_legacy_trees(tmp_path)
    (progress_dir(tmp_path) / "sentinel.json").write_text("{}", encoding="utf-8")

    clear_machine_state(tmp_path)

    assert not progress_dir(tmp_path).exists(), (
        "clear_machine_state cleared nothing, so the survival below is vacuous"
    )
    assert (
        _retention(tmp_path) / "image_complete" / "plate" / "a.json"
    ).is_file(), "--restart destroyed the revert path"


def test_the_flow_converts_in_dependency_order(tmp_path: Path) -> None:
    """The order is a dependency, not a preference.

    `plan_processing_state` reads per-image records
    (`_completed_is_fully_consumed`), so the records must exist before the
    state is planned; retention moves the trees the record conversion read, so
    moving them earlier would convert nothing.

    **Fires when** retention is hoisted above the record conversion: the
    record disappears, because there was no legacy tree left to read.
    """
    from phenotypic._cli._cli_migrate_state import migrate_machine_state
    from phenotypic.sdk_ import progress_dir

    _plant_legacy_markers(tmp_path, dataset="plate", stem="a", image_complete=True)
    plan = migrate_machine_state(tmp_path)

    assert len(plan.records) == 1, "nothing was converted, so the order is untested"
    assert image_record_path(tmp_path, "plate", "a").is_file()
    assert not (progress_dir(tmp_path) / "image_complete").exists()
    assert (
        _retention(tmp_path) / "image_complete" / "plate" / "a.json"
    ).is_file()
    assert plan.state_is_conditional is False


def test_the_rendered_plan_says_when_its_state_half_is_a_prediction(
    tmp_path: Path,
) -> None:
    """The dry run must not present a guess as the plan it will execute.

    On a tree needing per-image conversion, `plan_processing_state` runs
    against records that do not exist yet, so its result is a prediction. A
    renderer has to say so; the flag is how it knows.

    **Fires when** `plan_machine_state_migration` is folded into a single pass
    that renders the state plan as exact -- which is the defect the composition
    callout feared, arriving from the direction it did not consider.
    """
    from phenotypic._cli._cli_migrate_state import plan_machine_state_migration

    _plant_legacy_markers(tmp_path, dataset="plate", stem="a", image_complete=True)
    conditional = plan_machine_state_migration(tmp_path)
    assert conditional.records, "no records planned, so the flag below is vacuous"
    assert conditional.state_is_conditional is True

    # A tree with no legacy markers has nothing the state plan depends on.
    clean = tmp_path / "clean"
    clean.mkdir()
    assert plan_machine_state_migration(clean).state_is_conditional is False


def test_planning_the_whole_migration_writes_nothing(tmp_path: Path) -> None:
    """The dry-run seam for the folded flow, with a co-witness."""
    from phenotypic._cli._cli_migrate_state import plan_machine_state_migration

    _plant_legacy_markers(tmp_path, dataset="plate", stem="a", image_complete=True)
    before = {
        p.relative_to(tmp_path): p.read_bytes()
        for p in sorted(tmp_path.rglob("*"))
        if p.is_file()
    }

    plan = plan_machine_state_migration(tmp_path)

    assert plan.records, "planning found nothing, so the check below is vacuous"
    assert {
        p.relative_to(tmp_path): p.read_bytes()
        for p in sorted(tmp_path.rglob("*"))
        if p.is_file()
    } == before


# ---------------------------------------------------------------------------
# MIG-11: a pre-markers process tree's outputs ARE its completion record
# ---------------------------------------------------------------------------


def _build_pre_markers_process_tree(root: Path, *, stems=("a",)) -> Path:
    """A pre-markers ``--mode process`` run: flat layers, no store, no markers.

    `ext: "png"` is copied from the shipped gate fixture deliberately, and it
    is **wrong there** -- `process_only_output_path` derives the extension from
    the layer and format, giving `.tiff` for `rgb`. Reproduced so this suite
    proves the walk does not key on a field that is written at four sites and
    read at none.
    """
    from phenotypic.sdk_ import resolve_processing_state_path

    (root / "plate").mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (root / "plate" / f"{stem}.tiff").write_bytes(b"pixels-" + stem.encode())
    state = resolve_processing_state_path(root)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(
        json.dumps(
            {
                "version": "2.0.0",
                "config": {"process_only_layer": "rgb", "ext": "png"},
            }
        ),
        encoding="utf-8",
    )
    return root


def test_a_process_output_becomes_a_record(tmp_path: Path) -> None:
    """MIG-11. The output is the only surviving statement that the image was
    produced, so migrate records exactly that.

    Three co-witnesses, because each kills a different stub: `provenance` is
    `migrated` (not the forward default), the artifact descriptor resolves to
    the file on disk (not an empty record), and `work_id` is the sentinel
    (not a digest a future implementation quietly computes).
    """
    from phenotypic._cli._cli_migrate_state import convert_process_output_records
    from phenotypic.sdk_._image_record import (
        PROVENANCE_MIGRATED,
        WORK_ID_UNRECOVERABLE,
    )

    _build_pre_markers_process_tree(tmp_path)
    assert convert_process_output_records(tmp_path) == 1

    record = _record(tmp_path, "plate", "a")
    assert record["provenance"] == PROVENANCE_MIGRATED
    assert record["work_id"] == WORK_ID_UNRECOVERABLE, (
        "an identity was minted where U-10 forbids one"
    )
    assert len(record["work_id"]) != 64, "the work_id looks like a sha256"
    descriptor = record["artifacts"]["process_output"]
    assert (tmp_path / descriptor["path"]).is_file(), (
        "the record certifies an artifact that is not there"
    )
    assert record["mode"] == "process"


def test_the_minted_record_is_not_rejected_for_its_work_id(
    tmp_path: Path,
) -> None:
    """The sentinel works because `record_rejection` skips the comparison for
    migrated records -- which is what `PROVENANCE_MIGRATED` means.

    **Fires when** the provenance stamp is dropped: the record would then be
    fenced on a `work_id` that was never an identity, and the tree it was
    minted for could never be read.
    """
    from phenotypic._cli._cli_migrate_state import convert_process_output_records
    from phenotypic.sdk_ import read_image_record
    from phenotypic.sdk_._image_record import record_rejection

    _build_pre_markers_process_tree(tmp_path)
    convert_process_output_records(tmp_path)

    record = read_image_record(tmp_path, "plate", "a")
    assert record is not None
    assert (
        record_rejection(
            record, dataset="plate", image_stem="a", work_id="anything-at-all"
        )
        is None
    )


def test_a_tree_that_does_not_declare_a_process_run_mints_nothing(
    tmp_path: Path,
) -> None:
    """The narrowness that keeps migrate from accepting any folder of images.

    **Fires when** the classifier is made tolerant of an empty store set
    instead of keying on the declared `process_only_layer` -- which would make
    every directory of pictures a migratable process tree.
    """
    from phenotypic._cli._cli_migrate_state import plan_process_output_records
    from phenotypic.sdk_ import resolve_processing_state_path

    (tmp_path / "plate").mkdir(parents=True)
    (tmp_path / "plate" / "a.tiff").write_bytes(b"pixels")
    state = resolve_processing_state_path(tmp_path)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"version": "2.0.0", "config": {}}), encoding="utf-8")

    assert plan_process_output_records(tmp_path) == ()


def test_planning_process_records_writes_nothing(tmp_path: Path) -> None:
    """The dry-run seam, with a co-witness that planning found an output."""
    from phenotypic._cli._cli_migrate_state import plan_process_output_records

    _build_pre_markers_process_tree(tmp_path, stems=("a", "b"))
    before = {
        p.relative_to(tmp_path): p.read_bytes()
        for p in sorted(tmp_path.rglob("*"))
        if p.is_file()
    }

    planned = plan_process_output_records(tmp_path)

    assert len(planned) == 2, "planning found nothing, so the check is vacuous"
    assert {
        p.relative_to(tmp_path): p.read_bytes()
        for p in sorted(tmp_path.rglob("*"))
        if p.is_file()
    } == before


def test_the_process_arm_is_a_no_op_on_a_full_run_tree(tmp_path: Path) -> None:
    """Every conversion in this module is unconditional and cheap elsewhere."""
    from phenotypic._cli._cli_migrate_state import convert_process_output_records

    _plant_legacy_markers(tmp_path, dataset="plate", stem="a", image_complete=True)
    assert convert_process_output_records(tmp_path) == 0


# ---------------------------------------------------------------------------
# MIG-23: `False` means the documented no-op, and nothing else
# ---------------------------------------------------------------------------


def _write_legacy_state(
    root: Path,
    *,
    datasets: dict[str, dict[str, object]] | None = None,
    **config: object,
) -> Path:
    """A readable legacy `processing_state.json`, with the fields readers index.

    `timestamp` and `last_updated` are not decoration: `load_processing_state`
    subscripts both unguarded, so a state without them is *unreadable* rather
    than legacy -- which is the fault case, not the no-op.

    `datasets` defaults to empty, which is what every MIG-23 caller wants. The
    Blocker 5 fixtures pass it, because `_current_success_counts` derives its
    `total` from `initial_images`: a state that claims no images can never
    report the shortfall those tests are about.
    """
    from phenotypic.sdk_ import resolve_processing_state_path

    state = resolve_processing_state_path(root)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(
        json.dumps(
            {
                "version": "2.0.0",
                "pipeline_path": str(root / "pipeline.json"),
                "input_path": str(root / "input"),
                "output_dir": str(root),
                "timestamp": "2026-09-03T00:00:00",
                "last_updated": "2026-09-03T00:00:00",
                "execution_mode": "local",
                "datasets": dict(datasets or {}),
                "config": dict(config),
            }
        ),
        encoding="utf-8",
    )
    return state


def test_a_pre_markers_archive_is_a_no_op_not_a_failure(tmp_path: Path) -> None:
    """MIG-23, and the user-visible defect it caused.

    `republish_aggregate`'s docstring says a legacy tree with no markers is a
    documented no-op and warns that aborting there "would leave the stores
    written and the run reported as failed". Its own return value made that
    outcome unavoidable: `False` meant both "nothing to do" and "it went
    wrong", so the only safe reading at the caller was to treat every `False`
    as fatal.

    **Fires when** the no-op raises again -- which is what a pre-markers
    archive, the likeliest migration subject there is, would hit first.
    """
    from phenotypic.sdk_._hdf_to_zarr import republish_aggregate

    _write_legacy_state(tmp_path)
    assert republish_aggregate(tmp_path) is False


def test_markers_required_with_none_authorized_is_also_a_no_op(
    tmp_path: Path,
) -> None:
    """The third no-op, and the one an exception-message match would lose.

    A tree that requires markers but has none authorized yet has nothing to
    publish. The old code reached that verdict by letting
    `publish_aggregate_snapshot`'s "No marker-authorized measurements" fall
    into the same `except` as a genuine I/O failure.

    **Fires when** that case is folded back into the failure branch -- which
    my own first draft of this fix did, turning a no-op into a fault.
    """
    from phenotypic.sdk_._hdf_to_zarr import republish_aggregate

    _write_legacy_state(tmp_path, success_markers_required=True)
    assert republish_aggregate(tmp_path) is False


def test_an_unreadable_state_raises_rather_than_returning_false(
    tmp_path: Path,
) -> None:
    """The fault half, which is why the raise could not simply be deleted.

    Making the caller's raise conditional without separating the two would
    have traded a false failure for a **silent** one: a corrupt tree
    migrating "successfully" on the phase that cannot be rolled back.

    **Fires when** a fault returns `False` again. The message must name the
    file, because a refusal the user cannot act on is the bug class this
    whole change exists to remove.
    """
    from phenotypic.sdk_ import resolve_processing_state_path
    from phenotypic.sdk_._hdf_to_zarr import republish_aggregate

    state = resolve_processing_state_path(tmp_path)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("{truncated", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unreadable"):
        republish_aggregate(tmp_path)


def test_the_no_op_and_the_fault_are_distinguishable_at_all(
    tmp_path: Path,
) -> None:
    """The property the fix is actually for, asserted directly.

    Before, both returned `False` from four sites meaning two things. A caller
    given only that boolean could not act correctly on either. This asserts
    the two are now different observable outcomes -- not that each is right
    in isolation, which the tests above cover, but that they *differ*.
    """
    from phenotypic.sdk_ import resolve_processing_state_path
    from phenotypic.sdk_._hdf_to_zarr import republish_aggregate

    no_op = tmp_path / "pre-markers"
    no_op.mkdir()
    _write_legacy_state(no_op)

    fault = tmp_path / "corrupt"
    fault.mkdir()
    state = resolve_processing_state_path(fault)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("{truncated", encoding="utf-8")

    assert republish_aggregate(no_op) is False
    with pytest.raises(RuntimeError):
        republish_aggregate(fault)


# ---------------------------------------------------------------------------
# Blocker 5: migrate converts a tree, it does not assert the run completed
# ---------------------------------------------------------------------------

_B5_GENERATION = "b5-generation"
_B5_METADATA_DIGEST = "b5-metadata-digest"
_B5_MANIFEST_DIGEST = "b5-manifest-digest"


def _incomplete_run(root: Path) -> Path:
    """A converted tree whose run never finished.

    Two accepted images, neither carrying a valid success marker. That is the
    same shortfall as the real subject -- a run interrupted after its first
    image, where `work_ids` claims two and one marker exists -- taken to the
    extreme that needs no valid record to build, and it lands on the same
    `successful != total` branch of `_all_accepted_images_succeeded`.
    """
    root.mkdir(parents=True, exist_ok=True)
    _write_legacy_state(
        root,
        datasets={"plate": {"initial_images": ["a.tiff", "b.tiff"]}},
        success_markers_required=True,
    )
    return root


def _drive_finalizer(
    monkeypatch: pytest.MonkeyPatch,
    run: Path,
    *,
    seam_publisher: bool = False,
) -> tuple[object, list[str], dict[str, object]]:
    """Run `finalize_migration_attempt` past its authority gates.

    Everything patched here is a gate this test is not about -- metadata
    authority, the manifest/image-seal pair, the canonical view, the aggregate
    (MIG-23's subject, covered above). What is *not* patched is the thing under
    test: the completion gate reads the real tree, and with `seam_publisher`
    false so does `publish_run_completion_evidence`.

    Args:
        monkeypatch: pytest patcher.
        run: Output root to finalize.
        seam_publisher: Replace the publisher and its validator with recorders.
            Used by the two arms whose point is *that* publication was reached,
            not what the publisher then wrote.

    Returns:
        The final report, the ordered event names, and the kwargs the terminal
        status publisher was called with.
    """
    from types import SimpleNamespace

    from phenotypic._cli import _cli_migrate as subject
    from phenotypic._cli import _cli_migrate_manifest as manifest_module
    from phenotypic._cli._cli_migrate import (
        MetadataPassResult,
        finalize_migration_attempt,
    )
    from phenotypic.sdk_ import deliverables_dir
    from phenotypic.sdk_._hdf_to_zarr import MigrationReport

    events: list[str] = []
    terminal: dict[str, object] = {}
    authority = SimpleNamespace(
        terminal_receipt_digest=_B5_METADATA_DIGEST,
        status_path=run / "metadata_status.json",
    )

    monkeypatch.setattr(
        subject, "metadata_migration_authority", lambda *_: authority
    )
    monkeypatch.setattr(
        manifest_module,
        "_read_manifest",
        lambda *_, **__: (
            None,
            SimpleNamespace(
                generation=_B5_GENERATION,
                inventory_digest=_B5_MANIFEST_DIGEST,
            ),
        ),
    )
    monkeypatch.setattr(
        subject, "valid_migration_image_seal", lambda *_, **__: True
    )
    monkeypatch.setattr(
        subject,
        "emit_canonical_metadata_view",
        lambda *_, **__: events.append("canonical") or Path("canonical.csv"),
    )
    monkeypatch.setattr(
        subject,
        "_publish_migration_aggregate",
        lambda *_, **__: events.append("aggregate"),
    )
    if seam_publisher:
        monkeypatch.setattr(
            subject,
            "publish_run_completion_evidence",
            lambda *_, **__: (
                events.append("completion") or run / "completion.json"
            ),
        )
        monkeypatch.setattr(
            subject, "valid_run_completion", lambda *_: {"status": "complete"}
        )
    monkeypatch.setattr(
        subject,
        "publish_migration_terminal_status",
        lambda *_, **kwargs: (
            events.append("terminal")
            or terminal.update(kwargs)
            or run / "terminal.json"
        ),
    )
    monkeypatch.setattr(
        subject, "close_migration_generation", lambda *_, **__: None
    )

    report = finalize_migration_attempt(
        run,
        manifest_path=run / "manifest.json",
        expected_scientific_output=deliverables_dir(run),
        generation=_B5_GENERATION,
        metadata_pass=MetadataPassResult(
            headers_migrated=0, failures=(), authority=authority
        ),
        image_seal=SimpleNamespace(
            generation=_B5_GENERATION,
            manifest_digest=_B5_MANIFEST_DIGEST,
            metadata_terminal_digest=_B5_METADATA_DIGEST,
            clean=True,
            failures=(),
        ),
        reclaim_seal=None,
        deletion_requested=False,
        dry_run=False,
        report=MigrationReport(converted=2),
        image_failures=(),
        reclaim_failures=(),
        commit_guard=None,
    )
    return report, events, terminal


def test_an_incomplete_run_converts_without_claiming_it_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blocker 5, and the defect it names.

    `publish_run_completion_evidence` raises "Current run does not have
    complete publication evidence" for a run whose accepted images did not all
    succeed. The finalizer called it unconditionally and its `except` turned
    that raise into a terminal `completion` failure, so migrate refused the
    *entire* conversion over a run state it was never asked to fix -- with the
    stores already written, on the one phase §15.1 says cannot be rolled back
    by reverting code.

    The ruling is to skip the publication, not to soften the raise: publishing
    anyway would write a run proof asserting a run completed that did not.

    **Fires when** the call goes back to unconditional, or when a completion
    marker is written for an incomplete run -- the two failure directions are
    separately asserted below, because a fix that satisfies one by breaking
    the other is exactly what the ruling excludes.
    """
    from phenotypic.sdk_ import run_completion_marker_path
    from phenotypic.sdk_._run_state import resolve_run_state

    run = _incomplete_run(tmp_path / "run")

    report, events, terminal = _drive_finalizer(monkeypatch, run)

    assert report.ok, report.publication_failures
    assert terminal["succeeded"] is True
    assert events == ["canonical", "aggregate", "terminal"]
    # Not "the publisher was skipped" by proxy: the marker itself is absent.
    assert not run_completion_marker_path(run).exists()
    # The co-witness. Skipping publication is only honest if the tree then
    # says so out loud -- a silent skip would leave migrate reporting success
    # over a tree indistinguishable from a finished one.
    assert resolve_run_state(run).completion == "incomplete"


def test_the_skipped_publication_is_exactly_the_one_that_would_raise(
    tmp_path: Path,
) -> None:
    """The witness that the test above is not vacuous.

    A skip that fires on a tree the publisher would have accepted proves
    nothing. This asserts the fixture really is in the raising condition, by
    calling the real publisher on it directly, and that the predicate the gate
    reads reports it as `False` rather than the legacy `None`.
    """
    from phenotypic._cli._cli_completion import (
        _all_accepted_images_succeeded,
        publish_run_completion_evidence,
    )

    run = _incomplete_run(tmp_path / "run")

    assert _all_accepted_images_succeeded(run) is False
    with pytest.raises(
        RuntimeError, match="does not have complete publication evidence"
    ):
        publish_run_completion_evidence(run, execution_epoch=_B5_GENERATION)


def test_a_legacy_tree_still_gets_its_completion_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate is `is False`, not `is not True`, and the difference is a tree.

    `_all_accepted_images_succeeded` returns `None` for a state that does not
    require success markers, and the publisher's *first* branch answers that
    arm by writing a state-free marker -- it never reaches the raise. Gating on
    `is True` would read that `None` as "not complete" and stop publishing for
    the pre-markers archive, which is the likeliest migration subject there is.

    **Fires when** the gate is widened to the `None` arm.
    """
    from phenotypic._cli._cli_completion import _all_accepted_images_succeeded

    run = tmp_path / "run"
    run.mkdir()
    _write_legacy_state(run)

    assert _all_accepted_images_succeeded(run) is None

    _, events, terminal = _drive_finalizer(monkeypatch, run, seam_publisher=True)

    assert "completion" in events
    assert terminal["succeeded"] is True


def test_a_complete_run_still_gets_its_completion_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The third arm, so the gate is pinned across the whole tri-state.

    A process-only run whose claimed images all succeeded returns `True`
    without consulting an aggregate proof, which is the cheapest honest `True`
    a fixture can reach. The image count is zero, so this witnesses the
    *predicate arm* rather than a realistic run -- which is the point: the gate
    must key on the predicate, not on how many images a tree happens to have.

    **Fires when** the gate stops admitting a complete run at all.
    """
    from phenotypic._cli._cli_completion import _all_accepted_images_succeeded

    run = tmp_path / "run"
    run.mkdir()
    _write_legacy_state(
        run,
        success_markers_required=True,
        process_only_layer="rgb",
        work_ids={},
    )

    assert _all_accepted_images_succeeded(run) is True

    _, events, terminal = _drive_finalizer(monkeypatch, run, seam_publisher=True)

    assert "completion" in events
    assert terminal["succeeded"] is True


# ---------------------------------------------------------------------------
# Both execution paths convert machine state, not just the local one
# ---------------------------------------------------------------------------


def _calls_in(module_path: Path, function: str, callee: str) -> list:
    """Return every bare-name call to `callee` inside `function`.

    Returns the `ast.Call` nodes rather than their line numbers, because the
    first version of this helper returned lines and the test then reached for
    `min()` to pick among them -- choosing one of N without ever establishing
    what N was, or which one it wanted. The nodes carry the keywords that say
    which call is which.
    """
    import ast

    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function:
            return [
                call
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == callee
            ]
    raise AssertionError(f"{function} is not defined in {module_path}")


def _literal_dry_run(call) -> bool:
    """Return one call's literal `dry_run=` keyword.

    Raises rather than defaulting: a call that passes the flag through a
    variable is a call this guard cannot classify, and silently sorting it
    into one bucket is how a guard starts agreeing with whatever it is shown.
    """
    import ast

    for keyword in call.keywords:
        if keyword.arg == "dry_run":
            if not isinstance(keyword.value, ast.Constant) or not isinstance(
                keyword.value.value, bool
            ):
                raise AssertionError(
                    f"dry_run= at line {call.lineno} is no longer a literal"
                )
            return keyword.value.value
    raise AssertionError(f"call at line {call.lineno} passes no dry_run=")


def _not_dry_run_spans(module_path: Path, function: str) -> list[tuple[int, int]]:
    """Return the line spans of `if not <something>.dry_run:` bodies."""
    import ast

    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == function):
            continue
        for branch in ast.walk(node):
            if not isinstance(branch, ast.If):
                continue
            test = branch.test
            if not (
                isinstance(test, ast.UnaryOp)
                and isinstance(test.op, ast.Not)
                and isinstance(test.operand, ast.Attribute)
                and test.operand.attr == "dry_run"
            ):
                continue
            lines = [
                inner.lineno
                for statement in branch.body
                for inner in ast.walk(statement)
                if hasattr(inner, "lineno")
            ]
            spans.append((min(lines), max(lines)))
        return spans
    raise AssertionError(f"{function} is not defined in {module_path}")


def test_both_migrate_execution_paths_convert_machine_state() -> None:
    """The local driver and the SLURM chain must both run the conversion.

    Wiring `migrate_machine_state` into `_run_migrate_owned` alone converted
    the local tree and left the SLURM tree unconverted. The visible symptom
    was one field: the SLURM tree kept the **forward** run's
    `processing_generation`, minted with a real `per_image_config` digest,
    while the local tree re-derived it under migrate's inputs
    (`per_image_config=None`, U-7/U-10). Two trees migrated from one archive
    then disagreed about configuration identity, which is precisely what a
    content-derived generation exists to make impossible.

    Asserted structurally because the defect is structural -- a path that does
    not call it at all cannot be caught by a seam. The end-to-end equivalence
    is `test_local_and_synchronous_slurm_migration_publish_equivalent_runs`.

    **Fires when** either call site is dropped, duplicated, or moved across
    one of the two orderings below.
    """
    from phenotypic._cli import _cli_migrate, _cli_migrate_worker

    local = Path(_cli_migrate.__file__)
    worker = Path(_cli_migrate_worker.__file__)

    local_conversion = _calls_in(
        local, "_run_migrate_owned", "migrate_machine_state"
    )
    worker_conversion = _calls_in(
        worker, "_run_metadata_worker", "migrate_machine_state"
    )
    assert len(local_conversion) == 1, [
        call.lineno for call in local_conversion
    ]
    assert len(worker_conversion) == 1, [
        call.lineno for call in worker_conversion
    ]

    # `_execute_migration_tasks` is called TWICE in this one function, and the
    # two calls want opposite orderings. Partitioned by the flag that says
    # which is which, never by position: the earlier call is the dry run.
    by_mode: dict[bool, list[int]] = {}
    for call in _calls_in(
        local, "_run_migrate_owned", "_execute_migration_tasks"
    ):
        by_mode.setdefault(_literal_dry_run(call), []).append(call.lineno)
    assert sorted(by_mode) == [False, True], by_mode
    assert len(by_mode[False]) == 1, by_mode[False]
    assert len(by_mode[True]) == 1, by_mode[True]

    conversion = local_conversion[0].lineno
    # The crash-window argument in the call site's own comment: an
    # interruption between the two must leave the more converted tree.
    assert conversion < by_mode[False][0]
    # And the opposite ordering, which nothing pinned before. A dry run
    # writes nothing, so converting state ahead of it would be a bug -- the
    # `if dry_run:` branch returns before ever reaching the conversion.
    assert conversion > by_mode[True][0]


def test_the_slurm_conversion_is_guarded_by_the_worker_dry_run_flag() -> None:
    """The SLURM half of the same property, which line order cannot express.

    `_run_metadata_worker` has no dry-run early return to sit behind: it takes
    the flag as configuration and guards the writing half with
    `if not config.dry_run:`. The conversion has to be inside that guard, and
    "inside" is a containment question, not a position on the page.

    **Fires when** the call is lifted out of the guard, which would convert
    machine state during `--mode migrate --dry-run` on the SLURM path.
    """
    from phenotypic._cli import _cli_migrate_worker

    worker = Path(_cli_migrate_worker.__file__)
    conversion = _calls_in(
        worker, "_run_metadata_worker", "migrate_machine_state"
    )
    assert len(conversion) == 1, [call.lineno for call in conversion]
    line = conversion[0].lineno

    spans = _not_dry_run_spans(worker, "_run_metadata_worker")
    assert spans, "the worker no longer guards its writes on dry_run"
    assert any(start <= line <= end for start, end in spans), (line, spans)


def test_the_provenance_only_chain_converts_machine_state_too() -> None:
    """The same gap, in the second topology, found by looking rather than failing.

    `_run_migrate_owned` is the local arm for **both** target kinds, so the
    local path converts machine state for a process-output tree as well. The
    provenance-only SLURM chain is store array -> seal -> finalizer and had no
    stage that did -- so a SLURM migration of a process tree skipped the MIG-11
    record minting its local counterpart performs.

    The seal, not a new pre-array stage: `_pre_markers_process_outputs` mints
    records **from the outputs**, so a stage running before the store array
    would run before the outputs it reads exist. The seal is the first
    singleton after the store work, which mirrors the local *data* order.

    No test failed to find this one; it was found by asking what else had the
    shape of the bug already caught. That is why the guard exists.

    **Fires when** the call is dropped, duplicated, lifted out of the dry-run
    guard, or moved after the seal it must precede.
    """
    from phenotypic._cli import _cli_migrate_provenance_worker

    worker = Path(_cli_migrate_provenance_worker.__file__)
    conversion = _calls_in(worker, "_run_seal_worker", "migrate_machine_state")
    assert len(conversion) == 1, [call.lineno for call in conversion]
    line = conversion[0].lineno

    # A dry run writes nothing, and this stage has no early return to sit
    # behind -- the flag is configuration, so the guard is containment.
    spans = _not_dry_run_spans(worker, "_run_seal_worker")
    assert spans, "the seal stage no longer guards its writes on dry_run"
    assert any(start <= line <= end for start, end in spans), (line, spans)

    seals = _calls_in(worker, "_run_seal_worker", "seal_provenance_migration")
    assert len(seals) == 1, [call.lineno for call in seals]
    # Locally the conversion precedes the seal; the mirror must too, or the
    # seal binds a generation over a tree that is about to change.
    assert line < seals[0].lineno


# ---------------------------------------------------------------------------
# The fourth combination: local x provenance-only
# ---------------------------------------------------------------------------


def _minimal_store(path: Path) -> Path:
    """One schema-1 PhenoTypic store: enough to classify and to upgrade."""
    path.mkdir(parents=True)
    (path / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "ome": {"version": "0.5"},
                    "phenotypic": {
                        "store_schema_version": 3,
                        "provenance": {
                            "schema_version": 1,
                            "status": "complete",
                            "pipeline": None,
                            "retry_base_length": 0,
                            "operations": [],
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _migrate_locally(output: Path) -> object:
    """Invoke the real entry point, locally, on one migration target."""
    from click.testing import CliRunner

    from phenotypic.phenotypicCLI import phenotypic_cli

    return CliRunner().invoke(
        phenotypic_cli, ["--mode", "migrate", "--output", str(output)]
    )


def test_a_pre_markers_process_tree_is_migrated_by_the_local_cli(
    tmp_path: Path,
) -> None:
    """MIG-11's minting must be reachable through `--mode migrate`.

    It was not. `run_migrate` branches on `target.kind != "full_run"` into
    `execute_provenance_migration`, which iterates `target.stores` and nothing
    else -- and a pre-markers process tree has **no stores**, by definition:
    the kind exists because `--mode process` gained OME-Zarr output in a later
    release than the trees it converts. So the local arm did nothing, reported
    `provenance_upgraded=0`, and **exited 0**. A silent successful no-op on
    the exact tree the arm exists for.

    The SLURM arm could not reach it either, so before this fix the minting
    was reachable from no execution mode at all -- its passing tests call the
    converter directly.

    **Fires when** the local provenance branch stops converting machine state,
    which returns the minting to being unreachable.
    """
    tree = _build_pre_markers_process_tree(tmp_path / "run", stems=("a", "b"))

    result = _migrate_locally(tree)

    assert result.exit_code == 0, result.output
    # The co-witness that matters: exit 0 was already true when nothing
    # happened, so success is not the assertion -- the records are.
    assert _record(tree, "plate", "a")["mode"] == "process"
    assert _record(tree, "plate", "b")["mode"] == "process"


def test_a_process_tree_is_migrated_by_the_local_cli(tmp_path: Path) -> None:
    """The same gap for the store-bearing kind, where it is less visible.

    A process tree does get its per-store provenance upgraded locally, so the
    migration looks like it worked. What it did not get was any machine-state
    conversion -- which is why `image_complete/` survived a local migration
    and the schema gate went on reading CONVERT.

    **Fires when** the conversion is dropped from the local branch, which
    would leave a tree that migrates successfully and stays unconverted.
    """
    from phenotypic._cli._cli_migrate_state import legacy_retention_dir
    from phenotypic.sdk_ import progress_dir

    tree = tmp_path / "run"
    _minimal_store(tree / "dataset" / "a.ome.zarr")
    _plant_legacy_markers(tree, dataset="dataset", stem="a", image_complete=True)

    result = _migrate_locally(tree)

    assert result.exit_code == 0, result.output
    assert not (progress_dir(tree) / "image_complete").exists(), (
        "the legacy tree survived a local migration"
    )
    assert legacy_retention_dir(tree).is_dir(), "nothing was retained"


def test_a_direct_store_gets_no_machine_state_written_into_it(
    tmp_path: Path,
) -> None:
    """The kind the conversion must NOT reach, asserted rather than assumed.

    A direct store's lifecycle state is a hashed sibling below `.phenotypic`,
    never inside the store -- so converting machine state for it would write
    `.phenotypic/` exactly where that kind's contract forbids. Every arm of
    `migrate_machine_state` happens to no-op on a bare store today, which is
    why the gate is on `kind` and not on that accident.

    **Fires when** the kind gate is dropped and the conversion is applied to
    every provenance-only target.
    """
    store = _minimal_store(tmp_path / "solo.ome.zarr")

    result = _migrate_locally(store)

    assert result.exit_code == 0, result.output
    assert not (store / ".phenotypic").exists(), (
        "machine state was written inside the store"
    )
