"""The per-image record's writers and readers (P3 Task 1, spec §6.1).

**The record replaces three independently-written marker trees with one
file**, which is what makes "is this image done?" one JSON read instead of a
read plus up to three ``is_file()`` probes across three directory trees. It is
also what creates the hazard these tests exist for: three writers that could
not lose each other's writes now share a file.
"""

from __future__ import annotations

import json

import pytest

from phenotypic.sdk_ import image_record_path


def _a_promoted_store(root, stem="a"):
    """Return a minimal store the record writer will accept as an artifact.

    ``publish_image_record`` resolves every artifact ``strict=True``, so a
    path that does not exist fails there rather than in the assertion -- and a
    test passing a stub would exercise a different branch from the one under
    test.
    """
    store = root / "results" / "plate" / "zarr" / f"{stem}.ome.zarr"
    store.mkdir(parents=True, exist_ok=True)
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    return store


def test_the_record_is_one_file_carrying_every_stage(tmp_path):
    """Spec §6.1: one read replaces one read plus three `is_file()` probes."""
    from phenotypic.sdk_._image_record import read_image_record

    from phenotypic._cli._cli_image_record import publish_image_record

    publish_image_record(
        tmp_path,
        work_id="w",
        dataset="plate",
        image_stem="a",
        relative_image_path="a.tif",
        mode="full",
        stages={
            "stage1": {"at": "2026-09-03T00:00:00Z"},
            "stage2": {"at": "2026-09-03T00:00:01Z", "objmap_shape": [8, 8]},
            "stage3": {"at": "2026-09-03T00:00:02Z"},
            "measured": {"at": "2026-09-03T00:00:03Z"},
        },
        artifacts={"store": _a_promoted_store(tmp_path)},
        attempt_id="attempt",
        lifecycle_epoch="epoch",
    )

    assert image_record_path(tmp_path, "plate", "a").is_file()
    record = read_image_record(tmp_path, "plate", "a")
    assert set(record["stages"]) == {
        "stage1",
        "stage2",
        "stage3",
        "measured",
    }
    assert record["artifacts"]["store"]["kind"] == "store"


def test_stages_is_an_open_map(tmp_path):
    """§6.1: an unknown stage is additive, not a schema break.

    The shared `STAGE_*` constants close the *typo* class -- the names this
    build writes have one spelling -- without closing the *extension* class. A
    reader that rejected unknown keys would turn every future stage into a
    migration.
    """
    from phenotypic.sdk_._image_record import read_image_record

    from phenotypic._cli._cli_image_record import publish_image_record

    publish_image_record(
        tmp_path,
        work_id="w",
        dataset="plate",
        image_stem="a",
        relative_image_path="a.tif",
        mode="full",
        stages={"stage1": {"at": "t"}, "some_future_stage": {"at": "t"}},
        artifacts={},
        attempt_id="x",
        lifecycle_epoch="e",
    )
    record = read_image_record(tmp_path, "plate", "a")
    assert "some_future_stage" in record["stages"]


def test_recording_one_stage_leaves_the_others_untouched(tmp_path):
    """The three collapsed trees were independently writable; stay so.

    Stage 2 and Stage 3 run in different jobs, on different nodes, minutes
    apart. Before the collapse they wrote three files and could not lose each
    other's writes. After it they share one, and this is the property that
    replaces the separation.
    """
    from phenotypic.sdk_._image_record import read_image_record

    from phenotypic._cli._cli_image_record import record_stage

    record_stage(tmp_path, "plate", "a", "stage1", {"at": "t1"})
    record_stage(tmp_path, "plate", "a", "stage2", {"at": "t2"})

    record = read_image_record(tmp_path, "plate", "a")
    assert set(record["stages"]) == {"stage1", "stage2"}
    assert record["stages"]["stage1"] == {"at": "t1"}


def test_publishing_merges_stages_rather_than_replacing_them(tmp_path):
    """CAN-6 rule 1, and the bug it prevents is live in the SLURM worker.

    The Stage-3 worker calls `publish_image_success` **before** recording
    stage 3, so a `publish_image_record` that wrote a complete `stages` map
    would erase `stage1` and `stage2` -- and the image would look unprocessed
    to the next resume, having done all of its work. Today that survives by
    ordering luck; this makes it survive by construction.
    """
    from phenotypic.sdk_._image_record import read_image_record

    from phenotypic._cli._cli_image_record import (
        publish_image_record,
        record_stage,
    )

    record_stage(tmp_path, "plate", "a", "stage1", {"at": "t1"})
    record_stage(tmp_path, "plate", "a", "stage2", {"at": "t2"})
    publish_image_record(
        tmp_path,
        work_id="w",
        dataset="plate",
        image_stem="a",
        relative_image_path="a.tif",
        mode="full",
        stages={"measured": {"at": "t4"}},
        artifacts={},
        attempt_id="x",
        lifecycle_epoch="e",
    )

    stages = read_image_record(tmp_path, "plate", "a")["stages"]
    assert set(stages) == {"stage1", "stage2", "measured"}, (
        "publish replaced the stages map instead of merging into it; a "
        "Stage-3 publish would erase the two stages that preceded it"
    )


def test_consuming_a_stage_is_idempotent(tmp_path):
    """A retried worker must be able to clean up after its predecessor.

    The bool is the whole interface: consuming twice is not an error, so the
    return value says what happened rather than an exception saying it did
    not.
    """
    from phenotypic._cli._cli_image_record import consume_stage, record_stage

    record_stage(tmp_path, "plate", "a", "stage2", {"at": "t2"})

    assert consume_stage(tmp_path, "plate", "a", "stage2") is True
    assert consume_stage(tmp_path, "plate", "a", "stage2") is False
    assert consume_stage(tmp_path, "plate", "missing", "stage2") is False


def test_reading_a_corrupt_record_is_none_not_an_error(tmp_path):
    """INV-VERDICT, degrade half: unreadable makes an image look LESS done."""
    from phenotypic.sdk_._image_record import read_image_record

    path = image_record_path(tmp_path, "plate", "a")
    path.parent.mkdir(parents=True)
    path.write_text("{truncated", encoding="utf-8")

    assert read_image_record(tmp_path, "plate", "a") is None


def test_a_record_that_is_not_an_object_is_also_none(tmp_path):
    """A JSON array parses cleanly and is still not a record.

    Separate from the truncation case because it exercises a different
    branch: `json.loads` succeeds, so only the isinstance check stands
    between a list and an `AttributeError` at the first `.get`.
    """
    from phenotypic.sdk_._image_record import read_image_record

    path = image_record_path(tmp_path, "plate", "a")
    path.parent.mkdir(parents=True)
    path.write_text("[1, 2, 3]", encoding="utf-8")

    assert read_image_record(tmp_path, "plate", "a") is None


@pytest.mark.parametrize(
    "record, expected",
    [
        pytest.param({}, "forward", id="absent-means-forward"),
        pytest.param(None, "forward", id="unreadable-means-forward"),
        pytest.param(
            {"provenance": "migrated"}, "migrated", id="explicitly-migrated"
        ),
        pytest.param(
            {"provenance": "nonsense"}, "forward", id="unknown-is-fenced"
        ),
    ],
)
def test_absent_provenance_reads_as_forward(record, expected):
    """U-10 rule 1: **the default is the strict one.**

    `"forward"` is the value that KEEPS the `work_id` fence, so a record
    written before this field existed -- or by a writer that forgets it -- is
    fenced like any other. Defaulting the other way, or treating an
    unrecognized value as "not forward", would strip the fence from every tree
    written before P3 and do it silently.
    """
    from phenotypic.sdk_._image_record import record_provenance

    assert record_provenance(record) == expected


def test_a_forward_publish_clears_a_migrated_marking(tmp_path):
    """U-10 rule 2: `"migrated"` is write-once and non-propagating.

    Only `--mode migrate` writes it, and any forward run that rewrites the
    record takes the default and restores `"forward"`. That is what makes the
    relaxation self-limiting rather than a permanent hole in the fence.
    """
    from phenotypic.sdk_._image_record import (
        PROVENANCE_MIGRATED,
        read_image_record,
        record_provenance,
    )

    from phenotypic._cli._cli_image_record import publish_image_record

    common = dict(
        work_id="w",
        dataset="plate",
        image_stem="a",
        relative_image_path="a.tif",
        mode="full",
        stages={"measured": {"at": "t"}},
        artifacts={},
        attempt_id="x",
        lifecycle_epoch="e",
    )
    publish_image_record(tmp_path, provenance=PROVENANCE_MIGRATED, **common)
    assert (
        record_provenance(read_image_record(tmp_path, "plate", "a"))
        == "migrated"
    )

    publish_image_record(tmp_path, **common)
    assert (
        record_provenance(read_image_record(tmp_path, "plate", "a"))
        == "forward"
    ), (
        "a forward publish left the migrated marking in place; U-10's "
        "relaxation is now permanent for this image"
    )


def test_the_stage_names_have_exactly_one_home(tmp_path):
    """CAN-27: close the typo class rather than reporting it.

    O-2 proposed a `KNOWN_STAGES` frozenset feeding an advisory emitted by
    `resolve_run_state` -- which INV-LAYER forbids from importing `_cli`.
    Resolving that meant duplicating the frozenset or breaking the invariant,
    so the names became one shared constant instead: a misspelled stage cannot
    be constructed, which is strictly less code than the advisory it replaces.

    This is the half of the plan's `test_the_stage_names_come_from_one_shared
    _constant` that belongs to Task 1. The other half asserts
    `_cli_stage2_token.STAGE_STAGE2 is STAGE_STAGE2` and the `_cli_staged
    _resume` equivalent, and those modules do not import the constants until
    **Task 3** -- so it moves there with the imports it checks, keeping `is`
    rather than `==` because a shared-object check is the whole content of
    CAN-27.
    """
    from phenotypic.sdk_ import _run_state
    from phenotypic.sdk_._image_record import (
        PROVENANCE_MIGRATED,
        STAGE_MEASURED,
    )

    assert _run_state.STAGE_MEASURED is STAGE_MEASURED
    assert _run_state.PROVENANCE_MIGRATED is PROVENANCE_MIGRATED


def test_the_record_carries_the_lifecycle_epoch_under_that_name(tmp_path):
    """Not `scheduler_epoch`. §5.1's collapse was withdrawn, not deferred.

    The value already has exactly one on-disk name -- it is what
    `publish_image_success` writes into every image marker -- so giving this
    new artifact a second spelling would be that collapse arriving from the
    other direction.
    """
    from phenotypic._cli._cli_image_record import publish_image_record

    path = publish_image_record(
        tmp_path,
        work_id="w",
        dataset="plate",
        image_stem="a",
        relative_image_path="a.tif",
        mode="full",
        stages={},
        artifacts={},
        attempt_id="x",
        lifecycle_epoch="the-epoch",
    )
    record = json.loads(path.read_text(encoding="utf-8"))

    assert record["lifecycle_epoch"] == "the-epoch"
    assert "scheduler_epoch" not in record
