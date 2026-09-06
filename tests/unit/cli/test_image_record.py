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


# ---------------------------------------------------------------------------
# Task 2 — the publishers move onto the record, and the gate arms with them
# ---------------------------------------------------------------------------


def _publish_a_successful_image(root, *, dataset="plate", stem="a"):
    """Publish one image through the REAL publisher, with a measurements file.

    Through `publish_image_success` rather than by hand: a hand-planted
    fixture keeps validating after the schema moves, which is the drift this
    phase is entirely about. The `measurements` artifact is here because
    `authorized_measurement_sources` reads that role by name.
    """
    from phenotypic._cli._cli_completion import publish_image_success

    store = _a_promoted_store(root, stem=stem)
    measurements = root / "results" / dataset / "measurements" / f"{stem}.pq"
    measurements.parent.mkdir(parents=True, exist_ok=True)
    measurements.write_bytes(b"parquet-ish")
    publish_image_success(
        root,
        work_id="w",
        dataset=dataset,
        relative_image_path=f"{stem}.tif",
        image_stem=stem,
        mode="full",
        attempt_id="attempt",
        lifecycle_epoch="local",
        artifacts={"store": store, "measurements": measurements},
    )
    return store


def test_publish_image_success_writes_the_record_not_the_legacy_marker(
    tmp_path,
):
    """D1 is a clean break, not a dual write.

    A dual write would leave every tree carrying both shapes, and the schema
    gate could no longer tell a legacy tree from a current one -- which is
    the distinction the whole arming decision rests on.
    """
    from phenotypic.sdk_ import (
        image_completion_marker_path,
        image_record_path,
    )

    _publish_a_successful_image(tmp_path)

    assert image_record_path(tmp_path, "plate", "a").is_file()
    assert not image_completion_marker_path(tmp_path, "plate", "a").exists(), (
        "the legacy image_complete/ marker is still being written; D1 is a "
        "clean break, not a dual write"
    )


def test_valid_image_success_still_rejects_a_tampered_artifact(tmp_path):
    """The artifact-digest contract is unchanged by the collapse.

    Only *where the descriptors live* moved. This is the property
    `_walk_current_success` has today and the one P6 will lean on.
    """
    from phenotypic._cli._cli_completion import valid_image_success

    store = _publish_a_successful_image(tmp_path)
    assert valid_image_success(
        tmp_path, dataset="plate", image_stem="a", work_id="w"
    )

    (store / "zarr.json").write_text('{"tampered": true}', encoding="utf-8")
    assert not valid_image_success(
        tmp_path, dataset="plate", image_stem="a", work_id="w"
    )


def test_a_stage2_only_record_is_not_a_success_proof(tmp_path):
    """CAN-23, and the collapse is what makes it possible to get wrong.

    A Stage-2 GPU worker writes `stages.stage2` and **no artifacts** into the
    same file a success proof lives in. Before the collapse those were two
    trees and mistaking one for the other was unrepresentable; now it is one
    missing check away.
    """
    from phenotypic._cli._cli_completion import valid_image_success
    from phenotypic._cli._cli_image_record import record_stage

    record_stage(tmp_path, "plate", "a", "stage2", {"at": "t"})

    assert not valid_image_success(
        tmp_path, dataset="plate", image_stem="a", work_id="w"
    ), "a record with no artifacts certified an image"


def test_authorized_sources_reads_records_not_the_deleted_tree(tmp_path):
    """CAN-22, and an empty result here is VALID, which is what makes it bad.

    This arm used to re-open `image_complete/` after `valid_image_success`
    passed. After the clean break that file is gone, so every image would
    raise `OSError`, `continue`, and leave the mapping empty -- and `{}` is a
    legitimate schema-3 answer meaning "nothing succeeded yet". P4's
    `finalize_run` would then write an empty master and raise nothing: a
    successful-looking run that discarded every measurement.
    """
    from tests._output_layout import write_processing_state

    from phenotypic._cli._cli_completion import (
        authorized_measurement_sources,
    )

    _publish_a_successful_image(tmp_path, stem="a")
    _publish_a_successful_image(tmp_path, stem="b")
    write_processing_state(
        tmp_path, work_ids={"plate": {"a.tif": "w", "b.tif": "w"}}
    )

    sources = authorized_measurement_sources(tmp_path)

    assert sources, (
        "authorized_measurement_sources returned nothing; it is still "
        "reading image_complete/, and P4 would publish an empty master"
    )
    assert len(sources) == 2
    assert set(sources.values()) == {"plate"}


def test_a_forward_run_does_not_reintroduce_the_demoted_dataset_sets(
    tmp_path,
):
    """§4.2, and without it signal 3 is permanently un-dischargeable.

    P7 deletes these keys from the file; if `save_processing_state` re-adds
    them, the gate fires again on the very next run and the tree is refused
    by every writing mode, forever -- escapable only by `--overwrite`, which
    destroys the outputs.
    """
    from tests._output_layout import write_processing_state

    from phenotypic.sdk_ import resolve_processing_state_path

    write_processing_state(tmp_path, work_ids={"plate": {"a.tif": "w"}})
    state = json.loads(
        resolve_processing_state_path(tmp_path).read_text(encoding="utf-8")
    )

    for name, entry in state["datasets"].items():
        assert "completed" not in entry, f"{name}: the demoted sets came back"
        assert "failed" not in entry, f"{name}: the demoted sets came back"
        assert "errors" not in entry, f"{name}: the demoted sets came back"
        # AND the inventory survives. `initial_images` is not a demoted set:
        # it is the accepted inventory, re-read from the stored state at
        # `_cli_state_management.py:166` because nothing in the event log can
        # reconstruct it. Asserting only the absences would have passed
        # against a writer that dropped this too -- which is what the first
        # version of the writer did, emptying every dataset's inventory on
        # one save/load round trip.
        assert entry["initial_images"] == ["a.tif"], (
            f"{name}: the accepted inventory was dropped with the demoted "
            "sets; nothing else on disk carries it"
        )


# --------------------------------------------------------------------------
# The two GUI tripwires
#
# `save_processing_state` has 17 call sites across the test suite and NOT ONE
# reaches a GUI consumer.
#
# COUNTED, and the counting is its own small lesson. Two of us said "13" from
# memory -- that is the number of FILES, not call sites, a units error rather
# than an arithmetic one. Re-derive by grepping tests/ for the writer's name,
# keeping the lines where an open paren follows it and dropping the imports.
# Do NOT paste a pattern that includes the paren: the naive form matches THIS
# COMMENT and returns 18, because writing the measurement down changed the
# population it measures. The point survives any of the three numbers; the
# number does not survive not being checked.
#
# Both readers below are covered only by fixtures that
# hand-write `"completed": []` and `"failed": []` as literals
# (`test_output_discovery_contracts.py:264`, `test_runs_registry.py:291`), so
# they keep passing against a writer that no longer emits either key. That gap
# is why P3's demotion broke two readers with a green suite, and closing it is
# the point of these two tests -- the xfail marker is the temporary half.
#
# THEY LIVE BESIDE THE WRITER, NOT IN `tests/gui/`. P6 rewrites the GUI and
# its tests; a tripwire filed with the code it guards gets rewritten alongside
# it and never fires. Filed with the writer, P6's rewrite turns them XPASS,
# and `strict=True` turns XPASS into a failure someone has to look at.
#
# Each can end three ways, all deliberate:
#   xfail  -- today; the defect is present.
#   XPASS  -- a FAILURE under `strict=True`. The reader was fixed: drop the
#             marker and keep the test, which is then the only thing in the
#             suite binding this writer to that reader.
#   skip   -- the symbol is gone, P6's other legitimate ending. Skips are
#             visible under `-rs`; an ImportError absorbed by `xfail` is not,
#             which is why neither test lets one reach the marker.
# --------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "P3 stopped writing datasets.{completed,failed}; "
        "_output_consistency._processing_counts:527 bails to (None,)*4 when "
        "they are absent, silencing three contradiction detectors, so a "
        "manifest that disagrees with the inventory now reads COHERENT. "
        "Owned by P6 Task 2 (deletes the module)."
    ),
)
def test_the_gui_still_contradicts_a_manifest_that_disagrees_with_inventory(
    tmp_path,
):
    """Tripwire 1 of 2. Fails toward MORE finished, which INV-VERDICT forbids.

    The event log holds the true counts and would override them -- but at
    ``:534``, *seven lines after* the bail at ``:527``, so it never runs. A
    reader that gives up before consulting its own authority is the shape
    worth pinning, independent of which keys are involved.
    """
    from tests._output_layout import write_processing_state

    from phenotypic.sdk_ import resolve_processing_state_path

    try:
        from phenotypic.gui.results_viewer import _output_consistency
    except ImportError as exc:  # pragma: no cover - P6 Task 2 deletes it
        # Two causes, and the reason has to say which rather than assuming
        # the flattering one: P6 Task 2 deleting the module (the defect's
        # other legitimate ending), or the `gui` extra simply not being
        # installed (this tripwire then never fires -- a gap, not a pass).
        # `results_viewer/__init__` is lazy, so this import needs no Dash.
        pytest.skip(f"_output_consistency unavailable ({exc!r})")

    write_processing_state(
        tmp_path, work_ids={"plate": {"a.tif": "w", "b.tif": "w"}}
    )
    payload = json.loads(
        resolve_processing_state_path(tmp_path).read_text(encoding="utf-8")
    )

    total, completed, failed, _unfinished = (
        _output_consistency._processing_counts(payload)
    )

    assert total == 2, (
        "the reader could not count an inventory this build just wrote -- "
        "the demoted-key bail fired, and every count below is None"
    )

    # A manifest describing ONE image against an inventory of TWO. Complete,
    # self-consistent, and about a different run: exactly what the silenced
    # `manifest_total != state_total` detector exists to catch.
    report = _output_consistency.classify_output_consistency(
        _output_consistency.OutputCompletionEvidence(
            standalone_bundle=False,
            processing_state_present=True,
            processing_state_readable=True,
            processing_total=total,
            processing_completed=completed,
            processing_failed=failed,
            manifest_present=True,
            manifest_readable=True,
            manifest_is_complete=True,
            manifest_completed=1,
            manifest_failed=0,
            manifest_total=1,
        )
    )

    assert report.state == "contradictory", (
        "a complete manifest for 1 image against an inventory of 2 must be "
        f"surfaced as a contradiction; the GUI reported {report.state!r}"
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "P3 stopped writing datasets.{completed,failed}; "
        "_runs_registry._string_set:1300 RAISES TypeError on the absent key, "
        "caught at :1163 and reported as 'unreadable processing state' -- a "
        "false diagnosis of a perfectly readable file. Owned by P6's "
        "consumer-table row 2 (rewrites `_processing_state_conflict`)."
    ),
)
def test_the_gui_does_not_call_a_state_this_build_wrote_unreadable(tmp_path):
    """Tripwire 2 of 2. Fails toward LESS finished -- safe, but misdiagnosed.

    Reachable whenever ``processing_events.log`` does not exist, because
    ``:1102`` then hands every dataset the no-event branch: the window between
    run start and the first image event, and any state-writing run that emits
    no events at all.

    The direction is what separates this from tripwire 1. Refusing to bind is
    permitted by INV-VERDICT; **naming the wrong cause is not a verdict
    problem at all**, it is a user sent to look for corruption that is not
    there. Only the first tripwire guards the invariant; this one guards the
    message.
    """
    from tests._output_layout import write_processing_state

    from phenotypic.sdk_ import resolve_event_log_path

    try:
        from phenotypic.gui.shell._runs_registry import RunRegistry
    except ImportError as exc:  # pragma: no cover - relocation or no extra
        # Unlike tripwire 1, this import is NOT free: `gui/shell/__init__`
        # eagerly imports `_app`, so it pulls in Dash. Without the `gui`
        # extra this tripwire skips and never fires, which is why the reason
        # carries the exception instead of asserting a cause.
        pytest.skip(f"RunRegistry unavailable ({exc!r})")
    if not hasattr(RunRegistry, "_processing_state_conflict"):
        pytest.skip(
            "_processing_state_conflict was renamed by P6 row 2: re-point "
            "this tripwire at the replacement"
        )

    write_processing_state(tmp_path, work_ids={"plate": {"a.tif": "w"}})
    assert not resolve_event_log_path(tmp_path).exists(), (
        "this test is about the NO-event-log branch; an event log here "
        "would route around the code under test and pass for free"
    )

    conflict = RunRegistry._processing_state_conflict(tmp_path)

    assert "unreadable processing state" not in (conflict or ""), (
        "the GUI called a state file this build just wrote unreadable; it "
        f"is readable, it merely lacks a demoted key -- got {conflict!r}"
    )


def test_a_tree_this_build_wrote_needs_no_conversion(tmp_path):
    """**The ordering evidence.** Nothing else in the suite asserts this.

    The publisher and the demoted-set writer must move together: stop one and
    not the other and signal 1 or signal 3 still fires -- on trees this build
    has just written. The arming test pins publisher-vs-flag; this pins
    publisher-vs-state, which is the half a mis-ordered transition breaks.

    **Says nothing about ``SCHEMA_GATE_ARMED``, deliberately.** It was named
    ``test_the_armed_gate_does_not_refuse_...`` and opened with
    ``assert SCHEMA_GATE_ARMED is True`` as a "only meaningful against an
    armed gate" guard. Both were wrong in the same way: the claim is about
    **detection**, and detection is correct whether or not the refusal is
    surfaced. The ruling that disarmed the gate for P3 would have failed that
    guard and renamed nothing, leaving a green-then-red test whose failure had
    no relationship to what it asserts -- a precondition breaking for an
    unrelated reason, which is the shape this file exists to catch.

    Under the disarmed gate the test earns more than it did armed: it is the
    standing evidence that arming is safe for forward-written trees, which is
    what P7 Task 5 Step 1b needs before it flips the flag. It passes in both
    states, which is the property a piece of evidence should have.
    """
    from tests._output_layout import build_complete_run

    from phenotypic.sdk_._schema_shape import requires_conversion

    root = build_complete_run(tmp_path)

    assert requires_conversion(root) is None, (
        "a tree this build just wrote classifies as needing conversion, so "
        "arming the gate would make every writing mode refuse its own "
        "output -- the publisher and the demoted-set writer are out of step"
    )


def test_the_republish_probe_names_the_record_not_the_legacy_marker():
    """The measure path's re-publish, and the way it fails is the point.

    `_cli_process_single` guards its re-publish with `<path>.is_file()`.
    Point that at `image_complete/` after D1's clean break and the guard is
    False on every forward tree: the re-publish is **skipped with no
    exception and no log**, and the function still returns `True`. A
    `--mode measure` that changes the table's descriptor then rewrites the
    store root, leaving the record's store descriptor stale, and the image
    reads as unprocessed after successfully re-measuring.

    That also falsifies `_cli/CLAUDE.md`'s *"no store write outlives the
    publication that certifies it"*, which the root-only fingerprint
    argument rests on.

    Structural, and deliberately so: the behavioural version needs a real
    re-measure with a descriptor change, which is P4's fixture. What makes
    this worth having anyway is that the failure it guards is **silent** --
    a behavioural test that forgot to assert re-publication would pass too.
    Here the wrong path cannot be spelled without the assertion seeing it.
    """
    import ast
    from pathlib import Path

    import phenotypic._cli._cli_process_single as process_single
    import phenotypic._cli._cli_recompile_tables as recompile_tables

    for module in (process_single, recompile_tables):
        names = {
            node.func.id
            for node in ast.walk(
                ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
            )
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id
            in {"image_record_path", "image_completion_marker_path"}
        }
        assert "image_record_path" in names, (
            f"{module.__name__} does not resolve the record path at all; "
            "its re-publish probe cannot be looking at the right file"
        )

    republish_sources = Path(process_single.__file__).read_text(
        encoding="utf-8"
    )
    assert "image_completion_marker_path" not in republish_sources, (
        "the measure path still names the legacy marker; after the clean "
        "break that probe is False on every forward tree and the "
        "re-publish is skipped silently"
    )


def test_the_migrator_publishes_through_the_shared_writer():
    """CAN-7: the migrator is a second PRODUCER of this schema, not a stage.

    `_cli_migrate_image` calls `publish_image_success` directly, so it emits
    records for free -- but only while that stays the single call. A second
    publisher added here would write the old shape into a tree the forward
    path then reads as a record.

    Structural rather than a migrated-tree fixture, and the gap is worth
    naming: this catches a *new writer* appearing, not a drift in what the
    existing one emits. The plan asks for a real-migrator fixture asserting
    the record validates under `valid_image_success`; that is heavier than
    this cluster and is not done here.
    """
    import ast
    from pathlib import Path

    import phenotypic._cli._cli_migrate_image as migrate_image

    tree = ast.parse(
        Path(migrate_image.__file__).read_text(encoding="utf-8")
    )
    publishers = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        in {
            "publish_image_success",
            "publish_image_record",
            "atomic_write_json",
        }
    }

    assert "publish_image_success" in publishers
    assert "publish_image_record" not in publishers, (
        "the migrator gained a second per-image publisher; it must publish "
        "through publish_image_success so one writer owns the record shape"
    )
