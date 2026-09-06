"""The run-state reader: types, identity, verdict matrix, advisories.

Phase 1 Task 2 lands the state types, Task 4 the identity readers, Tasks 5
and 6 the verdict matrix, the advisories, the depth behaviour and the degrade
half of INV-VERDICT.

The fixtures are `tests._output_layout`'s `build_complete_run` /
`build_incomplete_run`, which publish through the **real** publishers. Nothing
in this file hand-writes a marker or a proof it then asserts on -- a fixture
that hand-writes the format under test keeps passing after the format changes,
which is the failure mode this whole plan is about. The one place a marker is
edited by hand is `_mark_migrated`, and it edits a marker the real publisher
wrote, for a shape whose writer does not exist until P7.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from phenotypic.sdk_ import clear_verification_cache


@pytest.fixture(autouse=True)
def _isolate_cache():
    """The verification cache is a module global; a leaked entry makes the
    next test lie about the depth it performed."""
    clear_verification_cache()
    yield
    clear_verification_cache()


@pytest.fixture
def complete_run(tmp_path):
    from tests._output_layout import build_complete_run

    return build_complete_run(tmp_path)


@pytest.fixture
def incomplete_run(tmp_path):
    from tests._output_layout import build_incomplete_run

    return build_incomplete_run(tmp_path)


def _read_state(output_dir):
    from phenotypic.sdk_ import resolve_processing_state_path

    path = resolve_processing_state_path(output_dir)
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_the_demoted_sources_live_only_under_diagnostics():
    """Spec §9: a predicate reaching into state.diagnostics is visibly wrong.

    This does not stop someone writing `if state.diagnostics.verified ==
    state.diagnostics.accepted`, but it does pin WHERE the demoted evidence
    lives. manifest counts and the event log were evidence; §4.2 demoted them.
    If they reappear as top-level RunState fields, the demotion has been undone.
    """
    from phenotypic.sdk_ import RunDiagnostics, RunState

    top = {f.name for f in dataclasses.fields(RunState)}
    assert top == {
        "completion",
        "identity",
        "images",
        "advisories",
        "diagnostics",
        "depth",
        "verified_at",
    }
    diag = {f.name for f in dataclasses.fields(RunDiagnostics)}
    assert diag == {"accepted", "verified", "failed"}, (
        "U-5 dropped manifest_completed/manifest_total/event_log_present after "
        "verifying zero consumers survive P6. Carrying demoted evidence into "
        "RunState is what keeps it alive as a quasi-evidence surface."
    )


def test_image_state_stages_is_an_open_map():
    """Spec 6.1: `stages` takes a key nothing has heard of, unvalidated.

    This is the property D-A's "no backfill stage" note *depends* on -- that
    re-adding a stage later is additive rather than a schema break -- and it is
    the one worth pinning, because it is the one a well-meaning change would
    remove. Adding validation to close the map (an enum, a `__post_init__`
    check, a `Literal`) is a natural-looking tightening that would turn every
    future stage into a breaking change, and nothing else here would notice.

    **This replaces a test that could not fail** (test-review finding 6). The
    original constructed an `ImageState` with a literal `stages` dict and
    asserted `"backfilled" not in state.stages` -- a claim about a literal the
    test itself wrote three lines earlier. `ImageState` is a frozen dataclass
    with no `__post_init__` (`_state_types.py:86-107`), so no change to
    `_run_state.py` could ever have made it red.

    **And forbidding that key was the wrong assertion anyway**, in two ways.
    D-A's constraint is phase-scoped -- *nothing in THIS phase writes or reads
    it* -- which is design intent for review to enforce, not an invariant. And
    a test that forbids one key in a map whose whole point is being open argues
    against the design it claims to protect. The producer today emits only
    `{measured: ...}` or `{}` (`_run_state.py:621,634`); P3 adds
    `stage1`/`stage2`/`stage3`, and a closed-set assertion would have to be
    edited for each -- which is exactly the schema break the open map exists to
    avoid.

    D-A's decision keeps its home in `_state_types.py`'s docstring, which is
    where a reader deciding whether to add a stage will actually look.
    """
    from phenotypic.sdk_ import ImageState

    state = ImageState(
        work_id="w",
        dataset="d",
        image_stem="s",
        stages={"a-stage-invented-by-this-test": {"at": "2026-09-03T00:00:00Z"}},
        verdict="verified",
        reason=None,
    )

    assert state.stages["a-stage-invented-by-this-test"] == {
        "at": "2026-09-03T00:00:00Z"
    }


# ------------------------------------------------------------ Task 4: identity


def test_run_identity_is_none_for_a_tree_with_no_processing_state(tmp_path):
    """The GUI points at arbitrary directories. An unmanaged one has no
    identity, and asking for it is not an error."""
    from phenotypic.sdk_ import run_identity

    assert run_identity(tmp_path) is None


def test_run_identity_degrades_on_an_unreadable_state_file(complete_run):
    """INV-VERDICT's degrade half, at the lowest level: a truncated state
    file yields no identity rather than a traceback out of a reader."""
    from phenotypic.sdk_ import run_identity

    path, _ = _read_state(complete_run)
    path.write_text("{", encoding="utf-8")

    assert run_identity(complete_run) is None


def test_run_identity_reads_todays_state_file(complete_run):
    """P1 lands before P2, so it must work on a uuid4-shaped
    processing_generation and a state file with no restart_epoch field at
    all. That is what makes this phase independently landable."""
    from phenotypic.sdk_ import run_identity

    identity = run_identity(complete_run)
    assert identity is not None
    assert identity.restart_epoch == 0
    assert len(identity.inventory_digest) == 64
    assert identity.finalization_input_digest


def test_assert_identity_current_accepts_the_current_identity(complete_run):
    """The positive control. Without it, `assert_identity_current` could
    raise unconditionally and the mismatch test below would still pass."""
    from phenotypic.sdk_ import assert_identity_current, run_identity

    identity = run_identity(complete_run)
    assert identity is not None
    assert_identity_current(complete_run, identity)  # must not raise


def test_assert_identity_current_names_the_field_that_changed(complete_run):
    """D6: a config change still hard-errors with the SPECIFIC mismatch. A
    generic 'identity changed' would make the content-derived generation a
    worse diagnostic than the uuid it replaces."""
    from phenotypic.sdk_ import assert_identity_current, run_identity

    identity = run_identity(complete_run)
    assert identity is not None
    stale = dataclasses.replace(identity, inventory_digest="0" * 64)
    with pytest.raises(RuntimeError, match="inventory_digest"):
        assert_identity_current(complete_run, stale)


def test_assert_identity_current_raises_when_there_is_no_state(tmp_path):
    """An output with no state cannot be current with anything. Returning
    quietly would let a caller act on an identity for a tree that no longer
    has one."""
    from phenotypic.sdk_ import RunIdentity, assert_identity_current

    identity = RunIdentity(
        processing_generation="g",
        restart_epoch=0,
        scheduler_epoch=None,
        owner_generation=None,
        inventory_digest="0" * 64,
        scientific_config_digest="1" * 64,
        finalization_input_digest="2" * 64,
    )
    with pytest.raises(RuntimeError, match="no readable processing state"):
        assert_identity_current(tmp_path, identity)


def test_finalization_input_digest_is_a_versioned_object(complete_run):
    """Spec §5.5: adding a field is a schema_version bump handled by the
    reader, not a second tree migration."""
    from phenotypic.sdk_ import finalization_input_object

    obj = finalization_input_object(complete_run)
    assert obj["schema_version"] == 1
    assert set(obj) == {
        "schema_version",
        "metadata_sha256",
        "include_dataset_column",
        "no_qc",
    }


def test_finalization_input_object_never_subscripts_the_config(tmp_path):
    """flow-r4 N-4. U-6's detection signal is the ABSENCE of keys, so the
    readers must use `.get`. A tree with no state at all is the extreme
    case: it must still answer with the four keys."""
    from phenotypic.sdk_ import finalization_input_object

    obj = finalization_input_object(tmp_path)
    assert obj["schema_version"] == 1
    assert obj["metadata_sha256"] is None
    assert obj["no_qc"] is False


def test_scheduler_epoch_and_owner_generation_are_not_in_the_digest(
    complete_run,
):
    """They are liveness facts, not configuration. Folding them in would
    discard the verification cache every time a job is submitted against
    unchanged work."""
    from phenotypic.sdk_ import run_identity

    identity = run_identity(complete_run)
    assert identity is not None
    moved = dataclasses.replace(
        identity, scheduler_epoch="other", owner_generation="other"
    )
    assert moved.digest() == identity.digest()


def test_each_digest_token_moves_the_digest(complete_run):
    """The companion to the test above: the five tokens the digest DOES fold
    in must each change it. Without this, an implementation that folded in
    nothing at all would pass the exclusion test."""
    from phenotypic.sdk_ import run_identity

    identity = run_identity(complete_run)
    assert identity is not None
    moved = {
        "processing_generation": "other",
        "restart_epoch": 7,
        "inventory_digest": "0" * 64,
        "scientific_config_digest": "0" * 64,
        "finalization_input_digest": "0" * 64,
    }
    for field, value in moved.items():
        changed = dataclasses.replace(identity, **{field: value})
        assert changed.digest() != identity.digest(), (
            f"{field} is not fenced by RunIdentity.digest()"
        )


# --------------------------------------------------------- tree mutations
#
# Each one is what a real event leaves behind, not the shortest edit that
# flips an assertion. Two of them are worth the extra sentence:
#
#  * `_fail_one_image` REMOVES the marker as well as appending the journal
#    row. Rule 3 is "terminal-failure records exist with no superseding
#    success proof", so a failure row beside a valid marker is superseded and
#    must NOT read `failed` -- pinned separately below. A real terminal
#    failure leaves a row and no marker, and `append_terminal_failure` itself
#    refuses to write one while the marker validates.
#  * `_accept_an_unprocessed_image` edits `config.work_ids` only. That is the
#    rolling-input case: a new file is accepted, and nothing has processed it
#    yet.


def _mutation_of(field, value, path_helper):
    """Return a mutation that rewrites one field of one published proof.

    Falsifying the PROOF rather than the tree is what isolates a single
    comparison. Every realistic tree change moves several digests at once
    (adding an image moves the inventory, the source set and the count), so a
    tree-level case cannot show that one specific comparison is the one doing
    the work -- delete any one of the three and the other two still catch it.
    """

    def mutate(root):
        proof = path_helper(root)
        payload = json.loads(proof.read_text(encoding="utf-8"))
        payload[field] = value
        proof.write_text(json.dumps(payload), encoding="utf-8")

    return mutate


def _falsify_run_proof(field, value=("0" * 64)):
    from phenotypic.sdk_ import run_completion_marker_path

    return _mutation_of(field, value, run_completion_marker_path)


def _falsify_aggregate_proof(field, value=("0" * 64)):
    from phenotypic.sdk_ import aggregate_publication_marker_path

    return _mutation_of(field, value, aggregate_publication_marker_path)


def _leave_untouched(root):
    return None


def _remove_one_image_marker(root):
    from phenotypic.sdk_ import image_record_path
    from tests._output_layout import FIXTURE_DATASET, FIXTURE_STEMS

    image_record_path(
        root, FIXTURE_DATASET, FIXTURE_STEMS[1]
    ).unlink()


def _remove_run_proof(root):
    from phenotypic.sdk_ import run_completion_marker_path

    run_completion_marker_path(root).unlink()


def _corrupt_run_proof(root):
    from phenotypic.sdk_ import run_completion_marker_path

    run_completion_marker_path(root).write_text("{", encoding="utf-8")


def _corrupt_processing_state(root):
    _read_state(root)[0].write_text("{", encoding="utf-8")


def _record_terminal_failure(root, stem, *, attempt="attempt-1"):
    """Append one terminal-failure journal row for an unmarked image.

    Split out from `_fail_one_image` because the two trees this file uses
    differ in whether the marker is there to remove: `build_incomplete_run`
    is *defined* by the second image having none, so a helper that unlinks
    unconditionally raises `FileNotFoundError` on it.

    The `assert` is not decoration. `append_terminal_failure` **refuses** to
    write while the image's marker still validates -- a failure beside a
    valid success proof is superseded by it, and the writer enforces that
    rather than trusting its caller. Asserting the return makes "the row was
    written" a checked fact, so a test that depends on a `failed` verdict
    cannot quietly pass for the wrong reason.
    """
    from phenotypic._cli._cli_failure_tracker import append_terminal_failure
    from tests._output_layout import FIXTURE_DATASET

    assert append_terminal_failure(
        root,
        work_id=f"work-{stem}",
        dataset=FIXTURE_DATASET,
        relative_image_path=f"{stem}.tif",
        failed_stage="measure",
        exception=ValueError("no colonies detected"),
        attempt_id=attempt,
        lifecycle_epoch="local",
    ), "append_terminal_failure refused -- the marker still validates"


def _fail_one_image(root):
    """Leave what a terminal scientific failure actually leaves.

    No marker, one journal row. For a tree whose second image is still
    marked; on one that is already unmarked, call
    :func:`_record_terminal_failure` directly.
    """
    from tests._output_layout import FIXTURE_STEMS

    _remove_one_image_marker(root)
    _record_terminal_failure(root, FIXTURE_STEMS[1])


def _mark_slurm_lifecycle_active(root):
    from phenotypic._cli._cli_slurm_lifecycle import (
        initialize_slurm_lifecycle,
    )

    initialize_slurm_lifecycle(root, generation="gen-1", mode="slurm")


def _accept_an_unprocessed_image(root):
    """Rolling input: a new file is accepted and not yet processed."""
    from tests._output_layout import FIXTURE_DATASET, write_processing_state

    _, payload = _read_state(root)
    work_ids = payload["config"]["work_ids"]
    work_ids[FIXTURE_DATASET]["c.tif"] = "work-c"
    write_processing_state(root, work_ids=work_ids)


def _bump_metadata_snapshot(root):
    from tests._output_layout import bump_metadata_snapshot_digest

    bump_metadata_snapshot_digest(root)


def _bump_pipeline_digest(root):
    from tests._output_layout import bump_scientific_config_digest

    bump_scientific_config_digest(root)


def _a_dead_pid() -> int:
    import psutil

    pid = max(psutil.pids()) + 1
    while psutil.pid_exists(pid):
        pid += 1
    return pid


def _write_owner_record(root, *, status, pid):
    """Write a GUI launch owner record.

    Hand-written, and that is a real compromise: the production writer is
    ``RunRegistry``'s CAS persist path, which needs a sandbox root and a
    registered run. What is asserted here is only the three fields the
    liveness check reads -- ``status``, ``pid``, ``generation`` -- and P6
    Task 5, which rewrites ``_assert_output_claimable_locked``, is where this
    record's reader and writer land in one place.
    """
    from phenotypic.sdk_ import gui_launch_owner_path

    path = gui_launch_owner_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "generation": "11111111111111111111111111111111",
                "mode": "local",
                "status": status,
                "pid": pid,
            }
        ),
        encoding="utf-8",
    )


def _mark_migrated(root, stem):
    """Rewrite one marker into U-10's migrated shape.

    Hand-edited on purpose: ``--mode migrate`` is the writer and it does not
    exist until P7, so there is nothing real to drive. The marker being
    edited was written by the real publisher, and the edit is exactly the two
    fields U-10 specifies -- ``provenance: "migrated"``, and a ``work_id``
    that no longer matches the inventory, which is the whole point: a
    pre-markers tree never had one to match.
    """
    from phenotypic.sdk_ import image_record_path
    from tests._output_layout import FIXTURE_DATASET

    path = image_record_path(root, FIXTURE_DATASET, stem)
    marker = json.loads(path.read_text(encoding="utf-8"))
    marker["provenance"] = "migrated"
    marker["work_id"] = "work-id-that-never-existed"
    path.write_text(json.dumps(marker), encoding="utf-8")


def _record_an_older_metadata_snapshot(root, stem, digest="e" * 64):
    """Record a diverging metadata snapshot on one store, and re-prove it.

    The store root is what carries the fact (D-A), and rewriting it
    invalidates that image's marker -- so the marker is re-published
    afterwards through the real publisher. Without that the tree would be
    incomplete for an unrelated reason and the advisory assertion would be
    reading the wrong tree.
    """
    from phenotypic._cli._cli_completion import publish_image_success
    from phenotypic.sdk_ import dataset_overlays_dir, zarr_store_path
    from phenotypic.sdk_._run_state import (
        _METADATA_TABLE_ATTR,
        _SNAPSHOT_SHA256_ATTR,
    )
    from phenotypic.sdk_.ngff_ import STORE_ROOT_JSON, PhenotypicAttr
    from tests._output_layout import FIXTURE_DATASET

    store = zarr_store_path(root, FIXTURE_DATASET, stem)
    root_json = store / STORE_ROOT_JSON
    payload = json.loads(root_json.read_text(encoding="utf-8"))
    payload["attributes"][PhenotypicAttr.ROOT][_METADATA_TABLE_ATTR] = {
        _SNAPSHOT_SHA256_ATTR: digest
    }
    root_json.write_text(json.dumps(payload), encoding="utf-8")
    publish_image_success(
        root,
        work_id=f"work-{stem}",
        dataset=FIXTURE_DATASET,
        relative_image_path=f"{stem}.tif",
        image_stem=stem,
        mode="full",
        attempt_id=f"attempt-{stem}",
        lifecycle_epoch="local",
        artifacts={
            "store": store,
            "overlay": dataset_overlays_dir(root, FIXTURE_DATASET)
            / f"{stem}.png",
        },
    )


# ------------------------------------------------------- Task 5: the ladder


def test_an_empty_directory_is_incomplete_and_never_raises(tmp_path):
    """INV-VERDICT, degrade half. An unmanaged directory is not an error --
    the GUI points at arbitrary paths and must get an answer, not a
    traceback."""
    from phenotypic.sdk_ import resolve_run_state

    state = resolve_run_state(tmp_path, depth="deep")
    assert state.completion == "incomplete"
    assert state.images == {}
    assert state.advisories


def test_a_pre_markers_tree_is_incomplete_and_never_raises(
    complete_run, monkeypatch
):
    """U-6 / flow-r4 N-4. The pre-markers shape is schema 2.0.0 with **no
    `work_ids` key**, and the detection signal is that absence.

    So every read of `config` in this module has to be `.get`: a subscript
    would raise `KeyError` from inside the one function whose job is to
    classify that tree, and a traceback is not a verdict. `resolve_run_state`'s
    job is to answer `incomplete` and say why.

    The advisory half is gated -- see
    `test_the_schema_advisory_is_gated_and_never_a_lie` for why -- so it is
    asserted here with the flag armed, on the tree the gate is really about.
    """
    from phenotypic.sdk_ import _schema_shape, resolve_run_state

    path, payload = _read_state(complete_run)
    del payload["config"]["work_ids"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    state = resolve_run_state(complete_run, depth="deep")
    assert state.completion == "incomplete"
    assert state.images == {}

    monkeypatch.setattr(_schema_shape, "SCHEMA_GATE_ARMED", True)
    armed = resolve_run_state(complete_run, depth="deep")
    assert armed.completion == "incomplete", "the advisory must not gate"
    assert any("migrate" in advisory for advisory in armed.advisories)


def test_the_schema_advisory_is_gated_and_never_a_lie(
    complete_run, monkeypatch
):
    """§4.3's reader half, and the reason it waits for the same flag the
    refusal waits for.

    Armed, the advisory appears and the verdict is **unchanged** -- which is
    the whole of "informational, not a gate".

    **The premise inverted at P3, and the legacy shape is now PLANTED rather
    than inherited.** This test used to open by asserting that
    `requires_conversion(complete_run)` is `CONVERT`, on the grounds that
    "the current shape IS the legacy shape" -- true at P1, because
    `publish_image_success` created `image_complete/` (signal 1) and
    `save_processing_state` wrote `datasets.<ds>.completed` (signal 3). P3
    stopped both, so this fixture -- built by those very publishers -- is now
    a *current* tree and `requires_conversion` returns `None`. The old
    assertion did not merely go stale: **it was measuring P3's success and
    failing because P3 succeeded.**

    So signal 1 is planted explicitly. An empty directory is the faithful
    plant: `_schema_shape.py:255` probes `(progress / segment).is_dir()`, and
    `test_schema_gate.py` pins that emptying the tree leaves the verdict at
    `CONVERT` while removing it flips to `None` -- a half-migrated tree is not
    a converted one. Planting a marker *file* would additionally imply a
    publisher that no longer exists.

    What the gating protects is unchanged and is why the advisory waits for
    the same flag the refusal does: an advisory that fires on every output
    teaches people to ignore the one that will matter.
    """
    from phenotypic.sdk_ import (
        DIR_IMAGE_COMPLETE,
        _schema_shape,
        resolve_progress_dir,
        resolve_run_state,
    )
    from phenotypic.sdk_._schema_shape import (
        ConversionVerdict,
        requires_conversion,
    )

    assert requires_conversion(complete_run) is None, (
        "this fixture is supposed to be a CURRENT tree before the plant; a "
        "verdict here means some other signal is live and the plant below "
        "would prove nothing about signal 1"
    )
    (resolve_progress_dir(complete_run) / DIR_IMAGE_COMPLETE).mkdir(
        parents=True, exist_ok=True
    )
    assert requires_conversion(complete_run) is ConversionVerdict.CONVERT, (
        "the premise of this test: the tree must need conversion, or the "
        "advisory being absent below would mean nothing"
    )

    disarmed = resolve_run_state(complete_run, depth="deep")
    assert disarmed.completion == "complete"
    assert not any("migrate" in a for a in disarmed.advisories)

    monkeypatch.setattr(_schema_shape, "SCHEMA_GATE_ARMED", True)
    armed = resolve_run_state(complete_run, depth="deep")
    assert armed.completion == "complete", (
        "the schema advisory became a gate; §4.3 says it never is"
    )
    assert any("migrate" in a for a in armed.advisories)


@pytest.mark.parametrize(
    "mutate,expected",
    [
        pytest.param(_leave_untouched, "complete", id="untouched"),
        pytest.param(
            _remove_one_image_marker, "incomplete", id="missing-marker"
        ),
        pytest.param(_remove_run_proof, "incomplete", id="no-run-proof"),
        pytest.param(
            _corrupt_run_proof, "incomplete", id="unreadable-proof"
        ),
        pytest.param(
            _corrupt_processing_state, "incomplete", id="unreadable-state"
        ),
        pytest.param(_fail_one_image, "failed", id="terminal-failure"),
    ],
)
def test_the_verdict_matrix(complete_run, mutate, expected):
    from phenotypic.sdk_ import resolve_run_state

    mutate(complete_run)
    assert resolve_run_state(complete_run, depth="deep").completion == expected


def test_a_live_worker_over_an_unfinished_run_reads_active(incomplete_run):
    """Rule 2, where rule 1 does not fire. The plan's verdict matrix put this
    row on the COMPLETE fixture, where rule 1 wins and the row can only ever
    read `complete` -- which its own neighbouring test asserts. This is the
    tree the row was describing."""
    from phenotypic.sdk_ import resolve_run_state

    _mark_slurm_lifecycle_active(incomplete_run)
    assert (
        resolve_run_state(incomplete_run, depth="deep").completion == "active"
    )


def test_a_live_worker_does_not_mask_a_valid_run_proof(complete_run):
    """Q2: `complete` outranks `active`.

    A run proof covers the CURRENT inventory, so a live worker at that point
    is either fenced by restart_epoch or is a new invocation that has already
    changed the inventory -- in which case rule 1 does not fire and this is
    not the case being decided.
    """
    from phenotypic.sdk_ import resolve_run_state

    _mark_slurm_lifecycle_active(complete_run)
    assert (
        resolve_run_state(complete_run, depth="deep").completion == "complete"
    )


def test_an_active_run_outranks_a_stale_terminal_failure(incomplete_run):
    """Q2 rule 2 over rule 3: a failure from a previous attempt must not mask
    an attempt currently **retrying it**.

    Same image, two attempts -- which is what the rule is about. The second
    image already has no marker (that is what `build_incomplete_run` means),
    so the journal row lands on exactly the state a failed attempt leaves,
    and the live lifecycle fence is the retry now in flight.

    The `failed == 1` assertion is load-bearing. Without it this test would
    pass on a tree where the journal row never registered, because rule 2
    would then be outranking rule *4* -- which proves nothing about the
    precedence this test is named for.
    """
    from phenotypic.sdk_ import resolve_run_state
    from tests._output_layout import FIXTURE_STEMS

    _record_terminal_failure(incomplete_run, FIXTURE_STEMS[1])
    _mark_slurm_lifecycle_active(incomplete_run)

    state = resolve_run_state(incomplete_run, depth="deep")
    assert state.diagnostics.failed == 1, (
        "the terminal-failure row did not register, so rule 3 is not the "
        "rule being outranked"
    )
    assert state.completion == "active"


def test_a_superseded_failure_does_not_make_the_run_failed(complete_run):
    """Rule 3's second half -- "with no superseding success proof" -- which
    nothing else in this file exercises. A journal row for an image that
    subsequently succeeded is history, not a verdict."""
    from phenotypic.sdk_ import resolve_run_state
    from tests._output_layout import FIXTURE_STEMS, _publish_one_image

    stem = FIXTURE_STEMS[1]
    _remove_one_image_marker(complete_run)
    _record_terminal_failure(complete_run, stem)
    # The retry succeeds: the marker comes back, the journal row stays.
    _publish_one_image(complete_run, stem=stem, mode="full")

    state = resolve_run_state(complete_run, depth="deep")
    assert state.completion == "complete"
    assert state.diagnostics.failed == 0


def test_a_dead_gui_owner_does_not_pin_an_unfinished_run_at_active(
    incomplete_run,
):
    """CAN-24, on the tree where the check can actually be observed.

    Nothing in this codebase repairs `gui_launch_owner.json` (audit S7,
    verified), so a SIGKILLed GUI leaves `status: "running"` forever. Without
    a liveness check on the authority itself, rule 2 is unsound and this
    output reads `active` until someone edits JSON by hand.

    The plan put this assertion on the COMPLETE fixture, where rule 1 already
    returns `complete` and the test passes whether or not the liveness check
    exists. That version is kept below as the precedence check it actually
    is; this one is the one that fails when the check is deleted.
    """
    from phenotypic.sdk_ import resolve_run_state

    _write_owner_record(incomplete_run, status="running", pid=_a_dead_pid())
    assert (
        resolve_run_state(incomplete_run, depth="deep").completion
        == "incomplete"
    )


def test_a_dead_gui_owner_does_not_pin_the_verdict_at_active(complete_run):
    """The precedence half: a valid run proof outranks any owner record."""
    from phenotypic.sdk_ import resolve_run_state

    _write_owner_record(complete_run, status="running", pid=_a_dead_pid())
    assert (
        resolve_run_state(complete_run, depth="deep").completion == "complete"
    )


def test_a_live_gui_owner_reads_active(incomplete_run):
    """The positive control for the liveness check. Without it, an
    implementation that always answered "not alive" would pass the CAN-24
    test above and silently delete rule 2's GUI half."""
    import os

    from phenotypic.sdk_ import resolve_run_state

    _write_owner_record(incomplete_run, status="running", pid=os.getpid())
    assert (
        resolve_run_state(incomplete_run, depth="deep").completion == "active"
    )


# --------------------------------------------------- Task 5: rule 1 in full


def test_clause_one_is_load_bearing(complete_run):
    """U-2 restored §4.3's FIRST clause, which an earlier draft dropped: every
    accepted image has a valid proof.

    It is **not** redundant with `source_set_digest`. An aggregate proof whose
    source set is exactly the succeeded subset is what a *partial*
    publication looks like, and every one of clause 2's five comparisons is
    satisfied by it -- the inventory still covers two images, the pipeline and
    the finalization inputs are unchanged, and the source set matches the one
    image that succeeded. Only clause 1 notices that the run is not done.

    This is also what makes completion O(N) in per-image proofs, and therefore
    what makes the verification cache load-bearing rather than marginal.
    """
    from phenotypic.sdk_ import resolve_run_state
    from phenotypic.sdk_._digests import canonical_digest
    from tests._output_layout import FIXTURE_STEMS

    _remove_one_image_marker(complete_run)
    surviving = [f"work-{FIXTURE_STEMS[0]}"]
    _falsify_aggregate_proof(
        "source_set_digest", canonical_digest(surviving)
    )(complete_run)
    _falsify_aggregate_proof("source_image_count", len(surviving))(
        complete_run
    )

    assert (
        resolve_run_state(complete_run, depth="deep").completion
        == "incomplete"
    )


@pytest.mark.parametrize(
    "mutate,comparison",
    [
        pytest.param(
            _falsify_run_proof("inventory_digest"),
            "inventory_digest",
            id="inventory_digest",
        ),
        pytest.param(
            _falsify_run_proof("scientific_config_digest"),
            "scientific_config_digest",
            id="scientific_config_digest",
        ),
        pytest.param(
            _falsify_run_proof("finalization_input_digest"),
            "finalization_input_digest",
            id="finalization_input_digest",
        ),
        pytest.param(
            _falsify_aggregate_proof("source_set_digest"),
            "source_set_digest",
            id="source_set_digest",
        ),
        pytest.param(
            _falsify_aggregate_proof("source_image_count", 99),
            "source_image_count",
            id="source_image_count",
        ),
        pytest.param(
            _falsify_run_proof("publication_id", "deadbeef"),
            "publication_id",
            id="publication_id",
        ),
    ],
)
def test_each_of_rule_ones_comparisons_is_load_bearing(
    complete_run, mutate, comparison
):
    """CAN-4. The one-line rule 1 kept only `inventory_digest`; each of these
    would have read `complete` under it.

    Exactly one comparison can catch each case, which is what makes deleting
    that comparison show up here rather than being absorbed by a neighbour.
    """
    from phenotypic.sdk_ import resolve_run_state

    mutate(complete_run)
    assert (
        resolve_run_state(complete_run, depth="deep").completion
        == "incomplete"
    ), f"{comparison} is not being compared; §7.4 and CAN-5 both depend on it"


@pytest.mark.parametrize(
    "mutate,event",
    [
        pytest.param(
            _bump_metadata_snapshot,
            "the metadata snapshot changed",
            id="late-metadata",
        ),
        pytest.param(
            _bump_pipeline_digest,
            "the pipeline changed",
            id="pipeline-edit",
        ),
        pytest.param(
            _accept_an_unprocessed_image,
            "a rolling input accepted a new image",
            id="rolling-input",
        ),
    ],
)
def test_a_real_change_after_publication_invalidates_completion(
    complete_run, mutate, event
):
    """The same five comparisons, driven by what actually happens rather than
    by editing a proof. §7.4's late-metadata guarantee is real today *only*
    because of the finalization comparison: a metadata edit leaves `work_ids`
    untouched, so nothing else would notice."""
    from phenotypic.sdk_ import resolve_run_state

    mutate(complete_run)
    assert (
        resolve_run_state(complete_run, depth="deep").completion
        == "incomplete"
    ), f"{event} left the run reading complete"


def test_a_fully_republished_rolling_input_reads_complete(complete_run):
    """The positive control for the case above. A new image that was
    processed AND re-proved is complete -- otherwise "detects a change" would
    be indistinguishable from "never reads complete twice"."""
    from phenotypic.sdk_ import resolve_run_state
    from tests._output_layout import extend_complete_run

    extend_complete_run(complete_run, stem="c")

    state = resolve_run_state(complete_run, depth="deep")
    assert state.completion == "complete"
    assert state.diagnostics.accepted == 3


def test_a_process_run_reads_complete(tmp_path):
    """N-4. A process run publishes no aggregate proof, so three of rule 1's
    five comparisons are inapplicable -- not merely different. The flat
    conjunction CAN-4's fix introduced made every process tree read
    `incomplete` forever.

    `_cli_completion.py` carries five carve-outs for this, and process mode
    is in scope elsewhere in this change (CAN-20 parametrizes identity over
    it, CAN-32 classifies it for `requires_conversion`), so it cannot be
    waved off.
    """
    from phenotypic.sdk_ import resolve_run_state
    from tests._output_layout import build_complete_run

    output = build_complete_run(tmp_path, process_only_layer="objmap")
    assert resolve_run_state(output, depth="deep").completion == "complete"


def test_a_process_run_still_detects_a_pipeline_edit(tmp_path):
    """The carve-out narrows the comparison set; it does not disable it."""
    from phenotypic.sdk_ import resolve_run_state
    from tests._output_layout import (
        build_complete_run,
        bump_scientific_config_digest,
    )

    output = build_complete_run(tmp_path, process_only_layer="objmap")
    bump_scientific_config_digest(output)
    assert resolve_run_state(output, depth="deep").completion == "incomplete"


def test_a_process_run_detects_a_changed_export_layer(tmp_path):
    """A process run's finalization input IS its exported layer, so changing
    it must invalidate the proof exactly as a metadata change does for a full
    run."""
    from phenotypic.sdk_ import resolve_run_state
    from tests._output_layout import build_complete_run

    output = build_complete_run(tmp_path, process_only_layer="objmap")
    path, payload = _read_state(output)
    payload["config"]["process_only_layer"] = "rgb"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert resolve_run_state(output, depth="deep").completion == "incomplete"


# ------------------------------------------------- Task 5: advisories (§4.3)


def test_an_unconverted_h5_is_an_advisory_and_never_a_gate(complete_run):
    """Spec §4.3: half-migrated trees contribute an advisory --
    informational, not a gate. Today they reach `contradictory` and flag the
    whole output read-only for a reason the user cannot act on."""
    from phenotypic.sdk_ import resolve_run_state
    from tests._output_layout import FIXTURE_DATASET

    hdf = complete_run / "results" / FIXTURE_DATASET / "hdf"
    hdf.mkdir(parents=True, exist_ok=True)
    (hdf / "legacy.h5").write_bytes(b"\x89HDF\r\n\x1a\n")

    state = resolve_run_state(complete_run, depth="deep")
    assert state.completion == "complete"
    assert any(
        "migrate" in advisory and FIXTURE_DATASET in advisory
        for advisory in state.advisories
    )


def test_a_clean_tree_carries_no_advisories(complete_run):
    """The control for every advisory test: an untouched complete run says
    nothing. Without it, an implementation that always emitted all four
    advisories would pass each of them."""
    from phenotypic.sdk_ import resolve_run_state

    assert resolve_run_state(complete_run, depth="deep").advisories == ()


def test_a_store_built_against_older_metadata_is_an_advisory(complete_run):
    """D-A: stores keep the metadata snapshot they were built against, and
    each store's root records which one. When that differs from the run's
    current metadata_sha256, say so -- derived from what the store already
    carries, never tracked, and never a gate."""
    from phenotypic.sdk_ import resolve_run_state
    from tests._output_layout import FIXTURE_DATASET, FIXTURE_STEMS

    _record_an_older_metadata_snapshot(complete_run, FIXTURE_STEMS[0])

    state = resolve_run_state(complete_run, depth="deep")
    assert state.completion == "complete"
    assert any(
        "metadata" in advisory
        and f"{FIXTURE_DATASET}/{FIXTURE_STEMS[0]}" in advisory
        for advisory in state.advisories
    )


def test_a_matching_metadata_snapshot_raises_no_advisory(complete_run):
    """The other side of the same check: a store built against the run's
    CURRENT snapshot is not a divergence. Without this, an advisory that
    fired on the key's mere presence would pass the test above."""
    from phenotypic.sdk_ import resolve_run_state
    from tests._output_layout import FIXTURE_STEMS

    _, payload = _read_state(complete_run)
    _record_an_older_metadata_snapshot(
        complete_run,
        FIXTURE_STEMS[0],
        digest=payload["config"]["metadata_sha256"],
    )

    assert resolve_run_state(complete_run, depth="deep").advisories == ()


def test_a_migrated_record_is_accepted_on_artifact_validity_alone(
    complete_run,
):
    """U-10. A pre-markers tree has zero work_ids, so migrate marks the
    record rather than fabricating an identity, and such a record is accepted
    with no work_id comparison."""
    from phenotypic.sdk_ import resolve_run_state
    from tests._output_layout import FIXTURE_STEMS

    _mark_migrated(complete_run, FIXTURE_STEMS[0])

    state = resolve_run_state(complete_run, depth="deep")
    assert state.completion == "complete"
    assert state.diagnostics.verified == 2


def test_the_unavailable_fence_is_surfaced_as_an_advisory(complete_run):
    """The visible half of U-10, and it is not optional.

    The ruling accepts a real weakening -- a migrated image is verified on
    artifact validity alone -- on the grounds that it is VISIBLE rather than
    silent. Delete the advisory and the trade the user agreed to is no longer
    the trade being made: what remains is an invisible hole.
    """
    from phenotypic.sdk_ import resolve_run_state
    from tests._output_layout import FIXTURE_DATASET, FIXTURE_STEMS

    _mark_migrated(complete_run, FIXTURE_STEMS[0])

    advisories = resolve_run_state(complete_run, depth="deep").advisories
    assert any(
        "fence" in advisory
        and f"{FIXTURE_DATASET}/{FIXTURE_STEMS[0]}" in advisory
        for advisory in advisories
    ), advisories


def test_an_unmarked_record_is_still_fenced_by_work_id(complete_run):
    """The control for U-10: the acceptance is scoped to marked records.
    Without this, "skip the work_id comparison" could have been implemented
    unconditionally and every test above would still pass."""
    from phenotypic.sdk_ import image_record_path, resolve_run_state
    from tests._output_layout import FIXTURE_DATASET, FIXTURE_STEMS

    path = image_record_path(
        complete_run, FIXTURE_DATASET, FIXTURE_STEMS[0]
    )
    marker = json.loads(path.read_text(encoding="utf-8"))
    marker["work_id"] = "work-id-that-never-existed"
    path.write_text(json.dumps(marker), encoding="utf-8")

    state = resolve_run_state(complete_run, depth="deep")
    assert state.completion == "incomplete"


# ------------------------------------------ Task 5: the sdk_/CLI cross-check


def _tamper_one_overlay(root):
    from phenotypic.sdk_ import dataset_overlays_dir
    from tests._output_layout import FIXTURE_DATASET, FIXTURE_STEMS

    overlay = (
        dataset_overlays_dir(root, FIXTURE_DATASET)
        / f"{FIXTURE_STEMS[0]}.png"
    )
    overlay.write_bytes(overlay.read_bytes() + b"tamper")


def _tamper_one_store_root(root):
    from phenotypic.sdk_ import zarr_store_path
    from phenotypic.sdk_.ngff_ import STORE_ROOT_JSON
    from tests._output_layout import FIXTURE_DATASET, FIXTURE_STEMS

    path = zarr_store_path(root, FIXTURE_DATASET, FIXTURE_STEMS[0])
    path = path / STORE_ROOT_JSON
    path.write_text(
        path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )


def _mark_one_image_migrated(root):
    """Apply U-10's migrated shape to the first image, for the tamper list.

    A one-argument **adapter** over `_mark_migrated`, not a second
    implementation of it: the parametrization below hands a tamper only the
    run root, while `_mark_migrated` takes a stem because its three other
    callers each choose a different image.

    **The one input on which the two readers provably differed**, and the one
    the parametrization excluded. U-10 rules that a migrated record is
    accepted on artifact validity alone -- a pre-markers tree never had a
    `work_id` to match -- so `record_rejection` skips the comparison when
    `provenance == "migrated"`, while `valid_image_success` used to compare
    `work_id` unconditionally with no provenance branch.

    Added and run *before* the fix, and it failed: sdk said `verified`, the
    CLI validator said `False`. That ordering is the point. A test whose
    whole purpose is "the two agree, image by image" was green over four
    tamperings -- untouched, marker-gone, overlay-rewritten,
    store-root-rewritten -- every one of them an input where the two readers
    were never in question. It could not have gone red for the reason its
    own name gives.

    It passes now because `valid_image_success` reads `record_rejection`
    rather than restating it, so there is one implementation to agree with.
    """
    from tests._output_layout import FIXTURE_STEMS

    _mark_migrated(root, FIXTURE_STEMS[0])


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param(_leave_untouched, id="untouched"),
        pytest.param(_remove_one_image_marker, id="marker-gone"),
        pytest.param(_tamper_one_overlay, id="overlay-rewritten"),
        pytest.param(_tamper_one_store_root, id="store-root-rewritten"),
        pytest.param(_mark_one_image_migrated, id="migrated-provenance"),
    ],
)
def test_the_sdk_reader_agrees_with_the_cli_validator(complete_run, tamper):
    """The same tree, the same tamperings, image by image.

    INV-LAYER once forced `_run_state` to re-derive what `valid_image_success`
    decides, because it may not import the CLI half. It no longer does: the
    predicate is `record_rejection` (`sdk_/_image_record`) and the artifact
    walk is `fenced_artifact_path` (`sdk_/_run_state`), both read
    by the CLI function, which is the direction INV-LAYER permits.

    So this test's remit has changed and is worth stating. It no longer keeps
    two implementations honest -- there is one. It pins that the CLI half
    still routes through it, and that the `bool` it returns still tracks the
    sdk verdict image by image. The regression it now catches is someone
    re-inlining a clause here for speed or convenience.

    The shared constants (`SUCCESS_MARKER_VERSION`, the artifact kinds, the
    two proof versions) live in `sdk_/_io_constants` and are imported by both,
    so a version bump cannot desynchronize them either.
    """
    from phenotypic._cli._cli_completion import valid_image_success
    from phenotypic.sdk_ import resolve_run_state

    tamper(complete_run)
    state = resolve_run_state(complete_run, depth="deep")
    assert state.images
    for image in state.images.values():
        assert (image.verdict == "verified") is valid_image_success(
            complete_run,
            dataset=image.dataset,
            image_stem=image.image_stem,
            work_id=image.work_id,
        ), image


def test_the_marker_schema_constants_have_exactly_one_home():
    """The relocation is only worth anything while it stays a relocation.

    `SUCCESS_MARKER_VERSION`, the two artifact kinds and the two proof
    versions moved to `sdk_/_io_constants` because the run-state reader has
    to branch on the same numbers and INV-LAYER forbids it importing the
    writer. `_cli_completion` imports and re-exports them, so its own
    importers -- `_cli_recompile_recovery`, `_cli_recompile_slurm_scripts`,
    `_cli_migrate_image`, `sdk_/_hdf_to_zarr`, and two CLI test modules --
    are unchanged.

    Equality alone would not catch the regression that matters: someone
    "tidying" the import away by re-declaring a literal would leave the two
    values equal on the day of the edit and free to drift afterwards. So this
    asserts **structurally** that `_cli_completion` assigns none of these
    names, which is the property that keeps the relocation real.
    """
    import ast
    from pathlib import Path

    import phenotypic._cli._cli_completion as completion
    from phenotypic.sdk_ import _io_constants

    names = {
        "SUCCESS_MARKER_VERSION",
        "ARTIFACT_KIND_FILE",
        "ARTIFACT_KIND_STORE",
        "AGGREGATE_PROOF_VERSION",
        "RUN_PROOF_VERSION",
    }
    for name in sorted(names):
        assert getattr(completion, name) == getattr(_io_constants, name)

    tree = ast.parse(
        Path(completion.__file__).read_text(encoding="utf-8")
    )
    assigned: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            assigned |= {
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            }
        elif isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        ):
            assigned.add(node.target.id)
    assert not (assigned & names), (
        f"_cli_completion re-declares {sorted(assigned & names)} instead of "
        "importing it; the run-state reader branches on the same value and "
        "cannot see a second copy drift"
    )


def test_an_unverified_image_says_why(incomplete_run):
    """`images` is "work_id -> stages + VERDICT" (spec §9), and the verdict
    carries its reason. A work_id with no marker is an UNVERIFIED ImageState,
    not an absent one -- that is what makes "which images are missing?"
    answerable without re-walking the tree."""
    from phenotypic.sdk_ import resolve_run_state
    from tests._output_layout import FIXTURE_STEMS

    state = resolve_run_state(incomplete_run, depth="deep")
    missing = state.images[f"work-{FIXTURE_STEMS[1]}"]
    assert missing.verdict == "unverified"
    assert missing.reason
    assert state.diagnostics.accepted == 2
    assert state.diagnostics.verified == 1
    assert state.diagnostics.failed == 0


# ------------------------------------------------- Task 6: the shallow path


def _count_sha256(call):
    """Return how many `hashlib.sha256` objects `call()` constructs.

    Every digest in this stack goes through the module attribute at call
    time -- `_digest_file`, `file_fingerprint`, `canonical_digest` and
    `RunIdentity.digest` -- so one patch counts all of them.
    """
    import hashlib

    calls = {"n": 0}
    real = hashlib.sha256

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    hashlib.sha256 = counting  # type: ignore[assignment]
    try:
        call()
    finally:
        hashlib.sha256 = real  # type: ignore[assignment]
    return calls["n"]


def test_shallow_reuse_is_independent_of_the_image_count(tmp_path):
    """Spec §9.2: adding 10 images to 6,000 should cost 6,000 stats and 10
    deep verifications, not 6,000 re-hashes. On a 10,000-image run on GPFS,
    one badge refresh is currently ~10^4 marker reads and 2-3 x 10^4 file
    hashes. Per tab. Every five seconds.

    The claim is asymptotic, so the test is too. A fixed bound on one tree
    ("at most 8 sha256 calls") cannot tell "constant" from "small", and would
    have to be re-tuned every time the run-level proof gains an artifact.
    Two tree sizes can: a warm shallow pass must cost the SAME on both, while
    a deep pass must cost more on the larger one.
    """
    from phenotypic.sdk_ import resolve_run_state
    from tests._output_layout import build_complete_run

    small = build_complete_run(tmp_path / "small", stems=("a", "b"))
    large = build_complete_run(
        tmp_path / "large", stems=("a", "b", "c", "d", "e", "f")
    )

    deep_small = _count_sha256(
        lambda: resolve_run_state(small, depth="deep")
    )
    deep_large = _count_sha256(
        lambda: resolve_run_state(large, depth="deep")
    )
    assert deep_large > deep_small, (
        "a deep pass is not re-hashing per-image artifacts at all"
    )

    shallow_small = _count_sha256(
        lambda: resolve_run_state(small, depth="shallow")
    )
    shallow_large = _count_sha256(
        lambda: resolve_run_state(large, depth="shallow")
    )
    assert shallow_small == shallow_large, (
        f"a warm shallow pass grew with the image count "
        f"({shallow_small} -> {shallow_large}); §9.1's whole point is that "
        "it re-stats instead of re-hashing"
    )
    assert shallow_large < deep_large

    state = resolve_run_state(large, depth="shallow")
    assert state.depth == "shallow"
    assert state.completion == "complete"


def test_a_new_image_escalates_the_whole_resolution(complete_run):
    """A cache miss escalates, and `depth` reports what was performed rather
    than what was asked for. "Mostly shallow" is not a useful third value."""
    from phenotypic.sdk_ import resolve_run_state
    from tests._output_layout import extend_complete_run

    resolve_run_state(complete_run, depth="deep")
    extend_complete_run(complete_run, stem="c")

    state = resolve_run_state(complete_run, depth="shallow")
    assert state.depth == "deep", "a cache miss must escalate"
    assert state.completion == "complete"


def test_shallow_with_a_cold_cache_equals_deep(complete_run):
    from phenotypic.sdk_ import clear_verification_cache, resolve_run_state

    clear_verification_cache()
    cold = resolve_run_state(complete_run, depth="shallow")
    deep = resolve_run_state(complete_run, depth="deep")
    assert cold.completion == deep.completion
    assert set(cold.images) == set(deep.images)
    assert cold.depth == "deep", (
        "a cold shallow call is a deep call, and says so"
    )


def test_a_warm_shallow_pass_reports_the_same_advisories(complete_run):
    """Advisories are projections over `images`, so a reused ImageState must
    carry everything they are derived from. An advisory that only a deep pass
    can emit would flicker on and off with the cache, which is worse than not
    having it."""
    from phenotypic.sdk_ import resolve_run_state
    from tests._output_layout import FIXTURE_STEMS

    _record_an_older_metadata_snapshot(complete_run, FIXTURE_STEMS[0])
    _mark_migrated(complete_run, FIXTURE_STEMS[1])

    deep = resolve_run_state(complete_run, depth="deep")
    shallow = resolve_run_state(complete_run, depth="shallow")
    assert shallow.depth == "shallow"
    assert shallow.advisories == deep.advisories
    assert len(deep.advisories) == 2, deep.advisories


def test_a_finished_lifecycle_record_is_not_a_live_authority(incomplete_run):
    """Rule 2's negative control, and the one every liveness test was missing.

    Every other test here drives `_live_authority` POSITIVELY -- an active
    lifecycle, a running owner. That leaves the predicate itself unproved:
    replace `lifecycle.get("active") is True` with `lifecycle is not None` and
    every one of them still passes.

    `active: False` is not hypothetical. It is what finalize and clear write
    (`_cli_slurm_lifecycle.py:661,679`), so it is the state of every SLURM run
    that has ENDED. Under that mutation a finished run with a failed image
    reports `active` forever: rule 1 cannot fire (clause 1 fails), rule 2 now
    does, and the failure is masked by a worker that is not there.
    """
    import json

    from phenotypic.sdk_ import resolve_run_state, slurm_lifecycle_path

    _mark_slurm_lifecycle_active(incomplete_run)
    path = slurm_lifecycle_path(incomplete_run)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["active"] = False
    path.write_text(json.dumps(record), encoding="utf-8")

    state = resolve_run_state(incomplete_run, depth="deep")
    assert state.completion != "active", (
        "a lifecycle record marked finished still reported live work"
    )


def test_a_terminal_owner_status_is_not_a_live_authority(incomplete_run):
    """The same control for the other predicate.

    `_OWNER_STATUSES_IN_FLIGHT` is `{"running", "submitting"}`; every test
    writes `"running"`, so the membership test is unproved -- widen it to any
    non-empty status and nothing goes red. A GUI that exited cleanly leaves a
    terminal status behind, and believing it pins the verdict at `active`
    exactly as the dead-pid case does, but without a pid to disbelieve.
    """
    import os

    from phenotypic.sdk_ import resolve_run_state

    _write_owner_record(incomplete_run, status="finished", pid=os.getpid())

    state = resolve_run_state(incomplete_run, depth="deep")
    assert state.completion != "active", (
        "a terminal owner status still reported live work"
    )


def test_a_post_u4_run_proof_binds_without_the_aggregate(complete_run):
    """Rule 1's modern branch -- dead today, live the moment P4 lands.

    `_source_set_binding` returns the run proof directly when it carries
    `source_set_digest`, and otherwise follows `publication_id` to the
    aggregate proof. Today's `publish_run_completion_evidence` writes neither
    field, so **only the legacy branch has ever executed** and the modern one
    arrives in P4 unproven.

    The failure that would land is silent and total. `:910-911` reads BOTH
    `source_set_digest` and `source_image_count` off whichever proof the
    binding returns -- so a P4 publishing the digest without the count makes
    the arity check compare `None` to an int, rule 1 stops firing, and **every
    full run reads `incomplete` forever**. That is N-4's shape, which this dual
    read exists to prevent, and the plan's claim of "no window in which the two
    comparisons silently stop being made" is the one part of rule 1 that
    nothing else checks.

    **The aggregate binding is deliberately broken here.** Stamping the run
    proof alone would pass via the legacy branch and prove nothing, so the
    `publication_id` link is severed: `complete` is then reachable ONLY through
    line 930.
    """
    import json

    from phenotypic.sdk_ import (
        aggregate_publication_marker_path,
        resolve_run_state,
        run_completion_marker_path,
    )
    from phenotypic.sdk_._digests import canonical_digest

    before = resolve_run_state(complete_run, depth="deep")
    assert before.completion == "complete", "fixture is not complete"
    verified = sorted(
        work_id
        for work_id, image in before.images.items()
        if image.verdict == "verified"
    )

    agg_path = aggregate_publication_marker_path(complete_run)
    aggregate = json.loads(agg_path.read_text(encoding="utf-8"))
    aggregate["publication_id"] = "severed-so-the-legacy-branch-cannot-bind"
    agg_path.write_text(json.dumps(aggregate), encoding="utf-8")

    path = run_completion_marker_path(complete_run)
    proof = json.loads(path.read_text(encoding="utf-8"))
    assert "source_set_digest" not in proof, (
        "today's publisher already writes source_set_digest -- this test's "
        "premise has expired and it should assert the real shape instead"
    )
    proof["source_set_digest"] = canonical_digest(verified)
    proof["source_image_count"] = len(verified)
    path.write_text(json.dumps(proof), encoding="utf-8")

    state = resolve_run_state(complete_run, depth="deep")
    assert state.completion == "complete", (
        "a run proof carrying its own source-set binding did not satisfy "
        "rule 1 -- the post-U-4 branch is broken and P4 would ship it"
    )


# ---------------------------------------------------------------------------
# The shared state-check helpers (P2 gate, `p2-check-reuse.md`)
#
# Destinations, not migrations: P6 Task 7 moves the ~20 call sites. What these
# pin is that each helper keeps the implementation the gate report named as
# the correct one -- "whichever was easiest to call" being the failure mode a
# consolidation invites.
# ---------------------------------------------------------------------------


def test_run_proof_refuses_a_version_it_cannot_interpret(complete_run):
    """The strictness four open-coded readers dropped.

    `RUN_PROOF_VERSION` is 2 and version-1 proofs exist on trees written by
    an earlier release, so this is reachable today rather than on a schedule.
    `_cli/CLAUDE.md` states the policy as "a version mismatch invalidates
    rather than migrates"; a reader that skips the check certifies a proof
    this build cannot interpret.
    """
    from phenotypic.sdk_ import run_completion_marker_path
    from phenotypic.sdk_._run_state import run_proof

    assert run_proof(complete_run) is not None

    path = run_completion_marker_path(complete_run)
    proof = json.loads(path.read_text(encoding="utf-8"))
    proof["version"] = 1
    path.write_text(json.dumps(proof), encoding="utf-8")

    assert run_proof(complete_run) is None, (
        "run_proof accepted a version-1 proof -- it has picked up the "
        "laxness of the readers it exists to replace"
    )


def test_run_proof_is_current_is_the_bindings_half_not_the_image_walk(
    complete_run,
):
    """The split is the point: structure is O(1), coverage is O(N).

    Deleting every per-image marker leaves the proof's four bindings intact,
    so this must still answer True. If it started walking images it would
    return False here -- and it would stop being the cheap question a GUI
    surface polls, which is what pushed callers into open-coding to begin
    with.
    """
    from phenotypic.sdk_ import image_record_path
    from phenotypic.sdk_._run_state import run_proof_is_current
    from tests._output_layout import FIXTURE_DATASET, FIXTURE_STEMS

    assert run_proof_is_current(complete_run) is True

    for stem in FIXTURE_STEMS:
        image_record_path(
            complete_run, FIXTURE_DATASET, stem
        ).unlink()

    assert run_proof_is_current(complete_run) is True, (
        "run_proof_is_current consulted per-image markers -- it has "
        "absorbed current_run_is_complete, which valid_run_completion keeps "
        "as a separate conjunct"
    )


def test_run_proof_is_current_notices_a_moved_inventory(complete_run):
    """A new image under a rolling input must invalidate the binding."""
    from phenotypic.sdk_._run_state import run_proof_is_current
    from tests._output_layout import FIXTURE_DATASET

    assert run_proof_is_current(complete_run) is True

    state_path, document = _read_state(complete_run)
    document["config"]["work_ids"][FIXTURE_DATASET]["c.tif"] = "work-c"
    state_path.write_text(json.dumps(document), encoding="utf-8")

    assert run_proof_is_current(complete_run) is False, (
        "the accepted inventory moved and the run proof still read current"
    )


def test_the_publishers_digest_is_one_of_the_validators_accepted_ones(
    complete_run,
):
    """F4's pair, and why it is a pair rather than a choice.

    `finalization_input_digest` is what a NEW proof carries -- versioned, one
    spelling. `accepted_finalization_digests` is every spelling a proof
    already on disk may legitimately carry. Today's publisher writes the
    unversioned one, so the two are deliberately *not* equal: keeping only
    the strict one rejects every existing tree, keeping only the tolerant one
    lets a publisher emit a spelling no validator was told about.
    """
    from phenotypic.sdk_ import aggregate_publication_marker_path
    from phenotypic.sdk_._run_state import (
        accepted_finalization_digests,
        finalization_input_digest,
    )

    _, document = _read_state(complete_run)
    config = document["config"]

    accepted = accepted_finalization_digests(config)
    fresh = finalization_input_digest(config)
    assert fresh in accepted

    published = json.loads(
        aggregate_publication_marker_path(complete_run).read_text(
            encoding="utf-8"
        )
    )["finalization_input_digest"]
    assert published in accepted, (
        "the validator would reject the digest today's publisher writes"
    )
    assert published != fresh, (
        "the publisher already writes the versioned spelling -- P4 has "
        "landed, so the unversioned member of the accepted set is now dead "
        "and should be dropped rather than left as a tolerance nothing needs"
    )


def test_a_moved_finalization_input_moves_both_spellings(complete_run):
    """The tolerance weakens nothing, which is why it is safe to keep.

    Both spellings are functions of exactly the same three values, so a
    change to any of them moves the whole accepted set. Accepting two
    spellings is not accepting two answers.
    """
    from phenotypic.sdk_._run_state import accepted_finalization_digests

    _, document = _read_state(complete_run)
    config = dict(document["config"])
    before = accepted_finalization_digests(config)

    config["metadata_sha256"] = "d" * 64
    after = accepted_finalization_digests(config)

    assert before.isdisjoint(after), (
        "a moved metadata snapshot left a spelling the validator still "
        "accepts -- §7.4's late-metadata guarantee is broken"
    )


def test_fenced_artifact_path_dispatches_on_kind(complete_run):
    """F8/F9: the copy that knows about stores is the one that survives.

    `valid_aggregate_snapshot` applies the file predicate unconditionally, so
    a store descriptor would be read as a file and rejected. This helper
    returns the store's root `zarr.json` -- the path to stat next time, not a
    bool -- because a directory's mtime tracks only its own entries.
    """
    from phenotypic.sdk_ import STORE_ROOT_JSON, image_record_path
    from phenotypic.sdk_._run_state import fenced_artifact_path
    from tests._output_layout import FIXTURE_DATASET, FIXTURE_STEMS

    marker = json.loads(
        image_record_path(
            complete_run, FIXTURE_DATASET, FIXTURE_STEMS[0]
        ).read_text(encoding="utf-8")
    )
    root = complete_run.resolve()
    fenced = {
        fenced_artifact_path(root, descriptor)
        for descriptor in marker["artifacts"].values()
    }
    assert None not in fenced, "a published marker failed its own fence"
    assert any(path.endswith(f"/{STORE_ROOT_JSON}") for path in fenced), (
        "no store descriptor fenced on its root zarr.json -- the kind "
        "dispatch is gone and stores are being fenced as directories"
    )
    assert (
        fenced_artifact_path(root, {"path": "x", "kind": "something-new"})
        is None
    ), "an unrecognized kind must fail closed, not default to file"


def test_staged_completeness_reads_the_embedded_table_not_the_legacy_one(
    complete_run,
):
    """F12: the two sides of the SLURM loop stat different files.

    The recovery controller stats
    `results/<ds>/measurements/<stem>.parquet`, which `_cli/CLAUDE.md`
    records as legacy migration input that forward staged runs never write.
    That guard is dead on every run it guards, so a finished image is
    reclassified retryable, excluded from the worker's candidates, and
    terminalized after two wasted array rounds.

    Creating the legacy file must not make this answer True.
    """
    from phenotypic.sdk_ import (
        MEASUREMENT_TABLE_RELATIVE_PATH,
        dataset_measurements_dir,
        zarr_store_path,
    )
    from phenotypic.sdk_._run_state import staged_image_is_complete
    from tests._output_layout import FIXTURE_DATASET, FIXTURE_STEMS

    stem = FIXTURE_STEMS[0]
    embedded = (
        zarr_store_path(complete_run, FIXTURE_DATASET, stem)
        / MEASUREMENT_TABLE_RELATIVE_PATH
    )
    assert embedded.is_file(), "fixture no longer embeds a measurement table"
    assert (
        staged_image_is_complete(
            complete_run,
            FIXTURE_DATASET,
            stem,
            markers_required=False,
            resume=True,
        )
        is True
    )

    embedded.unlink()
    legacy = dataset_measurements_dir(complete_run, FIXTURE_DATASET)
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / f"{stem}.parquet").write_bytes(b"")

    assert (
        staged_image_is_complete(
            complete_run,
            FIXTURE_DATASET,
            stem,
            markers_required=False,
            resume=True,
        )
        is False
    ), (
        "the legacy per-image parquet satisfied the staged completeness "
        "guard -- the controller's pre-OME-Zarr path has been kept"
    )


def test_staged_completeness_defers_to_the_marker_when_markers_are_on(
    complete_run,
):
    """`markers_required=True` means "the marker is the signal, not this".

    Returning True here would let a file stat certify an image the Stage-3
    marker has not, which is the unsafe direction: the conservative answer
    routes a doubtful image back through a stage.
    """
    from phenotypic.sdk_._run_state import staged_image_is_complete
    from tests._output_layout import FIXTURE_DATASET, FIXTURE_STEMS

    for markers_required, resume in ((True, True), (False, False)):
        assert (
            staged_image_is_complete(
                complete_run,
                FIXTURE_DATASET,
                FIXTURE_STEMS[0],
                markers_required=markers_required,
                resume=resume,
            )
            is False
        ), f"markers_required={markers_required}, resume={resume}"


def test_the_dead_second_worklist_definition_is_gone():
    """F11: deletion, not relocation.

    `_cli_update_state.get_remaining_images` derived the remaining set from
    `datasets.{completed,failed}` -- fields spec §4.2 removes from the state
    file -- and had no callers. Once those fields are gone it would answer
    "everything remains" for every run, so relocating it into a shared module
    would have laundered dead code as current.
    """
    from phenotypic._cli import _cli_update_state

    assert not hasattr(_cli_update_state, "get_remaining_images"), (
        "the dead worklist definition is back; the live one is "
        "_cli_state_management.get_remaining_images_for_datasets"
    )
