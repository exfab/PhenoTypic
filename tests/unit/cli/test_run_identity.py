"""`restart_epoch`: the one tracked counter, and the fence it buys.

Spec §5.1 D4. Three groups, and the middle one is the reason the first exists.

**The counter** (`read_restart_epoch` / `bump_restart_epoch`) is the only
tracked value this design adds, and it is worthless if a restart resets it --
the whole point is to distinguish "deliberately fresh attempt" from "same
config again", which is exactly what a restart is.

**Rule 2's first half** (`_live_authority`): an authority counts only when it
reports work in flight *for the current identity*. P1 shipped the second half
-- the pid probe -- and could not build the first, because before this counter
existed the identity and the authority were read from the same file and the
comparison would have been a value against itself.

**The asymmetry** is deliberate and tested in both directions: the lifecycle
record is epoch-fenced, the GUI owner record is not. See
`test_a_live_gui_owner_still_reports_active_across_a_restart` for why that is
a decision rather than an omission.

**A note on what Task 1 does and does not write.** `bump_restart_epoch`
persists `.phenotypic/restart_epoch.json`; `processing_state.json`'s
`config.restart_epoch` -- which is what `RunIdentity.restart_epoch` reads --
is written by P2 Task 3's minting. So the fence tests here set the config
value explicitly rather than expecting a bump to move it. That the two homes
can differ is the design (CONFLICT-1), not a gap in the fixture: one is *the
counter*, the other is *the value this state was minted under*.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from phenotypic._cli._cli_failure_tracker import file_sha256
from phenotypic._cli._cli_identity import (
    bump_restart_epoch,
    read_restart_epoch,
)
from phenotypic.sdk_ import (
    clear_machine_state,
    gui_launch_owner_path,
    phenotypic_cache_dir,
    resolve_processing_state_path,
    resolve_run_state,
    restart_epoch_path,
    slurm_lifecycle_path,
)


@pytest.fixture
def incomplete_run(tmp_path):
    """A run whose verdict is `incomplete` unless an authority says otherwise.

    `incomplete` is the discriminator every fence test here needs: with a live
    authority the ladder returns `active`, and without one it falls through.
    A complete run would report `complete` either way and prove nothing.
    """
    from tests._output_layout import build_incomplete_run

    return build_incomplete_run(tmp_path)


def _set_config_restart_epoch(output_dir: Path, epoch: int) -> None:
    """Write `config.restart_epoch`, which P2 Task 3's minting will write.

    Edited directly rather than through a helper so this file does not depend
    on a writer that does not exist yet, and so the value under test is
    visible in the test that sets it.
    """
    path = resolve_processing_state_path(output_dir)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["config"]["restart_epoch"] = epoch
    path.write_text(json.dumps(document), encoding="utf-8")


def _state_config(output_dir: Path) -> dict:
    return json.loads(
        resolve_processing_state_path(output_dir).read_text(encoding="utf-8")
    )["config"]


def _seed_converted_stores(output_dir: Path, *stems: str) -> None:
    """Seed the shape `_ensure_migration_processing_state` synthesizes from.

    **Stores, and deliberately NO processing state.** That function returns
    early when state already exists (`existing is not None`), so seeding state
    would skip the generation block entirely and the test would assert
    nothing. It builds `work_ids` by scanning
    `results/<ds>/zarr/*.ome.zarr/zarr.json`, which is the only thing it
    checks for, so an empty JSON file per store is a sufficient stand-in.
    """
    for stem in stems:
        store = (
            output_dir / "results" / "plate" / "zarr" / f"{stem}.ome.zarr"
        )
        store.mkdir(parents=True, exist_ok=True)
        (store / "zarr.json").write_text("{}", encoding="utf-8")


def _publish_lifecycle(output_dir: Path, *, generation: str) -> dict:
    from phenotypic._cli._cli_slurm_lifecycle import (
        initialize_slurm_lifecycle,
    )

    return initialize_slurm_lifecycle(
        output_dir, generation=generation, mode="slurm"
    )


def _lifecycle_record(output_dir: Path) -> dict:
    return json.loads(
        slurm_lifecycle_path(output_dir).read_text(encoding="utf-8")
    )


def _claim_as_live_gui(output_dir: Path) -> None:
    """Write a GUI owner record naming THIS process, which is alive.

    `os.getpid()` rather than a fabricated number: `_live_authority` probes
    the pid, so a record naming a dead process is refused by rule 2's *second*
    half and would make the epoch question unreachable.
    """
    path = gui_launch_owner_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"status": "running", "pid": os.getpid(), "generation": "gui-1"}
        ),
        encoding="utf-8",
    )


# -------------------------------------------------- the digest, D-C / §5.4


def test_per_image_config_digest_is_the_work_id_digest_itself():
    """D-C / §5.4: "not a new digest ... reused verbatim".

    §5.4's argument is that if the generation and `work_id` could disagree
    about what counts as scientific configuration, a change could invalidate
    per-image proofs without minting a new generation, or the reverse.
    Identity is the strongest form of agreement available, so this is an `is`
    check rather than an equality one: an equal-but-separate function is equal
    today and one edit away from not being, and nothing would fail at the
    moment it stopped.
    """
    from phenotypic._cli._cli_failure_tracker import (
        processing_configuration_digest,
    )
    from phenotypic._cli._cli_identity import per_image_config_digest

    assert per_image_config_digest is processing_configuration_digest


def test_the_proof_side_digest_is_a_different_value_entirely():
    """The rename's reason, pinned so nobody re-merges the two.

    `scientific_config_digest` is already taken, by the pipeline FILE's
    digest: the proofs write `state.config["pipeline_sha256"]` under that key
    and `RunIdentity` reads it back. `per_image_config_digest` contains no
    pipeline bytes, so the two cannot be substituted for one another.

    The failure this guards is a migration wearing a rename's clothes.
    "Unifying" them by pointing the proofs at the per-image digest rewrites
    the value in every aggregate and run proof on disk, and every previously
    complete run reads `incomplete` until it is re-finalized.
    """
    from phenotypic._cli._cli_failure_tracker import (
        processing_configuration_digest_from_values,
    )

    payload_fields = (
        processing_configuration_digest_from_values.__code__.co_varnames
    )

    assert "pipeline_sha256" not in payload_fields, (
        "the per-image digest grew a pipeline component; it is now capable "
        "of answering the proof-side digest's question and the two will be "
        "merged by the next reader who notices"
    )


# ------------------------------------- the content-derived generation (D3/D7)


def test_the_same_components_mint_the_same_generation():
    """D3: same inputs -> same token.

    This is what makes resume and fencing **emergent** rather than
    bookkeeping. Two invocations with the same configuration mint the same
    generation without either having read the other's state -- which is
    exactly what lets a SLURM worker starting cold fence itself correctly
    against a run it has never seen. A `uuid4()` cannot do that, and every
    site this replaces used one.
    """
    from phenotypic._cli._cli_identity import derive_processing_generation

    kwargs = dict(
        pipeline_sha256="pipe-1", per_image_config="cfg-1", restart_epoch=0
    )

    assert derive_processing_generation(
        **kwargs
    ) == derive_processing_generation(**kwargs)


@pytest.mark.parametrize(
    "field, value",
    [
        ("pipeline_sha256", "pipe-2"),
        ("per_image_config", "cfg-2"),
        ("restart_epoch", 1),
    ],
)
def test_every_component_moves_the_generation(field, value):
    """Each of the three is load-bearing, checked one at a time.

    A component folded in but never actually reaching the digest is invisible
    from a same-inputs-same-token test, which passes just as well when the
    function ignores two of its three arguments.

    Short stand-ins rather than 64-char digests: the function does not parse
    its inputs, so a realistic-looking digest would buy nothing and makes the
    parametrized test ids unreadable.
    """
    from phenotypic._cli._cli_identity import derive_processing_generation

    base = dict(
        pipeline_sha256="pipe-1", per_image_config="cfg-1", restart_epoch=0
    )
    moved = {**base, field: value}

    assert derive_processing_generation(
        **base
    ) != derive_processing_generation(**moved)


def test_an_absent_component_is_the_empty_one():
    """`None` and `""` are the same input, so a pipeline-less run still mints
    a stable generation rather than a different one each time the caller
    happens to pass a different spelling of "nothing"."""
    from phenotypic._cli._cli_identity import derive_processing_generation

    assert derive_processing_generation(
        pipeline_sha256=None, per_image_config=None, restart_epoch=0
    ) == derive_processing_generation(
        pipeline_sha256="", per_image_config="", restart_epoch=0
    )


def test_a_migrated_trees_generation_ignores_its_inventory(tmp_path):
    """D7, at the site that violated it — CAN-7.

    `_ensure_migration_processing_state` folded the full
    `dataset/stem:work_id` listing into the generation from `dd18d9c7`
    (2026-08-26), **eight days before D7 was written**. Every migrated tree
    therefore behaved the way D7 exists to prevent: each new image under a
    rolling input changed the generation, resetting live progress and fencing
    in-flight workers.

    This is a **pre-existing defect**, not a regression introduced by the
    change that names it. Left unfixed it would surface in P5's rolling-input
    matrix and look like a bug in P5 rather than an unrevised writer.
    """
    from phenotypic._cli._cli_migrate import (
        _ensure_migration_processing_state,
    )

    def _generation_for(root: Path, *stems: str) -> str:
        _seed_converted_stores(root, *stems)
        _ensure_migration_processing_state(root)
        return _state_config(root)["processing_generation"]

    one = _generation_for(tmp_path / "one", "a")
    two = _generation_for(tmp_path / "two", "a", "b")

    assert one == two, (
        "an image arrival changed a migrated tree's generation; D7 keeps "
        "inventory out of the generation precisely so it cannot"
    )


def test_a_migrated_tree_records_a_restart_epoch(tmp_path):
    """CAN-7 + CAN-11. The migrator wrote `work_ids` with **no**
    `restart_epoch`, which is P1's `requires_conversion` signal 4 — so a
    freshly migrated tree was refused by the very next `--mode full`. The gate
    firing on its own migrator's output."""
    from phenotypic._cli._cli_migrate import (
        _ensure_migration_processing_state,
    )

    root = tmp_path / "run"
    _seed_converted_stores(root, "a")
    _ensure_migration_processing_state(root)

    assert _state_config(root)["restart_epoch"] == 0


# -------------------------------------------- mint_run_identity (CAN-21, E4)


@pytest.fixture
def minted(tmp_path, make_exec_config):
    """A factory returning a FRESH config each call, and its identity.

    Fresh because `mint_run_identity` sets a once-only flag on the config
    object (CAN-21), so a fixture handing back one shared config would make
    the second call in any test raise for a reason unrelated to that test.
    """

    def _mint(*, restart: bool = False, **overrides):
        root = tmp_path / "run"
        root.mkdir(exist_ok=True)
        phenotypic_cache_dir(root).mkdir(parents=True, exist_ok=True)
        pipeline = overrides.pop("pipeline_json", None)
        if pipeline is None:
            pipeline = tmp_path / "pipeline.json"
            if not pipeline.is_file():
                pipeline.write_text("{}", encoding="utf-8")
        config = make_exec_config(
            pipeline_json=pipeline,
            input_path=tmp_path / "input",
            output_dir=root,
            **overrides,
        )
        from phenotypic._cli._cli_identity import mint_run_identity

        return config, mint_run_identity(config, restart=restart)

    return _mint


def test_minting_twice_in_one_invocation_is_a_programming_error(minted):
    """CAN-21. Two mints in one run give it two generations and burn an
    epoch. `ExecutionConfig.output_dir` is `Optional[Path]`, so this cannot
    make itself idempotent by re-reading — the guard is a loud failure, and
    the rule it enforces is structural: the entry point mints once and threads
    the `RunIdentity` down."""
    from phenotypic._cli._cli_identity import mint_run_identity

    config, _ = minted()

    with pytest.raises(RuntimeError, match="already minted"):
        mint_run_identity(config, restart=True)


def test_a_resume_does_not_bump_the_epoch_but_a_restart_does(minted):
    """**A resume is not a restart**, which is the conclusion the rewritten
    `phenotypicCLI.py:2422` comment records.

    A resume's configuration has not changed, so it mints the *same*
    generation — that is what D3 is for. If a resume bumped, every resume
    would fence its own in-flight workers, which is the failure D5 exists to
    prevent and the opposite of what a resume is for.
    """
    _, first = minted(restart=False)
    _, resumed = minted(restart=False)
    _, restarted = minted(restart=True)

    assert resumed.restart_epoch == first.restart_epoch
    assert resumed.processing_generation == first.processing_generation

    assert restarted.restart_epoch == first.restart_epoch + 1
    assert restarted.processing_generation != first.processing_generation


def test_a_pipeline_edit_mints_a_new_generation(tmp_path, minted):
    """The pipeline is a generation component, threaded from the config."""
    pipeline = tmp_path / "edited.json"
    pipeline.write_text("{}", encoding="utf-8")
    _, before = minted(pipeline_json=pipeline)

    pipeline.write_text('{"changed": true}', encoding="utf-8")
    _, after = minted(pipeline_json=pipeline)

    assert after.processing_generation != before.processing_generation


def test_a_per_image_config_change_mints_a_new_generation(minted):
    """**Category E #4, and the reason this test exists separately.**

    `mint_run_identity` accepts an `ExecutionConfig` and threads several
    components into the digest. A component accepted and never folded in
    leaves the signature correct, mypy silent, and
    `derive_processing_generation`'s own tests passing — because those test
    the primitive, not the caller. Only a test that moves a config field and
    watches the minted generation can see the difference.
    """
    _, before = minted(detect_mode="gray")
    _, after = minted(detect_mode="rgb")

    assert after.processing_generation != before.processing_generation


def test_the_minted_proof_side_digest_is_the_pipeline_digest(
    tmp_path, minted
):
    """Category E #4 again, on the component most likely to be mis-threaded.

    `RunIdentity.scientific_config_digest` is the **proof-side** token (A) --
    the pipeline file's bytes -- not `per_image_config_digest` (B). Writing B
    here would make every proof this run publishes disagree with every proof
    already on disk, and nothing else in this suite would notice.
    """
    import hashlib

    pipeline = tmp_path / "proof.json"
    pipeline.write_text('{"ops": []}', encoding="utf-8")
    config, identity = minted(pipeline_json=pipeline)

    from phenotypic._cli._cli_identity import per_image_config_digest

    expected = hashlib.sha256(pipeline.read_bytes()).hexdigest()

    assert identity.scientific_config_digest == expected
    assert identity.scientific_config_digest != per_image_config_digest(
        config
    )


# ------------------------------------------- the wiring (increment 3)


def test_create_initial_state_records_the_minted_identity(
    tmp_path, make_exec_config
):
    """The generation and the epoch both land in `config`, from the identity.

    `create_initial_state` minted its own `uuid4().hex` before this change.
    Taking the identity as a **required** keyword is what stops the five
    minting sites this task consolidates from quietly becoming six: an
    optional one would let a caller fall back to a uuid with nothing failing.
    """
    from phenotypic._cli._cli_identity import mint_run_identity
    from phenotypic._cli._cli_state_management import create_initial_state

    root = tmp_path / "run"
    phenotypic_cache_dir(root).mkdir(parents=True)
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text("{}", encoding="utf-8")
    config = make_exec_config(
        pipeline_json=pipeline,
        input_path=tmp_path / "input",
        output_dir=root,
    )
    identity = mint_run_identity(config, restart=False)

    state = create_initial_state(config, [], root, identity=identity)

    assert state.config["processing_generation"] == (
        identity.processing_generation
    )
    assert state.config["restart_epoch"] == identity.restart_epoch


def test_the_state_generation_is_not_a_uuid(tmp_path, make_exec_config):
    """D3, at the site that used to break it.

    A `uuid4().hex` is 32 lowercase hex characters; the content-derived
    generation is a sha256 and is 64. Asserting the *length* rather than a
    specific digest keeps this a test about "derived, not random" rather than
    a golden value that has to be updated whenever a component moves.
    """
    from phenotypic._cli._cli_identity import mint_run_identity
    from phenotypic._cli._cli_state_management import create_initial_state

    root = tmp_path / "run"
    phenotypic_cache_dir(root).mkdir(parents=True)
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text("{}", encoding="utf-8")
    config = make_exec_config(
        pipeline_json=pipeline,
        input_path=tmp_path / "input",
        output_dir=root,
    )

    state = create_initial_state(
        config, [], root, identity=mint_run_identity(config, restart=False)
    )

    assert len(str(state.config["processing_generation"])) == 64


def test_the_two_metadata_digests_agree(tmp_path, make_exec_config):
    """**The cross-module agreement test.** Category E, not spec drift.

    `mint_run_identity` must recompute `metadata_sha256` rather than read it,
    because `phenotypicCLI.py` stamps that value into the state only *after*
    state creation -- at mint time there is nothing to read. That constraint
    is real and forced; it is not a deviation and owes no experiment.

    The **risk** it creates is that the two computations drift apart. If they
    do, the minted `finalization_input_digest` disagrees with the one every
    later reader derives from the state, and §7.4's late-metadata guarantee
    fires on **every** run instead of on a real edit -- a re-finalize per
    invocation, forever, with nothing failing.

    A docstring saying "keep these identical" is prevention with no
    detection. This is the detection: compute both ways over one input and
    require them equal.
    """
    from phenotypic._cli._cli_identity import _metadata_digest_for
    from phenotypic.phenotypicCLI import _snapshot_metadata_csv
    from phenotypic.sdk_ import metadata_csv_deliverable_path

    root = tmp_path / "run"
    phenotypic_cache_dir(root).mkdir(parents=True)
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "Metadata_Well,Metadata_Strain\nA1,wt\n", encoding="utf-8"
    )
    config = make_exec_config(
        pipeline_json=tmp_path / "pipeline.json",
        input_path=tmp_path / "input",
        output_dir=root,
        metadata_csv=metadata,
    )

    # Drive the CLI's OWN copier rather than restating its arithmetic. An
    # earlier version of this test hand-copied the one-line sha256 and called
    # that "the CLI side" -- so it constrained only `_metadata_digest_for`,
    # the half already under a docstring, and stayed green through every
    # drift it named.
    _snapshot_metadata_csv(root, metadata)
    snapshot = metadata_csv_deliverable_path(root)

    assert _metadata_digest_for(config) == file_sha256(snapshot)

    # ARM 2 -- the continuation, and the case that used to fail. No
    # `--metadata` is re-passed, so `config.metadata_csv` is None and the
    # digest must come from the existing snapshot rather than reading None.
    continued = make_exec_config(
        pipeline_json=tmp_path / "pipeline.json",
        input_path=tmp_path / "input",
        output_dir=root,
        metadata_csv=None,
    )

    assert _metadata_digest_for(continued) == file_sha256(snapshot)

    # ARM 3 -- the guard. Measure and process runs SKIP the snapshot, so
    # falling back to it would invent a digest for a file those modes never
    # wrote. The two arms fail in opposite directions; the guard is what a
    # later edit will drop, so it is what this arm defends.
    measuring = make_exec_config(
        pipeline_json=tmp_path / "pipeline.json",
        input_path=tmp_path / "input",
        output_dir=root,
        metadata_csv=None,
        measure_only=True,
    )

    assert _metadata_digest_for(measuring) is None


# --------------------------------------------- D5 and mode parity (3b)


def _config_for(tmp_path, make_exec_config, **overrides):
    """A config over a real input image, for `work_id_for_image`."""
    root = tmp_path / "run"
    phenotypic_cache_dir(root).mkdir(parents=True, exist_ok=True)
    inputs = tmp_path / "input"
    inputs.mkdir(exist_ok=True)
    image = inputs / "a.tiff"
    if not image.is_file():
        image.write_bytes(b"pixels")
    pipeline = tmp_path / "pipeline.json"
    if not pipeline.is_file():
        pipeline.write_text("{}", encoding="utf-8")
    config = make_exec_config(
        pipeline_json=pipeline,
        input_path=inputs,
        output_dir=root,
        **overrides,
    )
    return config, image


def test_a_restart_moves_the_generation_but_not_any_work_id(
    tmp_path, make_exec_config
):
    """**D5**: the epoch fixes the stale-worker hazard *without* turning
    ``--restart`` into ``--overwrite``.

    Continuation skips an image whose ``work_id`` already carries a valid
    marker, so a ``work_id`` that moved on restart would reprocess every
    surviving store from zero -- exactly what D5 forbids. The two values must
    therefore move **independently**: the generation changes so stale workers
    are fenced, and the work_id does not, so finished images are still reused.

    Asserting both in one test is the point. Either alone is satisfied by a
    broken implementation -- one where nothing changes, or one where
    everything does.

    Structural today, because ``ExecutionConfig`` carries no ``restart_epoch``
    field for ``work_id_for_image`` to see. The leak the plan warns about
    would arrive by someone *adding* one, which is a one-line change nothing
    else here would catch.
    """
    from phenotypic._cli._cli_failure_tracker import work_id_for_image
    from phenotypic._cli._cli_identity import mint_run_identity

    before_config, image = _config_for(tmp_path, make_exec_config)
    before_work_id = work_id_for_image(before_config, "plate", image)[0]
    before_gen = mint_run_identity(
        before_config, restart=False
    ).processing_generation

    # A second config object: the mint-once guard is per invocation, and a
    # restart IS a second invocation.
    after_config, _ = _config_for(tmp_path, make_exec_config)
    after_gen = mint_run_identity(
        after_config, restart=True
    ).processing_generation
    after_work_id = work_id_for_image(after_config, "plate", image)[0]

    assert after_gen != before_gen, (
        "a restart did not mint a new generation, so a pre-restart worker "
        "would still pass the fence"
    )
    assert after_work_id == before_work_id, (
        "the restart epoch leaked into work_id; --restart would reprocess "
        "every surviving store, which is D5's whole prohibition"
    )


def test_measure_mints_the_identity_a_full_run_would(
    tmp_path, make_exec_config
):
    """DF-16 / CAN-20: measure runs under the SAME content-derived identity.

    Measure mode minted its own ``uuid4()`` and could therefore never match
    the run it was measuring. §7.4 routes measure through ``finalize_run`` and
    P4 Task 4 parametrizes a byte-identical master over
    ``["full", "measure", "recompile"]`` -- none of which is coherent if a
    measure invocation has an identity of its own.

    D3 supplies the mechanism: the pipeline and the per-image configuration
    are unchanged, so the generation is the same *value*.
    """
    from phenotypic._cli._cli_identity import mint_run_identity

    full_config, _ = _config_for(tmp_path, make_exec_config)
    measure_config, _ = _config_for(
        tmp_path, make_exec_config, measure_only=True
    )

    assert (
        mint_run_identity(measure_config, restart=False).processing_generation
        == mint_run_identity(full_config, restart=False).processing_generation
    )


def test_process_mints_a_DIFFERENT_identity_and_that_is_correct(
    tmp_path, make_exec_config
):
    """**The plan's own parametrized test would fail here, and should.**

    Task 3 Step 5 specifies
    ``@pytest.mark.parametrize("mode", ["full", "measure", "process"])`` with
    every mode asserted equal to ``full``. That holds for ``measure`` and is
    **wrong for ``process``**: ``process_only_layer`` is a per-image
    configuration field, and ``processing_configuration_digest_from_values``
    branches on it (`_cli_failure_tracker.py:216-233`) -- a process run
    digests ``{process_only_layer, ext, process_format}`` where a full run
    digests ``{include_dataset_column, overlay_alpha, save_overlays}``.

    Different payload, different digest, different generation -- and that is
    the **right** answer, not a defect to paper over. A process run exports
    one layer and publishes no master; a full run does neither. Forcing them
    to share a generation would say two genuinely different configurations
    were the same one, which is what the generation exists to deny.

    So DF-16's "same statement" for ``--mode process`` cannot be "same
    generation as full". It is: a process run mints the generation **its own**
    configuration implies, by the same rule as every other mode.
    """
    from phenotypic._cli._cli_identity import mint_run_identity

    full_config, _ = _config_for(tmp_path, make_exec_config)
    process_config, _ = _config_for(
        tmp_path, make_exec_config, process_only_layer="rgb"
    )

    assert (
        mint_run_identity(
            process_config, restart=False
        ).processing_generation
        != mint_run_identity(
            full_config, restart=False
        ).processing_generation
    )


# ------------------------------------- the event-log generation fence (§14)


def _log_event(output_dir: Path, *, generation: str, image: str) -> None:
    """Append one completed event tagged with ``generation``."""
    from phenotypic._cli._cli_update_state import append_event
    from phenotypic.sdk_ import resolve_event_log_path

    log = resolve_event_log_path(output_dir)
    log.parent.mkdir(parents=True, exist_ok=True)
    append_event(
        log,
        "plate",
        image,
        "completed",  # type: ignore[arg-type]
        generation=generation,
    )


def _aggregate_under(output_dir: Path, generation: str) -> set[str]:
    """Return the images counted completed under ``generation``."""
    from phenotypic._cli._cli_update_state import aggregate_state_from_events
    from phenotypic.sdk_ import resolve_event_log_path

    states = aggregate_state_from_events(
        resolve_event_log_path(output_dir),
        inventory={"plate": {"a.tif"}},
        generation=generation,
    )
    return set(states["plate"].completed) if "plate" in states else set()


def test_a_restart_excludes_events_from_the_previous_generation(tmp_path):
    """Spec §14: a worker holding the pre-restart generation must not have
    its **events counted**.

    The fence itself is not new -- `aggregate_state_from_events` has always
    ignored events tagged with another generation. What P2 changed is the
    *value* it fences on: before `3220a740` the generation was a `uuid4()`
    that churned on every invocation, so this fired constantly. Now it fires
    on a restart and only on a restart, which is what makes it a fence rather
    than an eraser.

    Nothing pinned this before. It is the half a reader would assume is
    covered.
    """
    _log_event(tmp_path, generation="gen-before", image="a.tif")

    assert _aggregate_under(tmp_path, "gen-after-restart") == set(), (
        "an event tagged with the pre-restart generation was counted; a "
        "worker the restart abandoned is still reporting progress"
    )


def test_a_resume_counts_events_from_its_own_generation(tmp_path):
    """**The half that changed, and the half nothing defends.**

    A resume mints the *same* generation (D3), so its prior events are its
    own and must be counted. Before `3220a740` a resume minted a fresh uuid,
    every prior event read as "other generation", and the event log was
    discarded on every resume -- silently defeating the merge point's own
    stated design, which is `prefer event log as source of truth`.

    That change shipped inside a commit about minting and was accepted by
    the user after the fact, on the evidence that the failure direction is
    safe: the work list is `completed | failed`, `started` is not in it, so a
    stale marker costs reprocessing and never a wrongly skipped image.

    **A regression here is silent and cheap to introduce** -- it looks like
    restoring a fence rather than deleting one. This test is what makes it
    loud. Paired with the restart test above on purpose: either alone is
    satisfied by an implementation that counts everything, or nothing.
    """
    _log_event(tmp_path, generation="gen-stable", image="a.tif")

    assert _aggregate_under(tmp_path, "gen-stable") == {"a.tif"}, (
        "a resume discarded its own prior events; the generation is stable "
        "across a resume (D3) precisely so they are kept"
    )


# ------------------------------------------------------------- the counter


def test_restart_epoch_survives_clear_machine_state(tmp_path):
    """D4. The counter is worthless if a restart resets it: the whole point is
    to distinguish "deliberately fresh attempt" from "same config again", and
    a restart is exactly the first."""
    phenotypic_cache_dir(tmp_path).mkdir(parents=True)
    assert read_restart_epoch(tmp_path) == 0
    assert bump_restart_epoch(tmp_path) == 1
    assert bump_restart_epoch(tmp_path) == 2

    clear_machine_state(tmp_path)

    assert read_restart_epoch(tmp_path) == 2, (
        "clear_machine_state destroyed the restart epoch; the fence it exists "
        "for cannot survive the operation it exists to fence"
    )


def test_reading_a_corrupt_restart_epoch_is_zero_not_an_error(tmp_path):
    """INV-VERDICT's degrade half. A restart must not be blocked by an
    unparseable counter -- reading 0 understates the restarts and so fails to
    fence a stale worker, which is the pre-counter status quo. Raising would
    make one bad byte a reason the user cannot restart at all."""
    phenotypic_cache_dir(tmp_path).mkdir(parents=True)
    restart_epoch_path(tmp_path).write_text("{not json", encoding="utf-8")

    assert read_restart_epoch(tmp_path) == 0


def test_a_boolean_is_not_a_restart_epoch(tmp_path):
    """`True` is an `int` in Python, so an unguarded reader accepts it as
    epoch 1 -- a fence silently advanced by a type error."""
    phenotypic_cache_dir(tmp_path).mkdir(parents=True)
    restart_epoch_path(tmp_path).write_text(
        json.dumps({"restart_epoch": True}), encoding="utf-8"
    )

    assert read_restart_epoch(tmp_path) == 0


def test_a_failed_write_raises_rather_than_returning_quietly(tmp_path):
    """The asymmetry with `read_restart_epoch`, which degrades to 0.

    A silently swallowed write failure is *worse* than the pre-counter status
    quo: the next invocation reads the stale epoch, mints the generation the
    abandoned workers are already holding, and the fence passes for exactly
    the workers it exists to exclude. Reading a missing fence is recoverable;
    failing to write one is not.
    """
    cache = phenotypic_cache_dir(tmp_path)
    cache.mkdir(parents=True)
    os.chmod(cache, 0o500)
    try:
        if os.access(cache, os.W_OK):
            pytest.skip("cannot drop write permission here (running as root?)")

        with pytest.raises(OSError):
            bump_restart_epoch(tmp_path)
    finally:
        os.chmod(cache, 0o700)


# ------------------------------------------------- the writer, on its own


def test_the_lifecycle_record_carries_the_epoch_current_at_publication(
    tmp_path,
):
    """The writer's half of rule 2, pinned WITHOUT going through the fence.

    If this only held via `_live_authority`, a later change to the reader
    could make it vacuous and nothing would say so. The epoch is read at
    publication rather than passed in precisely so it cannot be a caller's
    belief -- so the test bumps the counter *between* two publications and
    asserts the second record moved with it.
    """
    phenotypic_cache_dir(tmp_path).mkdir(parents=True)
    _publish_lifecycle(tmp_path, generation="gen-1")
    assert _lifecycle_record(tmp_path)["restart_epoch"] == 0

    bump_restart_epoch(tmp_path)
    slurm_lifecycle_path(tmp_path).unlink()
    _publish_lifecycle(tmp_path, generation="gen-2")

    assert _lifecycle_record(tmp_path)["restart_epoch"] == 1


def test_an_existing_active_fence_is_not_re_dated(tmp_path):
    """Re-publishing the same generation returns the standing record.

    Re-stamping would silently re-date an old fence to the current epoch,
    which is the precise failure the fence exists to prevent: a worker from
    before the restart would look current again.
    """
    phenotypic_cache_dir(tmp_path).mkdir(parents=True)
    _publish_lifecycle(tmp_path, generation="gen-1")
    bump_restart_epoch(tmp_path)

    _publish_lifecycle(tmp_path, generation="gen-1")

    assert _lifecycle_record(tmp_path)["restart_epoch"] == 0


# ------------------------------------------------------ rule 2, first half


def test_a_current_authority_still_reports_the_run_active(incomplete_run):
    """The control. Every other fence test asserts something is refused, and
    all of them would pass against an implementation that refused
    everything -- including the one that reports no run as active, ever."""
    _publish_lifecycle(incomplete_run, generation="gen-1")
    _set_config_restart_epoch(incomplete_run, 0)

    state = resolve_run_state(incomplete_run, depth="deep")

    assert state.completion == "active"


def test_a_pre_restart_authority_does_not_report_the_run_active(
    incomplete_run,
):
    """Rule 2's first half, and the failure it excludes.

    A `--restart` mints a new epoch; a worker from the previous epoch is
    still draining and its lifecycle record still says `active`. Without the
    fence, rule 2 fires and the run looks alive on the strength of a worker
    the restart already abandoned -- a stale authority outranking a valid
    verdict, in the one direction P1 could not construct.
    """
    _publish_lifecycle(incomplete_run, generation="gen-1")
    _set_config_restart_epoch(incomplete_run, 1)

    state = resolve_run_state(incomplete_run, depth="deep")

    assert state.completion != "active", (
        "a lifecycle record from a superseded epoch reported the run alive"
    )


def test_a_record_without_an_epoch_still_counts_on_an_unrestarted_run(
    incomplete_run,
):
    """Backward compatibility, in the direction that must not regress.

    Records written before this field existed read as epoch 0. On a
    never-restarted run that is still current, so an existing SLURM launch
    must not be fenced by an upgrade.

    ``pop(..., None)`` rather than ``del``: the subject is *a record with no
    epoch*, and a ``del`` would raise -- coupling this test to the writer
    still stamping the field, which is a different test's job.
    """
    _publish_lifecycle(incomplete_run, generation="gen-1")
    record = _lifecycle_record(incomplete_run)
    record.pop("restart_epoch", None)
    slurm_lifecycle_path(incomplete_run).write_text(
        json.dumps(record), encoding="utf-8"
    )
    _set_config_restart_epoch(incomplete_run, 0)

    assert resolve_run_state(incomplete_run, depth="deep").completion == (
        "active"
    )


def test_a_record_without_an_epoch_is_fenced_on_a_restarted_run(
    incomplete_run,
):
    """The other direction of the same default, and the one that matters.

    ``_record_restart_epoch`` degrades a missing or corrupt field to ``0``,
    and the direction is the whole point: reading ``0`` makes a record look
    *older* than it may be, so a doubtful authority is **fenced** rather than
    believed. That moves the verdict away from ``active`` and toward
    ``incomplete`` -- INV-VERDICT's direction.

    Without this, the paired test above is satisfied by a default that
    degrades *upward*: ``sys.maxsize`` would also let a pre-field record count
    on an unrestarted run, while believing every stale authority on every
    restarted one. Both tests are needed to pin a default; one pins only that
    it exists.
    """
    _publish_lifecycle(incomplete_run, generation="gen-1")
    record = _lifecycle_record(incomplete_run)
    record.pop("restart_epoch", None)
    slurm_lifecycle_path(incomplete_run).write_text(
        json.dumps(record), encoding="utf-8"
    )
    _set_config_restart_epoch(incomplete_run, 1)

    assert resolve_run_state(incomplete_run, depth="deep").completion != (
        "active"
    ), "a record predating the epoch field was believed on a restarted run"


# --------------------------------------------- the asymmetry, pinned as one


def test_a_live_gui_owner_still_reports_active_across_a_restart(
    incomplete_run,
):
    """**The GUI owner record is deliberately NOT epoch-fenced.**

    This is the one place the fence covers half its surface on purpose, so it
    is the one place a future contributor "completes" the job by stamping an
    epoch onto the owner record too -- a one-line change that reads as
    finishing what Task 1 started, and that nothing else here would catch.

    Why the asymmetry is right: the owner record is a *local process* claim,
    already believed only while the pid it names is alive (P1's CAN-24 probe).
    That is a **stronger** check than an epoch comparison, not a weaker one --
    it asks whether the process exists rather than whether a number matches.
    A GUI still running across a restart is genuinely still running, which is
    not the stale-authority case the lifecycle fence is for. Fencing it would
    kill a live process's claim on the strength of a counter it never read.
    """
    _claim_as_live_gui(incomplete_run)
    _set_config_restart_epoch(incomplete_run, 3)

    state = resolve_run_state(incomplete_run, depth="deep")

    assert state.completion == "active", (
        "a live GUI owner was fenced by the restart epoch; the owner record "
        "is a process claim and is bounded by the pid probe, not the counter"
    )
