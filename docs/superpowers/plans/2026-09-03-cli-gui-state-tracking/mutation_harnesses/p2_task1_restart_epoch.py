"""P2 Task 1: prove every test in the restart-epoch suite can fail.

Ten mutations across three targets — `_cli/_cli_identity.py` (the counter),
`_cli/_cli_slurm_lifecycle.py` (the writer that stamps it) and
`sdk_/_run_state.py` (rule 2's fence that reads it) — plus
`sdk_/_io_constants.py` for the preserve set.

**Why the writer and the reader are both mutated.** The fence is only as good
as the field it compares, and the two can fail independently: a writer that
stops stamping leaves the reader comparing against a default, and a reader
that stops comparing leaves the writer stamping a field nobody consults.
Neither shows up in the other's tests, which is why
`test_the_lifecycle_record_carries_the_epoch_current_at_publication` exists
outside the fence at all.

**The asymmetry mutation is the one worth reading.** `_live_authority`
deliberately fences the SLURM lifecycle record and deliberately does *not*
fence the GUI owner record. That is the one place this change covers half a
surface on purpose, so it is where a future contributor "completes" the job.
The mutation stamps an epoch onto the owner record too — a one-line change
that reads as finishing what Task 1 started — and
`test_a_live_gui_owner_still_reports_active_across_a_restart` is what goes
red.

Run from the worktree root::

    uv run python docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/\
mutation_harnesses/p2_task1_restart_epoch.py

Safety and concurrency rules are identical to the Task 0 harness: full-path
backups outside the repo, restore in a `finally`, sha256 compared at the end,
and **do not edit a target while this runs** — it holds pristine source in
memory and writes it back after every mutation, so a concurrent edit is
silently reverted.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_ENV = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}

TARGETS = (
    "src/phenotypic/_cli/_cli_identity.py",
    "src/phenotypic/_cli/_cli_slurm_lifecycle.py",
    "src/phenotypic/sdk_/_run_state.py",
    "src/phenotypic/sdk_/_io_constants.py",
    "src/phenotypic/_cli/_cli_failure_tracker.py",
    "src/phenotypic/_cli/_cli_migrate.py",
    "src/phenotypic/_cli/_cli_state_management.py",
    "src/phenotypic/_cli/_cli_update_state.py",
)
SUITE = "tests/unit/cli/test_run_identity.py"

# (label, old, new, tests that MUST fail)
MUTATIONS: list[tuple[str, str, str, tuple[str, ...]]] = [
    (
        "per_image_config_digest becomes a WRAPPER instead of an alias --"
        " equal today, and one edit away from not being, with nothing failing"
        " at the moment it stops. D-C's whole point is that the generation and"
        " work_id cannot be allowed to drift apart about what counts as"
        " scientific configuration, and identity is the only agreement that"
        " cannot.",
        "per_image_config_digest = processing_configuration_digest\n",
        "def per_image_config_digest(config):\n"
        "    return processing_configuration_digest(config)\n",
        ("test_per_image_config_digest_is_the_work_id_digest_itself",),
    ),
    (
        "create_initial_state goes back to minting its own uuid4 instead of"
        " recording the threaded identity -- the fifth site becoming a sixth,"
        " with the state's generation no longer matching the one the run is"
        " fenced by",
        '            "processing_generation": '
        "identity.processing_generation,\n",
        '            "processing_generation": '
        '__import__("uuid").uuid4().hex,\n',
        (
            "test_create_initial_state_records_the_minted_identity",
            "test_the_state_generation_is_not_a_uuid",
        ),
    ),
    (
        "create_initial_state stops recording restart_epoch -- P1's"
        " requires_conversion signal 4 is its ABSENCE, so the schema gate"
        " fires on every tree the current build writes",
        '            "restart_epoch": identity.restart_epoch,\n        }',
        "        }",
        ("test_create_initial_state_records_the_minted_identity",),
    ),
    (
        "the two metadata digests drift apart: the minter hashes the path"
        " instead of the bytes. §7.4's late-metadata guarantee then fires on"
        " EVERY run instead of on a real edit -- a re-finalize per"
        " invocation, forever, with nothing failing. Category E: the"
        " docstring saying 'keep these identical' is prevention; this is the"
        " detection.",
        "    return hashlib.sha256(Path(path).read_bytes()).hexdigest()\n",
        "    return hashlib.sha256(str(path).encode()).hexdigest()\n",
        ("test_the_two_metadata_digests_agree",),
    ),
    (
        "D5 BROKEN: the restart epoch leaks into work_id, so --restart"
        " reprocesses every surviving store from zero -- turning --restart"
        " into --overwrite, which is the one thing D5 forbids. Modelled the"
        " way the leak would actually arrive: the epoch reaching the id"
        " through a field somebody added to make it available.",
        "            mode=mode,\n",
        "            mode=mode\n"
        "            + str(\n"
        '                __import__('
        '                    "phenotypic._cli._cli_identity", '
        'fromlist=["x"]\n'
        "                ).read_restart_epoch(config.output_dir)\n"
        "            ),\n",
        ("test_a_restart_moves_the_generation_but_not_any_work_id",),
    ),
    (
        "measure stops sharing the full run's identity -- it mints under its"
        " own mode, so §7.4 cannot route it through finalize_run and P4's"
        " byte-identical master across [full, measure, recompile] is"
        " unreachable",
        "            per_image_config=per_image_config_digest(config),\n"
        "            restart_epoch=epoch,\n",
        "            per_image_config=per_image_config_digest(config)\n"
        '            + str(config.measure_only),\n'
        "            restart_epoch=epoch,\n",
        ("test_measure_mints_the_identity_a_full_run_would",),
    ),
    (
        "process stops being distinguishable from full: process_only_layer"
        " is dropped from the per-image digest, so a layer export and a full"
        " run share a generation and the fence says two different"
        " configurations are the same one."
        " A SECOND TEST WAS CLAIMED HERE AND REMOVED. The reasoning was that"
        " dropping the field also hides config changes in the process arm --"
        " sound for a PROCESS config, and irrelevant to a test that uses a"
        " FULL one: `process_only_layer is None` takes the `else` branch,"
        " which never contained the field. A mutation is invisible to a test"
        " that does not enter the branch it edits."
        " THIS IS THE SECOND OVER-CLAIM OF ITS FAMILY. The first assumed a"
        " test SEES state it merely passes through; this one assumed a test"
        " ENTERS the branch being edited. Both are 'the mutation perturbs"
        " something nearby, therefore the test notices'.",
        "    if process_only_layer is not None:\n",
        "    if False:\n",
        ("test_process_mints_a_DIFFERENT_identity_and_that_is_correct",),
    ),
    (
        "THE FENCE IS REMOVED: the event-log generation filter never"
        " excludes anything, so a worker abandoned by a --restart keeps"
        " reporting progress into the post-restart state. Spec §14's"
        " requirement, deleted."
        " THIS IS THE POSITIVE PREDICTION and it must go red for the RIGHT"
        " reason -- `a_restart_excludes_events_from_the_previous_generation`"
        " fails because a foreign-generation event is now COUNTED, not"
        " because a signature moved. Its pair must stay GREEN: removing an"
        " exclusion cannot break a test that asserts inclusion, and if it"
        " does then the two tests are not the independent halves they claim"
        " to be. Category E shape #2 -- a guard whose branch does nothing --"
        " is exactly what this proves absent.",
        "                generation is not None\n"
        "                and event.generation is not None\n"
        "                and event.generation != generation\n",
        "                False\n",
        (
            "test_a_restart_excludes_events_from_the_previous_generation",
        ),
    ),
    (
        "the fence is INVERTED: it excludes an event whose generation"
        " MATCHES. A run then discards its own history on every resume --"
        " the pre-3220a740 behaviour, restored, and the one a reader would"
        " mistake for tightening a fence rather than deleting one.",
        "                and event.generation != generation\n",
        "                and event.generation == generation\n",
        ("test_a_resume_counts_events_from_its_own_generation",),
    ),
    (
        "CAN-21's mint-once guard is removed, so one invocation can mint two"
        " generations and burn a restart epoch silently",
        "    if getattr(config, _MINTED_FLAG, False):\n",
        "    if False:\n",
        (
            "test_minting_twice_in_one_invocation_is_a_programming_error",
        ),
    ),
    (
        "a RESUME bumps the epoch too, so every resume fences its own"
        " in-flight workers -- the failure D5 exists to prevent, and the"
        " opposite of what a resume is for."
        " ONE TEST WAS CLAIMED HERE AND REMOVED, because the claim did not"
        " follow: this mutation DOES perturb state the mint-once test's"
        " fixture passes through -- the first mint now burns an epoch -- and"
        " that test is still blind to it, because it asserts that a SECOND"
        " mint raises, which it still does, for the guard. Perturbing state a"
        " test passes through is not the same as perturbing state it asserts"
        " on.",
        "    epoch = (\n"
        "        bump_restart_epoch(output_dir)\n"
        "        if restart\n"
        "        else read_restart_epoch(output_dir)\n"
        "    )\n",
        "    epoch = bump_restart_epoch(output_dir)\n",
        ("test_a_resume_does_not_bump_the_epoch_but_a_restart_does",),
    ),
    (
        "CATEGORY E #4: `per_image_config` is ACCEPTED AND NEVER READ."
        " mint_run_identity still takes an ExecutionConfig, the signature is"
        " unchanged, mypy is silent, and derive_processing_generation's own"
        " tests all pass -- because they test the primitive, not the caller."
        " Only a test that moves a config field and watches the MINTED"
        " generation can see it.",
        "            per_image_config=per_image_config_digest(config),\n",
        "            per_image_config=None,\n",
        (
            "test_a_per_image_config_change_mints_a_new_generation",
            "test_process_mints_a_DIFFERENT_identity_and_that_is_correct",
        ),
    ),
    (
        "the minted generation stops folding in the pipeline digest",
        "            pipeline_sha256=pipeline_sha256,\n"
        "            per_image_config=per_image_config_digest(config),\n",
        "            pipeline_sha256=None,\n"
        "            per_image_config=per_image_config_digest(config),\n",
        ("test_a_pipeline_edit_mints_a_new_generation",),
    ),
    (
        "THE A/B SWAP: the minted proof-side token becomes the PER-IMAGE"
        " digest. Every proof this run publishes then disagrees with every"
        " proof already on disk, and nothing else in the suite notices --"
        " drift register entry 14, arriving through the minter instead of"
        " through a rename.",
        "        scientific_config_digest=pipeline_sha256 or \"\",\n",
        "        scientific_config_digest=per_image_config_digest(config),\n",
        ("test_the_minted_proof_side_digest_is_the_pipeline_digest",),
    ),
    (
        "the generation goes back to being a uuid4 -- D3 gone entirely, and"
        " every site this task converted regresses at once"
        " [inherently broad: when the generation stops being a function of its"
        " inputs, every test about those inputs fails; that is a property of"
        " deleting determinism, not a weakness in those tests]",
        "    return canonical_digest(\n"
        "        {\n"
        '            "pipeline_sha256": pipeline_sha256 or "",\n'
        '            "per_image_config_digest": per_image_config or "",\n'
        '            "restart_epoch": restart_epoch,\n'
        "        }\n"
        "    )\n",
        "    from uuid import uuid4\n\n    return uuid4().hex\n",
        ("test_the_same_components_mint_the_same_generation",),
    ),
    (
        "the generation ignores per_image_config -- a pipeline-only fence,"
        " so a detect_mode or bit_depth change mints the same generation and"
        " a worker holding it is not fenced."
        " The minter test here is NOT independent evidence -- the"
        " minter calls the primitive, so one dropped component fails"
        " both. It is claimed because it genuinely fails, not because"
        " it adds a witness. The E#4 mutation is what proves the minter"
        " tests earn their place; a later reader trimming 'redundant'"
        " tests must cut from here and never from there.",
        '            "per_image_config_digest": per_image_config or "",\n',
        '            "per_image_config_digest": "",\n',
        (
            "test_every_component_moves_the_generation"
            "[per_image_config-cfg-2]",
            "test_a_per_image_config_change_mints_a_new_generation",
            "test_process_mints_a_DIFFERENT_identity_and_that_is_correct",
        ),
    ),
    (
        "the generation ignores restart_epoch -- D4's whole purpose gone, so"
        " a deliberately fresh attempt is indistinguishable from the run it"
        " replaces and the pre-restart workers pass the fence."
        " The minter test here is NOT independent evidence -- the"
        " minter calls the primitive, so one dropped component fails"
        " both. It is claimed because it genuinely fails, not because"
        " it adds a witness. The E#4 mutation is what proves the minter"
        " tests earn their place; a later reader trimming 'redundant'"
        " tests must cut from here and never from there.",
        '            "restart_epoch": restart_epoch,\n        }\n    )\n',
        '            "restart_epoch": 0,\n        }\n    )\n',
        (
            "test_every_component_moves_the_generation[restart_epoch-1]",
            "test_a_resume_does_not_bump_the_epoch_but_a_restart_does",
            "test_a_restart_moves_the_generation_but_not_any_work_id",
        ),
    ),
    (
        "the generation ignores pipeline_sha256 -- an edited pipeline mints"
        " the same generation."
        " The minter test here is NOT independent evidence -- the"
        " minter calls the primitive, so one dropped component fails"
        " both. It is claimed because it genuinely fails, not because"
        " it adds a witness. The E#4 mutation is what proves the minter"
        " tests earn their place; a later reader trimming 'redundant'"
        " tests must cut from here and never from there.",
        '            "pipeline_sha256": pipeline_sha256 or "",\n',
        '            "pipeline_sha256": "",\n',
        (
            "test_every_component_moves_the_generation"
            "[pipeline_sha256-pipe-2]",
            "test_a_pipeline_edit_mints_a_new_generation",
        ),
    ),
    (
        "None and empty string stop being the same input, so a pipeline-less"
        " run mints a different generation depending on which spelling of"
        " nothing its caller happened to pass",
        '            "pipeline_sha256": pipeline_sha256 or "",\n'
        '            "per_image_config_digest": per_image_config or "",\n',
        '            "pipeline_sha256": pipeline_sha256,\n'
        '            "per_image_config_digest": per_image_config,\n',
        ("test_an_absent_component_is_the_empty_one",),
    ),
    (
        "the MIGRATOR folds the inventory back into the generation -- CAN-7,"
        " restoring the D7 violation that shipped in dd18d9c7 and made every"
        " image arrival under a rolling input look like a config change",
        '            "processing_generation": '
        "derive_processing_generation(\n"
        "                pipeline_sha256=_file_sha256(pipeline_path),\n"
        "                per_image_config=None,\n"
        "                restart_epoch=0,\n"
        "            ),\n",
        '            "processing_generation": '
        "derive_processing_generation(\n"
        "                pipeline_sha256=_file_sha256(pipeline_path),\n"
        "                per_image_config=repr(sorted(work_ids.items())),\n"
        "                restart_epoch=0,\n"
        "            ),\n",
        ("test_a_migrated_trees_generation_ignores_its_inventory",),
    ),
    (
        "the migrator stops recording restart_epoch -- P1's"
        " requires_conversion signal 4 is its ABSENCE, so a freshly migrated"
        " tree is refused by the very next --mode full",
        '            "restart_epoch": 0,\n            "pipeline_sha256":',
        '            "pipeline_sha256":',
        ("test_a_migrated_tree_records_a_restart_epoch",),
    ),
    (
        "the per-image digest grows a pipeline component, so it becomes"
        " capable of answering the PROOF-side digest's question too -- which"
        " is the state in which the next reader notices one name for two"
        " values and 'unifies' them, rewriting the digest in every proof on"
        " disk. Drift register entry 14.",
        "    drop_originals: bool = False,\n) -> str:",
        "    drop_originals: bool = False,\n"
        '    pipeline_sha256: str = "",\n'
        ") -> str:",
        ("test_the_proof_side_digest_is_a_different_value_entirely",),
    ),
    (
        "the restart epoch is dropped from the preserve set -- a restart"
        " resets the counter that fences it, which is not a fence",
        "    {TERMINAL_FAILURES_JSONL, RESTART_EPOCH_JSON}\n",
        "    {TERMINAL_FAILURES_JSONL}\n",
        ("test_restart_epoch_survives_clear_machine_state",),
    ),
    (
        "a corrupt counter raises instead of degrading to 0 -- one bad byte"
        " becomes a reason the user cannot restart at all",
        "    try:\n"
        "        # UnicodeDecodeError is a ValueError, so undecodable bytes "
        "and\n"
        "        # malformed JSON are one case here, as they are one case to "
        "a caller.\n"
        "        document = json.loads(raw)\n"
        "    except ValueError:\n"
        "        return 0\n",
        "    document = json.loads(raw)\n",
        ("test_reading_a_corrupt_restart_epoch_is_zero_not_an_error",),
    ),
    (
        "bool is accepted as an epoch, so `true` reads as 1 -- a fence"
        " silently advanced by a type error",
        "    if not isinstance(epoch, int) or isinstance(epoch, bool) or "
        "epoch < 0:\n",
        "    if not isinstance(epoch, int):\n",
        ("test_a_boolean_is_not_a_restart_epoch",),
    ),
    (
        "a failed write is swallowed and returns quietly -- the next"
        " invocation mints the generation the abandoned workers already hold",
        "    atomic_write_json(restart_epoch_path(root), "
        "{_EPOCH_KEY: updated})\n",
        "    try:\n"
        "        atomic_write_json(restart_epoch_path(root), "
        "{_EPOCH_KEY: updated})\n"
        "    except OSError:\n"
        "        pass\n",
        ("test_a_failed_write_raises_rather_than_returning_quietly",),
    ),
    (
        "the writer stops stamping the epoch. NOTE WHAT THIS DOES *NOT*"
        " BREAK: the fence tests still pass, because an unstamped record"
        " degrades to 0 and 0 >= a bumped epoch is False -- so the run is"
        " still fenced, for the wrong reason. Only the writer's own tests"
        " see it, which is exactly why they exist outside the fence.",
        '            "restart_epoch": read_restart_epoch(output_dir),\n',
        "",
        (
            "test_the_lifecycle_record_carries_the_epoch_current_at_"
            "publication",
            "test_an_existing_active_fence_is_not_re_dated",
        ),
    ),
    (
        "an existing active fence is RE-DATED to the current epoch, so a"
        " worker from before the restart looks current again -- the precise"
        " failure the fence exists to prevent."
        " THE MUTATION MUST PERSIST, and the first version did not: it set"
        " the key on `existing`, which is the in-memory dict"
        " `load_slurm_lifecycle` returned, and the early return writes"
        " nothing. The file stayed correct, so the test passed and the"
        " mutation reported NOT PROVED against a test that was right all"
        " along. `_live_authority` reads the FILE, so an in-memory re-stamp"
        " is a no-op for the fence -- it did not model the bug at all.",
        "            # Deliberately NOT re-stamped: this fence was published "
        "earlier,\n"
        "            # and the epoch live at *that* moment is the one it "
        "asserts.\n"
        "            return existing\n",
        "            existing[\"restart_epoch\"] = read_restart_epoch("
        "output_dir)\n"
        "            atomic_write_json(\n"
        "                lifecycle_state_path(output_dir), existing\n"
        "            )\n"
        "            return existing\n",
        ("test_an_existing_active_fence_is_not_re_dated",),
    ),
    (
        "rule 2's first half is dropped -- a lifecycle record from a"
        " superseded epoch reports the run alive, which is a stale authority"
        " outranking a valid verdict. Takes down BOTH fenced-authority tests,"
        " which is right rather than broad: deleting the comparison stops"
        " fencing a stamped stale record AND a record that predates the"
        " field, and those are two separate claims about the same line.",
        "        and _record_restart_epoch(lifecycle) >= "
        "identity.restart_epoch\n",
        "",
        (
            "test_a_pre_restart_authority_does_not_report_the_run_active",
            "test_a_record_without_an_epoch_is_fenced_on_a_restarted_run",
        ),
    ),
    (
        "the fence is strict, so a record at the CURRENT epoch is refused --"
        " no run is ever active and every 'is refused' test still passes",
        "        and _record_restart_epoch(lifecycle) >= "
        "identity.restart_epoch\n",
        "        and _record_restart_epoch(lifecycle) > "
        "identity.restart_epoch\n",
        (
            "test_a_current_authority_still_reports_the_run_active",
            "test_a_record_without_an_epoch_still_counts_on_an_unrestarted_"
            "run",
        ),
    ),
    (
        "a missing epoch field degrades UPWARD instead of to 0, so a doubtful"
        " authority is believed rather than fenced -- away from incomplete,"
        " against INV-VERDICT's direction",
        "    epoch = record.get(\"restart_epoch\")\n"
        "    if not isinstance(epoch, int) or isinstance(epoch, bool):\n"
        "        return 0\n",
        "    epoch = record.get(\"restart_epoch\")\n"
        "    if not isinstance(epoch, int) or isinstance(epoch, bool):\n"
        "        import sys as _s\n"
        "        return _s.maxsize\n",
        ("test_a_record_without_an_epoch_is_fenced_on_a_restarted_run",),
    ),
    (
        "THE ASYMMETRY: the GUI owner record is epoch-fenced too -- a"
        " one-line change that reads as finishing the job, and kills a LIVE"
        " process's claim on the strength of a counter it never read",
        "        if (\n"
        "            isinstance(pid, int)\n"
        "            and not isinstance(pid, bool)\n"
        "            and _process_is_alive(pid)\n"
        "        ):\n",
        "        if (\n"
        "            isinstance(pid, int)\n"
        "            and not isinstance(pid, bool)\n"
        "            and _process_is_alive(pid)\n"
        "            and _record_restart_epoch(owner) >= "
        "identity.restart_epoch\n"
        "        ):\n",
        ("test_a_live_gui_owner_still_reports_active_across_a_restart",),
    ),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _suite_test_names() -> set[str]:
    """Return every id pytest collects, **including parametrized cases**.

    **Collected from pytest, not by AST, and that is the whole point.** An
    ``ast.FunctionDef`` carries only the bare function name, so
    ``test_x[a-b]`` is invisible to it: every bracketed expectation read as a
    typo and the harness aborted before running a single mutation, naming
    three tests that pytest collects verbatim. This is the first harness to
    claim a parametrized case, which is why it had never fired.

    Note that ``check_mutation_coverage.py`` has the **opposite** blind spot
    on purpose -- it strips ``[...]`` to the stem so it can stay fast and
    pytest-free, which is exactly why it reported ``COVERAGE_OK=True`` on the
    state that aborted here. Neither gate is individually complete and making
    them agree would remove the backstop: only pytest knows the ids, and only
    the checker can run without it.

    Two output shapes are accepted because pytest's differs by version and by
    ``addopts``: ``path::test_x[a-b]`` and ``<Function test_x[a-b]>``. A parse
    that matched neither would yield an empty set, which would make **every**
    expectation "unknown" and abort with precisely the misleading message
    this function was rewritten to stop producing -- so an empty result is
    raised as itself instead.
    """
    proc = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            SUITE,
            "--collect-only",
            "-q",
            "--no-header",
        ],
        capture_output=True,
        text=True,
        env={**_ENV},
    )
    names: set[str] = set()
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if "::" in stripped:
            candidate = stripped.rsplit("::", 1)[1]
        elif stripped.startswith("<Function "):
            candidate = stripped[len("<Function ") :].rstrip(">")
        else:
            continue
        if candidate.startswith("test_"):
            names.add(candidate)
    if not names:
        raise SystemExit(
            "ABORT: pytest --collect-only produced no recognizable test ids "
            "for "
            f"{SUITE}. The parse matched neither `path::name` nor "
            "`<Function name>`; fix the parse rather than the expectations. "
            f"First lines were:\n{chr(10).join(proc.stdout.splitlines()[:5])}"
        )
    return names


def _failed_tests() -> set[str]:
    proc = subprocess.run(
        ["uv", "run", "pytest", SUITE, "-q", "--no-header", "-rf"],
        capture_output=True,
        text=True,
        env={**_ENV},
    )
    failed: set[str] = set()
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("FAILED ") or stripped.startswith("ERROR "):
            name = stripped.split("::", 1)[-1].split(" ", 1)[0]
            failed.add(name)
    return failed


def _owner(sources: dict[Path, str], old: str) -> Path | None:
    """Return the one target containing ``old`` exactly once, else ``None``."""
    owners = [path for path, text in sources.items() if text.count(old) == 1]
    return owners[0] if len(owners) == 1 else None


def main() -> int:
    targets = [Path(name).resolve() for name in TARGETS]
    missing = [t for t in targets if not t.is_file()]
    if missing:
        print(f"ABORT: run me from the worktree root -- {missing} not found")
        return 4
    backup_dir = Path(tempfile.mkdtemp(prefix="phenotypic-mutation-"))
    print(f"backup: {backup_dir}")
    sources: dict[Path, str] = {}
    originals: dict[Path, str] = {}
    for name, target in zip(TARGETS, targets):
        backup = backup_dir / name
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
        sources[target] = target.read_text(encoding="utf-8")
        originals[target] = _sha256(target)

    rows: list[tuple[str, str, str]] = []
    try:
        defined = _suite_test_names()
        named = {name for _l, _o, _n, exp in MUTATIONS for name in exp}
        unknown = named - defined
        if unknown:
            print(
                "ABORT: MUTATIONS names tests that do not exist: "
                f"{sorted(unknown)}"
            )
            return 3
        unclaimed = defined - named
        print(
            f"suite defines {len(defined)} tests; {len(named)} are claimed by "
            f"a mutation"
        )
        if unclaimed:
            print(f"NOT COVERED by any mutation: {sorted(unclaimed)}")

        unowned = [
            label
            for label, old, _new, _exp in MUTATIONS
            if _owner(sources, old) is None
        ]
        if unowned:
            print(
                "ABORT: these anchors match no target exactly once: "
                f"{[label[:60] for label in unowned]}"
            )
            return 3

        baseline = _failed_tests()
        if baseline:
            print(f"ABORT: suite is not green to begin with: {baseline}")
            return 2
        print("baseline: suite green\n")

        for label, old, new, expected in MUTATIONS:
            target = _owner(sources, old)
            assert target is not None
            source = sources[target]
            target.write_text(source.replace(old, new, 1), encoding="utf-8")
            failed = _failed_tests()
            target.write_text(source, encoding="utf-8")

            missing_tests = set(expected) - failed
            extra = failed - set(expected)
            if missing_tests:
                verdict = "NOT PROVED"
                detail = f"did not fail: {sorted(missing_tests)}"
            elif extra:
                verdict, detail = (
                    "PROVED (broad)",
                    f"also failed: {sorted(extra)}",
                )
            else:
                verdict, detail = "PROVED", f"exactly {sorted(expected)}"
            rows.append((label, verdict, detail))
            print(
                f"{verdict:<14} [{target.name}] {label}\n"
                f"               {detail}"
            )
    finally:
        for name, target in zip(TARGETS, targets):
            shutil.copy2(backup_dir / name, target)
            restored = _sha256(target)
            status = "OK" if restored == originals[target] else "MISMATCH"
            print(f"restored {name}: {status} ({restored[:12]})")

    print("\n--- summary ---")
    for label, verdict, detail in rows:
        print(f"{verdict:<14} | {label} | {detail}")
    unproved = [r for r in rows if r[1] not in {"PROVED", "PROVED (broad)"}]
    print(f"\nMUTATIONS_ALL_PROVED={not unproved}")
    return 0 if not unproved else 1


if __name__ == "__main__":
    sys.exit(main())
