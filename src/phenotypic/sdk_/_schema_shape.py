"""Detect a run output still written in the pre-consolidation shape.

**A pure reader, which is why it lives here.** This module writes nothing,
reads raw JSON and two directory facts, and imports nothing from
:mod:`phenotypic._cli`. It has two consumers on opposite sides of INV-LAYER:

* ``_cli/_cli_schema_gate.refuse_unconverted_schema`` -- the **writer** half.
  Every mode that writes or reprocesses (``full``, ``measure``, ``recompile``,
  ``process``) refuses an unconverted tree and names ``--mode migrate``.
* :func:`phenotypic.sdk_._run_state.resolve_run_state` -- the **reader** half.
  Spec §4.3 makes an unconverted tree an *advisory* on ``RunState``, never a
  gate, so the GUI keeps displaying one.

Those two are the same detection surfaced to different audiences, so the
detection has one home. It was originally written entirely CLI-side, and its
module docstring drew the line in the right place -- *"a refusal that raises
``click.UsageError`` is the other side of that line"* -- while leaving the
predicate on the wrong side of it. INV-LAYER forced the correction: the
reader could not import the predicate, and a second copy would let the GUI's
advisory and the CLI's refusal disagree about whether a tree needs migrating,
which is CAN-4's shape.

**Why the detection has to exist before the consolidated record does
(CAN-11).** P3 makes a clean break -- ``publish_image_success`` writes the new
per-image record and ``valid_image_success`` reads it. On an unconverted tree
``authorized_measurement_sources`` then yields an **empty mapping**, and ``{}``
is a *valid* schema-3 answer meaning "nothing has succeeded yet", not a
failure. ``finalize_run`` would publish an empty master and raise nothing: a
successful-looking run that discarded every measurement. This module is what
turns that into an error, so it is built in P1 rather than P7.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Final, cast

from ._io_constants import (
    DIR_IMAGE_COMPLETE,
    MIGRATION_REMEDY,
    resolve_processing_state_path,
    resolve_progress_dir,
)

__all__ = [
    "SCHEMA_GATE_ARMED",
    "STATE_SCHEMA_VERSION",
    "ConversionVerdict",
    "describe_conversion_advisory",
    "describe_required_conversion",
    "requires_conversion",
]

#: Major component of the ``version`` string ``save_processing_state`` writes.
#:
#: **Never a detection signal.** U-6: ``state.version`` cannot express a floor
#: -- ``"2.0.0"`` is the value at v0.17.3 *and* immediately before ``"3.0.0"``
#: was introduced, and today's markers-era tree already carries ``"3.0.0"``
#: while still needing conversion. It is recorded here so that a future bump of
#: the writer's literal fails a test in this module's suite rather than
#: silently changing what "current" means.
STATE_SCHEMA_VERSION: Final[int] = 3

#: Whether a ``CONVERT`` verdict may be **surfaced**. **Flipped to ``True`` by
#: P3 Task 2**, in the same commit that makes ``publish_image_success`` write
#: the consolidated record -- see
#: ``test_the_gate_is_armed_exactly_when_the_forward_path_stops_writing_the``
#: ``_legacy_marker``, which fails the moment those two disagree.
#:
#: It is ``False`` today because the legacy shape and the **current** shape
#: still overlap: the forward path writes ``image_complete/`` and writes
#: ``datasets.<ds>.completed``, so **two** of the five signals below fire on a
#: tree the running build has just written. **Detection is correct now; only
#: the surfacing waits.**
#:
#: It said *three* until P2, and P2 is what made that false: signal 4 is the
#: **absence** of ``restart_epoch``, and ``create_initial_state`` now writes it
#: (``_cli_state_management.py:261``). The count is not decoration -- it is the
#: claim that decides whether arming this flag would refuse trees the current
#: build wrote, so a stale one is a wrong answer to the only question a reader
#: opens this docstring to ask (gate IMPL-F6 / SPEC-C3, found independently by
#: two reviewers). Signal 3 (``datasets.<ds>.completed``) is §4.2's to remove
#: in P3+, and the count drops to one when it does.
#:
#: It gates **both** consumers, because both are surfacings of one detection:
#: an armed gate would refuse every resume of every mode, and an armed
#: advisory would banner "run ``--mode migrate``" on every GUI output --
#: advice the user cannot act on, since migrate does not convert
#: ``.phenotypic/`` until P7 Tasks 2, 2b and 3. An advisory that is always on
#: is worse than none: it teaches people to ignore the one that will matter.
#:
#: **Patching note. There is exactly one binding: this one.** Both consumers
#: read it through this module -- ``_run_state`` for the advisory,
#: ``refuse_unconverted_schema`` for the refusal -- so **every test arming
#: either one patches ``_schema_shape``**, never ``_cli_schema_gate``.
#:
#: An earlier draft kept a re-export on ``_cli_schema_gate`` and this note
#: described how to patch each side separately. That was rejected: a
#: re-exported copy reads correctly while being **inert under monkeypatch**, so
#: a test patching the name on the module it is testing would change nothing,
#: and the flag is the last place anyone looks.
#: ``test_the_arming_flag_has_one_source`` asserts no ``Assign``, ``AnnAssign``
#: or ``ImportFrom`` in ``_cli_schema_gate`` binds the name, and that
#: ``hasattr`` is ``False`` -- so following the old note now fails a test
#: rather than silently doing nothing.
#:
#: The name predates the second consumer and is now slightly narrow; renaming
#: it would churn cluster 1.4's suite mid-phase, so P3 Task 2 -- where someone
#: next has this constant in front of them -- is where that happens.
#:
#: **STILL DISARMED after P3, by user ruling, and the coupling that governs
#: it CHANGED.** P3 armed this and P3 un-armed it again; the reversal is the
#: useful record, so it is written down rather than tidied away.
#:
#: The original rule was *arm in the same commit that moves the publisher onto
#: the record*, because moving the publisher alone leaves a legacy tree with
#: no valid records (CAN-11). P3 did arm it on that reasoning -- and the gate
#: lane then showed the consequence nobody had traced: **``--mode migrate``
#: does not discharge this gate.** ``_hdf_to_zarr._republish_image_marker``
#: rewrites the *legacy* marker and nothing removes ``image_complete/``, so
#: signal 1 fires on a tree migrate has just finished. Armed, the error
#: message names migrate as the remedy, migrate succeeds, and the tree is
#: refused again -- a loop with no exit but ``--overwrite``, which destroys
#: the outputs. INV-DISCHARGEABLE, on the escape hatch itself.
#:
#: So the binding constraint is not *"arm with the publisher"* but **"arm with
#: DISCHARGEABILITY"**, and that lands in **P7 Task 5 Step 1b**, which renames
#: the legacy trees into ``.phenotypic/legacy-v2/`` -- outside ``progress/``
#: and therefore invisible to signal 1 (see the note at the directory probe
#: below). Arm here in that commit, not before.
#:
#: **What disarming costs, stated so P7 does not have to rediscover it.** It
#: is not a clean return to pre-P3 behaviour. Pre-P3 the publisher *and* the
#: reader were both on ``image_complete/``, so a legacy tree resumed
#: correctly. Now the reader is on the record and nothing stops a legacy tree
#: from entering a writing mode, so ``valid_image_success`` is false for every
#: image and the run **reprocesses from source** instead of resuming -- which
#: on a migrated archive whose inputs are gone is a failure rather than a
#: waste. That is worse than pre-P3 and better than the refusal loop, which is
#: the trade the ruling makes.
SCHEMA_GATE_ARMED: bool = False

#: ``<progress>/stage3_complete/``. A module-private literal, and as of P3 a
#: **read-only** one: the writer it used to be pinned against
#: (``_cli_staged_resume.stage3_completion_marker_path``) is deleted, and the
#: only remaining reference is signal 1's directory probe at ``:247``. So it
#: stays out of ``sdk_/_io_constants.py`` -- that module names paths the
#: project *writes*, and nothing writes this tree any more.
#:
#: **It therefore has no anchor test. That is a gap, not a simplification.**
#: ``test_the_stage3_directory_name_matches_the_writer`` compared this constant
#: to the writer's own path and was deleted with the writer; a replacement
#: could only compare the constant to itself. A wrong value here fails
#: silently and in the dangerous direction -- signal 1 stops firing, and a
#: legacy staged tree is neither converted nor refused (INV-DISCHARGEABLE).
#: Its remaining ground truth is P7's migrate tests against the real pre-record
#: test bed.
_DIR_STAGE3_COMPLETE: Final[str] = "stage3_complete"

#: A JSON object carrying neither key is not a processing state at all:
#: ``load_processing_state`` raises ``KeyError`` reading ``version`` (
#: ``_cli_state_management.py:167``) and ``datasets`` (``:148``). Such a file
#: is *unreadable*, not *legacy* -- see :class:`ConversionVerdict`.
_STATE_SHAPE_KEYS: Final[tuple[str, ...]] = ("version", "datasets")

#: Operational half of the refusal (P7 Task 5 Step 1c, CAN-13). A SLURM array
#: launched from the old build holds the old schema for its whole lifetime --
#: up to 30 d -- and writes the legacy trees directly, so a tree migrated
#: underneath a live array re-acquires the old shape and is refused again.
_DRAIN_ADVICE: Final[str] = (
    "Drain or `scancel` any in-flight SLURM array for this output before "
    "migrating: a worker from the old build writes the old shape for its "
    "whole lifetime and would put this output back into it."
)


class ConversionVerdict(Enum):
    """Why an output directory cannot be written to as it stands.

    Two members, and there is deliberately **no ``BELOW_FLOOR``** (U-6): there
    is no version floor to be below. A pre-markers tree is supported however
    old, because every pre-markers tree is the *same shape* and the ported
    promoter (P7 Task 2b) handles them identically.
    """

    #: The tree is in the old shape and ``--mode migrate`` converts it.
    CONVERT = "convert"

    #: ``processing_state.json`` exists but cannot be read as one. **Not
    #: ``CONVERT``**: migrate cannot repair a truncated state file, so pointing
    #: at it would strand the tree behind a refusal in every writing mode
    #: (INV-DISCHARGEABLE). The message names the file instead.
    UNREADABLE_STATE = "unreadable_state"


def _read_raw_state(
    state_path: Path,
) -> tuple[Mapping[str, object] | None, bool]:
    """Return ``(payload, unreadable)`` for a raw processing-state file.

    Reads the JSON itself and **never** calls ``load_processing_state``
    (MIG-14b). Two reasons, both disqualifying:

    * ``load_processing_state`` calls ``migrate_legacy_machine_state``
      (``_cli_state_management.py:108``) -- **a write**. A refusal gate that
      mutates the tree before refusing it is worse than the silent path it
      replaces.
    * It indexes ``state_dict[ProcessingStateKey.VERSION]`` unguarded
      (``:167``) and ``json.loads`` at ``:115`` raises on a truncated file, so
      a malformed tree would crash the gate. INV-VERDICT's degrade half applies
      here as much as to the reader: absent, unparseable and malformed all map
      to a verdict, never to an exception.

    Args:
        state_path: Candidate ``processing_state.json``.

    Returns:
        ``(None, False)`` when the file is absent -- mirroring
        ``load_processing_state``'s own ``return None`` (``:111-112``);
        ``(None, True)`` when it is present but not readable as a state file;
        otherwise the decoded object and ``False``.
    """
    try:
        raw = state_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None, False
    except (OSError, ValueError):
        return None, True
    try:
        payload = json.loads(raw)
    except ValueError:
        return None, True
    if not isinstance(payload, Mapping):
        return None, True
    if not any(key in payload for key in _STATE_SHAPE_KEYS):
        return None, True
    return cast("Mapping[str, object]", payload), False


def _relative(path: Path, root: Path) -> str:
    """Return *path* spelled relative to *root* when it is below it."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _classify(output_dir: Path) -> tuple[ConversionVerdict | None, str]:
    """Return the verdict for *output_dir* and the evidence behind it.

    The five detection signals of :func:`requires_conversion`, in the order a
    verdict is reached. Readability of ``processing_state.json`` is settled
    first, because a *modern* tree with a truncated state file must not
    classify ``CONVERT`` (INV-DISCHARGEABLE).

    Args:
        output_dir: Run output root.

    Returns:
        ``(None, "")`` for a tree the forward path can write to, otherwise the
        verdict and a human-readable phrase naming what was found.
    """
    root = Path(output_dir)
    state_path = resolve_processing_state_path(root)
    payload, unreadable = _read_raw_state(state_path)
    if unreadable:
        return ConversionVerdict.UNREADABLE_STATE, _relative(state_path, root)

    # Signals 1 and 2 are directory facts, independent of the state file: they
    # are what "the old shape" means. `stage2_done/` is deliberately absent
    # from this pair (U-9) -- that tree is current, not legacy, so firing on it
    # would classify every modern GPU run CONVERT and strand it. The retained
    # `.phenotypic/legacy-v2/` migrate renames the trees into (P7 Task 5
    # Step 1b) is invisible here because it is not below `progress/`.
    progress = resolve_progress_dir(root)
    for segment in (DIR_IMAGE_COMPLETE, _DIR_STAGE3_COMPLETE):
        if (progress / segment).is_dir():
            return (
                ConversionVerdict.CONVERT,
                f"{_relative(progress / segment, root)}/ exists",
            )

    if payload is None:
        # An absent state file is not a schema to be wrong about. Refusing one
        # would make every new run start with an error, and would refuse a
        # standalone deliverables bundle, which `BundleLayout.detect`
        # explicitly supports.
        return None, ""

    raw_config = payload.get("config")
    config: Mapping[str, object] = (
        raw_config if isinstance(raw_config, Mapping) else {}
    )
    datasets = payload.get("datasets")

    # Signal 3: the derived per-dataset sets §4.2 deletes from the file.
    if isinstance(datasets, Mapping) and any(
        isinstance(entry, Mapping) and "completed" in entry
        for entry in datasets.values()
    ):
        return (
            ConversionVerdict.CONVERT,
            "processing_state.json carries datasets.<dataset>.completed",
        )

    # Signal 5: the pre-markers shape. `work_ids` did not exist at v0.17.3, so
    # its absence -- not `version` -- is what separates that era. Checked
    # before signal 4 only because the two partition on the same key.
    if "work_ids" not in config:
        return (
            ConversionVerdict.CONVERT,
            "processing_state.json has no config.work_ids (pre-markers run)",
        )

    # Signal 4: markers-era. Accepted inventory present, restart epoch not yet.
    if "restart_epoch" not in config:
        return (
            ConversionVerdict.CONVERT,
            "processing_state.json has config.work_ids and no "
            "config.restart_epoch",
        )

    return None, ""


def requires_conversion(output_dir: Path) -> ConversionVerdict | None:
    """Return why *output_dir* needs ``--mode migrate``, or ``None``.

    Two outcomes plus honest failure (U-6):

    * ``None`` -- already current
    * :attr:`ConversionVerdict.CONVERT` -- convertible; the message from
      :func:`describe_required_conversion` names the evidence
    * :attr:`ConversionVerdict.UNREADABLE_STATE` -- present but unreadable
      state file; the message names *that file* and does not send the user to
      migrate, which cannot repair it

    **It never raises.** Absent, unparseable and malformed state all map to a
    verdict, because a refusal gate that crashes on a malformed tree is worse
    than the silent path it replaces (MIG-14b).

    **Detection is not surfacing.** This answers "is this tree in the old
    shape?", which at P1 is ``CONVERT`` for every tree the running build
    writes. Whether that may be shown to anyone is
    :data:`SCHEMA_GATE_ARMED`'s question, and both consumers ask it.

    **There is no version floor.** U-1 named v0.17.3, but ``state.version``
    cannot express it -- ``"2.0.0"`` is the value both at v0.17.3 and
    immediately before ``"3.0.0"`` was introduced. Detection is by SHAPE:

    1. ``.phenotypic/progress/image_complete/`` exists
    2. ``stage3_complete/`` exists. **NOT ``stage2_done/``** (U-9): that tree
       is current, not legacy, so firing on it would classify every modern GPU
       run ``CONVERT`` and strand it -- an INV-DISCHARGEABLE violation.
    3. ``processing_state.json`` carries ``datasets.<ds>.completed`` (deleted
       by §4.2)
    4. ``processing_state.json`` has ``config.work_ids`` and no
       ``config.restart_epoch``
    5. a **present, parseable, object-shaped** ``processing_state.json`` that
       carries ``version`` (or ``datasets``) and has **no ``work_ids`` key** --
       the pre-markers shape

    Shapes that classify without obvious behaviour, each with a test:

    ===================================== ==================================
    Shape                                 Verdict
    ===================================== ==================================
    fresh / absent output directory       ``None`` -- no schema to be wrong
                                          about
    standalone deliverables bundle        ``None`` -- no state file, so
                                          signal 5 cannot fire; supported by
                                          ``BundleLayout.detect``
    unreadable ``processing_state.json``  ``UNREADABLE_STATE``, never
                                          ``CONVERT``
    modern ``--mode process`` tree        ``CONVERT`` on signal 1 today
                                          (process publishes image markers),
                                          ``None`` once P3 converts its
                                          records
    pre-markers ``--mode process`` tree   ``CONVERT`` on signal 3 or 5; P7
                                          Task 2b's ported promoter is what
                                          discharges it (U-10)
    interrupted migrate                   ``CONVERT``; the re-run completes it
    retained ``.phenotypic/legacy-v2/``   ``None`` -- renamed aside, read by
                                          nothing
    ===================================== ==================================

    Reads the raw JSON directly. Never ``load_processing_state``, which writes
    via ``migrate_legacy_machine_state`` (``_cli_state_management.py:108``) and
    raises on an absent version (``:167``) -- a gate must not mutate the tree
    it is deciding about.

    Args:
        output_dir: Run output root; need not exist.

    Returns:
        The verdict, or ``None`` when the forward path can write to this tree.
    """
    return _classify(output_dir)[0]


def describe_required_conversion(
    output_dir: Path, *, mode: str
) -> str | None:
    """Return the **refusal** message for *output_dir*, or ``None``.

    Split from :func:`requires_conversion` so the verdict stays a plain enum
    while the evidence still reaches the user: a refusal the user cannot act on
    is the bug class this whole change exists to remove.

    Args:
        output_dir: Run output root.
        mode: The mode being refused, for the message.

    Returns:
        The message, or ``None`` when no conversion is required.
    """
    verdict, evidence = _classify(output_dir)
    if verdict is None:
        return None
    if verdict is ConversionVerdict.UNREADABLE_STATE:
        return (
            f"--mode {mode} cannot read this output: {evidence} is not "
            "readable as a processing state file (it must be a JSON object "
            'carrying "version" or "datasets"). Repair or remove that file; '
            "conversion cannot recover a state file it cannot read."
        )
    return (
        f"--mode {mode} cannot read this output: it was written before the "
        f"consolidated run-state schema ({evidence}). Convert it with:\n"
        f"  python -m phenotypic {MIGRATION_REMEDY} --output {output_dir}\n"
        f"{_DRAIN_ADVICE}"
    )


def describe_conversion_advisory(output_dir: Path) -> str | None:
    """Return the **reader-facing** note for *output_dir*, or ``None``.

    A separate function from :func:`describe_required_conversion` because the
    two make different *claims*, not merely different noises. The refusal says
    "cannot read this output"; the advisory exists precisely because the
    reader **can** read it -- spec §4.3 makes an unconverted tree an advisory
    on ``RunState`` and never a gate, so the GUI keeps displaying one. Reusing
    the refusal's wording here would tell a user their output is unreadable
    while they are looking at it.

    It takes no ``mode``: a reader has none.

    Args:
        output_dir: Run output root.

    Returns:
        The note, or ``None`` when no conversion is required. **Not** gated by
        :data:`SCHEMA_GATE_ARMED` -- the caller gates, because the caller is
        the surfacing.
    """
    verdict, evidence = _classify(output_dir)
    if verdict is None:
        return None
    if verdict is ConversionVerdict.UNREADABLE_STATE:
        return (
            f"This output's processing state ({evidence}) is not readable as "
            "a state file, so its completion cannot be established. What is "
            "displayed comes from the artifacts on disk. Repair or remove "
            "that file; conversion cannot recover one it cannot read."
        )
    return (
        "This output was written before the consolidated run-state schema "
        f"({evidence}). It is displayed as-is and remains readable; "
        f"reprocessing it requires `{MIGRATION_REMEDY}` first. Advisory "
        "only -- it does not gate the verdict."
    )
