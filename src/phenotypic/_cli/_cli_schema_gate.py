"""Refuse a run output still written in the pre-consolidation shape.

Spec D1 (clean break), §11.1. Every mode that writes or reprocesses --
``full``, ``measure``, ``recompile``, ``process`` -- refuses an unconverted
tree and names ``--mode migrate``. ``migrate`` itself is exempt: it is the
remedy, and guarding it with its own predicate makes the tree unmigratable
(ledger MIG-19). Readers are exempt too: spec §4.3 makes a half-migrated tree
an **advisory** on ``RunState``, never a gate, so the GUI keeps displaying one.

**Why the gate has to exist before the consolidated record does (CAN-11).**
P3 makes a clean break -- ``publish_image_success`` writes the new per-image
record and ``valid_image_success`` reads it. On an unconverted tree
``authorized_measurement_sources`` then yields an **empty mapping**, and ``{}``
is a *valid* schema-3 answer meaning "nothing has succeeded yet", not a
failure. ``finalize_run`` would publish an empty master and raise nothing: a
successful-looking run that discarded every measurement. This module is what
turns that into an error, so it is built in P1 rather than P7.

**This is a CLI-side, writer-adjacent module and does not belong in ``sdk_``.**
INV-LAYER keeps ``sdk_/_run_state.py`` free of ``phenotypic._cli``; a refusal
that raises ``click.UsageError`` is the other side of that line.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Final, cast

from phenotypic.sdk_ import (
    DIR_IMAGE_COMPLETE,
    MIGRATION_REMEDY,
    resolve_processing_state_path,
    resolve_progress_dir,
)

__all__ = [
    "SCHEMA_GATE_ARMED",
    "STATE_SCHEMA_VERSION",
    "ConversionVerdict",
    "describe_required_conversion",
    "refuse_unconverted_schema",
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

#: Whether the refusal is live. **Flipped to ``True`` by P3 Task 2**, in the
#: same commit that makes ``publish_image_success`` write the consolidated
#: record -- see ``test_the_gate_is_armed_exactly_when_the_forward_path_stops``
#: ``_writing_the_legacy_marker``, which fails the moment those two disagree.
#:
#: It is ``False`` today because at P1 the legacy shape and the **current**
#: shape are the same shape: the forward path still writes
#: ``image_complete/``, still writes ``datasets.<ds>.completed``, and does not
#: yet write ``restart_epoch``, so three of the five signals below fire on a
#: tree the running build has just written. An armed gate would refuse every
#: resume of every mode. Detection is correct now; only the refusal waits.
SCHEMA_GATE_ARMED: bool = False

#: ``<progress>/stage3_complete/``. A module-private literal for the same
#: reason ``_cli_stage2_token._STAGE2_DIR`` is one: the segment is hand-joined
#: by its writer (``_cli_staged_resume.stage3_completion_marker_path``) and P3
#: deletes the tree, so promoting it to ``sdk_/_io_constants.py`` now would add
#: a constant this change is about to remove.
#: ``test_the_stage3_directory_name_matches_the_writer`` pins the two
#: together meanwhile.
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
    """Return the refusal message for *output_dir*, or ``None``.

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


def refuse_unconverted_schema(output_dir: Path, *, mode: str) -> None:
    """Raise when *output_dir* holds a schema the forward path cannot read.

    Called for ``full``, ``measure``, ``recompile`` and ``process`` from
    :func:`phenotypic.phenotypicCLI._refuse_unmigrated_output`, so the two
    reasons a tree is unwritable share one call site and one severity story.

    Inert until :data:`SCHEMA_GATE_ARMED`; see that constant for why, and for
    the test that fails when arming is overdue.

    Args:
        output_dir: Run output root.
        mode: The mode being refused, for the message.

    Raises:
        click.UsageError: The tree needs converting, or its state file cannot
            be read.
    """
    if not SCHEMA_GATE_ARMED:
        return
    message = describe_required_conversion(output_dir, mode=mode)
    if message is None:
        return
    import click

    raise click.UsageError(message)
