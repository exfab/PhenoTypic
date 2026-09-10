"""Refuse a run output still written in the pre-consolidation shape.

Spec D1 (clean break), §11.1. Every mode that writes or reprocesses --
``full``, ``measure``, ``recompile``, ``process`` -- refuses an unconverted
tree and names ``--mode migrate``. ``migrate`` itself is exempt: it is the
remedy, and guarding it with its own predicate makes the tree unmigratable
(ledger MIG-19). Readers are exempt too: spec §4.3 makes an unconverted tree
an **advisory** on ``RunState``, never a gate, so the GUI keeps displaying one.

**The detection moved to :mod:`phenotypic.sdk_._schema_shape`; the refusal
stayed here.** This module's earlier docstring drew the line correctly -- *"a
refusal that raises ``click.UsageError`` is the other side of that line"* --
but kept the predicate on this side of it, and the predicate is a pure reader
with no ``phenotypic._cli`` dependency at all. INV-LAYER forced the correction
when ``resolve_run_state`` needed that same detection to emit §4.3's advisory
and could not import this package. A second copy would have let the GUI's
advisory and this refusal disagree about whether a tree needs migrating, which
is CAN-4's shape.

What is left here is exactly what belongs on the writer side: the one function
that imports ``click`` and raises. The rest is re-exported unchanged, so every
existing importer of this module is untouched.
"""

from __future__ import annotations

from pathlib import Path

from phenotypic.sdk_ import _schema_shape
from phenotypic.sdk_._schema_shape import (
    STATE_SCHEMA_VERSION,
    ConversionVerdict,
    describe_required_conversion,
    requires_conversion,
)

# Re-exported for `test_the_stage3_directory_name_matches_the_writer`, which
# anchors this hand-joined segment against its real writer *through this
# module*. The redundant alias is the explicit re-export form, so the name is
# not read as an unused import.
from phenotypic.sdk_._schema_shape import (
    _DIR_STAGE3_COMPLETE as _DIR_STAGE3_COMPLETE,
)

__all__ = [
    "STATE_SCHEMA_VERSION",
    "ConversionVerdict",
    "describe_required_conversion",
    "refuse_unconverted_schema",
    "requires_conversion",
]


def refuse_unconverted_schema(output_dir: Path, *, mode: str) -> None:
    """Raise when *output_dir* holds a schema the forward path cannot read.

    Called for ``full``, ``measure``, ``recompile`` and ``process`` from
    :func:`phenotypic.phenotypicCLI._refuse_unmigrated_output`, so the two
    reasons a tree is unwritable share one call site and one severity story.

    Inert until ``_schema_shape.SCHEMA_GATE_ARMED``; see that constant for
    why, and for the test that fails when arming is overdue.

    It is read **through the module**, and this module deliberately does not
    re-export the name. One flag, one home, one patch point: a re-exported
    copy here would still read correctly while being **inert under
    monkeypatch**, so a test patching ``_cli_schema_gate.SCHEMA_GATE_ARMED``
    -- the name on the module it is testing -- would change nothing, and the
    flag is the last place anyone looks.
    ``test_the_arming_flag_has_one_source`` pins that this module never
    reintroduces such a copy.

    Args:
        output_dir: Run output root.
        mode: The mode being refused, for the message.

    Raises:
        click.UsageError: The tree needs converting, or its state file cannot
            be read.
    """
    if not _schema_shape.SCHEMA_GATE_ARMED:
        return
    message = describe_required_conversion(output_dir, mode=mode)
    if message is None:
        return
    import click

    raise click.UsageError(message)
