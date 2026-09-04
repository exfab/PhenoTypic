"""Read-only resolution of a run's completion state.

**Readers only.** Spec §5.2 makes the read/write asymmetry structural: every
function that *publishes* state stays in :mod:`phenotypic._cli`, so a GUI
import of this module cannot reach one. INV-LAYER
(``tests/unit/sdk_/test_run_state_layering.py``) enforces both halves -- no
``phenotypic._cli`` import, and no writer in ``__all__``.

This module reads ``processing_state.json`` as plain JSON and never replays the
event log. That is possible because spec §4.2 demotes the event log out of the
evidence set and deletes ``processing_state.datasets.{completed,failed,started}``
from the file: what remains that a verdict depends on is ``config.work_ids``
and the digests, all literal JSON fields. See OPEN-QUESTIONS Q4.

The four frozen dataclasses are defined in
:mod:`phenotypic.sdk_._state_types` and re-exported here, which is where the
spec's function surface puts them. They live one module down so that
:mod:`phenotypic.sdk_._verification_cache` can cache whole ``ImageState``
objects without this module and that one importing each other.
"""

from __future__ import annotations

from ._state_types import ImageState, RunDiagnostics, RunIdentity, RunState

#: Grows one name at a time, in the task that defines it. ``run_identity``,
#: ``assert_identity_current``, ``finalization_input_object`` and
#: ``resolve_run_state`` are named by spec §5.2 and belong here, but listing a
#: name this module does not yet bind is ruff **F822** -- an error under the
#: default ``F`` rule set this repo runs -- so each arrives with its own
#: implementation. Keeping the two in step is also what keeps every commit
#: importable, which is the phase-gate contract.
__all__ = [
    "ImageState",
    "RunDiagnostics",
    "RunIdentity",
    "RunState",
]
