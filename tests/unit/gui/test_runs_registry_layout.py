"""The runs registry resolves run status from new and legacy layouts.

P6 Task 4 retired ``RunRegistry._read_status_from_manifest``. These tests
followed the capability, not the symbol: what they pinned was that a
**pre-migration legacy tree still reports a status instead of ``unknown``**,
and that survives the migration -- the reader changed from
``resolve_manifest_json_path`` to :func:`~phenotypic.sdk_.resolve_run_state`,
whose ``resolve_processing_state_path`` carries the same legacy fallback
(``_io_constants.py:1127-1135``).
"""
from __future__ import annotations

import json
from pathlib import Path

from phenotypic._gui.shell._runs_registry import RunRegistry
from phenotypic.sdk_ import processing_state_path
from tests._output_layout import build_complete_run


def test_status_read_finds_new_layout(tmp_path: Path) -> None:
    """A tree this build wrote keeps its state under ``.phenotypic/``."""
    output = build_complete_run(tmp_path)
    assert processing_state_path(output).is_file()

    status, _ = RunRegistry._rehydrated_status(output)

    assert status == "complete"


def test_status_read_finds_legacy_layout(tmp_path: Path) -> None:
    """A pre-migration tree keeps its state at the output root.

    Fires if the reader stops going through ``resolve_processing_state_path``
    and hard-codes ``.phenotypic/``: the moved file would then be invisible
    and every legacy run in a sandbox would rehydrate as ``unknown``.
    """
    output = build_complete_run(tmp_path)
    current = processing_state_path(output)
    legacy = output / current.name
    legacy.write_bytes(current.read_bytes())
    current.unlink()
    assert not processing_state_path(output).exists()

    status, _ = RunRegistry._rehydrated_status(output)

    assert status == "complete"


def test_a_manifest_alone_is_not_a_status(tmp_path: Path) -> None:
    """Spec §4.2: ``manifest.json`` is no longer evidence in either layout.

    The retired reader answered ``complete`` for exactly this tree, in both
    the ``.phenotypic/progress/`` and the legacy ``progress/`` spelling. It is
    a cache of what a run reported and is never rewritten when the tree
    beneath it changes, so nothing reads it here now.
    """
    output = tmp_path / "legacy-manifest-only"
    (output / "progress").mkdir(parents=True)
    (output / "progress" / "manifest.json").write_text(
        json.dumps({"is_complete": True, "execution_mode": "local"}),
        encoding="utf-8",
    )

    status, _ = RunRegistry._rehydrated_status(output)

    assert status == "unknown"
