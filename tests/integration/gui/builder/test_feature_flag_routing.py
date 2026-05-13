"""Test that the ``PHENOTYPIC_GUI_DAG`` feature flag drives layout routing.

Spec §5.7a — the flag is read once at module import of
``phenotypic.gui.builder._state``.  Phase 2 wires the new
``build_canvas_elements_dag`` renderer behind the flag; until the flag
is on, the legacy ``build_canvas_elements`` continues to drive the
canvas.

This test exercises both halves:

* When the flag is **off**, ``BuilderScope`` / ``BuilderState`` resolve
  to the legacy types; the renderer dispatch in ``_callbacks._render_views``
  selects ``build_canvas_elements`` (the linear-list renderer).
* When the flag is **on** (verified via a subprocess so a process
  restart simulates a real cold start), ``BuilderScope`` /
  ``BuilderState`` resolve to the DAG types.

Subprocess isolation keeps the parent test process's class identities
stable — reloading ``_state`` in-process would corrupt sibling tests
that imported the DAG dataclasses before the reload.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from phenotypic.gui.builder import _layout, _state


def test_flag_off_resolves_to_legacy_types() -> None:
    """Default (flag off) → ``BuilderScope`` is the legacy linear-list type."""

    if _state.PHENOTYPIC_GUI_DAG:
        pytest.skip("Flag is on in this process; covered by subprocess test")
    # Public type alias points at the legacy class.
    assert _state.BuilderScope is _state._LegacyBuilderScope
    assert _state.BuilderState is _state._LegacyBuilderState


def test_dag_render_function_is_always_available() -> None:
    """``build_canvas_elements_dag`` is importable regardless of the flag.

    Per spec §5.7a, the DAG schema dataclasses are *always* defined so
    sibling modules can import them.  The renderer needs the same
    treatment — the legacy callbacks shouldn't break if a developer
    imports ``build_canvas_elements_dag`` outside the flag-gated code
    paths.
    """

    assert callable(_layout.build_canvas_elements_dag)
    assert callable(_layout.build_canvas_elements)
    # The two functions are distinct.
    assert _layout.build_canvas_elements is not _layout.build_canvas_elements_dag


def test_dag_render_accepts_dag_scope() -> None:
    """``build_canvas_elements_dag`` works on a DAG scope independent of flag."""

    # The DAG dataclasses are always defined per spec §5.7a even when the
    # flag is off.  Importing them gives us a real DAG scope to render.
    from phenotypic.gui.builder._state import _DagBuilderScope

    scope = _DagBuilderScope()
    elements = _layout.build_canvas_elements_dag(scope)
    assert isinstance(elements, list)
    # The auto-seeded InputImage block produces one node + one output port.
    assert len(elements) >= 2


def test_subprocess_with_flag_on_resolves_to_dag_types() -> None:
    """Subprocess with ``PHENOTYPIC_GUI_DAG=1`` flips ``BuilderScope`` to DAG."""

    script = (
        "import os, json, sys\n"
        "from phenotypic.gui.builder import _state\n"
        "result = {\n"
        "    'flag': _state.PHENOTYPIC_GUI_DAG,\n"
        "    'is_dag_scope': _state.BuilderScope is _state._DagBuilderScope,\n"
        "    'is_dag_state': _state.BuilderState is _state._DagBuilderState,\n"
        "}\n"
        "json.dump(result, sys.stdout)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env={"PHENOTYPIC_GUI_DAG": "1", "PATH": _safe_path()},
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(proc.stdout)
    assert result["flag"] is True
    assert result["is_dag_scope"] is True
    assert result["is_dag_state"] is True


def test_subprocess_with_flag_off_resolves_to_legacy_types() -> None:
    """Subprocess without ``PHENOTYPIC_GUI_DAG`` keeps the legacy types active."""

    script = (
        "import os, json, sys\n"
        "from phenotypic.gui.builder import _state\n"
        "result = {\n"
        "    'flag': _state.PHENOTYPIC_GUI_DAG,\n"
        "    'is_legacy_scope': _state.BuilderScope is _state._LegacyBuilderScope,\n"
        "    'is_legacy_state': _state.BuilderState is _state._LegacyBuilderState,\n"
        "}\n"
        "json.dump(result, sys.stdout)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env={"PATH": _safe_path()},
        capture_output=True,
        text=True,
        check=True,
    )
    result = json.loads(proc.stdout)
    assert result["flag"] is False
    assert result["is_legacy_scope"] is True
    assert result["is_legacy_state"] is True


def test_callbacks_module_imports_legacy_renderer() -> None:
    """The legacy callback path still imports ``build_canvas_elements``.

    Phase 2 leaves the legacy callbacks intact; the DAG-flag-on routing
    lands in a later phase.  This guard keeps the legacy import alive
    so the existing fan-in callback resolves.
    """

    from phenotypic.gui.builder import _callbacks

    assert hasattr(_callbacks, "build_canvas_elements"), (
        "Legacy build_canvas_elements must remain importable from "
        "_callbacks until Phase 7 removes it"
    )


def _safe_path() -> str:
    """Return a minimal PATH so the subprocess can locate Python."""

    import os

    return os.environ.get("PATH", "/usr/bin:/bin")
