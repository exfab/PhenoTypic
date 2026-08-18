"""Back-compat shim for the Space view, which is now split in two.

Pure half — :mod:`phenotypic._services.tune_spec`: infer a search space, apply
per-knob edits, build a :class:`~phenotypic.tune.TuningSpec`, load a run's spec
or pipeline. Dash-free, so the MCP server can call it.

Dash half — :mod:`phenotypic.gui.tune._space_view`: the knob-form rendering and
the Space view body.

Both halves re-export the *same* objects here, so ``space_to_spec`` is one
function no matter which path a caller imports it through. The private names
travel too: ``_callbacks.py:2227`` imports ``_load_space_source``, and
``tests/unit/gui/tune/test_space.py`` imports ``_apply_edits`` and ``_knob_form``
by name.

**The view half resolves lazily** (PEP 562), the same way
:mod:`phenotypic.gui.tune` resolves ``create_app``. An eager
``from ._space_view import build_space_view`` here would put ``dash`` back on
this module's import path and leave the split cosmetic: ``_setup_authoring``
imports ``space_to_spec`` through this shim and would pay for a renderer it
never calls. ``tests/unit/services/test_lazy_gui_packages.py`` holds that line.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from phenotypic._services.tune_spec import (  # noqa: F401
    _apply_edits,
    _build_search_space,
    _default_qc_scorer,
    _editable_knobs,
    _is_tuning_spec,
    _load_space_source,
    _recover_typed_choices,
    _try_load_pipeline,
    _try_load_spec,
    apply_space_edits,
    space_to_spec,
)

if TYPE_CHECKING:  # type-checker only; never executed at runtime
    from phenotypic.gui.tune._space_view import (  # noqa: F401
        _REVIEW_BADGE,
        _categorical_input,
        _domain_editor,
        _knob_form,
        _range_inputs,
        _tunable_toggle,
        build_space_view,
        setup_knob_forms,
    )

#: Every name this shim forwards to the Dash half, resolved on first access.
_VIEW_NAMES = frozenset(
    {
        "_REVIEW_BADGE",
        "_categorical_input",
        "_domain_editor",
        "_knob_form",
        "_range_inputs",
        "_tunable_toggle",
        "build_space_view",
        "setup_knob_forms",
    }
)

__all__ = [
    "apply_space_edits",
    "build_space_view",
    "setup_knob_forms",
    "space_to_spec",
]


def __getattr__(name: str) -> Any:
    if name in _VIEW_NAMES:
        from phenotypic.gui.tune import _space_view

        return getattr(_space_view, name)
    raise AttributeError(name)
