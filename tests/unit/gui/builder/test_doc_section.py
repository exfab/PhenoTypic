"""Unit tests for the Inspector "Documentation" collapse section.

The pipeline-builder Inspector renders an operation's class docstring as a
collapsed-by-default ``dbc.Collapse`` so users can browse "what does this
op do?" without leaving the canvas. ``_doc_section_widgets`` is the pure
factory behind that section; it returns either:

* a single visible ``html.Div`` containing a toggle ``dbc.Button`` and a
  ``dbc.Collapse`` (when the operation has a non-empty docstring), or
* two hidden placeholder widgets carrying the same component ids (when
  the docstring is missing or whitespace-only).

The hidden-placeholder branch keeps Section 10 of ``register_callbacks``
resolvable on every Inspector render path — a missing id at click time
would crash the toggle callback. These tests pin both branches and the
toggle defaults so a future refactor can't silently break the contract.
"""

from __future__ import annotations

from typing import Any, Iterable

import pytest

from phenotypic.gui.builder import _ids as ids
from phenotypic.gui.builder._layout import _doc_section_widgets


def _walk_components(tree: Any) -> Iterable[Any]:
    """Yield every nested Dash component in a subtree (depth-first)."""

    yield tree
    children = getattr(tree, "children", None)
    if isinstance(children, (list, tuple)):
        for ch in children:
            yield from _walk_components(ch)
    elif children is not None and hasattr(children, "children"):
        yield from _walk_components(children)


def _find_by_id(components: Iterable[Any], component_id: str) -> Any:
    for c in components:
        if getattr(c, "id", None) == component_id:
            return c
    return None


class TestVisibleDocSection:
    """Branch: docstring is non-empty -> render the visible section."""

    def test_returns_single_div(self):
        widgets = _doc_section_widgets("Brief description.\n\nMore detail.")
        assert len(widgets) == 1

    def test_toggle_button_visible_with_label(self):
        widgets = _doc_section_widgets("Brief.")
        components = list(_walk_components(widgets[0]))

        toggle = _find_by_id(components, ids.INSPECTOR_DOC_TOGGLE)
        assert toggle is not None, "doc toggle must be present"
        assert toggle.children == "Documentation ▾"
        # Visible branch must NOT carry display:none. ``style`` may be
        # absent entirely on the visible button.
        style = getattr(toggle, "style", None)
        assert style is None or "display" not in style

    def test_collapse_starts_closed(self):
        widgets = _doc_section_widgets("Brief.")
        components = list(_walk_components(widgets[0]))

        collapse = _find_by_id(components, ids.INSPECTOR_DOC_COLLAPSE)
        assert collapse is not None, "doc collapse must be present"
        assert collapse.is_open is False, (
            "collapse must default to closed so the section is folded"
        )

    def test_docstring_is_cleandoc_normalized(self):
        # ``inspect.cleandoc`` strips uniform leading-line indentation and
        # outer blank lines; we should see the result, not the raw input.
        raw = "    First line.\n\n    Second line.\n    "
        widgets = _doc_section_widgets(raw)
        components = list(_walk_components(widgets[0]))

        # The cleaned text lives in an html.Pre inside the collapse.
        pre = next(
            (c for c in components if type(c).__name__ == "Pre"), None
        )
        assert pre is not None, "doc body must render as html.Pre"
        assert pre.children == "First line.\n\nSecond line."


class TestHiddenPlaceholders:
    """Branch: docstring is None / empty / whitespace -> hidden placeholders."""

    @pytest.mark.parametrize("docstring", [None, "", "   ", "\n\n  \n"])
    def test_returns_two_hidden_widgets(self, docstring):
        widgets = _doc_section_widgets(docstring)
        assert len(widgets) == 2

    @pytest.mark.parametrize("docstring", [None, "", "   "])
    def test_both_ids_present_with_display_none(self, docstring):
        widgets = _doc_section_widgets(docstring)
        ids_seen = {getattr(w, "id", None): w for w in widgets}

        assert ids.INSPECTOR_DOC_TOGGLE in ids_seen
        assert ids.INSPECTOR_DOC_COLLAPSE in ids_seen

        toggle = ids_seen[ids.INSPECTOR_DOC_TOGGLE]
        collapse = ids_seen[ids.INSPECTOR_DOC_COLLAPSE]
        assert toggle.style == {"display": "none"}
        assert collapse.style == {"display": "none"}

    def test_hidden_collapse_starts_closed(self):
        widgets = _doc_section_widgets(None)
        collapse = next(
            w for w in widgets if w.id == ids.INSPECTOR_DOC_COLLAPSE
        )
        # Even when hidden, ``is_open`` must default to False so the toggle
        # callback's ``State`` reads a consistent initial value.
        assert collapse.is_open is False


class TestInspectorRenderPathsEmitDocIds:
    """Every ``build_inspector`` branch must include both doc ids exactly once."""

    @pytest.fixture(scope="class")
    def registry(self):
        from phenotypic.gui._operation_registry import OperationRegistry

        reg = OperationRegistry()
        reg.discover()
        return reg

    def _doc_id_count(self, tree: Any) -> dict[str, int]:
        # The param form embedded in the non-pipeline Inspector branch
        # carries pattern-matching ids (dicts) for per-parameter widgets;
        # those are unhashable and must be skipped before the membership
        # test against our string-id counter.
        counts = {ids.INSPECTOR_DOC_TOGGLE: 0, ids.INSPECTOR_DOC_COLLAPSE: 0}
        for c in _walk_components(tree):
            cid = getattr(c, "id", None)
            if isinstance(cid, str) and cid in counts:
                counts[cid] += 1
        return counts

    def test_empty_inspector_branch_emits_doc_ids_once(self, registry):
        from phenotypic.gui.builder._layout import build_inspector
        from phenotypic.gui.builder._state import _LegacyBuilderScope as BuilderScope, _LegacyBuilderState as BuilderState

        state = BuilderState(
            root=BuilderScope(nodes=[]),
            breadcrumb=[],
            selected_node_id=None,
        )
        counts = self._doc_id_count(build_inspector(state, registry))
        assert counts[ids.INSPECTOR_DOC_TOGGLE] == 1
        assert counts[ids.INSPECTOR_DOC_COLLAPSE] == 1

    def test_pipeline_node_branch_emits_doc_ids_once(self, registry):
        from phenotypic.gui.builder._layout import build_inspector
        from phenotypic.gui.builder._state import (
            _LegacyBuilderScope as BuilderScope,
            _LegacyBuilderState as BuilderState,
            _LegacyStepNode as StepNode,
            PIPELINE_CLASS_NAME,
            _new_node_id,
        )

        node = StepNode(
            node_id=_new_node_id(),
            class_name=PIPELINE_CLASS_NAME,
        )
        scope = BuilderScope(nodes=[node])
        state = BuilderState(
            root=scope,
            breadcrumb=[],
            selected_node_id=node.node_id,
        )
        counts = self._doc_id_count(build_inspector(state, registry))
        assert counts[ids.INSPECTOR_DOC_TOGGLE] == 1
        assert counts[ids.INSPECTOR_DOC_COLLAPSE] == 1

    def test_operation_node_with_docstring_emits_doc_ids_once(self, registry):
        from phenotypic.gui.builder._layout import build_inspector
        from phenotypic.gui.builder._state import (
            _LegacyBuilderScope as BuilderScope,
            _LegacyBuilderState as BuilderState,
            _LegacyStepNode as StepNode,
            _new_node_id,
        )

        # BlurGauss is registered with a real Google-style docstring;
        # this branch should emit the *visible* toggle button.
        node = StepNode(
            node_id=_new_node_id(),
            class_name="BlurGauss",
        )
        scope = BuilderScope(nodes=[node])
        state = BuilderState(
            root=scope,
            breadcrumb=[],
            selected_node_id=node.node_id,
        )
        tree = build_inspector(state, registry)
        counts = self._doc_id_count(tree)
        assert counts[ids.INSPECTOR_DOC_TOGGLE] == 1
        assert counts[ids.INSPECTOR_DOC_COLLAPSE] == 1

        # Confirm the *visible* branch was taken: toggle has the label
        # text, not a display:none style.
        toggle = next(
            c for c in _walk_components(tree)
            if isinstance(getattr(c, "id", None), str)
            and c.id == ids.INSPECTOR_DOC_TOGGLE
        )
        assert toggle.children == "Documentation ▾"
        style = getattr(toggle, "style", None)
        assert style is None or "display" not in style
