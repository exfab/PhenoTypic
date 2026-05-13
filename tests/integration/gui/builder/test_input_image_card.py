"""Integration tests for the Input Image inspector card (spec §4.5).

Exercises the DAG branch of
:func:`phenotypic.gui.builder._layout.build_inspector` for the
"Input Image selected" state.  When ``state.selected_block_id``
resolves to a block with ``class_name == INPUT_IMAGE_CLASS_NAME``,
the inspector renders a dedicated card carrying:

* Heading: ``"Input Image — pipeline source"``.
* A description paragraph explaining the runtime-loader contract.
* A ``Re-anchor`` button (id :data:`ids.BTN_REANCHOR`) that
  dispatches the ``reanchor`` viewport op.
* A "Re-layout" affordance — rendered as a labelled button forwarded
  to the toolbar's canonical :data:`ids.BTN_RELAYOUT` (Dash forbids
  duplicate ids; the card uses a CSS-selector hook instead).
* No param form (InputImage has no parameters).
* No Delete button (spec §4.1 — the Input Image sentinel cannot be
  deleted because every scope must contain exactly one).

The tests walk the rendered :class:`dash.html.Div` tree to assert
these invariants without booting Dash — keeping the suite fast and
independent of clientside JS.
"""

from __future__ import annotations

from phenotypic.gui.builder import _ids as ids
from phenotypic.gui.builder._layout import (
    _build_dag_inspector,
    _build_input_image_card,
)
from phenotypic.gui.builder._state import (
    INPUT_IMAGE_CLASS_NAME,
    _DagBuilderScope,
    _DagBuilderState,
)

# Component-tree walking helpers shared with the other inspector
# integration tests; see ``conftest.py`` in this directory.
from .conftest import _collect_text, _find_by_id, _walk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_with_input_image_selected() -> _DagBuilderState:
    """Construct a state whose Input Image sentinel is selected."""

    scope = _DagBuilderScope()
    # ``_DagBuilderScope.__post_init__`` auto-seeds the InputImage block.
    input_image = scope.blocks[0]
    assert input_image.class_name == INPUT_IMAGE_CLASS_NAME
    state = _DagBuilderState(root=scope)
    state.selected_block_id = input_image.block_id
    state.selected_edge_id = None
    return state


# ---------------------------------------------------------------------------
# Card-level smoke tests
# ---------------------------------------------------------------------------


def test_input_image_card_renders_when_input_image_selected() -> None:
    """Selecting the Input Image sentinel shows the dedicated card."""

    state = _state_with_input_image_selected()
    inspector = _build_dag_inspector(state, registry=None)  # type: ignore[arg-type]

    cards = _find_by_id(inspector, ids.INSPECTOR_INPUT_IMAGE_CARD)
    assert len(cards) == 1, (
        f"Expected one Input Image card; found {len(cards)}: "
        f"{[getattr(c, 'id', None) for c in cards]}"
    )


def test_input_image_card_has_no_param_form_fields() -> None:
    """The Input Image card emits no param form (InputImage has no params)."""

    state = _state_with_input_image_selected()
    inspector = _build_dag_inspector(state, registry=None)  # type: ignore[arg-type]

    # The param-form container ``INSPECTOR_PARAM_FORM`` must not appear.
    forms = _find_by_id(inspector, ids.INSPECTOR_PARAM_FORM)
    assert len(forms) == 0, (
        f"Expected no param form on the Input Image card; "
        f"found {len(forms)}: {forms}"
    )


def test_input_image_card_has_re_layout_button() -> None:
    """The card surfaces a Re-layout affordance.

    Dash forbids duplicating ``ids.BTN_RELAYOUT`` (which lives on the
    toolbar); the card uses a label-only button with the sentinel
    class ``inspector-input-image-relayout-btn`` as its test handle.
    The button must be reachable through the card so the spec §4.5
    surface is complete.
    """

    state = _state_with_input_image_selected()
    inspector = _build_dag_inspector(state, registry=None)  # type: ignore[arg-type]

    text = _collect_text(inspector)
    assert "Re-layout" in text, (
        f"Expected the Re-layout label in the card; rendered text: {text!r}"
    )

    # Walk the component tree and look for a button carrying the
    # sentinel className.  The button has no Dash id (would clash with
    # the toolbar's BTN_RELAYOUT).  Reuses ``_walk`` from the shared
    # builder integration conftest instead of redefining it inline.
    proxy_buttons: list = []
    for node in _walk(inspector):
        class_name = getattr(node, "className", None)
        if isinstance(class_name, str) and (
            "inspector-input-image-relayout-btn" in class_name
        ):
            proxy_buttons.append(node)

    assert len(proxy_buttons) == 1, (
        "Expected exactly one Re-layout proxy button on the Input Image card"
    )


def test_input_image_card_has_re_anchor_button() -> None:
    """``ids.BTN_REANCHOR`` is present and clickable."""

    state = _state_with_input_image_selected()
    inspector = _build_dag_inspector(state, registry=None)  # type: ignore[arg-type]

    btns = _find_by_id(inspector, ids.BTN_REANCHOR)
    assert len(btns) == 1, (
        f"Expected exactly one Re-anchor button; found {len(btns)}: {btns}"
    )
    assert btns[0].n_clicks == 0

    # Sanity: the button text is the spec-mandated label.
    text = _collect_text(btns[0])
    assert "Re-anchor view to Input Image" in text


def test_input_image_card_has_no_delete_button() -> None:
    """Spec §4.1: the Input Image sentinel cannot be deleted.

    The :data:`ids.BTN_DELETE_NODE` and :data:`ids.BTN_DELETE_WIRE`
    must not be rendered inside the Input Image card.  ``BTN_DRILL_IN``
    is allowed as a hidden placeholder (the fan-in callback's
    ``Input(BTN_DRILL_IN)`` requires the id to resolve) but it
    carries the ``style: display: none`` flag so the user never sees
    it.
    """

    state = _state_with_input_image_selected()
    inspector = _build_dag_inspector(state, registry=None)  # type: ignore[arg-type]

    # No visible delete buttons.
    assert _find_by_id(inspector, ids.BTN_DELETE_NODE) == []
    assert _find_by_id(inspector, ids.BTN_DELETE_WIRE) == []

    # ``BTN_DRILL_IN`` may be rendered as a hidden placeholder; if it
    # is, it must be hidden (display: none) so it doesn't render an
    # interactive surface.
    drill_btns = _find_by_id(inspector, ids.BTN_DRILL_IN)
    for btn in drill_btns:
        style = getattr(btn, "style", None) or {}
        assert style.get("display") == "none", (
            f"BTN_DRILL_IN found visible on Input Image card: style={style}"
        )


def test_input_image_card_helper_returns_correct_container_id() -> None:
    """``_build_input_image_card`` returns a Div with INSPECTOR_CONTAINER id."""

    scope = _DagBuilderScope()
    input_image = scope.blocks[0]
    state = _DagBuilderState(root=scope, selected_block_id=input_image.block_id)

    div = _build_input_image_card(state, input_image)
    assert getattr(div, "id", None) == ids.INSPECTOR_CONTAINER


def test_input_image_card_replaces_empty_state_on_selection() -> None:
    """Selecting Input Image swaps the empty-state card for the input-image card.

    Before any selection the inspector shows ``ids.INSPECTOR_EMPTY_STATE``;
    once the InputImage block is selected, the card swaps to
    ``ids.INSPECTOR_INPUT_IMAGE_CARD``.
    """

    # Empty state — no selection.
    empty_state = _DagBuilderState(root=_DagBuilderScope())
    empty_inspector = _build_dag_inspector(empty_state, registry=None)  # type: ignore[arg-type]
    assert _find_by_id(empty_inspector, ids.INSPECTOR_EMPTY_STATE)
    assert not _find_by_id(empty_inspector, ids.INSPECTOR_INPUT_IMAGE_CARD)

    # Now select the Input Image block.
    state = _state_with_input_image_selected()
    inspector = _build_dag_inspector(state, registry=None)  # type: ignore[arg-type]
    assert _find_by_id(inspector, ids.INSPECTOR_INPUT_IMAGE_CARD)
    assert not _find_by_id(inspector, ids.INSPECTOR_EMPTY_STATE)


# ---------------------------------------------------------------------------
# Empty-state placeholder (spec §4.5)
# ---------------------------------------------------------------------------


def test_empty_state_placeholder_describes_validation_badge() -> None:
    """The empty-state card mentions the toolbar badge so users orient.

    Spec §4.5: the empty placeholder shows "Drag an operation from
    the palette to begin." plus a one-line hint about the validation
    badge.  Both copy strings must appear in the rendered text.
    """

    state = _DagBuilderState(root=_DagBuilderScope())
    inspector = _build_dag_inspector(state, registry=None)  # type: ignore[arg-type]

    text = _collect_text(inspector)
    assert "Drag an operation from the palette to begin." in text
    # The hint copy mentions "badge" and "validation" (one-liner).
    assert "badge" in text.lower()
    assert "validation" in text.lower()


def test_empty_state_placeholder_has_stable_id() -> None:
    """The placeholder uses ``ids.INSPECTOR_EMPTY_STATE`` as its handle."""

    state = _DagBuilderState(root=_DagBuilderScope())
    inspector = _build_dag_inspector(state, registry=None)  # type: ignore[arg-type]

    placeholders = _find_by_id(inspector, ids.INSPECTOR_EMPTY_STATE)
    assert len(placeholders) == 1, (
        f"Expected one empty-state placeholder; got {len(placeholders)}"
    )
