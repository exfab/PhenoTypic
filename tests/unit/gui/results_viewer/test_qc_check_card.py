"""Unit tests for the per-card builder in the QC tab.

The card builder is a side-effect-free pure function: it accepts one
:class:`~phenotypic.gui._qc_recipe.QcRecipeEntry` and returns a fully
mounted :class:`dash_bootstrap_components.Card`. The tests walk the
returned component tree to confirm every pattern-matching id Wave E's
callbacks rely on is present.
"""
from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]

from phenotypic.analysis import ReplicateAgreement
from phenotypic.gui._qc_recipe import QcRecipeEntry
from phenotypic.gui.results_viewer._qc_tab import _ids as ids
from phenotypic.gui.results_viewer._qc_tab._check_card import build_check_card


def _make_entry(instance_id: str = "qc-SE-deadbeef") -> QcRecipeEntry:
    """Build a minimal entry for the card-builder tests."""
    return QcRecipeEntry(
        cls=ReplicateAgreement,
        params={"on": "Size_Area", "groupby": ["Plate"]},
        instance_id=instance_id,
        enabled=True,
    )


def _walk(component: Any):
    """Yield every component (including the root) recursively."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, list):
        for child in children:
            if child is None:
                continue
            yield from _walk(child)
    else:
        yield from _walk(children)


def _find_id(component: Any, target_id: Any) -> Any:
    """Return the first descendant whose ``id`` equals ``target_id``."""
    for node in _walk(component):
        node_id = getattr(node, "id", None)
        if node_id == target_id:
            return node
    return None


def test_build_check_card_returns_dbc_card() -> None:
    """The card builder returns a ``dbc.Card`` instance."""
    card = build_check_card(_make_entry())
    assert isinstance(card, dbc.Card)


def test_card_has_pattern_matching_figure_id() -> None:
    """The figure ``dcc.Graph`` carries the qc-card-figure pattern id."""
    instance_id = "qc-SE-cafe1234"
    card = build_check_card(_make_entry(instance_id))
    figure_node = _find_id(card, ids.qc_card_figure_id(instance_id))
    assert figure_node is not None
    assert figure_node.id == {"type": "qc-card-figure", "index": instance_id}


def test_card_has_status_badge_for_status() -> None:
    """The status badge is mounted with the qc-card-status-badge pattern id."""
    instance_id = "qc-SE-99887766"
    card = build_check_card(_make_entry(instance_id))
    badge_node = _find_id(card, ids.qc_card_status_badge_id(instance_id))
    assert badge_node is not None
    assert badge_node.id == {
        "type": "qc-card-status-badge",
        "index": instance_id,
    }


def test_card_has_edit_toggle_duplicate_delete_buttons() -> None:
    """Four lifecycle buttons are present with the right pattern ids."""
    instance_id = "qc-SE-77665544"
    card = build_check_card(_make_entry(instance_id))
    for builder in (
        ids.qc_card_edit_id,
        ids.qc_card_toggle_id,
        ids.qc_card_duplicate_id,
        ids.qc_card_delete_id,
    ):
        node = _find_id(card, builder(instance_id))
        assert node is not None, f"button id {builder(instance_id)} missing"


def test_card_has_mark_flagged_button() -> None:
    """The Mark-flagged-for-removal button uses the qc-card-mark-flag id."""
    instance_id = "qc-SE-11223344"
    card = build_check_card(_make_entry(instance_id))
    node = _find_id(card, ids.qc_card_mark_flag_id(instance_id))
    assert node is not None
    assert node.id == {"type": "qc-card-mark-flag", "index": instance_id}
