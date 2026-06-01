"""Unit tests for the aux-port mutation kinds in :func:`_dispatch_state_update`.

Wave 5 (Agent B) of the aux-port popover redesign tracks the new
embedded-aux state model. Aux operations now live as embedded
:class:`StepNode` instances inside ``consumer.aux_ports[param][slot]``
(replacing the old free-floating ``aux_nodes`` list + ID references),
and aux lifecycle is tied to slot lifecycle — there is no separate
``aux_add`` / ``aux_delete`` step. The dispatch kinds covered here are:

* ``wire_create`` — materialise a fresh aux ``StepNode`` from a class
  name and embed it at ``consumer.aux_ports[param][slot]``. Auto-focuses
  the new aux in the inspector.
* ``wire_delete`` — clear the slot (drop the embedded aux) and clear
  ``inspector_focus_aux`` if it was pointing at the slot.
* ``port_slot_add`` / ``port_slot_remove`` — grow / shrink list-typed
  port slot lists.
* ``drill_in_aux`` — push an aux-slot breadcrumb segment so the canvas
  refocuses on the wired aux's scope.
* ``set_inspector_focus`` — set or clear the ``inspector_focus_aux``
  override (swaps the inspector pane between consumer params and a
  wired aux's params without changing the canvas selection).

Tests are JSON-in / JSON-out without booting Dash, mirroring the
style of :mod:`tests.gui.builder.test_callbacks`.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from phenotypic.gui._operation_registry import OperationRegistry
from phenotypic.gui.builder._app import create_app
from phenotypic.gui.builder._callbacks import _dispatch_state_update
from phenotypic.gui.builder._state import (
    _LegacyBuilderScope as BuilderScope,
    _LegacyBuilderState as BuilderState,
    StepNode,
    state_to_json,
)


@pytest.fixture(scope="module")
def app_ctx() -> Any:
    """Yield an active Flask app-context with the registry stashed.

    ``_dispatch_state_update`` reads the registry off ``current_app.config``
    so we need a Flask app context active for any test that triggers a
    kind whose validator consults the registry (``wire_create``,
    ``port_slot_add``, ``port_slot_remove``).
    """

    registry = OperationRegistry()
    registry.discover()
    app = create_app(registry=registry)
    with app.server.app_context():
        yield app


def _state_with_consumer(
    class_name: str, node_id: str = "consumer"
) -> Dict[str, Any]:
    """Return a JSON state dict with one main-ribbon node of *class_name*.

    The consumer node starts with an empty ``aux_ports`` map; tests that
    need pre-wired aux can either run ``wire_create`` themselves or
    construct the embedded shape directly on the returned dict.
    """

    return state_to_json(
        BuilderState(
            root=BuilderScope(
                nodes=[StepNode(node_id=node_id, class_name=class_name)],
                name="root",
            )
        )
    )


def _dispatch(
    state: Dict[str, Any], kind: str, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Thin wrapper around :func:`_dispatch_state_update`.

    Exists purely as a readable alias inside the test bodies — every
    test follows the JSON-in / JSON-out pattern and benefits from a
    short call site.
    """

    return _dispatch_state_update(state, kind, payload)


# ---------------------------------------------------------------------------
# wire_create
# ---------------------------------------------------------------------------


def test_wire_create_materializes_embedded_aux(app_ctx: Any) -> None:
    """``wire_create`` embeds a fresh aux ``StepNode`` at the given slot.

    The new aux carries default params for *class_name*, gets a fresh
    8-char node id, and the dispatch also sets ``inspector_focus_aux``
    so the user can immediately edit the aux's params without leaving
    the canvas selection.
    """

    state = _state_with_consumer("FilamentousFungiDetector", node_id="fungi")
    out = _dispatch(
        state,
        "wire_create",
        {
            "target_node_id": "fungi",
            "param": "inoculum_detector",
            "slot": 0,
            "class_name": "OtsuDetector",
        },
    )

    fungi = out["root"]["nodes"][0]
    slot_list = fungi["aux_ports"]["inoculum_detector"]
    assert len(slot_list) == 1
    aux = slot_list[0]
    assert isinstance(aux, dict)
    assert aux["class_name"] == "OtsuDetector"
    # Fresh node id is an 8-char hex string.
    assert isinstance(aux["node_id"], str)
    assert len(aux["node_id"]) == 8
    # ``params`` is a dict (even if empty for OtsuDetector defaults).
    assert isinstance(aux["params"], dict)
    # Auto-focus: the new aux is now the inspector's focus.
    assert out["inspector_focus_aux"] == {
        "target_node_id": "fungi",
        "param": "inoculum_detector",
        "slot": 0,
    }
    # Input state is not mutated.
    assert state["root"]["nodes"][0].get("aux_ports") in ({}, None)


def test_wire_create_rejects_unknown_class(app_ctx: Any) -> None:
    """An unknown ``class_name`` makes ``wire_create`` a no-op.

    The aux registry lookup fails to materialise the embedded node so
    the slot is left untouched and ``inspector_focus_aux`` is not set.
    """

    state = _state_with_consumer("FilamentousFungiDetector", node_id="fungi")
    out = _dispatch(
        state,
        "wire_create",
        {
            "target_node_id": "fungi",
            "param": "inoculum_detector",
            "slot": 0,
            "class_name": "DoesNotExist",
        },
    )

    fungi = out["root"]["nodes"][0]
    assert fungi.get("aux_ports", {}).get("inoculum_detector") is None
    assert out.get("inspector_focus_aux") is None


def test_wire_create_rejects_type_incompatible_class(app_ctx: Any) -> None:
    """Wiring a non-ImageOperation source is rejected by type validation.

    ``EdgeCorrector`` is a :class:`SetAnalyzer`, NOT an
    :class:`ImageOperation` subclass.
    ``FilamentousFungiDetector.inoculum_detector`` accepts
    ``ObjectDetector | ImagePipeline`` only, so the dispatch returns
    state unchanged and the slot remains unwired.
    """

    state = _state_with_consumer("FilamentousFungiDetector", node_id="fungi")
    out = _dispatch(
        state,
        "wire_create",
        {
            "target_node_id": "fungi",
            "param": "inoculum_detector",
            "slot": 0,
            "class_name": "EdgeCorrector",
        },
    )

    fungi = out["root"]["nodes"][0]
    assert fungi.get("aux_ports", {}).get("inoculum_detector") is None
    assert out.get("inspector_focus_aux") is None


def test_wire_create_grows_slot_list_for_list_typed(app_ctx: Any) -> None:
    """For list-typed ports, wiring at an unallocated slot extends the list.

    ``CompositeDetector.detectors`` is list-typed; wiring at slot 2 on
    an empty consumer must pad slots 0 and 1 with ``None`` placeholders
    so the resulting list has length 3 with only slot 2 occupied.
    """

    state = _state_with_consumer("CompositeDetector", node_id="composite")
    out = _dispatch(
        state,
        "wire_create",
        {
            "target_node_id": "composite",
            "param": "detectors",
            "slot": 2,
            "class_name": "OtsuDetector",
        },
    )

    composite = out["root"]["nodes"][0]
    slots = composite["aux_ports"]["detectors"]
    assert len(slots) == 3
    assert slots[0] is None
    assert slots[1] is None
    assert isinstance(slots[2], dict)
    assert slots[2]["class_name"] == "OtsuDetector"


# ---------------------------------------------------------------------------
# wire_delete
# ---------------------------------------------------------------------------


def test_wire_delete_drops_aux_and_clears_focus(app_ctx: Any) -> None:
    """Disconnecting a wired aux drops the embedded ``StepNode`` and clears focus.

    After ``wire_create`` (which auto-focuses), ``wire_delete`` on the
    same slot must blank the slot and reset ``inspector_focus_aux`` so
    the inspector falls back to the canvas-selected consumer.
    """

    state = _state_with_consumer("FilamentousFungiDetector", node_id="fungi")
    state = _dispatch(
        state,
        "wire_create",
        {
            "target_node_id": "fungi",
            "param": "inoculum_detector",
            "slot": 0,
            "class_name": "OtsuDetector",
        },
    )
    # Sanity: focus is now on the new aux.
    assert state["inspector_focus_aux"] is not None

    out = _dispatch(
        state,
        "wire_delete",
        {
            "target_node_id": "fungi",
            "param": "inoculum_detector",
            "slot": 0,
        },
    )

    fungi = out["root"]["nodes"][0]
    assert fungi["aux_ports"]["inoculum_detector"] == [None]
    assert out["inspector_focus_aux"] is None


def test_wire_delete_other_slot_preserves_focus(app_ctx: Any) -> None:
    """Disconnecting a different slot leaves the focus on the still-wired one.

    Wire slot 0 + slot 1 of a list-typed port, set focus on slot 0,
    then disconnect slot 1. Focus on slot 0 must remain set; only the
    matching-slot ``wire_delete`` clears focus.
    """

    state = _state_with_consumer("CompositeDetector", node_id="composite")
    # Wire slot 0 (auto-focuses on slot 0).
    state = _dispatch(
        state,
        "wire_create",
        {
            "target_node_id": "composite",
            "param": "detectors",
            "slot": 0,
            "class_name": "OtsuDetector",
        },
    )
    # Wire slot 1 (auto-focus moves to slot 1).
    state = _dispatch(
        state,
        "wire_create",
        {
            "target_node_id": "composite",
            "param": "detectors",
            "slot": 1,
            "class_name": "RoundPeaksDetector",
        },
    )
    # Restore focus on slot 0 explicitly so the test exercises the
    # "focus stays on slot 0 when slot 1 is deleted" branch.
    state = _dispatch(
        state,
        "set_inspector_focus",
        {
            "focus": "aux",
            "target_node_id": "composite",
            "param": "detectors",
            "slot": 0,
        },
    )
    assert state["inspector_focus_aux"]["slot"] == 0

    out = _dispatch(
        state,
        "wire_delete",
        {
            "target_node_id": "composite",
            "param": "detectors",
            "slot": 1,
        },
    )

    composite = out["root"]["nodes"][0]
    slots = composite["aux_ports"]["detectors"]
    # Slot 0 still wired; slot 1 cleared.
    assert isinstance(slots[0], dict)
    assert slots[1] is None
    # Focus on slot 0 preserved.
    assert out["inspector_focus_aux"] == {
        "target_node_id": "composite",
        "param": "detectors",
        "slot": 0,
    }


# ---------------------------------------------------------------------------
# port_slot_add / port_slot_remove
# ---------------------------------------------------------------------------


def test_port_slot_add_appends_none(app_ctx: Any) -> None:
    """``port_slot_add`` appends a ``None`` slot to a list-typed port.

    Scalar ports are not affected (the dispatch is a no-op for them);
    only list-typed ports grow.
    """

    state = _state_with_consumer("CompositeDetector", node_id="composite")
    out = _dispatch(
        state,
        "port_slot_add",
        {"node_id": "composite", "param": "detectors"},
    )
    assert out["root"]["nodes"][0]["aux_ports"]["detectors"] == [None]

    # Adding twice grows to two placeholders.
    out = _dispatch(
        out,
        "port_slot_add",
        {"node_id": "composite", "param": "detectors"},
    )
    assert out["root"]["nodes"][0]["aux_ports"]["detectors"] == [None, None]


def test_port_slot_remove_pops(app_ctx: Any) -> None:
    """``port_slot_remove`` deletes the indexed slot and reindexes the list.

    Builds three wired slots, removes the middle one, and asserts that
    the remaining list contains exactly slot 0 and slot 2 (re-indexed
    to positions 0 and 1).
    """

    state = _state_with_consumer("CompositeDetector", node_id="composite")
    # Wire three slots in sequence.
    for slot in range(3):
        state = _dispatch(
            state,
            "wire_create",
            {
                "target_node_id": "composite",
                "param": "detectors",
                "slot": slot,
                "class_name": "OtsuDetector",
            },
        )
    pre_slots = state["root"]["nodes"][0]["aux_ports"]["detectors"]
    slot0_id = pre_slots[0]["node_id"]
    slot2_id = pre_slots[2]["node_id"]

    out = _dispatch(
        state,
        "port_slot_remove",
        {"node_id": "composite", "param": "detectors", "slot": 1},
    )

    remaining = out["root"]["nodes"][0]["aux_ports"]["detectors"]
    assert len(remaining) == 2
    assert remaining[0]["node_id"] == slot0_id
    assert remaining[1]["node_id"] == slot2_id


# ---------------------------------------------------------------------------
# drill_in_aux
# ---------------------------------------------------------------------------


def test_drill_in_aux_pushes_breadcrumb_and_clears_focus(app_ctx: Any) -> None:
    """``drill_in_aux`` pushes a ``{target_node_id, param, slot}`` segment.

    The canvas scope swap takes over so ``inspector_focus_aux`` is
    cleared — the inspector now belongs to the new (drilled-into)
    scope, not the parent's wired aux.
    """

    state = _state_with_consumer("FilamentousFungiDetector", node_id="fungi")
    state = _dispatch(
        state,
        "wire_create",
        {
            "target_node_id": "fungi",
            "param": "inoculum_detector",
            "slot": 0,
            "class_name": "OtsuDetector",
        },
    )
    # Sanity: wire_create set focus.
    assert state["inspector_focus_aux"] is not None

    out = _dispatch(
        state,
        "drill_in_aux",
        {
            "target_node_id": "fungi",
            "param": "inoculum_detector",
            "slot": 0,
        },
    )

    assert out["breadcrumb"] == [
        {
            "target_node_id": "fungi",
            "param": "inoculum_detector",
            "slot": 0,
        }
    ]
    assert out["selected_node_id"] is None
    assert out["inspector_focus_aux"] is None


def test_drill_in_aux_rejects_empty_slot(app_ctx: Any) -> None:
    """Drilling into an unwired slot is a no-op.

    Without an embedded aux to drill into, the breadcrumb must not
    grow and the state is returned unchanged.
    """

    state = _state_with_consumer("FilamentousFungiDetector", node_id="fungi")
    # No wire_create — slot is missing entirely.

    out = _dispatch(
        state,
        "drill_in_aux",
        {
            "target_node_id": "fungi",
            "param": "inoculum_detector",
            "slot": 0,
        },
    )

    assert out["breadcrumb"] == []
    assert out.get("inspector_focus_aux") is None


# ---------------------------------------------------------------------------
# set_inspector_focus
# ---------------------------------------------------------------------------


def test_set_inspector_focus_aux(app_ctx: Any) -> None:
    """``set_inspector_focus`` with ``focus="aux"`` records the slot.

    Wire an aux, clear the auto-set focus via a ``"consumer"`` focus,
    then explicitly re-set it via ``"aux"`` — the resulting focus must
    point at the wired slot.
    """

    state = _state_with_consumer("FilamentousFungiDetector", node_id="fungi")
    state = _dispatch(
        state,
        "wire_create",
        {
            "target_node_id": "fungi",
            "param": "inoculum_detector",
            "slot": 0,
            "class_name": "OtsuDetector",
        },
    )
    # Clear the auto-set focus first to isolate the set_inspector_focus path.
    state = _dispatch(state, "set_inspector_focus", {"focus": "consumer"})
    assert state["inspector_focus_aux"] is None

    out = _dispatch(
        state,
        "set_inspector_focus",
        {
            "focus": "aux",
            "target_node_id": "fungi",
            "param": "inoculum_detector",
            "slot": 0,
        },
    )

    assert out["inspector_focus_aux"] == {
        "target_node_id": "fungi",
        "param": "inoculum_detector",
        "slot": 0,
    }


def test_set_inspector_focus_consumer_clears(app_ctx: Any) -> None:
    """``set_inspector_focus`` with ``focus="consumer"`` clears the override.

    After ``wire_create`` auto-focuses on the new aux, dispatching
    ``set_inspector_focus`` with ``focus="consumer"`` (or any non-aux
    value) must reset ``inspector_focus_aux`` to ``None`` so the
    inspector falls back to the canvas-selected node's params.
    """

    state = _state_with_consumer("FilamentousFungiDetector", node_id="fungi")
    state = _dispatch(
        state,
        "wire_create",
        {
            "target_node_id": "fungi",
            "param": "inoculum_detector",
            "slot": 0,
            "class_name": "OtsuDetector",
        },
    )
    assert state["inspector_focus_aux"] is not None

    out = _dispatch(state, "set_inspector_focus", {"focus": "consumer"})
    assert out["inspector_focus_aux"] is None


def test_set_inspector_focus_aux_rejects_empty_slot(app_ctx: Any) -> None:
    """Trying to focus on a ``None`` slot is rejected — focus stays put.

    A list-typed port with an empty slot 0 cannot be focused via
    ``set_inspector_focus({"focus": "aux", ...})``; the validator
    rejects the payload and the existing ``inspector_focus_aux`` value
    (whatever it was) is preserved.
    """

    state = _state_with_consumer("CompositeDetector", node_id="composite")
    # Grow the port to one empty slot.
    state = _dispatch(
        state,
        "port_slot_add",
        {"node_id": "composite", "param": "detectors"},
    )
    assert state["root"]["nodes"][0]["aux_ports"]["detectors"] == [None]
    # Stash a sentinel focus value so we can prove it survives the
    # rejected dispatch.
    state["inspector_focus_aux"] = {"sentinel": True}

    out = _dispatch(
        state,
        "set_inspector_focus",
        {
            "focus": "aux",
            "target_node_id": "composite",
            "param": "detectors",
            "slot": 0,
        },
    )

    assert out["inspector_focus_aux"] == {"sentinel": True}
