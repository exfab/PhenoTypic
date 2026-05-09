"""Unit tests for the aux-port mutation kinds in :func:`_dispatch_state_update`.

Wave 3 (Agent B) of the Galaxy-style aux input ports plan adds seven new
dispatch kinds to the pipeline builder: ``aux_add``, ``aux_delete``,
``wire_create``, ``wire_delete``, ``port_slot_add``, ``port_slot_remove``,
and ``drill_in_aux``.  Each is exercised here in JSON-in / JSON-out form
without booting Dash, mirroring the testing style of
:mod:`tests.gui.builder.test_callbacks`.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from phenotypic.gui._operation_registry import OperationRegistry
from phenotypic.gui.builder._app import create_app
from phenotypic.gui.builder._callbacks import _dispatch_state_update
from phenotypic.gui.builder._state import (
    BuilderScope,
    BuilderState,
    StepNode,
    from_pipeline,
    state_from_json,
    state_to_json,
    to_pipeline,
)


@pytest.fixture(scope="module")
def app_ctx() -> Any:
    """Yield an active Flask app-context with the registry stashed.

    ``_dispatch_state_update`` reads the registry off ``current_app.config``
    so we need a Flask request/app context active for any test that
    triggers a kind whose validator consults the registry (``aux_add``,
    ``wire_create``, ``port_slot_add``, ``port_slot_remove``).
    """

    registry = OperationRegistry()
    registry.discover()
    app = create_app(registry=registry)
    with app.server.app_context():
        yield app


def _empty_state() -> Dict[str, Any]:
    """Return a JSON state dict with an empty root scope (no nodes)."""

    return state_to_json(BuilderState(root=BuilderScope(name="root")))


def _state_with_consumer(class_name: str, node_id: str = "consumer") -> Dict[str, Any]:
    """Return a JSON state dict with one main-ribbon node of *class_name*."""

    return state_to_json(
        BuilderState(
            root=BuilderScope(
                nodes=[StepNode(node_id=node_id, class_name=class_name)],
                name="root",
            )
        )
    )


# ---------------------------------------------------------------------------
# aux_add
# ---------------------------------------------------------------------------


def test_aux_add_appends_to_scope(app_ctx: Any) -> None:
    """``aux_add`` creates a fresh aux node in the current scope's ``aux_nodes``."""

    state = _empty_state()
    out = _dispatch_state_update(state, "aux_add", {"class_name": "OtsuDetector"})

    aux_nodes = out["root"]["aux_nodes"]
    assert len(aux_nodes) == 1
    aux = aux_nodes[0]
    assert aux["class_name"] == "OtsuDetector"
    # Node-id is a fresh 8-char hex string (not bound to any consumer).
    assert isinstance(aux["node_id"], str)
    assert len(aux["node_id"]) == 8
    # Original input is not mutated.
    assert state["root"]["aux_nodes"] == []


def test_aux_add_unknown_class_is_noop(app_ctx: Any) -> None:
    """``aux_add`` with an unknown class name returns the unchanged state."""

    state = _empty_state()
    out = _dispatch_state_update(
        state, "aux_add", {"class_name": "NoSuchClass"}
    )

    assert out["root"]["aux_nodes"] == []


# ---------------------------------------------------------------------------
# wire_create
# ---------------------------------------------------------------------------


def test_wire_create_sets_slot(app_ctx: Any) -> None:
    """Wiring a compatible aux into a scalar port populates ``aux_ports[param]``."""

    state = _state_with_consumer("FilamentousFungiDetector", node_id="fungi")
    state = _dispatch_state_update(
        state, "aux_add", {"class_name": "OtsuDetector"}
    )
    aux_id = state["root"]["aux_nodes"][0]["node_id"]

    out = _dispatch_state_update(
        state,
        "wire_create",
        {
            "source_aux_id": aux_id,
            "target_node_id": "fungi",
            "param": "inoculum_detector",
            "slot": 0,
        },
    )
    fungi = out["root"]["nodes"][0]
    assert fungi["aux_ports"]["inoculum_detector"] == [aux_id]


def test_wire_create_rejects_type_mismatch(app_ctx: Any) -> None:
    """Wiring a non-ImageOperation source (e.g. an analysis Filter) is rejected.

    ``EdgeCorrector`` is a :class:`SetAnalyzer`, NOT an
    :class:`ImageOperation` subclass.  ``FilamentousFungiDetector.inoculum_detector``
    accepts ``ObjectDetector | ImagePipeline``; the dispatch should
    return state unchanged and leave ``aux_ports`` empty / unset.
    """

    state = _state_with_consumer("FilamentousFungiDetector", node_id="fungi")
    state = _dispatch_state_update(
        state, "aux_add", {"class_name": "EdgeCorrector"}
    )
    aux_id = state["root"]["aux_nodes"][0]["node_id"]

    out = _dispatch_state_update(
        state,
        "wire_create",
        {
            "source_aux_id": aux_id,
            "target_node_id": "fungi",
            "param": "inoculum_detector",
            "slot": 0,
        },
    )
    fungi = out["root"]["nodes"][0]
    # No wire was created; the aux_ports map is untouched.
    assert fungi.get("aux_ports", {}).get("inoculum_detector") in (None,)


def test_wire_create_initializes_missing_aux_ports_dict(app_ctx: Any) -> None:
    """The target's ``aux_ports`` key is created on demand when wiring."""

    state = _state_with_consumer("FilamentousFungiDetector", node_id="fungi")
    # Strip the auto-generated aux_ports dict to simulate an older save shape.
    state["root"]["nodes"][0].pop("aux_ports", None)

    state = _dispatch_state_update(
        state, "aux_add", {"class_name": "OtsuDetector"}
    )
    aux_id = state["root"]["aux_nodes"][0]["node_id"]

    out = _dispatch_state_update(
        state,
        "wire_create",
        {
            "source_aux_id": aux_id,
            "target_node_id": "fungi",
            "param": "inoculum_detector",
            "slot": 0,
        },
    )
    fungi = out["root"]["nodes"][0]
    assert "aux_ports" in fungi
    assert fungi["aux_ports"]["inoculum_detector"] == [aux_id]


# ---------------------------------------------------------------------------
# wire_delete
# ---------------------------------------------------------------------------


def test_wire_delete_orphans_aux_node(app_ctx: Any) -> None:
    """``wire_delete`` clears the slot but leaves the aux in ``aux_nodes``."""

    state = _state_with_consumer("FilamentousFungiDetector", node_id="fungi")
    state = _dispatch_state_update(
        state, "aux_add", {"class_name": "OtsuDetector"}
    )
    aux_id = state["root"]["aux_nodes"][0]["node_id"]
    state = _dispatch_state_update(
        state,
        "wire_create",
        {
            "source_aux_id": aux_id,
            "target_node_id": "fungi",
            "param": "inoculum_detector",
            "slot": 0,
        },
    )

    out = _dispatch_state_update(
        state,
        "wire_delete",
        {"target_node_id": "fungi", "param": "inoculum_detector", "slot": 0},
    )
    fungi = out["root"]["nodes"][0]
    assert fungi["aux_ports"]["inoculum_detector"] == [None]
    # Aux node still in the dock as an orphan.
    assert any(a["node_id"] == aux_id for a in out["root"]["aux_nodes"])


# ---------------------------------------------------------------------------
# port_slot_add / port_slot_remove
# ---------------------------------------------------------------------------


def test_port_slot_add_grows_list_only(app_ctx: Any) -> None:
    """``port_slot_add`` appends ``None`` for list-typed ports; no-op for scalar."""

    # Composite ribbon node — ``detectors`` is list-typed.
    list_state = _state_with_consumer("CompositeDetector", node_id="composite")
    out = _dispatch_state_update(
        list_state,
        "port_slot_add",
        {"node_id": "composite", "param": "detectors"},
    )
    composite = out["root"]["nodes"][0]
    assert composite["aux_ports"]["detectors"] == [None]

    # Add another slot to verify it appends rather than replacing.
    out = _dispatch_state_update(
        out,
        "port_slot_add",
        {"node_id": "composite", "param": "detectors"},
    )
    composite = out["root"]["nodes"][0]
    assert composite["aux_ports"]["detectors"] == [None, None]

    # Scalar ribbon node — ``inoculum_detector`` is non-list; should no-op.
    scalar_state = _state_with_consumer(
        "FilamentousFungiDetector", node_id="fungi"
    )
    out = _dispatch_state_update(
        scalar_state,
        "port_slot_add",
        {"node_id": "fungi", "param": "inoculum_detector"},
    )
    fungi = out["root"]["nodes"][0]
    # Either the key was never created or remains scalar/empty.
    assert "inoculum_detector" not in fungi.get("aux_ports", {})


def test_port_slot_remove_pops_index(app_ctx: Any) -> None:
    """``port_slot_remove`` deletes the indexed slot and reindexes the list."""

    state = _state_with_consumer("CompositeDetector", node_id="composite")
    # Populate three slots manually with placeholder ids ``a``, ``b``, ``c``.
    state["root"]["nodes"][0]["aux_ports"] = {"detectors": ["a", "b", "c"]}

    out = _dispatch_state_update(
        state,
        "port_slot_remove",
        {"node_id": "composite", "param": "detectors", "slot": 1},
    )
    composite = out["root"]["nodes"][0]
    assert composite["aux_ports"]["detectors"] == ["a", "c"]


# ---------------------------------------------------------------------------
# aux_delete
# ---------------------------------------------------------------------------


def test_aux_delete_clears_dependent_wires(app_ctx: Any) -> None:
    """Deleting an aux clears all wires referencing it across the scope."""

    # Build: two CompositeDetector consumers + one OtsuDetector aux wired
    # into both of them at slot 0.
    state = state_to_json(
        BuilderState(
            root=BuilderScope(
                nodes=[
                    StepNode(node_id="cd1", class_name="CompositeDetector"),
                    StepNode(node_id="cd2", class_name="CompositeDetector"),
                ],
                name="root",
            )
        )
    )
    state = _dispatch_state_update(
        state, "aux_add", {"class_name": "OtsuDetector"}
    )
    aux_id = state["root"]["aux_nodes"][0]["node_id"]
    state = _dispatch_state_update(
        state,
        "wire_create",
        {
            "source_aux_id": aux_id,
            "target_node_id": "cd1",
            "param": "detectors",
            "slot": 0,
        },
    )
    state = _dispatch_state_update(
        state,
        "wire_create",
        {
            "source_aux_id": aux_id,
            "target_node_id": "cd2",
            "param": "detectors",
            "slot": 0,
        },
    )

    # Sanity: both wires exist.
    cd1 = state["root"]["nodes"][0]
    cd2 = state["root"]["nodes"][1]
    assert cd1["aux_ports"]["detectors"] == [aux_id]
    assert cd2["aux_ports"]["detectors"] == [aux_id]

    out = _dispatch_state_update(state, "aux_delete", {"node_id": aux_id})

    # Aux is gone.
    assert all(
        a["node_id"] != aux_id for a in out["root"]["aux_nodes"]
    )
    # Both consumer slots are now ``None``.
    cd1 = out["root"]["nodes"][0]
    cd2 = out["root"]["nodes"][1]
    assert cd1["aux_ports"]["detectors"] == [None]
    assert cd2["aux_ports"]["detectors"] == [None]


# ---------------------------------------------------------------------------
# drill_in_aux
# ---------------------------------------------------------------------------


def test_drill_in_aux_pushes_breadcrumb(app_ctx: Any) -> None:
    """``drill_in_aux`` appends an ``{"aux_id": ...}`` segment to the breadcrumb."""

    state = _empty_state()
    state = _dispatch_state_update(
        state, "aux_add", {"class_name": "OtsuDetector"}
    )
    aux_id = state["root"]["aux_nodes"][0]["node_id"]

    out = _dispatch_state_update(state, "drill_in_aux", {"aux_id": aux_id})
    assert out["breadcrumb"] == [{"aux_id": aux_id, "param": None}]
    assert out["selected_node_id"] is None


# ---------------------------------------------------------------------------
# Wave 4: orphan-aux detection at save time
# ---------------------------------------------------------------------------


def test_collect_orphan_aux_ids_returns_empty_for_wired_aux(app_ctx: Any) -> None:
    """An aux node wired into a consumer is NOT an orphan."""

    from phenotypic.gui.builder._callbacks import _collect_orphan_aux_ids

    state = _state_with_consumer("FilamentousFungiDetector", node_id="fungi")
    state = _dispatch_state_update(
        state, "aux_add", {"class_name": "OtsuDetector"}
    )
    aux_id = state["root"]["aux_nodes"][0]["node_id"]
    state = _dispatch_state_update(
        state,
        "wire_create",
        {
            "source_aux_id": aux_id,
            "target_node_id": "fungi",
            "param": "inoculum_detector",
            "slot": 0,
        },
    )

    assert _collect_orphan_aux_ids(state) == []


def test_save_emits_orphan_warning_toast(app_ctx: Any) -> None:
    """Aux nodes with no consumer are surfaced by ``_collect_orphan_aux_ids``."""

    from phenotypic.gui.builder._callbacks import _collect_orphan_aux_ids

    state = _empty_state()
    state = _dispatch_state_update(
        state, "aux_add", {"class_name": "OtsuDetector"}
    )
    aux_id = state["root"]["aux_nodes"][0]["node_id"]

    orphans = _collect_orphan_aux_ids(state)
    assert orphans == [aux_id]


def test_collect_orphan_aux_ids_handles_partial_wiring(app_ctx: Any) -> None:
    """Two aux nodes, only one wired -> the unwired one is reported as orphan."""

    from phenotypic.gui.builder._callbacks import _collect_orphan_aux_ids

    state = _state_with_consumer("FilamentousFungiDetector", node_id="fungi")
    state = _dispatch_state_update(
        state, "aux_add", {"class_name": "OtsuDetector"}
    )
    state = _dispatch_state_update(
        state, "aux_add", {"class_name": "RoundPeaksDetector"}
    )
    wired_id = state["root"]["aux_nodes"][0]["node_id"]
    orphan_id = state["root"]["aux_nodes"][1]["node_id"]
    state = _dispatch_state_update(
        state,
        "wire_create",
        {
            "source_aux_id": wired_id,
            "target_node_id": "fungi",
            "param": "inoculum_detector",
            "slot": 0,
        },
    )

    orphans = _collect_orphan_aux_ids(state)
    assert orphans == [orphan_id]


def test_orphan_collection_handles_nested_scopes(app_ctx: Any) -> None:
    """Aux nodes inside a nested ImagePipeline scope are walked too.

    A wired aux inside the nested scope must not be reported; an orphan
    aux inside the same nested scope must be reported.  The outer scope's
    walk also catches its own orphans, so the helper returns the union.
    """

    from phenotypic.gui.builder._callbacks import _collect_orphan_aux_ids

    state = state_to_json(
        BuilderState(
            root=BuilderScope(
                nodes=[
                    StepNode(
                        node_id="outer-pipe",
                        class_name="ImagePipeline",
                        nested=BuilderScope(
                            nodes=[
                                StepNode(
                                    node_id="inner-fungi",
                                    class_name="FilamentousFungiDetector",
                                ),
                            ],
                            name="inner",
                        ),
                    ),
                ],
                name="outer",
            )
        )
    )
    # Drill into the nested pipeline so dispatch operates on the inner scope.
    state["breadcrumb"] = [{"node_id": "outer-pipe", "param": None}]

    state = _dispatch_state_update(
        state, "aux_add", {"class_name": "OtsuDetector"}
    )
    state = _dispatch_state_update(
        state, "aux_add", {"class_name": "RoundPeaksDetector"}
    )
    nested_scope = state["root"]["nodes"][0]["nested"]
    wired_id = nested_scope["aux_nodes"][0]["node_id"]
    nested_orphan_id = nested_scope["aux_nodes"][1]["node_id"]
    state = _dispatch_state_update(
        state,
        "wire_create",
        {
            "source_aux_id": wired_id,
            "target_node_id": "inner-fungi",
            "param": "inoculum_detector",
            "slot": 0,
        },
    )

    # Pop back to root scope and add another orphan there.
    state["breadcrumb"] = []
    state = _dispatch_state_update(
        state, "aux_add", {"class_name": "OtsuDetector"}
    )
    outer_orphan_id = state["root"]["aux_nodes"][-1]["node_id"]

    orphans = _collect_orphan_aux_ids(state)
    assert outer_orphan_id in orphans
    assert nested_orphan_id in orphans
    # The wired inner aux must NOT be flagged.
    assert wired_id not in orphans


def test_validate_main_ribbon_linear_passes_for_v1_chain(app_ctx: Any) -> None:
    """The v1 main ribbon is always linear by construction."""

    from phenotypic.gui.builder._callbacks import _validate_main_ribbon_linear

    state = state_to_json(
        BuilderState(
            root=BuilderScope(
                nodes=[
                    StepNode(node_id="a", class_name="GaussianBlur"),
                    StepNode(node_id="b", class_name="OtsuDetector"),
                ],
                name="root",
            )
        )
    )
    # Doesn't raise.
    _validate_main_ribbon_linear(state)


def test_validate_main_ribbon_linear_raises_when_root_missing(app_ctx: Any) -> None:
    """Defensive: callers can rely on the helper to flag malformed payloads."""

    from phenotypic.gui.builder._callbacks import _validate_main_ribbon_linear

    with pytest.raises(ValueError, match="root scope"):
        _validate_main_ribbon_linear({})


# ---------------------------------------------------------------------------
# Wave 4: pending-wire state machine (click-then-click flow)
# ---------------------------------------------------------------------------


def test_pending_wire_starts_from_port_handle(app_ctx: Any) -> None:
    """Tapping a port handle while pending=None records a port endpoint."""

    from phenotypic.gui.builder._callbacks import _resolve_pending_wire_tap
    from phenotypic.gui.builder._layout import _encode_port_handle_id

    state = _state_with_consumer("FilamentousFungiDetector", node_id="fungi")
    handle = _encode_port_handle_id("fungi", "inoculum_detector", 0)

    new_pending, completion, recognized = _resolve_pending_wire_tap(
        state, None, handle
    )
    assert recognized is True
    assert completion is None
    assert new_pending == {
        "endpoint_kind": "port",
        "node_id": "fungi",
        "param": "inoculum_detector",
        "slot": 0,
    }


def test_pending_wire_starts_from_aux_node(app_ctx: Any) -> None:
    """Tapping an aux node while pending=None records an aux endpoint."""

    from phenotypic.gui.builder._callbacks import _resolve_pending_wire_tap

    state = _empty_state()
    state = _dispatch_state_update(
        state, "aux_add", {"class_name": "OtsuDetector"}
    )
    aux_id = state["root"]["aux_nodes"][0]["node_id"]

    new_pending, completion, recognized = _resolve_pending_wire_tap(
        state, None, aux_id
    )
    assert recognized is True
    assert completion is None
    assert new_pending == {"endpoint_kind": "aux", "aux_id": aux_id}


def test_pending_wire_completes_aux_then_port(app_ctx: Any) -> None:
    """Aux pending + port tapped -> wire_create payload, store cleared."""

    from phenotypic.gui.builder._callbacks import _resolve_pending_wire_tap
    from phenotypic.gui.builder._layout import _encode_port_handle_id

    state = _state_with_consumer("FilamentousFungiDetector", node_id="fungi")
    state = _dispatch_state_update(
        state, "aux_add", {"class_name": "OtsuDetector"}
    )
    aux_id = state["root"]["aux_nodes"][0]["node_id"]
    pending = {"endpoint_kind": "aux", "aux_id": aux_id}
    handle = _encode_port_handle_id("fungi", "inoculum_detector", 0)

    new_pending, completion, recognized = _resolve_pending_wire_tap(
        state, pending, handle
    )
    assert recognized is True
    assert completion == "wire_create"
    # In completion mode the helper packs the dispatch payload, NOT a real
    # pending shape — caller is responsible for clearing the store.
    assert new_pending == {
        "source_aux_id": aux_id,
        "target_node_id": "fungi",
        "param": "inoculum_detector",
        "slot": 0,
    }


def test_pending_wire_completes_port_then_aux(app_ctx: Any) -> None:
    """Port pending + aux tapped -> wire_create payload (mirror of the above)."""

    from phenotypic.gui.builder._callbacks import _resolve_pending_wire_tap

    state = _state_with_consumer("FilamentousFungiDetector", node_id="fungi")
    state = _dispatch_state_update(
        state, "aux_add", {"class_name": "OtsuDetector"}
    )
    aux_id = state["root"]["aux_nodes"][0]["node_id"]
    pending = {
        "endpoint_kind": "port",
        "node_id": "fungi",
        "param": "inoculum_detector",
        "slot": 0,
    }

    new_pending, completion, recognized = _resolve_pending_wire_tap(
        state, pending, aux_id
    )
    assert recognized is True
    assert completion == "wire_create"
    assert new_pending == {
        "source_aux_id": aux_id,
        "target_node_id": "fungi",
        "param": "inoculum_detector",
        "slot": 0,
    }


def test_pending_wire_re_taps_clear_pending(app_ctx: Any) -> None:
    """Tapping the same port twice cancels the pending wire."""

    from phenotypic.gui.builder._callbacks import _resolve_pending_wire_tap
    from phenotypic.gui.builder._layout import _encode_port_handle_id

    state = _state_with_consumer("FilamentousFungiDetector", node_id="fungi")
    handle = _encode_port_handle_id("fungi", "inoculum_detector", 0)
    pending = {
        "endpoint_kind": "port",
        "node_id": "fungi",
        "param": "inoculum_detector",
        "slot": 0,
    }

    new_pending, completion, recognized = _resolve_pending_wire_tap(
        state, pending, handle
    )
    assert recognized is True
    assert completion is None
    assert new_pending is None


def test_pending_wire_main_ribbon_tap_is_unrecognized(app_ctx: Any) -> None:
    """Taps on main-ribbon nodes fall through to ``select_node`` (recognized=False)."""

    from phenotypic.gui.builder._callbacks import _resolve_pending_wire_tap

    state = _state_with_consumer("GaussianBlur", node_id="gauss")
    new_pending, completion, recognized = _resolve_pending_wire_tap(
        state, None, "gauss"
    )
    assert recognized is False
    assert completion is None
    # ``new_pending`` is the unchanged input when the tap isn't handled.
    assert new_pending is None


# ---------------------------------------------------------------------------
# Wave 4: aux-palette helper (drop-on-port shortcut)
# ---------------------------------------------------------------------------


def test_last_aux_id_for_class_picks_rightmost(app_ctx: Any) -> None:
    """``_last_aux_id_for_class`` returns the most-recently-added aux of a class."""

    from phenotypic.gui.builder._callbacks import _last_aux_id_for_class

    state = _empty_state()
    state = _dispatch_state_update(
        state, "aux_add", {"class_name": "OtsuDetector"}
    )
    state = _dispatch_state_update(
        state, "aux_add", {"class_name": "RoundPeaksDetector"}
    )
    state = _dispatch_state_update(
        state, "aux_add", {"class_name": "OtsuDetector"}
    )

    expected = state["root"]["aux_nodes"][-1]["node_id"]
    assert _last_aux_id_for_class(state, "OtsuDetector") == expected
    assert _last_aux_id_for_class(state, "NoSuchClass") is None


# ---------------------------------------------------------------------------
# Wave 5: end-to-end user-session integration
# ---------------------------------------------------------------------------


class TestAuxEndToEnd:
    """Higher-level scenarios that mirror a realistic user session.

    The earlier tests exercise individual dispatch kinds in isolation.
    These walk through the same kinds in sequence — adding a main-ribbon
    op, an aux op, wiring them, then crossing the JSON / runtime
    boundary in both directions — to catch regressions at the seams
    between dispatch, ``to_pipeline``, and ``from_pipeline``.
    """

    def test_user_session_add_filamentous_with_inoculum_aux(
        self, app_ctx: Any
    ) -> None:
        """Ribbon op + aux op + wire + save/reload survives the full loop.

        Simulates: empty canvas -> add ``FilamentousFungiDetector`` to
        the ribbon -> add ``OtsuDetector`` as aux -> wire the aux into
        the consumer's ``inoculum_detector`` port -> serialize through
        ``to_pipeline``/``from_pipeline`` -> assert the reload retains
        one main node, one aux node, and the wire on the same port.
        """

        # 1. Start from an empty canvas (no nodes, no aux, no breadcrumbs).
        state = _empty_state()
        assert state["root"]["nodes"] == []
        assert state["root"]["aux_nodes"] == []

        # 2. User adds FilamentousFungiDetector to the main ribbon.
        state = _dispatch_state_update(
            state, "add_node", {"class_name": "FilamentousFungiDetector"}
        )
        assert len(state["root"]["nodes"]) == 1
        consumer_id = state["root"]["nodes"][0]["node_id"]
        assert state["root"]["nodes"][0]["class_name"] == "FilamentousFungiDetector"

        # 3. User adds an OtsuDetector aux node (palette button).
        state = _dispatch_state_update(
            state, "aux_add", {"class_name": "OtsuDetector"}
        )
        assert len(state["root"]["aux_nodes"]) == 1
        aux_id = state["root"]["aux_nodes"][0]["node_id"]

        # 4. User wires the aux into the consumer's ``inoculum_detector`` port.
        state = _dispatch_state_update(
            state,
            "wire_create",
            {
                "source_aux_id": aux_id,
                "target_node_id": consumer_id,
                "param": "inoculum_detector",
                "slot": 0,
            },
        )
        consumer = state["root"]["nodes"][0]
        assert consumer["aux_ports"]["inoculum_detector"] == [aux_id]

        # 5. Save -> ImagePipeline (mirrors the save handler's path).
        builder_state = state_from_json(state)
        pipeline = to_pipeline(builder_state.root)

        # The runtime pipeline should carry the FilamentousFungiDetector
        # with an OtsuDetector folded into its inoculum_detector slot.
        ops = pipeline.get_ops()
        assert "FilamentousFungiDetector" in ops
        consumer_op = ops["FilamentousFungiDetector"]
        inner_op = consumer_op.inoculum_detector
        assert type(inner_op).__name__ == "OtsuDetector"

        # 6. Reload from the same pipeline (mirrors a fresh load).
        rebuilt_scope = from_pipeline(pipeline)

        # 7. Confirm the reloaded state matches the saved structure.
        assert len(rebuilt_scope.nodes) == 1
        assert len(rebuilt_scope.aux_nodes) == 1
        rebuilt_consumer = rebuilt_scope.nodes[0]
        rebuilt_aux = rebuilt_scope.aux_nodes[0]
        assert rebuilt_consumer.class_name == "FilamentousFungiDetector"
        assert rebuilt_aux.class_name == "OtsuDetector"
        # The port wire is preserved (with a fresh aux node id minted
        # on reload — the wire still points at the right aux).
        slots = rebuilt_consumer.aux_ports["inoculum_detector"]
        assert slots == [rebuilt_aux.node_id]
        # And the consumer's params no longer carries the inoculum_detector
        # marker inline — the aux port is now the source of truth.
        assert "inoculum_detector" not in rebuilt_consumer.params

    def test_user_session_composite_with_two_aux_then_remove_one(
        self, app_ctx: Any
    ) -> None:
        """Build CompositeDetector with two aux, then disconnect the second slot.

        Exercises ``port_slot_add`` growing the list-typed port, two
        consecutive ``wire_create`` calls into different slots, and
        ``port_slot_remove`` shrinking the list — verifying the wire
        for slot 0 is preserved while slot 1 is collapsed.
        """

        state = _empty_state()
        state = _dispatch_state_update(
            state, "add_node", {"class_name": "CompositeDetector"}
        )
        consumer_id = state["root"]["nodes"][0]["node_id"]

        # Add two aux nodes in sequence.
        state = _dispatch_state_update(
            state, "aux_add", {"class_name": "OtsuDetector"}
        )
        otsu_id = state["root"]["aux_nodes"][0]["node_id"]
        state = _dispatch_state_update(
            state, "aux_add", {"class_name": "RoundPeaksDetector"}
        )
        peaks_id = state["root"]["aux_nodes"][1]["node_id"]

        # Grow the list-typed port to two slots and wire each.
        state = _dispatch_state_update(
            state,
            "port_slot_add",
            {"node_id": consumer_id, "param": "detectors"},
        )
        state = _dispatch_state_update(
            state,
            "port_slot_add",
            {"node_id": consumer_id, "param": "detectors"},
        )
        state = _dispatch_state_update(
            state,
            "wire_create",
            {
                "source_aux_id": otsu_id,
                "target_node_id": consumer_id,
                "param": "detectors",
                "slot": 0,
            },
        )
        state = _dispatch_state_update(
            state,
            "wire_create",
            {
                "source_aux_id": peaks_id,
                "target_node_id": consumer_id,
                "param": "detectors",
                "slot": 1,
            },
        )

        consumer = state["root"]["nodes"][0]
        assert consumer["aux_ports"]["detectors"] == [otsu_id, peaks_id]

        # User decides slot 1 was a mistake and removes it via the ``×`` UI.
        state = _dispatch_state_update(
            state,
            "port_slot_remove",
            {"node_id": consumer_id, "param": "detectors", "slot": 1},
        )
        consumer = state["root"]["nodes"][0]
        # Only the OtsuDetector wire remains.
        assert consumer["aux_ports"]["detectors"] == [otsu_id]

        # Save -> reload, and verify the runtime pipeline carries
        # exactly one detector (OtsuDetector).
        builder_state = state_from_json(state)
        pipeline = to_pipeline(builder_state.root)
        runtime_consumer = pipeline.get_ops()["CompositeDetector"]
        runtime_detectors = list(runtime_consumer.detectors)
        assert len(runtime_detectors) == 1
        assert type(runtime_detectors[0]).__name__ == "OtsuDetector"

        # The orphan RoundPeaksDetector aux remains in the JSON state
        # (it's only dropped at to_pipeline time as part of the fold).
        # Confirm the reload from the runtime pipeline does NOT
        # resurrect it — orphans are dropped on save.
        rebuilt_scope = from_pipeline(pipeline)
        # One main node + one wired aux (RoundPeaks dropped).
        assert len(rebuilt_scope.nodes) == 1
        assert len(rebuilt_scope.aux_nodes) == 1
        assert rebuilt_scope.aux_nodes[0].class_name == "OtsuDetector"
