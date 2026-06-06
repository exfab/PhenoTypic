"""Unit guards for fixed linear builder callback routing helpers."""

from __future__ import annotations

import inspect

from phenotypic.gui.builder import _callbacks
from phenotypic.gui.builder import _ids as ids


def test_linear_pattern_id_decoders_round_trip_dash_safe_fields():
    triggered = ids.linear_port_id(
        kind="parameter_slot",
        scope_path=["pipe-a", "pipe-b"],
        block_id="consumer",
        param="detectors",
        slot=2,
        surface="side",
    )

    target = _callbacks._linear_target_payload_from_id(triggered)

    assert target == {
        "kind": "parameter_slot",
        "scope_path": ["pipe-a", "pipe-b"],
        "block_id": "consumer",
        "param": "detectors",
        "slot": 2,
    }


def test_linear_param_action_decoder_classifies_scalar_target():
    triggered = ids.linear_param_action_id(
        action="replace",
        block_id="consumer",
        param="detector",
        surface="side",
    )

    target = _callbacks._linear_param_target_payload_from_id(triggered)

    assert target["kind"] == "parameter"
    assert target["block_id"] == "consumer"
    assert target["param"] == "detector"
    assert target["slot"] is None


def test_register_callbacks_routes_palette_clicks_and_retires_drag_stores():
    source = inspect.getsource(_callbacks.register_callbacks)

    assert '"linear_palette_add"' in source
    assert "elif triggered == ids.STORE_PALETTE_DROP:" in source
    assert "elif triggered == ids.STORE_EDGE_EVENT:" in source
    assert source.count("return _NOOP_FAN_IN") >= 2


def test_asset_status_callback_keeps_retired_js_assets_inert():
    source = inspect.getsource(_callbacks.register_callbacks)

    assert "def asset_status_disables(" in source
    assert 'return True, {}, [], {"display": "none"}' in source
    assert "Block creation offline" not in source
