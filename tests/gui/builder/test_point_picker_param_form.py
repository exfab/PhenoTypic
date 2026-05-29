"""Verify the param form swaps in the picker widget for pickable ops' centres."""

from __future__ import annotations

from phenotypic.gui._operation_registry import OperationRegistry, ParamInfo
from phenotypic.gui.builder._param_form import param_form, parse_widget_value


def _walk(node):
    yield node
    children = getattr(node, "children", None)
    if children is None:
        return
    if isinstance(children, list):
        for c in children:
            if c is not None:
                yield from _walk(c)
    else:
        yield from _walk(children)


def _id_types_in_form(form):
    """Return the set of `type` strings used by every dict-id in the form."""
    types = set()
    for node in _walk(form):
        nid = getattr(node, "id", None)
        if isinstance(nid, dict):
            t = nid.get("type")
            if isinstance(t, str):
                types.add(t)
    return types


def _ids_in_form(form):
    ids = []
    for node in _walk(form):
        nid = getattr(node, "id", None)
        if isinstance(nid, dict):
            ids.append(nid)
    return ids


def test_picker_widget_replaces_input_for_manual_point_detector():
    reg = OperationRegistry()
    reg.discover()

    info = reg.get("ManualPointDetector")
    assert info is not None and info.is_point_pickable
    form = param_form(info, current_values={}, form_id_prefix="mpd")

    types = _id_types_in_form(form)
    # Picker components present...
    assert "param-point-picker-store" in types
    assert "param-point-picker-btn" in types
    assert "param-point-picker-count" in types
    # ...and the default list/tuple/str input for ``centers`` is NOT.
    # (Other params like ``shape`` and ``width`` still emit their own inputs;
    # we just check that no list/tuple input keys ``centers``.)
    centers_inputs = [
        getattr(node, "id", None)
        for node in _walk(form)
        if isinstance(getattr(node, "id", None), dict)
           and getattr(node, "id", {}).get("name") == "centers"
           and getattr(node, "id", {}).get("type") in ("param-list", "param-tuple",
                                                       "param-str")
    ]
    assert centers_inputs == []


def test_non_pickable_op_has_no_picker_widget():
    reg = OperationRegistry()
    reg.discover()

    info = reg.get("OtsuDetector")
    assert info is not None and not info.is_point_pickable
    form = param_form(info, current_values={}, form_id_prefix="otsu")

    types = _id_types_in_form(form)
    assert "param-point-picker-store" not in types
    assert "param-point-picker-btn" not in types


def test_picker_store_carries_initial_centers():
    reg = OperationRegistry()
    reg.discover()
    info = reg.get("ManualPointDetector")

    form = param_form(
            info,
            current_values={"centers": [[10, 20], [30, 40]]},
            form_id_prefix="seeded",
    )

    for node in _walk(form):
        nid = getattr(node, "id", None)
        if isinstance(nid, dict) and nid.get("type") == "param-point-picker-store":
            assert node.data == [[10, 20], [30, 40]] or node.data == [[10.0, 20.0],
                                                                      [30.0, 40.0]]
            return
    raise AssertionError("picker store not found in form")


def test_picker_store_handles_numpy_initial():
    """np.ndarray current_values flow through cleanly (PointPickerMixin coerces on __setattr__)."""
    import numpy as np

    reg = OperationRegistry()
    reg.discover()
    info = reg.get("ManualRefine")

    form = param_form(
            info,
            current_values={"centers": np.array([[1.5, 2.5]])},
            form_id_prefix="np",
    )
    for node in _walk(form):
        nid = getattr(node, "id", None)
        if isinstance(nid, dict) and nid.get("type") == "param-point-picker-store":
            # Either list-of-lists with floats or list-of-tuples; allow either.
            assert len(node.data) == 1
            assert list(node.data[0]) == [1.5, 2.5]
            return
    raise AssertionError("picker store not found in form")


def test_pep604_optional_int_param_uses_numeric_widget_and_parser():
    """``int | None`` operation params must not fall back to text handling."""

    reg = OperationRegistry()
    reg.discover()
    info = reg.get("ImageCropper")

    assert info is not None
    left = info.parameters["left"]
    assert left.is_optional is True

    form = param_form(info, current_values={}, form_id_prefix="crop")
    left_ids = [
        nid
        for nid in _ids_in_form(form)
        if nid.get("prefix") == "crop" and nid.get("name") == "left"
    ]

    assert {"type": "param-num", "prefix": "crop", "name": "left"} in left_ids
    assert {"type": "param-str", "prefix": "crop", "name": "left"} not in left_ids
    assert parse_widget_value("12", left) == 12


def test_parse_widget_value_coerces_pep604_optional_float():
    """PEP 604 optional floats share the same coercion path as Optional[float]."""

    param = ParamInfo(
            name="sigma",
            type_hint=float | None,
            default=None,
            has_default=True,
            is_operation=False,
            is_pipeline=False,
            is_optional=True,
    )

    assert parse_widget_value("2.5", param) == 2.5
