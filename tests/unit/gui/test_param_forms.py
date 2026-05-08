"""Unit tests for the shared param-form module ``gui/_param_forms.py``.

Covers:
- Type-classification helpers (multi-union detection in particular).
- Each widget kind builds the expected component shape.
- ``parse_widget_value`` round-trips for every supported type.
- The new multi-union widget + tagged coercion path.
- Builder back-compat shim re-exports the same symbols.
"""

from __future__ import annotations

import pytest

from phenotypic.gui._param_forms import (
    _is_multi_union,
    _multi_union_branches,
    _unwrap_optional,
    _widget_for_param,
    param_form,
    parse_widget_value,
)


class _StubParamInfo:
    """Minimal ParamInfo stand-in for unit tests."""

    def __init__(
        self,
        name: str,
        type_hint,
        default=None,
        description: str | None = None,
        is_optional: bool = False,
        has_default: bool = True,
        is_operation: bool = False,
        is_pipeline: bool = False,
        column_ref=None,
    ):
        self.name = name
        self.type_hint = type_hint
        self.default = default
        self.description = description
        self.is_optional = is_optional
        self.has_default = has_default
        self.is_operation = is_operation
        self.is_pipeline = is_pipeline
        self.column_ref = column_ref


class TestUnwrapOptional:
    def test_unwraps_t_or_none_pep604(self):
        assert _unwrap_optional(int | None) is int

    def test_passes_through_plain_type(self):
        assert _unwrap_optional(int) is int

    def test_passes_through_multi_union(self):
        # Multi-type unions don't reduce to a single T.
        hint = bool | float | int | str | None
        assert _unwrap_optional(hint) is hint


class TestMultiUnion:
    def test_detects_multi_union(self):
        assert _is_multi_union(bool | float | int | str | None)
        assert _is_multi_union(int | str)

    def test_rejects_single_optional(self):
        assert not _is_multi_union(int | None)
        assert not _is_multi_union(int)

    def test_branches_strip_none(self):
        branches = _multi_union_branches(bool | float | int | str | None)
        assert set(branches) == {bool, float, int, str}


class TestParseWidgetValue:
    def test_bool_passthrough(self):
        p = _StubParamInfo("flag", bool, default=False)
        assert parse_widget_value(True, p) is True
        assert parse_widget_value("true", p) is True
        assert parse_widget_value("no", p) is False

    def test_int_coerce(self):
        p = _StubParamInfo("n", int, default=0)
        assert parse_widget_value("42", p) == 42

    def test_float_coerce(self):
        p = _StubParamInfo("x", float, default=0.0)
        assert parse_widget_value("1.5", p) == 1.5

    def test_str_passthrough(self):
        p = _StubParamInfo("s", str, default="")
        assert parse_widget_value("hello", p) == "hello"

    def test_optional_int(self):
        p = _StubParamInfo("n", int | None, default=None)
        assert parse_widget_value(None, p) is None
        assert parse_widget_value("3", p) == 3

    def test_list_str_coerce(self):
        p = _StubParamInfo("xs", list[str], default=None)
        assert parse_widget_value("a, b ,c", p) == ["a", "b", "c"]

    def test_tuple_int_coerce(self):
        p = _StubParamInfo("pair", tuple[int, ...], default=None)
        assert parse_widget_value("1, 2, 3", p) == (1, 2, 3)

    def test_literal_match(self):
        from typing import Literal

        p = _StubParamInfo("loss", Literal["linear", "huber"], default="huber")
        assert parse_widget_value("huber", p) == "huber"

    def test_multi_union_tag_none(self):
        p = _StubParamInfo("u", bool | float | int | str | None, default=None)
        assert parse_widget_value(("none", ""), p) is None

    def test_multi_union_tag_true(self):
        p = _StubParamInfo("u", bool | float | int | str | None, default=None)
        assert parse_widget_value(("true", ""), p) is True

    def test_multi_union_tag_false(self):
        p = _StubParamInfo("u", bool | float | int | str | None, default=None)
        assert parse_widget_value(("false", ""), p) is False

    def test_multi_union_tag_number(self):
        p = _StubParamInfo("u", bool | float | int | str | None, default=None)
        assert parse_widget_value(("number", "0.5"), p) == 0.5

    def test_multi_union_tag_string(self):
        p = _StubParamInfo("u", bool | float | int | str | None, default=None)
        assert parse_widget_value(("string", "Metadata_Strain"), p) == "Metadata_Strain"


class TestWidgetForParam:
    def test_bool_renders_switch(self):
        p = _StubParamInfo("flag", bool, default=False)
        widget = _widget_for_param(p, current_value=False, form_id_prefix="t")
        assert widget.id["type"] == "param-bool"

    def test_int_renders_num_input(self):
        p = _StubParamInfo("n", int, default=0)
        widget = _widget_for_param(p, current_value=3, form_id_prefix="t")
        assert widget.id["type"] == "param-num"

    def test_str_renders_text_input(self):
        p = _StubParamInfo("s", str, default="")
        widget = _widget_for_param(p, current_value="x", form_id_prefix="t")
        assert widget.id["type"] == "param-str"

    def test_literal_renders_select(self):
        from typing import Literal

        p = _StubParamInfo("loss", Literal["linear", "huber"], default="huber")
        widget = _widget_for_param(p, current_value="huber", form_id_prefix="t")
        assert widget.id["type"] == "param-enum"

    def test_multi_union_renders_tagged_widget(self):
        p = _StubParamInfo("u", bool | float | int | str | None, default=None)
        widget = _widget_for_param(p, current_value=None, form_id_prefix="t")
        # _multi_union_widget returns an html.Div with two children.
        assert hasattr(widget, "children")
        # Tag selector + value input.
        types = {c.id.get("type") for c in widget.children if hasattr(c, "id")}
        assert "param-multi-tag" in types
        assert "param-multi-value" in types

    def test_picker_factory_called_when_param_matches(self):
        called = {}

        def fake_factory(*, form_id_prefix, name, current_value):
            called["args"] = (form_id_prefix, name, current_value)
            return "PICKER_PLACEHOLDER"

        p = _StubParamInfo("centers", list[tuple[float, float]], default=None)
        result = _widget_for_param(
            p,
            current_value=None,
            form_id_prefix="t",
            point_picker_param="centers",
            picker_factory=fake_factory,
        )
        assert result == "PICKER_PLACEHOLDER"
        assert called["args"] == ("t", "centers", None)

    def test_picker_factory_not_called_for_non_picker_param(self):
        def boom(**_):
            raise AssertionError("picker_factory should not run for non-picker params")

        p = _StubParamInfo("sigma", float, default=1.0)
        widget = _widget_for_param(
            p,
            current_value=1.0,
            form_id_prefix="t",
            point_picker_param="centers",  # different name
            picker_factory=boom,
        )
        assert widget.id["type"] == "param-num"


class TestParamFormViaRegistry:
    """End-to-end: build a form from a real registered analyzer."""

    @pytest.fixture(scope="class")
    def registry(self):
        from phenotypic.gui._operation_registry import OperationRegistry

        reg = OperationRegistry()
        reg.discover()
        return reg

    def test_edge_corrector_registered(self, registry):
        info = registry.get("EdgeCorrector")
        assert info is not None
        assert info.category == "Filter"
        assert "on" in info.parameters
        assert "groupby" in info.parameters

    def test_log_growth_model_registered(self, registry):
        info = registry.get("LogGrowthModel")
        assert info is not None
        assert info.category == "Model"
        assert "loss" in info.parameters

    def test_linear_softplus_has_multi_union(self, registry):
        info = registry.get("LinearSoftplusModel")
        assert info is not None
        sp = info.parameters.get("s0_prior")
        assert sp is not None
        assert _is_multi_union(sp.type_hint)

    def test_param_form_renders_for_filter(self, registry):
        info = registry.get("EdgeCorrector")
        form = param_form(info, current_values={}, form_id_prefix="ec")
        # ``dbc.Form`` is a list-like container; one row per parameter.
        assert len(form.children) == len(info.parameters)


class TestColumnWidgets:
    """Coverage for the column-aware dropdowns + two-button mode toggle."""

    def _columns(self, _src):
        return ["Metadata_Strain", "Metadata_Time", "Shape_Area"]

    def _scalar_spec(self, with_alt=False):
        from phenotypic.gui._operation_registry import ColumnRefSpec

        return ColumnRefSpec(source="measurements", multi=False, with_alt=with_alt)

    def _multi_spec(self):
        from phenotypic.gui._operation_registry import ColumnRefSpec

        return ColumnRefSpec(source="measurements", multi=True, with_alt=False)

    def test_scalar_dropdown_renders_for_columnref(self):
        from phenotypic.tools_ import ColumnRef

        p = _StubParamInfo("on", ColumnRef, column_ref=self._scalar_spec())
        w = _widget_for_param(
            p,
            current_value="Shape_Area",
            form_id_prefix="t",
            columns_provider=self._columns,
        )
        # dbc.Select sets `value=`; check via vars().
        assert w.value == "Shape_Area"
        assert {o["value"] for o in w.options} == {
            "Metadata_Strain",
            "Metadata_Time",
            "Shape_Area",
        }

    def test_multi_dropdown_renders_for_columnreflist(self):
        from phenotypic.tools_ import ColumnRefList

        p = _StubParamInfo(
            "groupby", ColumnRefList, column_ref=self._multi_spec()
        )
        w = _widget_for_param(
            p,
            current_value=["Metadata_Strain"],
            form_id_prefix="t",
            columns_provider=self._columns,
        )
        # dcc.Dropdown(multi=True)
        assert w.multi is True
        assert w.value == ["Metadata_Strain"]

    def test_stale_value_renders_with_tooltip(self):
        from phenotypic.tools_ import ColumnRef

        p = _StubParamInfo("on", ColumnRef, column_ref=self._scalar_spec())
        w = _widget_for_param(
            p,
            current_value="MissingCol",
            form_id_prefix="t",
            columns_provider=self._columns,
        )
        # The stale wrapper Div carries a `title` for the tooltip.
        assert getattr(w, "title", None)
        assert "MissingCol" in w.title

    def test_columnref_or_none_renders_two_button_toggle(self):
        from phenotypic.tools_ import ColumnRef

        p = _StubParamInfo(
            "Kmax_label",
            ColumnRef | None,
            column_ref=self._scalar_spec(with_alt=True),
        )
        w = _widget_for_param(
            p,
            current_value="Shape_Area",
            form_id_prefix="t",
            columns_provider=self._columns,
        )
        # Wrapper Div with [RadioItems, Div(dropdown)] children.
        kids = w.children
        assert len(kids) == 2
        radio = kids[0]
        # RadioItems options include a Column branch + a None branch.
        values = {o["value"] for o in radio.options}
        assert {"column", "none"}.issubset(values)
        assert radio.value == "column"  # current is a string -> column mode

    def test_columnref_or_none_default_mode_is_none_when_value_is_none(self):
        from phenotypic.tools_ import ColumnRef

        p = _StubParamInfo(
            "Kmax_label",
            ColumnRef | None,
            column_ref=self._scalar_spec(with_alt=True),
        )
        w = _widget_for_param(
            p,
            current_value=None,
            form_id_prefix="t",
            columns_provider=self._columns,
        )
        radio = w.children[0]
        assert radio.value == "none"
        # The dropdown is disabled when mode is "none".
        dropdown_wrapper = w.children[1]
        dropdown = dropdown_wrapper.children
        assert dropdown.disabled is True

    def test_columnref_list_with_alt_raises(self):
        """``ColumnRefList | None`` is not yet wired end-to-end.

        Guard exists so adding the first multi+alt param surfaces the
        gap immediately rather than silently rendering a scalar widget
        on a list value.
        """
        from phenotypic.gui._operation_registry import ColumnRefSpec
        from phenotypic.tools_ import ColumnRefList

        p = _StubParamInfo(
            "future_param",
            ColumnRefList | None,
            column_ref=ColumnRefSpec(
                source="measurements", multi=True, with_alt=True
            ),
        )
        with pytest.raises(NotImplementedError, match="ColumnRefList"):
            _widget_for_param(
                p,
                current_value=None,
                form_id_prefix="t",
                columns_provider=self._columns,
            )


class TestParseColumnValue:
    def test_scalar_passthrough(self):
        from phenotypic.gui._operation_registry import ColumnRefSpec

        p = _StubParamInfo(
            "on",
            str,
            column_ref=ColumnRefSpec("measurements", False, False),
        )
        assert parse_widget_value("Shape_Area", p) == "Shape_Area"
        assert parse_widget_value("", p) is None
        assert parse_widget_value(None, p) is None

    def test_multi_passthrough(self):
        from phenotypic.gui._operation_registry import ColumnRefSpec

        p = _StubParamInfo(
            "groupby",
            list,
            column_ref=ColumnRefSpec("measurements", True, False),
        )
        assert parse_widget_value(["a", "b"], p) == ["a", "b"]
        assert parse_widget_value(None, p) == []

    def test_mode_column_returns_scalar(self):
        from phenotypic.gui._operation_registry import ColumnRefSpec

        p = _StubParamInfo(
            "Kmax_label",
            str,
            column_ref=ColumnRefSpec("measurements", False, True),
        )
        assert parse_widget_value(("column", "Shape_Area"), p) == "Shape_Area"

    def test_mode_none_returns_none(self):
        from phenotypic.gui._operation_registry import ColumnRefSpec

        p = _StubParamInfo(
            "Kmax_label",
            str,
            column_ref=ColumnRefSpec("measurements", False, True),
        )
        assert parse_widget_value(("none", "Shape_Area"), p) is None


class TestColumnsProviderPlumbing:
    def test_param_form_passes_provider_to_columnref_params(self):
        from phenotypic.gui._operation_registry import get_registry

        reg = get_registry()
        info = reg.get("EdgeCorrector")
        captured: list[str] = []

        def provider(source: str) -> list[str]:
            captured.append(source)
            return ["Shape_Area", "Metadata_Strain", "Metadata_Time"]

        param_form(
            info,
            current_values={
                "on": "Shape_Area",
                "groupby": ["Metadata_Strain"],
                "time_label": "Metadata_Time",
            },
            form_id_prefix="ec",
            columns_provider=provider,
        )
        # on, groupby, time_label all carry column_ref → 3 calls minimum.
        assert captured.count("measurements") >= 3


class TestBuilderShimReExports:
    def test_param_form_importable_from_builder(self):
        from phenotypic.gui.builder._param_form import (  # noqa: F401
            param_form,
            parse_widget_value,
            serialize_param_for_widget,
            parse_list_value,
        )

    def test_builder_param_form_injects_picker(self):
        from phenotypic.gui.builder._param_form import param_form as builder_form
        from phenotypic.gui._operation_registry import OperationRegistry

        reg = OperationRegistry()
        reg.discover()
        # Find a point-pickable op to verify the picker lights up.
        pickable = [
            info for info in reg.get_all().values() if info.is_point_pickable
        ]
        if not pickable:
            pytest.skip("No point-pickable operations registered")
        info = pickable[0]
        form = builder_form(info, current_values={}, form_id_prefix="t")
        assert len(form.children) >= 1
