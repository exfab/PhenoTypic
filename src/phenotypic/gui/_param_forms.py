"""Auto-generate Dash form widgets from ``ParamInfo`` metadata.

Pure-function tier shared by the builder and the analysis sub-app's
section forms. Extracted from ``gui/builder/_param_form.py`` so the
analysis sub-app can author the same kind of inline forms for
``SetAnalyzer`` / ``ModelFitter`` / ``PostMeasurement`` params without
duplicating the type-classification + coercion machinery.

Builder-specific widgets (the point picker) inject themselves through
the ``picker_factory`` parameter on :func:`_widget_for_param`,
:func:`_param_row`, and :func:`param_form`. Callers that don't need a
picker (the analysis sub-app) leave it as ``None`` and the picker
branch is dead code on their path.
"""

from __future__ import annotations

import collections.abc as abc
import enum
import inspect
import types
import typing
from typing import TYPE_CHECKING, Any, Callable, Optional, get_args, get_origin

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
import numpy as np
from dash import dcc, html

if TYPE_CHECKING:
    from phenotypic.gui._operation_registry import OperationInfo, ParamInfo


#: Tag used by the column / alt-dtype mode toggle.
COLUMN_MODE_TAG = "column"
NONE_MODE_TAG = "none"


# ---------------------------------------------------------------------------
# Type-classification helpers
# ---------------------------------------------------------------------------


def _unwrap_optional(hint: Any) -> Any:
    """Return ``T`` from ``T | None`` annotations; otherwise pass through.

    Recognises both ``typing.Union[T, None]`` and ``T | None`` (PEP 604).
    Multi-type unions (e.g. ``int | str | None``) pass through unchanged
    because there's no single ``T`` to extract.
    """
    origin = get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in get_args(hint) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return hint


def _literal_options(hint: Any) -> list[Any] | None:
    """Extract literal values from ``Literal[...]`` annotations."""
    inner = _unwrap_optional(hint)
    if get_origin(inner) is typing.Literal:
        return list(get_args(inner))
    return None


def _enum_options(hint: Any) -> list[Any] | None:
    """Extract member values from ``enum.Enum`` annotations."""
    inner = _unwrap_optional(hint)
    if inspect.isclass(inner) and issubclass(inner, enum.Enum):
        return [m.value for m in inner]
    return None


def _is_list_type(hint: Any) -> bool:
    """Return ``True`` if ``hint`` is ``list[T]`` / ``Iterable[T]``."""
    inner = _unwrap_optional(hint)
    origin = get_origin(inner)
    if origin is list:
        return True
    if origin in (
        abc.Iterable,
        abc.Sequence,
        abc.MutableSequence,
        typing.Iterable,
        typing.Sequence,
        typing.MutableSequence,
    ):
        return True
    return inner is list


def _is_tuple_type(hint: Any) -> bool:
    """Return ``True`` if ``hint`` is ``tuple[...]``."""
    inner = _unwrap_optional(hint)
    origin = get_origin(inner)
    if origin is tuple:
        return True
    return inner is tuple


def _list_item_type(hint: Any) -> Any:
    """Return the element type for ``list[T]`` / ``tuple[T, ...]``."""
    inner = _unwrap_optional(hint)
    args = get_args(inner)
    if not args:
        return str
    return args[0]


def _is_multi_union(hint: Any) -> bool:
    """Return ``True`` for unions of 2+ non-None types like ``bool | float | None``.

    Single-type optionals (``T | None``) report False — :func:`_unwrap_optional`
    handles those. Multi-type unions need the type-tag widget instead.
    """
    origin = get_origin(hint)
    if origin not in (typing.Union, types.UnionType):
        return False
    non_none = [a for a in get_args(hint) if a is not type(None)]
    return len(non_none) >= 2


def _multi_union_branches(hint: Any) -> list[Any]:
    """Return the non-``None`` member types of a multi-union annotation."""
    return [a for a in get_args(hint) if a is not type(None)]


# ---------------------------------------------------------------------------
# Value conversion helpers
# ---------------------------------------------------------------------------


def _coerce_scalar(text: Any, item_type: Any) -> Any:
    """Coerce a single token to a Python scalar using ``item_type``."""
    if text is None or text == "":
        return None
    if item_type is bool:
        if isinstance(text, bool):
            return text
        return str(text).strip().lower() in {"1", "true", "yes", "on"}
    if item_type is int:
        try:
            return int(float(text))
        except (TypeError, ValueError):
            return text
    if item_type is float:
        try:
            return float(text)
        except (TypeError, ValueError):
            return text
    return str(text)


def parse_list_value(text: str | None, item_type: Any = str) -> list[Any]:
    """Parse a comma-separated text value into a list of typed elements."""
    if text is None:
        return []
    cleaned = str(text).strip()
    if not cleaned:
        return []
    tokens = [t.strip() for t in cleaned.split(",")]
    return [_coerce_scalar(t, item_type) for t in tokens if t != ""]


#: Tag values used by the multi-union widget's selector dropdown. The tag
#: tells :func:`parse_widget_value` which branch of the union the value
#: payload should coerce to.
MULTI_UNION_TAGS = ("none", "true", "false", "number", "string")


def parse_widget_value(raw: Any, p: "ParamInfo") -> Any:
    """Convert a Dash widget's reported value back to its declared Python type.

    Handles tuple/list comma-split, int/float coercion, enum/literal
    pass-through, multi-union type-tag dispatch, column-ref pass-through
    (single + multi + mode-toggle), and bool fall-through.
    """
    inner = _unwrap_optional(p.type_hint)

    # Column-ref params: column-with-alt packs ``(mode, scalar)`` so the
    # active branch is decided at parse time without needing two Inputs.
    # Plain dropdowns return their natural type (str / list[str]).
    column_ref = getattr(p, "column_ref", None)
    if column_ref is not None:
        if isinstance(raw, tuple) and len(raw) == 2:
            mode, scalar = raw
            return None if mode == NONE_MODE_TAG else scalar
        if column_ref.multi:
            if isinstance(raw, list):
                return [str(v) for v in raw if v is not None]
            return []
        # ``dbc.Select`` reports ``""`` for "no selection" — fold to None.
        if raw == "" or raw is None:
            return None
        return str(raw)

    # Multi-type unions arrive as a tuple (tag, value) from the widget. Any
    # other shape (None, primitive) is treated as the natural-type case.
    if _is_multi_union(p.type_hint) and isinstance(raw, (list, tuple)) and len(raw) == 2:
        tag, value = raw
        if tag == "none":
            return None
        if tag == "true":
            return True
        if tag == "false":
            return False
        if tag == "number":
            try:
                return float(value)
            except (TypeError, ValueError):
                return value
        if tag == "string":
            return "" if value is None else str(value)
        return value

    if raw is None:
        return None

    literals = _literal_options(p.type_hint)
    if literals is not None:
        for lit in literals:
            if str(lit) == str(raw):
                return lit
        return raw

    enum_vals = _enum_options(p.type_hint)
    if enum_vals is not None:
        for v in enum_vals:
            if str(v) == str(raw):
                return v
        return raw

    if inner is bool:
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    if inner is int:
        try:
            return int(raw)
        except (TypeError, ValueError):
            return raw
    if inner is float:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return raw

    if _is_list_type(p.type_hint):
        return parse_list_value(raw, _list_item_type(p.type_hint))
    if _is_tuple_type(p.type_hint):
        return tuple(parse_list_value(raw, _list_item_type(p.type_hint)))

    return raw


def serialize_param_for_widget(value: Any, p: "ParamInfo") -> Any:
    """Format a stored value for display in a Dash widget."""
    if value is None:
        return None

    if isinstance(value, enum.Enum):
        return value.value

    if isinstance(value, np.ndarray):
        # Picker-style ndarrays (N x 2) round-trip via ``.tolist()``.
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        return ", ".join(_format_token(v) for v in value)

    if isinstance(value, bool):
        return value

    return value


def _multi_union_tag_for(value: Any) -> str:
    """Pick the matching tag for a multi-union value's existing storage."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return "number"
    return "string"


def _format_token(v: Any) -> str:
    """Render a sequence element as a stable string for the widget."""
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return repr(v)
    return str(v)


# ---------------------------------------------------------------------------
# Widget builders
# ---------------------------------------------------------------------------


def _column_options(columns: list[str]) -> list[dict[str, str]]:
    """Build a Dash options list from a column-name list."""
    return [{"label": c, "value": c} for c in columns]


def _column_widget(
    *,
    p: "ParamInfo",
    current_value: Any,
    form_id_prefix: str,
    columns: list[str],
) -> Any:
    """Render a column-aware dropdown for plain ``ColumnRef`` / ``ColumnRefList``.

    Stale values (selected but absent from ``columns``) are surfaced via a
    wrapper-level tooltip — multi widgets keep them selectable as
    ``(missing) <name>`` entries; scalars clear the value (Dash's
    ``dbc.Select`` does not render an option that isn't in ``options``).
    """
    spec = p.column_ref
    assert spec is not None  # caller guarantees this
    component_id = {
        "prefix": form_id_prefix,
        "name": p.name,
    }

    if spec.multi:
        value_list = list(current_value) if isinstance(current_value, list) else []
        stale = [
            v for v in value_list
            if isinstance(v, str) and v and v not in columns
        ]
        options = _column_options(columns) + [
            {"label": f"{v} (missing)", "value": v} for v in stale
        ]
        widget = dcc.Dropdown(
            id={"type": "param-column-multi", **component_id},
            options=options,
            value=value_list,
            multi=True,
            placeholder="Pick one or more columns…",
        )
        if stale:
            return html.Div(
                widget,
                title=(
                    f"missing column(s): {', '.join(stale)} "
                    f"(not in {spec.source} file)"
                ),
                className="param-column-stale",
            )
        return widget

    scalar = current_value if isinstance(current_value, str) else None
    is_stale = bool(scalar) and scalar not in columns
    widget = dbc.Select(
        id={"type": "param-column-scalar", **component_id},
        options=_column_options(columns),
        value=None if is_stale else scalar,
        placeholder="Pick a column…",
    )
    if is_stale:
        return html.Div(
            widget,
            title=f'previously: "{scalar}" (not in {spec.source} file)',
            className="param-column-stale",
        )
    return widget


def _column_or_alt_widget(
    *,
    p: "ParamInfo",
    current_value: Any,
    form_id_prefix: str,
    columns: list[str],
) -> Any:
    """Render a two-button mode toggle for ``ColumnRef | <alt>`` unions.

    The toggle is a ``dbc.RadioItems`` styled as a Bootstrap button group.
    "Column" enables the dropdown; the alt branch ("None" for
    ``ColumnRef | None``) disables it and the param saves as the alt's
    natural value. Mode is derived from ``current_value``: a string means
    column mode, otherwise the alt branch is active.

    Raises:
        NotImplementedError: When ``spec.multi`` is True. v1 only renders
            the scalar dropdown inside the toggle; ``ColumnRefList | None``
            needs both a multi dropdown branch in this widget AND a
            matching ``param-column-multi`` arm in
            ``_apply_param_edit``'s with-alt dispatch — neither is wired
            up. Any future use must update both sites in lockstep.
    """
    spec = p.column_ref
    assert spec is not None
    if spec.multi:
        raise NotImplementedError(
            f"{p.name}: ColumnRefList with an alternate branch is not "
            "supported yet. Add both a multi-select dropdown to "
            "_column_or_alt_widget and a `param-column-multi` arm to "
            "_apply_param_edit's with-alt dispatch."
        )
    component_id = {
        "prefix": form_id_prefix,
        "name": p.name,
    }

    has_none = any(a is type(None) for a in get_args(p.type_hint))
    mode = COLUMN_MODE_TAG if isinstance(current_value, str) else NONE_MODE_TAG
    mode_options: list[dict[str, Any]] = [
        {"label": "Column", "value": COLUMN_MODE_TAG},
    ]
    if has_none:
        mode_options.append({"label": "None", "value": NONE_MODE_TAG})

    scalar_value = current_value if isinstance(current_value, str) else None
    is_stale = bool(scalar_value) and scalar_value not in columns
    # Stale values stay selectable here (unlike _column_widget) so the user
    # can see what was there; the wrapper tooltip surfaces the staleness.
    column_options = _column_options(columns)
    if is_stale:
        column_options.append(
            {"label": f"{scalar_value} (missing)", "value": scalar_value}
        )

    dropdown = dbc.Select(
        id={"type": "param-column-scalar", **component_id},
        options=column_options,
        value=scalar_value,
        placeholder="Pick a column…",
        disabled=(mode == NONE_MODE_TAG),
    )
    radio = dbc.RadioItems(
        id={"type": "param-column-mode", **component_id},
        options=mode_options,
        value=mode,
        inline=True,
        class_name="btn-group param-column-mode",
        input_class_name="btn-check",
        label_class_name="btn btn-outline-secondary btn-sm",
    )

    wrapper_kwargs: dict[str, Any] = {"className": "param-column-toggle"}
    if is_stale:
        wrapper_kwargs["title"] = (
            f'previously: "{scalar_value}" (not in {spec.source} file)'
        )
    return html.Div(
        [radio, html.Div(dropdown, style={"marginTop": "0.4rem"})],
        **wrapper_kwargs,
    )


def _multi_union_widget(
    *,
    p: "ParamInfo",
    current_value: Any,
    form_id_prefix: str,
) -> Any:
    """Render a type-tag dropdown plus an adaptive value input for multi-unions.

    Used for parameters whose annotation is a union of 2+ non-None types
    (e.g. ``LinearSoftplusModel.s0_prior: bool | float | int | str | None``).
    The dropdown picks the active branch; the input adapts type accordingly.
    A pattern-matching callback fans the (tag, value) pair back into
    :func:`parse_widget_value` which dispatches on the tag.
    """
    branches = _multi_union_branches(p.type_hint)
    options = [{"label": "None", "value": "none"}]
    if bool in branches:
        options.extend([
            {"label": "True", "value": "true"},
            {"label": "False", "value": "false"},
        ])
    if any(t in branches for t in (int, float)):
        options.append({"label": "number", "value": "number"})
    if str in branches:
        options.append({"label": "string", "value": "string"})

    tag = _multi_union_tag_for(current_value)
    value_text = (
        ""
        if current_value is None or isinstance(current_value, bool)
        else str(current_value)
    )

    return html.Div(
        [
            dbc.Select(
                id={"type": "param-multi-tag", "prefix": form_id_prefix, "name": p.name},
                options=options,
                value=tag,
                style={"width": "8rem", "display": "inline-block"},
            ),
            dbc.Input(
                id={"type": "param-multi-value", "prefix": form_id_prefix, "name": p.name},
                type="text",
                value=value_text,
                placeholder="value",
                debounce=True,
                style={"display": "inline-block", "marginLeft": "0.5rem"},
            ),
        ],
        className="d-flex align-items-center",
    )


def _widget_for_param(
    p: "ParamInfo",
    *,
    current_value: Any,
    form_id_prefix: str,
    point_picker_param: Optional[str] = None,
    picker_factory: Optional[Callable[..., Any]] = None,
    columns_provider: Optional[Callable[[str], list[str]]] = None,
) -> Any:
    """Build the primary input widget for a single parameter.

    Args:
        p: :class:`ParamInfo` from the operation registry.
        current_value: Existing value to populate the widget with.
        form_id_prefix: Prefix added to every generated component id.
        point_picker_param: When set AND ``picker_factory`` is provided,
            the named parameter renders as the picker widget instead of a
            text input. Builder injects this; the analysis sub-app leaves
            both ``None`` so the picker branch is dead code on its path.
        picker_factory: Callable that builds the picker widget when
            ``point_picker_param`` matches. Signature:
            ``factory(*, form_id_prefix, name, current_value) -> Component``.
        columns_provider: Callable returning the column-name list for a
            given source (``"measurements"`` / ``"master_measurements"``).
            Analysis sub-app passes ``MeasurementSchema.columns_for``;
            builder leaves it as ``None``. When provided AND the param
            carries a ``column_ref``, renders a column dropdown instead
            of a free-text input.
    """
    if (
        point_picker_param is not None
        and picker_factory is not None
        and p.name == point_picker_param
    ):
        return picker_factory(
            form_id_prefix=form_id_prefix,
            name=p.name,
            current_value=current_value,
        )

    # ``columns_provider`` is supplied by the analysis sub-app only —
    # builder ops carry no column-ref params, so this branch is dead
    # code on the builder path.
    column_ref = getattr(p, "column_ref", None)
    if column_ref is not None and columns_provider is not None:
        try:
            columns = columns_provider(column_ref.source)
        except Exception:  # noqa: BLE001
            columns = []
        builder = (
            _column_or_alt_widget if column_ref.with_alt else _column_widget
        )
        return builder(
            p=p,
            current_value=current_value,
            form_id_prefix=form_id_prefix,
            columns=columns,
        )

    # Multi-type unions (e.g. ``bool | float | int | str | None``) take
    # priority over the inner-type dispatch below since ``_unwrap_optional``
    # would not collapse them to a single type.
    if _is_multi_union(p.type_hint):
        return _multi_union_widget(
            p=p, current_value=current_value, form_id_prefix=form_id_prefix
        )

    inner = _unwrap_optional(p.type_hint)
    initial = (
        serialize_param_for_widget(current_value, p)
        if current_value is not None
        else serialize_param_for_widget(p.default, p)
    )

    if p.is_operation or p.is_pipeline:
        return dbc.Button(
            "Edit ▸",
            id={
                "type": "param-edit-nested",
                "prefix": form_id_prefix,
                "name": p.name,
            },
            color="secondary",
            size="sm",
            outline=True,
        )

    options = _literal_options(p.type_hint) or _enum_options(p.type_hint)
    if options is not None:
        return dbc.Select(
            id={"type": "param-enum", "prefix": form_id_prefix, "name": p.name},
            options=[{"label": str(v), "value": str(v)} for v in options],
            value=str(initial) if initial is not None else None,
        )

    if inner is bool:
        return dbc.Switch(
            id={"type": "param-bool", "prefix": form_id_prefix, "name": p.name},
            label="",
            value=bool(initial) if initial is not None else False,
        )

    if inner is int:
        return dbc.Input(
            type="number",
            id={"type": "param-num", "prefix": form_id_prefix, "name": p.name},
            value=initial,
            step=1,
            debounce=True,
        )
    if inner is float:
        return dbc.Input(
            type="number",
            id={"type": "param-num", "prefix": form_id_prefix, "name": p.name},
            value=initial,
            step=0.01,
            debounce=True,
        )

    if _is_list_type(p.type_hint):
        return dbc.Input(
            type="text",
            id={"type": "param-list", "prefix": form_id_prefix, "name": p.name},
            value=initial if isinstance(initial, str) else "",
            placeholder="comma-separated",
            debounce=True,
        )

    if _is_tuple_type(p.type_hint):
        return dbc.Input(
            type="text",
            id={"type": "param-tuple", "prefix": form_id_prefix, "name": p.name},
            value=initial if isinstance(initial, str) else "",
            placeholder="comma-separated",
            debounce=True,
        )

    return dbc.Input(
        type="text",
        id={"type": "param-str", "prefix": form_id_prefix, "name": p.name},
        value=initial if initial is not None else "",
        debounce=True,
    )


def _optional_toggle(p: "ParamInfo", *, form_id_prefix: str) -> Any:
    """Build the "Use default" toggle shown beside Optional widgets."""
    return dbc.Switch(
        id={
            "type": "param-optional-toggle",
            "prefix": form_id_prefix,
            "name": p.name,
        },
        label="Use default",
        value=p.default is None,
    )


def _error_slot(p: "ParamInfo", *, form_id_prefix: str) -> Any:
    """Reserve an empty validation-message div for callbacks to populate."""
    return html.Div(
        id={"type": "param-error", "prefix": form_id_prefix, "name": p.name},
        className="invalid-feedback d-block",
    )


def _param_row(
    p: "ParamInfo",
    *,
    current_values: dict[str, Any],
    form_id_prefix: str,
    point_picker_param: Optional[str] = None,
    picker_factory: Optional[Callable[..., Any]] = None,
    columns_provider: Optional[Callable[[str], list[str]]] = None,
) -> Any:
    """Render one parameter as a labelled ``dbc.Row``."""
    widget = _widget_for_param(
        p,
        current_value=current_values.get(p.name),
        form_id_prefix=form_id_prefix,
        point_picker_param=point_picker_param,
        picker_factory=picker_factory,
        columns_provider=columns_provider,
    )

    label_children: list[Any] = [p.name]
    label = dbc.Label(label_children, html_for=None, className="fw-semibold")

    helper: list[Any] = []
    if p.description:
        helper.append(dbc.FormText(p.description, color="secondary"))

    main_col_children: list[Any] = [
        widget,
        *helper,
        _error_slot(p, form_id_prefix=form_id_prefix),
    ]
    cols: list[Any] = [
        dbc.Col(label, width=4),
        dbc.Col(main_col_children, width=6),
    ]
    if p.is_optional and p.has_default:
        cols.append(
            dbc.Col(_optional_toggle(p, form_id_prefix=form_id_prefix), width=2)
        )
    else:
        cols.append(dbc.Col(width=2))

    return dbc.Row(cols, className="mb-3 align-items-center")


def param_form(
    op_info: "OperationInfo",
    current_values: dict[str, Any],
    *,
    form_id_prefix: str,
    picker_factory: Optional[Callable[..., Any]] = None,
    columns_provider: Optional[Callable[[str], list[str]]] = None,
) -> dbc.Form:
    """Generate a parameter form for an operation.

    Args:
        op_info: Registry metadata for the operation being edited.
        current_values: Mapping of parameter-name → current value used to
            seed each widget. Missing keys fall back to the parameter
            default.
        form_id_prefix: Prefix added to every emitted component id.
        picker_factory: Optional builder-side callable that renders the
            point picker for parameters whose owning op declares
            ``point_picker_param``. The analysis sub-app leaves this as
            ``None``.
        columns_provider: Optional analysis-side callable that resolves a
            ``ColumnSource`` to a list of column names (typically
            :meth:`MeasurementSchema.columns_for`). When set, params whose
            type carries a ``ColumnRef`` marker render as dropdowns
            populated from the live measurements schema instead of a
            free-text input.

    Returns:
        ``dbc.Form`` whose children are one ``dbc.Row`` per parameter.
    """
    point_picker_param = getattr(op_info, "point_picker_param", None)
    rows: list[Any] = []
    for p in op_info.parameters.values():
        rows.append(
            _param_row(
                p,
                current_values=current_values,
                form_id_prefix=form_id_prefix,
                point_picker_param=point_picker_param,
                picker_factory=picker_factory,
                columns_provider=columns_provider,
            )
        )
    return dbc.Form(rows)


__all__ = [
    "param_form",
    "parse_list_value",
    "parse_widget_value",
    "serialize_param_for_widget",
]
