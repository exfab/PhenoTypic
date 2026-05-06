"""Auto-generate Dash form widgets from OperationRegistry parameter metadata.

Maps :class:`~phenotypic.gui._operation_registry.ParamInfo` entries to Dash
Bootstrap widgets used by the pipeline builder's inspector pane. Pure functions
only — no Dash callbacks live here. Phase 3 wires the IDs emitted from this
module to ``dcc.Store`` updates.
"""

from __future__ import annotations

import collections.abc as abc
import enum
import inspect
import types
import typing
from typing import TYPE_CHECKING, Any, Optional, get_args, get_origin

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
import numpy as np
from dash import dcc, html

from phenotypic.gui.builder import _ids as ids

if TYPE_CHECKING:
    from phenotypic.gui._operation_registry import OperationInfo, ParamInfo


# ---------------------------------------------------------------------------
# Type-classification helpers
# ---------------------------------------------------------------------------


def _unwrap_optional(hint: Any) -> Any:
    """Return ``T`` from ``T | None`` annotations; otherwise pass through.

    Args:
        hint: A type hint (possibly a ``Union[T, None]`` / ``Optional[T]``).

    Returns:
        The non-``None`` member of an Optional/Union, or the original hint when
        the hint is not Optional.
    """
    origin = get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        non_none = [a for a in get_args(hint) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return hint


def _literal_options(hint: Any) -> list[Any] | None:
    """Extract literal values from ``Literal[...]`` annotations.

    Args:
        hint: Type hint to inspect.

    Returns:
        List of literal values when ``hint`` is a ``Literal[...]``; otherwise
        ``None``.
    """
    inner = _unwrap_optional(hint)
    if get_origin(inner) is typing.Literal:
        return list(get_args(inner))
    return None


def _enum_options(hint: Any) -> list[Any] | None:
    """Extract member values from ``enum.Enum`` annotations.

    Args:
        hint: Type hint to inspect.

    Returns:
        List of enum member values when ``hint`` is an Enum subclass; otherwise
        ``None``.
    """
    inner = _unwrap_optional(hint)
    if inspect.isclass(inner) and issubclass(inner, enum.Enum):
        return [m.value for m in inner]
    return None


def _is_list_type(hint: Any) -> bool:
    """Return ``True`` if ``hint`` is ``list[T]`` / ``List[T]`` / ``Iterable[T]``.

    Tuples report as ``False`` here — see :func:`_is_tuple_type`.
    """
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
    """Return ``True`` if ``hint`` is ``tuple[...]`` / ``Tuple[...]``."""
    inner = _unwrap_optional(hint)
    origin = get_origin(inner)
    if origin is tuple:
        return True
    return inner is tuple


def _list_item_type(hint: Any) -> Any:
    """Return the element type for ``list[T]`` / ``tuple[T, ...]`` annotations.

    Returns ``str`` when the element type cannot be determined.
    """
    inner = _unwrap_optional(hint)
    args = get_args(inner)
    if not args:
        return str
    # tuple[T, ...] — the ellipsis means homogeneous; both args[0] is the type
    return args[0]


# ---------------------------------------------------------------------------
# Value conversion helpers (used by Phase-3 callbacks)
# ---------------------------------------------------------------------------


def _coerce_scalar(text: Any, item_type: Any) -> Any:
    """Coerce a single token to a Python scalar using ``item_type``.

    Falls back to the original text when coercion fails so that validation can
    raise a meaningful error from the operation's ``__init__`` / ``__setattr__``.
    """
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
    """Parse a comma-separated text value into a list of typed elements.

    Args:
        text: Raw widget text (may be ``None`` or empty).
        item_type: Python type to coerce each token to.

    Returns:
        List of coerced values; empty list when ``text`` is empty.
    """
    if text is None:
        return []
    cleaned = str(text).strip()
    if not cleaned:
        return []
    tokens = [t.strip() for t in cleaned.split(",")]
    return [_coerce_scalar(t, item_type) for t in tokens if t != ""]


def parse_widget_value(raw: Any, p: "ParamInfo") -> Any:
    """Convert a Dash widget's reported value back to the right Python type.

    Handles tuple/list comma-split, int/float coercion, enum/literal pass-
    through, and bool fall-through. Tuples return ``tuple(...)`` so that ops
    like :class:`FrangiVesselness` (which re-coerces in ``__setattr__``) round-
    trip cleanly.

    Args:
        raw: Value reported by the Dash component.
        p: :class:`ParamInfo` describing the target parameter.

    Returns:
        Value coerced to the parameter's expected Python type.
    """
    inner = _unwrap_optional(p.type_hint)

    if raw is None:
        return None

    # Literal / Enum: values come back as strings (Select stores .value)
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

    # Bool
    if inner is bool:
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    # Numeric scalars
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

    # Lists / tuples
    if _is_list_type(p.type_hint):
        return parse_list_value(raw, _list_item_type(p.type_hint))
    if _is_tuple_type(p.type_hint):
        return tuple(parse_list_value(raw, _list_item_type(p.type_hint)))

    # Default: leave as text
    return raw


def serialize_param_for_widget(value: Any, p: "ParamInfo") -> Any:
    """Format a stored value for display in a Dash widget.

    Inverse of :func:`parse_widget_value`. Tuples / lists become comma-separated
    text suitable for the underlying ``dbc.Input``; enum members are reduced to
    their ``.value`` attribute; booleans pass through unchanged.

    Args:
        value: Value currently stored for the parameter.
        p: :class:`ParamInfo` describing the target parameter.

    Returns:
        A widget-ready representation (text, number, or bool).
    """
    if value is None:
        return None

    if isinstance(value, enum.Enum):
        return value.value

    if isinstance(value, np.ndarray):
        # Picker-style ndarrays (N x 2) round-trip via ``.tolist()`` so the
        # downstream list/tuple branch can render them. Scalar callers don't
        # reach this path (operations store ndarrays only for picker params).
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        return ", ".join(_format_token(v) for v in value)

    if isinstance(value, bool):
        return value

    return value


def _format_token(v: Any) -> str:
    """Render a sequence element as a stable string for the widget."""
    if isinstance(v, float):
        # Avoid ``2.0`` → ``"2.0"`` flapping when user originally typed ``2``.
        if v.is_integer():
            return str(int(v))
        return repr(v)
    return str(v)


# ---------------------------------------------------------------------------
# Widget builders
# ---------------------------------------------------------------------------


def _widget_for_param(
    p: "ParamInfo",
    *,
    current_value: Any,
    form_id_prefix: str,
    point_picker_param: Optional[str] = None,
) -> Any:
    """Build the primary input widget for a single parameter.

    Args:
        p: :class:`ParamInfo` from the operation registry.
        current_value: Existing value to populate the widget with.
        form_id_prefix: Prefix added to every generated component id so multiple
            forms can coexist on the same page without id collisions.
        point_picker_param: If set, the named parameter of the owning operation
            that should render as the point-picker widget (button + count +
            hidden ``dcc.Store``) instead of the default text input. Comes from
            :attr:`OperationInfo.point_picker_param`.

    Returns:
        A Dash component (typically a ``dbc.Input`` / ``dbc.Switch`` /
        ``dbc.Select`` / ``dbc.Button``).
    """
    # Picker swap takes precedence over every other dispatch branch — the
    # owning operation declares the parameter as point-pickable, and the
    # default text-input would lose the (y, x) structure.
    if point_picker_param is not None and p.name == point_picker_param:
        return _picker_widget(
            form_id_prefix=form_id_prefix,
            name=p.name,
            current_value=current_value,
        )

    inner = _unwrap_optional(p.type_hint)
    initial = (
        serialize_param_for_widget(current_value, p)
        if current_value is not None
        else serialize_param_for_widget(p.default, p)
    )

    # Operation / pipeline params: emit the placeholder "Edit" button.
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

    # Literal / Enum dropdowns.
    options = _literal_options(p.type_hint) or _enum_options(p.type_hint)
    if options is not None:
        return dbc.Select(
            id={"type": "param-enum", "prefix": form_id_prefix, "name": p.name},
            options=[{"label": str(v), "value": str(v)} for v in options],
            value=str(initial) if initial is not None else None,
        )

    # Bool → Switch.
    if inner is bool:
        return dbc.Switch(
            id={"type": "param-bool", "prefix": form_id_prefix, "name": p.name},
            label="",
            value=bool(initial) if initial is not None else False,
        )

    # int / float → numeric input.
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

    # list[T]
    if _is_list_type(p.type_hint):
        return dbc.Input(
            type="text",
            id={"type": "param-list", "prefix": form_id_prefix, "name": p.name},
            value=initial if isinstance(initial, str) else "",
            placeholder="comma-separated",
            debounce=True,
        )

    # tuple[...]
    if _is_tuple_type(p.type_hint):
        return dbc.Input(
            type="text",
            id={"type": "param-tuple", "prefix": form_id_prefix, "name": p.name},
            value=initial if isinstance(initial, str) else "",
            placeholder="comma-separated",
            debounce=True,
        )

    # str / fallback.
    return dbc.Input(
        type="text",
        id={"type": "param-str", "prefix": form_id_prefix, "name": p.name},
        value=initial if initial is not None else "",
        debounce=True,
    )


def _initial_picker_data(current_value: Any) -> list[list[float]]:
    """Normalise ``current_value`` into a JSON-safe ``[[y, x], …]`` list.

    Accepts ``None``, an N×2 ``numpy.ndarray``, or any sequence of
    length-2 pairs. Returns ``[]`` for empty / unrecognised input so the
    backing :class:`dcc.Store` always carries a serialisable payload.
    """
    if current_value is None:
        return []
    try:
        arr = np.asarray(current_value)
    except (TypeError, ValueError):
        return []
    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] != 2:
        return []
    return arr.tolist()


def _picker_widget(
    *,
    form_id_prefix: str,
    name: str,
    current_value: Any,
) -> Any:
    """Build the point-picker widget for a ``PointPickerMixin`` parameter.

    Returns a ``html.Div`` containing three components:

    * ``dcc.Store`` (``type=PICKER_PARAM_STORE_TYPE``) holding the
      JSON-safe list of ``(y, x)`` pairs. The picker fan-in callback in
      :mod:`._callbacks` reads this store's ``data`` directly —
      bypassing :func:`parse_widget_value` because the payload is
      already structured — and writes it into the owning node's
      ``params``.
    * ``dbc.Button`` (``type=PICKER_PARAM_BTN_TYPE``) that opens the
      modal picker (modal wiring lives in :mod:`._point_picker`).
    * ``html.Span`` (``type=PICKER_PARAM_COUNT_TYPE``) displaying the
      current point count.

    Args:
        form_id_prefix: Form id prefix (typically the owning node's id) so
            multiple forms can coexist without id collisions.
        name: Name of the picker-bound parameter (e.g. ``"centers"``).
        current_value: Existing parameter value to seed the store with.
            Accepts ``None``, ``list``, ``tuple``, or ``numpy.ndarray``.
    """
    initial_list = _initial_picker_data(current_value)
    return html.Div(
        [
            dcc.Store(
                id={
                    "type": ids.PICKER_PARAM_STORE_TYPE,
                    "prefix": form_id_prefix,
                    "name": name,
                },
                data=initial_list,
            ),
            dbc.Button(
                "Pick on image…",
                id={
                    "type": ids.PICKER_PARAM_BTN_TYPE,
                    "prefix": form_id_prefix,
                    "name": name,
                },
                color="primary",
                outline=True,
                size="sm",
                n_clicks=0,
            ),
            html.Span(
                f"{len(initial_list)} points",
                id={
                    "type": ids.PICKER_PARAM_COUNT_TYPE,
                    "prefix": form_id_prefix,
                    "name": name,
                },
                className="ms-2 text-muted small",
            ),
        ],
        className="d-flex align-items-center",
    )


def _optional_toggle(p: "ParamInfo", *, form_id_prefix: str) -> Any:
    """Build the "Use default" toggle shown beside Optional widgets.

    Args:
        p: Parameter being annotated. The toggle defaults to ``True`` when the
            parameter currently has its default-of-``None`` value.
        form_id_prefix: Prefix forwarded to the component id.
    """
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
    """Reserve an empty validation-message div for Phase-3 callbacks to fill."""
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
) -> Any:
    """Render one parameter as a labelled ``dbc.Row``.

    Args:
        p: :class:`ParamInfo` to render.
        current_values: Mapping of parameter-name → current value (may be empty).
        form_id_prefix: Prefix forwarded to every component id in the row.
        point_picker_param: Optional name of the parameter that should swap
            in the point-picker widget; threaded through to
            :func:`_widget_for_param`.
    """
    widget = _widget_for_param(
        p,
        current_value=current_values.get(p.name),
        form_id_prefix=form_id_prefix,
        point_picker_param=point_picker_param,
    )

    label_children: list[Any] = [p.name]
    label = dbc.Label(label_children, html_for=None, className="fw-semibold")

    helper: list[Any] = []
    if p.description:
        helper.append(dbc.FormText(p.description, color="secondary"))

    main_col_children: list[Any] = [widget, *helper, _error_slot(p, form_id_prefix=form_id_prefix)]
    cols: list[Any] = [
        dbc.Col(label, width=4),
        dbc.Col(main_col_children, width=6),
    ]
    # The "Use default" toggle only makes sense when the param has a default to
    # fall back to. ``Optional`` params without a default are required-with-a-
    # ``None``-sentinel and would otherwise silently produce ``TypeError`` from
    # ``create_instance`` if the toggle stripped them from the params dict.
    if p.is_optional and p.has_default:
        cols.append(
            dbc.Col(_optional_toggle(p, form_id_prefix=form_id_prefix), width=2)
        )
    else:
        cols.append(dbc.Col(width=2))

    return dbc.Row(cols, className="mb-3 align-items-center")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def param_form(
    op_info: "OperationInfo",
    current_values: dict[str, Any],
    *,
    form_id_prefix: str,
) -> dbc.Form:
    """Generate a parameter form for an operation.

    Walks ``op_info.parameters`` in declaration order and emits one labelled
    row per parameter. Widget choice is driven by the parameter's type hint:
    booleans become switches, numerics become number inputs, ``Literal`` and
    ``Enum`` types become dropdowns, sequences become comma-separated text
    inputs, and parameters typed as :class:`ImageOperation` /
    :class:`ImagePipeline` become a placeholder "Edit" button (Phase 3 wires
    the drill-down).

    Component ids follow the pattern-matching shape ``{"type": "param-<kind>",
    "prefix": form_id_prefix, "name": p.name}`` so multiple forms can coexist
    without collisions.

    Args:
        op_info: Registry metadata for the operation being edited.
        current_values: Mapping of parameter-name → current value used to seed
            each widget. Missing keys fall back to the parameter default.
        form_id_prefix: Prefix added to every emitted component id.

    Returns:
        ``dbc.Form`` whose children are one ``dbc.Row`` per parameter.

    Examples:
        >>> from phenotypic.gui import OperationRegistry
        >>> registry = OperationRegistry()
        >>> registry.discover()
        >>> info = registry.get('GaussianBlur')
        >>> form = param_form(info, current_values={'sigma': 2.0}, form_id_prefix='blur')
        >>> isinstance(form.children, list)
        True
    """
    point_picker_param = op_info.point_picker_param
    rows: list[Any] = []
    for p in op_info.parameters.values():
        rows.append(
            _param_row(
                p,
                current_values=current_values,
                form_id_prefix=form_id_prefix,
                point_picker_param=point_picker_param,
            )
        )
    return dbc.Form(rows)


__all__ = [
    "param_form",
    "parse_list_value",
    "parse_widget_value",
    "serialize_param_for_widget",
]
