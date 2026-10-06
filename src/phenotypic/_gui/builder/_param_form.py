"""Builder-flavoured parameter forms.

Re-exports the shared :mod:`phenotypic._gui._param_forms` machinery and
adds the builder's point-picker widget. Other modules (the analysis
sub-app, future tools) import from ``_gui/_param_forms.py`` directly;
the builder imports from here so its existing call sites
(``_callbacks.py``, the inspector pane) continue to work without
rewiring.
"""

from __future__ import annotations

from typing import Any, Callable

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
import numpy as np
from dash import dcc, html

from phenotypic._gui._operation_registry import OperationInfo
from phenotypic._gui._param_forms import (
    parse_list_value,
    parse_widget_value,
    serialize_param_for_widget,
)
from phenotypic._gui._param_forms import param_form as _shared_param_form  # noqa: F401  - re-export
from phenotypic._gui.builder import _ids as ids
from phenotypic._gui.builder._reference_metadata import REFERENCE_SOURCE
from phenotypic.sdk_ import ensure_metadata_prefix


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
      JSON-safe list of ``(y, x)`` pairs.
    * ``dbc.Button`` (``type=PICKER_PARAM_BTN_TYPE``) that opens the
      modal picker.
    * ``html.Span`` (``type=PICKER_PARAM_COUNT_TYPE``) displaying the
      current point count.
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


def _prefixed_reference_columns(
    op_info: OperationInfo,
    current_values: dict[str, Any],
    columns: list[str],
) -> dict[str, Any]:
    """Show a bare ``BlankImage`` as the ``Metadata_BlankImage`` it resolves to.

    ``ReferenceContext`` accepts either spelling, but the dropdown matches
    its normalized columns literally, so the bare one would render as stale
    although the run succeeds.
    """
    resolved = dict(current_values)
    for name, p in op_info.parameters.items():
        value = resolved.get(name)
        if (
            p.column_ref is None
            or p.column_ref.source != REFERENCE_SOURCE
            or not isinstance(value, str)
            or value in columns
        ):
            continue
        prefixed = ensure_metadata_prefix(value)
        if prefixed in columns:
            resolved[name] = prefixed
    return resolved


def param_form(
    op_info: OperationInfo,
    current_values: dict[str, Any],
    *,
    form_id_prefix: str,
    columns_provider: Callable[[str], list[str]] | None = None,
) -> dbc.Form:
    """Builder-flavoured ``param_form`` that injects the point picker.

    Delegates to :func:`phenotypic._gui._param_forms.param_form` after
    binding :func:`_picker_widget` as the picker factory. Other tools
    (analysis sub-app) call the shared function directly without the
    picker injection.

    Renders the parameter form for *one* block / step at a time. Aux-port
    wiring affordances (per-row Disconnect / reorder) live in the
    inspector's dedicated aux-ports section on the DAG path — they are
    not rendered inline by this function.

    Args:
        op_info: Registry metadata for the operation being edited.
        current_values: Mapping of parameter-name → current value used to
            seed each widget.
        form_id_prefix: Prefix added to every emitted component id —
            typically the consumer node's ``node_id`` (or the focused
            aux node's ``node_id`` when the inspector is focused on a
            wired aux).
        columns_provider: Column choices for ``RefColumn`` parameters, from
            the session's reference metadata table
            (:func:`._reference_metadata.reference_columns_provider`).
            ``None`` without a table: those parameters stay free text.
    """
    if columns_provider is not None:
        # A column dropdown shows only ``current_values`` (unlike the text
        # input, which falls back to the default), so an untouched param
        # would read "Pick a column…" while its default is what runs.
        current_values = {
            **{
                name: p.default
                for name, p in op_info.parameters.items()
                if p.column_ref is not None and p.has_default
            },
            **current_values,
        }
        current_values = _prefixed_reference_columns(
            op_info, current_values, columns_provider(REFERENCE_SOURCE)
        )
    return _shared_param_form(
        op_info,
        current_values,
        form_id_prefix=form_id_prefix,
        picker_factory=_picker_widget,
        columns_provider=columns_provider,
    )


__all__ = [
    "param_form",
    "parse_list_value",
    "parse_widget_value",
    "serialize_param_for_widget",
]
