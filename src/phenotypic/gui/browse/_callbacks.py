"""Browse-tab callbacks + the pure helpers they wrap.

The helpers (``dataset_options``/``image_options``/``dataset_row_hidden``/
``sandbox_rel``/``current_image_payload``) are unit-tested; the Dash
callbacks are thin adapters so the live wiring is the only thing that needs
a browser smoke check.
"""
from __future__ import annotations

import logging
from pathlib import PurePosixPath
from typing import Any

import dash
from dash import Input, Output, State, ctx, no_update

from phenotypic.gui._shared._picker_navigation import (
    picker_button_disabled_states,
    step_picker_value,
)
from phenotypic.gui.browse import _ids as ids
from phenotypic.gui.browse import _metadata, _source_lister, _source_render
from phenotypic.gui.browse._layout import DATASET_ROW_STYLE
from phenotypic.gui.shell._ids import SHELL_SOURCE_IMAGE_ROOT_STORE
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.shell._source_context import resolve_source_image_root

logger = logging.getLogger(__name__)

__all__ = ["register_callbacks"]

_ROOT_LABEL = "(root)"


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------
def dataset_options(datasets: dict[str, list[str]]) -> list[dict[str, str]]:
    """Dropdown options for the dataset picker (``.`` shown as ``(root)``)."""
    return [
        {"label": _ROOT_LABEL if key == "." else key, "value": key}
        for key in datasets
    ]


def image_options(
    datasets: dict[str, list[str]], dataset: str | None
) -> list[dict[str, str]]:
    """Dropdown options for the image picker within ``dataset``."""
    return [{"label": name, "value": name} for name in datasets.get(dataset or "", [])]


def dataset_row_hidden(datasets: dict[str, list[str]]) -> bool:
    """True when the dataset dropdown should be hidden (flat or empty source)."""
    return set(datasets.keys()) in ({"."}, set())


def sandbox_rel(src_root_rel: str, dataset_rel: str, filename: str) -> str:
    """Join the image's path relative to the sandbox root (POSIX)."""
    parts = [p for p in (src_root_rel, dataset_rel) if p and p != "."]
    return PurePosixPath(*parts, filename).as_posix() if parts else filename


def current_image_payload(
    src_root_rel: str, dataset_rel: str, filename: str
) -> dict[str, str]:
    """Build the ``{token, label}`` current-image store payload."""
    rel = sandbox_rel(src_root_rel, dataset_rel, filename)
    return {"token": _source_render.encode_token(rel), "label": rel}


def _src_root_rel(sandbox: SandboxRoot, payload: Any) -> str | None:
    """Resolve the source root and return its path relative to the sandbox."""
    resolved = resolve_source_image_root(sandbox, payload)
    if resolved is None:
        return None
    try:
        return resolved.relative_to(sandbox.root).as_posix()
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Callback registration
# --------------------------------------------------------------------------
def register_callbacks(app: dash.Dash, sandbox: SandboxRoot) -> None:
    """Register every Browse callback on ``app``."""

    @app.callback(
        Output(ids.BROWSE_DATASETS_STORE, "data"),
        Output(ids.BROWSE_DATASET_PICKER, "options"),
        Output(ids.BROWSE_DATASET_PICKER, "value"),
        Output(ids.BROWSE_DATASET_ROW, "style"),
        Output(ids.BROWSE_EMPTY_HINT, "style"),
        Input(SHELL_SOURCE_IMAGE_ROOT_STORE, "data"),
    )
    def _load_datasets(payload: Any):
        hidden_row = {**DATASET_ROW_STYLE, "display": "none"}
        resolved = resolve_source_image_root(sandbox, payload)
        if resolved is None:
            return {}, [], None, hidden_row, {"display": "block"}
        datasets = _source_lister.list_datasets(resolved)
        options = dataset_options(datasets)
        value = options[0]["value"] if options else None
        row_style = hidden_row if dataset_row_hidden(datasets) else dict(DATASET_ROW_STYLE)
        hint_style = {"display": "block"} if not datasets else {"display": "none"}
        return datasets, options, value, row_style, hint_style

    @app.callback(
        Output(ids.BROWSE_IMAGE_PICKER, "options"),
        Output(ids.BROWSE_IMAGE_PICKER, "value"),
        Input(ids.BROWSE_DATASET_PICKER, "value"),
        State(ids.BROWSE_DATASETS_STORE, "data"),
    )
    def _cascade_images(dataset: str | None, datasets: dict | None):
        options = image_options(datasets or {}, dataset)
        value = options[0]["value"] if options else None
        return options, value

    @app.callback(
        Output(ids.BROWSE_IMAGE_PICKER, "value", allow_duplicate=True),
        Input(ids.BROWSE_PREV_BTN, "n_clicks"),
        Input(ids.BROWSE_NEXT_BTN, "n_clicks"),
        State(ids.BROWSE_IMAGE_PICKER, "value"),
        State(ids.BROWSE_IMAGE_PICKER, "options"),
        prevent_initial_call=True,
    )
    def _step_image(_p, _n, value, options):
        triggered = ctx.triggered_id
        direction = "previous" if triggered == ids.BROWSE_PREV_BTN else "next"
        return step_picker_value(value, options, direction) or no_update

    @app.callback(
        Output(ids.BROWSE_PREV_BTN, "disabled"),
        Output(ids.BROWSE_NEXT_BTN, "disabled"),
        Input(ids.BROWSE_IMAGE_PICKER, "value"),
        Input(ids.BROWSE_IMAGE_PICKER, "options"),
    )
    def _bounds(value, options):
        return picker_button_disabled_states(value, options)

    @app.callback(
        Output(ids.BROWSE_CURRENT_IMAGE_STORE, "data"),
        Input(ids.BROWSE_IMAGE_PICKER, "value"),
        State(ids.BROWSE_DATASET_PICKER, "value"),
        State(SHELL_SOURCE_IMAGE_ROOT_STORE, "data"),
    )
    def _current_image(filename, dataset, payload):
        if not filename:
            return None
        src_root_rel = _src_root_rel(sandbox, payload)
        if src_root_rel is None:
            return None
        return current_image_payload(src_root_rel, dataset or ".", filename)

    @app.callback(
        Output(ids.BROWSE_META_DIMS, "children"),
        Output(ids.BROWSE_META_SIZE, "children"),
        Output(ids.BROWSE_META_CAPTURED, "children"),
        Output(ids.BROWSE_META_CAMERA, "children"),
        Input(ids.BROWSE_CURRENT_IMAGE_STORE, "data"),
    )
    def _metadata_panel(payload: dict | None):
        if not payload or not payload.get("token"):
            return "—", "—", "—", "—"
        try:
            rel = _source_render.decode_token(payload["token"])
            original = sandbox.resolve(rel)
        except Exception:  # noqa: BLE001 - metadata is best-effort
            return "—", "—", "—", "—"
        info = _metadata.read(original)
        dims = (
            f"{info['width']} × {info['height']} px"
            if info["width"] and info["height"]
            else "—"
        )
        size = _humanize_bytes(info["bytes"]) if info["bytes"] else "—"
        exif = info.get("exif", {})
        captured = exif.get("captured", "—")
        camera = " ".join(p for p in (exif.get("make"), exif.get("model")) if p) or "—"
        return dims, size, captured, camera

    # Clientside: mount/replace the single OSD viewer on image change.
    app.clientside_callback(
        """
        function(payload) {
            if (window.__phenotypicBrowse) {
                window.__phenotypicBrowse.applyImage(payload);
            }
            return "";
        }
        """,
        Output(ids.BROWSE_OSD_SYNC, "data"),
        Input(ids.BROWSE_CURRENT_IMAGE_STORE, "data"),
    )


def _humanize_bytes(n: int) -> str:
    """Compact human-readable file size."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
