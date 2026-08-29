"""Browse-tab callbacks + the pure helpers they wrap.

The helpers (``dataset_options``/``image_options``/``dataset_row_hidden``/
``sandbox_rel``/``current_image_payload``) are unit-tested; the Dash
callbacks are thin adapters so the live wiring is the only thing that needs
a browser smoke check.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Literal

import dash
from dash import Input, Output, State, ctx, html, no_update

from phenotypic.gui._config import (
    CFG_URL_PREFIX,
    MOUNT_HOME,
)
from phenotypic.gui._shared._picker_navigation import (
    enabled_picker_values,
    offset_picker_value,
    picker_button_disabled_states,
    picker_position,
    step_picker_value,
)
from phenotypic.gui.browse import _ids as ids
from phenotypic.gui.browse import _metadata, _source_lister, _source_render
from phenotypic.gui.browse._preparation_routes import BrowsePreparationApi
from phenotypic.gui.browse._source_probe import SourceRevision, probe_source
from phenotypic.gui.browse._layout import DATASET_ROW_STYLE
from phenotypic.gui.shell._ids import (
    SHELL_CLASSIFIER_CACHE_STORE,
    SHELL_METADATA_CSV_STORE,
    SHELL_SOURCE_IMAGE_ROOT_STORE,
)
from phenotypic.gui.shell._metadata_context import (
    MetadataLookupResult,
    read_metadata_row_for_image_stem,
)
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.shell._source_context import resolve_source_image_root

logger = logging.getLogger(__name__)

__all__ = ["register_callbacks"]

_ROOT_LABEL = "(root)"

#: How many images on each side of the active one to pre-warm. The Browse
#: client background-fetches these neighbours' ``.dzi`` so a ‹/› step lands on
#: an already-converted image (instant) instead of paying the multi-second
#: normalize + DZI-tile on click.
_PREFETCH_RADIUS = 3

CsvMetadataPanelState = Literal[
    "unset",
    "unavailable",
    "missing_image_name",
    "ambiguous_image_name",
    "no_match",
    "matched",
]


@dataclass(frozen=True)
class CsvMetadataPanelModel:
    """Display model for Browse's optional CSV metadata panel."""

    state: CsvMetadataPanelState
    image_stem: str
    rows: list[dict[str, str]]


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
    return [
        {"label": name, "value": name}
        for name in datasets.get(dataset or "", [])
    ]


def dataset_row_hidden(datasets: dict[str, list[str]]) -> bool:
    """True when the dataset dropdown should be hidden (flat or empty source)."""
    return set(datasets.keys()) in ({"."}, set())


def sandbox_rel(src_root_rel: str, dataset_rel: str, filename: str) -> str:
    """Join the image's path relative to the sandbox root (POSIX)."""
    parts = [p for p in (src_root_rel, dataset_rel) if p and p != "."]
    return PurePosixPath(*parts, filename).as_posix() if parts else filename


def neighbor_filenames(
    option_values: Sequence[str], current: str, radius: int = _PREFETCH_RADIUS
) -> list[str]:
    """The up-to-``radius`` filenames on each side of ``current`` in nav order.

    Returns the immediate neighbours within the dataset's ordered file list,
    excluding ``current`` and clamping at the list bounds — exactly the images
    a ‹/› step can reach next, so the client can pre-warm their conversion.
    Returns ``[]`` when ``current`` is not in the list.
    """
    values = list(option_values)
    try:
        idx = values.index(current)
    except ValueError:
        return []
    return (
        values[max(0, idx - radius) : idx] + values[idx + 1 : idx + 1 + radius]
    )


def directional_neighbor_filenames(
    option_values: Sequence[str],
    current: str,
    direction: Literal["forward", "backward", "unknown"] = "unknown",
) -> list[str]:
    """Return the fixed directional preparation order around ``current``."""
    values = list(option_values)
    try:
        index = values.index(current)
    except ValueError:
        return []
    offsets = {
        "forward": (1, 2, 3, -1, -2),
        "backward": (-1, -2, -3, 1, 2),
        "unknown": (1, -1, 2, -2, 3),
    }[direction]
    return [
        values[target]
        for offset in offsets
        if 0 <= (target := index + offset) < len(values)
    ]


def filmstrip_filenames(
    option_values: Sequence[str], current: str, radius: int = 4
) -> list[str]:
    """Return a centered, clamped filmstrip window of at most ``2r + 1``."""
    values = list(option_values)
    try:
        index = values.index(current)
    except ValueError:
        return []
    width = radius * 2 + 1
    start = max(0, min(len(values) - width, index - radius))
    return values[start : start + width]


def current_image_payload(
    src_root_rel: str,
    dataset_rel: str,
    filename: str,
    neighbor_files: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the ``{token, label[, prefetch]}`` current-image store payload.

    When ``neighbor_files`` is given, ``prefetch`` carries their tokens (same
    dataset, nav order) so the Browse client background-fetches each ``.dzi``
    to pre-warm the server-side normalize + DZI-tile cache.
    """
    rel = sandbox_rel(src_root_rel, dataset_rel, filename)
    payload: dict[str, Any] = {
        "token": _source_render.encode_token(rel),
        "label": rel,
    }
    if neighbor_files:
        payload["prefetch"] = [
            _source_render.encode_token(
                sandbox_rel(src_root_rel, dataset_rel, name)
            )
            for name in neighbor_files
        ]
    return payload


def render_csv_metadata_panel(model: CsvMetadataPanelModel) -> Any:
    """Render Browse's optional metadata CSV section."""
    if model.state == "unset":
        return html.Div("No metadata CSV selected", className="text-muted")
    if model.state == "unavailable":
        return html.Div(
            "Metadata CSV is unavailable", className="text-warning"
        )
    if model.state == "missing_image_name":
        return html.Div(
            "Metadata CSV has no image-name column",
            className="text-warning",
        )
    if model.state == "ambiguous_image_name":
        return html.Div(
            "Metadata CSV has conflicting image-name columns",
            className="text-warning",
        )
    if model.state == "no_match":
        return html.Div(
            f"No metadata row for {model.image_stem}",
            className="text-muted",
        )
    columns = list(dict.fromkeys(key for row in model.rows for key in row))
    rows = [
        html.Tr([html.Td(row.get(column, "-") or "-") for column in columns])
        for row in model.rows
    ]
    return html.Div(
        [
            html.Div(
                f"{len(model.rows)} metadata row"
                f"{'' if len(model.rows) == 1 else 's'} for {model.image_stem}",
                className="browse-csv-metadata-title",
            ),
            html.Div(
                html.Table(
                    [
                        html.Thead(
                            html.Tr([html.Th(column) for column in columns])
                        ),
                        html.Tbody(rows),
                    ],
                    className="table table-sm mb-0 browse-csv-metadata-table",
                ),
                className="browse-csv-metadata-scroll",
            ),
        ]
    )


def _csv_panel_model(result: MetadataLookupResult) -> CsvMetadataPanelModel:
    return CsvMetadataPanelModel(
        state=result.state,
        image_stem=result.image_stem,
        rows=result.rows,
    )


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
def register_callbacks(
    app: dash.Dash,
    sandbox: SandboxRoot,
    preparation_api: BrowsePreparationApi | None = None,
) -> None:
    """Register every Browse callback on ``app``."""
    @app.callback(
        Output(ids.BROWSE_DATASETS_STORE, "data"),
        Output(ids.BROWSE_DATASET_PICKER, "options"),
        Output(ids.BROWSE_DATASET_PICKER, "value"),
        Output(ids.BROWSE_DATASET_ROW, "style"),
        Output(ids.BROWSE_EMPTY_HINT, "style"),
        Input(SHELL_SOURCE_IMAGE_ROOT_STORE, "data"),
        Input(SHELL_CLASSIFIER_CACHE_STORE, "data"),
    )
    def _load_datasets(payload: Any, _refresh_revision: object):
        hidden_row = {**DATASET_ROW_STYLE, "display": "none"}
        resolved = resolve_source_image_root(sandbox, payload)
        if resolved is None:
            return {}, [], None, hidden_row, {"display": "block"}
        datasets = _source_lister.list_datasets(resolved)
        options = dataset_options(datasets)
        value = options[0]["value"] if options else None
        row_style = (
            hidden_row
            if dataset_row_hidden(datasets)
            else dict(DATASET_ROW_STYLE)
        )
        hint_style = {"display": "none"} if datasets else {"display": "block"}
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
        Output(ids.BROWSE_IMAGE_PICKER, "value", allow_duplicate=True),
        Input(ids.BROWSE_NAV_EVENT_STORE, "data"),
        State(ids.BROWSE_IMAGE_PICKER, "value"),
        State(ids.BROWSE_IMAGE_PICKER, "options"),
        prevent_initial_call=True,
    )
    def _navigate_from_client(event, value, options):
        if not isinstance(event, Mapping):
            raise dash.exceptions.PreventUpdate
        kind = event.get("kind")
        if kind == "offset":
            delta = event.get("delta")
            if not isinstance(delta, int) or delta not in {-10, -1, 1, 10}:
                raise dash.exceptions.PreventUpdate
            return offset_picker_value(value, options, delta) or no_update
        if kind == "select":
            selected = event.get("value")
            values = enabled_picker_values(options)
            return (
                selected
                if isinstance(selected, str) and selected in values
                else no_update
            )
        raise dash.exceptions.PreventUpdate

    @app.callback(
        Output(ids.BROWSE_PREPARATION_STATUS, "data-client-state-sync"),
        Input(ids.BROWSE_NAV_EVENT_STORE, "data"),
        prevent_initial_call=True,
    )
    def _sync_preparation_client_state(event):
        if preparation_api is None or not isinstance(event, Mapping):
            raise dash.exceptions.PreventUpdate
        client_id = event.get("session_id")
        enabled = event.get("speculation_enabled")
        if not isinstance(client_id, str) or not client_id:
            raise dash.exceptions.PreventUpdate
        if isinstance(enabled, bool):
            preparation_api.manager.set_speculation_enabled(client_id, enabled)
        return client_id

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
        State(ids.BROWSE_IMAGE_PICKER, "options"),
        State(ids.BROWSE_DATASET_PICKER, "value"),
        State(SHELL_SOURCE_IMAGE_ROOT_STORE, "data"),
        State(ids.BROWSE_CURRENT_IMAGE_STORE, "data"),
        State(ids.BROWSE_NAV_EVENT_STORE, "data"),
    )
    def _current_image(filename, options, dataset, payload, prior, nav_event):
        if not filename:
            return None
        src_root_rel = _src_root_rel(sandbox, payload)
        source_root = resolve_source_image_root(sandbox, payload)
        if src_root_rel is None or source_root is None:
            return None
        option_values = enabled_picker_values(options)
        relative = sandbox_rel(src_root_rel, dataset or ".", filename)
        token = _source_render.encode_token(relative)
        try:
            selected_revision = probe_source(
                sandbox.resolve(relative),
                sandbox_root=sandbox.root,
                relative_path=relative,
            )
        except Exception:  # noqa: BLE001 - selection can disappear during refresh
            return {"token": token, "label": relative, "filename": filename}

        previous = (
            prior.get("filename") if isinstance(prior, Mapping) else None
        )
        try:
            previous_index = option_values.index(str(previous))
            current_index = option_values.index(filename)
        except ValueError:
            direction: Literal["forward", "backward", "unknown"] = "unknown"
        else:
            direction = (
                "forward"
                if current_index > previous_index
                else "backward"
                if current_index < previous_index
                else "unknown"
            )
        neighbor_names = directional_neighbor_filenames(
            option_values,
            filename,
            direction,
        )
        filmstrip_names = filmstrip_filenames(option_values, filename)
        client_id = "browse-server"
        generation = 0
        if isinstance(nav_event, Mapping):
            candidate = nav_event.get("session_id")
            sequence = nav_event.get("sequence")
            if isinstance(candidate, str) and candidate:
                client_id = candidate
            if isinstance(sequence, int) and sequence >= 0:
                generation = sequence

        revisions: dict[str, SourceRevision] = {filename: selected_revision}
        for name in dict.fromkeys([*neighbor_names, *filmstrip_names]):
            rel = sandbox_rel(src_root_rel, dataset or ".", name)
            try:
                revisions[name] = probe_source(
                    sandbox.resolve(rel),
                    sandbox_root=sandbox.root,
                    relative_path=rel,
                )
            except Exception:  # noqa: BLE001 - inventory can change in place
                continue
        if preparation_api is not None:
            preparation_api.replace_nearby(
                client_id,
                generation,
                [
                    revisions[name]
                    for name in neighbor_names
                    if name in revisions
                ],
            )

        prefix = str(app.server.config.get(CFG_URL_PREFIX, MOUNT_HOME))
        if not prefix.endswith("/"):
            prefix += "/"
        position, total = picker_position(filename, options)
        filmstrip = []
        for name in filmstrip_names:
            revision = revisions.get(name)
            if revision is None:
                continue
            item_token = _source_render.encode_token(
                sandbox_rel(src_root_rel, dataset or ".", name)
            )
            status = "queued"
            if preparation_api is not None:
                entry = preparation_api.cache.entry(revision)
                if entry.dzi_ready:
                    status = "ready"
                else:
                    try:
                        phase = preparation_api.manager.snapshot(
                            revision
                        ).phase
                    except KeyError:
                        phase = "queued"
                    status = (
                        "failed"
                        if phase in {"failed", "cancelled"}
                        else "preparing"
                        if phase not in {"queued", "ready"}
                        else phase
                    )
            filmstrip.append(
                {
                    "value": name,
                    "label": name,
                    "preview_url": (
                        f"{prefix}assets/{item_token}/{revision.cache_key}/"
                        "preview-if-ready.png"
                    ),
                    "status": status,
                    "current": name == filename,
                }
            )
        return {
            "token": token,
            "label": relative,
            "filename": filename,
            "value": filename,
            "revision": selected_revision.cache_key,
            "width": selected_revision.width,
            "height": selected_revision.height,
            "position": {"index": position, "total": total},
            "filmstrip": filmstrip,
            "preview_url": (
                f"{prefix}assets/{token}/{selected_revision.cache_key}/preview.png"
            ),
            "dzi_url": (
                f"{prefix}assets/{token}/{selected_revision.cache_key}/image.dzi"
            ),
        }

    @app.callback(
        Output(ids.BROWSE_META_IMAGE_NAME, "children"),
        Output(ids.BROWSE_META_DIMS, "children"),
        Output(ids.BROWSE_META_SIZE, "children"),
        Output(ids.BROWSE_META_CAPTURED, "children"),
        Output(ids.BROWSE_META_CAMERA, "children"),
        Input(ids.BROWSE_CURRENT_IMAGE_STORE, "data"),
    )
    def _metadata_panel(payload: dict | None):
        if not payload or not payload.get("token"):
            return "—", "—", "—", "—", "—"
        image_name = payload.get("filename")
        if not isinstance(image_name, str) or not image_name:
            image_name = "—"
        try:
            rel = _source_render.decode_token(payload["token"])
            original = sandbox.resolve(rel)
        except Exception:  # noqa: BLE001 - metadata is best-effort
            return image_name, "—", "—", "—", "—"
        image_name = Path(rel).name or "—"
        info = _metadata.read(original)
        dims = (
            f"{info['width']} × {info['height']} px"
            if info["width"] and info["height"]
            else "—"
        )
        size = _humanize_bytes(info["bytes"]) if info["bytes"] else "—"
        exif = info.get("exif", {})
        captured = exif.get("captured", "—")
        camera = (
            " ".join(p for p in (exif.get("make"), exif.get("model")) if p)
            or "—"
        )
        return image_name, dims, size, captured, camera

    @app.callback(
        Output(ids.BROWSE_CSV_METADATA_PANEL, "children"),
        Input(ids.BROWSE_CURRENT_IMAGE_STORE, "data"),
        Input(SHELL_METADATA_CSV_STORE, "data"),
    )
    def _csv_metadata_panel(
        image_payload: dict | None,
        metadata_payload: object,
    ) -> Any:
        if not image_payload or not image_payload.get("token"):
            return render_csv_metadata_panel(
                CsvMetadataPanelModel("unset", "", [])
            )
        try:
            rel = _source_render.decode_token(image_payload["token"])
            original = sandbox.resolve(rel)
        except Exception:  # noqa: BLE001 - metadata display is best-effort
            return render_csv_metadata_panel(
                CsvMetadataPanelModel("unavailable", "", [])
            )
        result = read_metadata_row_for_image_stem(
            sandbox,
            metadata_payload,
            Path(original).stem,
        )
        return render_csv_metadata_panel(_csv_panel_model(result))

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

    @app.callback(
        Output(ids.BROWSE_PREPARATION_STATUS_STORE, "data"),
        Input(ids.BROWSE_PREPARE_BTN, "n_clicks"),
        Input(ids.BROWSE_STOP_PREPARE_BTN, "n_clicks"),
        Input(ids.BROWSE_CLEAR_CACHE_BTN, "n_clicks"),
        Input(ids.BROWSE_PREPARATION_POLL, "n_intervals"),
        State(ids.BROWSE_IMAGE_PICKER, "options"),
        State(ids.BROWSE_DATASET_PICKER, "value"),
        State(SHELL_SOURCE_IMAGE_ROOT_STORE, "data"),
        State(ids.BROWSE_CURRENT_IMAGE_STORE, "data"),
        State(ids.BROWSE_NAV_EVENT_STORE, "data"),
    )
    def _preparation_controls(
        prepare_clicks,
        _stop_clicks,
        _clear_clicks,
        _poll,
        options,
        dataset,
        source_payload,
        current_image,
        nav_event,
    ):
        if preparation_api is None:
            return {
                "state": "idle",
                "ready": 0,
                "total": 0,
                "failed": 0,
                "message": "Images prepare as you browse.",
            }
        client_id = "browse-server"
        generation = int(prepare_clicks or 0)
        if isinstance(nav_event, Mapping):
            candidate = nav_event.get("session_id")
            if isinstance(candidate, str) and candidate:
                client_id = candidate
        triggered = ctx.triggered_id
        if triggered == ids.BROWSE_PREPARE_BTN:
            src_root_rel = _src_root_rel(sandbox, source_payload)
            if src_root_rel is None:
                raise dash.exceptions.PreventUpdate
            revisions = []
            for filename in enabled_picker_values(options):
                relative = sandbox_rel(src_root_rel, dataset or ".", filename)
                try:
                    revisions.append(
                        probe_source(
                            sandbox.resolve(relative),
                            sandbox_root=sandbox.root,
                            relative_path=relative,
                        )
                    )
                except Exception:  # noqa: BLE001 - source changed during scan
                    continue
            payload = preparation_api.start_dataset(
                client_id,
                generation,
                revisions,
            )
        elif triggered == ids.BROWSE_STOP_PREPARE_BTN:
            payload = preparation_api.stop_dataset(client_id)
        elif triggered == ids.BROWSE_CLEAR_CACHE_BTN:
            revision = (
                current_image.get("revision")
                if isinstance(current_image, Mapping)
                else None
            )
            cleared = preparation_api.clear(current_revision=revision)
            payload = preparation_api.status(client_id)
            payload["message"] = (
                f"Cleared {cleared['removed_entries']} prepared entries."
            )
        else:
            payload = preparation_api.status(client_id)
        usage = payload.get("cache_usage")
        if isinstance(usage, Mapping):
            payload["cache_usage"] = (
                f"{_humanize_bytes(int(usage.get('bytes', 0)))} in "
                f"{int(usage.get('entries', 0))} entries ({usage.get('tier', 'unknown')})"
            )
        return payload

    app.clientside_callback(
        """
        function(payload) {
            if (window.__phenotypicBrowse) {
                return window.__phenotypicBrowse.applyPreparationStatus(payload);
            }
            return "";
        }
        """,
        Output(ids.BROWSE_PREPARATION_STATUS, "data-render-sync"),
        Input(ids.BROWSE_PREPARATION_STATUS_STORE, "data"),
    )


def _humanize_bytes(n: int) -> str:
    """Compact human-readable file size."""
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return (
                f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
            )
        size /= 1024
    return f"{size:.1f} GB"
