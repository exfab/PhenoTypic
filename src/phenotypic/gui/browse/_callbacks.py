"""Browse-tab callbacks + the pure helpers they wrap.

The helpers (``dataset_options``/``image_options``/``dataset_row_hidden``/
``sandbox_rel``/``current_image_payload``) are unit-tested; the Dash
callbacks are thin adapters so the live wiring is the only thing that needs
a browser smoke check.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import dash
from dash import Input, Output, State, ctx, html, no_update

from phenotypic.gui._config import (
    BROWSE_THUMB_URL_SEGMENT,
    CFG_URL_PREFIX,
    MOUNT_HOME,
    TIMELINE_TILE_SIZE_DEFAULT,
    snap_thumb_bucket,
    stepped_timeline_tile_size_from_trigger,
)
from phenotypic.gui._shared._picker_navigation import (
    enabled_picker_values,
    offset_picker_value,
    picker_button_disabled_states,
    picker_position,
    step_picker_value,
)
from phenotypic.gui._shared.timeline import build_matrix, build_timeline_grid
from phenotypic.gui.browse import _ids as ids
from phenotypic.gui.browse import _metadata, _source_lister, _source_render
from phenotypic.gui.browse._preparation_routes import BrowsePreparationApi
from phenotypic.gui.browse._source_item import source_item_relative_path
from phenotypic.gui.browse._source_probe import SourceRevision, probe_source
from phenotypic.gui.browse._capture_time import read_capture_time
from phenotypic.gui.browse._layout import DATASET_ROW_STYLE
from phenotypic.gui.browse._plate_pattern import (
    PatternError,
    parse_plate_identity,
)
from phenotypic.gui.browse._timeline_records import (
    BrowseAxisConfig,
    build_browse_records,
)
from phenotypic.gui.shell._ids import (
    SHELL_CLASSIFIER_CACHE_STORE,
    SHELL_METADATA_CSV_STORE,
    SHELL_SOURCE_IMAGE_ROOT_STORE,
)
from phenotypic.gui.shell._metadata_context import (
    MetadataLookupResult,
    read_metadata_csv_table,
    read_metadata_row_for_image_stem,
    resolve_metadata_csv,
    resolve_metadata_image_identity,
)
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.gui.shell._source_context import resolve_source_image_root
from phenotypic.sdk_ import source_image_stem

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


@dataclass(frozen=True)
class _AuthorizedTimelineRevision:
    """Server-side authority for one browser's live Timeline generation."""

    generation: int
    revision: str
    source_root: Path


@dataclass
class _SourceRevisionState:
    """Mutable server authority for one browser's source refresh lifecycle."""

    generation: int = 0
    revision: str | None = None
    reset_in_progress: bool = False
    grid_revisions: set[str] = field(default_factory=set)


class TimelineRevisionAuthority:
    """Thread-safe current Timeline revision authority keyed by browser session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_session: dict[str, _AuthorizedTimelineRevision] = {}

    def authorize(
        self,
        session_id: str,
        generation: int,
        revision: str,
        source_root: Path,
    ) -> bool:
        """Publish a revision unless a newer browser generation already won."""
        with self._lock:
            current = self._by_session.get(session_id)
            if current is not None:
                if generation < current.generation:
                    return False
                if (
                    generation == current.generation
                    and revision != current.revision
                ):
                    return False
            self._by_session[session_id] = _AuthorizedTimelineRevision(
                generation=generation,
                revision=revision,
                source_root=source_root,
            )
        return True

    def current(
        self,
        session_id: str,
        generation: int,
        revision: str,
    ) -> _AuthorizedTimelineRevision | None:
        """Return authority only when every live-generation field matches."""
        with self._lock:
            current = self._by_session.get(session_id)
        if (
            current is None
            or current.generation != generation
            or current.revision != revision
        ):
            return None
        return current

    def retire(self, session_id: str) -> None:
        """Retire every popout event authorized for ``session_id``."""
        with self._lock:
            self._by_session.pop(session_id, None)

    def retire_if_matches(
        self,
        session_id: str,
        generation: int,
        revision: str,
    ) -> None:
        """Retire one stale authorization without clobbering a newer one."""
        with self._lock:
            current = self._by_session.get(session_id)
            if (
                current is not None
                and current.generation == generation
                and current.revision == revision
            ):
                self._by_session.pop(session_id, None)


class SourceRevisionAuthority:
    """Thread-safe source refresh authority isolated by browser session."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_session: dict[str, _SourceRevisionState] = {}

    def ensure_session(self, session_id: str) -> None:
        """Create the initial, unrefreshed authority for ``session_id``."""
        with self._lock:
            self._by_session.setdefault(session_id, _SourceRevisionState())

    def begin_reset(self, session_id: str) -> int:
        """Start a newer reset and immediately retire its session's grids."""
        with self._lock:
            return self._begin_reset_locked(session_id)

    def begin_reset_and_retire(
        self,
        session_id: str,
        revision_authority: TimelineRevisionAuthority,
    ) -> int:
        """Start a reset and retire popouts under one source-generation lock."""
        with self._lock:
            generation = self._begin_reset_locked(session_id)
            revision_authority.retire(session_id)
            return generation

    def _begin_reset_locked(self, session_id: str) -> int:
        """Begin a reset while ``self._lock`` is held."""
        current = self._by_session.setdefault(
            session_id,
            _SourceRevisionState(),
        )
        current.generation += 1
        current.revision = None
        current.reset_in_progress = True
        current.grid_revisions.clear()
        return current.generation

    def publish_reset(
        self,
        session_id: str,
        generation: int,
        revision: str,
    ) -> bool:
        """Publish only if no newer reset for this session has started."""
        with self._lock:
            current = self._by_session.get(session_id)
            if current is None or current.generation != generation:
                return False
            current.revision = revision
            current.reset_in_progress = False
            return True

    def is_current(self, session_id: str, revision: str | None) -> bool:
        """Return whether ``revision`` is the live source refresh revision."""
        with self._lock:
            current = self._by_session.get(session_id)
            return (
                current is not None
                and not current.reset_in_progress
                and current.revision == revision
            )

    def authorize_grid(
        self,
        session_id: str,
        source_revision: str | None,
        grid_revision: str,
    ) -> bool:
        """Publish a grid identity only while its source revision is current."""
        with self._lock:
            current = self._by_session.get(session_id)
            if (
                current is None
                or current.reset_in_progress
                or current.revision != source_revision
            ):
                return False
            current.grid_revisions.add(grid_revision)
            return True

    def grid_is_current(self, session_id: str, grid_revision: str) -> bool:
        """Return whether a grid was published for the live source revision."""
        with self._lock:
            current = self._by_session.get(session_id)
            return (
                current is not None
                and not current.reset_in_progress
                and grid_revision in current.grid_revisions
            )

    def grid_authorization_generation(
        self,
        session_id: str,
        grid_revision: str,
    ) -> int | None:
        """Return the source generation authorizing ``grid_revision``."""
        with self._lock:
            current = self._by_session.get(session_id)
            if (
                current is None
                or current.reset_in_progress
                or grid_revision not in current.grid_revisions
            ):
                return None
            return current.generation

    def grid_generation_is_current(
        self,
        session_id: str,
        grid_revision: str,
        generation: int,
    ) -> bool:
        """Return whether the exact source/grid generation remains current."""
        with self._lock:
            current = self._by_session.get(session_id)
            return (
                current is not None
                and not current.reset_in_progress
                and current.generation == generation
                and grid_revision in current.grid_revisions
            )


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
    return source_item_relative_path(src_root_rel, dataset_rel, filename)


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


def source_asset_payload(
    prefix: str,
    token: str,
    revision: str,
    *,
    is_store: bool,
) -> dict[str, str | None]:
    """Return the renderer-specific, generation-addressed asset URLs.

    Flat images keep Browse's normalized DZI path. A Zarr store exposes its
    immutable members directly for the shared Viv facade, so no level-0 PNG or
    redundant server-side pyramid is materialized.
    """
    base = prefix if prefix.endswith("/") else f"{prefix}/"
    asset = f"{base}assets/{token}/{revision}"
    if is_store:
        return {
            "render_kind": "ome-zarr",
            "preview_url": None,
            "dzi_url": None,
            "store_url": f"{asset}/zarr/",
        }
    return {
        "render_kind": "dzi",
        "preview_url": f"{asset}/preview.png",
        "dzi_url": f"{asset}/image.dzi",
        "store_url": None,
    }


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
# Timeline pure helpers (unit-tested; the Dash callbacks below are thin wrappers)
# --------------------------------------------------------------------------
def timeline_thumb_url(prefix: str, token: str, fetch_size: int) -> str:
    """Build a thumbnail ``<img>`` URL for the Browse thumb route."""
    return f"{prefix}{BROWSE_THUMB_URL_SEGMENT}/{token}?size={fetch_size}"


def timeline_revision_token(*parts: object) -> str:
    """Return a deterministic identity for one rendered Timeline generation.

    The token is browser-session state, not filesystem authority. It binds a
    client event to the exact source, metadata, axis, pattern, and tile inputs
    that produced the live grid so delayed events from a retired generation
    fail closed.
    """
    encoded = json.dumps(
        parts,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_timeline_grid(
    records: Sequence[dict[str, object]], *, display_size: int, prefix: str
) -> Any:
    """Build matrix → grid component (encoding each cell_ref to a thumb token).

    Each record's sandbox-rel ``cell_ref`` is encoded to a base64url token via
    :func:`_source_render.encode_token` before the thumbnail URL is built; the
    same token is written into each cell's ``data-ref`` (the pop-out identity).
    """
    fetch_size = snap_thumb_bucket(display_size)

    def _url_builder(cell_ref: object, fetch: int) -> str:
        token = _source_render.encode_token(str(cell_ref))
        return timeline_thumb_url(prefix, token, fetch)

    def _ref_builder(cell_ref: object) -> str:
        return _source_render.encode_token(str(cell_ref))

    matrix = build_matrix(records)
    component, _grid_order = build_timeline_grid(
        matrix,
        url_builder=_url_builder,
        display_size=display_size,
        fetch_size=fetch_size,
        ref_builder=_ref_builder,
    )
    return component


def pattern_preview_rows(
    datasets: dict[str, list[str]], pattern: str, advanced: bool
) -> Any:
    """Render a small live preview of the plate-identity pattern over stems.

    Shows, per dataset folder, how the first few filenames resolve into
    ``{plate}`` / ``{time}`` captures so the user can iterate on the pattern
    before applying it. Invalid patterns surface their :class:`PatternError`.
    """
    if not pattern:
        return html.Div(
            "Enter a pattern to preview matches.", className="text-muted"
        )

    flat_stems = [
        source_image_stem(Path(name))
        for files in datasets.values()
        for name in files
    ]
    preview_stems = flat_stems[:8]
    try:
        matches = parse_plate_identity(
            preview_stems, pattern, advanced=advanced
        )
    except PatternError as exc:
        return html.Div(f"Invalid pattern: {exc}", className="text-danger")

    if not matches:
        return html.Div("No filenames to preview.", className="text-muted")

    body = [
        html.Tr(
            [
                html.Td(match.stem),
                html.Td(match.plate if match.plate is not None else "—"),
                html.Td(match.time if match.time is not None else "—"),
            ]
        )
        for match in matches
    ]
    return html.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Filename"),
                        html.Th("{plate}"),
                        html.Th("{time}"),
                    ]
                )
            ),
            html.Tbody(body),
        ],
        className="table table-sm mb-0 browse-tl-pattern-preview-table",
    )


def _csv_column_options(columns: Sequence[str]) -> list[dict[str, str]]:
    """Dropdown options for the CSV column / image-name dropdowns."""
    return [{"label": column, "value": column} for column in columns]


def csv_column_options_and_image_default(
    columns: Sequence[str],
    rows: Sequence[dict[str, str]],
) -> tuple[list[dict[str, str]], str | None]:
    """Return Timeline CSV options and a compatible image-column default."""
    options = _csv_column_options(columns)
    identity = resolve_metadata_image_identity(columns, rows)
    default = identity.column if identity.state == "resolved" else None
    return options, default


def strip_popout_nonce(value: str) -> str:
    """Strip the ``#<nonce>`` uniqueness suffix timeline.js appends to a token.

    ``setBridge`` (timeline.js) appends ``#<monotonic-counter>`` to every
    bridge write so re-opening the pop-out on the SAME cell still changes the
    controlled ``dcc.Input`` value (Dash's onChange only fires on a change). As
    ``#`` is outside the base64url token alphabet, splitting on the first ``#``
    recovers the original token (POP-OUT M5). Surface-agnostic: Results decodes
    the same shape.
    """
    return value.split("#", 1)[0]


def warnings_alert_state(warnings: Sequence[str] | None) -> tuple[Any, bool]:
    """Render the CSV-join warnings alert body + open-state.

    Returns ``(children, is_open)``: a stacked list of warning lines (open)
    when ``warnings`` is non-empty, else ``(None, False)`` so the alert stays
    hidden. Surfaces the otherwise-dead ``BROWSE_TL_STORE_WARNINGS`` store
    (e.g. cross-folder stem collisions) to the user.
    """
    items = [w for w in (warnings or []) if w]
    if not items:
        return None, False
    return [html.Div(message) for message in items], True


def source_reset_values(
    source_payload: object,
    refresh_revision: object = None,
) -> tuple[object, ...]:
    """Return the complete source-dependent Timeline reset transaction."""
    revision = timeline_revision_token(
        "source",
        source_payload,
        "refresh",
        refresh_revision,
    )
    return (
        "folder",
        "exif",
        None,
        None,
        None,
        "",
        [],
        pattern_preview_rows({}, "", False),
        f"{TIMELINE_TILE_SIZE_DEFAULT} px",
        TIMELINE_TILE_SIZE_DEFAULT,
        html.Div("Loading current source…", className="text-muted"),
        [],
        f"{revision}:reset",
        revision,
        None,
        None,
        False,
        None,
        "",
    )


def authorize_revision_candidate(
    authority: TimelineRevisionAuthority,
    sandbox: SandboxRoot,
    candidate: Mapping[str, object] | None,
    source_payload: object,
) -> dict[str, object] | None:
    """Authorize one browser-applied grid generation on the server."""
    if not isinstance(candidate, Mapping):
        return None
    session_id = candidate.get("session_id")
    generation = candidate.get("generation")
    revision = candidate.get("revision")
    if (
        not isinstance(session_id, str)
        or not session_id
        or type(generation) is not int
        or generation < 1
        or not isinstance(revision, str)
        or not revision
    ):
        return None
    source_root = resolve_source_image_root(sandbox, source_payload)
    if source_root is None:
        return None
    if not authority.authorize(
        session_id,
        generation,
        revision,
        source_root,
    ):
        return None
    return {
        "session_id": session_id,
        "generation": generation,
        "revision": revision,
    }


def authorize_current_revision_candidate(
    source_authority: SourceRevisionAuthority,
    revision_authority: TimelineRevisionAuthority,
    sandbox: SandboxRoot,
    candidate: Mapping[str, object] | None,
    source_payload: object,
) -> dict[str, object] | None:
    """Authorize a grid only while its exact source generation stays current."""
    if not isinstance(candidate, Mapping):
        return None
    session_id = candidate.get("session_id")
    grid_revision = candidate.get("revision")
    browser_generation = candidate.get("generation")
    if (
        not isinstance(session_id, str)
        or not session_id
        or not isinstance(grid_revision, str)
        or not grid_revision
        or type(browser_generation) is not int
    ):
        return None
    source_generation = source_authority.grid_authorization_generation(
        session_id,
        grid_revision,
    )
    if source_generation is None:
        return None
    authorized = authorize_revision_candidate(
        revision_authority,
        sandbox,
        candidate,
        source_payload,
    )
    if authorized is None:
        return None
    if not source_authority.grid_generation_is_current(
        session_id,
        grid_revision,
        source_generation,
    ):
        revision_authority.retire_if_matches(
            session_id,
            browser_generation,
            grid_revision,
        )
        return None
    return authorized


def resolve_popout_event(
    sandbox: SandboxRoot,
    authority: TimelineRevisionAuthority,
    event: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Validate a popout event against live server revision authority.

    Args:
        sandbox: Frozen launch-time filesystem boundary.
        authority: Per-browser current generation authority.
        event: Client event carrying an opaque token and grid revision.

    Returns:
        An approved event for a current-source image, or ``None`` when stale,
        malformed, unavailable, or outside the selected source.
    """
    if not isinstance(event, Mapping):
        return None
    session_id = event.get("session_id")
    generation = event.get("generation")
    revision = event.get("revision")
    token = event.get("token")
    sequence = event.get("sequence")
    if (
        not isinstance(session_id, str)
        or not session_id
        or type(generation) is not int
        or generation < 1
        or not isinstance(revision, str)
        or not revision
        or not isinstance(token, str)
        or not token
        or type(sequence) is not int
        or sequence < 1
    ):
        return None
    current = authority.current(session_id, generation, revision)
    if current is None:
        return None
    try:
        relative_path = _source_render.decode_token(token)
        target = sandbox.resolve(relative_path)
        target.relative_to(current.source_root)
        if not target.is_file():
            return None
        label = target.relative_to(sandbox.root).as_posix()
    except (OSError, RuntimeError, ValueError):
        return None
    return {
        "session_id": session_id,
        "generation": generation,
        "revision": revision,
        "sequence": sequence,
        "token": token,
        "label": label,
    }


# --------------------------------------------------------------------------
# Callback registration
# --------------------------------------------------------------------------
def register_callbacks(
    app: dash.Dash,
    sandbox: SandboxRoot,
    preparation_api: BrowsePreparationApi | None = None,
) -> None:
    """Register every Browse callback on ``app``."""
    revision_authority = TimelineRevisionAuthority()
    source_revision_authority = SourceRevisionAuthority()

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
        Output(ids.BROWSE_TL_ROW_SOURCE, "value"),
        Output(ids.BROWSE_TL_TIME_SOURCE, "value"),
        Output(ids.BROWSE_TL_ROW_CSV_COL, "value"),
        Output(ids.BROWSE_TL_TIME_CSV_COL, "value"),
        Output(ids.BROWSE_TL_CSV_IMAGE_COL, "value", allow_duplicate=True),
        Output(ids.BROWSE_TL_PATTERN_INPUT, "value"),
        Output(ids.BROWSE_TL_PATTERN_ADVANCED, "value"),
        Output(
            ids.BROWSE_TL_PATTERN_PREVIEW,
            "children",
            allow_duplicate=True,
        ),
        Output(
            ids.BROWSE_TL_TILE_SIZE_READOUT,
            "children",
            allow_duplicate=True,
        ),
        Output(
            ids.BROWSE_TL_STORE_TILE_SIZE,
            "data",
            allow_duplicate=True,
        ),
        Output(ids.BROWSE_TL_GRID, "children", allow_duplicate=True),
        Output(
            ids.BROWSE_TL_STORE_WARNINGS,
            "data",
            allow_duplicate=True,
        ),
        Output(
            ids.BROWSE_TL_GRID,
            "data-grid-revision",
            allow_duplicate=True,
        ),
        Output(ids.BROWSE_TL_SOURCE_REVISION, "data"),
        Output(ids.BROWSE_TL_POPOUT_EVENT, "data"),
        Output(
            ids.BROWSE_TL_POPOUT_APPROVED,
            "data",
            allow_duplicate=True,
        ),
        Output(
            ids.BROWSE_TL_POPOUT_MODAL,
            "is_open",
            allow_duplicate=True,
        ),
        Output(
            ids.BROWSE_TL_POPOUT_STORE,
            "data",
            allow_duplicate=True,
        ),
        Output(
            ids.BROWSE_TL_POPOUT_TITLE,
            "children",
            allow_duplicate=True,
        ),
        Input(SHELL_SOURCE_IMAGE_ROOT_STORE, "data"),
        Input(SHELL_CLASSIFIER_CACHE_STORE, "data"),
        State(ids.BROWSE_TL_SESSION, "data"),
        prevent_initial_call=True,
    )
    def _reset_timeline_for_source(
        source_payload: object,
        refresh_revision: object,
        session_id: object,
    ):
        # One callback response retires every source-derived authoring and
        # rendered value. Downstream callbacks may then build a fresh matrix,
        # but no old-source state remains authoritative in the interim.
        if not isinstance(session_id, str) or not session_id:
            raise dash.exceptions.PreventUpdate
        reset_generation = source_revision_authority.begin_reset_and_retire(
            session_id,
            revision_authority,
        )
        values = source_reset_values(source_payload, refresh_revision)
        if not source_revision_authority.publish_reset(
            session_id,
            reset_generation,
            str(values[13]),
        ):
            raise dash.exceptions.PreventUpdate
        return values

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
                    and revisions[name].store_revision is None
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
            status = "ready" if revision.store_revision is not None else "queued"
            if preparation_api is not None:
                if revision.store_revision is None:
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
            filmstrip_assets = source_asset_payload(
                prefix,
                item_token,
                revision.cache_key,
                is_store=revision.store_revision is not None,
            )
            filmstrip.append(
                {
                    "value": name,
                    "label": name,
                    "preview_url": filmstrip_assets["preview_url"],
                    "status": status,
                    "current": name == filename,
                }
            )
        assets = source_asset_payload(
            prefix,
            token,
            selected_revision.cache_key,
            is_store=selected_revision.store_revision is not None,
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
            **assets,
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
            source_image_stem(Path(original)),
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

    # ----------------------------------------------------------------------
    # Timeline view (Phase 2)
    # ----------------------------------------------------------------------
    @app.callback(
        Output(ids.BROWSE_SINGLE_BODY, "style"),
        Output(ids.BROWSE_TIMELINE_BODY, "style"),
        Input(ids.BROWSE_VIEW_MODE_TOGGLE, "value"),
    )
    def _toggle_view_mode(mode: str | None):
        is_timeline = mode == "timeline"
        single_style = (
            {"display": "none"} if is_timeline else {"display": "block"}
        )
        timeline_style = (
            {"display": "block"} if is_timeline else {"display": "none"}
        )
        return single_style, timeline_style

    # Clientside companion: cancel any in-flight warm when leaving Timeline,
    # re-attach the controller (re-render the centered window) when entering it.
    # The body is shown by the server callback above; attach()'s first-paint
    # requestAnimationFrame guard self-corrects if it fires before the show.
    app.clientside_callback(
        """
        function(mode) {
            if (window.__phenotypicTimeline) {
                if (mode === "timeline") {
                    window.__phenotypicTimeline.attach("%s");
                } else if (window.__phenotypicTimeline.cancelWarm) {
                    window.__phenotypicTimeline.cancelWarm();
                }
            }
            return "";
        }
        """
        % ids.BROWSE_TL_GRID,
        Output(ids.BROWSE_TL_GRID, "data-attach-sync"),
        Input(ids.BROWSE_VIEW_MODE_TOGGLE, "value"),
    )

    @app.callback(
        Output(ids.BROWSE_TL_TILE_SIZE_READOUT, "children"),
        Output(ids.BROWSE_TL_STORE_TILE_SIZE, "data"),
        Input(ids.BROWSE_TL_TILE_SIZE_MINUS, "n_clicks"),
        Input(ids.BROWSE_TL_TILE_SIZE_PLUS, "n_clicks"),
        State(ids.BROWSE_TL_STORE_TILE_SIZE, "data"),
        prevent_initial_call=True,
    )
    def _step_tile_size(_minus, _plus, current):
        size = stepped_timeline_tile_size_from_trigger(
            ctx.triggered_id,
            current,
            plus_id=ids.BROWSE_TL_TILE_SIZE_PLUS,
            minus_id=ids.BROWSE_TL_TILE_SIZE_MINUS,
        )
        return f"{size} px", size

    @app.callback(
        Output(ids.BROWSE_TL_NUDGE, "style"),
        Input(SHELL_METADATA_CSV_STORE, "data"),
    )
    def _nudge_visibility(metadata_payload: object):
        # Shown only when no CSV is loaded (richer axes need one).
        has_csv = resolve_metadata_csv(sandbox, metadata_payload) is not None
        return {"display": "none"} if has_csv else {"display": "block"}

    @app.callback(
        Output(ids.BROWSE_TL_ROW_CSV_COL, "options"),
        Output(ids.BROWSE_TL_TIME_CSV_COL, "options"),
        Output(ids.BROWSE_TL_CSV_IMAGE_COL, "options"),
        Output(ids.BROWSE_TL_CSV_IMAGE_COL, "value"),
        Input(SHELL_METADATA_CSV_STORE, "data"),
    )
    def _populate_csv_columns(metadata_payload: object):
        path = resolve_metadata_csv(sandbox, metadata_payload)
        if path is None:
            return [], [], [], None
        try:
            columns, rows = read_metadata_csv_table(path)
        except (OSError, UnicodeError):
            return [], [], [], None
        options, image_default = csv_column_options_and_image_default(
            columns,
            rows,
        )
        return options, options, options, image_default

    @app.callback(
        Output(ids.BROWSE_TL_PATTERN_PREVIEW, "children"),
        Input(ids.BROWSE_TL_PATTERN_INPUT, "value"),
        Input(ids.BROWSE_TL_PATTERN_ADVANCED, "value"),
        State(ids.BROWSE_DATASETS_STORE, "data"),
    )
    def _pattern_preview(
        pattern: str | None, advanced_value, datasets: dict | None
    ):
        advanced = bool(advanced_value) and "advanced" in advanced_value
        return pattern_preview_rows(datasets or {}, pattern or "", advanced)

    @app.callback(
        Output(ids.BROWSE_TL_GRID, "children"),
        Output(ids.BROWSE_TL_STORE_WARNINGS, "data"),
        Output(ids.BROWSE_TL_GRID, "data-grid-revision"),
        Input(ids.BROWSE_VIEW_MODE_TOGGLE, "value"),
        Input(ids.BROWSE_TL_ROW_SOURCE, "value"),
        Input(ids.BROWSE_TL_TIME_SOURCE, "value"),
        Input(ids.BROWSE_TL_ROW_CSV_COL, "value"),
        Input(ids.BROWSE_TL_TIME_CSV_COL, "value"),
        Input(ids.BROWSE_TL_CSV_IMAGE_COL, "value"),
        Input(ids.BROWSE_TL_PATTERN_INPUT, "value"),
        Input(ids.BROWSE_TL_PATTERN_ADVANCED, "value"),
        Input(ids.BROWSE_TL_STORE_TILE_SIZE, "data"),
        Input(ids.BROWSE_TL_SOURCE_REVISION, "data"),
        Input(SHELL_METADATA_CSV_STORE, "data"),
        State(SHELL_SOURCE_IMAGE_ROOT_STORE, "data"),
        State(ids.BROWSE_TL_SESSION, "data"),
    )
    def _render_grid(
        mode: str | None,
        row_source: str | None,
        time_source: str | None,
        row_csv_col: str | None,
        time_csv_col: str | None,
        csv_image_col: str | None,
        pattern: str | None,
        advanced_value,
        tile_size,
        source_revision: str | None,
        metadata_payload: object,
        source_payload: object,
        session_id: object,
    ):
        if mode != "timeline":
            raise dash.exceptions.PreventUpdate
        if not isinstance(session_id, str) or not session_id:
            raise dash.exceptions.PreventUpdate
        source_revision_authority.ensure_session(session_id)
        if not source_revision_authority.is_current(
            session_id,
            source_revision,
        ):
            raise dash.exceptions.PreventUpdate
        revision = timeline_revision_token(
            source_payload,
            metadata_payload,
            row_source,
            time_source,
            row_csv_col,
            time_csv_col,
            csv_image_col,
            pattern,
            advanced_value,
            tile_size,
            source_revision,
        )
        resolved = resolve_source_image_root(sandbox, source_payload)
        src_root_rel = _src_root_rel(sandbox, source_payload)
        if resolved is None or src_root_rel is None:
            return (
                html.Div("No current source.", className="text-muted"),
                [],
                revision,
            )
        datasets = _source_lister.list_datasets(resolved)

        csv_rows: list[dict[str, str]] | None = None
        if "csv" in (row_source, time_source):
            csv_path = resolve_metadata_csv(sandbox, metadata_payload)
            if csv_path is not None:
                try:
                    _columns, csv_rows = read_metadata_csv_table(csv_path)
                except OSError:
                    csv_rows = None

        config = BrowseAxisConfig(
            row_source=row_source or "folder",
            time_source=time_source or "exif",
            pattern=pattern or "",
            advanced_pattern=bool(advanced_value)
            and "advanced" in advanced_value,
            csv_image_col=csv_image_col,
            row_csv_col=row_csv_col,
            time_csv_col=time_csv_col,
        )

        def _capture_time_of(rel: str) -> str | None:
            try:
                return read_capture_time(sandbox.resolve(rel))
            except (OSError, RuntimeError, ValueError):
                return None

        records, warnings = build_browse_records(
            datasets,
            src_root_rel,
            config,
            csv_rows=csv_rows,
            capture_time_of=_capture_time_of,
        )
        display_size = int(tile_size or TIMELINE_TILE_SIZE_DEFAULT)
        prefix = app.server.config.get(CFG_URL_PREFIX, MOUNT_HOME)
        component = render_timeline_grid(
            records, display_size=display_size, prefix=prefix
        )
        if not source_revision_authority.authorize_grid(
            session_id,
            source_revision,
            revision,
        ):
            raise dash.exceptions.PreventUpdate
        return component, warnings, revision

    @app.callback(
        Output(ids.BROWSE_TL_WARNINGS_ALERT, "children"),
        Output(ids.BROWSE_TL_WARNINGS_ALERT, "is_open"),
        Input(ids.BROWSE_TL_STORE_WARNINGS, "data"),
    )
    def _surface_warnings(warnings: list[str] | None):
        # Surface the CSV-join warnings the render callback wrote to the
        # otherwise-dead warnings store; hidden when the list is empty.
        return warnings_alert_state(warnings)

    # Each browser gets a session-scoped identity. It keys the server-side
    # revision authority without coupling independent tabs/users.
    app.clientside_callback(
        """
        function(children, current) {
            if (current) {
                return current;
            }
            if (window.crypto && window.crypto.randomUUID) {
                return window.crypto.randomUUID();
            }
            return "tl-" + Date.now().toString(36)
                + "-" + Math.random().toString(36).slice(2);
        }
        """,
        Output(ids.BROWSE_TL_SESSION, "data"),
        Input(ids.BROWSE_TL_GRID, "children"),
        State(ids.BROWSE_TL_SESSION, "data"),
    )

    # Re-attach after each render, retire all client-owned state, and publish
    # the generation that the browser actually applied for server authority.
    app.clientside_callback(
        """
        function(children, sessionId, gridRevision) {
            let generation = 0;
            if (window.__phenotypicBrowse
                && window.__phenotypicBrowse.resetTimelineRevision) {
                generation = window.__phenotypicBrowse.resetTimelineRevision("%s");
            }
            if (window.__phenotypicTimeline) {
                window.__phenotypicTimeline.attach("%s");
            }
            const grid = document.getElementById("%s");
            const revision = gridRevision
                || (grid && grid.getAttribute("data-grid-revision"));
            const noUpdate = window.dash_clientside.no_update;
            if (!sessionId || !revision || !generation) {
                return ["", String(generation || ""), noUpdate];
            }
            return [
                "",
                String(generation),
                {
                    session_id: sessionId,
                    generation: generation,
                    revision: revision,
                },
            ];
        }
        """
        % (ids.BROWSE_TL_GRID, ids.BROWSE_TL_GRID, ids.BROWSE_TL_GRID),
        Output(ids.BROWSE_TL_GRID, "data-render-sync"),
        Output(ids.BROWSE_TL_GRID, "data-revision-generation"),
        Output(ids.BROWSE_TL_REVISION_CANDIDATE, "data"),
        Input(ids.BROWSE_TL_GRID, "children"),
        Input(ids.BROWSE_TL_SESSION, "data"),
        Input(ids.BROWSE_TL_GRID, "data-grid-revision"),
    )

    @app.callback(
        Output(ids.BROWSE_TL_REVISION_AUTHORIZED, "data"),
        Input(ids.BROWSE_TL_REVISION_CANDIDATE, "data"),
        State(SHELL_SOURCE_IMAGE_ROOT_STORE, "data"),
    )
    def _authorize_revision(
        candidate: Mapping[str, object] | None,
        source_payload: object,
    ):
        authorized = authorize_current_revision_candidate(
            source_revision_authority,
            revision_authority,
            sandbox,
            candidate,
            source_payload,
        )
        if authorized is None:
            raise dash.exceptions.PreventUpdate
        return authorized

    app.clientside_callback(
        """
        function(authorized, generation, sessionId) {
            const noUpdate = window.dash_clientside.no_update;
            if (!authorized || !sessionId
                || authorized.session_id !== sessionId
                || String(authorized.generation) !== String(generation)) {
                return [noUpdate, sessionId || ""];
            }
            return [authorized.revision || "", sessionId];
        }
        """,
        Output(ids.BROWSE_TL_GRID, "data-authorized-revision"),
        Output(ids.BROWSE_TL_GRID, "data-session-id"),
        Input(ids.BROWSE_TL_REVISION_AUTHORIZED, "data"),
        Input(ids.BROWSE_TL_GRID, "data-revision-generation"),
        State(ids.BROWSE_TL_SESSION, "data"),
    )

    # ----------------------------------------------------------------------
    # Single-image deep-zoom pop-out (Task 9)
    # ----------------------------------------------------------------------
    # browse.js publishes a revision-stamped event through Dash's supported
    # set_props API. The event remains connected after source/metadata/grid
    # revisions and wholesale DOM remounts because its listener is delegated
    # on document rather than attached to one React-controlled input node.
    @app.callback(
        Output(ids.BROWSE_TL_POPOUT_APPROVED, "data"),
        Input(ids.BROWSE_TL_POPOUT_EVENT, "data"),
    )
    def _approve_popout(
        event: Mapping[str, object] | None,
    ):
        payload = resolve_popout_event(
            sandbox,
            revision_authority,
            event,
        )
        if payload is None:
            raise dash.exceptions.PreventUpdate
        return payload

    # Final publication is gated against the live browser generation and
    # latest event. A delayed server response for revision A therefore cannot
    # reopen or overwrite revision B even if request A began before B existed.
    app.clientside_callback(
        """
        function(approved, event) {
            const noUpdate = window.dash_clientside.no_update;
            const grid = document.getElementById("%s");
            if (!approved || !event || !grid) {
                return [noUpdate, noUpdate, noUpdate];
            }
            const revision = grid.getAttribute("data-grid-revision");
            const generation = grid.getAttribute("data-revision-generation");
            const sessionId = grid.getAttribute("data-session-id");
            const authorized = grid.getAttribute("data-authorized-revision");
            if (approved.revision !== revision
                || approved.revision !== authorized
                || String(approved.generation) !== String(generation)
                || approved.session_id !== sessionId
                || approved.sequence !== event.sequence
                || approved.token !== event.token) {
                return [noUpdate, noUpdate, noUpdate];
            }
            const payload = {token: approved.token, label: approved.label};
            return [true, payload, approved.label];
        }
        """
        % ids.BROWSE_TL_GRID,
        Output(ids.BROWSE_TL_POPOUT_MODAL, "is_open"),
        Output(ids.BROWSE_TL_POPOUT_STORE, "data"),
        Output(ids.BROWSE_TL_POPOUT_TITLE, "children"),
        Input(ids.BROWSE_TL_POPOUT_APPROVED, "data"),
        Input(ids.BROWSE_TL_POPOUT_EVENT, "data"),
    )

    # Clientside: mount the deep-zoom OSD viewer into the pop-out modal's
    # dedicated OSD div. Dash requires every clientside callback to declare an
    # Output sink (applyPopoutImage returns nothing useful), so mirror the
    # existing BROWSE_OSD_SYNC idiom and write a throwaway synthetic data-attr
    # on the pop-out OSD div — it never disturbs the OSD canvas children.
    app.clientside_callback(
        """
        function(payload) {
            if (window.__phenotypicBrowse
                && window.__phenotypicBrowse.applyPopoutImage) {
                window.__phenotypicBrowse.applyPopoutImage(payload);
            }
            return "";
        }
        """,
        Output(ids.BROWSE_TL_POPOUT_OSD, "data-popout-sync"),
        Input(ids.BROWSE_TL_POPOUT_STORE, "data"),
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
