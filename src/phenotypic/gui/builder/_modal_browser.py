"""Layout-only :class:`dbc.Modal` factories for the pipeline builder's file flows.

This module provides three modal factories used by the Dash pipeline builder
to replace typed-path inputs with point-and-click file browsers:

* :func:`save_pipeline_modal` — folder browser + filename input for writing
  ``ImagePipeline.to_json()`` output to disk.
* :func:`load_picker_modal` — two-stage chooser letting the user browse for a
  saved JSON pipeline file or pick a built-in prefab from
  :mod:`phenotypic.prefab`.
* :func:`load_image_modal` — image-file browser filtered to common plate image
  formats (``.tif``, ``.nef``, ``.cr2``, etc.), with a "Use synthetic plate"
  shortcut in the footer for quick testing.

The module is layout-only: it builds static :class:`dbc.Modal` trees using
:func:`~phenotypic.gui.builder._directory_browser.directory_tree` and the id
constants defined in :mod:`~phenotypic.gui.builder._ids`. Callbacks
(open/close, page swaps, tree re-renders, save/load actions) are wired up in
:mod:`~phenotypic.gui.builder._callbacks`; nothing here registers callbacks or
imports the builder state model.

All modals are rendered with ``is_open=False`` so the layout module can mount
them once at app start. The callback layer flips :attr:`is_open` in response
to the matching trigger-button click (e.g. :data:`~_ids.BTN_SAVE`,
:data:`~_ids.BTN_LOAD`, :data:`~_ids.BTN_LOAD_IMAGE`).
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html

from . import _ids as ids
from ._directory_browser import IMAGE_EXTS, PIPELINE_EXTS, directory_tree

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _no_root_placeholder() -> html.Div:
    """Return an italic muted ``Div`` shown when no working directory is set."""
    return html.Div(
        "(no working directory configured)",
        className="text-muted small fst-italic",
    )


def _first_doc_line(cls: object) -> str:
    """Return the first non-blank stripped line of ``cls.__doc__``, or ``""``."""
    doc = getattr(cls, "__doc__", None) or ""
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _chooser_buttons() -> html.Div:
    """Render the JSON / Prefab chooser-button stack used by the load picker.

    Returned as a single :class:`html.Div` (id
    ``"load-picker-chooser-buttons"``) so a callback can toggle its
    ``style.display`` to hide the chooser when the user navigates to the JSON
    or Prefab subpages. The buttons stay in the DOM on every page so
    ``swap_load_picker_page``'s ``Input`` subscriptions to
    :data:`ids.BTN_LOAD_JSON_CHOICE` and :data:`ids.BTN_LOAD_PREFAB_CHOICE`
    always resolve.
    """
    json_card = html.Div(
        [
            dbc.Button(
                "JSON",
                id=ids.BTN_LOAD_JSON_CHOICE,
                color="primary",
                size="lg",
                n_clicks=0,
                className="w-100",
            ),
            html.Div(
                "Browse for a pipeline file saved with"
                " ImagePipeline.to_json().",
                className="text-muted small mt-1",
            ),
        ],
        className="mb-3",
    )

    prefab_card = html.Div(
        [
            dbc.Button(
                "Prefab",
                id=ids.BTN_LOAD_PREFAB_CHOICE,
                color="primary",
                size="lg",
                n_clicks=0,
                className="w-100",
            ),
            html.Div(
                "Pick a built-in pipeline shipped under phenotypic.prefab.",
                className="text-muted small mt-1",
            ),
        ],
    )

    return html.Div(
        [json_card, prefab_card],
        id="load-picker-chooser-buttons",
    )


def _back_button() -> dbc.Button:
    """Render the load-picker "← Back" button (always present, visibility toggled)."""
    return dbc.Button(
        "← Back",
        id=ids.BTN_LOAD_PICKER_BACK,
        color="secondary",
        outline=True,
        size="sm",
        n_clicks=0,
        className="mb-2",
        style={"display": "none"},
    )


def _json_page(root: Path | None) -> List:
    """Build the JSON-browser body: :func:`directory_tree` filtered to ``*.json``."""
    if root is None:
        return [_no_root_placeholder()]
    return [
        directory_tree(
            root,
            current=None,
            extensions=PIPELINE_EXTS,
            select_files=True,
            id_type=ids.DIR_ENTRY_TYPE_JSON,
        ),
    ]


def _prefab_page() -> List:
    """Build the prefab-list body: one clickable card per entry in ``phenotypic.prefab.__all__``."""
    try:
        import phenotypic.prefab as _prefab
    except Exception:
        return [
            html.Div(
                "(prefab module unavailable)",
                className="text-muted small fst-italic",
            ),
        ]

    items: List = []
    for name in getattr(_prefab, "__all__", []):
        cls = getattr(_prefab, name, None)
        if cls is None:
            continue
        items.append(
            dbc.ListGroupItem(
                [
                    html.Strong(name),
                    html.Div(
                        _first_doc_line(cls),
                        className="text-muted small",
                    ),
                ],
                id=ids.prefab_card_id(name),
                action=True,
                n_clicks=0,
            )
        )

    if not items:
        return [
            html.Div(
                "(no prefabs registered)",
                className="text-muted small fst-italic",
            ),
        ]

    return [dbc.ListGroup(items, flush=True)]


def render_load_picker_body(page: str, root: Path | None) -> List:
    """Dispatch to the correct subpage builder for :data:`ids.MODAL_LOAD_PICKER_BODY`.

    Called by the callback layer whenever :data:`ids.STORE_LOAD_PICKER_PAGE`
    changes. Returns the component list that replaces the body container's
    ``children``. Unknown ``page`` values fall back to the chooser so a
    corrupt store value cannot blank the modal.

    Args:
        page: Active page token. One of ``"chooser"``, ``"json"``,
            ``"prefab"``. Any other value renders the chooser page.
        root: Configured working directory. Only consulted for the JSON page,
            where it is passed to :func:`directory_tree` as the security
            boundary. The chooser and prefab pages ignore this argument.

    Returns:
        A list of Dash components ready to assign to
        :data:`ids.MODAL_LOAD_PICKER_BODY` ``children``.
    """
    if page == "json":
        return _json_page(root)
    if page == "prefab":
        return _prefab_page()
    # Chooser page: body is empty — the chooser buttons live as siblings of
    # the body container, toggled via style by the callback layer.
    return []


# ---------------------------------------------------------------------------
# Public modal factories
# ---------------------------------------------------------------------------

def save_pipeline_modal(root: Path | None) -> dbc.Modal:
    """Build the "Save pipeline" modal (id :data:`ids.MODAL_SAVE`).

    Renders a folder-only directory tree so the user navigates to a target
    directory, then types a filename in the :data:`ids.INPUT_SAVE_FILENAME`
    footer input. The confirm callback joins the selected directory from
    :data:`ids.STORE_BROWSE_DIR_SAVE` with the filename and writes
    ``ImagePipeline.to_json()`` output to that path.

    Files are not selectable in the tree (``select_files=False``): the user
    provides the filename explicitly so overwriting an existing pipeline is a
    deliberate choice, not an accidental click.

    Args:
        root: Configured working directory used as the security boundary for
            the folder tree. When ``None``, the body shows a muted
            "(no working directory configured)" placeholder instead of a tree;
            the footer input is still rendered.

    Returns:
        A :class:`dbc.Modal` with ``id=ids.MODAL_SAVE``,
        ``is_open=False``, ``size="lg"``, ``scrollable=True``,
        ``backdrop=True``, and ``centered=True``, ready to be mounted once
        at app start by :func:`~_layout.build_app_layout`.
    """
    if root is None:
        body_children: List = [_no_root_placeholder()]
    else:
        body_children = [
            directory_tree(
                root,
                current=None,
                extensions=None,
                select_files=False,
                id_type=ids.DIR_ENTRY_TYPE_SAVE,
            )
        ]

    footer = dbc.ModalFooter(
        [
            dbc.Input(
                id=ids.INPUT_SAVE_FILENAME,
                type="text",
                value="pipeline.json",
                placeholder="filename.json",
                debounce=True,
                className="w-100 mb-2",
            ),
            dbc.Button(
                "Cancel",
                id=ids.BTN_SAVE_CANCEL,
                color="secondary",
                outline=True,
                n_clicks=0,
            ),
            dbc.Button(
                "Save",
                id=ids.BTN_SAVE_CONFIRM,
                color="primary",
                n_clicks=0,
            ),
        ]
    )

    # ``STORE_BROWSE_DIR_SAVE`` is a sibling of (not inside) ``MODAL_SAVE_BODY``
    # so re-rendering the body doesn't replace the store mid-callback (which
    # would cycle ``render_save_body`` -> store -> body re-render).
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Save pipeline")),
            dbc.ModalBody(
                [
                    dcc.Store(
                        id=ids.STORE_BROWSE_DIR_SAVE,
                        data=str(root) if root else None,
                    ),
                    html.Div(id=ids.MODAL_SAVE_BODY, children=body_children),
                ]
            ),
            footer,
        ],
        id=ids.MODAL_SAVE,
        is_open=False,
        size="lg",
        scrollable=True,
        backdrop=True,
        centered=True,
    )


def load_picker_modal(root: Path | None) -> dbc.Modal:
    """Build the two-stage "Load pipeline" modal (id :data:`ids.MODAL_LOAD_PICKER`).

    The modal opens on a chooser page with two large buttons:

    * **JSON** — navigates to a :func:`directory_tree` filtered to
      :data:`~_directory_browser.PIPELINE_EXTS` (``.json``) so the user can
      locate a pipeline previously saved with ``ImagePipeline.to_json()``.
    * **Prefab** — shows a scrollable list of built-in pipelines from
      :mod:`phenotypic.prefab`; clicking any card instantiates that pipeline
      and replaces the current builder state.

    The active page is tracked in :data:`ids.STORE_LOAD_PICKER_PAGE` and the
    body container :data:`ids.MODAL_LOAD_PICKER_BODY` is re-rendered by the
    callback layer when that store changes. The two :class:`dcc.Store` widgets
    are mounted as siblings of the body container (not inside it) so they
    survive body re-renders without losing their values.

    Args:
        root: Configured working directory. Used as the security boundary for
            the JSON page's directory tree. When ``None``, the JSON page shows
            a muted placeholder; the chooser and prefab pages are unaffected.

    Returns:
        A :class:`dbc.Modal` with ``id=ids.MODAL_LOAD_PICKER``,
        ``is_open=False``, ``size="lg"``, ``scrollable=True``,
        ``backdrop=True``, and ``centered=True``.
    """
    # The Back button and the chooser-buttons stack are permanent siblings
    # of the body so swap_load_picker_page's pattern-matching Inputs always
    # resolve. A separate callback toggles their inline ``style.display``
    # based on STORE_LOAD_PICKER_PAGE.
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Load pipeline")),
            dbc.ModalBody(
                [
                    dcc.Store(
                        id=ids.STORE_LOAD_PICKER_PAGE,
                        data="chooser",
                    ),
                    dcc.Store(
                        id=ids.STORE_BROWSE_DIR_JSON,
                        data=str(root) if root else None,
                    ),
                    _back_button(),
                    _chooser_buttons(),
                    html.Div(
                        id=ids.MODAL_LOAD_PICKER_BODY,
                        children=[],
                    ),
                ]
            ),
        ],
        id=ids.MODAL_LOAD_PICKER,
        is_open=False,
        size="lg",
        scrollable=True,
        backdrop=True,
        centered=True,
    )


def load_image_modal(root: Path | None) -> dbc.Modal:
    """Build the "Load image" modal (id :data:`ids.MODAL_LOAD_IMAGE`).

    Renders a :func:`directory_tree` filtered to :data:`~_directory_browser.IMAGE_EXTS`
    (``.tif``, ``.tiff``, ``.nef``, ``.cr2``, ``.arw``, ``.dng``, and common
    lossy formats) so the user can select a plate image captured by a DSLR or
    flatbed scanner. The chosen path is written to :data:`ids.STORE_IMAGE_PATH`
    and consumed by the "Run preview" callback.

    A "Use synthetic plate" button in the modal footer (id
    :data:`ids.BTN_USE_SYNTHETIC_MODAL`) lets the user skip the browser and
    load :func:`phenotypic.data.load_synth_yeast_plate` without leaving the
    modal. It shares a callback handler with the equivalent top-level button
    (id :data:`ids.BTN_USE_SYNTHETIC`) in the footer bar.

    Args:
        root: Configured image-root directory used as the security boundary
            for the tree. When ``None``, the body shows a muted
            "(no working directory configured)" placeholder; the synthetic
            plate shortcut remains fully functional.

    Returns:
        A :class:`dbc.Modal` with ``id=ids.MODAL_LOAD_IMAGE``,
        ``is_open=False``, ``size="lg"``, ``scrollable=True``,
        ``backdrop=True``, and ``centered=True``.
    """
    if root is None:
        body_children: List = [_no_root_placeholder()]
    else:
        body_children = [
            directory_tree(
                root,
                current=None,
                extensions=IMAGE_EXTS,
                select_files=True,
                id_type=ids.DIR_ENTRY_TYPE_IMAGE,
            )
        ]

    footer = dbc.ModalFooter(
        dbc.Button(
            "Use synthetic plate",
            id=ids.BTN_USE_SYNTHETIC_MODAL,
            color="secondary",
            outline=True,
            n_clicks=0,
        )
    )

    # ``STORE_BROWSE_DIR_IMAGE`` is a sibling of (not inside) ``MODAL_LOAD_IMAGE_BODY``
    # so re-rendering the body doesn't replace the store mid-callback.
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Load image")),
            dbc.ModalBody(
                [
                    dcc.Store(
                        id=ids.STORE_BROWSE_DIR_IMAGE,
                        data=str(root) if root else None,
                    ),
                    html.Div(id=ids.MODAL_LOAD_IMAGE_BODY, children=body_children),
                ]
            ),
            footer,
        ],
        id=ids.MODAL_LOAD_IMAGE,
        is_open=False,
        size="lg",
        scrollable=True,
        backdrop=True,
        centered=True,
    )


__all__ = [
    "save_pipeline_modal",
    "load_picker_modal",
    "load_image_modal",
    "render_load_picker_body",
]
