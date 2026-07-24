"""Run console form — pickers + mode + advanced + slurm config.

Builds the left-column form for the Run console: three picker buttons
(pipeline config, input directory, output directory), a Local/SLURM mode
toggle, inline ``Dry-run`` / ``Resume`` checkboxes, an Advanced collapse
(``--sample``, ``--nrows``, ``--ncols``,
``--image-type``, ``--workers``, ``--log-level``), and a SLURM config
collapse (typed common fields plus a free-form ``k=v`` textarea).

Three :class:`dbc.Modal` factories are also defined here:

    * :func:`build_pipeline_picker_modal` — wraps
      :func:`~phenotypic.gui.builder._directory_browser.directory_tree`
      filtered to ``.json`` files.
    * :func:`build_input_picker_modal` — wraps the same with
      ``select_files=False`` so the user picks a directory containing
      images.
    * The output picker lives in :mod:`._directory_picker` because it
      needs different "may-not-yet-exist" semantics.

Layout-only: callbacks are wired by :mod:`._callbacks`.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html

from phenotypic.gui._config import IMAGE_EXTS
from phenotypic.gui._design import FONT_FAMILY_MONO
from phenotypic.gui.builder._directory_browser import (
    PIPELINE_EXTS,
    directory_tree,
)
from phenotypic.gui.run_console import _ids as ids
from phenotypic.gui.run_console._directory_picker import (
    build_output_picker_modal,
)
from phenotypic.gui.shell._sandbox import SandboxRoot

__all__ = [
    "build_form",
    "build_pipeline_picker_modal",
    "build_input_picker_modal",
    "build_output_picker_modal",
    "render_pipeline_tree",
    "render_input_tree",
]


# ---------------------------------------------------------------------------
# Tree-render helpers (used by callbacks on browse-dir change)
# ---------------------------------------------------------------------------


def render_pipeline_tree(
    sandbox: SandboxRoot, current: Path | None = None
) -> html.Div:
    """Render the pipeline-JSON tree for the currently-browsed directory.

    Reuses the builder's :func:`directory_tree` (filtered to ``.json``)
    with the Run console's distinct ``id_type`` so callbacks here do not
    collide with the builder's load-picker callbacks.
    """
    return directory_tree(
        sandbox.root,
        current=current,
        extensions=PIPELINE_EXTS,
        select_files=True,
        id_type=ids.RC_DIR_ENTRY_TYPE_PIPELINE_JSON,
    )


def render_input_tree(
    sandbox: SandboxRoot, current: Path | None = None
) -> html.Div:
    """Render the input-directory tree (no file selection).

    Folders only. ``IMAGE_EXTS`` is still passed as the shared discovery
    contract even though ``directory_tree`` ignores it while
    ``select_files=False``.
    """
    return directory_tree(
        sandbox.root,
        current=current,
        extensions=IMAGE_EXTS,
        select_files=False,
        id_type=ids.RC_DIR_ENTRY_TYPE_INPUT_DIR,
    )


# ---------------------------------------------------------------------------
# Modal factories
# ---------------------------------------------------------------------------


def build_pipeline_picker_modal(sandbox: SandboxRoot) -> dbc.Modal:
    """Build the pipeline-JSON picker modal.

    The user navigates the sandbox tree filtered to ``*.json``. Confirming
    a file copies its path into :data:`ids.RC_STORE_PIPELINE_PATH`. The
    confirm callback also performs a cheap "looks like a pipeline" check
    by reading the first 4 KB of the file and looking for an
    ``"operations"`` token; missing that, it surfaces a warning toast but
    still allows the selection.

    Args:
        sandbox: Containment primitive — the tree never escapes
            ``sandbox.root``.

    Returns:
        A :class:`dbc.Modal` ready to mount once at app start.
    """
    body = html.Div(
        [
            dcc.Store(
                id=ids.RC_STORE_BROWSE_DIR_PIPELINE,
                data=str(sandbox.root),
            ),
            html.Div(
                id=ids.RC_MODAL_PIPELINE_BODY,
                children=[render_pipeline_tree(sandbox)],
            ),
        ]
    )

    footer = dbc.ModalFooter(
        [
            dbc.Button(
                "Cancel",
                id=ids.RC_BTN_PIPELINE_CANCEL,
                color="secondary",
                outline=True,
                n_clicks=0,
            ),
        ]
    )
    # The Confirm button is implicit — clicking a file in the tree
    # selects it. We retain a hidden ``RC_BTN_PIPELINE_CONFIRM`` so the
    # callback wiring is consistent across modals.
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Pick pipeline config")),
            dbc.ModalBody(body),
            footer,
            html.Div(
                dbc.Button(id=ids.RC_BTN_PIPELINE_CONFIRM, n_clicks=0),
                style={"display": "none"},
            ),
        ],
        id=ids.RC_MODAL_PIPELINE,
        is_open=False,
        size="lg",
        scrollable=True,
        backdrop=True,
        centered=True,
    )


def build_input_picker_modal(sandbox: SandboxRoot) -> dbc.Modal:
    """Build the input-directory picker modal.

    Folder-only tree. The selected directory is committed via the
    "Use this directory" footer button (the user must explicitly confirm
    so accidental clicks while exploring don't pin the wrong directory).

    Args:
        sandbox: Containment primitive.

    Returns:
        A :class:`dbc.Modal` ready to mount once at app start.
    """
    body = html.Div(
        [
            dcc.Store(
                id=ids.RC_STORE_BROWSE_DIR_INPUT,
                data=str(sandbox.root),
            ),
            html.Div(
                id=ids.RC_MODAL_INPUT_BODY,
                children=[render_input_tree(sandbox)],
            ),
        ]
    )

    footer = dbc.ModalFooter(
        [
            dbc.Button(
                "Cancel",
                id=ids.RC_BTN_INPUT_CANCEL,
                color="secondary",
                outline=True,
                n_clicks=0,
            ),
            dbc.Button(
                "Use this directory",
                id=ids.RC_BTN_INPUT_CONFIRM,
                color="primary",
                n_clicks=0,
            ),
        ]
    )
    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Pick input directory")),
            dbc.ModalBody(body),
            footer,
        ],
        id=ids.RC_MODAL_INPUT,
        is_open=False,
        size="lg",
        scrollable=True,
        backdrop=True,
        centered=True,
    )


# ---------------------------------------------------------------------------
# Form sub-sections
# ---------------------------------------------------------------------------


def _picker_row(
    *,
    label: str,
    button_id: str,
    label_id: str,
    placeholder: str,
) -> html.Div:
    """Build one picker row: title + button + selected-path label."""
    return html.Div(
        [
            html.Div(label, className="run-console-picker-label"),
            html.Div(
                [
                    dbc.Button(
                        "Browse...",
                        id=button_id,
                        color="primary",
                        outline=True,
                        size="sm",
                        n_clicks=0,
                    ),
                    html.Span(
                        placeholder,
                        id=label_id,
                        className="run-console-picker-value",
                    ),
                ],
                className="run-console-picker-row",
            ),
        ],
        className="run-console-picker-section",
    )


def _build_advanced_section() -> html.Div:
    """Build the Advanced collapse body (six form fields)."""
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label(
                                "Sample",
                                html_for=ids.RC_INPUT_SAMPLE,
                                className="run-console-form-label",
                            ),
                            dbc.Input(
                                id=ids.RC_INPUT_SAMPLE,
                                type="number",
                                min=1,
                                step=1,
                                placeholder="all",
                                debounce=True,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label(
                                "Workers",
                                html_for=ids.RC_INPUT_WORKERS,
                                className="run-console-form-label",
                            ),
                            dbc.Input(
                                id=ids.RC_INPUT_WORKERS,
                                type="number",
                                min=1,
                                step=1,
                                placeholder="auto",
                                debounce=True,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label(
                                "Log level",
                                html_for=ids.RC_INPUT_LOG_LEVEL,
                                className="run-console-form-label",
                            ),
                            dcc.Dropdown(
                                id=ids.RC_INPUT_LOG_LEVEL,
                                options=[
                                    {"label": "DEBUG", "value": "DEBUG"},
                                    {"label": "INFO", "value": "INFO"},
                                    {"label": "WARNING", "value": "WARNING"},
                                    {"label": "ERROR", "value": "ERROR"},
                                ],
                                placeholder="INFO",
                                clearable=True,
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="g-2 mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label(
                                "nrows",
                                html_for=ids.RC_INPUT_NROWS,
                                className="run-console-form-label",
                            ),
                            dbc.Input(
                                id=ids.RC_INPUT_NROWS,
                                type="number",
                                min=1,
                                step=1,
                                placeholder="auto",
                                debounce=True,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label(
                                "ncols",
                                html_for=ids.RC_INPUT_NCOLS,
                                className="run-console-form-label",
                            ),
                            dbc.Input(
                                id=ids.RC_INPUT_NCOLS,
                                type="number",
                                min=1,
                                step=1,
                                placeholder="auto",
                                debounce=True,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label(
                                "Image type",
                                html_for=ids.RC_INPUT_IMAGE_TYPE,
                                className="run-console-form-label",
                            ),
                            dcc.Dropdown(
                                id=ids.RC_INPUT_IMAGE_TYPE,
                                options=[
                                    {
                                        "label": "GridImage",
                                        "value": "GridImage",
                                    },
                                    {"label": "Image", "value": "Image"},
                                ],
                                placeholder="GridImage (default)",
                                clearable=True,
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="g-2",
            ),
        ]
    )


def _build_slurm_section() -> html.Div:
    """Build the common CPU profile and staged-GPU controls."""
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label(
                                "Partition",
                                html_for=ids.RC_INPUT_SLURM_PARTITION,
                                className="run-console-form-label",
                            ),
                            dbc.Input(
                                id=ids.RC_INPUT_SLURM_PARTITION,
                                type="text",
                                placeholder="general",
                                debounce=True,
                            ),
                        ],
                        md=6,
                    ),
                    dbc.Col(
                        [
                            dbc.Label(
                                "Time limit",
                                html_for=ids.RC_INPUT_SLURM_TIME,
                                className="run-console-form-label",
                            ),
                            dbc.Input(
                                id=ids.RC_INPUT_SLURM_TIME,
                                type="text",
                                placeholder="04:00:00",
                                debounce=True,
                            ),
                            dbc.FormText(
                                "Minutes or SLURM duration "
                                "(HH:MM:SS, D-HH:MM:SS)"
                            ),
                        ],
                        md=6,
                    ),
                ],
                className="g-2 mb-2",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label(
                                "Memory",
                                html_for=ids.RC_INPUT_SLURM_MEM,
                                className="run-console-form-label",
                            ),
                            dbc.Input(
                                id=ids.RC_INPUT_SLURM_MEM,
                                type="text",
                                placeholder="16G",
                                debounce=True,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label(
                                "CPUs / task",
                                html_for=ids.RC_INPUT_SLURM_CPUS,
                                className="run-console-form-label",
                            ),
                            dbc.Input(
                                id=ids.RC_INPUT_SLURM_CPUS,
                                type="number",
                                min=1,
                                step=1,
                                placeholder="4",
                                debounce=True,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label(
                                "CPU-stage GPUs",
                                html_for=ids.RC_INPUT_SLURM_GPUS,
                                className="run-console-form-label",
                            ),
                            dbc.Input(
                                id=ids.RC_INPUT_SLURM_GPUS,
                                type="number",
                                min=0,
                                step=1,
                                placeholder="0",
                                debounce=True,
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="g-2 mb-2",
            ),
            html.Div(
                [
                    dbc.Label(
                        "Extra SLURM (one ``key=value`` per line)",
                        html_for=ids.RC_INPUT_SLURM_EXTRA,
                        className="run-console-form-label",
                    ),
                    dbc.Textarea(
                        id=ids.RC_INPUT_SLURM_EXTRA,
                        placeholder="account=lab\nqos=normal",
                        rows=4,
                        style={"fontFamily": FONT_FAMILY_MONO},
                    ),
                ]
            ),
            html.Div(
                [
                    html.Hr(),
                    html.Div(
                        "Staged GPU detection",
                        className="run-console-picker-label",
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "GPU-stage delta profile",
                                        html_for=ids.RC_INPUT_GPU_SLURM,
                                        className="run-console-form-label",
                                    ),
                                    dbc.Textarea(
                                        id=ids.RC_INPUT_GPU_SLURM,
                                        placeholder=(
                                            "slurm_partition=gpu\n"
                                            "slurm_account=lab"
                                        ),
                                        rows=4,
                                        style={
                                            "fontFamily": FONT_FAMILY_MONO
                                        },
                                    ),
                                ],
                                md=8,
                            ),
                            dbc.Col(
                                [
                                    dbc.Label(
                                        "GPU shards",
                                        html_for=ids.RC_INPUT_GPU_SHARDS,
                                        className="run-console-form-label",
                                    ),
                                    dbc.Input(
                                        id=ids.RC_INPUT_GPU_SHARDS,
                                        type="number",
                                        min=1,
                                        step=1,
                                        value=1,
                                        debounce=True,
                                    ),
                                    html.Small(
                                        "One resident model per whole-GPU shard.",
                                        className="text-muted",
                                    ),
                                ],
                                md=4,
                            ),
                        ],
                        className="g-2",
                    ),
                ],
                id=ids.RC_STAGED_GPU_SECTION,
                style={"display": "none"},
            ),
        ]
    )


def _build_action_buttons() -> html.Div:
    """Build the action-button row (validate / run / cancel / save / load)."""
    return html.Div(
        [
            dbc.Button(
                "Validate (dry-run)",
                id=ids.RC_BTN_VALIDATE,
                color="secondary",
                outline=True,
                n_clicks=0,
                className="me-2",
            ),
            dbc.Button(
                "Run",
                id=ids.RC_BTN_RUN,
                color="primary",
                n_clicks=0,
                className="me-2",
            ),
            dbc.Button(
                "Cancel",
                id=ids.RC_BTN_CANCEL,
                color="danger",
                outline=True,
                n_clicks=0,
                disabled=True,
                className="me-2",
            ),
            html.Div(style={"flex": "1 1 auto"}),
            dbc.Input(
                id=ids.RC_INPUT_PRESET_NAME,
                type="text",
                placeholder="preset name",
                size="sm",
                debounce=True,
                style={"maxWidth": "180px"},
                className="me-2",
            ),
            dbc.Button(
                "Save preset",
                id=ids.RC_BTN_SAVE_PRESET,
                color="secondary",
                outline=True,
                size="sm",
                n_clicks=0,
                className="me-2",
            ),
            dcc.Dropdown(
                id=ids.RC_DROPDOWN_LOAD_PRESET,
                options=[],
                placeholder="Load preset...",
                clearable=True,
                style={"minWidth": "200px"},
            ),
        ],
        className="run-console-actions",
    )


# ---------------------------------------------------------------------------
# Public form builder
# ---------------------------------------------------------------------------


def build_form(sandbox: SandboxRoot) -> html.Div:
    """Build the Run console form column.

    Sections (top to bottom):

        1. Three picker rows (pipeline / input / output) — buttons that
           open modal pickers; the selected path is shown in monospace
           next to each button.
        2. Mode toggle (Local / SLURM radio).
        3. Inline checkboxes (``Dry-run``, ``Resume``).
        4. Advanced collapse (default closed).
        5. SLURM-config collapse (default closed).
        6. Action buttons row (``Validate``, ``Run``, ``Cancel``,
           preset name input, ``Save preset``, ``Load preset`` dropdown).

    Args:
        sandbox: Containment primitive used to seed picker modals.

    Returns:
        A :class:`dash.html.Div` with class ``run-console-form-col``.
    """
    pickers = html.Div(
        [
            _picker_row(
                label="Pipeline config file",
                button_id=ids.RC_BTN_PICK_PIPELINE,
                label_id=ids.RC_LABEL_PIPELINE,
                placeholder="(no pipeline selected)",
            ),
            _picker_row(
                label="Input directory",
                button_id=ids.RC_BTN_PICK_INPUT,
                label_id=ids.RC_LABEL_INPUT,
                placeholder="(no directory selected)",
            ),
            _picker_row(
                label="Output directory",
                button_id=ids.RC_BTN_PICK_OUTPUT,
                label_id=ids.RC_LABEL_OUTPUT,
                placeholder="(no directory selected)",
            ),
        ],
        className="run-console-pickers",
    )

    mode_row = html.Div(
        [
            dbc.Label("Mode", className="run-console-form-label"),
            dbc.RadioItems(
                id=ids.RC_RADIO_MODE,
                options=[
                    {"label": "Local", "value": "local"},
                    {"label": "SLURM", "value": "slurm"},
                ],
                value="local",
                inline=True,
            ),
        ],
        className="run-console-mode-row",
    )

    flags_row = html.Div(
        [
            dbc.Checklist(
                id=ids.RC_CHECKS_FLAGS,
                options=[
                    {"label": "Dry-run", "value": "dry_run"},
                    {"label": "Resume", "value": "resume"},
                ],
                value=[],
                inline=True,
            ),
        ],
        className="run-console-flags-row",
    )

    advanced_collapse = html.Div(
        [
            dbc.Button(
                "Advanced ▾",
                id=ids.RC_BTN_TOGGLE_ADVANCED,
                color="link",
                size="sm",
                n_clicks=0,
                className="run-console-collapse-toggle",
            ),
            dbc.Collapse(
                _build_advanced_section(),
                id=ids.RC_COLLAPSE_ADVANCED,
                is_open=False,
            ),
        ]
    )

    slurm_collapse = html.Div(
        [
            dbc.Button(
                "SLURM config ▾",
                id=ids.RC_BTN_TOGGLE_SLURM,
                color="link",
                size="sm",
                n_clicks=0,
                className="run-console-collapse-toggle",
            ),
            dbc.Collapse(
                _build_slurm_section(),
                id=ids.RC_COLLAPSE_SLURM,
                is_open=False,
            ),
        ]
    )

    sections: List = [
        pickers,
        html.Hr(),
        mode_row,
        flags_row,
        html.Hr(),
        advanced_collapse,
        slurm_collapse,
        html.Hr(),
        _build_action_buttons(),
    ]

    return html.Div(sections, id=ids.RC_FORM_COL, className="run-console-form-col")
