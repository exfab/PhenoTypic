"""Home (landing) pane for the GUI shell.

Renders a welcome card with the resolved sandbox root, a tutorial-pointer
section, and the capability summary computed by walking the sandbox top
level once (no recursion — the sidebar handles deep navigation).

The summary is computed at layout-build time. Phase 5 will swap this for a
clientside fetch against ``/sandbox/api/root`` so refresh works without a
full layout re-render.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import html

from phenotypic.gui.shell._classifier import classify
from phenotypic.gui.shell._sandbox import SandboxRoot

__all__ = ["build_home_layout"]

#: Cap on how many top-level children the home page classifies at boot.
#: Mirrors the sidebar + JSON-API caps (``_CHILDREN_CLASSIFY_CAP``) so the
#: home tile counts stay snappy even on a sandbox with thousands of plate
#: folders. The summary is "at least N" beyond the cap.
_HOME_CLASSIFY_CAP = 500


def build_home_layout(sandbox: SandboxRoot) -> html.Div:
    """Build the home page layout.

    Args:
        sandbox: Frozen-at-launch sandbox root.

    Returns:
        ``html.Div`` ready to mount as the active tool's main pane.
    """
    summary = _summarise_sandbox(sandbox)

    return html.Div(
        [
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H2("PhenoTypic GUI", className="card-title"),
                        html.P(
                            "Unified hub for the pipeline builder, results "
                            "viewer, and run console.",
                            className="text-muted",
                        ),
                        html.Hr(),
                        html.Div(
                            [
                                html.Strong("Sandbox root: "),
                                html.Code(str(sandbox.root)),
                            ],
                            className="shell-home-root",
                        ),
                        html.Div(
                            _summary_grid(summary),
                            className="shell-home-summary",
                        ),
                        html.Hr(),
                        _quick_links(),
                    ]
                ),
                className="shell-home-card",
            ),
        ],
        className="shell-home",
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _summarise_sandbox(sandbox: SandboxRoot) -> dict[str, int]:
    """Walk up to ``_HOME_CLASSIFY_CAP`` direct children + tally counts.

    Counts include only direct children of the root — deep walks are
    deferred to clicked sidebar expansions. The cap keeps boot fast even
    on an HPCC scratch dir with thousands of plate folders; beyond it the
    home page surfaces "at least N" rather than walking the whole tree.
    """
    counts = {"images": 0, "outputs": 0, "pipelines": 0}
    try:
        children = list(sandbox.list_children())
    except (PermissionError, FileNotFoundError):
        return counts
    for idx, child in enumerate(children):
        if idx >= _HOME_CLASSIFY_CAP:
            break
        caps = classify(child)
        if caps.is_image_dir:
            counts["images"] += 1
        if caps.is_cli_output:
            counts["outputs"] += 1
        if caps.has_pipeline_json:
            counts["pipelines"] += 1
    return counts


def _summary_grid(counts: dict[str, int]) -> list:
    """Three-cell grid: images / outputs / pipelines."""
    cells = [
        ("Image dirs", counts["images"], "shell-summary-img"),
        ("CLI outputs", counts["outputs"], "shell-summary-out"),
        ("Pipeline files", counts["pipelines"], "shell-summary-cfg"),
    ]
    return [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div(str(value), className="shell-summary-num"),
                                html.Div(label, className="shell-summary-lbl"),
                            ]
                        ),
                        className=f"shell-summary-card {cls}",
                    ),
                )
                for label, value, cls in cells
            ],
            className="g-3",
        )
    ]


def _quick_links() -> html.Div:
    return html.Div(
        [
            html.H5("Tutorials", className="shell-home-tutorials-title"),
            html.Ul(
                [
                    html.Li(
                        html.A(
                            "Pipeline builder walkthrough",
                            href="https://wheeldon-lab.github.io/PhenoTypic/",
                            target="_blank",
                            rel="noopener",
                        )
                    ),
                    html.Li(
                        html.A(
                            "Results viewer cookbook",
                            href="https://wheeldon-lab.github.io/PhenoTypic/",
                            target="_blank",
                            rel="noopener",
                        )
                    ),
                    html.Li(
                        html.A(
                            "Running pipelines via the CLI",
                            href="https://wheeldon-lab.github.io/PhenoTypic/",
                            target="_blank",
                            rel="noopener",
                        )
                    ),
                ]
            ),
        ],
        className="shell-home-quicklinks",
    )
