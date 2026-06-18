"""Launch view layout — the live ``python -m phenotypic.tune run`` card (C1).

The Launch view lets a user assemble the exact CLI invocation that runs (or
resumes) the bound run's ``tuning_spec.json`` over its calibration images. It is
composed of a small form — strategy / trial-budget / storage-URL / screen / slurm
— and a **live command card** that re-renders as the form changes.

The card's source-of-truth is the pure
:func:`~phenotypic.gui.tune._command.render_launch_command`; the clientside
callback (:func:`~phenotypic.gui.tune._callbacks._register_launch_command_mirror`)
mirrors the same logic into the card in the browser, and the initial render is
done server-side here so the card is correct before the first form interaction.

The Launch view only RENDERS a command string — it never spawns a process (the
no-re-optimize lock). Importing this module must never drag ``optuna`` into
``sys.modules``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import dash_bootstrap_components as dbc  # type: ignore[import-untyped]
from dash import dcc, html

from phenotypic.gui.tune import _ids as ids
from phenotypic.gui.tune._command import render_launch_command
from phenotypic.tune._strategies._config import STRATEGY_CHOICES

if TYPE_CHECKING:
    from phenotypic.gui.tune._run_root import TuneRunRoot

#: The default strategy when the Launch form first renders (TPE handles our mixed
#: categorical / conditional space and is the documented default sampler).
_DEFAULT_STRATEGY: str = "tpe"

#: The default trial budget the form pre-fills (mirrors the CLI's
#: ``_DEFAULT_N_TRIALS``; the user edits or clears it).
_DEFAULT_N_TRIALS: int = 50

#: The checklist option value used by the ``--screen`` / ``--slurm`` single-item
#: toggles (the presence of this value in the checklist's value list = on).
_TOGGLE_ON: str = "on"


def _spec_input_output(root: "TuneRunRoot") -> tuple[str, str, str]:
    """Resolve the ``(spec_path, input_dir, output_dir)`` for the run.

    The output directory is the run root's path; the spec is its
    ``deliverables/tuning_spec.json``; the input is the bound ``images_dir`` when
    known, else a ``<images>`` placeholder the user edits in the rendered command.

    Args:
        root: The validated tune output handle.

    Returns:
        The ``(spec_path, input_dir, output_dir)`` string triple.
    """
    from phenotypic.sdk_ import resolve_tuning_spec_path

    output_dir = str(root.path)
    spec_path = str(resolve_tuning_spec_path(root.path))
    input_dir = str(root.images_dir) if root.images_dir is not None else "<images>"
    return spec_path, input_dir, output_dir


def _strategy_dropdown() -> dbc.Select:
    """The strategy ``Select`` (options sourced from the real CLI choices)."""
    return dbc.Select(
        id=ids.TUNE_LAUNCH_STRATEGY,
        options=[{"label": choice, "value": choice} for choice in STRATEGY_CHOICES],
        value=_DEFAULT_STRATEGY,
    )


def _toggle(component_id: str, label: str) -> dbc.Checklist:
    """A single-option on/off checklist toggle (off by default)."""
    return dbc.Checklist(
        id=component_id,
        options=[{"label": label, "value": _TOGGLE_ON}],
        value=[],
        switch=True,
    )


def build_launch_view(root: "TuneRunRoot") -> html.Div:
    """Render the Launch view body for the bound run ``root``.

    Builds the strategy / budget / storage-URL / screen / slurm form, a hidden
    paths store (so the clientside command mirror reads the spec / input / output
    without re-deriving them), and the live command ``<code>`` card. The card is
    rendered server-side from :func:`render_launch_command` so it is correct on
    first load; the clientside callback keeps it in sync as the form changes.

    Args:
        root: The validated tune output handle.

    Returns:
        The Launch view body.
    """
    spec_path, input_dir, output_dir = _spec_input_output(root)
    initial_command = render_launch_command(
        spec_path,
        input_dir,
        output_dir,
        strategy=_DEFAULT_STRATEGY,
        n_trials=_DEFAULT_N_TRIALS,
        storage_url=None,
        screen=False,
        slurm=False,
    )
    form = html.Div(
        [
            html.Div(
                [
                    html.Label("Strategy", className="tune-launch-label"),
                    _strategy_dropdown(),
                ],
                className="tune-launch-field",
            ),
            html.Div(
                [
                    html.Label("Trials", className="tune-launch-label"),
                    dbc.Input(
                        id=ids.TUNE_LAUNCH_N_TRIALS,
                        type="number",
                        min=1,
                        step=1,
                        value=_DEFAULT_N_TRIALS,
                    ),
                ],
                className="tune-launch-field",
            ),
            html.Div(
                [
                    html.Label("Storage URL", className="tune-launch-label"),
                    dbc.Input(
                        id=ids.TUNE_LAUNCH_STORAGE_URL,
                        type="text",
                        placeholder="postgresql+psycopg://… (blank → local study.db)",
                        value="",
                    ),
                ],
                className="tune-launch-field tune-launch-field-wide",
            ),
            html.Div(
                [
                    _toggle(ids.TUNE_LAUNCH_SCREEN, "Two-round screening (--screen)"),
                    _toggle(ids.TUNE_LAUNCH_SLURM, "Distributed fleet (--slurm)"),
                ],
                className="tune-launch-toggles",
            ),
        ],
        className="tune-launch-form",
    )

    return html.Div(
        [
            dcc.Store(
                id=ids.TUNE_LAUNCH_PATHS_STORE,
                data={
                    "spec": spec_path,
                    "input": input_dir,
                    "output": output_dir,
                },
            ),
            html.P(
                "Assemble the command to run (or resume) this study, then copy it "
                "into a terminal. The GUI never spawns the run itself.",
                className="tune-launch-intro",
            ),
            form,
            html.Code(
                initial_command,
                id=ids.TUNE_LAUNCH_COMMAND,
                className="tune-launch-command",
            ),
        ],
        className="tune-launch",
    )


__all__ = ["build_launch_view"]
