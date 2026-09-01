"""Integration tests for the Launch view (Task C1).

The Launch view exposes the strategy / trials / storage-URL / screen / slurm form
plus a live command card whose initial server-side render comes from the pure
:func:`~phenotypic.gui.tune._command.render_launch_command`. The clientside mirror
keeps the card in sync, but the server-rendered initial command must already be a
valid ``uv run phenotypic-tune run …`` invocation for the bound run.
"""
from __future__ import annotations

import shlex
from pathlib import Path


def _launch_app(tmp_path: Path):  # type: ignore[no-untyped-def]
    """Build a loaded tune app over a 1-trial journal + a marked run."""
    from phenotypic.gui.tune import create_app
    from phenotypic.gui.tune._run_root import TuneRunRoot
    from phenotypic.sdk_ import trials_parquet_path
    from phenotypic.tune._study_store import JournalStudyStore, Trial

    images = tmp_path / "calibration"
    images.mkdir()
    parquet = trials_parquet_path(tmp_path)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    JournalStudyStore(
        trials=[
            Trial(number=0, params={"0.sigma": 1.0}, score=0.5, terms={}, n_images=2),
        ]
    ).to_parquet(parquet)
    root = TuneRunRoot.discover(tmp_path)
    return create_app(root=root, url_prefix="/tune/")


def test_launch_exposes_form_and_command_ids(tmp_path: Path) -> None:
    app = _launch_app(tmp_path)
    layout = str(app.layout)
    for component_id in (
        "tune-launch-strategy",
        "tune-launch-n-trials",
        "tune-launch-storage-url",
        "tune-launch-screen",
        "tune-launch-slurm",
        "tune-launch-paths-store",
        "tune-launch-command",
    ):
        assert component_id in layout


def _find_component(component, target_id):  # type: ignore[no-untyped-def]
    """Depth-first search for the Dash component whose ``id`` is ``target_id``."""
    if getattr(component, "id", None) == target_id:
        return component
    children = getattr(component, "children", None)
    if children is None:
        return None
    if not isinstance(children, list):
        children = [children]
    for child in children:
        if child is None or isinstance(child, str):
            continue
        found = _find_component(child, target_id)
        if found is not None:
            return found
    return None


def test_launch_initial_command_is_a_valid_run_invocation(tmp_path: Path) -> None:
    from phenotypic.gui.tune._command import render_launch_command
    from phenotypic.sdk_ import tuning_spec_path

    app = _launch_app(tmp_path)
    card = _find_component(app.layout, "tune-launch-command")
    assert card is not None
    rendered = card.children

    # A bare trials.parquet root carries no run.json, so images_dir is unknown
    # and the input renders as the ``<images>`` placeholder the user edits.
    expected = render_launch_command(
        str(tuning_spec_path(tmp_path)),
        "<images>",
        str(tmp_path),
        strategy="tpe",
        n_trials=50,
        storage_url=None,
        screen=False,
        slurm=False,
    )
    assert rendered == expected
    # The server-side initial render is a valid, re-parseable run invocation.
    tokens = shlex.split(rendered)
    assert tokens[:4] == ["uv", "run", "phenotypic-tune", "run"]
    assert "--strategy" in tokens and "tpe" in tokens
    assert "--n-trials" in tokens and "50" in tokens
