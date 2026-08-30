"""Process-mode stores cross the real CLI and Browse consumer seams."""

from __future__ import annotations

import hashlib
from pathlib import Path

import dash
from click.testing import CliRunner

from phenotypic.gui.browse import _source_render, _tile_routes
from phenotypic.gui.browse._source_probe import probe_source
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.phenotypicCLI import phenotypic_cli


def _tree_bytes(root: Path) -> dict[str, str]:
    """Return exact file-byte digests without importing production helpers."""
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_process_output_is_full_cli_input_and_browse_store_asset(
    tmp_path: Path,
    simple_pipeline_json: Path,
    synth_one_level_input: Path,
) -> None:
    """Both consumers accept one output while every source byte stays fixed."""
    runner = CliRunner()
    process_output = tmp_path / "process"
    source_before = _tree_bytes(synth_one_level_input)
    process_result = runner.invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(synth_one_level_input),
            "--output",
            str(process_output),
            "--mode",
            "process",
            "--layer",
            "rgb",
            "--force-local",
            "--njobs",
            "1",
        ],
    )
    assert process_result.exit_code == 0, process_result.output
    store = process_output / "day1" / "plateA.ome.zarr"
    assert store.is_dir()
    assert _tree_bytes(synth_one_level_input) == source_before
    process_before = _tree_bytes(process_output)

    full_result = runner.invoke(
        phenotypic_cli,
        [
            "--pipeline",
            str(simple_pipeline_json),
            "--input",
            str(process_output),
            "--output",
            str(tmp_path / "full"),
            "--force-local",
            "--njobs",
            "1",
        ],
    )
    assert full_result.exit_code == 0, full_result.output
    assert _tree_bytes(synth_one_level_input) == source_before
    assert _tree_bytes(process_output) == process_before

    sandbox = SandboxRoot.from_path(process_output)
    app = dash.Dash(__name__)
    app.layout = dash.html.Div()
    _tile_routes.register(app, sandbox)
    revision = probe_source(store, sandbox_root=process_output)
    token = _source_render.encode_token("day1/plateA.ome.zarr")
    response = app.server.test_client().get(
        f"/assets/{token}/{revision.cache_key}/zarr/zarr.json"
    )
    assert response.status_code == 200
    assert response.data == (store / "zarr.json").read_bytes()
    assert _tree_bytes(process_output) == process_before
