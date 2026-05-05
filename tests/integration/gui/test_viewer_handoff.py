"""Integration tests for the sidebar -> viewer hand-off.

Covers ``POST /sandbox/api/viewer/output-root`` and the
``compose_hub`` plumbing that lets the endpoint swap the viewer's
``OutputRoot`` and trigger a rebuild on the next request to
``/results/``.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from phenotypic.gui.shell import SandboxRoot
from phenotypic.gui.shell._app import compose_hub


def _make_minimal_output(root: Path, dataset: str = "d1") -> None:
    """Mirror ``tests/gui/results_viewer/test_output_root._make_minimal_output``.

    Kept inline here to avoid a cross-package test import.
    """
    (root / "results" / dataset / "overlays").mkdir(parents=True)
    (root / "results" / dataset / "measurements").mkdir(parents=True)
    pl.DataFrame(
        {
            "Metadata_Dataset": [dataset, dataset],
            "Metadata_ImageFile": ["a", "b"],
            "Metadata_Strain": ["s1", "s2"],
            "ObjectLabel": [1, 1],
            "Size_Area": [100.0, 200.0],
        }
    ).write_parquet(root / "master_measurements.parquet")
    for stem in ("a", "b"):
        (root / "results" / dataset / "overlays" / f"{stem}.png").touch()


@pytest.fixture()
def sandbox_with_output(tmp_path: Path) -> tuple[SandboxRoot, str]:
    out_dir = tmp_path / "output_synth"
    out_dir.mkdir()
    _make_minimal_output(out_dir)
    return SandboxRoot.from_path(tmp_path), "output_synth"


def test_post_swaps_output_root_and_rebuilds(
    sandbox_with_output: tuple[SandboxRoot, str],
) -> None:
    sandbox, rel = sandbox_with_output
    shell_app, viewer_session = compose_hub(sandbox, start_idle_thread=False)
    client = shell_app.server.test_client()

    # First /results/ hit builds the empty-state viewer. Dash returns the
    # bare index HTML; the actual layout is fetched via /_dash-layout.
    assert client.get("/results/").status_code == 200
    assert viewer_session.is_built() is True
    layout_before = client.get("/results/_dash-layout").get_json()
    assert "results-viewer-empty-state" in str(layout_before)

    # POST hand-off; endpoint validates layout, swaps state, releases session.
    post_resp = client.post(
        "/sandbox/api/viewer/output-root",
        data=json.dumps({"path": rel}),
        content_type="application/json",
    )
    assert post_resp.status_code == 200, post_resp.data
    payload = post_resp.get_json()
    assert payload["status"] == "ok"
    assert payload["abs_path"].endswith(rel)
    assert viewer_session.is_built() is False, "session should be released"

    # Next /results/ hit rebuilds with the new OutputRoot — empty-state gone.
    assert client.get("/results/").status_code == 200
    assert viewer_session.is_built() is True
    layout_after = client.get("/results/_dash-layout").get_json()
    layout_after_s = str(layout_after)
    assert "results-viewer-empty-state" not in layout_after_s
    assert "results-viewer-root" in layout_after_s


def test_post_rejects_path_outside_sandbox(
    sandbox_with_output: tuple[SandboxRoot, str],
) -> None:
    sandbox, _ = sandbox_with_output
    shell_app, _viewer_session = compose_hub(sandbox, start_idle_thread=False)
    client = shell_app.server.test_client()

    resp = client.post(
        "/sandbox/api/viewer/output-root",
        data=json.dumps({"path": "../../etc"}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "path escapes sandbox"


def test_post_rejects_invalid_layout(
    tmp_path: Path,
) -> None:
    # A directory inside the sandbox that is NOT a CLI output (no
    # master_measurements.parquet). OutputRoot.discover raises
    # FileNotFoundError; the endpoint surfaces it as 400.
    bad = tmp_path / "not_an_output"
    bad.mkdir()
    sandbox = SandboxRoot.from_path(tmp_path)
    shell_app, _viewer_session = compose_hub(sandbox, start_idle_thread=False)
    client = shell_app.server.test_client()

    resp = client.post(
        "/sandbox/api/viewer/output-root",
        data=json.dumps({"path": "not_an_output"}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert "master_measurements.parquet" in resp.get_json()["error"]


def test_post_requires_path_in_body(
    sandbox_with_output: tuple[SandboxRoot, str],
) -> None:
    sandbox, _ = sandbox_with_output
    shell_app, _viewer_session = compose_hub(sandbox, start_idle_thread=False)
    client = shell_app.server.test_client()

    resp = client.post(
        "/sandbox/api/viewer/output-root",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "missing 'path'"
