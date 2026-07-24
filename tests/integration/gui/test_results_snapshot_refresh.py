"""Shared Results/Analysis snapshot binding and refresh integration tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from phenotypic import ImagePipeline
from phenotypic.gui import analysis, results_viewer
from phenotypic.gui._config import (
    CFG_ANALYSIS_SESSION,
    CFG_OUTPUT_ROOT,
    CFG_RESULTS_BINDING_STATE,
)
from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.gui.shell import SandboxRoot
from phenotypic.gui.shell._app import compose_hub
from phenotypic.gui.shell._session import ToolSession
from phenotypic.schema import METADATA
from phenotypic.sdk_ import atomic_write_json, gui_launch_owner_path
from tests._output_layout import seed_output_dir


def _seed_output(parent: Path, name: str = "output") -> Path:
    """Create a minimal complete output accepted by both sub-apps."""
    output = parent / name
    frame = pl.DataFrame(
        {
            "MetadataExperiment_Dataset": ["dataset"],
            str(METADATA.IMAGE_NAME): ["plate"],
            "Object_Label": [1],
            "Shape_Area": [100.0],
        }
    )
    seed_output_dir(
        output,
        frame,
        mirror=frame,
        pipeline=ImagePipeline(name="snapshot-refresh"),
    )
    (output / "results" / "dataset" / "measurements").mkdir(
        parents=True,
        exist_ok=True,
    )
    overlay = output / "deliverables" / "overlays" / "dataset" / "plate.png"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_bytes(b"overlay")
    return output


def _source_tree(root: Path) -> dict[str, tuple[str, bytes | None]]:
    """Capture relative type and bytes for an immutable-tree assertion."""
    snapshot: dict[str, tuple[str, bytes | None]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            snapshot[relative] = ("dir", None)
        elif path.is_file():
            snapshot[relative] = ("file", path.read_bytes())
    return snapshot


def _bind(client: Any, relative: str) -> Any:
    """Bind one output and assert the route accepted it."""
    response = client.post(
        "/sandbox/api/viewer/output-root",
        data=json.dumps({"path": relative}),
        content_type="application/json",
    )
    assert response.status_code == 200, response.get_json()
    return response


def _rewrite_mirror(output_root: OutputRoot, increment: float = 1.0) -> None:
    """Atomically publish a distinct measurements mirror revision."""
    mirror = output_root.layout.mirror_parquet
    replacement = mirror.with_name("measurements-replacement.parquet")
    (
        pl.read_parquet(mirror)
        .with_columns((pl.col("Shape_Area") + increment).alias("Shape_Area"))
        .write_parquet(replacement)
    )
    replacement.replace(mirror)


def test_bind_is_source_preserving_and_uses_external_cache(
    tmp_path: Path,
) -> None:
    """Binding reads the selected tree without migrating or caching in it."""
    output = _seed_output(tmp_path)
    legacy_sidecar = output / ".viewer_cache" / "qc_recipe.json"
    legacy_sidecar.parent.mkdir(parents=True)
    legacy_sidecar.write_text('{"checks": []}', encoding="utf-8")
    before = _source_tree(output)

    shell_app, viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    response = _bind(shell_app.server.test_client(), output.name)
    bound = viewer_session.get().server.config[CFG_OUTPUT_ROOT]

    assert response.get_json()["snapshot"]["processing_fingerprint"]
    assert _source_tree(output) == before
    assert bound.cache_dir.is_relative_to(
        tmp_path / ".phenotypic-gui" / "viewer_cache"
    )
    assert not bound.cache_dir.is_relative_to(output)


def test_refresh_atomically_swaps_results_and_analysis_to_one_descriptor(
    tmp_path: Path,
) -> None:
    """A successful refresh publishes both consumers at one revision."""
    output = _seed_output(tmp_path)
    shell_app, viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    client = shell_app.server.test_client()
    _bind(client, output.name)

    analysis_session: ToolSession[Any] = shell_app.server.config[
        CFG_ANALYSIS_SESSION
    ]
    old_viewer = viewer_session.get()
    old_analysis = analysis_session.get()
    old_root = old_viewer.server.config[CFG_OUTPUT_ROOT]
    assert old_root is old_analysis.server.config[CFG_OUTPUT_ROOT]

    _rewrite_mirror(old_root)
    response = client.post(
        "/sandbox/api/viewer/output-root",
        json={"refresh": True},
    )

    assert response.status_code == 200, response.get_json()
    new_viewer = viewer_session.get()
    new_analysis = analysis_session.get()
    new_root = new_viewer.server.config[CFG_OUTPUT_ROOT]
    assert new_viewer is not old_viewer
    assert new_analysis is not old_analysis
    assert new_root is new_analysis.server.config[CFG_OUTPUT_ROOT]
    assert new_root.snapshot != old_root.snapshot
    assert shell_app.server.config[CFG_RESULTS_BINDING_STATE]["snapshot"] is (
        new_root.snapshot
    )


def test_failed_refresh_keeps_both_live_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction failure rolls back without publishing either candidate."""
    output = _seed_output(tmp_path)
    shell_app, viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    client = shell_app.server.test_client()
    _bind(client, output.name)
    analysis_session: ToolSession[Any] = shell_app.server.config[
        CFG_ANALYSIS_SESSION
    ]
    old_viewer = viewer_session.get()
    old_analysis = analysis_session.get()
    old_state = dict(shell_app.server.config[CFG_RESULTS_BINDING_STATE])
    old_root = old_viewer.server.config[CFG_OUTPUT_ROOT]
    _rewrite_mirror(old_root)

    def _raise_analysis_failure(**_kwargs: Any) -> Any:
        raise RuntimeError("candidate analysis failed")

    monkeypatch.setattr(analysis, "create_app", _raise_analysis_failure)
    response = client.post(
        "/sandbox/api/viewer/output-root",
        json={"refresh": True},
    )

    assert response.status_code == 500
    assert response.get_json()["status"] == "unavailable"
    assert viewer_session.get() is old_viewer
    assert analysis_session.get() is old_analysis
    assert shell_app.server.config[CFG_RESULTS_BINDING_STATE] == old_state


def test_concurrent_source_change_is_stale_and_keeps_old_sessions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source race after Results construction is refused before publish."""
    output = _seed_output(tmp_path)
    shell_app, viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )
    client = shell_app.server.test_client()
    _bind(client, output.name)
    analysis_session: ToolSession[Any] = shell_app.server.config[
        CFG_ANALYSIS_SESSION
    ]
    old_viewer = viewer_session.get()
    old_analysis = analysis_session.get()
    old_root = old_viewer.server.config[CFG_OUTPUT_ROOT]
    real_results_factory = results_viewer.create_app

    def _build_then_change(**kwargs: Any) -> Any:
        candidate = real_results_factory(**kwargs)
        _rewrite_mirror(kwargs["output_root"])
        return candidate

    monkeypatch.setattr(results_viewer, "create_app", _build_then_change)
    response = client.post(
        "/sandbox/api/viewer/output-root",
        json={"refresh": True},
    )

    assert response.status_code == 409
    assert response.get_json()["status"] == "stale"
    assert viewer_session.get() is old_viewer
    assert analysis_session.get() is old_analysis
    assert viewer_session.get().server.config[CFG_OUTPUT_ROOT] is old_root


def test_cache_ownership_relocates_with_equivalent_sandbox(
    tmp_path: Path,
) -> None:
    """Equivalent copied outputs use the configured sandbox cache owner."""
    roots: list[OutputRoot] = []
    for sandbox_name in ("sandbox-a", "sandbox-b"):
        sandbox_path = tmp_path / sandbox_name
        sandbox_path.mkdir()
        output = _seed_output(sandbox_path)
        before = _source_tree(output)
        shell_app, viewer_session = compose_hub(
            SandboxRoot.from_path(sandbox_path),
            start_idle_thread=False,
        )
        _bind(shell_app.server.test_client(), output.name)
        bound = viewer_session.get().server.config[CFG_OUTPUT_ROOT]
        roots.append(bound)
        assert bound.cache_dir.is_relative_to(
            sandbox_path / ".phenotypic-gui" / "viewer_cache"
        )
        assert _source_tree(output) == before

    assert roots[0].cache_dir != roots[1].cache_dir
    assert roots[0].root != roots[1].root


def test_nonterminal_owner_marks_both_apps_as_active_snapshot(
    tmp_path: Path,
) -> None:
    """The shared descriptor records and displays active-run capture state."""
    output = _seed_output(tmp_path)
    owner = gui_launch_owner_path(output)
    owner.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        owner,
        {
            "version": 1,
            "run_id": "active",
            "generation": "generation-active",
            "status": "running",
        },
    )
    shell_app, viewer_session = compose_hub(
        SandboxRoot.from_path(tmp_path),
        start_idle_thread=False,
    )

    response = _bind(shell_app.server.test_client(), output.name)
    analysis_session: ToolSession[Any] = shell_app.server.config[
        CFG_ANALYSIS_SESSION
    ]
    viewer_root = viewer_session.get().server.config[CFG_OUTPUT_ROOT]
    analysis_root = analysis_session.get().server.config[CFG_OUTPUT_ROOT]

    assert response.get_json()["snapshot"]["active_run"] is True
    assert viewer_root.snapshot.active_run is True
    assert analysis_root.snapshot is viewer_root.snapshot
    assert "Active run snapshot" in str(viewer_session.get().layout)
    assert "Active run snapshot" in str(analysis_session.get().layout)
