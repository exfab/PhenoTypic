"""Tests for exact GUI-to-local-CLI completion publication."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from phenotypic._cli._cli_gui_lifecycle import (
    gui_record_generation_from_environment,
    publish_local_gui_completion,
)
from phenotypic.sdk_ import manifest_json_path, run_completion_marker_path
from phenotypic.sdk_._io_constants import GUI_RECORD_GENERATION_ENV_VAR


def _write_manifest(
    output: Path,
    *,
    gui_generation: object | None = None,
    complete: bool = True,
    completed: int = 1,
    failed: int = 0,
    total: int = 1,
    mode: str = "local",
) -> None:
    path = manifest_json_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "execution_mode": mode,
        "is_complete": complete,
        "completed": completed,
        "failed": failed,
        "total_images": total,
    }
    if gui_generation is not None:
        payload["gui_record_generation"] = str(gui_generation)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_non_gui_local_cli_does_not_publish_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(GUI_RECORD_GENERATION_ENV_VAR, raising=False)
    _write_manifest(tmp_path)

    assert publish_local_gui_completion(tmp_path) is False
    assert not run_completion_marker_path(tmp_path).exists()


def test_gui_local_cli_publishes_exact_generation_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation = uuid4()
    monkeypatch.setenv(GUI_RECORD_GENERATION_ENV_VAR, str(generation))
    _write_manifest(tmp_path, gui_generation=generation)

    assert publish_local_gui_completion(tmp_path) is True
    marker = json.loads(
        run_completion_marker_path(tmp_path).read_text(encoding="utf-8")
    )
    assert marker["generation"] == str(generation)
    assert marker["mode"] == "local"
    assert marker["status"] == "complete"
    assert marker["finalizer_succeeded"] is True


@pytest.mark.parametrize(
    "manifest",
    [
        {"complete": False},
        {"completed": 0, "total": 1},
        {"failed": 1},
        {"mode": "slurm"},
    ],
)
def test_gui_local_cli_rejects_incoherent_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: dict[str, object],
) -> None:
    generation = uuid4()
    monkeypatch.setenv(GUI_RECORD_GENERATION_ENV_VAR, str(generation))
    _write_manifest(
        tmp_path,
        gui_generation=generation,
        **manifest,  # type: ignore[arg-type]
    )

    with pytest.raises(
        RuntimeError,
        match="incomplete, failed, or non-local",
    ):
        publish_local_gui_completion(tmp_path)
    assert not run_completion_marker_path(tmp_path).exists()


def test_gui_generation_environment_must_be_uuid() -> None:
    with pytest.raises(RuntimeError, match="not a valid UUID"):
        gui_record_generation_from_environment(
            {GUI_RECORD_GENERATION_ENV_VAR: "not-a-generation"}
        )


def test_gui_local_cli_rejects_coherent_manifest_from_prior_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(GUI_RECORD_GENERATION_ENV_VAR, str(uuid4()))
    _write_manifest(tmp_path, gui_generation=uuid4())

    with pytest.raises(RuntimeError, match="stale-generation"):
        publish_local_gui_completion(tmp_path)
    assert not run_completion_marker_path(tmp_path).exists()
