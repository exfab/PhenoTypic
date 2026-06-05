"""The runs registry resolves run status from new and legacy layouts."""
from __future__ import annotations

from pathlib import Path

from phenotypic.gui.shell._runs_registry import RunRegistry
from phenotypic.tools_ import manifest_json_path


def _write_manifest(
    path: Path,
    payload: str = '{"is_complete": true, "execution_mode": "local"}',
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def test_status_read_finds_new_layout(tmp_path: Path) -> None:
    _write_manifest(manifest_json_path(tmp_path))
    mode, status, _ = RunRegistry._read_status_from_manifest(tmp_path)
    assert status != "unknown"


def test_status_read_finds_legacy_layout(tmp_path: Path) -> None:
    _write_manifest(tmp_path / "progress" / "manifest.json")
    mode, status, _ = RunRegistry._read_status_from_manifest(tmp_path)
    assert status != "unknown"
