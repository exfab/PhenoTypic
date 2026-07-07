"""Tests for shared SDK atomic write helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from phenotypic.sdk_ import (
    PARQUET_WRITE_OPTIONS,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_with_writer,
)


def _no_tmp_debris(directory: Path) -> bool:
    """Return whether ``directory`` has no lingering temp files."""
    return not any(p.name.endswith(".tmp") for p in directory.iterdir())


def test_atomic_write_with_writer_removes_temp_after_writer_failure(
    tmp_path: Path,
) -> None:
    """A writer failure must leave the old file and no temp debris."""
    target = tmp_path / "out.txt"
    target.write_text("old", encoding="utf-8")

    def writer(path: str) -> None:
        Path(path).write_text("new", encoding="utf-8")
        raise OSError("writer failed")

    with pytest.raises(OSError, match="writer failed"):
        atomic_write_with_writer(target, writer)

    assert target.read_text(encoding="utf-8") == "old"
    assert _no_tmp_debris(tmp_path)


def test_atomic_write_json_writes_pretty_json_with_trailing_newline(
    tmp_path: Path,
) -> None:
    """JSON helper should provide deterministic pretty output."""
    target = tmp_path / "state.json"

    atomic_write_json(target, {"b": 2, "a": 1})

    assert target.read_text(encoding="utf-8") == '{\n  "a": 1,\n  "b": 2\n}\n'
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert _no_tmp_debris(tmp_path)


def test_atomic_write_parquet_uses_shared_default_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parquet writes should use the one shared compression policy."""
    target = tmp_path / "frame.parquet"
    captured: dict[str, object] = {}

    def fake_to_parquet(self, path, **kwargs):  # noqa: ANN001
        captured["path"] = path
        captured["kwargs"] = kwargs
        Path(path).write_bytes(b"PARQUET")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    atomic_write_parquet(target, pd.DataFrame({"x": [1]}))

    assert target.read_bytes() == b"PARQUET"
    assert captured["kwargs"] == {"index": False, **PARQUET_WRITE_OPTIONS}
    assert _no_tmp_debris(tmp_path)
