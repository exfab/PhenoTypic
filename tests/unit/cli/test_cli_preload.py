"""Tests for custom-operation preloading in staged remote processes."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import patch

import pytest

from phenotypic._cli import _cli_staged_slurm_worker as staged_worker
from phenotypic._cli._cli_preload import preload_custom_operation_modules


def test_preload_ignores_whitespace_and_empty_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only non-empty, trimmed module names are imported in list order."""
    monkeypatch.setenv(
        "PHENOTYPIC_PRELOAD_MODULES",
        "  custom.first , ,custom.second,   ",
    )

    with patch.object(importlib, "import_module") as import_module:
        preload_custom_operation_modules()

    assert [call.args[0] for call in import_module.call_args_list] == [
        "custom.first",
        "custom.second",
    ]


def test_preload_propagates_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing registration module is reported without being swallowed."""
    monkeypatch.setenv(
        "PHENOTYPIC_PRELOAD_MODULES",
        "custom.missing",
    )
    error = ImportError(
        "No module named 'custom.missing'",
        name="custom.missing",
    )

    with (
        patch.object(importlib, "import_module", side_effect=error),
        pytest.raises(ImportError, match="custom\\.missing") as raised,
    ):
        preload_custom_operation_modules()

    assert raised.value is error


def test_staged_worker_preloads_before_loading_manifest(
    tmp_path: Path,
) -> None:
    """The worker keeps preloading registrations before stage dispatch."""
    events: list[str] = []
    argv = [
        "--stage",
        "1",
        "--pipeline",
        str(tmp_path / "pipeline.json"),
        "--output-dir",
        str(tmp_path / "out"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--index",
        "0",
        "--epoch",
        "epoch-1",
    ]

    with (
        patch.object(
            staged_worker,
            "preload_custom_operation_modules",
            side_effect=lambda: events.append("preload"),
        ),
        patch.object(
            staged_worker,
            "load_staged_manifest",
            side_effect=lambda _path: events.append("manifest") or [],
        ),
        patch.object(
            staged_worker,
            "run_stage1_step",
            side_effect=lambda *_args, **_kwargs: events.append("stage"),
        ),
    ):
        result = staged_worker.main(argv)

    assert result == 0
    assert events == ["preload", "manifest", "stage"]
