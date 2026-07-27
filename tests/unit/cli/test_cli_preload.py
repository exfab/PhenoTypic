"""Tests for custom-operation preloading in staged remote processes."""

from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from phenotypic._cli import _cli_checkpoint_handler as checkpoint_handler
from phenotypic._cli import _cli_staged_slurm_worker as staged_worker
from phenotypic._cli._cli_preload import preload_custom_operation_modules
from phenotypic.sdk_ import JobMetadataKey


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


def test_staged_finalizer_imports_registration_before_deserialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The finalizer registers custom operations before loading its pipeline."""
    module_name = "_phenotypic_finalizer_preload_probe"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        "import builtins\n"
        "builtins._phenotypic_preload_events.append('import')\n",
        encoding="utf-8",
    )
    events: list[str] = []
    monkeypatch.setattr(
        builtins,
        "_phenotypic_preload_events",
        events,
        raising=False,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("PHENOTYPIC_PRELOAD_MODULES", f"  {module_name}  ")
    sys.modules.pop(module_name, None)

    pipeline_path = tmp_path / "pipeline.json"
    output_dir = tmp_path / "out"
    job_metadata = {
        JobMetadataKey.DATASETS: {},
        JobMetadataKey.START_TIME: "2026-07-25T00:00:00",
        JobMetadataKey.PIPELINE_PATH: str(pipeline_path),
        JobMetadataKey.IMAGE_TYPE: "Image",
    }

    def deserialize_after_registration(_path: Path) -> object:
        events.append("deserialize")
        assert events == ["import", "deserialize"]
        return object()

    with (
        patch(
            "phenotypic._cli._cli_update_state.aggregate_state_from_events",
            return_value={},
        ),
        patch(
            "phenotypic._cli._cli_report_generator."
            "HTMLReportGenerator.generate_report"
        ),
        patch(
            "phenotypic._cli._cli_staged_orchestration.assert_active_epoch"
        ),
        patch(
            "phenotypic._cli._cli_readme_generator.READMEGenerator"
        ) as readme_generator,
        patch(
            "phenotypic.ImagePipeline.from_json",
            side_effect=deserialize_after_registration,
        ),
    ):
        checkpoint_handler._publish_staged_report_and_readme(
            output_dir,
            job_metadata,
            "epoch-1",
        )

    assert events == ["import", "deserialize"]
    readme_generator.return_value.generate.assert_called_once_with(
        output_dir,
        [],
    )
