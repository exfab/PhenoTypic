"""Pipeline-running entry points import the deferred libraries before any image work."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest import mock

import pytest
from click.testing import CliRunner

PACKAGE_ROOT = Path(__file__).resolve().parents[3] / "src" / "phenotypic"

PIPELINE_WORKER_ENTRY_MODULES = (
    "_cli/_cli_process_single.py",
    "_cli/_cli_staged_slurm_worker.py",
    "_cli/_cli_recompile_worker.py",
    "_cli/_cli_checkpoint_handler.py",
)


def test_cli_aborts_before_any_output_when_a_runtime_dependency_is_broken(tmp_path: Path, monkeypatch) -> None:
    import phenotypic.phenotypicCLI as cli

    calls: list[str] = []

    def broken_install() -> None:
        calls.append("preload")
        raise ImportError("simulated broken numba install")

    monkeypatch.setattr(cli, "load_runtime_dependencies", broken_install)
    output_dir = tmp_path / "out"
    result = CliRunner().invoke(cli.phenotypic_cli, ["--input", str(tmp_path), "--output", str(output_dir)])

    assert calls == ["preload"]
    assert result.exit_code != 0
    assert isinstance(result.exception, ImportError)
    assert not output_dir.exists()


@pytest.mark.parametrize("relative_path", PIPELINE_WORKER_ENTRY_MODULES)
def test_pipeline_worker_entry_preloads_runtime_dependencies(relative_path: str) -> None:
    tree = ast.parse((PACKAGE_ROOT / relative_path).read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    assert any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "load_runtime_dependencies"
        for node in ast.walk(main)
    ), f"{relative_path}: main() never calls load_runtime_dependencies()"


def test_a_patched_deferred_cli_name_stays_patched_through_the_loader() -> None:
    import phenotypic.phenotypicCLI as cli
    from phenotypic._cli._cli_execution_strategies import create_execution_strategy

    with mock.patch("phenotypic.phenotypicCLI.create_execution_strategy") as patched:
        cli._load_cli_runtime()
        assert cli.create_execution_strategy is patched
    assert cli.create_execution_strategy is create_execution_strategy
