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

#: Workers whose ``main`` must call the preload as its very first statement (after an
#: optional docstring). Anything later is not a fail-fast preload: the point is to blow
#: up on a broken install before the first image is touched, not afterwards.
FIRST_STATEMENT_WORKERS = (
    "_cli/_cli_process_single.py",
    "_cli/_cli_recompile_worker.py",
    "_cli/_cli_checkpoint_handler.py",
)

#: ``_cli_staged_slurm_worker.main`` builds its ``argparse`` parser inline, so statement 0
#: cannot be the preload. Its contract is the next-strictest one available: the preload is
#: the statement immediately after ``parse_args``, i.e. before the first statement that
#: does any work.
PARSE_ARGS_FIRST_WORKERS = ("_cli/_cli_staged_slurm_worker.py",)


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
    """The preload must run *first*, not merely somewhere inside ``main``.

    An ``ast.walk`` presence check passes with the call moved to the last line of
    ``main`` -- after every image has been processed -- which is the opposite of the
    fail-fast contract this module's docstring states. Position is the property; presence
    is not. The import check is the other half: a call whose name is never imported passes
    a presence check and raises ``NameError`` at run start on the cluster.
    """
    tree = ast.parse((PACKAGE_ROOT / relative_path).read_text(encoding="utf-8"))
    imported = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "load_runtime_dependencies" in imported, f"{relative_path}: the name is never imported"

    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    body = [
        node
        for node in main.body
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
    ]
    preload_indexes = [
        index
        for index, node in enumerate(body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "load_runtime_dependencies"
    ]
    assert preload_indexes, f"{relative_path}: main() never calls load_runtime_dependencies() as a statement"

    if relative_path in FIRST_STATEMENT_WORKERS:
        limit = 0
    elif relative_path in PARSE_ARGS_FIRST_WORKERS:
        parse_args_indexes = [
            index
            for index, node in enumerate(body)
            if isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "parse_args"
        ]
        assert parse_args_indexes, f"{relative_path}: main() no longer calls parse_args()"
        limit = parse_args_indexes[0] + 1
    else:
        # A new worker added to PIPELINE_WORKER_ENTRY_MODULES must state which contract it
        # holds to, rather than silently getting the weakest one.
        pytest.fail(f"{relative_path}: not classified in FIRST_STATEMENT_WORKERS or PARSE_ARGS_FIRST_WORKERS")

    assert preload_indexes[0] <= limit, (
        f"{relative_path}: the preload is statement {preload_indexes[0]} of main(), not <= {limit}"
    )


def test_a_patched_deferred_cli_name_stays_patched_through_the_loader() -> None:
    """The loader's ``all(...) -> continue`` skip is what protects a live patch.

    ``mock.patch.__enter__`` reads the original through ``phenotypicCLI.__getattr__``,
    which runs the loader and binds every name of every module before this body starts.
    The call below is therefore the same no-op the real command bodies make while a test's
    patch is active, and the assertion is that it leaves the Mock alone. Deleting the skip
    reddens this test: the loop then reassigns the real function over the Mock.
    """
    import phenotypic.phenotypicCLI as cli
    from phenotypic._cli._cli_execution_strategies import create_execution_strategy

    with mock.patch("phenotypic.phenotypicCLI.create_execution_strategy") as patched:
        cli._load_cli_runtime()
        assert cli.create_execution_strategy is patched
    assert cli.create_execution_strategy is create_execution_strategy
