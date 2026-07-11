"""Tests for generated reference check/write helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import get_args

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_script(script_name: str) -> ModuleType:
    """Import a script module from the repository's scripts directory."""
    script_path = REPO_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name, script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[script_name] = module
    spec.loader.exec_module(module)
    return module


def test_check_mode_reports_missing_output(tmp_path, capsys) -> None:
    """Missing generated docs should fail with the regenerate command."""
    from scripts._reference_generator import write_or_check_generated_file

    output = tmp_path / "missing" / "reference.rst"
    rc = write_or_check_generated_file(
        output_path=output,
        rendered="new content",
        check=True,
        regenerate_command="uv run python scripts/example.py",
    )

    assert rc == 1
    assert not output.exists()
    err = capsys.readouterr().err
    assert f"{output} does not exist" in err
    assert "uv run python scripts/example.py" in err


def test_check_mode_reports_drift(tmp_path, capsys) -> None:
    """Out-of-date generated docs should fail without writing."""
    from scripts._reference_generator import write_or_check_generated_file

    output = tmp_path / "reference.rst"
    output.write_text("old content", encoding="utf-8")

    rc = write_or_check_generated_file(
        output_path=output,
        rendered="new content",
        check=True,
        regenerate_command="uv run python scripts/example.py",
    )

    assert rc == 1
    assert output.read_text(encoding="utf-8") == "old content"
    err = capsys.readouterr().err
    assert f"{output} is out of date" in err
    assert "uv run python scripts/example.py" in err


def test_check_mode_returns_zero_for_matching_output(tmp_path, capsys) -> None:
    """Matching generated docs should pass silently."""
    from scripts._reference_generator import write_or_check_generated_file

    output = tmp_path / "reference.rst"
    output.write_text("current content", encoding="utf-8")

    rc = write_or_check_generated_file(
        output_path=output,
        rendered="current content",
        check=True,
        regenerate_command="uv run python scripts/example.py",
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_write_mode_creates_parent_and_writes_output(tmp_path, capsys) -> None:
    """Non-check mode should create parent directories and write bytes."""
    from scripts._reference_generator import write_or_check_generated_file

    output = tmp_path / "generated" / "reference.rst"

    rc = write_or_check_generated_file(
        output_path=output,
        rendered="new content",
        check=False,
        regenerate_command="uv run python scripts/example.py",
    )

    assert rc == 0
    assert output.read_text(encoding="utf-8") == "new content"
    assert f"Wrote {output}" in capsys.readouterr().out


def test_validation_reference_mapping_covers_live_issue_kinds() -> None:
    """Validation reference metadata should cover the live issue taxonomy."""
    module = _load_script("generate_validation_reference.py")

    assert module._check_coverage(module._RULES) is None


def test_dispatch_reference_mapping_covers_live_dispatch_kinds() -> None:
    """Dispatch reference metadata should cover the live GUI dispatch surface."""
    module = _load_script("generate_dispatch_reference.py")

    assert module._check_coverage(module._DISPATCH_KINDS) is None


def test_dispatch_reference_order_matches_live_dispatch_kind_order() -> None:
    """The generated page should follow the live dispatcher declaration order."""
    module = _load_script("generate_dispatch_reference.py")

    from phenotypic.gui.builder._callbacks import DispatchKind

    assert [entry["kind"] for entry in module._DISPATCH_KINDS] == list(
        get_args(DispatchKind)
    )


def test_validation_reference_documents_blocking_runtime_issues() -> None:
    """Non-numbered runtime validation issues should document real severity."""
    module = _load_script("generate_validation_reference.py")
    entries_by_kind = {
        kind: entry
        for entry in module._RULES
        for kind in entry["kinds"]
    }

    assert entries_by_kind["unknown_class"]["severity"] == "error"
    assert entries_by_kind["unsupported_linear"]["severity"] == "error"
    assert "fixed linear" in entries_by_kind["unsupported_linear"]["mechanic"]


def test_validation_reference_renders_non_numbered_errors_as_blocking() -> None:
    """Error-severity entries without rule numbers should not render as advisory."""
    module = _load_script("generate_validation_reference.py")

    rendered = module._render_rst(module._RULES)

    assert "Blocking — Class not in the operation registry" in rendered
    assert (
        "Blocking — Fixed linear editor cannot represent this DAG shape"
        in rendered
    )
    assert "Advisory — Class not in the operation registry" not in rendered


def test_dispatch_reference_notes_match_live_fixed_linear_routes() -> None:
    """Docs should distinguish staged requests from validated commits."""
    module = _load_script("generate_dispatch_reference.py")
    entries = {entry["kind"]: entry for entry in module._DISPATCH_KINDS}

    assert "any string" in entries["linear_delete_node_request"]["notes"]
    assert "existence and spine validation" in entries[
        "linear_delete_node_confirm"
    ]["notes"]
    assert "not emitted by the current fixed-linear UI" in entries[
        "linear_select_aux_value"
    ]["notes"]


@pytest.mark.parametrize(
    ("script_name", "entries_name", "command"),
    [
        (
            "generate_validation_reference.py",
            "_RULES",
            "uv run python scripts/generate_validation_reference.py",
        ),
        (
            "generate_dispatch_reference.py",
            "_DISPATCH_KINDS",
            "uv run python scripts/generate_dispatch_reference.py",
        ),
    ],
)
def test_reference_generators_delegate_check_write(
    monkeypatch,
    script_name: str,
    entries_name: str,
    command: str,
) -> None:
    """Generator mains should share the check/write mechanics."""
    module = _load_script(script_name)
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(module, "_check_coverage", lambda entries: None)
    monkeypatch.setattr(module, "_render_rst", lambda entries: "rendered")

    def fake_write_or_check_generated_file(**kwargs):
        calls.append(kwargs)
        return 0

    monkeypatch.setattr(
        module,
        "write_or_check_generated_file",
        fake_write_or_check_generated_file,
    )

    assert module.main(["--check"]) == 0
    assert calls == [
        {
            "output_path": module.OUTPUT_RST,
            "rendered": "rendered",
            "check": True,
            "regenerate_command": command,
        }
    ]
    assert getattr(module, entries_name)
