"""split_master_by_category has exactly one caller: finalize_post_master_outputs.

That function is the single finalization path shared by full, measure and
recompile (spec §5.3). A second call site would let one mode drift.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "src" / "phenotypic"


def _calls_by_enclosing_function(name: str) -> list[str]:
    found: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(func):
                if not isinstance(node, ast.Call):
                    continue
                target = node.func
                called = (
                    target.id
                    if isinstance(target, ast.Name)
                    else target.attr
                    if isinstance(target, ast.Attribute)
                    else None
                )
                if called == name:
                    found.append(f"{path.name}:{func.name}")
    return found


def test_category_split_is_called_only_from_finalize_post_master_outputs() -> None:
    assert _calls_by_enclosing_function("split_master_by_category") == [
        "_cli_output_manager.py:finalize_post_master_outputs"
    ]
