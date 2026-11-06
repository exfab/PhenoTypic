"""No code in ``src/`` calls ``remove_small_objects`` except the compat helper.

scikit-image 0.26 deprecated ``remove_small_objects(min_size=...)`` (strictly
smaller than) in favour of ``max_size=...`` (smaller than or equal), and the
lock pins 0.25.2, which has no ``max_size``. No single direct call is silent
and correct on both, so every call goes through
``phenotypic.sdk_._skimage_compat.remove_objects_smaller_than``. This test
keeps a new direct call from bringing the ``FutureWarning`` back.

Two checks: the AST of every module (imports of, attribute access to, and
bare references to the function), and the raw text for call-shaped
occurrences with arguments, which is what catches a docstring example a
reader would copy. A bare prose mention, ``remove_small_objects()``, is fine.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src" / "phenotypic"
HELPER = SRC / "sdk_" / "_skimage_compat.py"
NAME = "remove_small_objects"
_CALL_OPEN = re.compile(rf"\b{NAME}\(")


def _code_references(path: Path) -> list[str]:
    """``file:line`` for each import of, attribute to, or name ``NAME``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(REPO_ROOT).as_posix()
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(
            alias.name == NAME for alias in node.names
        ):
            hits.append(f"{rel}:{node.lineno} imports {NAME}")
        elif isinstance(node, ast.Attribute) and node.attr == NAME:
            hits.append(f"{rel}:{node.lineno} uses .{NAME}")
        elif isinstance(node, ast.Name) and node.id == NAME:
            hits.append(f"{rel}:{node.lineno} uses {NAME}")
    return hits


def _text_calls_with_arguments(path: Path) -> list[str]:
    """``file:line`` for each ``NAME(<args>)`` in the raw text, docstrings too."""
    text = path.read_text(encoding="utf-8")
    rel = path.relative_to(REPO_ROOT).as_posix()
    hits = []
    for match in _CALL_OPEN.finditer(text):
        depth, end = 1, match.end()
        while end < len(text) and depth:
            depth += {"(": 1, ")": -1}.get(text[end], 0)
            end += 1
        arguments = text[match.end() : end - 1]
        if arguments.strip():
            line = text.count("\n", 0, match.start()) + 1
            hits.append(f"{rel}:{line} {NAME}({' '.join(arguments.split())})")
    return hits


def _modules_other_than_the_helper() -> list[Path]:
    return sorted(path for path in SRC.rglob("*.py") if path != HELPER)


def test_the_scanners_find_the_helpers_own_call() -> None:
    """Control: both scanners see the call the helper itself makes."""
    assert _code_references(HELPER)
    assert _text_calls_with_arguments(HELPER)


def test_no_module_references_remove_small_objects_outside_the_helper() -> (
    None
):
    hits = [
        hit
        for path in _modules_other_than_the_helper()
        for hit in _code_references(path)
    ]
    assert not hits, (
        "call phenotypic.sdk_._skimage_compat.remove_objects_smaller_than "
        f"instead of {NAME} directly: {hits}"
    )


def test_no_docstring_or_comment_shows_a_direct_call() -> None:
    hits = [
        hit
        for path in _modules_other_than_the_helper()
        for hit in _text_calls_with_arguments(path)
    ]
    assert not hits, (
        f"a {NAME}(...) example warns on scikit-image 0.26 (min_size=) or "
        f"fails on the locked 0.25.2 (max_size=); show something that "
        f"works on both: {hits}"
    )
