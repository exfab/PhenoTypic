"""split_master_by_category has exactly one caller: _publish_split_tables.

That function is the ``tables`` group of the single finalization path shared by
full, measure, recompile and ``--mode migrate`` (spec §5.3), reached in-process
through ``finalize_post_master_outputs`` and on SLURM through the finalizer
chain's ``tables`` task. A second call site would let one
mode drift, and a call outside the publication fence would let a SLURM
generation that lost ownership still write ``measurements_by_category/``.

The checks are reference-based, not call-based: an aliased import, a
``functools.partial``, a callback argument or a ``getattr`` string each count
as a reference, so none of them can add a second route to the function.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[3] / "src" / "phenotypic"
_OUTPUT_MANAGER = "_cli/_cli_output_manager.py"


def _modules() -> list[tuple[str, ast.Module]]:
    return [
        (path.relative_to(_SRC).as_posix(), ast.parse(path.read_text(encoding="utf-8")))
        for path in sorted(_SRC.rglob("*.py"))
    ]


class _ReferenceFinder(ast.NodeVisitor):
    """Record every reference to *name*, tagged ``<file>:<enclosing scope>``.

    A reference is a Load-context ``Name``/``Attribute``, an import ``alias``,
    or a string constant equal to *name* (``getattr``, ``__all__``). A
    ``def`` of *name* is a definition, recorded separately.
    """

    def __init__(self, name: str, filename: str) -> None:
        self.name = name
        self.filename = filename
        self.scope: list[str] = []
        self.references: list[tuple[str, ast.AST]] = []
        self.definitions: list[str] = []

    def _where(self) -> str:
        return f"{self.filename}:{'.'.join(self.scope) or '<module>'}"

    def _visit_scope(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        if node.name == self.name:
            self.definitions.append(self._where())
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    visit_FunctionDef = _visit_scope
    visit_AsyncFunctionDef = _visit_scope
    visit_ClassDef = _visit_scope

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == self.name and isinstance(node.ctx, ast.Load):
            self.references.append((self._where(), node))

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == self.name and isinstance(node.ctx, ast.Load):
            self.references.append((self._where(), node))
        self.generic_visit(node)

    def visit_alias(self, node: ast.alias) -> None:
        if self.name in (node.name, node.asname):
            self.references.append((self._where(), node))

    def visit_Constant(self, node: ast.Constant) -> None:
        if node.value == self.name:
            self.references.append((self._where(), node))


def _find(name: str) -> tuple[list[tuple[str, ast.AST, ast.Module]], list[str]]:
    references: list[tuple[str, ast.AST, ast.Module]] = []
    definitions: list[str] = []
    for filename, tree in _modules():
        finder = _ReferenceFinder(name, filename)
        finder.visit(tree)
        references.extend((where, node, tree) for where, node in finder.references)
        definitions.extend(finder.definitions)
    return references, definitions


def _parents(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def test_category_split_is_referenced_only_from_the_tables_group() -> None:
    references, definitions = _find("split_master_by_category")
    assert definitions == [f"{_OUTPUT_MANAGER}:<module>"]
    assert [where for where, _, _ in references] == [
        f"{_OUTPUT_MANAGER}:_publish_split_tables"
    ]


def test_the_tables_group_has_one_caller() -> None:
    # A second caller of the group would be the second route this file
    # exists to rule out, one level up.
    references, _ = _find("_publish_split_tables")
    assert [where for where, _, _ in references] == [
        f"{_OUTPUT_MANAGER}:publish_finalization_outputs"
    ]


def test_category_split_runs_inside_the_publication_fence() -> None:
    # The one reference must be `_guarded_terminal_best_effort(commit_guard,
    # lambda: split_master_by_category(...), ...)`: called directly by a lambda
    # that is an argument of the fenced wrapper, fenced by `commit_guard`.
    references, _ = _find("split_master_by_category")
    assert len(references) == 1
    _, node, tree = references[0]
    parents = _parents(tree)

    call = parents[node]
    assert isinstance(call, ast.Call) and call.func is node, "not called directly"
    lam = parents[call]
    assert isinstance(lam, ast.Lambda) and lam.body is call, "not the body of a lambda"
    guard = parents[lam]
    assert isinstance(guard, ast.Call) and lam in guard.args, "lambda is not a positional argument"
    assert isinstance(guard.func, ast.Name)
    assert guard.func.id == "_guarded_terminal_best_effort"
    assert guard.args and isinstance(guard.args[0], ast.Name)
    assert guard.args[0].id == "commit_guard"


def test_category_dir_is_referenced_only_by_split_master_by_category() -> None:
    references, definitions = _find("measurements_by_category_dir")
    assert definitions == ["sdk_/_io_constants.py:<module>"]
    assert sorted(where for where, _, _ in references) == [
        f"{_OUTPUT_MANAGER}:<module>",  # import
        f"{_OUTPUT_MANAGER}:split_master_by_category",  # the one use
        "sdk_/__init__.py:<module>",  # re-export import
        "sdk_/__init__.py:<module>",  # __all__ entry
    ]
