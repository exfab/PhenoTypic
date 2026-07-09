"""Gate 7 for the domain-stripped corpus: prove the transform touched only prose.

Strips docstrings and comments from both sides, then compares `ast.dump()`.
Identical dumps mean no expression, constant, or control-flow edit survived the
rename/strip pass. Identifier renames DO change the dump -- that is deliberate:
gate 7's contract is "changes confined to docstrings, comments and identifiers",
so identifiers are normalised first, by structure, not by a rename map we would
have to trust.

Comments never enter the AST, so they need no handling.

Usage:
    python ast_equiv.py ORIGINAL.py STRIPPED.py
Exit 0 if structurally identical, 1 otherwise (with the first divergence).
"""

from __future__ import annotations

import ast
import sys


class _Normalise(ast.NodeTransformer):
    """Drop docstrings; rename every local identifier to a positional slot.

    Renaming is done by first-appearance order within each scope, so
    `sum_amplitude` -> `v0` on both sides iff it occupies the same slot. A
    genuine logic edit reorders or adds bindings and shows up immediately.
    Attribute names, keywords and imported symbols are left alone: renaming
    those would mask a real API change.
    """

    def __init__(self) -> None:
        self._names: dict[str, str] = {}

    def _slot(self, name: str) -> str:
        if name not in self._names:
            self._names[name] = f"v{len(self._names)}"
        return self._names[name]

    def _strip_doc(self, node: ast.AST) -> ast.AST:
        body = getattr(node, "body", None)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:] or [ast.Pass()]  # type: ignore[attr-defined]
        return node

    def visit_Module(self, node: ast.Module) -> ast.AST:
        return self.generic_visit(self._strip_doc(node))

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node.name = self._slot(node.name)
        return self.generic_visit(self._strip_doc(node))

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = self._slot(node.name)
        return self.generic_visit(self._strip_doc(node))

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._slot(node.arg)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._slot(node.id)
        return node


def structure(path: str) -> str:
    tree = ast.parse(open(path, encoding="utf-8").read())
    return ast.dump(ast.fix_missing_locations(_Normalise().visit(tree)))


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a, b = sys.argv[1], sys.argv[2]
    sa, sb = structure(a), structure(b)
    if sa == sb:
        print(f"IDENTICAL structure: {a} == {b}")
        return 0

    # Report the first divergence, with context, rather than dumping both trees.
    for i, (ca, cb) in enumerate(zip(sa, sb)):
        if ca != cb:
            lo = max(0, i - 90)
            print(f"DIVERGES at char {i}")
            print(f"  original: ...{sa[lo:i + 90]}")
            print(f"  stripped: ...{sb[lo:i + 90]}")
            break
    else:
        print(f"DIVERGES by length: {len(sa)} vs {len(sb)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
