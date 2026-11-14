"""Every name ``_gui/`` imports from ``phenotypic.sdk_`` still resolves (spec B8).

Issue #281 changes ``phenotypic.sdk_`` while leaving ``src/phenotypic/_gui/``
untouched until #283. These two guards fail the moment a change removes a name
the GUI imports, or re-signs an ``sdk_`` function the GUI calls:

- the first walks ``_gui/**`` with :mod:`ast` and asserts every imported name
  still exists on its module;
- the second compares each callable's :func:`inspect.signature` with
  ``gui_sdk_signatures.json``, generated once from ``main`` @ ``3248a9173`` (the
  signatures the GUI was written against). Classes are excluded from the
  baseline; a class's constructor is a pydantic or dataclass detail the GUI
  tests already exercise.

Regenerate the baseline only when #283 lands a GUI that calls the new
signatures; never to make a red guard green.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
from pathlib import Path

import phenotypic

GUI_ROOT = Path(phenotypic.__file__).parent / "_gui"
BASELINE_PATH = Path(__file__).parent / "gui_sdk_signatures.json"
SDK_PACKAGE = "phenotypic.sdk_"


def _module_of(path: Path, gui_root: Path) -> str:
    """Return the dotted module name of ``path``, a file under ``gui_root``."""
    relative = path.relative_to(gui_root.parent.parent).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _absolute_from(node: ast.ImportFrom, module: str, is_package: bool) -> str:
    """Resolve an ``ImportFrom``'s target, including a relative one."""
    if node.level == 0:
        return node.module or ""
    package = module.split(".") if is_package else module.split(".")[:-1]
    base = package[: len(package) - (node.level - 1)]
    return ".".join([*base, node.module] if node.module else base)


def _is_sdk(module: str) -> bool:
    return module == SDK_PACKAGE or module.startswith(f"{SDK_PACKAGE}.")


def gui_sdk_imports(gui_root: Path) -> dict[str, set[str]]:
    """Return every name a file under ``gui_root`` imports from ``phenotypic.sdk_``.

    Covers ``from phenotypic.sdk_[.<module>] import name`` (absolute or
    relative, at any nesting depth, ``TYPE_CHECKING`` blocks included) and
    ``import phenotypic.sdk_.<module>``, which is recorded as the name
    ``<module>`` on its parent package.

    Args:
        gui_root: The ``_gui`` package directory to walk.

    Returns:
        Imported names keyed by the dotted module they are imported from.
    """
    found: dict[str, set[str]] = {}
    for path in sorted(gui_root.rglob("*.py")):
        module = _module_of(path, gui_root)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                target = _absolute_from(node, module, path.name == "__init__.py")
                if not _is_sdk(target):
                    continue
                names = {alias.name for alias in node.names if alias.name != "*"}
                found.setdefault(target, set()).update(names)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_sdk(alias.name) and alias.name != SDK_PACKAGE:
                        parent, _, leaf = alias.name.rpartition(".")
                        found.setdefault(parent, set()).add(leaf)
    return found


def test_the_walk_finds_the_gui_imports_the_spec_names():
    """The walk is not vacuous: it sees the names spec B8 lists by hand."""
    names = set().union(*gui_sdk_imports(GUI_ROOT).values())
    assert {"MEASUREMENTS_CSV", "measurements_csv_path"} <= names


def test_every_gui_import_from_sdk_still_resolves():
    for module, names in gui_sdk_imports(GUI_ROOT).items():
        mod = importlib.import_module(module)
        for name in names:
            assert hasattr(mod, name), (
                f"{module}.{name} is imported by _gui/ and no longer exists"
            )


def test_every_sdk_function_the_gui_calls_keeps_its_signature():
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline, "the signature baseline is empty"
    for qualified, signature in baseline.items():
        module, _, name = qualified.rpartition(".")
        obj = getattr(importlib.import_module(module), name)
        assert str(inspect.signature(obj)) == signature, qualified
