"""A deferred CLI name must be patched on ``phenotypic.phenotypicCLI``, not on its source.

``phenotypicCLI._load_cli_runtime()`` binds each deferred name once and never re-reads the
module it came from -- that is what the ``all(...) -> continue`` skip buys, and it is what
keeps a live ``mock.patch("phenotypic.phenotypicCLI.<name>")`` in force. The cost is that a
patch on the **defining** module which is open across the first load in a process leaves the
Mock bound in ``phenotypicCLI`` after the patch exits, for the life of the interpreter.
Under ``-p no:randomly`` that is reproducible; under random ordering it is a shard-dependent
flake whose failure points at an unrelated test.

Before the lazy-startup change the bind happened at ``phenotypicCLI`` import time, i.e.
before any test could patch anything, so the window did not exist. This guard reddens the
moment someone opens it.

**This is a partial net, not a proof. It has two blind spots, and they are equally
real -- do not treat the second as a footnote to the first:**

1. **It matches the dotted-string form only.** ``patch.object(module, "name")`` and
   ``monkeypatch.setattr(module, "name", ...)`` name the module by object, so no text
   search can see them. One such patch exists today
   (``test_lifecycle_publication_races.py:122``) and is inert for an unrelated reason.
2. **It only reads files that also mention ``phenotypicCLI``.** A file that never names
   the CLI module cannot reach ``_load_cli_runtime()``, so its source-module patches are
   inert -- three such patches exist today, in
   ``tests/unit/cli/test_cli_checkpoint_handler.py``, and are correct as written. But a
   route through a shared fixture in a ``conftest.py`` that the patching file does not
   itself name would reach the loader and **evade this guard entirely**. Widening the
   scope to every file costs those three false positives; that trade was made
   deliberately, and it is the first thing to revisit if this guard ever misses a real
   one.

A third exclusion is not a blind spot but a correctness rule: a dotted path that
continues past the deferred name (``...OutputManager.from_config``) sets an attribute on
the bound object, which is the same object the loader binds, so the patch is visible
through the CLI binding and restored on exit. The trailing-boundary check excludes it
because it is not the hazard, not because it is hard to see.
"""

from __future__ import annotations

import re
from pathlib import Path

from phenotypic.phenotypicCLI import _CLI_RUNTIME_IMPORTS

TESTS_ROOT = Path(__file__).resolve().parents[2]

#: This file spells no target literally -- it builds them from the shipped table -- but it
#: is excluded anyway so the guard can never trip over its own prose.
SELF = Path(__file__).resolve()

#: Marker that a file can reach the loader at all. See limit 2 above.
CLI_MODULE_MARKER = "phenotypicCLI"


def test_no_test_patches_a_deferred_cli_name_on_its_defining_module() -> None:
    """Fail on any ``"<source module>.<deferred name>"`` literal in a CLI-touching test."""
    # Derived from the shipped module, not transcribed, so the pair list cannot drift from
    # the code it guards.
    patterns = {
        f"{module_name}.{name}": re.compile(re.escape(f"{module_name}.{name}") + r"(?![\w.])")
        for module_name, names in _CLI_RUNTIME_IMPORTS.items()
        for name in names
    }
    assert patterns, "the deferred-name table is empty, so this guard cannot fail"

    scanned = 0
    offenders: list[str] = []
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        if path.resolve() == SELF:
            continue
        source = path.read_text(encoding="utf-8")
        if CLI_MODULE_MARKER not in source:
            continue
        scanned += 1
        for target, pattern in sorted(patterns.items()):
            for line_number, line in enumerate(source.splitlines(), start=1):
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(TESTS_ROOT)}:{line_number}: {target}")

    assert scanned, "no test file mentions the CLI module, so this guard cannot fail"
    assert offenders == [], (
        "a deferred CLI name is patched on its defining module; the binding "
        "phenotypicCLI takes is a one-time snapshot, so the Mock outlives the patch. "
        f"Patch 'phenotypic.phenotypicCLI.<name>' instead: {offenders}"
    )
