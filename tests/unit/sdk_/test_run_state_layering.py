"""INV-LAYER: sdk_/_run_state.py never reaches into phenotypic._cli.

Spec §5.2 calls the read/write asymmetry "structural, not conventional":
_run_state.py exports only readers, so the GUI cannot reach a publish_*
function. Structure that nothing tests is convention with extra steps -- the
GUI's 25 private phenotypic._cli imports across 9 modules are what that looks
like at scale (audit §7).

A LAZY import inside a function body is also a violation, not a loophole: it
would drag back load_processing_state's event-log replay, which spec §4.2
deletes. See OPEN-QUESTIONS Q4. The AST walk catches both forms.
"""

from __future__ import annotations

import ast
from pathlib import Path

import phenotypic.sdk_._run_state as run_state
import phenotypic.sdk_._schema_shape as schema_shape
import phenotypic.sdk_._state_types as state_types
import phenotypic.sdk_._verification_cache as verification_cache

# All FOUR modules. Three of them are the cycle-breaking split (gen-r3 C5):
# the dataclasses live in their own leaf so _verification_cache can cache
# whole ImageState objects without it and _run_state importing each other, and
# INV-LAYER binds the leaf exactly as it binds the other two.
#
# _schema_shape is the fourth, added when the pre-consolidation detection moved
# out of _cli/_cli_schema_gate so that resolve_run_state could emit spec
# §4.3's reader advisory from the same predicate the CLI refuses with.
# _run_state imports it, so a phenotypic._cli import there would be a
# TRANSITIVE INV-LAYER violation this walk would otherwise miss. The guarantee
# has to follow the code, or it silently stops covering it.
_MODULES = (
    Path(state_types.__file__),
    Path(verification_cache.__file__),
    Path(schema_shape.__file__),
    Path(run_state.__file__),
)


def test_neither_module_ever_names_the_cli_package():
    offenders: list[str] = []
    for source in _MODULES:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(("phenotypic._cli", "._cli")):
                    offenders.append(
                        f"{source.name}:{node.lineno} from {node.module}"
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("phenotypic._cli"):
                        offenders.append(
                            f"{source.name}:{node.lineno} import {alias.name}"
                        )
    assert not offenders, (
        "INV-LAYER: the run-state readers must not import phenotypic._cli. "
        f"Found: {offenders}"
    )


def test_run_state_exports_no_writer():
    forbidden = ("publish", "write", "mint", "append", "save", "delete")
    exported = getattr(run_state, "__all__", None)
    assert exported is not None, "_run_state.py must declare __all__"
    bad = [
        name
        for name in exported
        if any(name.lower().startswith(prefix) for prefix in forbidden)
    ]
    assert not bad, f"_run_state.py exports writers: {bad}"
