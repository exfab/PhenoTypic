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

import phenotypic.sdk_ as sdk_package
import phenotypic.sdk_._image_record as image_record
import phenotypic.sdk_._io_constants as io_constants
import phenotypic.sdk_._master_io as master_io
import phenotypic.sdk_._run_state as run_state
import phenotypic.sdk_._schema_shape as schema_shape
import phenotypic.sdk_._state_types as state_types
import phenotypic.sdk_._verification_cache as verification_cache

# All EIGHT modules. Three of them are the cycle-breaking split (gen-r3 C5):
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
#
# _image_record is the FIFTH, added in P3 -- and the one whose *whole reason
# for existing* is this invariant. Its own module docstring calls the
# reader/writer split "forced rather than stylistic" because INV-LAYER forbids
# sdk_ importing phenotypic._cli "at module scope or inside a function": the
# readers live here precisely so sdk_ never has to. _run_state's deep path asks
# it for record_rejection, so the same transitive argument that added
# _schema_shape applies, and P6 Task 0 makes the edge unconditional when
# valid_image_success moves here.
#
# A module created to satisfy a rule is the last one that should be exempt from
# testing it. It was unwatched from its creation in P3 cluster 3.1 until this
# line -- clean throughout, which is the point: nothing would have noticed if it
# had stopped being.
# _io_constants and the package __init__ are the SIXTH and SEVENTH, and they
# are a different argument from the five above. Those five are watched because
# `_run_state` reaches them; these two are watched because **everything**
# reaches them. `sdk_/__init__.py` is the module the GUI actually imports, so a
# `phenotypic._cli` import there would be dragged in by every one of the five
# -- INV-LAYER would hold in each watched file and be violated in practice on
# every import path through the package. A guarantee that five files are clean
# is worth very little if the door they all come through is not.
#
# _master_io is the EIGHTH, added in P4. It is the one home of the v1/v2
# master discrimination, `sdk_/__init__.py` re-exports it, and the readers
# that will branch on it are GUI modules (P6) -- so it is reached through the
# seventh module's door by exactly the consumers INV-LAYER exists for. A new
# sdk_ module that the package __init__ re-exports is unwatched until it is
# named here, and the cost of noticing that later is a violation that already
# shipped.
_MODULES = (
    Path(state_types.__file__),
    Path(verification_cache.__file__),
    Path(schema_shape.__file__),
    Path(image_record.__file__),
    Path(io_constants.__file__),
    Path(sdk_package.__file__),
    Path(run_state.__file__),
    Path(master_io.__file__),
)


def _dotted_name(node: ast.AST) -> str:
    """Flatten an attribute chain to its dotted spelling, or ``""``.

    ``phenotypic._cli.x`` parses as nested ``Attribute`` nodes over a ``Name``,
    with no ``Import`` node anywhere -- so ``import phenotypic`` followed by an
    attribute access reaches the CLI package without tripping any of the four
    import shapes below. It is the one remaining way through the walk.
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


#: The package every watched module lives in. Relative imports resolve
#: against it, so ``level=1`` is ``phenotypic.sdk_`` and ``level=2`` is
#: ``phenotypic``.
_PACKAGE = "phenotypic.sdk_"

_FORBIDDEN = "phenotypic._cli"


def _names_the_cli(dotted: str) -> bool:
    """Whether *dotted* is ``phenotypic._cli`` or something inside it."""
    return dotted == _FORBIDDEN or dotted.startswith(_FORBIDDEN + ".")


def _absolute_module(node: ast.ImportFrom) -> str:
    """Resolve an ``ImportFrom`` to an absolute dotted module.

    ``ast`` puts the leading dots of a relative import in ``level`` and
    **strips them from** ``module``, so ``from .._cli import x`` arrives as
    ``module='_cli', level=2`` -- with no dot anywhere in the string.
    """
    if node.level == 0:
        return node.module or ""
    parts = _PACKAGE.split(".")
    base = ".".join(parts[: len(parts) - node.level + 1])
    return f"{base}.{node.module}" if node.module else base


def test_neither_module_ever_names_the_cli_package():
    """INV-LAYER, over every syntactic form that can reach the CLI package.

    The first version checked ``node.module.startswith(("phenotypic._cli",
    "._cli"))`` and missed four shapes -- three of them plausible, one of them
    the *most likely accidental violation there is*. Each is now covered and
    each is proved by a mutation:

    ===================================== =================================
    Form                                  Why the old walk missed it
    ===================================== =================================
    ``from .._cli import x``              ``module='_cli'``, dots moved to
                                          ``level``. The ``"._cli"`` prefix
                                          in the old test could therefore
                                          **never match anything** -- it was
                                          written believing ``ast`` keeps
                                          the dot. This is the natural
                                          relative form from ``sdk_``.
    ``from phenotypic import _cli``       ``module='phenotypic'``; the name
                                          is in ``node.names``
    ``from .. import _cli``               ``module`` is ``None``, so the
                                          ``and node.module`` guard skipped
                                          the node entirely
    ``importlib.import_module("...")``    not an import node at all
    ===================================== =================================

    Relative imports are resolved to absolute before comparison, and the
    imported *names* are checked too, not only the module they come from.
    """
    offenders: list[str] = []
    for source in _MODULES:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = _absolute_module(node)
                # `from phenotypic import _cli` names the package in an
                # alias rather than in `module`, so both are checked.
                candidates = [module] + [
                    f"{module}.{alias.name}" for alias in node.names
                ]
                if any(_names_the_cli(c) for c in candidates):
                    offenders.append(
                        f"{source.name}:{node.lineno} from {module} import "
                        + ", ".join(a.name for a in node.names)
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if _names_the_cli(alias.name):
                        offenders.append(
                            f"{source.name}:{node.lineno} import {alias.name}"
                        )
            elif isinstance(node, ast.Attribute):
                # `import phenotypic` + `phenotypic._cli.thing` -- no
                # ImportFrom, no Import naming the subpackage, no
                # import_module call. The fifth shape, and the only one of
                # the five that needs no import statement at all.
                dotted = _dotted_name(node)
                if _names_the_cli(dotted):
                    offenders.append(
                        f"{source.name}:{node.lineno} {dotted}"
                    )
            elif isinstance(node, ast.Call):
                # importlib.import_module("phenotypic._cli...") and
                # __import__("...") route around the import statement
                # entirely. Only a literal argument is checkable, which is
                # the point: a computed one is not something a reader can
                # audit either, and belongs nowhere near this boundary.
                func = node.func
                name = (
                    func.attr
                    if isinstance(func, ast.Attribute)
                    else func.id
                    if isinstance(func, ast.Name)
                    else ""
                )
                if name in {"import_module", "__import__"}:
                    for arg in node.args:
                        if isinstance(
                            arg, ast.Constant
                        ) and isinstance(arg.value, str):
                            if _names_the_cli(arg.value):
                                offenders.append(
                                    f"{source.name}:{node.lineno} "
                                    f"{name}({arg.value!r})"
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
