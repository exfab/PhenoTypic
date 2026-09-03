"""Shared runner for the import-surface goldens.

Imported by BOTH ``scripts/capture_import_laziness_goldens.py`` (which writes)
and ``tests/unit/test_import_surface.py`` (which compares), so the capture pass
and the verification pass can never disagree about how a surface is measured.
This mirrors the arrangement in :mod:`tests.migration._runner`.

The goldens exist to make an import-laziness refactor checkable. Moving an
``import`` from module scope into a function body is invisible to an ordinary
test suite: the library still loads the first time anything calls the function,
so every test still passes whether the deferral worked or not. Three things are
therefore recorded:

* **Export surface** -- ``__all__`` for every public package, plus the resolved
  identity (module + qualname + kind) of every exported name. Asserted as
  strict equality. A lazy ``__getattr__`` that returns a *different* object, or
  silently stops serving a name, fails here.
* **Eager module set** -- what ``sys.modules`` holds after a bare
  ``import phenotypic`` and after ``import phenotypic.phenotypicCLI``, measured
  in a subprocess. This is the number the refactor is trying to move.
* **Forbidden-eager ratchet** -- :data:`FORBIDDEN_EAGER`, which starts empty and
  gains a library each time a stage evicts one. Once a name is listed it may
  never come back.

``dir()`` is recorded but deliberately **not** asserted: ``dir(module)`` reads
``__dict__`` and does not trigger PEP-562 ``__getattr__``, so it legitimately
shrinks whenever a name becomes lazy. ``__all__`` is the contract; ``dir()`` is
a diagnostic.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = REPO_ROOT / "tests" / "fixtures" / "import_laziness"
IMPORT_SURFACE_GOLDEN = GOLDEN_DIR / "import_surface.json"
TIMINGS_LOG = GOLDEN_DIR / "timings.json"

SCHEMA_VERSION = 1

#: Public packages whose export surface is pinned. ``phenotypic.gui`` is out of
#: scope: it is an optional extra whose ``__getattr__`` imports dash, so
#: resolving its exports would make the golden depend on the GUI extra being
#: installed.
PUBLIC_PACKAGES: tuple[str, ...] = (
    "phenotypic",
    "phenotypic.abc_",
    "phenotypic.abc_.plotting",
    "phenotypic.analysis",
    "phenotypic.correction",
    "phenotypic.data",
    "phenotypic.detect",
    "phenotypic.detect.nn",
    "phenotypic.enhance",
    "phenotypic.grid",
    "phenotypic.measure",
    "phenotypic.plotting",
    "phenotypic.post",
    "phenotypic.prefab",
    "phenotypic.refine",
    "phenotypic.schema",
    "phenotypic.sdk_",
    "phenotypic.settings",
    "phenotypic.tune",
    "phenotypic.tune.score",
    "phenotypic.tune.strategy",
    "phenotypic.util",
)

#: ``phenotypic.detect.nn`` resolves its exports through a lazy ``__getattr__``
#: that imports torch and the transformers stack -- tens of seconds, and it is
#: gated behind an optional extra. Its ``__all__`` is still pinned; only the
#: per-name identity resolution is skipped.
RESOLVE_SKIP: frozenset[str] = frozenset({"phenotypic.detect.nn"})

#: Import targets whose eager module set is measured.
EAGER_TARGETS: tuple[str, ...] = ("phenotypic", "phenotypic.phenotypicCLI")

#: The ratchet. A library listed here must NOT be in ``sys.modules`` after any
#: target in :data:`EAGER_TARGETS` is imported. Empty at stage 0 -- each stage
#: of the laziness refactor appends what it evicted, and nothing is ever
#: removed. Keep the comment naming the stage that added each entry.
FORBIDDEN_EAGER: tuple[str, ...] = (
    # Stage 1. `colour` was the single largest cost on the startup path; `numba`
    # (with `llvmlite`, and `coverage` via `numba.misc.coverage_support`) came in
    # for two detectors' method bodies; `h5py` for a read/write surface nothing
    # calls since per-image storage moved to OME-Zarr.
    "colour",
    "coverage",
    "h5py",
    "llvmlite",
    "numba",
)

#: Files permitted to import a :data:`FORBIDDEN_EAGER` library at module scope.
#: Every one is genuinely off the ``import phenotypic`` path -- reachable only
#: through a lazy ``__getattr__`` or a function-body import -- and each would be
#: awkward to defer in place: the numba kernels apply ``@numba.njit`` as a
#: decorator at import, and ``hdf_.py`` has no ``from __future__ import
#: annotations``, so its ``h5py.File`` annotations evaluate at def time.
#:
#: This allowlist is what makes :func:`FORBIDDEN_MODULE_SCOPE_IMPORTS` obsolete
#: as a *sufficient* check. A hand-list of files a stage happened to touch cannot
#: see a NEW file that imports a deferred library at module scope -- off the eager
#: path today, one careless re-export away from being on it tomorrow, with
#: nothing failing in between. Scanning every file and allowing three named
#: exceptions inverts that: new offenders fail by default.
MODULE_SCOPE_IMPORT_ALLOWLIST: frozenset[tuple[str, str]] = frozenset(
    {
        ("sdk_/hdf_.py", "h5py"),
        ("sdk_/reconnect/_tensor_voting.py", "numba"),
        ("sdk_/branch_pathfinding/_dijkstra_kernels.py", "numba"),
    }
)

#: ``(relative source path, module that must not be imported at module scope)``
#: pairs checked by static AST analysis. This catches a module-scope import the
#: runtime probe would miss because the file happens to be off the eager path
#: today, and would only start costing on the day something imports it. Entries
#: may name a package (``phenotypic.sdk_.reconnect``) as well as a library.
#: Grows alongside :data:`FORBIDDEN_EAGER`.
FORBIDDEN_MODULE_SCOPE_IMPORTS: tuple[tuple[str, str], ...] = (
    # Stage 1 -- colour.
    ("sdk_/colourspace.py", "colour"),
    ("util/_robust_color_stats.py", "colour"),
    ("_core/_image_parts/color_space_accessors/_xyz_conversion.py", "colour"),
    ("_core/_image_parts/color_space_accessors/_xyz_d65_accessor.py", "colour"),
    ("_core/_image_parts/color_space_accessors/_cielab_accessor.py", "colour"),
    (
        "_core/_image_parts/color_space_accessors/_chromaticity_xy_accessor.py",
        "colour",
    ),
    ("correction/_color_correction/_helpers.py", "colour"),
    ("correction/_color_correction/_color_corrector.py", "colour"),
    ("correction/_color_correction/_color_correction_report.py", "colour"),
    ("correction/_color_correction/_color_checker_profile.py", "colour"),
    # Stage 1 -- numba, reached only through these two detectors.
    ("detect/_filamentous_fungi_detector.py", "phenotypic.sdk_.reconnect"),
    ("detect/_two_k_filamentous_detector.py", "phenotypic.sdk_.reconnect"),
    # Stage 1 -- h5py.
    ("_core/_image_parts/_image_io_handler.py", "h5py"),
)


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


def _probe(script: str) -> Any:
    """Run *script* in a clean subprocess and parse the JSON it prints.

    A subprocess is mandatory: ``sys.modules`` is process-global and pytest has
    already imported most of the library by the time any test runs, so an
    in-process measurement would report the test session's imports rather than
    the package's.
    """
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    # Warnings from third-party packages (mahotas emits SyntaxWarnings) land on
    # stderr, so only stdout is parsed -- and only its last line, in case an
    # import prints.
    return json.loads(completed.stdout.strip().splitlines()[-1])


_EAGER_PROBE = '''
import json, sys

forbidden = json.loads({forbidden!r})

before = set(sys.modules)
import {target}  # noqa: F401
added = set(sys.modules) - before

stdlib = sys.stdlib_module_names


def genuinely_loaded(root):
    """Whether *root* names a third-party package that is really imported.

    Two ways this can be wrong, both seen:

    * Deriving roots by splitting every module name counts a package whose
      *submodule key* is occupied by a lazy stand-in. `_startup_perf` puts one
      at `colour.plotting` so colour's own `__init__` binds it instead of
      importing matplotlib and the spectral datasets, which made `colour` look
      imported when the package never was. Hence: require the root's own entry.
    * Reading the stub marker with `getattr` runs the module's own
      `__getattr__`, so a package with a permissive one could pass itself off
      as a stub and vanish from the report. Hence: read `__dict__` directly.
    """
    if root in stdlib or root.startswith("_") or root == "phenotypic":
        return False
    module = sys.modules.get(root)
    if module is None:
        return False
    return not vars(module).get("__phenotypic_lazy_stub__", False)


third_party = sorted(
    {{root for root in {{name.split(".")[0] for name in added}} if genuinely_loaded(root)}}
)
phenotypic_modules = sorted(name for name in added if name.split(".")[0] == "phenotypic")
print(json.dumps({{
    "third_party_roots": third_party,
    "phenotypic_module_count": len(phenotypic_modules),
    "phenotypic_modules": phenotypic_modules,
    "total_modules": len(sys.modules),
    # A forbidden library already loaded before the import under test does not
    # show up in `added`, so the deferral check would pass vacuously. This is
    # not hypothetical: pytest-cov's .pth starts coverage in every subprocess,
    # which would silence the `coverage` entry precisely under `pytest --cov`.
    # Report both the pre-existing set (the measurement is void if non-empty)
    # and membership in the FINAL sys.modules, which no preload can hide.
    "forbidden_preloaded": sorted(r for r in forbidden if r in before),
    "forbidden_in_final": sorted(r for r in forbidden if genuinely_loaded(r)),
}}))
'''


def measure_eager_imports(target: str) -> dict[str, Any]:
    """Measure what importing *target* adds to ``sys.modules``.

    Only the *incremental* set is reported, following
    ``tests/unit/sdk_/reconnect/test_import_rules.py`` -- interpreter startup
    and any ``sitecustomize`` imports must not be attributed to the package.
    """
    return _probe(
        _EAGER_PROBE.format(target=target, forbidden=json.dumps(list(FORBIDDEN_EAGER)))
    )


_SURFACE_PROBE = '''
import importlib, json, sys

packages = json.loads({packages!r})
skip_resolve = set(json.loads({skip!r}))

def identify(obj):
    """A stable, diffable identity for an exported object."""
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if isinstance(obj, type) or callable(obj):
        return f"{{type(obj).__name__}}:{{module}}:{{qualname}}"
    if isinstance(obj, __import__("types").ModuleType):
        return f"module:{{obj.__name__}}"
    # set/frozenset repr order follows PYTHONHASHSEED, so canonicalize before
    # recording -- otherwise the golden fails on a re-run for no real reason.
    if isinstance(obj, (set, frozenset)):
        members = sorted(repr(item) for item in obj)
        return f"{{type(obj).__module__}}.{{type(obj).__name__}}:{{members}}"
    # Enum members, constants, dataclass instances: type plus repr, which is
    # what actually changes if a lazy path returns a different object.
    return f"{{type(obj).__module__}}.{{type(obj).__name__}}:{{obj!r}}"

surface = {{}}
for name in packages:
    module = importlib.import_module(name)
    exported = getattr(module, "__all__", None)
    entry = {{
        "__all__": sorted(exported) if exported is not None else None,
        "dir": sorted(dir(module)),
    }}
    if exported is not None and name not in skip_resolve:
        entry["resolved"] = {{
            item: identify(getattr(module, item)) for item in sorted(exported)
        }}
    surface[name] = entry
print(json.dumps(surface))
'''


def measure_export_surface() -> dict[str, Any]:
    """Capture ``__all__``, ``dir()`` and per-name identity for every public package."""
    return _probe(
        _SURFACE_PROBE.format(
            packages=json.dumps(list(PUBLIC_PACKAGES)),
            skip=json.dumps(sorted(RESOLVE_SKIP)),
        )
    )


_RESOLUTION_PROBE = '''
import json

from phenotypic._core._pipeline_parts._serializable_pipeline import (
    SerializablePipeline,
)
from tests.migration._scenarios import discover_analyzers, discover_operations

names = set()
for classes in discover_operations().values():
    names |= {cls.__name__ for cls in classes}
names |= {cls.__name__ for cls in discover_analyzers()}

resolved = {}
for name in sorted(names):
    found = SerializablePipeline._find_class_in_phenotypic(name)
    resolved[name] = (
        None if found is None else f"{found.__module__}:{found.__qualname__}"
    )
print(json.dumps(resolved))
'''


def measure_class_resolution() -> dict[str, Any]:
    """Record what ``_find_class_in_phenotypic`` resolves every operation name to.

    This is the direct guard on the ``hasattr(phenotypic, class_name)`` branch
    in ``_serializable_pipeline.py``. Under PEP-562 laziness that call still
    works, but only while ``__getattr__`` raises a clean ``AttributeError`` for
    unknown names -- anything else (an ``ImportError`` from a missing optional
    dependency, say) silently breaks pipeline deserialization for every class
    that resolves through the fallback list.

    Operation names come from ``tests.migration._scenarios``, which discovers
    every concrete operation across the public subpackages.
    """
    return _probe(_RESOLUTION_PROBE)


def capture() -> dict[str, Any]:
    """Build the complete import-surface golden."""
    return {
        "schema_version": SCHEMA_VERSION,
        "exports": measure_export_surface(),
        "eager": {target: measure_eager_imports(target) for target in EAGER_TARGETS},
        "class_resolution": measure_class_resolution(),
    }


# --------------------------------------------------------------------------
# Static analysis
# --------------------------------------------------------------------------


def toplevel_import_modules(path: Path) -> set[str]:
    """Fully dotted modules imported at *module scope* by one file.

    Only module-body statements are walked, so ``if TYPE_CHECKING:`` blocks and
    in-function lazy imports are correctly excluded — that distinction is the
    whole point, and it is what makes "imported at runtime" precise. Same
    technique as ``tests/unit/viz/test_import_rules.py``, but keeping the full
    dotted name rather than only the root: some of what this branch defers is an
    internal package (``phenotypic.sdk_.reconnect``) whose root would be
    indistinguishable from any other ``phenotypic`` import.

    Relative imports are skipped: resolving them needs the importing file's
    package, and nothing this is used for needs them.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


def imports_at_module_scope(path: Path, target: str) -> bool:
    """Whether *path* imports *target* (or a submodule of it) at module scope."""
    return any(
        module == target or module.startswith(f"{target}.")
        for module in toplevel_import_modules(path)
    )


# --------------------------------------------------------------------------
# Golden I/O
# --------------------------------------------------------------------------


def load_golden() -> dict[str, Any]:
    """Read the committed import-surface golden."""
    return json.loads(IMPORT_SURFACE_GOLDEN.read_text(encoding="utf-8"))


def write_golden(payload: dict[str, Any]) -> None:
    """Write *payload* as the import-surface golden, pretty-printed for diffing."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    IMPORT_SURFACE_GOLDEN.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
