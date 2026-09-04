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

#: The ratchet, per import target. A library listed under a target must NOT be
#: in that target's ``sys.modules`` after it is imported. Each stage appends
#: what it evicted; nothing is ever removed.
#:
#: Per-target rather than one global list, because the two targets genuinely
#: differ and pretending otherwise would mean either a false claim or a silent
#: allowlist. ``polars`` is off the library path but remains eager for the CLI:
#: six modules under ``_cli/`` use it across 87 call sites in the measurement
#: layer, every real run (full / measure / recompile) reaches it almost
#: immediately, and deferring it there is a separate change with a different
#: risk profile from moving an import out of a method body.
FORBIDDEN_EAGER_BY_TARGET: dict[str, tuple[str, ...]] = {
    "phenotypic": (
        # Stage 1.
        "colour",
        "coverage",
        "h5py",
        "llvmlite",
        "numba",
        # Stage 2.
        "cv2",
        "plotly",
        "polars",
    ),
    "phenotypic.phenotypicCLI": (
        "colour",
        "coverage",
        "h5py",
        "llvmlite",
        "numba",
        "cv2",
        "plotly",
        # NOTE: `polars` is deliberately absent -- see above.
    ),
}

# A target absent from the mapping would forbid nothing while the test still
# reported green, so the two must agree by construction.
assert set(EAGER_TARGETS) == set(FORBIDDEN_EAGER_BY_TARGET), (
    "every EAGER_TARGETS entry needs a FORBIDDEN_EAGER_BY_TARGET entry: "
    f"{set(EAGER_TARGETS) ^ set(FORBIDDEN_EAGER_BY_TARGET)}"
)

#: Every library any target forbids, for the static scan and the allowlist.
FORBIDDEN_EAGER: tuple[str, ...] = tuple(
    sorted({lib for libs in FORBIDDEN_EAGER_BY_TARGET.values() for lib in libs})
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
        # Figure/report builders that are NOT on either eager path. Verified,
        # not assumed: `_detect_modes_plotter` is reached only from inside
        # `PlotDetectModes.inspect`/`.report` (`plotting/_image_plots.py:128,137`),
        # so it stays out of `sys.modules` after `import phenotypic` -- unlike
        # its sibling `_diagnostics_plotter`, which the `@_diagnostics_figure`
        # decorator pulls in at class-body time and which therefore did need its
        # plotly imports deferred.
        ("_core/_image_parts/plot_accessor/_detect_modes_plotter.py", "plotly"),
        ("correction/_color_correction/_color_correction_report.py", "plotly"),
        ("grid/_grid_fit_report.py", "plotly"),
    }
)

#: ``(path prefix, library)`` pairs exempt from the module-scope scan.
#:
#: A whole subsystem, not a file: ``gui/`` is an optional extra that neither
#: import target reaches, and ``_cli/`` is where polars actually lives. Listing
#: 30 files individually would be noise that nobody re-reads; naming the
#: subsystem and the reason is the thing a future reader can check.
MODULE_SCOPE_ALLOWED_PREFIXES: tuple[tuple[str, str], ...] = (
    # The GUI is behind the `gui` extra and is imported by neither target.
    ("gui/", "polars"),
    ("gui/", "plotly"),
    ("gui/", "cv2"),
    # polars is the CLI measurement layer's working dependency; the CLI target
    # does not forbid it. See FORBIDDEN_EAGER_BY_TARGET.
    ("_cli/", "polars"),
)

# NOTE: only genuine directory prefixes belong above. Matching is
# `relative.startswith(prefix)`, so a full filename here would silently exempt
# every file whose name extends it -- `grid/_grid_fit_report.py` would also
# cover a future `grid/_grid_fit_report_v2.py`. Exact files go in
# MODULE_SCOPE_IMPORT_ALLOWLIST, which is matched by equality.

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
    # Stage 2 -- cv2.
    ("enhance/_subtract_opening.py", "cv2"),
    ("enhance/_flatten_illumination.py", "cv2"),
    ("refine/_extract_colony_core.py", "cv2"),
    # Stage 2 -- polars.
    ("util/_measurement_outputs.py", "polars"),
    # Stage 2 -- plotly.
    ("sdk_/viz/figures/_theme.py", "plotly"),
    ("_core/_image_parts/plot_accessor/_diagnostics_plotter.py", "plotly"),
    (
        "_core/_image_parts/accessor_abstracts/_image_accessor_base_parents/"
        "_accessor_dash_handler.py",
        "plotly",
    ),
    ("analysis/qc/_expected_vs_detected.py", "plotly"),
    ("analysis/qc/_grid_occupancy.py", "plotly"),
    ("analysis/qc/_replicate_agreement.py", "plotly"),
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
        _EAGER_PROBE.format(
            target=target,
            forbidden=json.dumps(list(FORBIDDEN_EAGER_BY_TARGET.get(target, ()))),
        )
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
    """Fully dotted modules one file imports *at import time*.

    "At import time" is not the same as "at top level". An import executes when
    the module is loaded unless it sits inside a function body, so this descends
    through ``try``/``except``, ``if``/``else``, ``with`` and class bodies — and
    stops at ``def``.

    That distinction was not academic. The first version walked only
    ``tree.body`` and therefore could not see::

        try:
            import plotly.express as px
            PLOTLY_AVAILABLE = True
        except ImportError:
            PLOTLY_AVAILABLE = False

    which is a module-scope import of plotly by any measure — and was on the
    startup path — yet a scan of every source file reported the tree clean.

    ``if TYPE_CHECKING:`` blocks are excluded because they genuinely do not
    execute at runtime, which is the whole point of writing one.

    Relative imports are skipped: resolving them needs the importing file's
    package, and nothing this is used for needs them.
    """

    def _is_type_checking_guard(test: ast.expr) -> bool:
        """Whether an ``if`` test is the ``TYPE_CHECKING`` guard."""
        if isinstance(test, ast.Name):
            return test.id == "TYPE_CHECKING"
        if isinstance(test, ast.Attribute):
            return test.attr == "TYPE_CHECKING"
        return False

    modules: set[str] = set()

    def _visit(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    modules.add(node.module)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue  # a function body runs when called, not when imported
            elif isinstance(node, ast.If):
                if not _is_type_checking_guard(node.test):
                    _visit(node.body)
                _visit(node.orelse)
            elif isinstance(node, ast.Try):
                _visit(node.body)
                for handler in node.handlers:
                    _visit(handler.body)
                _visit(node.orelse)
                _visit(node.finalbody)
            elif isinstance(node, (ast.With, ast.AsyncWith, ast.ClassDef, ast.For, ast.While)):
                _visit(node.body)
                _visit(getattr(node, "orelse", []))

    _visit(ast.parse(path.read_text(encoding="utf-8")).body)
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
