# Phase 1 gate review — lazy startup (Tasks 1–4)

**Reviewer:** independent implementation/test reviewer (opus), read-only over the branch
`refactor/private-gui` at `31c9e3c1c`.
**Scope:** the diff package `.superpowers/sdd/plan/review-d105397d0..31c9e3c1c.diff` (9 commits),
read as one change, against `docs/superpowers/specs/2026-09-11-lazy-startup/spec.md`
(with Amendment A P1–P14) and `docs/superpowers/plans/2026-09-11-lazy-startup/plan.md`.
Tiers 2 and 3, the CLI help path and the GUI launcher belong to Phase 2 and are noted, not faulted.

---

## Verdict

**Ready for Phase 2.**

The implementation is correct as written. I found **no Critical issues and no live defect** in
the 34 deferral sites or the three lazy `__init__`s: every deferred name is imported before every
runtime use of it, nothing deferred survives in a decorator, a default argument, a class body, a
pydantic field annotation or a runtime annotation, and the three lazy maps are complete against
their `__all__`s and resolve.

What I would not merge the branch on is the **guard set's forward-looking coverage**. Three gaps
(I1–I3 below) mean the property the spec claims — *"A deterministic guard keeps it that way"* —
is only partly held. In particular, the stated purpose of Task 4's two detector rows ("deferring
the `sdk_.reconnect` import at these two importers is what keeps numba out of
`import phenotypic.detect`") has **no runtime guard at all**, and a plain `import numba` re-added
at the top of either detector passes every test in the phase. None of the three blocks Phase 2 —
Phase 2 touches different files — but all three should land before the branch merges.

---

## Strengths

- **The probe helper is genuinely falsifiable.** I mutation-tested `run_startup_probe` directly:
  a failing import, an unserialisable `report`, and an early `sys.exit(0)` with no report each
  raise `AssertionError` with the child's stderr tail attached. The `_REPORT_MARKER` change from
  `startswith` to substring match (`tests/_startup_probe.py:56`) is a real fix, not cosmetic — a C
  extension writing to fd 1 without a newline would otherwise have swallowed the report and turned
  every guard into a hard failure.
- **The Task 1 false green is properly closed.** `test_load_runtime_dependencies_imports_every_deferred_module`
  (`tests/unit/ci/test_startup_imports.py:19`) now asserts `report["before"] == []` *and*
  `report["missing"] == []`. The `before` half is the control that makes it able to fail: without
  it, the test passed with `load_runtime_dependencies` deleted, because the eager package had
  already imported all eight. The inline comment says exactly that. This is the model the rest of
  the suite should follow.
- **The point-of-use rule was applied with unusual discipline.** I re-derived it independently with
  four AST passes over all 34 `DEFERRED_SITES` modules (see *Checks run*). Zero uncovered runtime
  loads, zero decorator/default/class-body uses, zero loads sequenced before their own local import
  (which would be `UnboundLocalError`, not `NameError` — the harder failure to spot), zero local
  imports buried inside an `if`/`try` rather than at function top level, and zero runtime
  annotations needing a deferred name in the three modules that lack
  `from __future__ import annotations`.
- **The lazy maps are complete and self-consistent.** `sdk_._LAZY_ATTRS` has exactly the 31 names
  removed from the eager blocks (1 module + 1 + 12 + 11 + 6); all 31 are in `sdk_.__all__` and all
  resolve. `abc_._LAZY_ATTRS`'s three names are all in `abc_.__all__`. Every name in
  `phenotypic.__all__` resolves, `phenotypic.plotting` resolves though it is deliberately outside
  `__all__`, and an unknown name raises `AttributeError`, which is what
  `SerializablePipeline._find_class_in_phenotypic`
  (`src/phenotypic/_core/_pipeline_parts/_serializable_pipeline.py:649`) needs to fall through to
  its submodule list.
- **`import phenotypic` is 71 modules**, matching the task report's claim, with none of
  `HEAVY_STARTUP_MODULES` present.
- **The sweep floor is meaningful.** 75 packages are discovered against a floor of 70; 16 of them
  are `_gui` packages, so a discovery change that silently dropped the GUI tree would land at 59
  and fail `test_package_discovery_found_the_tree`.
- **The guards are wired into CI.** `tests/unit/ci/` is already a path of the `foundation-schema`
  shard in `.github/pytest-shards.json`, and `tests/unit/ci/test_pytest_shard_manifest.py` keeps it
  that way. The spec's "the guards live in existing sharded directories" claim holds.
- **The one changed existing test is changed for the right reason.**
  `tests/unit/detect/test_filamentous_fungi_gwdt_seam.py` moves its monkeypatch target from
  `fungi_module` to `phenotypic.sdk_.reconnect`, and its docstring explains *why* — the local
  import re-resolves from the source module's namespace on every call. That is the correct target,
  and it is documented rather than silently edited.

---

## Critical

None.

---

## Important

### I1. No guard keeps a deferred library out of the operation subpackage that deferred it

**Evidence.** Plan Task 4: *"The two detectors: the numba kernel modules are **not** edited (D6).
Deferring the `sdk_.reconnect` import at these two importers is what keeps numba out of
`import phenotypic.detect`."* That sentence names the property the work exists to create. Nothing
asserts it.

- Tier 1 (`tests/unit/ci/test_startup_imports.py:76`) probes `import phenotypic`, which no longer
  imports `detect` at all.
- Tier 4 (`:119`) probes `from phenotypic import Image`, which does not import `detect`, `enhance`,
  `refine`, `correction` or `measure` either.
- Tier 3 (Phase 2) probes the composed hub, but the builder's `OperationRegistry.discover()` runs
  on the *first `/builder/` visit*, after tier 3's absence assertion has already been taken.
- The sweep (`:110`) imports `phenotypic.detect` first in a fresh interpreter but asserts only
  `report["imported"] is True` — it never looks at `sys.modules`.
- The per-site checker's keys for `detect/_filamentous_fungi_detector.py` are `ReconnectConfig`
  and the seven `sdk_.reconnect` functions. `numba` is not a key there, so
  `test_deferred_names_are_not_imported_at_module_level` does not watch it.
- `tests/unit/viz/test_import_rules.py` is the only other static import-rule test in the tree and
  covers `abc_/plotting/_pht_plot.py` and `sdk_/viz/figures/_theme.py` only.

**Concrete failure scenario.** A contributor adds `import numba` (or `import cv2`, `import bm3d`,
`import mahotas`) at module level in `src/phenotypic/detect/_filamentous_fungi_detector.py` —
say, to type a kernel signature. `tests/unit/ci` stays green (160 passed), the sweep stays green,
tiers 1–4 stay green, and `import phenotypic.detect` silently costs 0.10 s again. The same holds
for `enhance/_subtract_opening.py` re-acquiring `import cv2`, for
`measure/_measure_texture.py` re-acquiring `import mahotas`, and for any *new* operation module,
which is not in `DEFERRED_SITES` at all.

**Proposed fix** (a fifth tier, cheap, subprocess-based, same probe helper):

```python
#: Operation subpackage -> libraries that importing it must not load.
SUBPACKAGE_FORBIDDEN = {
    "phenotypic.detect":     ("numba",),
    "phenotypic.enhance":    ("cv2", "bm3d"),
    "phenotypic.refine":     ("cv2",),
    "phenotypic.measure":    ("mahotas",),
    "phenotypic.correction": ("bm3d",),
    "phenotypic.util":       ("colour",),
}


@pytest.mark.parametrize("package, forbidden", sorted(SUBPACKAGE_FORBIDDEN.items()))
def test_operation_subpackage_loads_no_deferred_library(package, forbidden) -> None:
    """Tier 5: the point of deferring at the importer is that the subpackage stays light."""
    report = run_startup_probe(
        "import importlib\n"
        f"importlib.import_module({package!r})\n"
        f"watched = {sorted(forbidden)!r}\n"
        "report = {'loaded': [m for m in watched if m in sys.modules],\n"
        f"          'control': {package!r} in sys.modules}}\n"
    )
    assert report["control"] is True
    assert report["loaded"] == []
```

Note `correction` deliberately still loads `colour` eagerly (spec §2, `_color_checker_profile.py`
and `_helpers.py` are on the unchanged list), so `colour` must *not* be in its row — which is
exactly the kind of decision the table should record.

### I2. The per-site checker is one-directional: a *missing* local import is unguarded

**Evidence.** `tests/unit/ci/test_deferred_imports.py:202`
(`test_each_user_of_a_deferred_name_imports_it_locally`) iterates the hand-written
`DEFERRED_SITES` table and asserts that each *listed* function imports the name. Nothing walks the
other way: no test asserts that every runtime *use* of `plt`/`go`/`colour`/`cv2`/… in those modules
lies inside a function that imports it.

The table carries ~90 `(name, function)` pairs over 34 modules, several of them long — the
`_diagnostics_plotter.py` row alone lists 7 `plt` functions, 13 `go` functions, 5 `NAVY` and 3
`OKABE_ITO`. Getting one wrong is a `NameError` at plot time.

**Concrete failure scenario.** A new `DiagnosticsPlotter.fig_edge_sharpness` uses `go.Figure(...)`
and the author forgets the local `import plotly.graph_objects as go` (module level no longer has
it). `test_deferred_names_are_not_imported_at_module_level` passes (nothing leaked to module
level), `test_each_user_of_a_deferred_name_imports_it_locally` passes (the new function is not in
the table), tiers 1 and 4 pass. The `NameError` reaches the first user who asks for that figure.

I verified the tree is clean today, so this is a coverage gap, not a live defect.

**Proposed fix** — one more parametrized test, reusing the helpers already in the file:

```python
@pytest.mark.parametrize("relative_path", sorted(DEFERRED_SITES))
def test_no_runtime_use_of_a_deferred_name_escapes_its_local_import(relative_path: str) -> None:
    """Every runtime load must sit inside a function that imports the name first."""
    tree = _parse(relative_path)
    parents = {c: p for p in ast.walk(tree) for c in ast.iter_child_nodes(p)}
    annotated = _annotation_node_ids(tree)   # return/param/AnnAssign annotations
    escaped = []
    for name in DEFERRED_SITES[relative_path]:
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Name) and node.id == name
                    and isinstance(node.ctx, ast.Load)):
                continue
            if id(node) in annotated:        # string annotations; every such module
                continue                     # carries `from __future__ import annotations`
            cur, covered = node, False
            while cur in parents:
                cur = parents[cur]
                if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                        name in _locally_imported_names(cur):
                    covered = True
                    break
            if not covered:
                escaped.append(f"{name} at line {node.lineno}")
    assert escaped == [], f"{relative_path}: {escaped}"
```

Add the `from __future__ import annotations` precondition as an explicit assertion in the same
test for any module with a `TYPE_CHECKING`-only key, so the "annotations are strings" premise the
skip relies on is pinned rather than assumed.

### I3. The watched sets are unpinned, so a guard can be retired without anything going red

**Evidence.** Tier 1 (`test_startup_imports.py:76`) and tier 4 (`:119`) interpolate
`sorted(HEAVY_STARTUP_MODULES)` / `sorted(DEFERRED_RUNTIME_MODULES)` into the probe body at
collection time and then assert `report["loaded"] == []`. Nothing asserts what those tuples
contain. `test_every_deferred_module_is_watched_at_startup` (`:38`) asserts only the subset
relation `DEFERRED ⊆ HEAVY ∪ {"matplotlib.pyplot"}` — which is satisfied by the empty set.

**Concrete failure scenario.** A future change makes tier 1 fail on `colour`. The cheapest way to
get green is to delete `"colour"` from `HEAVY_STARTUP_MODULES` in `_startup_perf.py:42`. Tier 1
then passes, `test_every_deferred_module_is_watched_at_startup` still passes (removing a name from
`HEAVY` while leaving it in `DEFERRED` *would* break the subset — but removing it from both does
not), and the 0.39 s regression ships. In the limit, emptying both tuples leaves every tier
green and every assertion vacuous.

**Proposed fix** — pin the contents where the constants are asserted:

```python
def test_the_watched_sets_are_the_ones_the_spec_names() -> None:
    """A guard asserted against a constant needs the constant pinned, or it can be retired silently."""
    assert set(HEAVY_STARTUP_MODULES) == {
        "bm3d", "colour", "cv2", "dash", "h5py", "mahotas", "matplotlib",
        "numba", "pandas", "plotly", "polars", "pyarrow", "scipy", "skimage",
    }
    assert set(DEFERRED_RUNTIME_MODULES) == {
        "bm3d", "colour", "cv2", "h5py", "mahotas", "matplotlib.pyplot", "numba", "plotly",
    }
```

This is the spec's tier-1/tier-4 table (spec §Tests) written down where it can fail. Changing
either set then requires changing the spec's own list in the same commit, which is the intent.

---

## Minor

### M1. `_plotly_imshow` imports plotly before the friendly guard runs

`src/phenotypic/_core/_image_parts/accessor_abstracts/_image_accessor_base_parents/_accessor_dash_handler.py:117-119`
now reads `import plotly.express as px` *then* `AccessorDashHandler._require_plotly()`. With plotly
genuinely absent, the raw `ModuleNotFoundError` fires first and the curated message
("plotly is required for interactive visualization… `pip install phenotypic[gui]`") never runs.

Impact is small: `plotly>=6.0.0` is a hard dependency (`pyproject.toml:68`), and the callers that
matter (`_single_channel_accessor.py:152`, `_multichannel_accessor.py:250`) check
`PLOTLY_AVAILABLE` before reaching here, which is why `tests/unit/core/test_plotly_fallback.py`
still passes with `PLOTLY_AVAILABLE` patched to `False`. Fix by putting `_require_plotly()` first:

```python
AccessorDashHandler._require_plotly()

import plotly.express as px
```

The plan's "first statement after the docstring" rule is what produced this ordering; it is
plan-mandated, and this is the one place in the phase where that rule is slightly wrong.

### M2. `configure_docs_build_plotly_renderer` drops the old `except ImportError`

`src/phenotypic/_startup_perf.py:134` does a bare `import plotly.io as pio`. The code it replaced
(`_accessor_dash_handler.py`, old lines 14–35) sat inside `try: … except ImportError:`. Under
`PHENOTYPIC_DOCS_BUILD=1` in an environment without plotly, `import phenotypic` now raises instead
of degrading. Again low impact because plotly is required; worth a `try`/`except ImportError:
return False` if the `PLOTLY_AVAILABLE` fallback is meant to keep meaning anything.

### M3. `PLOTLY_AVAILABLE` is now `find_spec`, which answers a slightly different question

`find_spec("plotly") is not None` says "installed and findable", where the old `try: import
plotly.express` said "imports cleanly". A plotly whose import raises (a broken `numpy` ABI, a
partially-installed wheel) now reports `True` and fails later at the first figure. That is the
right trade for startup, and `load_runtime_dependencies()` covers the pipeline paths — but the
interactive/notebook path (spec B3 "Interactive use") now surfaces it at the plot rather than at
the accessor, which is what B3 says, so this is a note rather than a fault.

### M4. The sweep covers packages and ten named entry modules, not top-level modules

`_package_modules()` (`test_startup_imports.py:63`) discovers only directories with `__init__.py`.
`src/phenotypic/settings.py` is a plain module and a member of `_LAZY_SUBPACKAGES`
(`src/phenotypic/__init__.py:44`); `src/phenotypic/_startup_perf.py` is the module every entry
point imports first. Neither is in `IMPORT_FIRST_ENTRY_MODULES`. Both import first cleanly today
(I checked `import phenotypic` reaches `_startup_perf` before anything else); adding them to the
tuple is two lines and closes the gap the spec's "Known risks" section already flags as
"cycles in leaf-first imports the sweep does not cover".

### M5. `ImageGridHandler.__init__` imports `phenotypic.grid` on every construction

`src/phenotypic/_core/_image_parts/_grid_image_handler.py:88` puts
`from phenotypic.grid import CenteredAutoGridFinder` at the top of `__init__`, but the name is used
only in the `elif grid_finder is None` branch (`:96`). Every `GridImage(...)` therefore pays a
`sys.modules` lookup, and the first one pays the whole `phenotypic.grid` import even when the
caller supplied a finder. Correct and plan-mandated; moving it into the branch would be strictly
better and costs nothing. Same shape, lower stakes, at `:392` for `MeasureBounds`.

### M6. The sweep adds 85 subprocesses to one shard

`PACKAGE_MODULES` (75) + `IMPORT_FIRST_ENTRY_MODULES` (10) parametrize
`test_module_imports_first_in_a_fresh_interpreter`, all inside the `foundation-schema` shard. The
plan measured ~95 s serial / ~24 s on 4 workers pre-change. Sixteen of the 75 are `_gui` packages,
which pull dash. This is a deliberate cost, recorded here so the Phase-1 lane timing is not
mistaken for a regression elsewhere.

### M7. Known-open Task 1 review minors 6 and 7 — my judgement

- **Minor 6 (the probe discards stderr on success).** *Now worth fixing, cheaply.* Tier 1 and
  tier 4 assert on `sys.modules` only; a library that emits a `RuntimeWarning` or a C-extension
  banner on import still passes silently. More to the point, the mahotas `SyntaxWarning` filter
  moved from module scope into `_mahotas()`/`load_runtime_dependencies()`
  (`_measure_texture.py:20`, `_startup_perf.py:112`), and a probe that returned stderr would let a
  future guard assert the filter still works outside pytest — where `pyproject.toml`'s
  `filterwarnings = ["ignore::SyntaxWarning:mahotas"]` does not apply. Returning
  `{"report": …, "stderr": result.stderr}` is a two-line change.
- **Minor 7 (the child inherits `PYTHONPATH`/`PYTHONWARNINGS`).** *Still not worth fixing.* The
  helper already scrubs the two variables that would change the answer
  (`PHENOTYPIC_DOCS_BUILD`, `PYTEST_CURRENT_TEST`) and pins `QT_QPA_PLATFORM`/`MPLBACKEND`. A
  `PYTHONPATH` that shadowed `phenotypic` would break the positive controls, not hide a failure;
  a `PYTHONWARNINGS` that turned warnings into errors would make the probe exit non-zero and the
  guard red. Both fail loudly. Leave it.

---

## Spec conformance

| Item | Status |
|---|---|
| Tier 1 (`import phenotypic` loads no `HEAVY_STARTUP_MODULES`) | Present and falsifiable. Verified: 71 modules, none heavy. |
| Tier 4 (`from phenotypic import Image` loads no `DEFERRED_RUNTIME_MODULES`) | Present, closed by Task 4 as the plan sequenced. Positive control `phenotypic._core._image` is correct. |
| Import-order sweep, floor ≥ 70 | Present; 75 discovered, floor meaningful. Probe raises on a failing import (mutation-tested). |
| Per-site checker | Present; one-directional (I2). |
| `HEAVY_STARTUP_MODULES` / `DEFERRED_RUNTIME_MODULES` constants | Present, match the spec's lists exactly, but unpinned (I3). |
| `load_runtime_dependencies()` | Present; imports all eight; guard has a real control. |
| Docs-build plotly renderer moved to `_startup_perf` (Amendment A P4) | Done; still runs at `import phenotypic`, only under `PHENOTYPIC_DOCS_BUILD`; guarded by `test_docs_build_still_selects_the_notebook_connected_renderer`. |
| `# noqa: E402` markers (Amendment A P14) | Present on all eager imports in `abc_/__init__.py` (20) and `sdk_/__init__.py` (13). Controller reports ruff matches baseline exactly. |
| P8 (`__getattr__` defined before the eager imports) | Honoured in all three `__init__`s. |
| P12 (`abc_` lazy re-export is the cycle fix; the two `_grid_image_handler` moves are weight moves) | Consistent with the code; the two moves are in the checker table, per P12's "M2 targets the deferral checker, not the sweep". |
| D6 (kernel modules untouched) | `_tensor_voting.py` and `_dijkstra_kernels.py` are not in the diff. |
| "No numeric change" | `sRGB_D50.whitepoint` mutation at `_xyz_conversion.py:38` still binds the same module-level object through the function-scope import; `colourspace.py` and `hdf_.py` are not in the diff. |
| Tiers 2, 3, CLI, GUI launcher | Phase 2. Not assessed. |

---

## Checks run

| Risk | Check | Result |
|---|---|---|
| A deferred name still reachable at runtime outside its importing function | AST pass over all 34 `DEFERRED_SITES` modules: every `ast.Name` load of every key, walking the enclosing scope chain for a local import or assignment; annotations excluded only where `from __future__ import annotations` is present | **0 uncovered loads** |
| A deferred name in a decorator, a default argument, or a class body (evaluated at definition time, outside any function) | AST pass over every `FunctionDef`/`ClassDef` `decorator_list`, `args.defaults`, `args.kw_defaults`, and class-body `Assign`/`AnnAssign` values | **0 hits** |
| A deferred name in a pydantic **field** annotation (pydantic resolves these at class build, so `TYPE_CHECKING`-only would raise) | AST pass over every class-level `AnnAssign` annotation in the 34 modules | **0 hits**; every `TYPE_CHECKING`-only key (`Figure`, `BM3DStages`, `go`, `Axes`, `Colormap`, `Normalize`, `PathCollection`, `Quiver`, `ReconnectConfig`) appears only in method signatures |
| A runtime annotation needing a deferred name in a module without `from __future__ import annotations` (NameError at class creation) | Per-module `__future__` detection + annotation-node scan | **0 hits**; the three modules lacking it (`_chromaticity_xy_accessor.py`, `_cielab_accessor.py`, `_xyz_d65_accessor.py`) have no `TYPE_CHECKING` keys, as the plan states |
| A local import sequenced *after* a use in the same function (`UnboundLocalError`, not `NameError`) | AST pass comparing each runtime load's line to the first local import line within the same function scope | **0 hits** |
| A local import buried inside an `if`/`try`/loop rather than at function top level | AST parent check on every local import of a deferred key | **0 hits** |
| Every removed module-level import is tracked by the checker | Extracted every `-` import line for `src/**` from the diff package and matched against `DEFERRED_SITES` keys | **all accounted for**; the only untracked removals are the three lazy `__init__` rewrites and `typing` re-orderings |
| `sdk_` lazy map incomplete against `__all__` / real importers | `[n for n in sdk_._LAZY_ATTRS if n not in sdk_.__all__]`; all 31 resolved via `getattr` | **[] / 31 resolve** |
| `abc_` lazy map incomplete | same for its three names | **[]** |
| `phenotypic` classes, subpackages, `plotting`, `__dir__`, unknown → `AttributeError` | `all(hasattr(phenotypic, n) for n in __all__)`; `phenotypic.plotting.__name__`; `dir()` membership; read `_find_class_in_phenotypic` | **all pass**; `hasattr` fall-through for `_find_class_in_phenotypic` preserved |
| `import phenotypic` still heavy | module count in a fresh interpreter | **71 modules** |
| Sweep floor is vacuous | counted discovered packages and the `_gui` share | **75 total, 16 `_gui`** — floor 70 is load-bearing |
| The probe helper cannot fail | ran `run_startup_probe` with (a) a failing import, (b) an unserialisable `report`, (c) `sys.exit(0)` before binding `report` | **all three raise `AssertionError`** |
| The guards never run in CI | `.github/pytest-shards.json` + `tests/unit/ci/test_pytest_shard_manifest.py` | `tests/unit/ci/` is in the `foundation-schema` shard; manifest test enforces it |
| `from tests._startup_probe import …` works by accident | root `conftest.py` puts rootdir on `sys.path`; 100 existing test files already use `from tests.…` | **established convention, no finding** |
| `import phenotypic.detect` still free of numba | searched every test for a `sys.modules` assertion over an operation subpackage | **none exists** → I1 |
| Deferred-name *users* outside the table | read `test_each_user_of_a_deferred_name_imports_it_locally` | one-directional → I2 |
| Watched sets pinned | read tier 1 / tier 4 / `test_every_deferred_module_is_watched_at_startup` | contents unpinned → I3 |
| mahotas `SyntaxWarning` filter lost | grepped every `import mahotas` in `src/` and `tests/` | **one importer** (`_measure_texture.py:27`, inside `_mahotas()` after the filter); `load_runtime_dependencies` installs the same filter before its loop; pytest covers the suite via `filterwarnings` |
| `PLOTLY_AVAILABLE` no longer patchable | traced `tests/unit/core/test_plotly_fallback.py:87` against `_require_plotly`'s module-global read | **still patchable**; `sdk_/_plotly_helpers.py:9` keeps its pre-existing stale copy |
| Monkeypatch targets moved silently | read the one changed existing test | target move is correct and documented in its docstring |
| plotly/bm3d/etc. optional vs required | `pyproject.toml:38-82` | all eight deferred libraries are hard dependencies, which bounds M1–M3 |

---

## Recommended changes, in priority order

1. **I1** — add the tier-5 subpackage guard (`SUBPACKAGE_FORBIDDEN`). This is the one missing guard
   whose absence lets the largest part of Task 4 silently revert.
2. **I3** — pin `HEAVY_STARTUP_MODULES` and `DEFERRED_RUNTIME_MODULES` by value. Three lines.
3. **I2** — add the reverse direction of the per-site checker.
4. **M1** — move `_require_plotly()` above the `import plotly.express` in `_plotly_imshow`.
5. **M4** — add `phenotypic.settings` and `phenotypic._startup_perf` to `IMPORT_FIRST_ENTRY_MODULES`.
6. **M7/minor 6** — have `run_startup_probe` return the child's stderr alongside the report.
7. **M2, M5** — optional polish.

None of 1–7 blocks Task 5. All of 1–3 should land before the branch merges.
