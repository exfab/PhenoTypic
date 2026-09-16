# Final whole-change review — lazy startup (Tasks 1–7)

**Reviewer:** Task 8 Step 1 (independent implementation + test review)
**Range:** `f0f9a544..c60d5483` — 21 commits, 59 files, +2058/−310
**Worktree:** `/bigdata/exfab/anguy344/PhenoTypic/.worktrees/private-gui`
**Authority:** spec `2026-09-11-lazy-startup/spec.md` (Amendment B supersedes A/P1), plan, Phase 1 & 2
gate reports, `mutation-proofs.md`, `startup-measurements.md`, `.superpowers/sdd/plan/progress.md`.

**Verdict: Ready to merge** — no Critical findings. Three Important findings; none blocks the
merge, all three are latent-hazard or guard-coverage issues over correct production code, and one
(I-2) was already raised and ruled in-spec at the Phase 2 gate.

**I-3 is this change's fourth false green**, confirmed by probe: AC1's public-name resolution
assertion is a tautology, and the one other guard that might have covered it structurally cannot.
**I-1 is a confirmed stale-binding leak** with zero live instances in the suite today — measured
across all three patch forms, not reasoned. Neither ships a defect; both are traps for the next
person, and both have a cheap, named remedy below.

---

## Scope of what I actually checked

Everything below was derived from the tree, not from the reports. Where I quote a report's number
(AC8, AC9) I say so and attribute it rather than re-deriving it.

Static analyses I ran over the worktree (read-only, `/usr/bin/python3.12` + `ast`):

1. Every runtime module-level import of the eight deferred libraries across all of
   `src/phenotypic` — **27 sites, all accounted for** by the spec's unchanged list (A/P3), the
   kernel rule (D6), `_cli_process_single.py`'s `matplotlib.use("Agg")`, `sdk_/colourspace.py`,
   `sdk_/hdf_.py`, and `_gui/` (outside the AST inventory by construction). No leak.
2. Every use of the 28 deferred `phenotypicCLI` names, mapped to its enclosing scope, plus a
   separate pass over module-execution-time positions (top-level `def` annotations, defaults,
   decorators, class bodies, module statements) — **zero module-exec-time uses**, and every runtime
   use sits in `phenotypic_cli`, `_migrate_legacy_success_evidence`, `_handle_recompile_slurm`, or
   inside `_regenerate_missing_overlays` (via its nested `_render_one`), each of which calls
   `_load_cli_runtime()` first. No `NameError` route.
3. Every name in `_CLI_RUNTIME_IMPORTS` resolved against its source module — **all 28 exist**, so
   `__getattr__`'s `return globals()[name]` cannot raise `KeyError`.
4. Every `__all__` list in every changed `src/` file, base vs. HEAD — **one change only**,
   `_startup_perf.__all__`, additive and on a private module (AC11, below).
5. Every `__all__` entry of the four lazy packages against its eager bindings ∪ lazy map —
   **0 unreachable** in all four (`phenotypic` 18, `sdk_` 324, `abc_` 28, `_gui.shell` 5).
6. Every `(module, name, function)` row of `DEFERRED_SITES` checked for *import-before-first-use*
   with annotations excluded — **all 48 sites correctly ordered** (see M-a for why nothing guards
   this).
7. Module-level `get_registry()` calls anywhere in `src/` — **none**, which is the premise
   `get_registry`'s property-3 docstring rests on.

---

## AC1–AC11 conformance

| AC | Verdict | Guard that establishes it |
|---|---|---|
| **1** Tier 1: `import phenotypic` loads no `HEAVY_STARTUP_MODULES`; public names resolve; `__all__ ⊆ dir()` | **Met in fact, unguarded in half** | `tests/unit/ci/test_startup_imports.py:119` (absence + `_startup_perf`/`__version__` control) is sound, and all 18 `__all__` entries do resolve today. But the *resolution* half at `:133` is only genuine for `Image`, `ImagePipeline`, `detect`; the `__all__ ⊆ dir()` assertion at `:141` is **confirmed by probe to be incapable of failing** — see **I-3** |
| **2** Tier 2: both `--help` paths exit 0, load nothing heavy; `--detect-mode` unchanged | **Met** | `:229` (CLI, via `runpy`), `:312` (GUI, via `_launcher.main(["--help"])` per A/P7), `:253` drift test. Phase 2 M5 is fixed: the RHS is now `list(available_modes())`, not `sorted(...)`, so the assertion is sensitive to the registry's own ordering. `sorted(get_args(DetectMode))` and `available_modes()` produce byte-identical choice lists (both plain `sorted()` over the same 11 strings) |
| **3** Tier 3: hub loads none of its forbidden set; first `/builder/` returns 200 and builds it; run console + observer still eager | **Met** | `tests/unit/gui/shell/test_hub_startup_imports.py:56` (`loaded_before == []`, `detect_before is False`, `status == 200`, `detect_after is True`). Phase 2 I2 is fixed: `HUB_WATCHED_MODULES` is pinned by value and asserted **disjoint** from the allow-list at `:33`, so an allow-list entry can no longer cancel its own watch. Run console / observer eagerness is not covered by *this* test (it passes `start_slurm_observer=False`); `_app.py:567` `_tick("run console")` is unchanged and `tests/integration/gui/test_scheduler_startup_wiring.py` is the guard — orchestrator's lane |
| **4** Tier 4: `from phenotypic import Image` loads no `DEFERRED_RUNTIME_MODULES` | **Met** | `:162`, with `phenotypic._core._image` as the positive control |
| **5** Sweep: every package (≥70) and every named entry module imports first | **Met** | `:151` parametrized over 75 discovered packages + 12 entry modules = 87 cases; floor `>= 70` pinned at `:146`. The probe fails on a non-zero child exit, so `report = {'imported': True}` is not a hard-coded pass |
| **6** Preload: CLI body + four worker entry modules call `load_runtime_dependencies()` before any image work | **Met for the enumerated set** | `tests/unit/cli/test_cli_runtime_preload.py:37` (abort-before-output, with `not output_dir.exists()` as the control) and `:57` (Phase 2 I4 fix — **position**, not presence: statement 0 for three workers, immediately post-`parse_args` for the staged worker, and an explicit `pytest.fail` for an unclassified new worker). I independently confirmed the set is complete *today* across all twelve `_cli/*` entry modules. Two gaps: the set is a hand-maintained tuple (no mechanical derivation), and `phenotypic-tune` is outside it — **I-2** |
| **7** Mutations M1–M6 each redden their guard | **Met, exceeded** | `mutation-proofs.md`: M1–M7, **7/7 killed, 7/7 restored byte-identically**, verified against the orchestrator's own independent `sha256sum` snapshot. M3's kill is honestly recorded as a collection-time `ImportError` rather than a named assertion, which is the right way to record it |
| **8** No regression (lanes, e2e, Sphinx, mypy/ruff) | **Orchestrator's — taken as given** | Phase 2 gate recorded 6228 tests / 0 failed on 16 shards and 22 passed / 86 attributed skips on the PLAYWRIGHT lanes. The final full-lane, e2e, Sphinx and mypy/ruff set-diffs are running separately |
| **9** Before/after table committed and quoted | **Met** | `startup-measurements.md`. The pair was correctly **re-measured on one machine** (HPCC `i24`) rather than pairing the committed macOS `before` with an HPCC `after`; the macOS table is kept as history. The `get_registry()`-fix caveat on the two hub rows is disclosed rather than buried |
| **10** Documentation lands in the same change | **Met** | Root `CLAUDE.md` Gotchas (new lazy-entry-points block, citing all three guard files), `sdk_/CLAUDE.md` ("Lazy re-exports"), `_gui/CLAUDE.md` (builder-on-first-request gotcha), `_gui/FEATURES.md` ("Staged startup feedback" row rewritten to two stages). Two small omissions: **M-d**, **M-i** |
| **11** No numeric change: no algorithm, constant, public name or `__all__` list moves | **Met — verified by reading the diff, not asserted** | I diffed every non-import, non-comment, non-docstring changed line across `_core`, `analysis`, `correction`, `detect`, `enhance`, `measure`, `refine`, `util`, `sdk_/orientation_fields`, `_cli`, `_gui/analysis`: the complete set is four `load_runtime_dependencies()` call statements, the `PLOTLY_AVAILABLE` rewrite, `if TYPE_CHECKING:` block headers, the `_mahotas()` cached loader, and `mh = _mahotas()`. Nothing else. The five files the plan forbids editing (`_tensor_voting.py`, `_dijkstra_kernels.py`, `colourspace.py`, `hdf_.py`, `refs/`) are untouched. `__all__` diff across all changed `src/` files: **one**, `_startup_perf.py`, additive and private — that is the change's own new API, not a movement of a public name |

---

## Whole-change coherence (the thing neither phase gate could see)

The two gates each read half. I read Tasks 1–6 together looking for the three interaction shapes
the brief names. Findings:

**A name deferred in Task 3/4 that Task 5/6 rebinds eagerly — none.** Analysis 1 above is the
mechanical form of that question, and it comes back clean at 27 known sites. The one near-miss is
`_gui/analysis/_render.py`, which Task 6 *did* have to fix (`matplotlib.pyplot` at module level,
reached eagerly from `compose_hub` through `analysis/_app.py`); the fix keeps
`matplotlib.use("Agg")` at module scope and moves only `pyplot` into `render_plot`, past the plotly
early return. That ordering is correct: `use("Agg")` still runs before any `import
matplotlib.pyplot` anywhere in the process.

**The four lazy `__getattr__`s plus the CLI's, as one system — no bad interaction.**
`phenotypic.__getattr__` raises `AttributeError` for unknown names, which is what
`SerializablePipeline._find_class_in_phenotypic` (`_serializable_pipeline.py:648-652`) needs: it
tries `hasattr(phenotypic, class_name)` and falls through to an explicit
`importlib.import_module` over 13 named submodules, `phenotypic.post` among them. `post` is absent
from `_LAZY_SUBPACKAGES`, but it was absent from the base `__init__` too, and every consumer spells
it `from phenotypic.post import X`, so nothing regressed. `phenotypic.plotting` is *newly* resolvable
(it used to work only as a side effect of the eager chain) — a deliberate widening, documented in
the `_LAZY_SUBPACKAGES` comment.

**An import-order cycle reachable only through a combination — none found.** The sweep covers all
75 packages including the `_gui` subtree, so the lazy `_gui.shell.__init__` is import-order-tested
as well. `abc_/__init__.py:50` `from phenotypic.sdk_ import FootprintMixin` is eager and therefore
pulls scipy/skimage into any `import phenotypic.abc_` — that is allowed (tier 5 watches only
`DEFERRED_RUNTIME_MODULES`) and matches the spec's non-goal on scipy/skimage.

**The `setdefault` removal (Amendment B/B1) is safe on every route, including the untested ones.**
This is the question the brief singles out, so the argument in full:

- The state the deleted `setdefault` guarded is *a module partly bound while a patch is live on
  another of its names*. In production that state is unreachable: `_load_cli_runtime()` binds all
  names of a module in one uninterrupted loop, nothing ever deletes a binding, and the four call
  sites are reached only after `load_runtime_dependencies()` has already imported the heavy
  libraries.
- In tests, the only thing that *deletes* a binding is `mock.patch.__exit__`. Read against the
  shipped stdlib rather than from memory: `/usr/lib64/python3.12/unittest/mock.py:1426-1431` sets
  `local` from `target.__dict__[name]`, so the patch that first triggered the loader (name absent
  from `__dict__`, resolved through `getattr` at `:1429`) records `is_local=False` and takes the
  `delattr` branch at `:1605`. That branch is immediately followed by `hasattr(self.target,
  self.attribute)` at `:1606`, which re-enters `phenotypicCLI.__getattr__` → `_load_cli_runtime()`
  → the module's `all(...)` is now False → **all** of its names are re-bound from the real module.
  So the partial state exists only between two adjacent lines of `__exit__`, inside a single thread.
- For a live patch to be clobbered in that window, a *second* patch on the same module would have
  to still be active while the first exits. `with`-nesting, stacked decorators, nested fixtures and
  `mock.patch.stopall()` are all LIFO — `_patch_stopall` at `mock.py:1972-1975` is literally
  `for patch in reversed(_patch._active_patches)` — which makes that impossible: the patch that
  takes the `delattr` branch is always the outermost one on that module, and every inner patch has
  already exited. The only construction that could break LIFO is out-of-order `patcher.start()` /
  `patcher.stop()`, and **`grep -rn "\.start()" tests/ | grep -i patch` is empty across the whole
  suite** — there is not one patcher-object idiom in the repo.
- The suite's only `mock.patch`/`monkeypatch` targets among the 28 deferred names are
  `phenotypic.phenotypicCLI.execute_dry_run` (two files, `monkeypatch.setattr`, which restores by
  `setattr` and never deletes) and `phenotypic.phenotypicCLI.create_execution_strategy` (the guard
  test itself).

**Confirmed by probe.** Two names of one module under nested patches, checked at all three depths,
plus an instrumented loader that records the binding state at every entry:

```
P2 inner: True True      # both Mocks in force after an explicit _load_cli_runtime()
P2 mid  : True True      # outer Mock still in force; inner name restored to the real function
P2 after: True True      # both names back to the real functions
P3 states observed at loader entry: None
```

`P3` is the decisive one: the loader is **never entered** during `mock.patch.__exit__`, so even the
two-adjacent-lines window described above is not reachable from the loader's side in practice.

The removal is correct, and `test_a_patched_deferred_cli_name_stays_patched_through_the_loader`
is now genuinely falsifiable: with the `continue` deleted, the loop reassigns the real function over
the Mock and the test's first assertion fails. That is a real fix for a real false green.

**`get_registry()` (`0e832d22`) is correct under every caller, not just the tested one.** I
enumerated all 24 call sites across `_gui/builder`, `_gui/analysis`, `_gui/results_viewer` and
`tests/`. Every one is inside a function; **no module in `src/` calls `get_registry()` at module
level**, which is precisely the premise property 3's docstring states and which makes the
"build outside the lock" decision safe rather than lucky. `discover()` imports only the eight
operation packages, none of which reaches `phenotypic._gui`, so the re-entry the docstring worries
about is unreachable today — and the implementation removes the dependency on that fact rather than
documenting it, which is the right call. The three concurrency tests are unusually well built: the
`discover_window` fixture parks **only the first** call, and the docstring explains at length why
parking every call would fail the correct implementation and pass the broken one. The
`observed["second_size"]` snapshot is taken inside the worker thread, not read back from the main
thread afterwards — the exact shape that makes this kind of test go vacuous, and it is avoided
deliberately.

The only caller-surface consequence of the fix is that a racing pair of first callers can now run
`discover()` twice and discard one registry. `discover()` imports packages and runs
`inspect.getmembers`; it registers nothing and mutates no global besides the local registry, so the
duplicate is genuinely benign. `startup-measurements.md` discloses it.

---

## Important

### I-1. `_load_cli_runtime()` takes a permanent snapshot, so a patch on a *source* module leaks a stale binding for the life of the process

**Confirmed by probe, not inferred.** `src/phenotypic/phenotypicCLI.py:296-303`.

```
P1 pre : False
P1 in  : SENTINEL
P1 post: SENTINEL | is real: False
```

(`mock.patch("phenotypic._cli._cli_validation.validate_pipeline", "SENTINEL")` held open across one
`cli._load_cli_runtime()`; `P1 post` is read *after* the patch has exited.) The module permanently
holds the dead object.

```python
for module_name, names in _CLI_RUNTIME_IMPORTS.items():
    if all(name in module_globals for name in names):
        continue
    module = importlib.import_module(module_name)
    for name in names:
        module_globals[name] = getattr(module, name)
```

The `continue` skip is documented, correctly, as what protects a live
`mock.patch("phenotypic.phenotypicCLI.<name>")`. It has a second consequence the docstring does not
mention: **once a module is bound, its names are never re-read.** Before this change the binding
happened at `phenotypicCLI` import time — i.e. at interpreter start, before any test could patch
anything. Now it happens on the first CLI command in the process.

**Failure scenario.** A test patches the *defining* module rather than the CLI module —
`mock.patch("phenotypic._cli._cli_validation.validate_pipeline")` — and inside that `with` block is
the first thing in the process to invoke a CLI command. `_load_cli_runtime()` runs, reads
`getattr(_cli_validation, "validate_pipeline")`, and binds the **Mock** into
`phenotypicCLI.__dict__`. The patch exits and restores `_cli_validation`, but `phenotypicCLI` is
never re-read: every later command in that worker calls a dead Mock. Under `-p no:randomly` it
would be reproducible; under random ordering it would be a shard-dependent flake whose failure
points at an unrelated test.

**Amendment B/B1 is not the cause; lazy binding is.** With `setdefault` the name is absent at that
moment, so `setdefault` would have bound `SENTINEL` identically. What changed is that
`phenotypicCLI` used to do `from phenotypic._cli._cli_validation import validate_pipeline` at module
level, binding the real function at interpreter start, where no patch could ever be open across it.
Deferring the bind moved it into a window a patch can span. This is a behaviour change relative to
`BASE_PRE`, and no guard covers it.

**Reach: zero live instances, so Important rather than Critical — measured, not reasoned.** I
derived the 28 `(source module, name)` pairs from the shipped `_CLI_RUNTIME_IMPORTS` table
(`phenotypicCLI.py:263`) and grepped `tests/` for each pair in all three patch forms:

- **Dotted-string form** (`patch("<module>.<name>")`, `monkeypatch.setattr("<module>.<name>", ...)`)
  — **3 hits**, all on `phenotypic._cli._cli_state_management.load_processing_state`, at
  `tests/unit/cli/test_cli_checkpoint_handler.py:150,655,679`. All three are inert: the patch body
  calls `_run_manifest(...)`, and `grep -rn "phenotypicCLI"` is **empty** in that test file and in
  all three modules on the call path (`_cli_checkpoint_handler.py`,
  `_dashboard/_manifest_builder.py`, `_cli_state_management.py`). `_load_cli_runtime()` is reachable
  only from `phenotypic_cli`, `_migrate_legacy_success_evidence`, `_regenerate_missing_overlays`,
  `_handle_recompile_slurm` and `phenotypicCLI.__getattr__`, none of which that path touches, so the
  loader never runs inside those windows. The file's one `CliRunner().invoke` (`:589`) targets
  `_cli_checkpoint_handler.main`, not `phenotypic_cli`, and is not inside a source-module patch.
- **Module-object form** (`patch.object(mod, "<name>")`, `monkeypatch.setattr(mod, "<name>", ...)`)
  — **1 hit**, `tests/integration/cli/test_lifecycle_publication_races.py:122`
  `monkeypatch.setattr(staged_worker, "ImagePipeline", serialized)`. Inert for a different reason:
  `staged_worker` is `phenotypic._cli._cli_staged_slurm_worker` (`:19`), which is not the source
  module for `ImagePipeline` in the table (that is `phenotypic._core._image_pipeline`), so it
  patches that worker's own binding and the loader never reads it.
- Grep sensitivity was confirmed with a positive control before trusting the empty results
  (`_cli_output_manager.aggregate_measurements` — a non-deferred name in the same modules — is found
  at two sites).

Production never patches anything, and pytest runs one test at a time per worker, so the window is
strictly the body of a `with` block in one thread. This is a trap laid for the next person, not a
defect shipping today.

**Recommendation: a documented constraint, not a code fix.** I would not take either code shape the
orchestrator sketched:

- *"Refresh bindings whose source module object no longer matches"* does not work. `mock.patch`
  mutates an attribute **on** the module; the module object itself never changes, so module identity
  is the wrong invariant. Detecting the stale value requires comparing each bound value against
  `getattr(module, name)` on **every** `_load_cli_runtime()` call — which is precisely what the
  `continue` skip exists to prevent, because it would overwrite a live
  `mock.patch("phenotypic.phenotypicCLI.<name>")` on every call. The two properties are in direct
  opposition, and nothing inside the loader can tell a Mock installed on the CLI module from a Mock
  installed on the source module. There is no correct code fix at this seam.
- That is the same judgement Amendment B already made: when a guarantee protects a state the loader
  cannot distinguish, remove the guarantee rather than keep untestable machinery.

So: one sentence in `_load_cli_runtime`'s docstring, next to the sentence that already explains the
skip — *the skip also makes each binding a one-time snapshot, so patch
`phenotypic.phenotypicCLI.<name>`, never the defining module; a source-module patch open across the
first load leaves the Mock bound after it exits.*

**If a guard is wanted rather than prose, the enforceable one is static, not runtime** (~15 lines,
`tests/unit/ci/`): parse `_CLI_RUNTIME_IMPORTS` from the shipped module — so the pair list cannot
drift from the code — and fail if any `tests/**.py` contains the string literal
`"<source module>.<deferred name>"` for any of the 28 pairs. Zero runtime cost, and it reddens the
moment someone lays the trap. State its limit in its own docstring: it cannot see the module-object
form (`patch.object(mod, "name")`), so it is a partial net, not a proof.

### I-2. `phenotypic-tune` runs pipelines and never calls `load_runtime_dependencies()`

`pyproject.toml:146` registers `phenotypic-tune = "phenotypic.tune.__main__:main"`, and
`tune/__main__.py:24` imports `ImagePipeline` and runs the engine over an image directory
(`_tune_cli/_run.run_tuning`). `grep -rn "load_runtime_dependencies" src/phenotypic/` returns five
files: the CLI, the four workers, and `_startup_perf` itself. Tune is not among them.

Spec D4 says *"Every pipeline-running entry point calls `load_runtime_dependencies()` after parsing
its options, so a broken required library fails at run start rather than being recorded as a
per-image scientific failure."* B3 then enumerates the call sites and omits tune; the Non-goals
section excludes tune from getting a *guard tier*, which is a different question from the preload.

**Before this change the gap did not exist**: tune paid every heavy import at `import phenotypic`,
so a broken numba/cv2/bm3d on a node failed at tune startup. After it, a tune worker on a bad node
fails inside its first trial — which the engine will record as a trial outcome, i.e. exactly the
"recorded as a scientific failure" mode D4 exists to prevent. Tune distributes across SLURM, so "one
bad node" is its motivating scenario, not a hypothetical.

This was raised as Phase 2 M8 and ruled *"in-spec and not a fault"*, with the note that *"if D4 is
meant literally, the enumeration in B3 is the thing to widen"*. I am not re-litigating the ruling;
I am recording that the item is still open and that it is a **behaviour regression relative to
`BASE_PRE`**, which the M8 note does not say. Either add the call to `tune/__main__.main` (one line,
same position as the CLI's) or narrow D4's wording so the spec stops claiming something the code
does not do.

### I-3. AC1's public-name resolution guard is structurally incapable of failing

`tests/unit/ci/test_startup_imports.py:141`:

```python
assert set(phenotypic.__all__) <= set(dir(phenotypic))
```

against `src/phenotypic/__init__.py:67-68`:

```python
def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
```

`dir()` is *defined as* a superset of `__all__`, so the assertion is a tautology. The surrounding
test resolves only three of the eighteen `__all__` entries for real — `Image`, `ImagePipeline` and
`detect` (`:138-140`). The other fourteen subpackages (`abc_`, `analysis`, `data`, `measure`,
`grid`, `refine`, `schema`, `prefab`, `correction`, `enhance`, `sdk_`, `util`, `settings`, `tune`)
plus `GridImage` are resolved by **nothing**.

**The mutation that proves it, run and confirmed.** Delete one name — `"prefab"` — from
`_LAZY_SUBPACKAGES` (`__init__.py:39-47`). Probe output:

```
guard assertion still passes: True
'prefab' still in dir(): True
resolution: BROKEN -> module 'phenotypic' has no attribute 'prefab'
importlib.import_module('phenotypic.prefab'): OK
```

Three legs, and **the third is what makes this non-obvious**:

1. `phenotypic.prefab` is broken for every real user process — `import phenotypic as pht;
   pht.prefab.X` is spec B4's first compatibility bullet.
2. The guard stays green, because `"prefab"` reaches `dir()` from `__all__` rather than from a
   binding.
3. **The one other guard that might have covered it structurally cannot.**
   `test_module_imports_first_in_a_fresh_interpreter` calls
   `importlib.import_module("phenotypic.prefab")`, which resolves through `sys.path` and never
   consults the parent package's `__getattr__` — so the sweep passes on a name the lazy map no
   longer carries. The same mechanism hides the breakage from the rest of the suite: importing
   `phenotypic.prefab` by path *binds* `prefab` on the package as an import side effect, so every
   later in-process `getattr` succeeds by accident.

**No live defect.** I checked all four lazy packages statically — every `__all__` entry is either
eagerly bound or present in the lazy map (`phenotypic` 18/18, `sdk_` 324/324, `abc_` 28/28,
`_gui.shell` 5/5). The property is true; it is simply unguarded.

**Not a finding, recorded to close it:** `phenotypic.post` raises `AttributeError`, and that is
correct. `post` is in neither `__all__` nor `_LAZY_SUBPACKAGES`, and it was not exported at top level
before this change either (`git show f0f9a544:src/phenotypic/__init__.py` imports 15 subpackages,
`post` not among them). Every consumer spells it `from phenotypic.post import X`, and
`_find_class_in_phenotypic` reaches it by explicit `importlib.import_module`
(`_serializable_pipeline.py:663`). No regression.

**Recommendation (not blocking): its own test, in a fresh interpreter, with a count floor.**

```python
def test_every_public_name_resolves_in_a_fresh_interpreter() -> None:
    """AC1's resolution half. ``__all__ <= dir()`` cannot fail: ``__dir__`` unions ``__all__`` in.

    Fresh interpreter, because importing ``phenotypic.<name>`` by path anywhere earlier in the
    worker binds ``<name>`` on the package as an import side effect -- so an in-process getattr
    passes for a name the lazy map no longer carries. That is the same blind spot that stops the
    import-order sweep from covering this: it uses ``importlib.import_module``, which never
    consults ``__getattr__``.
    """
    report = run_startup_probe(
        "import phenotypic\n"
        "missing = []\n"
        "for name in phenotypic.__all__:\n"
        "    try:\n"
        "        getattr(phenotypic, name)\n"
        "    except AttributeError:\n"
        "        missing.append(name)\n"
        "report = {'missing': missing, 'count': len(phenotypic.__all__)}\n"
    )
    assert report["count"] >= 18
    assert report["missing"] == []
```

Three things about that shape are load-bearing, in descending order of how easily they are dropped:

- **The `count` floor.** Without it, the cheapest way to green a failing resolution is to delete the
  offending entry from `__all__`, which passes everything and ships the regression — the exact move
  `test_the_watched_sets_are_the_ones_the_spec_names` (`:42`) exists to block for the watched
  tuples. The same discipline belongs here.
- **The fresh interpreter**, for the reason in the docstring above. An in-process version would be a
  second tautology — weaker than the first, and harder to spot.
- **A separate test, not folded into tier 1.** This is a hard constraint, not a style preference:
  resolving every `__all__` entry imports all 15 subpackages, i.e. it loads most of
  `HEAVY_STARTUP_MODULES`. Putting it in `test_import_phenotypic_loads_no_heavy_module`'s probe body
  would destroy that test's own `loaded == []` assertion.

The same treatment applies to `sdk_`, `abc_` and `_gui.shell`; one `@pytest.mark.parametrize` over
the four package names covers all of them, and it closes this finding and most of M-b at once.

---

## Minor

### M-a. The per-site deferral guard checks that the local import *exists*, not that it *precedes the use*

`tests/unit/ci/test_deferred_imports.py:294` walks the whole function for an `Import`/`ImportFrom`
binding the name. A function-local `import x` placed *after* a use of `x` makes `x` local for the
entire scope, so the earlier use raises `UnboundLocalError` — and the guard is green. That is the
one failure mode a moved import actually has, and the only thing standing between it and
production is that the rarely-exercised branches (`_plot_*` panel helpers, `inspect()` on the three
QC analyzers) get called by some test.

I ran the missing check over all 48 `(module, name, function)` rows with annotation nodes excluded:
**every site is correctly ordered today.** Adding it is ~10 lines and reuses the module's existing
`_annotation_node_ids` / `_has_future_annotations` helpers.

### M-b. `GUARDED_SUBPACKAGES` covers 13 of the 16 lazy subpackages

`test_startup_imports.py:178` lists `abc_, analysis, correction, detect, enhance, grid, measure,
plotting, post, refine, schema, sdk_, util`. `_LAZY_SUBPACKAGES` also contains **`data`,
`prefab`, `settings`, `tune`**, and none of them appears in any tier. A module-level `import colour`
added to `phenotypic/prefab/` or `phenotypic/tune/` today passes tier 1 (lazy), tiers 2 and 3 (not
on those paths), tier 4 (not on the `Image` path), tier 5 (not listed) and the sweep (which only
asks whether the import succeeds). This is the direct answer to *"where could a deferred library be
re-added today and pass every guard?"* — that, plus the `_gui/` subtree and `_cli/`, which are out
of scope by construction.

`mutation-proofs.md` is explicit that M1–M7 prove guard *correctness*, not guard-set completeness,
and names tier 5 as the completeness net. Extending that tuple by four names closes most of the
remaining hole for free.

### M-c. `PLOTLY_AVAILABLE` now answers "installed", not "importable"

`_accessor_dash_handler.py:19`. `importlib.util.find_spec("plotly") is not None` is `True` for a
plotly that is on disk but raises on import — a broken wheel, or (historically) `plotly.express`
refusing to import without pandas. The old `try: import plotly.express ... except ImportError`
returned `False` in exactly those cases, and `_require_plotly()` produced the actionable *"install
it with: pip install plotly>=6.0.0"* message. Now `_require_plotly()` passes and the raw
`ImportError` from `import plotly.express as px` (`:124`) reaches the user instead. Raised as
Phase 1 M3 and still open; the trade (not importing plotly on every `Image`) is clearly worth it,
and the degraded path is narrow.

### M-d. `IMPORT_STARTED_AT` has no runtime consumer left, and a docstring still says it does

Deleting `_core_import_elapsed` removed the only caller. `_startup_perf.IMPORT_STARTED_AT` and
`phenotypic._IMPORT_STARTED_AT` are now stamped and never read (the spec deliberately keeps them, so
this is not a defect), but `_gui/shell/_startup.py:10` still documents the reporter as measuring
core-library load *"against `phenotypic._startup_perf.IMPORT_STARTED_AT`"*, which no longer happens.
`StartupReporter.import_elapsed` itself is correctly preserved and still exercised by
`tests/unit/gui/shell/test_startup.py:47`, as the spec requires.

### M-e. `available_modes` is bound by the loader but used nowhere in `phenotypicCLI`

`phenotypicCLI.py:254` (`TYPE_CHECKING`, `# noqa: F401`) and `:265` (the loader table). Task 5
replaced its only use with `sorted(get_args(DetectMode))`. Keeping it in `_CLI_RUNTIME_IMPORTS`
preserves `mock.patch("phenotypic.phenotypicCLI.available_modes")` and
`from phenotypic.phenotypicCLI import available_modes` — neither of which any test or module does
(grep is empty). The cost is that `_migrate_legacy_success_evidence`, `_regenerate_missing_overlays`
and `_handle_recompile_slurm` each import the detection-mode registry they do not use. Harmless;
worth a comment saying it is a compatibility shim, or worth dropping.

### M-f. `DEFERRED_SITES` carries a row for a name that no longer exists

`test_deferred_imports.py:31`: `"_pio": ()` under `_accessor_dash_handler.py`. A/P4 moved the
docs-build renderer switch to `_startup_perf`, and `_pio` is gone from the file entirely. The row
passes both checks vacuously. One-line deletion.

### M-g. `__dir__` is not given the partial-initialisation treatment `__getattr__` got

All four lazy `__init__`s define `__dir__` as `sorted(set(globals()) | set(__all__))` *before*
`__all__` is assigned (`phenotypic/__init__.py:67` vs `:93`; `abc_/__init__.py:37` vs `:77`;
`sdk_/__init__.py:72`; `_gui/shell/__init__.py:43` vs `:53`). Spec A/P8 put `__getattr__` above the
eager imports precisely so a re-entrant import mid-initialisation can still resolve a name;
`dir()` in that same window raises `NameError: name '__all__' is not defined` instead of returning
the eager surface. No route reaches it today — nothing in `src/` calls `dir()` on a package during
its own import — but the two functions were written to the same shape and only one of them holds up
under the condition the shape exists for.

### M-h. `_gui/CLAUDE.md`'s new gotcha does not record how a broken builder now presents

Phase 2 M4: a `builder.create_app` failure used to abort the launcher before the port was bound; it
now surfaces as a 500 on `/builder/` and, because `ToolSession.get()` correctly declines to cache a
failed build, re-runs the failing build and its full import cost on **every** subsequent request.
The gotcha at `_gui/CLAUDE.md:586-591` says only that the builder is built on first request. One
sentence.

### M-i. `RESUME.md` is still in the tree

198 lines of stale scaffolding at the worktree root, contradicted by
`.superpowers/sdd/plan/progress.md`'s own "Corrections to RESUME.md, part 2". Task 8 deletes it;
recording it so it is not forgotten at commit time.

---

## What I did not verify

- **AC8** (full default lanes, PLAYWRIGHT e2e, Sphinx warning set-diff, mypy/ruff set-diffs) — the
  orchestrator is running these. The numbers in `phase1-review.md`, `phase2-review.md` and the
  ledger are taken as given and attributed, not re-derived.
- **AC9's timings** — reported in `startup-measurements.md`; I checked the *methodology* (same
  machine, `before` correctly re-measured on HPCC rather than paired with the committed macOS run,
  `src/` of `.worktrees/lazy-baseline` verified byte-identical to `BASE_PRE`) and the disclosed
  caveat, not the seconds.
- **Runtime behaviour of the three concurrency tests** — I read them and reasoned about the
  interleavings; I did not execute them.

---

## Verdict

**Ready to merge.** No Critical findings.

I-1 and I-3 are both confirmed by probe, and both are latent hazards over code that is correct
today — I-1 has zero live instances across all three patch forms, and every `__all__` entry of all
four lazy packages resolves. Neither should be fixed under time pressure; both should be recorded,
and each has a concrete cheap remedy named above (a docstring sentence plus an optional static
guard for I-1; one parametrized fresh-interpreter test with a count floor for I-3, which also closes
most of M-b). I-2 is a pre-existing ruling that is still open and whose spec wording should be
reconciled either way — it is the one item that is a genuine behaviour regression relative to
`BASE_PRE` rather than a coverage gap.

The nine Minor findings are all one-to-ten-line changes and none of them affects shipped behaviour.

**Also verified clean, and worth saying rather than leaving as silence:** AC11 by reading every
non-import changed line rather than trusting the commit messages; the Amendment B `setdefault`
removal, by argument from the shipped CPython stdlib plus a three-depth probe; and `get_registry()`
under all 24 of its callers, whose property-3 premise (no module-level `get_registry()` call
anywhere in `src/`) I checked mechanically rather than taking from its docstring.
