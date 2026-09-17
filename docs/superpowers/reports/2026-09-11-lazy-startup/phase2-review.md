# Phase 2 gate review — lazy startup (Tasks 5–6)

**Reviewer:** independent implementation/test reviewer (opus), read-only over the branch
`refactor/private-gui` in the worktree `.worktrees/private-gui`, at `0d0bc221`.
**Scope:** exactly two commits, `git diff 6abf8d96..0d0bc221` — `7049bfaa` (Task 5, CLI help path
and fail-fast preload) and `0d0bc221` (Task 6, GUI launcher, lazy shell package, builder on first
request), read as one change against
`docs/superpowers/specs/2026-09-11-lazy-startup/spec.md` (Amendment A P1–P14 superseding the body),
`docs/superpowers/plans/2026-09-11-lazy-startup/plan.md` (Task 5 §1183, Task 6 §1542) and the
Phase 1 gate at `docs/superpowers/reports/2026-09-11-lazy-startup/phase1-review.md`.
Tasks 7–8 (measurements, regression, docs) are out of scope and are noted, not faulted.

**Evidence supplied by the gate controller** (not measured by this reviewer, and attributed as
such wherever it is used below): the startup guards at **218 passed**; the 16-shard array over
`tests/unit/cli`, `tests/integration/cli`, `tests/unit/gui`, `tests/integration/gui`, `tests/gui`,
`tests/unit/ci` aggregating to **6228 tests, 0 failed, 0 errors, 33 skipped**; `PLAYWRIGHT=1` e2e
(builder + the six `ci_flaky` helper modules) at **22 passed, 86 skipped, 0 failed**, every skip
attributed to a documented surface retirement from the 2026-08-26 gui-simplification-removals spec.
There is therefore **no regression pressure on this verdict**; the guards are judged on their
merits, and a green suite is precisely what a false green looks like.

**Environment note.** The worktree venv was re-synced to
`uv sync --group dev --group test-qt --group docs --all-extras` between the static review and the
probes, so `optuna` is now installed where it was absent. I re-checked every reading that could
depend on it: none does. `test_every_deferred_runtime_module_is_a_hard_dependency` iterates
`DEFERRED_RUNTIME_MODULES` only, and all eight names there are unconditional
`[project.dependencies]` entries under either sync. The one downstream consequence is for a
*re-run* of Task 5's Step 5b.5 proof — see the Step 5b row of the conformance table.

---

## Verdict

**Ready for Task 7.**

Both commits do what they say. The CLI deferral is complete and safe: I traced every one of the 28
deferred names to every `ast.Name` load in `phenotypicCLI.py` and all of them lie inside the four
functions that call `_load_cli_runtime()` first — there is no reachable path that reads a heavy
name before it is bound, no module-scope use survives, and the one runtime-evaluated annotation
(`_render_one`'s `manager: OutputManager`) is inside a nested `def` that executes after the loader.
The preload placement outside the `try:` is correct and, unusually, *pinned* — `except Exception`
at `phenotypicCLI.py:3300` would convert the `ImportError` into `SystemExit(1)` and the abort test's
`isinstance(result.exception, ImportError)` would go red. AC6's four worker entry modules are the
complete set: I checked the other eight `__main__` modules under `_cli/` and none of them loads or
runs a pipeline.

**T6's headline claim is correct and is the best single decision in the change.**
`HUB_ALLOWED_BEFORE_FIRST_REQUEST` is subtracted from the probe's *watched* result, so adding
`matplotlib.pyplot` to it would have made `test_composed_hub_builds_the_builder_on_its_first_request`
structurally incapable of failing on pyplot while still reporting green. That is a real property of
the code as written, not a rationalization. Moving pyplot to the point of use was the right call.

What I would not merge on is four things, one of them now measured rather than argued.

**The third false green this change has produced is I1**, and it is measured: the one test whose
name asserts the `globals().setdefault` mechanism never executes `setdefault`. `mock.patch`'s own
`getattr` binds all 28 names through the module `__getattr__` before the test calls the loader, so
**11 of 11** modules take the `all(...) → continue` short-circuit. Replacing `setdefault` with a
plain assignment leaves the test green. The production code is correct — the same probe shows the
patch surviving and the real function restored — so this is a test fix, not a code fix.

The other three: the allow-list subtraction T6 correctly identified as dangerous is still in the
tier-3 guard, unfenced, and the new `HUB_WATCHED_MODULES` tuple repeats verbatim the Phase 1 I3
defect that `ffa77f5b` had just finished closing for the other two constants (I2); the worker
preload guard asserts that `load_runtime_dependencies()` appears *somewhere* in `main`, not that it
runs before any image work (I4); and `get_registry()` publishes its singleton **before** populating
it, which makes T6's "duplicate work, benign" ruling a description of a failure mode the code does
not have — the real one hands the second caller an empty registry for the whole `discover()` window
(I3). None of the four blocks Task 7, which touches measurements and docs rather than these files.
All four should land before the branch merges.

---

## Strengths

- **The preload placement is argued *and* pinned.** `load_runtime_dependencies()` and
  `_load_cli_runtime()` sit at `phenotypicCLI.py:1720-1721`, above the `try:` at `:1722`, with a
  five-line comment citing spec A/P13. I verified the counterfactual rather than taking it on
  trust: the body's `except Exception as e:` (`:3300`) echoes `"Unexpected error"` and calls
  `sys.exit(1)`, so an `ImportError` raised inside the `try` would reach
  `test_cli_aborts_before_any_output_when_a_runtime_dependency_is_broken` as a `SystemExit`, and
  its `assert isinstance(result.exception, ImportError)` would fail. The placement is therefore a
  guarded decision, not a convention.
- **The abort test has a real control.** `calls == ["preload"]` is what makes it able to fail:
  deleting the call (spec mutation M5) empties the list. `assert not output_dir.exists()` on its
  own would pass vacuously, and the test does not rely on it.
- **The four worker entry points are the right four, and the set is closed.** I grepped all twelve
  `__main__` modules under `src/phenotypic/_cli/` for `from_json` / `.apply(` / `.measure(` /
  `ImagePipeline` / `GridImage`. The eight that did not get a preload
  (`_cli_finalize_fanout`, `_cli_sentinel`, `_cli_migrate_provenance_worker`,
  `_cli_slurm_lifecycle`, `_cli_migrate_worker`, `_cli_staged_controller`, `_cli_chunk_writer`,
  `_cli_update_state`) have **zero** hits — the only matches are the word "Image" in docstrings and
  a click `help=` string. Spec B3's enumeration holds exactly.
- **No unbound-name path exists in the CLI.** For each of the 28 names in `_CLI_RUNTIME_IMPORTS` I
  located every occurrence in `phenotypicCLI.py` and mapped it to its enclosing top-level `def`.
  All uses fall inside `_migrate_legacy_success_evidence` (`:638`), `phenotypic_cli` (`:1635`),
  `_regenerate_missing_overlays` (`:3308`) and `_handle_recompile_slurm` (`:3455`) — exactly the
  four that call `_load_cli_runtime()` as their first statement. The functions those four call
  (`_handle_recompile`, `_discover_recompile_dataset_names`, `_initialize_recompile_slurm_attempt`,
  `_build_recompile_job_metadata_datasets`, `_recompile_dataset_image_names`,
  `_wait_for_recompile_finalizer_status`, `_read_recompile_wait_status`,
  `_raise_if_recompile_attempt_cannot_finish`, `_copy_pipeline_to_output`, `setup_logging`,
  `error_exit`) contain none of the 28.
- **The one runtime-evaluated annotation is safe, and it is the non-obvious case.**
  `phenotypicCLI.py` has **no** `from __future__ import annotations`, so annotations are evaluated.
  `finalizer_pipeline: Optional[ImagePipeline] = None` (`:3104`) is a *local* annotation and PEP 526
  leaves those unevaluated; `def _render_one(item: OverlayWork, manager: OutputManager)` (`:3398`)
  **is** evaluated, but it is a nested `def` inside `_regenerate_missing_overlays`, whose first
  statement is `_load_cli_runtime()`. No top-level signature in the file names a deferred type.
- **`mock.patch` teardown does not leave a stale binding, despite the `delattr` path.** I traced
  CPython's `_patch.get_original()` / `__exit__`: for a name not yet in `phenotypicCLI.__dict__`,
  `get_original` falls through to `getattr` (firing `__getattr__`), records `local = False`, and on
  exit `delattr`s, then tests `hasattr` — which re-fires `__getattr__`, re-imports, and rebinds the
  real object, so the `not hasattr` restore branch is correctly skipped and the name ends up bound
  to the genuine function. The `__getattr__`/`delattr` interaction the design invites is benign
  here, and `__getattr__` still raises `AttributeError` for unknown names, so `mock.patch(...,
  create=False)` on a typo still fails loudly.
- **The `--detect-mode` help really is byte-identical.** `available_modes()`
  (`_core/_image_parts/detection_modes/_detection_mode.py:95-97`) returns
  `tuple(sorted(_DETECTION_MODE_REGISTRY))`, and `DetectMode` (`sdk_/typing_.py:50`) is an 11-name
  `Literal`. `sorted(get_args(DetectMode))` and `list(available_modes())` are the same list in the
  same order, and the drift test pins it. A custom mode registered through
  `PHENOTYPIC_PRELOAD_MODULES` could not appear in the choices before this change either
  (`preload_custom_operation_modules()` runs inside the command body), so nothing user-visible
  moves.
- **The two plan-text defects T6 reports are both real.** I checked them against the pre-change
  blob. `git show 6abf8d96:src/phenotypic/_gui/shell/_app.py` line **539** is
  `# 3. Builder Dash (eager — single-process registry build).` and line 540 is `_tick("builder")`,
  so the plan's "lines 540–550" applied literally would indeed have stranded a comment saying
  "eager" above the lazy block. And nothing in the tree pins the launcher's three-stage sequence:
  the only hits for `_STARTUP_STEPS` / `"Core library loaded"` / `import_elapsed` in `tests/` are
  `tests/unit/gui/shell/test_startup.py:47,52,61,102`, which construct their own `StartupReporter`
  with `total_steps=3, import_elapsed=1.23` and call `record_done` directly — `StartupReporter`'s
  tests, not the launcher's.
- **T6's "never released" claim is verified at every release site.** The builder session is absent
  from `start_idle_release_thread([viewer_session, analysis_session])` (`_app.py:660-663`) and from
  `extra_release_sessions=(analysis_session,)` (`_app.py:528`), which is the only other release
  path (`_routes.py:399-404`). Nothing can release the builder.
- **`ToolSession` gives the builder the memoization and failure semantics the commit claims.**
  `get()` (`_session.py:118-127`) builds under `self._lock` only when `self._state is None`, so the
  session is memoized, concurrent first requests cannot double-build, and a build that **raises
  leaves `_state` at `None`** — a failed first build is not cached, and the next request retries.
- **Both tier-2 guards are genuinely falsifiable, and the GUI one covers two independent
  regressions.** `test_gui_help_loads_no_heavy_module` imports
  `phenotypic._gui.shell._launcher`, which necessarily executes
  `phenotypic._gui.shell.__init__` first — so it fails both on spec M6 (re-importing `create_app`
  at launcher module level) *and* on Amendment A/P2 (an eager `_app` import in the shell package),
  because `dash` is in `HEAVY_STARTUP_MODULES`. Its `"phenotypic-gui" in help` control is real:
  `_launcher.py:120` sets `prog="phenotypic-gui"`, so a probe that printed nothing fails.
- **The tier-3 hub guard's controls are the load-bearing half.** `detect_before is False` /
  `status == 200` / `detect_after is True` is exactly spec mutation M4: making the builder eager
  again flips `detect_before` and the test goes red. `app.server.test_client()` routes through
  `Flask.__call__ → wsgi_app`, which is the `DispatcherMiddleware`, so `/builder/` really does go
  through `_SessionProxy`.
- **The `_gui/analysis/_render.py` deferral is clean and fully covered.** `plt` has exactly one
  use, `plt.gcf()` at `:87`, inside `render_plot`, after the local import at `:61`; there is no
  second user and no module-scope use. The `DEFERRED_SITES` row added in the same commit therefore
  satisfies all three checkers in `tests/unit/ci/test_deferred_imports.py`, including the escape
  check added by `ffa77f5b`.
- **Keeping `matplotlib.use("Agg")` at module scope is the right call and is argued correctly.**
  Deferring it alongside pyplot would move a process-wide backend selection into a request thread.
  `matplotlib` (core, not pyplot) is already reachable before the first request through the
  scipy/skimage chain and is in `HUB_ALLOWED_BEFORE_FIRST_REQUEST`, so leaving it changes no
  assertion.
- **The Phase 1 findings were closed, and closed well.** `ffa77f5b` landed tier 5
  (`GUARDED_SUBPACKAGES`, `test_startup_imports.py:195-215`), the reverse per-site checker
  (`test_deferred_imports.py:259`), the pinned watched sets (`test_startup_imports.py:42`) and M4's
  two extra sweep entries (`phenotypic._startup_perf`, `phenotypic.settings`). Tier 5 in particular
  uses **exact equality** against `SUBPACKAGE_EXPECTED_DEFERRALS` rather than a subtraction — which
  is the idiom I1/I2 below say T6 should have reused.

---

## Critical

None.

---

## Important

### I1. `test_a_patched_deferred_cli_name_stays_patched_through_the_loader` cannot fail on a mutation of `setdefault` — the mechanism it is named for is untested

**File:** `tests/unit/cli/test_cli_runtime_preload.py:50-58`, against
`src/phenotypic/phenotypicCLI.py:302-315`.

**The production code is correct. Only the guard is worthless.** This is a test fix, not a code
fix. The probe below shows the patch surviving the loader (`patch still in force: True`) and the
real function restored on exit — the behaviour spec A/P1 asks for is delivered. What is missing is
any test that would notice if it stopped being delivered.

**Evidence.** `_load_cli_runtime()` has two independent protections, and the test exercises only
the cheaper one:

```python
    module_globals = globals()
    for module_name, names in _CLI_RUNTIME_IMPORTS.items():
        if all(name in module_globals for name in names):   # ← short-circuit
            continue
        module = importlib.import_module(module_name)
        for name in names:
            module_globals.setdefault(name, getattr(module, name))   # ← the documented mechanism
```

`mock.patch.__enter__` calls `_patch.get_original()`, which does `target.__dict__[name]` and, on
`KeyError`, falls through to `getattr(target, name, DEFAULT)`. That `getattr` fires
`phenotypicCLI.__getattr__` (`:318-323`), which calls `_load_cli_runtime()` — binding **all 28
names across all 11 modules** into `globals()` before the patch's `setattr` even runs. By the time
the test's own `cli._load_cli_runtime()` executes inside the `with` block, every module hits
`all(name in module_globals)` and `continue`s. `setdefault` is never reached. The same holds if a
previous test in the worker already warmed the module, because `_load_cli_runtime()` binds
all-or-nothing across the whole table.

**Measured, not inferred.** The gate controller ran this probe in the worktree
(`--all-extras` venv), verbatim output:

```
total modules in _CLI_RUNTIME_IMPORTS: 11
bound BEFORE patch enter: []
short-circuited at the moment _load_cli_runtime() is called inside the with: 11 of 11
patch still in force: True
after exit, name in __dict__: True
after exit, value: <function create_execution_strategy at 0x7f161f61e980>
```

Nothing is bound before `__enter__` — so `mock.patch`'s own `getattr` is what binds all 28 names —
and by the time `_load_cli_runtime()` runs inside the `with`, **11 of 11** modules take the
`continue`. The `setdefault` line has no reachable execution in this test. The last two lines are
the other half of the result: production behaviour is right.

**Concrete failure scenario.** A later refactor "simplifies" the loader to
`module_globals[name] = getattr(module, name)` — the obvious reading, since a `setdefault` whose
key is known-absent looks redundant. This test stays green. So does every other test, because the
three deferred names that are actually patched in the suite
(`create_execution_strategy` ×4, `generate_recompile_slurm_scripts` ×3, `execute_dry_run` ×3) all
reach `_load_cli_runtime()` through the same short-circuit. The protection is silently removed.

**How reachable is the state `setdefault` guards? Barely — and this correction matters for what to
do about it.** I traced it properly rather than assuming. A patch is `is_local=False` (the only
branch whose teardown `delattr`s) exactly when the name is absent from `cli.__dict__` at
`__enter__`. But *any* `getattr` on a deferred name — including `get_original`'s — binds all 28.
So the **first** deferred-name patch in a process is non-local and every subsequent one is local,
restoring by `setattr` and never deleting. The only window in which a deferred name is absent is
the two statements inside a non-local `__exit__` between `delattr` and `hasattr`, on one thread,
and `hasattr` closes it immediately by re-running the loader.

For `setdefault` to protect a *live mock*, a sibling name of the same module must be mocked at that
instant. Under `with`/decorator nesting that is impossible (LIFO: the inner patch has already been
restored). It requires `mock.patch(...).start()` / `.stop()` called **out of order** on two names of
one module, with the first of them being the process's first deferred-name access. I grepped: that
style is not used anywhere in the CLI test surface, and `monkeypatch.delattr` appears nowhere in
`tests/`. So the guarded state is reachable in principle and unreached in practice.

**Consequence: `setdefault` and the `all(...) → continue` short-circuit are redundant with each
other.** Either alone delivers spec A/P1's property; the short-circuit must stay (it is the
"cheap after the first call" half); therefore `setdefault` never executes. That is why *no single-line
mutation reddens the existing test* — remove `setdefault` and the short-circuit still protects the
patch; remove the short-circuit and `setdefault` still does. Redundancy plus no test is what
produced a guard with nothing behind it.

**The exact mutation that proves it.** In `phenotypicCLI.py:314`, change

```python
            module_globals.setdefault(name, getattr(module, name))
```

to

```python
            module_globals[name] = getattr(module, name)
```

and run `tests/unit/cli/test_cli_runtime_preload.py::test_a_patched_deferred_cli_name_stays_patched_through_the_loader`.
It passes. A guard that survives the deletion of the line its docstring is about is a false green.

**What the test should assert instead.** Since `mock.patch` cannot produce the partial-binding
state, a test that reaches `setdefault` must construct it directly: bind one of a module's two
names, leave the sibling unbound, call the loader, and assert the bound one survived *and* the
sibling was filled. The second assertion is the control — it proves the import branch actually ran,
so the test cannot pass by taking the short-circuit.

```python
_UNBOUND = object()


def test_the_runtime_loader_keeps_a_name_that_is_already_bound() -> None:
    """``_load_cli_runtime`` fills only the missing names of a partly-bound module (spec A/P1).

    Scope, stated plainly: ``mock.patch`` cannot produce this state. Its ``__enter__`` does a
    ``getattr``, which fires the module ``__getattr__`` and binds all 28 names, so every later
    call takes the loader's ``all(...) -> continue`` short-circuit. The state is constructed
    here because it is the loader's documented contract and nothing else in the suite executes
    ``setdefault`` at all.
    """
    import phenotypic.phenotypicCLI as cli
    from phenotypic._cli._cli_execution_strategies import (
        create_execution_strategy,
        uses_staged_gpu_strategy,
    )

    names = ("create_execution_strategy", "uses_staged_gpu_strategy")
    saved = {name: cli.__dict__.get(name, _UNBOUND) for name in names}
    stand_in = object()
    try:
        cli.__dict__["create_execution_strategy"] = stand_in   # already in force
        cli.__dict__.pop("uses_staged_gpu_strategy", None)     # its sibling is not bound

        cli._load_cli_runtime()

        assert cli.__dict__["create_execution_strategy"] is stand_in          # setdefault kept it
        assert cli.__dict__["uses_staged_gpu_strategy"] is uses_staged_gpu_strategy  # branch ran
    finally:
        for name, value in saved.items():
            if value is _UNBOUND:
                cli.__dict__.pop(name, None)
            else:
                cli.__dict__[name] = value
    assert cli.create_execution_strategy is create_execution_strategy
```

Under the mutation, the first assertion fails — plain assignment overwrites `stand_in` with the
real function. Under the shipped code it passes.

The **existing** test should be kept but renamed and re-docstringed, because it does not test what
its name says: it pins the end-to-end property (*a patched name survives a CLI run*) through the
short-circuit, which is worth having, and its docstring should say that the `setdefault` contract
is covered by the test above.

**Recommended action: test fix only. Do not change `phenotypicCLI.py` on this branch.**
Production behaviour is correct, and spec Amendment A/P1 names `globals().setdefault` explicitly,
so deleting it would contradict binding authority.

**The alternative, if the user will amend the spec.** Given the redundancy established above, the
simpler honest shape is to *remove the false guarantee rather than test a fictional one*: change
`setdefault` to plain assignment, delete the claim from the docstring, and let the surviving
`all(...) → continue` be the single named mechanism. The payoff is that the **existing** test
becomes genuinely falsifiable with no new test at all — delete the `continue` and it goes red,
because the loop then overwrites the mock. One mechanism, one claim, one test, one mutation. The
cost is that the unreached partial-binding case loses its belt, and that A/P1's wording must change
in the same commit, which is a user-gated decision rather than mine. I recommend the test fix for
this branch and this as a separate, gated follow-up.

### I2. The tier-3 hub guard keeps the exact mechanism T6 identified as dangerous, and its new watched tuple is unpinned

**File:** `tests/unit/gui/shell/test_hub_startup_imports.py:12-46`.

Two halves, same root cause. The commit message is right about the hazard; the code still contains it.

**(a) The allow-list can still cancel a watch, and nothing checks that it does not.**
The assertion is

```python
    assert sorted(set(report["loaded_before"]) - set(HUB_ALLOWED_BEFORE_FIRST_REQUEST)) == []
```

`HUB_WATCHED_MODULES` = `("bm3d", "colour", "cv2", "h5py", "mahotas", "matplotlib.pyplot",
"numba")` and `HUB_ALLOWED_BEFORE_FIRST_REQUEST` = `{"plotly", "pandas", "polars", "pyarrow",
"scipy", "skimage", "matplotlib"}`. The two are **disjoint**, and the probe only ever reports
members of `HUB_WATCHED_MODULES` — so the subtraction is a no-op today and the allow-list dict is
mechanically dead. Its only live effect is the one T6 named: a future maintainer facing a red
guard adds the offending name to the allow-list (which is what the dict's own comment invites:
*"a shell-side module-level chain may be listed here with its justification"*) and the watch is
cancelled with the test still green. Nothing asserts the disjointness that makes the current
arrangement safe.

**Concrete failure scenario.** Someone reintroduces a module-level `import matplotlib.pyplot` in
`_gui/analysis/_render.py` (or any module `compose_hub` reaches eagerly), the hub guard goes red,
and the cheapest green is a one-line dict entry:
`"matplotlib.pyplot": "analysis/_render.py:23, dash worker rendering"`. The subtraction removes it,
`loaded_before` nets to empty, the test passes, and the 270 ms is back. This is exactly what T6's
commit message says must not happen — and there is no test that stops it.

**(b) `HUB_WATCHED_MODULES` repeats the Phase 1 I3 defect in a brand-new constant.** Phase 1's I3
was *"the watched sets are unpinned, so a guard can be retired without anything going red"*; it was
closed one commit earlier by `test_the_watched_sets_are_the_ones_the_spec_names`
(`tests/unit/ci/test_startup_imports.py:42-73`), which pins `HEAVY_STARTUP_MODULES` and
`DEFERRED_RUNTIME_MODULES` by value. `HUB_WATCHED_MODULES` is new in T6, is the spec's tier-3
minimum (spec A/P11), and is pinned by nothing. Deleting `"matplotlib.pyplot"` from it greens a
pyplot regression with no other test noticing.

**The exact mutations that prove both.** (a) Add `"matplotlib.pyplot": "any string"` to
`HUB_ALLOWED_BEFORE_FIRST_REQUEST` *and* restore `import matplotlib.pyplot as plt` at
`_gui/analysis/_render.py:28` — the hub guard passes (only `test_deferred_names_are_not_imported_at_module_level`
goes red, and it is a different file guarding a different property). (b) Delete
`"matplotlib.pyplot"` from `HUB_WATCHED_MODULES` and make the same source mutation — every test in
`tests/unit/gui/shell/` passes.

**Proposed fix** — use the idiom `ffa77f5b` already established one commit earlier for tier 5
(`test_startup_imports.py:214`), which asserts exact equality against a table of *expected*
exceptions rather than subtracting an allow-list:

```python
def test_composed_hub_builds_the_builder_on_its_first_request(tmp_path: Path) -> None:
    # HUB_WATCHED_MODULES is the spec's tier-3 minimum (A/P11). Pinning it by value is what
    # stops a failing guard from being greened by deleting the offending name.
    assert set(HUB_WATCHED_MODULES) == {
        "bm3d", "colour", "cv2", "h5py", "mahotas", "matplotlib.pyplot", "numba",
    }
    assert set(HUB_WATCHED_MODULES).isdisjoint(HUB_ALLOWED_BEFORE_FIRST_REQUEST), (
        "an allowed module that is also watched cancels its own watch: the subtraction below "
        "would remove it and the guard would report green while proving nothing"
    )
    ...
    assert report["loaded_before"] == []
```

Asserting `report["loaded_before"] == []` directly is strictly safer than the subtraction: the
allow-list keeps its documentary value (it explains what the hub *does* load, which the probe
never measures anyway), and the only way to green a leak becomes deleting a name from
`HUB_WATCHED_MODULES` — a visible, reviewable line that the pin above then blocks.

### I3. The `get_registry()` ruling is right that the race is newly reachable and wrong about what it does

**File:** `src/phenotypic/_gui/_operation_registry.py:811-824`, exposed by
`src/phenotypic/_gui/shell/_app.py:543-556, 641`.

The commit message records: *"a concurrent first `/builder/` and first `/analysis/` request can both
enter it. The duplicate work is benign — equivalent registries, last writer wins, the loser garbage-
collected — and needs no lock."* The first half is correct; the second half describes a failure mode
the code does not have.

**Evidence.**

```python
_REGISTRY: Optional[OperationRegistry] = None


def get_registry() -> OperationRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = OperationRegistry()   # ← published here
        _REGISTRY.discover()              # ← populated only here
    return _REGISTRY
```

The singleton is **published before it is populated**. A second thread entering `get_registry()`
after line 822 and before line 823 returns does not do duplicate work and does not lose a race —
it receives the *same* object, empty, and returns it. `discover()` is the call that imports the
entire operation library (spec B2: ~1 s), so the window is not a few instructions; it is the whole
import.

**Concrete failure scenario.** Two browser tabs, `/builder/` and `/analysis/`, first hit within a
second of each other on the threaded Flask dev server (`_launcher.py:115`, `app.run(...)`, Flask's
default `threaded=True`). Builder's build closure reaches `get_registry()` and starts `discover()`.
Analysis's build closure reaches it, sees a non-`None` `_REGISTRY`, and builds its layout from an
empty registry: `_gui/analysis/_layout.py:64` (`get_registry().get_by_category(category)`) returns
`[]`, so every analyzer dropdown renders empty, and `:87`/`:846` resolve every instance to
`info = None`. The user gets a silently wrong page — no exception, no log line — and it persists
until that `ToolSession` is released. The `ToolSession` locks do not help: they are per-session, and
this is two different sessions.

Before T6 this was unreachable: the builder was built inside `compose_hub` on the composing thread,
so `discover()` had completed before anything was served. That is precisely why the commit is right
to flag it — the diagnosis is what needs correcting.

**Proposed fix** (report separately; this is a behaviour change and needs its own test and commit,
per the brief).

**Publish-after-build alone is necessary but not sufficient.** The bare form —
`registry = OperationRegistry(); registry.discover(); _REGISTRY = registry` — fixes completeness
but lets two threads each publish, so `id(get_registry())` can change between two calls in one
process. That is observable: `builder/_layout.py:967-1010`
(`_resolve_dag_accepts_for_class_port`) is a cache **keyed on `id(registry)`** and returns `None`
(forcing the uncached ~12 µs-per-port walk on every render) when the live registry's id does not
match the caller's. Its docstring states the assumption in writing — *"a registry-instance swap
(vanishingly rare — only happens in tests via monkeypatch)"*. A lock-free publish-after-build would
make instance swaps a production event and permanently demote that cache for whichever sub-app
captured the loser's id.

**Recommended: build outside the lock, publish inside it, first writer wins.**

```python
_REGISTRY: Optional[OperationRegistry] = None
_REGISTRY_LOCK = threading.Lock()


def get_registry() -> OperationRegistry:
    global _REGISTRY
    if _REGISTRY is not None:          # fast path: no lock on the overwhelmingly common call
        return _REGISTRY
    registry = OperationRegistry()
    registry.discover()                # built OUTSIDE the lock: `discover` imports eight
    with _REGISTRY_LOCK:               # operation packages, so no lock is held across imports
        if _REGISTRY is None:          # first writer wins, so id(_REGISTRY) is stable forever
            _REGISTRY = registry
        return _REGISTRY
```

Three properties, each for a reason: the published object is always complete (the defect); exactly
one object is ever published (the `id()`-keyed cache); and no lock is held while importing, so a
future module-scope `get_registry()` call cannot deadlock against the import lock. I checked: every
`get_registry()` call in `_gui/` today is inside a function, so holding the lock across `discover()`
would in fact be safe right now — the build-outside form simply makes that not something anyone has
to keep checking. The worst case is one wasted duplicate `discover()` in a genuinely concurrent
first call, with the loser garbage-collected — which is exactly the outcome the commit message
already describes. **The fix makes the commit's own sentence true.**

**The test that goes red on the current code and green after** — deterministic, no sleeps, no
racing. It opens the window by calling `get_registry()` from a second thread *from inside*
`discover`, and `join()`s it, so the interleaving is forced rather than hoped for:

```python
def test_get_registry_publishes_one_complete_registry_to_a_caller_arriving_mid_build(monkeypatch) -> None:
    """A second caller entering during ``discover()`` must get a populated registry, and the same one.

    Before the builder became lazy this was unreachable: ``compose_hub`` built the builder on the
    composing thread, so ``discover()`` had finished before anything was served. A first
    ``/builder/`` and a first ``/analysis/`` request can now enter concurrently.
    """
    import phenotypic._gui._operation_registry as reg

    monkeypatch.setattr(reg, "_REGISTRY", None)
    real_discover = reg.OperationRegistry.discover
    window_opened = threading.Event()
    observed: dict[str, object] = {}

    def discover_with_a_second_caller_in_the_window(self) -> None:
        if not window_opened.is_set():          # only the outermost build opens it
            window_opened.set()
            box: list = []
            thread = threading.Thread(target=lambda: box.append(reg.get_registry()))
            thread.start()
            thread.join()                        # deterministic: the second caller finishes here
            observed["second"] = box[0]
            observed["second_size"] = len(box[0].get_all())
        real_discover(self)

    monkeypatch.setattr(reg.OperationRegistry, "discover", discover_with_a_second_caller_in_the_window)
    first = reg.get_registry()

    assert observed["second_size"] > 0, "a caller arriving mid-build received an unpopulated registry"
    assert observed["second"] is first, "two instances were published; id(registry) is no longer stable"
```

It discriminates all three implementations, which is why it is one test rather than two:

| Implementation | `second_size` | `second is first` | Result |
|---|---|---|---|
| Shipped (publish-before-populate) | `0` — the second caller gets the empty global | n/a | **red on assert 1** |
| Bare publish-after-build, no lock | `> 0` | `False` — outer overwrites the nested publish | **red on assert 2** |
| Recommended (build outside, publish inside, first-writer-wins) | `> 0` | `True` | **green** |

On the fixed code it pays two real `discover()` runs (~2 s); mark it `slow` if the lane's budget
requires it.

### I4. The worker preload guard asserts presence, not position — "before any image work" is not tested

**File:** `tests/unit/cli/test_cli_runtime_preload.py:42-48`.

```python
    assert any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "load_runtime_dependencies"
        for node in ast.walk(main)
    ), f"{relative_path}: main() never calls load_runtime_dependencies()"
```

`ast.walk(main)` finds the call anywhere in `main` — including after the image work, inside an
`if` that never runs, or inside an `except` handler. The test's module docstring says *"Pipeline-
running entry points import the deferred libraries **before any image work**"*, and AC6 says the
same; neither is what is asserted. It also does not check that `load_runtime_dependencies` is
imported in the module at all, so a call with a missing import passes the guard and raises
`NameError` at run start on the cluster.

**The exact mutation that proves it.** In `src/phenotypic/_cli/_cli_recompile_worker.py`, move
`load_runtime_dependencies()` from `:74` (above the `try:`) to the last line of `main`. All four
parametrized cases still pass. Deleting `from phenotypic._startup_perf import
load_runtime_dependencies` at `:27` while keeping the call also passes.

**Proposed fix** — assert the call is the first *statement* of `main` (after an optional docstring),
which is what three of the four files already do and what the fourth
(`_cli_staged_slurm_worker`, deliberately after `parse_args`) can be spelled as an explicit
exception:

```python
FIRST_STATEMENT_WORKERS = (
    "_cli/_cli_process_single.py",
    "_cli/_cli_recompile_worker.py",
    "_cli/_cli_checkpoint_handler.py",
)

@pytest.mark.parametrize("relative_path", PIPELINE_WORKER_ENTRY_MODULES)
def test_pipeline_worker_entry_preloads_runtime_dependencies(relative_path: str) -> None:
    tree = ast.parse((PACKAGE_ROOT / relative_path).read_text(encoding="utf-8"))
    assert "load_runtime_dependencies" in {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }, f"{relative_path}: the name is never imported"
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    body = [n for n in main.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    calls = [
        i for i, n in enumerate(body)
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name) and n.value.func.id == "load_runtime_dependencies"
    ]
    assert calls, f"{relative_path}: main() never calls load_runtime_dependencies()"
    limit = 0 if relative_path in FIRST_STATEMENT_WORKERS else 1  # staged worker: after parse_args
    assert calls[0] <= limit, f"{relative_path}: preload is statement {calls[0]}, not <= {limit}"
```

---

## Minor

### M1. `render_plot` pays for pyplot on the plotly fast path

`src/phenotypic/_gui/analysis/_render.py:61`. The local `import matplotlib.pyplot as plt` is the
first statement of `render_plot`, but `plt` is used only at `:87`, inside the
`figure is None` matplotlib fallback — the function returns at `:77`
(`return _render_output(figure, ...)`) whenever the node produced a plotly figure. Every analysis
node that renders through plotly therefore still pays the ~270 ms on the first call. Moving the
import into the `try:` block at `:79` (or immediately before `:87`) is correct, costs nothing, and
keeps the `DEFERRED_SITES` row valid — the escape checker only requires an enclosing function that
binds the name. Same shape as Phase 1's M5.

### M2. WITHDRAWN — `import importlib.util` costs nothing, and the open question T5 raised is now closed

I raised this as a finding on the theory that Step 5b.1's widening of `import importlib` to
`import importlib.util` (`src/phenotypic/_startup_perf.py:26`) added startup cost for a
`DEFERRED_OPTIONAL_MODULES` loop that is empty by design. The measurement refutes it:

```
modules added by importlib.util: 0 []
import phenotypic module count: 71
```

`importlib.util` is already in `sys.modules` from interpreter startup, so the widened import adds
**zero** modules, and `import phenotypic` is still **71** — byte-identical to the Phase 1 figure.
There is no cost to object to, and the "dead code for an empty tuple" reading is wrong on its own
terms: Step 5b's whole point is to make the required/optional split a property the code enforces
rather than a fact someone remembers, and an empty tuple with a live `find_spec` loop is what that
enforcement looks like. **No action.**

This also closes the open question Task 5's own commit message raised — *"Step 5b.1 changes import
importlib to import importlib.util in `_startup_perf`, which `_startup_perf` imports at import
phenotypic. Task 7 should re-measure the module count rather than carry Phase 1's 71 forward."*
Re-measured under the `--all-extras` venv: **unchanged at 71**. Task 7 may carry 71 forward.

### M3. Three stale references to the eager builder survive in `_app.py`

- `src/phenotypic/_gui/shell/_app.py:8` — the module docstring still routes
  `` `/builder/...` → builder Dash factory (eager — small).`` It is the sentence a reader meets
  first, and it now says the opposite of the code.
- `:254` — `compose_hub`'s `progress` docstring still gives `"builder"` as an example tick label.
  `_tick("builder")` was removed with the eager block, so the surviving labels are
  `"custom operations"`, `"sub-app modules"`, `"shell"`, `"run console"`, `"browse"`.
- The plan's own Step 6 check (`git grep -n "builder_app" -- src/phenotypic/_gui/shell/_app.py`,
  *"Expected: no output"*) contradicts the replacement text the same step supplies, which binds
  `builder_app` inside `_build_builder`. This is a **third** plan-text defect, and the commit
  message reports only two. It is harmless — the implementation is right and the plan's expectation
  is wrong — but the self-report is incomplete on the point the brief asks about.

### M4. A builder build failure now surfaces per-request instead of at boot, and repeats

`_app.py:543-556`. Previously an exception inside `builder.create_app` (a duplicate Dash component
id from a `PHENOTYPIC_PRELOAD_MODULES` custom operation, a missing asset) failed the launcher
loudly before the server bound its port. Now it raises inside `_SessionProxy.__call__`, reaching
the user as a 500 on `/builder/`, and because `ToolSession.get()` correctly does not cache a failed
build (a strength, above) it re-runs the failing build — and its full import cost — on **every**
subsequent `/builder/` request. That is the right trade for the common case, but it is a
user-visible change in how a broken builder presents, and spec B2 discusses only the latency move.
Worth one sentence in `_gui/CLAUDE.md`'s new gotcha, which currently says only that the builder is
built on first request.

### M5. The drift test is one `sorted()` weaker than the claim it stands for

`tests/unit/ci/test_startup_imports.py:265`:
`assert list(option.type.choices) == sorted(available_modes())`. The claim the change rests on is
that the **help text is byte-identical**, i.e. that the new `sorted(get_args(DetectMode))` equals
the old `list(available_modes())`. Wrapping the right-hand side in `sorted()` makes the assertion
insensitive to `available_modes()`'s own ordering, which is the property doing the work
(`_detection_mode.py:97` returns `tuple(sorted(...))`). Dropping the `sorted()` pins the real
invariant and costs nothing. No live defect — the two are equal today.

### M6. `phenotypicCLI` gets `__getattr__` but no `__dir__`, unlike the three lazy `__init__`s

`src/phenotypic/phenotypicCLI.py:318`. The plan gave each lazy package `__init__` a `__dir__` so
`dir()` and `inspect.getmembers` keep seeing the full surface; the CLI module did not get one, so
until `_load_cli_runtime()` has run, `dir(phenotypic.phenotypicCLI)` omits all 28 names. I checked
`docs/source/` — the CLI module is not autodoc'd (only `docs/source/_downloads/phenotypic-cli.py`
imports `phenotypic_cli` by name) and no test enumerates it, so this is a note, not a fault. If a
Sphinx page is ever pointed at the module, the surface will look shrunken.

### M7. The execution ledger has no Task 6 entry, and Task 5's entry stops before its commit

`.superpowers/sdd/plan/progress.md` ends at *"Status: Steps 1–6 and 5b complete, tree clean. Step 7
test surface pending, then commit."* Both tasks are committed. Everything T6 decided — the pyplot
leak and its eleven-module pre-edit verdict check, the two plan-text defects, the `get_registry()`
observation, the two mutation proofs — exists only in the commit message. The ledger is the
artifact Task 7 is told to read and the place rulings are carried forward; a decision that lives
only in `git log` is one `git rebase -i` away from being unfindable. Task 7 should reconstruct a
T5-tail and T6 entry before it starts.

### M8. `phenotypic-tune` runs pipelines and gets no preload

Spec B3 enumerates the preload call sites as the CLI command body plus four `__main__` modules, and
the Non-goals section excludes guarding `phenotypic-tune`'s startup — so this is in-spec and not a
fault. Recording it because D4's wording is *"every pipeline-running entry point"*, and the tune
executor is one. If D4 is meant literally, the enumeration in B3 is the thing to widen.

---

## Spec conformance

| Item | Status |
|---|---|
| **AC2** — `python -m phenotypic --help` exits 0, loads no `HEAVY_STARTUP_MODULES`; `--detect-mode` choices unchanged | Present and falsifiable. Positive controls: exit code, `"Usage:"`, `"--detect-mode"`, `click` loaded. Choices verified equal by construction (`available_modes()` is sorted). |
| **AC2** — `phenotypic-gui --help` | Present. Runs `_launcher.main(["--help"])` in a fresh interpreter per A/P7; covers both M6 and P2 because the package `__init__` runs first. Console-script exit code still covered by `tests/integration/gui/test_console_script.py`. |
| **AC3** — tier 3: no forbidden module before the first request; `/builder/` returns 200 and builds the builder | Present; controls are real (M4 is caught). The asserted set is the spec's A/P11 minimum. **Weakened by I2**: the allow-list subtraction and the unpinned `HUB_WATCHED_MODULES`. |
| **AC3** — "the run console and SLURM observer still start at composition" | Not asserted by the new guard (which passes `start_slurm_observer=False`, as the spec's tier-3 row prescribes). Covered by the spec's named regression net (`test_scheduler_startup_wiring.py`), which is unchanged. |
| **AC6** — preload in the command body and the four pipeline-running entry modules, before any image work | Call sites all present and correctly placed; the four-module set is complete (verified against all twelve `_cli/` `__main__` modules). "Before any image work" is **not** asserted — I4. |
| **AC6** — a failing import aborts before any image is processed | Pinned, including the placement outside the `try:`. |
| **AC11** — no numeric change; no algorithm, constant, public name or `__all__` moved | Holds. No kernel, colour constant or algorithm is in the diff. `_startup_perf.__all__` gains `DEFERRED_OPTIONAL_MODULES` (Step 5b, spec-mandated addition); `_gui/shell/__init__.__all__`, `_launcher.__all__` and `_app.__all__` are byte-identical. `phenotypic.ImagePipeline is phenotypic._core._image_pipeline.ImagePipeline`, so rebinding the name from the package to its leaf changes no identity. |
| **D4 / A-P13** — preload is the first statement of the body, ahead of validation | Done, commented at the site, and pinned by the abort test. |
| **A-P1** — 28 names bound by `globals().setdefault`; module `__getattr__` + `TYPE_CHECKING` | Implemented exactly: 11 statements removed, 11 module keys, 28 names, one `# noqa: F401` on `available_modes` (A/P14 predicted exactly one). The `setdefault` half is untested — I1. |
| **A-P2** — `shell/__init__.py` gets PEP 562 re-exports | Done. `_LAZY_ATTRS` covers all five `__all__` names; `__getattr__` and `__dir__` precede `__all__` (P8's ordering rule is vacuous here — there are no eager imports left). Submodule imports (`from phenotypic._gui.shell import _ids/_routes/_sidebar/_runs_registry`, 4 sites) still resolve through `_handle_fromlist`, because `__getattr__` raises `AttributeError` for unknown names. |
| **A-P11** — tier 3's asserted set is the spec minimum; the rest listed with chains | Done; all seven allowed entries carry a chain and a justification. |
| **D5** — the run console stays eager | Unchanged (`_tick("run console")`, `_app.py:563`). |
| **Design §4** — the builder session is *not* registered with the idle-release thread | Verified at both release paths (`_app.py:660-663`, `:528`). |
| **Design §4** — launcher stops recording "Core library loaded"; `_STARTUP_STEPS` 3 → 2; `_core_import_elapsed` deleted; `StartupReporter`'s API and `IMPORT_STARTED_AT` unchanged | All done. `_STARTUP_STEPS = 2` matches the two surviving `reporter.stage()` calls. `StartupReporter.__init__` still accepts `import_elapsed` (`_startup.py:84,88`); `IMPORT_STARTED_AT` still stamped and re-exported (`__init__.py:24`). |
| **Spec M4 / M5 / M6 mutations** | M4 caught by tier 3 (`detect_before`), M5 by the abort test (`calls == []`), M6 by the GUI tier-2 guard (`dash` watched). |
| **Docs in the same change** | `_gui/FEATURES.md:583` row rewritten as the plan specifies; `_gui/CLAUDE.md` gotcha added as the first bullet under `## Common gotchas`. Root `CLAUDE.md` and `sdk_/CLAUDE.md` updates belong to earlier/later tasks and are out of scope here. |
| **Task 6 Step 9 second bullet** | Correctly reported as having no target; verified — nothing pins the three-stage launcher sequence. |
| **Step 5b** — the preload's optional-extras handling, and its mutation proof | `DEFERRED_OPTIONAL_MODULES` is added, `__all__`-exported, skipped via `find_spec`, and fenced from abuse by `test_the_watched_sets_are_the_ones_the_spec_names`' value-pin on `DEFERRED_RUNTIME_MODULES`. The two-mutation proof is **complete**: `assert unresolved` is `test_startup_imports.py:300` and `assert optional_only` is `:301`, so the recorded results (optuna → 300, rawpy → 301) show the load-bearing `optional_only` branch was genuinely exercised by rawpy, under the then-narrow venv where optuna could only reach 300. The commit's own analysis of why one mutation was insufficient is correct. **Forward note:** under the new `--all-extras` venv optuna is installed, so re-running mutation A would now exit through `:301` with the `conditionally-installed libraries…` message — the ledger predicts exactly this; it is the environment flipping which branch A exercises, not a regression. |
| **AC8** (partial) — ruff finding set not worse | Verified on exactly the 14 files the two commits touch: `All checks passed!`. The T5/T6 claims hold. The full `src/phenotypic` set-diff and the mypy diff remain Task 8's. |
| **AC1, AC4, AC5, AC7 (M1–M3), AC9, AC10** | Phase 1 / Tasks 7–8. Not assessed. |

---

## Checks run

| Risk | Check | Result |
|---|---|---|
| A deferred CLI name read before `_load_cli_runtime()` binds it | Located every occurrence of all 28 `_CLI_RUNTIME_IMPORTS` names in `phenotypicCLI.py`; mapped each line to its enclosing top-level `def` via the file's `^def ` line map | **0 uses outside the four loader-calling functions** |
| A deferred name in a runtime-evaluated annotation (the file has no `from __future__ import annotations`) | Checked every annotation naming a deferred type: `:3104` (local `AnnAssign`, unevaluated per PEP 526) and `:3398` (nested `def`, executes after the loader) | **0 hazards**; no top-level signature names one |
| A deferred name at module scope (decorator, default, class body) | The `--detect-mode` decorator was the only one; now `sorted(get_args(DetectMode))`. Full-file name grep found nothing else outside a function | **0 hits** |
| `mock.patch` teardown leaving a stale or missing binding through module `__getattr__` | Traced CPython `_patch.get_original` / `__exit__` for both `is_local` branches against `phenotypicCLI.__getattr__` | **Correct**; the `delattr` → `hasattr` → rebind path restores the real object |
| `setdefault` actually exercised by its own test | Traced `mock.patch.__enter__` → `getattr` → `__getattr__` → `_load_cli_runtime()` binding all 28 names before the test's own call | **Never reached** → I1 |
| A pipeline-running entry module missed | Grepped all twelve `src/phenotypic/_cli/*.py` with `__name__ == "__main__"` for `from_json` / `.apply(` / `.measure(` / `ImagePipeline` / `GridImage` | **The four preloaded modules are the complete set**; the other eight have zero hits |
| Preload inside the `try:` would pass for the wrong reason | Read `phenotypic_cli`'s handlers: `except KeyboardInterrupt` (`:3289`), `except click.UsageError` (`:3292`), `except click.ClickException` (`:3297`), `except Exception as e: … sys.exit(1)` (`:3300`) | **Placement is load-bearing and pinned** |
| The tier-3 allow-list cancels a watch | Compared `HUB_WATCHED_MODULES` against `HUB_ALLOWED_BEFORE_FIRST_REQUEST` | **Disjoint today, unenforced** → I2; T6's reasoning verified correct |
| `HUB_WATCHED_MODULES` pinned | Searched `tests/` for an assertion over its contents | **None** → I2(b) |
| `plt` escapes its local import in `_render.py` | Listed every `plt` occurrence and every `def` in the file | **1 use (`:87`), inside `render_plot`, after `:61`**; covered by all three checkers in `test_deferred_imports.py` |
| `plt` imported earlier than needed | Traced `render_plot`'s early return at `:77` | pays on the plotly path → M1 |
| The builder session is released somewhere | Grepped every `.release()` call site and both session lists (`_app.py:528`, `:660-663`, `_routes.py:399-404`) | **Never released**, as claimed |
| A failed first builder build is cached | Read `ToolSession.get()` (`_session.py:118-127`) | **Not cached**; `_state` stays `None` on exception, next `get()` retries → M4 |
| The builder session is rebuilt per request | Same | **Memoized** under `self._lock` |
| The `get_registry()` ruling | Read `_operation_registry.py:811-824`; traced both build closures to `get_registry()` call sites (`analysis/_layout.py:64,87,675,846`; `builder/_layout.py:999,1908`; `builder/_callbacks.py` ×5) | **Publish-before-populate**, not "duplicate work" → I3 |
| The builder's registry population was a cross-app side effect lost to deferral | Read `builder/_app.py:83-85,126` and `results_viewer/_app.py:325-328` | **No**; the builder writes `CFG_OPERATION_REGISTRY` on its *own* Flask server, and the viewer builds its own instance idempotently |
| `mounts["/builder"]` consumers left resolving a Flask app | Grepped `tests/` | **2 sites, both updated** (`test_smoke_shell.py:222,403`); a third consumer, `test_no_id_collisions.py:131`, goes through the test client and so builds on demand |
| Blueprint registration on a session-built builder | `ToolSession.get()` builds at that moment, before any request reaches the app | **Works** (plan review C4) |
| Submodule imports broken by the shell `__getattr__` | Grepped `from phenotypic._gui.shell import …` (4 submodule sites in `src/`+`tests/`) against `_handle_fromlist`'s `hasattr` fallback | **All resolve** |
| Plan-text defect 1 (Step 6 replace range) | `git show 6abf8d96:src/phenotypic/_gui/shell/_app.py \| sed -n '536,552p'` | **Confirmed**: 539 is the "eager" comment, 540 is `_tick("builder")` |
| Plan-text defect 2 (Step 9 second bullet) | Grepped `tests/` and `src/` for `_STARTUP_STEPS`, `"Core library loaded"`, `import_elapsed`, `_core_import_elapsed` | **Confirmed**: only `test_startup.py:47,52,61,102`, which are `StartupReporter`'s own tests |
| A third, unreported plan-text defect | Read Step 6's `git grep "builder_app"` check against its own replacement text | **Found** → M3 |
| `available_modes()` ordering (help byte-identity) | Read `_detection_mode.py:95-97` and `sdk_/typing_.py:50` | `tuple(sorted(...))`, 11 names; identical list → M5 on the test's strength only |
| Guards land in a CI shard | `.github/pytest-shards.json` | `tests/unit/ci/` → `foundation-schema`; `tests/unit/cli/` → `cli-packaging`; `tests/unit/gui/` → `gui-browser`. All three new/edited files are covered. |
| Phase 1's I1–I3 and M4 closed before this phase | Read `test_startup_imports.py:42-73, 173-215, 88-91` and `test_deferred_imports.py:259-290` | **All four landed in `ffa77f5b`**; tier 5's exact-equality idiom is the model I2 should follow |
| `DEFERRED_OPTIONAL_MODULES` can be used to smuggle a name out of the required set | `test_the_watched_sets_are_the_ones_the_spec_names` asserts `set(DEFERRED_RUNTIME_MODULES)` by value | **Fenced**; moving a name to the optional tuple goes red |
| The probe helper can pass vacuously | Re-read `tests/_startup_probe.py` (mutation-tested in Phase 1); confirmed non-zero exit or missing marker raises, and the env scrub is unchanged | **No change in this phase** |
| `setdefault` short-circuit — measured, not inferred | Controller-run probe over `mock.patch` + `_load_cli_runtime()` in the `--all-extras` venv | **11 of 11 modules short-circuit**; nothing bound before `__enter__`; patch survives; real function restored on exit → I1, and production confirmed correct |
| Step 5b.1's module-count cost | Controller-run `importlib.util` delta and `import phenotypic` count | **0 added; 71 total**, unchanged from Phase 1 → M2 withdrawn, and T5's open re-measure question closed |
| ruff on the changed files | Controller-run `uv run ruff check` over all 14 files in the two commits | **All checks passed!** |

---

## Recommended changes, in priority order

1. **I1** — measured, so no longer a hypothesis: make the patch-loader test reach `setdefault` by
   deleting one sibling name before calling the loader. **Test fix only; `phenotypicCLI.py` is
   correct and must not change.** One line of setup plus one control assertion.
2. **I2** — pin `HUB_WATCHED_MODULES` by value, assert it is disjoint from
   `HUB_ALLOWED_BEFORE_FIRST_REQUEST`, and assert `loaded_before == []` directly. Six lines. This
   is the finding T6 itself identified in prose and then did not fence in code; it is also the
   Phase 1 I3 defect recurring in a constant created after I3 was closed.
3. **I3** — fix `get_registry()` to publish after `discover()`, under a lock. Separate commit with
   its own test (two threads, one entering mid-`discover`), per the brief's rule on behaviour
   changes; correct the ruling's wording in the ledger at the same time. This is the only
   **code** fix in the list.
4. **I4** — assert the preload's *position* in `main`, and that the name is imported.
5. **M1** — move `import matplotlib.pyplot as plt` past `render_plot`'s plotly early return.
6. **M3** — update `_app.py:8` and `:254`; record the third plan-text defect.
7. **M7** — reconstruct the Task 5 tail and a Task 6 entry in `.superpowers/sdd/plan/progress.md`
   before Task 7 reads it. Carry in: `import phenotypic` re-measured at 71 (unchanged), ruff clean
   on all 14 changed files, and the I3 correction.
8. **M4, M5, M6, M8** — polish. (**M2 is withdrawn**; the measurement refuted it.)

None of 1–8 blocks Task 7. Items 1–4 should land before the branch merges.

---

**Verdict: Ready for Task 7** — no Critical findings and no blocking defect. Four Important
findings (I1 measured false green on the `setdefault` mechanism — test fix, production is correct;
I2 the tier-3 allow-list can cancel its own watch and `HUB_WATCHED_MODULES` is unpinned; I3 the
`get_registry()` publish-before-populate race that T6 newly exposes and rules benign on incorrect
grounds — the one code fix; I4 the worker preload guard asserts presence rather than position) must
land before merge. M2 is withdrawn: the measurement refuted it.
