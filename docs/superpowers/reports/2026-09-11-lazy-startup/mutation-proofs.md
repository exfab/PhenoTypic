# Lazy startup — mutation proofs M1–M7 (Task 7, Step 3)

Every guard in this change is required to be **proven able to fail**, by reintroducing the defect
it guards. That rule exists because this change has already produced **three** false greens — two
caught during Tasks 1–4, and a third (the `setdefault` guard) caught by the Phase 2 gate. A guard
that has never been observed failing is a guard with no evidence behind it.

**Result: 7 of 7 killed, 0 survived, 7 of 7 restored byte-identically.**

## How these were run

Harness: `/bigdata/exfab/anguy344/slurm_logs/run_m_mutations.py`, run in a background slot (a
foreground timeout is what SIGTERMs a harness mid-mutation and leaves a mutant in the tree).
It enforces the project's mutation-harness rules:

- backs up by **full relative path**, never basename, and **asserts backup count == target count**
  — the check a basename collision fails even when every surviving comparison passes;
- verifies **every** precondition first (each target exists; each anchor string occurs **exactly
  once**) and exits touching nothing if any fails;
- **refuses to report** unless the baseline is green, because mutation results against a red suite
  are noise;
- **restores and re-verifies after each mutation**, not once at the end.

The orchestrator additionally took its own independent `sha256sum` snapshot of all seven targets
before the run and verified it afterwards (`sha256sum -c`: 7 × `OK`), rather than relying on the
harness's own "restored cleanly" claim. `git status` shows no source file modified.

## Results

| ID | File | Mutation | Guard | Result |
|---|---|---|---|---|
| M1 | `sdk_/__init__.py` | add eager `from . import colourspace` | `test_cli_help_loads_no_heavy_module` | **KILLED** — `test_startup_imports.py:250` |
| M2 | `_core/_image_parts/_grid_image_handler.py` | hoist `from phenotypic.grid import CenteredAutoGridFinder` to module level | `test_deferred_names_are_not_imported_at_module_level[...]` | **KILLED** — `test_deferred_imports.py:198` |
| M3 | `abc_/__init__.py` | add eager `from ._prefab_pipeline import PrefabPipeline` | `test_module_imports_first_in_a_fresh_interpreter[phenotypic.analysis]` | **KILLED** — see note below |
| M4 | `_gui/shell/_app.py` | `_SessionProxy(builder_session)` → `builder_session.get().server` | `test_hub_startup_imports.py` | **KILLED** — `test_composed_hub_builds_the_builder_on_its_first_request`, `:71` |
| M5 | `phenotypicCLI.py` | delete `load_runtime_dependencies()` in `phenotypic_cli` | `test_cli_aborts_before_any_output_when_a_runtime_dependency_is_broken` | **KILLED** — `test_cli_runtime_preload.py:50` |
| M6 | `_gui/shell/_launcher.py` | add module-level `from ..._app import create_app` | `test_gui_help_loads_no_heavy_module` | **KILLED** — `test_startup_imports.py:333` |
| M7 | `enhance/_subtract_opening.py` | add module-level `import cv2` | `test_deferred_names_are_not_imported_at_module_level[...]` | **KILLED** — `test_deferred_imports.py:198` |

### M3 is a kill, but by a different route than the others — recorded rather than smoothed over

M1, M2 and M4–M7 each fail through their guard's own assertion. **M3 does not.** Restoring the
eager `PrefabPipeline` import re-creates a genuine circular import, and the run dies during plugin
collection:

```
File ".../_core/_pipeline_parts/_image_pipeline_core.py", line 32, in <module>
    from phenotypic.abc_ import MeasureFeatures, BaseOperation, ImageOperation
ImportError: Error importing plugin "tests.unit.test_fixtures": cannot import name
'BaseOperation' from partially initialized module 'phenotypic.abc_'
(most likely due to a circular import)
```

So the defect **cannot pass unnoticed** — which is what the proof needs to establish — but the
mechanism is a collection-time `ImportError` that takes the whole session down, not the sweep
guard reporting a failure. Worth knowing when reading a future red run: this defect announces
itself as a broken test session, not as a named assertion, so someone hitting it should look for a
re-introduced eager import in a lazy `__init__` rather than for a logic error in the guard.

## What is *not* proved here

These seven prove each named guard can fail. They do not prove the guard set is *complete* — a
deferred library re-added somewhere no guard watches would still pass. That is what the Phase 1
gate's tier-5 guard (`test_operation_subpackage_loads_no_deferred_runtime_library`, asserting the
whole deferred set is absent from each of 13 subpackages) and the Phase 2 gate's I2 fix (pinning
`HUB_WATCHED_MODULES` by value and asserting it disjoint from the allow-list) exist to cover.
Both were themselves added *because* a mutation showed the earlier guards could not fail.
