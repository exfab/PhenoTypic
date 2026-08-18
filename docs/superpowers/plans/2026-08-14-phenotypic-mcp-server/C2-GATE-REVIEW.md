# C2 gate review — `2cab819bc..95ed54359`

Reviewer: independent gate agent. Analysis only; no source file in the shared
tree was modified. Every mutation below was run against an **isolated copy** of
`src/` + `tests/unit/services` under the reviewer's scratch directory, imported
via `PYTHONPATH`, so the shared worktree stayed clean throughout
(`git status --porcelain` empty at start and end).

**Verdict: no blockers in the shipped code. The cluster is sound to merge.**
One gate weakness (G1) should be closed before the boundary is relied on by
later clusters; it is a defect in the *test*, not in the code being merged.

---

## 1. What was verified, with the evidence

### Sound — proven by mutation, not by reading

| # | Mutation | Result |
|---|---|---|
| M1 | `from phenotypic import gui` added to `_services/tune_spec.py` | **FAIL** — `test_import_purity.py::test_service_module_does_not_import_gui[…tune_spec]`, `imports ['phenotypic.gui']`. The aliased-import fix works. |
| M2b | `import phenotypic.gui.shell._app` added to `_services/tune_spec.py` | **FAIL ×2** — `test_service_module_imports_no_dash` + `test_service_module_does_not_import_gui` |
| M3 | third entry (`gui.shell._sandbox`) added to the **tune_spec** allowlist key | **FAIL** — `test_space_split.py:100` equality test |
| M5 | `_services/runs.py` imports `phenotypic.gui.shell._classifierX` and `…._classifier_evil` (both real modules created for the probe) | **FAIL** — both flagged as offenders. **The prefix match cannot over-match.** |
| M6 | the `_space.py` PEP 562 shim made eager | **FAIL** — `test_lazy_gui_packages.py:56`, `_space dragged ['dash','dash_bootstrap_components','flask','werkzeug']` |
| M7 | `grid_feasibility` re-export removed from `gui/tune/_domain_editor.py` | **FAIL** — `test_grid_feasibility_is_one_function` (ImportError) |
| M8 | `sandbox_fingerprint` re-export removed from `gui/shell/_source_context.py` | **FAIL** — hard ImportError (`_metadata_context` imports it) |
| M9 | `gui/_config.py` gains a parallel `SANDBOX_PRESETS_SUBDIR = "presets"` alongside the import | **FAIL** — `test_tune_presets_dir_is_one_function_and_three_constants` (`locally_defined` assert) |
| M10 | same, with the name dropped from the re-export import | **FAIL** — same test (`reexported` assert) |

**Area 2 is closed and the concern is unfounded.** `tests/unit/services/test_import_purity.py:118-120`
matches on a *dotted* boundary:

```python
if not any(name == entry or name.startswith(f"{entry}.") for entry in allowed)
```

`phenotypic.gui.shell._classifier_evil` and `…._classifierX` do **not** match
`phenotypic.gui.shell._classifier`; M5 shows both being rejected. No blocker.

### Sound — proven by construction

- **Scope leak (area 4): none.** An AST comparator diffed every top-level
  `def` / `class` / assignment between `2cab819bc~1` (8 source files) and
  `95ed54359` (12 destination files), docstrings normalised out:
  **0 definitions lost, 0 duplicated** (the one duplicate, `logger`, predates
  the cluster). 7 bodies differ:
  - `SANDBOX_GUI_DIRNAME` / `SANDBOX_PRESETS_SUBDIR` / `SANDBOX_TUNE_PRESETS_SUBDIR`
    — `str` → `Final[str]` annotation only.
  - `_load_space_source`, `_try_load_pipeline`, `_try_load_spec`, `space_to_spec`
    — the function-local lazy imports were removed in favour of the module-level
    ones D4 preserved. **Declared** in `C2-PROGRESS.md` (D4) and verified
    behaviour-preserving: `phenotypic.tune.TuningSpec is phenotypic.tune._spec.TuningSpec`
    → `True`, likewise `SearchSpace` and `build_pipeline`, and
    `src/phenotypic/tune/__init__.py` is docstring + imports + `__all__` with no
    side effects. `_load_space_source` also swapped its `TuneRunRoot` annotation
    for the `_RunRootLike` Protocol — that is approved D2.
- **D4 is faithful.** The union of the pre-move module-level imports of
  `_export.py`, `_command.py`, `_validation.py`, `_setup_authoring.py` appears
  at module level in `_services/tune_spec.py`, with only the GUI paths rewritten
  to their promoted homes.
- **`_services.tune_spec` is optuna-free.** A module-level import walker
  (`if TYPE_CHECKING:` bodies skipped) starting at `phenotypic._services.tune_spec`
  reaches no `optuna` import; the walker was validated by injecting
  `import optuna` at the top of `phenotypic/tune/_spec.py` in the sandbox, which
  it then reported as `tune_spec -> tune._spec -> optuna`. In fact **no module
  under `src/phenotypic` imports optuna at module level** — every occurrence is
  inside a function or a `TYPE_CHECKING` block. D4's substantive claim holds.
  (But see G4 for why the *test* that is supposed to protect this does not.)
- **Shim completeness (area 5): no missing names.** An AST harness parsed every
  `.py` under `src/`, `tests/` and `scripts/`, collecting from-imports (including
  multi-line parenthesised), aliased attribute access through both
  `import x.y as z` and `from pkg import submod`, fully-dotted
  `phenotypic.a.b.c` attribute chains, and string targets of
  `monkeypatch.setattr` / `mock.patch`, then resolved each `(module, name)` pair
  by `getattr`. **Zero misses across every C2 shim.** The six residual hits are
  all pre-existing and unrelated (`phenotypic.FakeGpuDetector` — registered at
  runtime by a test fake; `phenotypic.abstract` and `detect.nn.Sam2Detector` —
  `scripts/`; `_HAS_NAPARI` — monkeypatch-created; `sdk_.ClipControlMixin`;
  `gui.analysis.__main__.launch_results_viewer`). The harness was validated by
  deleting `write_setup_draft` from the `_setup_authoring` shim in the sandbox,
  which it detected and attributed to `tests/unit/gui/tune/test_setup_authoring.py:10`.
- **mypy (area 7): no new errors, and the instability is refuted here.**
  Pre-cluster and post-cluster source trees were reconstructed in scratch and
  checked separately: **421 errors in 125 files** on both. Line-normalised diff
  of the two full outputs differs in exactly 6 lines, all of them mypy-internal
  type-variable ids in `analysis/abc_/_quality_check.py`
  (`P\`81286` vs `P\`81289`) — i.e. **no new and no removed diagnostics**.
  `_services/tune_spec.py` (1,702 lines) and `_space_view.py` contribute **0**
  errors. Three runs on a byte-identical tree — two sharing a warm cache, one
  with a cold cache — all produced `421 errors in 125 files` with byte-identical
  output. I could not reproduce 420/124; a differing tree state is a likelier
  explanation than mypy nondeterminism. The diff method remains the right
  practice regardless.
- **ruff** clean on all 12 changed `src` paths plus `tests/unit/services/`.
- **Full suite** `tests/unit/gui tests/integration/gui tests/unit/services`:
  **1786 passed, 3 skipped** in 91s (no `deselected` line appeared in my
  invocation; the pass count matches the implementer's report exactly).
- **The M18 interned-string finding is real**, and now quantified. Measured
  across two separate modules in one interpreter: `"presets"` → same object,
  `"tune"` → same object, `"builder_tiles"` → same object,
  `".phenotypic-gui"` → **different** objects, `5` → same, `300` → different,
  `True`/`None` → same. So `SANDBOX_GUI_DIRNAME` was never forgeable (it
  contains `.` and `-`, so CPython does not intern it) — the log's "three of the
  four are short string literals, which CPython interns" is one too many. The
  AST binding check covers all four either way, so nothing follows from it.

---

## 2. Findings, ranked

### G1 — the allowlist exactness pin covers only one of its two keys *(highest; gate defect, not a code defect)*

`tests/unit/services/test_space_split.py:98-105` pins the GUI reach of
`phenotypic._services.tune_spec` to equal its allowlist entry. **Nothing pins
`phenotypic._services.runs`**, and the tier gate itself is subset-only. So the
`_classifier` entry can be widened arbitrarily and the whole suite stays green.

Proven twice:

- **M4** — `"phenotypic._services.runs"` entry changed to
  `{"phenotypic.gui.shell._classifier", "phenotypic.gui.shell._app", "phenotypic.gui"}`,
  no source change → **59 passed**.
- **M15** — entry replaced with `{"phenotypic.gui"}` *and* a genuinely new
  upward import added to `src/phenotypic/_services/runs.py`
  (`from phenotypic.gui.shell._metadata_context import resolve_metadata_csv`)
  → **59 passed**. A one-line allowlist edit dissolves the tier boundary for
  `_services.runs` with nothing failing.

`C2-PROGRESS.md`'s claim that "a stale entry with no matching import fails" is
therefore true only for the `tune_spec` key. Fix: parametrize the equality
assertion over every key in `GUI_IMPORT_ALLOWLIST` rather than hard-coding
`"phenotypic._services.tune_spec"`, so each entry must match a real import
exactly.

### G2 — seven identity assertions cannot fail (area 3, generalised)

Every `assert ... is ...` in `tests/unit/services/` was enumerated and each
asserted value classified by whether a parallel definition would satisfy `is`.
Nine are forgeable; two of those (`SANDBOX_PRESETS_SUBDIR`,
`SANDBOX_TUNE_PRESETS_SUBDIR`) are already covered by the AST binding check
added in the M18 fix. The remaining **seven prove nothing**:

| Name | Assertion | Why forgeable |
|---|---|---|
| `RunMode` | `test_shim_equivalence.py:86` | `typing.Literal[...]` is `typing._tp_cache`d — a parallel `Literal[...]` with the same members **is** the same object |
| `RunStatus` | `test_shim_equivalence.py:86` | same |
| `DEFAULT_STORAGE_ENV` | `test_shim_equivalence.py:172` | `str` `'PHENOTYPIC_STORAGE_URL'` — identifier-like, interned |
| `ExecutionTarget` | `test_shim_equivalence.py:172` | `Literal['local','slurm']` — `_tp_cache` |
| `StorageMode` | `test_shim_equivalence.py:172` | `Literal['spec','local','environment']` — `_tp_cache` |
| `Blocks` | `test_shim_equivalence.py:188` | `Literal['continue','deploy','both']` — `_tp_cache` |
| `SETUP_DRAFT_VERSION` | `test_shim_equivalence.py:225` | `int` `2` — CPython small-int cache |

Proven by mutation, not asserted:

- **M11** — `gui/tune/_command.py` drops `DEFAULT_STORAGE_ENV` from its
  re-export import and defines `DEFAULT_STORAGE_ENV = "PHENOTYPIC_STORAGE_URL"`
  itself → **19 passed** (false green).
- **M12** — same shape with `SETUP_DRAFT_VERSION = 2` in
  `gui/tune/_setup_authoring.py` → **19 passed** (false green).
- **M13** — `gui/shell/_runs_registry.py` defines its own
  `RunMode = Literal["local","slurm","validate","unknown"]` → **19 passed**
  (false green).

`SetupPathPayload` looks forgeable by type but is **not**: `TypedDict` classes
are not cached (verified — two identical `TypedDict` definitions are distinct
objects).

Fix: apply the M18 remedy generally — assert the *binding* from the parsed AST
(name arrives by `ImportFrom` of the canonical module, and is bound by no
module-level `Assign`/`AnnAssign`/`def`/`class`) for every shim, not only
`gui/_config`.

### G3 — `test_state_shim_reexports_every_public_name` asserts `is not None`

`tests/unit/services/test_argv_promotion.py:32-41` checks
`getattr(_state, name) is not None` for `RunConsoleState`, `run_state_to_json`,
`run_state_from_json`, `state_from_controls`, `to_argv`. That is weaker than an
identity check — it passes on a parallel re-definition, on a re-import from a
third module, on anything truthy. Pre-existing (C1, outside this diff); flagged
because it is the same class of hole as G2 and lives in the same directory.

### G4 — the optuna guards are unfalsifiable here and order-dependent everywhere

Two separate problems, both bearing on D4's stated safeguard
("`test_space_module_does_not_import_optuna` still means something"):

1. **`optuna` is not installed in this environment.** It is declared only in the
   optional `[project.optional-dependencies] tune` extra (`pyproject.toml:126-134`),
   and `import optuna` raises `ModuleNotFoundError` in `.venv`. Nothing can put
   it into `sys.modules`, so `tests/unit/gui/tune/test_space.py:331` and
   `tests/unit/gui/tune/test_command.py:235` **cannot fail locally**. The
   implementer's local green on them is not evidence. CI does
   `uv sync --group dev --group test-qt --all-extras`, so the guards are live
   there.
2. **The `sys.modules.pop(...)` + `importlib.import_module(...)` shape is a
   tautology once the module is already imported in the worker.** Demonstrated
   with a synthetic reproduction: import `space_like` (which imports
   `fake_optuna`), pop `fake_optuna` from `sys.modules`, re-`import_module`
   `space_like` → the module is served from cache, is never re-executed, and the
   guard **passes** although the module genuinely does import the forbidden
   dependency. `tests/unit/gui/tune/test_space.py` imports
   `phenotypic.gui.tune._space` in many earlier tests in the same file, so in CI
   this guard is green regardless of what `_space` imports.

Fix: convert both to the subprocess `_PROBE` shape already used in
`test_import_purity.py` / `test_lazy_gui_packages.py`, and — the durable
version — add `optuna` to a forbidden-import subprocess probe over the
`_services` tier so the tune-tier laziness is enforced by the same gate as
`dash`. (Today the property does hold; nothing in `src/phenotypic` imports
optuna eagerly. It just is not *guarded*.)

### G5 — one surviving source-substring assertion

`tests/unit/services/test_image_exts_relocation.py:29-35` asserts
`"_directory_browser" not in inspect.getsource(_classifier)`. That is precisely
the pattern C2-DECISION's standing lesson bans ("No assertion in this codebase
may be a substring search over source text"). It goes green on a
`from phenotypic.gui.builder import _directory_browser as _db` alias, and red on
a docstring that merely names the module. Pre-existing (C1's file, not in this
diff); named because the lesson is now repo policy and this is the last
instance in `tests/unit/services/`.

### Note (cosmetic, no action)

An eager GUI import placed in `_services/sandbox.py` (M2) does not produce a
clean gate failure — it produces a circular-import `ImportError` at conftest
plugin load, taking down collection for the whole session. Loud, so not a hole;
worth knowing when a future mutation on that module reports something odd.

---

## 3. Interfaces, for later clusters (area 6)

### `src/phenotypic/_services/tune_spec.py` — 1,702 lines

**`__all__` is not the surface.** It lists 18 names; the shims import **27 more
public module-level names by explicit name**, plus 8 privates.

- `__all__` (18): `Issue`, `PreparedPipelineExport`, `ValidatedTuneCommand`,
  `apply_space_edits`, `build_tune_command`, `can_deploy`, `export_best_from_run`,
  `export_pareto_pipeline`, `export_winning_pipeline`, `preflight_issues`,
  `prepare_best_from_run`, `publish_prepared_export`, `render_launch_command`,
  `render_tokens`, `space_to_spec`, `spec_path_issue`, `storage_url_preflight_issue`,
  `validate_setup`.
- Public, **not** in `__all__` (27): `Blocks`, `DEFAULT_STORAGE_ENV`,
  `ExecutionTarget`, `SETUP_DRAFT_VERSION`, `SetupAuthoringResult`, `SetupDraft`,
  `SetupDraftCache`, `SetupPathKind`, `SetupPathPayload`, `SetupPathResolution`,
  `SetupPathSource`, `SetupWriteReceipt`, `StorageMode`,
  `authored_content_fingerprint`, `authored_setup_spec_path`,
  `build_authored_setup_spec`, `build_setup_draft`, `grid_feasibility`,
  `load_pipeline_or_spec`, `path_content_fingerprint`, `resolve_picker_payload`,
  `resolve_setup_path`, `setup_draft_from_store`, `setup_path_payload`,
  `setup_path_resolution_from_store`, `write_authored_setup_spec`,
  `write_setup_draft`, `write_setup_draft_receipt`.
  (`SetupPathKind` / `SetupPathSource` are internal type aliases, referenced
  nowhere outside this module — they were internal pre-move too.)
- Privates imported by name elsewhere: `_RunRootLike`, `_apply_edits`,
  `_build_search_space`, `_default_qc_scorer`, `_editable_knobs`,
  `_is_tuning_spec`, `_load_space_source`, `_params_from_best_params_payload`,
  `_recover_typed_choices`, `_try_load_pipeline`, `_try_load_spec`.
- **One upward import**, allowlisted and expiry-tracked:
  `from phenotypic.gui.shell._metadata_context import resolve_metadata_csv`
  (`_services/tune_spec.py:112`).

### `src/phenotypic/_services/sandbox.py` — 219 lines (+14)

`__all__ = ["SandboxRoot", "sandbox_fingerprint"]`. `sandbox_fingerprint` is
new here (moved verbatim from `gui/shell/_source_context.py`; `hashlib` added,
and `hashlib` + `os` were removed from `_source_context.py` — verified no
remaining use of either there). Privates re-exported through the
`gui/shell/_sandbox` shim: `_is_safe_relative_path`,
`_v1_selection_matches_sandbox`.

### `src/phenotypic/sdk_/_io_constants.py` — 2,243 lines (+32)

Four names added, all moved verbatim from `gui/_config.py`:
`SANDBOX_GUI_DIRNAME`, `SANDBOX_PRESETS_SUBDIR`, `SANDBOX_TUNE_PRESETS_SUBDIR`
(each now `Final[str]`) and `tune_presets_dir`. The module has **no `__all__`**
— import by explicit name. Following the `IMAGE_EXTS` precedent, these are
**not** re-exported through `phenotypic.sdk_/__init__.py`; import them from
`phenotypic.sdk_._io_constants` directly. `SANDBOX_BUILDER_TILES_SUBDIR` stayed
in `gui/_config.py`.

### `src/phenotypic/gui/tune/_space_view.py` — 337 lines, new

`__all__ = ["build_space_view", "setup_knob_forms"]`; privates
`_REVIEW_BADGE`, `_categorical_input`, `_domain_editor`, `_knob_form`,
`_range_inputs`, `_tunable_toggle`, all reachable through the `_space` shim's
PEP 562 `__getattr__`.

---

## 4. Coverage statement

All seven requested areas were completed. Nothing was skipped.

- **1 (false greens)** — 15 mutations run (M1, M2, M2b, M3, M4, M5, M6, M7, M8,
  M9, M10, M11, M12, M13, M15), plus a shim-deletion probe validating the
  surface harness and an injection probe validating the optuna walker. Three
  previously-unreported false greens found (G2).
- **2 (prefix over-match)** — closed, no over-match, M5.
- **3 (interned strings)** — M18 reproduced and measured; full `is`-audit of
  `tests/unit/services/` done; seven unguarded assertions reported.
- **4 (scope leak)** — AST comparison of all 8 pre-move files against all 12
  post-move files; D4 confirmed faithful; optuna-freeness confirmed statically.
- **5 (shim completeness)** — AST harness over `src/`, `tests/`, `scripts/`;
  zero misses; harness validated.
- **6 (interfaces)** — above.
- **7 (mypy)** — before/after diff run; no new errors; instability refuted in
  3/3 runs on this tree.

Caveat worth stating plainly: my full-suite figure (1786 passed / 3 skipped)
came from `tests/unit/gui tests/integration/gui tests/unit/services`; I did not
run the whole repository suite, and no test outside those three trees was
exercised.
