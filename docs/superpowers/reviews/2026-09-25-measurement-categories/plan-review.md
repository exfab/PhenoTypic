# Plan review: measurement categories (pre-dispatch gate)

- **Plan:** `docs/superpowers/plans/2026-09-25-measurement-categories/plan.md` (commit `038c6948`)
- **Spec:** `docs/superpowers/specs/2026-09-25-measurement-categories/design.md`
- **Tree reviewed:** worktree `measurement-tags`, branch `feat/measurement-tags`
- **Reviewer:** plan-feasibility reviewer. This review is analysis only; nothing in the plan or in `src/` was edited.

## Summary verdict

**FEASIBLE WITH CONCERNS. It is not dispatch-ready until C1 and H1 are fixed.**

The design holds up against the real code:

- The `CATEGORIES` enum shape works.
- The `Entry.categories` normalizer works.
- The dual-hook rename guard works.
- The single finalize call site is correct.
- The split refactor works.

I verified the Enum and dataclass mechanics empirically on 3.12.10.

Two plan edits are mechanically wrong and would break things the plan never tests:

- **C1.** Task 7 changes the row shape of `_render_info_table`. It misses its second caller, `schema/_quality_check.py`. Every `QualityCheck` subclass would then raise `IndexError` on import. `phenotypic.analysis` would stop importing, which silently breaks both CSV splits in production.
- **H1.** Task 1 renames the table caption but leaves a docs-extension test that counts the old caption. Task 1's own Step 8 would then go red.

The remaining findings are coverage gaps, test-cadence gaps, and documentation drift.

**Counts:** 1 CRITICAL, 3 HIGH, 7 MEDIUM, 10 LOW.

---

## CRITICAL

### C1: Task 7 breaks `_render_info_table`'s second caller, so every QC module fails to import

- **Evidence:**
  - `src/phenotypic/schema/_quality_check.py:58-62` builds **5-tuples**, `(f"QC_{slug}_{m.label}", m.desc, m.bio_desc, m.image, m.use_badge)`, and calls `_render_info_table(rows, ...)`.
  - Task 7 Step 3 changes `_render_info_table` to compute `has_cat = any(row[5] for row in rows)` and to unpack `for name, desc, bio, img, use, cats in rows:`.
  - `row[5]` raises `IndexError` on a 5-tuple.
  - The call runs at **class-definition time**. `analysis/abc_/_quality_check.py:493-511` (`QualityCheck.__init_subclass__`) calls `QUALITY_CHECK.append_rst_to_doc(...)` for every subclass that has a docstring and a `name`. `ICC`, `MaxModifiedZScore`, `RelativeMAD`, `TukeyOutlierFraction`, Count, Occupancy and others all qualify.
- **Why it breaks:**
  - `import phenotypic.analysis` raises.
  - `util/_measurement_outputs._discover_measurement_producers()` imports `phenotypic.analysis` (line 140). So `split_measurements` and `split_measurements_by_category` both raise.
  - In the CLI those calls sit inside `_guarded_terminal_best_effort`. The feature split and the category split therefore **silently disappear** from every run, leaving only a WARNING.
  - `_emit_analysis_outputs` and QC would also fail.
- **Why the plan does not catch it:**
  - Task 7 Step 4 runs only `tests/unit/schema`.
  - Tasks 8 and 9 run only the docs-extension test and a Sphinx build. Under autodoc an import failure is a warning, and "exit 0 is not the check".
  - There is no Phase 3 gate (see M2), so the first detection is Task 10's full regression.
- **Fix to the plan text.** Task 7 must add `src/phenotypic/schema/_quality_check.py` to **Files**, and change its row comprehension to a 6-tuple ending `m.category_badges`. Alternatively, keep `_render_info_table` tolerant, for example `cats = row[5] if len(row) > 5 else ""`, but the explicit 6-tuple is cleaner and mypy-checkable.
- **Also add a guard.** Add a Task 7 test that `QUALITY_CHECK.append_rst_to_doc("Doc.", check_name="Count")` renders. Add `tests/unit/sdk_/test_quality_*_info.py` and `tests/unit/util` to Task 7 Step 4.

---

## HIGH

### H1: Task 1's caption rename breaks an existing docs test that Task 1 Step 8 expects to pass

- **Evidence:**
  - `tests/unit/docs/test_measurements_ref_extension.py:137` asserts `combined.count(".. list-table:: Category:") == len(public_classes)`.
  - Task 1 Step 5 changes the caption to `Metric family: **...**` (`_measurement_info.py:163`).
  - Task 1 Step 8 runs `tests/unit/docs/test_measurements_ref_extension.py` and expects "all PASS". It will fail, with the count at 0.
  - Task 8's Files list edits lines 67-78, 93-113 and 115-125 of that test file, but not 137.
- **Fix.** Task 1 Step 5 (or the Files list) must also change line 137 to `".. list-table:: Metric family:"`. A repo-wide grep for `Category: ` in `tests/` finds no other hits.

### H2: No test runs between Task 7 and Task 10, and the Phase 2 gate can be voided by parallel chains

- **Evidence:**
  - The plan defines a Phase 0 gate and a Phase 2 gate, but Phase 3 (Tasks 7-9) has **no affected-surface gate**. Task 7 edits `_measurement_info.py`, the base of every schema, and C1 shows that edit's blast radius reaches `analysis`, `util` and `_cli`.
  - Separately, the plan says "T4→T6 and T7→T9 are independent chains … can run in parallel". If both chains run in the same worktree, the Phase 2 gate measures a union of two trees. With T7 in flight, the gate would fail for C1's reason and be misattributed to Phase 2. Root `CLAUDE.md` names this failure mode: "A parallel gate measures ONE tree".
- **Fix.**
  1. Add a **Phase 3 gate** after Task 8. Derive it mechanically from the importers of `phenotypic.schema._measurement_info`, `_quality_check`, `phenotypic.analysis` and `phenotypic.util`. At minimum it must cover `tests/unit/schema`, `tests/unit/sdk_`, `tests/unit/util`, `tests/unit/cli` and `tests/unit/analysis`.
  2. State that each phase gate runs in a **worktree detached at the phase's last commit SHA**, or else serialize the two chains.

### H3: The dispatch commands contain the literal `docs/source`, which this session's worktree guard refuses

- **Evidence:** the team lead's brief says the guard "rejects Bash commands containing the literal path segment `docs/source`". The plan puts that literal in executable commands at:
  - Task 1 Step 9 (`git add … docs/source/explanation/metadata_namespace.md`)
  - Task 8 Step 6 (`uv run ruff check --fix docs/source/_extensions/measurements_ref.py …`, `git add docs/source/_extensions/… docs/source/_templates/…`)
  - Task 9 Step 6 (`git add docs/source/explanation/… docs/source/tutorials/…`)
  - The sbatch script (`docs/source docs/_build/categories`). That one runs in Slurm, not in this session, so it is probably fine, but the submission line only passes the script path.
- **Why it matters:** a subagent executing these steps verbatim is refused mid-task, after the tests pass but before the commit. Under the orchestrate-subagent round-trip, the orchestrator hits the same refusal.
- **Fix:** rewrite these command lines with a glob (`docs/sour*/…`) or with `git add -u` plus explicit new-file paths. Alternatively, have the orchestrator stage them. Add a one-line note to Global Constraints.

---

## MEDIUM

### M1: Stale skill example teaches the removed API

- **Evidence:**
  - `.claude/skills/adding-an-operation/SKILL.md:76` has a code example `def category(cls) -> str:`.
  - The plan's Step 7 fixes only line 37 (prose).
  - Step 3's perl runs over `src tests` only.
  - After Phase 0, following the skill produces a class that raises `TypeError` at import, which is exactly what the guard is for, but the skill is the first thing an agent reads when adding an enum.
- **Fix:** add `SKILL.md:76` to Step 7, or add the path to the perl pathspec.

### M2: Phase 0 gate surface is derived from package names, not from the importers of the edited modules

- **Evidence:**
  - The gate is `git grep -lE 'phenotypic\.schema|phenotypic\.util|_cli_readme_generator' -- tests`, which gives 180 of 928 test files.
  - Task 1 also edits `_gui/results_viewer/_scatter_tab/_grouping.py:71`, `colony_view/_grid.py:128`, `sdk_/_metadata_helpers.py:357`, `sdk_/constants_.py:78,145` and `measure/_measure_symzones.py:256`.
  - The tests of those modules (`tests/unit/gui/results_viewer/test_scatter_grouping.py`, `tests/gui/results_viewer/colony_view/test_grid*.py`, `tests/unit/measure/test_measure_symmetric_zones.py`, `tests/unit/gui/results_viewer/test_metadata_prefix_predicates.py`) import those modules, not necessarily `phenotypic.schema`.
  - Root `CLAUDE.md` asks for surfaces "derived from importers, mechanically".
- **Fix:** add the importers of every Task-1-edited module to the gate list. The `git diff --name-only` of the Task 1 commit, mapped to test importers, is the mechanical form.

### M3: Task 4, Task 6 and Task 8 run discovery and resolution three different ways, and the spec is contradicted on caching

- **Evidence:**
  - Spec §4.1 says `.members()` is "built lazily … and cached".
  - The plan's `members()` docstring says "**Not** cached, so schema classes registered later are seen", and `test_members_finds_tagged_public_members` depends on it not being cached.
  - `util._member_for_column` uses the `lru_cache`d `_public_info_classes()`.
- **Assessment:** the uncached choice is correct. It avoids stale results under monkeypatch, and the cost is negligible: one walk over ~40 classes per call, called only by docs and tests. But the plan silently deviates from an approved spec.
- **Fix:** add one line to the plan, or amend the spec, recording "not cached (deliberate)".

### M4: The README Categories section documents *configured* columns, where the spec says *present* columns

- **Evidence:**
  - Spec §5.5 says the section "lists each category that has columns *present in this run*".
  - The plan's `_generate_categories_section` derives columns from the configured measurers' `get_measurement_infoclasses()`. `READMEGenerator` has no access to the frame (`_cli_readme_generator.py:41-68`).
  - For example, a measurer that fails every image still gets a Categories entry, and its file will not exist.
- **Assessment:** this is a reasonable approximation, consistent with how the existing measurement tables are generated. The spec wording is what's wrong.
- **Fix:** either amend spec §5.5 to "configured", or say so in the plan's Task 6 Interfaces.

### M5: The Task 1 perl misses nothing in code, but several prefix-wording sites are absent from Step 7's list

- **Verified clean:**
  - `git grep` of `\.category\b` without `()` finds only operation-registry, warning and `unicodedata` uses, all unrelated.
  - `getattr/hasattr(..., "category")` appears only in `_gui/builder/_linear_layout.py:54` and `_preview_callbacks.py:53` (`OperationInfo.category`, unrelated).
  - No `CATEGORY` member names exist (`_curation.py:21` deliberately avoids one).
  - The "leave alone" list is correct.
- **Missed wording:**
  - `post/_append_string.py:22` and `post/_expand_metadata.py:26` ("The schema category …").
  - `_prepend_string.py:22` (same pattern; I have not checked the exact line).
  - `sdk_/_rembi_manifest.py:37` ("`<Category>_` prefix").
  - `schema/_curation.py:21`, whose comment says `CATEGORY` is "a reserved `MeasurementInfo` property". After the rename that is false, and the reserved name becomes `METRIC_FAMILY`.
  - `src/phenotypic/_cli/CLAUDE.md:865`, where the deliverables inventory should gain `measurements_by_category/`. The spec names only root `CLAUDE.md`, but `_cli/CLAUDE.md` owns the "full file inventory".
- **Contradiction:** the plan says "Leave alone: … all test docstrings", yet the perl pattern *will* rewrite `tests/unit/gui/results_viewer/test_measurement_prefixes.py:9`'s docstring (`TEXTURE.category()`). That is harmless, but the statement contradicts the command.

### M6: Task 1 Step 9 ruff invocation skips the new test files and may churn untouched lint

- **Evidence:** `uv run ruff check --fix $(git diff --name-only -- '*.py')`:
  - (a) Excludes untracked files, so the two new test files are never linted.
  - (b) Autofixes pre-existing lint in ~60 files the perl touched, adding unrelated churn to a "mechanical" commit.
- **Fix:** pass the new files explicitly. Consider running `ruff check` (no `--fix`) first to see what would change.

### M7: `Entry.categories` annotation makes mypy flag `MeasurementInfo.__new__`

- **Evidence:**
  - The field is declared `categories: "CATEGORIES | Iterable[CATEGORIES]"`.
  - `obj.categories = entry.categories` assigns that union to an attribute annotated `frozenset[CATEGORIES]` (Task 2 Step 4.5).
  - mypy reports an incompatible-types error.
  - Task 10 Step 3 reports "new errors relative to main", so this surfaces late.
- **Fix:** annotate the field `frozenset[CATEGORIES]` and accept the wider input via `# type: ignore[arg-type]` at call sites, or `cast` in `__new__`. Either way, state the choice in Task 2.

---

## LOW

1. **Python 3.11 is in CI but unverified.**
   - `pyproject.toml` has `requires-python = ">=3.11, <3.13"`. `.github/workflows/run-pytest.yml:94` and `run-pytest-full.yml:60` run 3.11 and 3.12.
   - The plan's "Tech Stack: Python 3.12" and "Python 3.12 builds enum members before `__init_subclass__`" cover only 3.12. No 3.11 interpreter is installed here.
   - The dual-hook guard (member `__new__` plus `__init_subclass__`) is robust to either ordering. `EnumType.__new__` also unwraps a `RuntimeError` from `__set_name__` to its cause on 3.11. But the claim should read "3.11 and 3.12; guard is in both hooks so ordering doesn't matter", and CI's 3.11 leg is the real check.
2. **`test_members_must_be_category_entries` never reaches `__new__`.** It calls `_validate_entry` directly, so deleting the call from `CATEGORIES.__new__` would still pass. The limitation is acknowledged in the test comment. Acceptable, but it is not the "CATEGORIES rejects a non-CategoryEntry value" guard the spec §7 row describes.
3. **The call-site test matches `ast.Name` only.** A future `_cli_output_manager.split_master_by_category(...)` attribute call would escape it. Consider also matching `ast.Attribute` with `attr == name`.
4. **The recompile coverage is the SLURM finalizer from shards, not "a recompile of an existing finalized tree".** Spec §7's CLI row and Review Focus #5 say "recompiling a run finalized before this feature". The extended test starts from shards with no prior `deliverables/`. The path is the same (`finalize_run`), so this is a wording gap, not a coverage hole.
5. **The Categories page header is `Metric family`; spec §6.1 says `Family`.** The plan's choice is better; note it.
6. **`Mapping` import.** Task 5 says twice to add `Mapping` to the `typing` import. The module already imports `Sequence` from `collections.abc` (`_cli_output_manager.py:18`), and ruff's UP035 prefers `collections.abc`. Say it once, from `collections.abc`.
7. **`_columns()` error text is stale.** The message names only `split_measurements()` and `generate_output_key()` (`_measurement_outputs.py:92-95`); the new public function also raises it.
8. **Task 1 Step 8's focused list omits the tests that read `.CATEGORY` directly:** `tests/unit/sdk_/test_quality_check_info.py`, `test_quality_count_info.py` and `test_quality_se_info.py`. The Phase 0 gate probably covers them, but they are the direct consumers.
9. **README text hand-writes `measurements_by_category/`.** It is documentation text, not a path, so it does not strictly violate "never hand-join". Interpolating `DIR_MEASUREMENTS_BY_CATEGORY` keeps it in lock-step with the constant.
10. **`"StartingMetrics" in entry.categories` is `True`.** The probe shows a raw string is a member of a frozenset of `CATEGORIES` (str-mixin hash and equality). This is harmless, because every *write* path is `isinstance`-guarded. But a reader might assume the frozenset is type-exact. Mention it in the `_normalize_categories` docstring.

---

## Validated aspects

- **The `CATEGORIES` shape is sound on 3.12.** Probe results:
  - Only `STARTING_METRICS` is a member. The bare annotations `label`/`desc`, the staticmethod `_validate_entry`, the properties, the classmethod `in_order` and the method `members` are all ignored.
  - `str()`, f-string formatting, `CATEGORIES("StartingMetrics")` lookup, pickling round-trip and frozenset membership all work.
  - `display_name` of `CIELabColor` is `CIE Lab Color`.
- **The `Entry` extension works.** A frozen, slotted, KW_ONLY dataclass with a `frozenset()` default and `object.__setattr__` in `__post_init__` normalizes a bare member to `frozenset({member})` and stays hashable. No test pins Entry's field list; I checked `fields(Entry)`, `asdict`, `__match_args__` and attribute-set assertions.
- **The rename guard fires on both paths.** Measured order for a member-ful class is `['member_new', 'init_subclass']`:
  - The member-ful legacy class raises `TypeError` from member `__new__`.
  - The member-less legacy class raises `TypeError` from `__init_subclass__`.
  - No `RuntimeError` wrapping on 3.12.
  - A normal subclass is unaffected.
- **The rename patterns are complete for code.**
  - 44 src files and 17 test files, matching the plan's count.
  - The `sdk_/constants_.py` `category` definers (`GAMMA_ENCODINGS` via `ConstantLabels`, and `PIPE_STATUS`) are `MeasurementInfo` subclasses and must be renamed; they are.
  - `hasattr`-style duck checks in src target `OperationInfo`, not `MeasurementInfo`.
  - No user-docs page or notebook outside `metadata_namespace.md` uses the API.
- **The symzones dead filter is real.** `SYMMETRIC_ZONES.CATEGORY` is a class-level `property` object, so the filter is a no-op. Deleting it is behaviour-neutral and avoids a nonsense `METRIC_FAMILY` rewrite.
- **There is a single finalize call site.** `split_master_by_feature` is called only from `finalize_post_master_outputs` (`_cli_output_manager.py:1292-1299`), and that function's only caller is `_cli_finalize_run.py:552`. Putting the category split beside it inside `_guarded_terminal_best_effort` is correct and minimal.
- **All the names the plan relies on exist**, with the signatures it uses:
  - `_guarded_terminal_best_effort` (`:91`), `normalize_measurement_metadata_columns` (imported at `:43`), `atomic_write_with_writer` and `PARQUET_WRITE_OPTIONS` (in the sdk_ import block at `:48-68`).
  - `_producer_column_groups`, `_public_info_classes` and `_describe_column` (`_measurement_outputs.py`).
  - `_section_label` (`measurements_ref.py:49`), which yields `measurement-info-size` for `SIZE`.
  - The helpers and fixtures the tests append into: `_write_parquet` with a `Size_Area` column (`test_cli_recompile_slurm.py:45-56`), `IMAGE` and `patch` imports, and `TestAggregateMeasurementsAutoResolve` (`test_cli_output_manager.py:307`).
- **The monkeypatch seams reach the code.**
  - `_category_column_groups` reads module-global `CATEGORIES` and `_member_for_column` at call time.
  - `_member_for_column` calls the module-global `_public_info_classes`. Patching the attribute replaces the cached function, so there is no `lru_cache` trap.
  - `split_measurements_by_category` is imported by name into `_cli_output_manager`, so the failure-isolation patch reaches it.
- **Context columns.** `OBJECT.LABEL` and `Metadata_Dataset` are context in the existing feature split (`tests/unit/util/test_measurement_outputs.py:45-70`), so the new test's `_CONTEXT` is right. Categorized columns are drawn only from non-context columns, so polars never sees a duplicate name, and the producer-ownership invariant test guards the one hole.
- **The three tagged families are owned by producers.** SIZE has exactly 14 members (`_size.py:28-120`), and `ColorLab` medoid lines are at 26-28 as the plan says.
- **sphinx-design 0.6.1 in `.venv` registers `bdg-ref-<color>-line`** (`badges_buttons.py:40`) and emits `sd-outline-<color>`, which Task 9's HTML grep relies on. `"dark"` is in `SEMANTIC_COLORS`.
- **The docs toctree string, navbar markup and anchor names are correct.**
  - The toctree string in Task 8's new test matches the implementation's join.
  - The navbar markup matches the existing entries (`navbar-nav.html:81-92`).
  - `measurement-categories` and `measurement-category-*` do not collide with existing anchors.
  - No test enumerates `schema.__all__` without an `issubclass` filter, so exporting `CATEGORIES`/`CategoryEntry` is safe.
- **No `*_REVISION` bump is needed.** The splits are derivations of the mirror, and no completion proof enumerates `measurements_by_feature/`.

## Concurrency

The only concurrency surface is the existing `--wait` double aggregation. It runs in-process and in the finalizer, serialized by `.aggregate_publication.lock`.

The category split is deterministic: dict order follows the `CATEGORIES` declaration, and column order follows master order. So the two writers produce identical bytes, as the feature split already does. Each write is atomic (`atomic_write_with_writer`) and runs under `publication_commit(commit_guard)`.

No new shared state is introduced. The parallel-chains hazard is a test-gate validity issue, covered in H2, not a runtime race.

## Verification record

**Run by the team lead at my request** (3.12.10, verbatim):

```
members: ['STARTING_METRICS']
StartingMetrics StartingMetrics Starting Metrics True True True True True
CIELabColor -> CIE Lab Color
entry: frozenset({<CATEGORIES.STARTING_METRICS: 'StartingMetrics'>}) True
memberful TypeError L defines category; use metric_family ['member_new']
memberless TypeError L2 defines category; use metric_family ['init_subclass']
ok order: ['member_new', 'init_subclass'] [<OK.V: 'Ok_V'>]
```

No 3.11 interpreter is installed, so the 3.11 run was not performed.

**Verified by reading source:**

- C1: `_quality_check.py:58-62`, `analysis/abc_/_quality_check.py:493-511`, `_measurement_outputs.py:140`.
- H1: `test_measurements_ref_extension.py:137`.
- All the line references in "Validated aspects".
- `.claude/skills/adding-an-operation/SKILL.md:76`.
- The CI matrix and `requires-python`.

**Not verified:**

- 3.11 enum ordering (see LOW 1).
- The exact full-suite impact of the new `deliverables/measurements_by_category/` folder on the before/after `rglob` snapshot tests in `tests/unit/cli/test_cli_migrate_mode.py` and `test_finalize_run.py`. These compare two runs of the same code, so they are probably unaffected. Task 10 is the check.
