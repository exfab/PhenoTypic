# Phase 2 review: `split_measurements_by_category`, the CLI seam, the README

Reviewer: Phase 2 deep-review gate (implementation correctness and test strength).
Scope: `git diff 9f122602 55cf7965`, which covers commits `27004d08` (Task 4), `542dfa44`
(Task 6) and `55cf7965` (Task 5). `f0d6ff71` is a docs-only gate note. Checked against plan
Tasks 4–6, the Global Constraints, the Review Focus, spec §5 (with the amended §5.5), root
`CLAUDE.md` ("Output layout") and `src/phenotypic/_cli/CLAUDE.md`.

**Verdict: pass. No CRITICAL or HIGH findings.** 0 CRITICAL / 0 HIGH / 2 MEDIUM / 9 LOW.

The implementation is correct:

- The category split uses exactly the feature split's context set.
- By construction, a column cannot be both context and a category member.
- `split_measurements` and the extracted `_write_split` loop behave exactly as before.
- Two independent writes produce byte-identical files, so the `--wait` double-writer claim holds for the new folder.
- Every mode that should reach the category split does, and process mode and the chunk writer do not (§C).

The weaknesses are in what the tests pin:

- Nothing pins that the new call runs **inside the publication fence**. A plain `try/except` outside `publication_commit` passes all 338 focused tests (M4).
- The single-call-site AST test misses an aliased call and a `functools.partial` (M12, M13).

---

## A. Execution evidence (run by main, verbatim excerpts)

Probes ran in the live worktree at `55cf7965`. The only uncommitted edits there were Task 9
`.md` prose, with no `.py` files. Mutations ran in a scratch worktree at `55cf7965`. Each
mutated file was restored and re-hashed, and the final `git status` was clean. The full output
is in main's scratchpad at `phase2_mutations.out`.

F is the focused set:

- `tests/unit/util/test_split_by_category.py`
- `tests/unit/util/test_measurement_outputs.py`
- `tests/unit/cli/test_cli_output_manager.py`
- `tests/unit/cli/test_category_split_call_site.py`
- `tests/unit/cli/test_readme_categories.py`
- `tests/unit/cli/test_readme_measurement_tables.py`
- `tests/unit/cli/test_readme_model_section.py`
- `tests/unit/cli/test_cli_recompile_slurm.py`
- `tests/unit/sdk_/test_io_constants.py`

All runs used `-o addopts= -m "not slow"`.

| Probe | Result |
|---|---|
| P0: F, unmutated | `338 passed in 29.63s` |
| P1: ruff on changed files | `All checks passed!` |
| P1: mypy on the 3 changed modules | `Success: no issues found in 3 source files` |
| P2: column order on a finalize-shaped frame | see below |
| P3: two independent `split_master_by_category` writes | `StartingMetrics .csv True`, `StartingMetrics .parquet True` |

P2 (verbatim):

```
ordered ['Metadata_Strain', 'Metadata_Dataset', 'Size_Area', 'Shape_Circularity', 'Metadata_ImageName', 'Metadata_UUID', 'Metadata_ImageName', 'Metadata_ParentImageName', 'Object_Label']
feature ['Metadata_Strain', 'Metadata_Dataset', 'Metadata_ImageName', 'Metadata_UUID', 'Metadata_ParentImageName', 'Object_Label', 'Size_Area']
category ['Metadata_Strain', 'Metadata_Dataset', 'Metadata_ImageName', 'Metadata_UUID', 'Metadata_ParentImageName', 'Object_Label', 'Size_Area']
```

On the real finalize frame, the IMAGE metadata columns and `Object_Label` come *after* the
measurements (`order_measurement_columns`). Both splits move them to the front, in the same
order. Context-first is therefore observable in real output, not just in test fixtures.

### Mutation results

| # | Mutation | Expected | Result | Verdict |
|---|---|---|---|---|
| M1 | delete the category block in `finalize_post_master_outputs` | red | 3 red: aggregate category test, call-site test, recompile-SLURM finalizer test | killed |
| M2 | category block moved before the feature block | green (equivalent) | 338 passed | equivalent; not a contract |
| M3 | bare call, no guard | red | 1 red: `test_failing_category_split_does_not_block_publication` | killed |
| M4 | `try/except` **outside** `publication_commit` | green | **338 passed** | **survives → finding M-1** |
| M5 | category context = columns not in any *category* group | red | 5 red | killed |
| M6 | `_write_split` skips Parquet | red | 2 red (feature and category writer tests) | killed |
| M7 | `mkdir` before the empty check | red | 2 red | killed |
| M8 | `_split_by_groups` in master order | green (my prediction) | 1 red: `test_no_state_file_keeps_master_and_splits_known_columns` | killed, but only through the **feature** split → L-4 |
| M9 | README section without `dict.fromkeys` | green | 338 passed | survives → L-6 |
| M10 | README Categories cell without `in_order` | green (equivalent today) | 338 passed | survives → L-6 |
| M11 | tag `OBJECT.LABEL` (unowned) with `STARTING_METRICS` | red | 2 red: `test_every_categorized_header_is_owned_by_a_producer` and the Phase 1 pin-18 test; **no duplicate-column error** | killed; confirms L-2 |
| M12 | aliased import + call in `_cli_finalize_run.py` | green | 1 passed | **survives → finding M-2** |
| M13 | `functools.partial(split_master_by_category, …)` | green | 1 passed | **survives → finding M-2** |
| M14 | drop `normalize_measurement_metadata_columns` in the category split | green | 338 passed | equivalent on the sole path (post_df is already normalized at `_cli_output_manager.py:1176`) |
| M15 | `_category_column_groups` keeps only the first category | red | 1 red: `test_column_in_two_categories_appears_in_both` | killed |

---

## B. Split semantics (`util/_measurement_outputs.py`)

- **The context set is identical to the feature split's.** Both call
  `_context_columns(columns, _producer_column_groups(columns))` (`_measurement_outputs.py:52` and
  `:80`). M5 shows the tests would catch a category-derived context.
- **A column cannot be both context and a category member.** Category groups are built only
  from `measured`, the exact complement of `context` (`:81-83`). A polars duplicate-name
  selection is therefore impossible by construction, and M11 confirms no duplicate error
  arises. The plan's Review Focus #2 describes the wrong failure mode (see L-2).
- **Within a category there are no duplicates.** `CATEGORIES.in_order` de-duplicates through
  `set()` (`_categories.py:102`), and each column is visited once (`:155`).
- **`split_measurements` is unchanged.** I compared it to
  `git show 9f122602:src/phenotypic/util/_measurement_outputs.py`. The context comprehension
  and the `context + group_columns` selection are the same expressions, moved into
  `_context_columns` and `_split_by_groups`.
- **Frame types.** pandas and polars behaviour is inherited from `_select_columns`, which is
  unchanged, and is tested both ways.
- **Degenerate frames.** A frame with only metadata returns `{}` (tested). A zero-row frame goes
  through the same `select`, and nothing is special about it.
- **Key order.** Keys follow `CATEGORIES` declaration order (`:154`, `:161`), so dict order is
  deterministic. Within a key, columns follow input order.
- **A column resolved by two public infos.** Membership comes from the *first* public class in
  `schema.__all__` whose `member_for_header` matches (`:333-339`). Ownership comes from the
  producer's own infos (`:173`). The two can disagree only if two public classes claim the same
  header. Today the only shared families are `QC` (`_quality_check.py:33`,
  `_metadata_match.py:19`) and `OrientZones` (`_orientation_zones.py:28`, `:153`), and neither
  carries a category. This is a latent hazard only (L-3).

## C. Reachability (read from code)

| Mode | Path to the category split |
|---|---|
| full / measure, local | `phenotypicCLI.py:3268` `aggregate_master_csv` → `_cli_output_manager.py:2117` `aggregate_measurements` → `:1588` → `:1553` `finalize_run` → `_cli_finalize_run.py:552` `finalize_post_master_outputs` → `_cli_output_manager.py:1304-1314`. `measure` shares this main flow. The only mode exit before `:3268` that I found is the process-only exit at `:3195-3211`. |
| full / measure, SLURM finalizer | `_cli_checkpoint_handler.py:195` `_run_finalize` → `:305` `aggregate_measurements`; the sentinel goes through `_cli_sentinel.py:163`. |
| `--wait` | Both of the above, serialized by `.aggregate_publication.lock` (`_cli_output_manager.py:1583-1587`). P3 shows the category writes are byte-identical. |
| recompile, local | `phenotypicCLI.py:4186` (inside `_handle_recompile`, `:4082`) → `aggregate_measurements`. |
| recompile, SLURM finalizer | `_cli_recompile_worker.py:973`/`:975` → `finalize_run`. Under a generation it runs inside `generation_publication_guard` (`:974`) with `commit_guard=None`, the same as the feature split. |
| `--mode migrate` (full-run target) | `_cli_migrate.py:1037` `_publish_migration_aggregate` → `:1051` `aggregate_measurements(no_qc=True, commit_guard=…)`. |
| `--mode process`, local | `phenotypicCLI.py:3195-3211` calls `sys.exit` before aggregation. |
| `--mode process`, SLURM | `_cli_slurm_array_scripts.py:499` selects the `manifest` checkpoint, not `finalize`. `_cli_checkpoint_handler.py:168-186` publishes only the completion marker. |
| chunk writer | `_cli_chunk_writer.py` references neither split; its only mentions are in the docstring at `:8` and `:11`. |

## D. The seam (`_cli/_cli_output_manager.py`)

- **Same guard as the feature split.** The call uses
  `_guarded_terminal_best_effort(commit_guard, …)` (`:1307-1314`), placed directly after the
  feature split. `_guarded_terminal_best_effort` swallows only `Exception` raised *inside*
  `publication_commit`, so a fence rejection still propagates (`:92-108`, `:82-89`). An
  exception anywhere in the split, including `normalize_measurement_metadata_columns`'s
  `ValueError`, is logged, the same as for the feature split. M3 proves the tests would catch an
  unguarded call. M4 proves they would not catch a guard placed outside the fence (M-1).
- **`_write_split` is unchanged in behaviour and log text.** Its loop is identical, line for
  line, to the pre-refactor `split_master_by_feature` body: the same warnings, the same
  `"Split %r: …"` info line, and CSV kept when Parquet fails. The only new log line is
  `"No categorized measurement columns -- skipping category split"`. No test or GUI code
  matches either skip message (grep).
- **Normalization** is applied the same way in both splits (`:1390`, `:1419`). It is redundant
  on the sole path, because `post_df` is normalized at `:1176`. M14 is equivalent, so keeping
  it for parity is harmless.
- **`--wait` determinism** holds, as shown in §A (P3) and §B (key order).

---

## Findings

### MEDIUM

**M-1: Nothing pins that the category split runs inside the publication fence.**
Evidence: M4 replaced the `_guarded_terminal_best_effort(commit_guard, …)` block with a plain
`try/except` outside `publication_commit`, and all 338 focused tests passed.
`test_failing_category_split_does_not_block_publication` proves only that *some* guard exists.
The call-site test counts callers and ignores the wrapper. In a SLURM generation that has lost
ownership, the unfenced variant would still write `measurements_by_category/` into the tree.
The spec requires "inside the same `_guarded_terminal_best_effort(commit_guard, …)` wrapper"
(§5.3). The feature split has the same gap, so this is inherited, but the new call site is
where the requirement is written.

Fix (either or both):

- Extend `test_category_split_call_site.py`. Assert that the `Call` to `split_master_by_category`
  sits inside a `Lambda` that is an argument of a `Call` to `_guarded_terminal_best_effort`,
  whose first positional argument is `Name(id="commit_guard")`.
- Add a behavioural test. Pass `finalize_post_master_outputs` a `commit_guard` whose
  `publication_commit` rejects. Assert that the rejection propagates and that
  `measurements_by_category_dir` does not exist.

**M-2: The single-call-site AST test misses indirect references.**
Evidence: M12 (an aliased import plus call) and M13 (`functools.partial`) both left
`test_category_split_is_called_only_from_finalize_post_master_outputs` green. By reading
`_calls_by_enclosing_function` (`test_category_split_call_site.py:15-35`), it also misses:

- a module-level or class-body call, since it walks only `FunctionDef`s;
- passing the function as a callback (`_guarded(…, split_master_by_category)`);
- `getattr(mod, "split_master_by_category")`;
- a re-implementation that calls `split_measurements_by_category` and
  `_write_split(measurements_by_category_dir(…), …)` directly.

The Global Constraint "called from exactly one place" is therefore guarded only against the
most literal violation.

Fix: make the test reference-based. Walk every module under `src/phenotypic` (not only function
bodies) and collect:

- every `ast.Name`/`ast.Attribute` in `Load` context named `split_master_by_category`;
- every `ast.alias` whose `name` is `split_master_by_category`.

Allow exactly the one reference in `finalize_post_master_outputs` and the import in
`_cli_output_manager.py`, and nothing else. Add a sibling assertion that
`measurements_by_category_dir` is referenced only inside `split_master_by_category` and the
`sdk_` definition and re-export.

### LOW

**L-1: A recompiled or migrated run gains the folder, but its README does not describe it.**
Spec §5.3 advertises `--mode recompile` as the upgrade path. `_handle_recompile`
(`phenotypicCLI.py:4082-…`), the recompile worker and `_cli_migrate.py` never call
`READMEGenerator` (grep finds no README reference in them). The README comes only from the
forward main flow (`phenotypicCLI.py:3320-3337`) and the staged finalizer
(`_cli_checkpoint_handler.py:611`). So an upgraded tree has `measurements_by_category/` and an
old README that lacks both the tree line and the Categories section. This behaviour predates
the change, and spec §5.5 does not promise otherwise. Fix: note it in the Task 9 prose, or
regenerate the README on recompile as a follow-up.

**L-2: Plan Review Focus #2 describes an impossible failure mode.**
The plan says that an unowned categorized column "would be both a context column and a group
column, and the split would select it twice (polars raises)". The implementation makes that
impossible (§B), and M11 shows no duplicate error. The real failure mode is silent: the column
stays context in every file and never appears as a member of its category.
`test_every_categorized_header_is_owned_by_a_producer` still guards it, and M11 kills it. Fix:
correct the plan's rationale sentence so a later reader does not "fix" a non-bug.

**L-3: The member lookup and producer ownership can disagree in future.**
`_member_for_column` returns the first public class in `schema.__all__` order (`:333-339`).
Ownership uses the producer's own infos (`:173`). If a second public class ever claims a
categorized header (the shared-family precedent exists: `QC`, `OrientZones`), the category
comes from whichever class comes first in the list. Fix: add a pin to
`test_split_by_category.py`:
`for c in CATEGORIES: for m in c.members(): assert mo._member_for_column(m.value) is m`.

**L-4: Nothing tests the category split's own column order on a real finalize frame.**
M8 was killed only by the feature split's
`test_no_state_file_keeps_master_and_splits_known_columns`, because `_split_by_groups` is
shared. Every category fixture puts context first, and P2 shows the real frame does not. A
category-specific ordering change, such as bypassing `_split_by_groups` in
`split_measurements_by_category`, would survive.
`test_context_matches_the_feature_split` compares only a 2-column prefix
(`test_split_by_category.py:59-65`). Fix: in
`test_aggregate_writes_category_split_beside_feature_split`, assert the exact column list, as
its feature sibling does at `test_cli_output_manager.py:482-487`. Also assert that the feature
split file exists, since the test name claims it. Build the context comparison on a frame whose
context trails the measurements, and compare the full lists.

**L-5: The Categories section adds an unguarded crash path to README generation.**
`_generate_measurement_table` wraps its body in `except Exception` (`_cli_readme_generator.py`,
the table function). `_generate_categories_section` does not. It iterates each configured
info's members and reads `member.categories`. A malformed info from a third-party
`get_measurement_infoclasses()` used to degrade to one missing table. Now it fails the whole
README:

- in the main flow, `finalization_succeeded = False` (`phenotypicCLI.py:3334-3337`);
- in the staged SLURM finalizer, the error propagates out of
  `_publish_staged_report_and_readme` (`_cli_checkpoint_handler.py:611`).

The likelihood is low, because every `Entry`-built member has `categories`
(`test_measurement_info_format.py` pins this for first-party classes). Fix: use
`getattr(member, "categories", ())`, or wrap the per-info loop the way the table function is
wrapped.

**L-6: The README's de-duplication and `in_order` are unpinned.**
M9 (no `dict.fromkeys`) and M10 (no `in_order`) both survive. M10 is equivalent while only one
category exists. M9 is observable today with two `MeasureSize` instances, or with `MeasureSize`
plus a measurer that shares `SIZE`. Fix:

- Add `_generator(MeasureSize(), MeasureSize())` and assert `section.count(f"`{SIZE.AREA}`") == 1`.
- For ordering, monkeypatch a two-category member, as `test_column_in_two_categories_appears_in_both` does.

**L-7: The README tree text is a literal that is not tied to the constant.**
The tree line (`_cli_readme_generator.py:105`) hard-codes `measurements_by_category/`, while the
section uses `DIR_MEASUREMENTS_BY_CATEGORY` (`:239`). `test_output_tree_lists_both_split_folders`
checks the literal, so renaming the constant would leave the tree wrong and the test green. The
rest of the tree is hand-written the same way, so this is consistent with the file. Fix: have
the test assert `f"{DIR_MEASUREMENTS_BY_CATEGORY}/" in tree` (and the same for
`DIR_MEASUREMENTS_BY_FEATURE`).

**L-8: Stale split files are inherited unchanged.**
When a later finalize at the same output produces no categorized column, `split_master_by_category`
returns `{}` and leaves an earlier `StartingMetrics.{csv,parquet}` in place. The same happens
when a category is renamed or removed in a later release. A Parquet failure also keeps the new
CSV beside the *old* Parquet. The feature split has always behaved this way, and spec §5.2
requires `_write_split` to be unchanged. Fix, as a follow-up for both splits: inside the
guard, remove `*.csv`/`*.parquet` in the split directory whose key is not in `split_frames`,
and remove a key's Parquet when its Parquet write fails.

**L-9: Docstrings omit migrate.**
`split_master_by_category`'s docstring (`:1407-1409`) and the call-site comment (`:1304-1306`)
say "full, measure and recompile". `--mode migrate` also reaches the call (§C). Fix: add
"and `--mode migrate`".

### Not findings (checked)

- **`_get_measurement_infoclasses` annotation** (`-> list[type[MeasurementInfo]]`, `:261`). The
  module has `from __future__ import annotations`, so importing `MeasurementInfo` only under
  `TYPE_CHECKING` is safe. mypy is clean, and no caller is mistyped: its three callers pass the
  result to an untyped parameter or iterate it.
- **The `DYNAMIC` test class** in `test_dynamic_header_resolves_through_member_for_header` stays
  in `MeasurementInfo.__subclasses__()`. Both subclass-walking tests filter on
  `__module__.startswith("phenotypic")`, and `CATEGORIES.members()` walks `schema.__all__`, so
  the leftover class does not leak into them.
- **The sdk_ constant and helper** are exported and listed in `__all__`. They are covered by
  `TestPathHelpers` and `TestDeliverablesLayout` (under `deliverables/`). No deliverable name is
  hand-joined in `src/`.

## Per-test verdicts (can each go red for its stated reason?)

| Test | Can it fail for its stated reason? |
|---|---|
| `test_starting_metrics_split_holds_context_then_categorized_columns[pandas/polars]` | Yes (M5). It cannot catch a context-first regression, because the fixture is already context-first (L-4). |
| `test_uncategorized_measurements_are_not_context` | Yes (M5) |
| `test_both_integrated_intensities_appear_once_each` | Only if a duplicate could arise, which is structurally impossible today. It is a regression pin, not a live discriminator. |
| `test_context_matches_the_feature_split` | Weak: it compares a 2-column prefix (L-4) |
| `test_category_with_no_present_columns_has_no_key` / `test_frame_with_no_measurements_splits_to_nothing` | Yes (a mutant that returned empty keys or skipped the early return would fail) |
| `test_non_frame_input_names_the_category_split` | Yes (message text) |
| `test_every_categorized_header_is_owned_by_a_producer` | Yes (M11) |
| `test_column_in_two_categories_appears_in_both` | Yes (M15) |
| `test_dynamic_header_resolves_through_member_for_header` | Yes for resolution. It bypasses producer ownership, which is acceptable because ownership uses the same `member_for_header`. |
| `TestSplitMasterByCategory` (2 tests) | Yes (M6, M7) |
| `test_aggregate_writes_category_split_beside_feature_split` | Yes (M1, M5). It does not check the feature split its name claims (L-4). |
| `test_failing_category_split_does_not_block_publication` | Yes for "a guard exists" (M3, M7). It cannot detect a guard outside the fence (M-1). |
| `test_category_split_is_called_only_from_finalize_post_master_outputs` | Yes for removal or a literal second call (M1). Blind to an alias, `partial`, a callback or a module-level call (M-2). |
| recompile-SLURM finalizer extension | Yes (M1) |
| `test_readme_categories.py` (6 tests) | Yes for presence and absence. De-duplication and ordering are unpinned (L-6), and the tree check uses a literal (L-7). |

## At risk outside the focused runs

- `tests/unit/cli/test_cli_v2.py` and `tests/integration/cli/test_staged_gpu_local.py`. Both bind
  `aggregate_master_csv`, which now also writes the category folder. Any assertion over an exact
  `deliverables/` listing would change.
- Any e2e or GUI test that snapshots the `deliverables/` tree: `tests/e2e/gui/`,
  results-viewer output-root discovery (`_gui/results_viewer/_output_root.py`), and CLI
  integration tests under `tests/integration/cli/`.
- `tests/unit/cli/test_cli_migrate*.py`, because migrate now writes the folder (§C).
- `tests/unit/cli/test_cli_finalize*.py` and the sentinel and checkpoint-handler tests. They
  share `finalize_post_master_outputs`, and the extra guarded write adds a publication-commit
  window.
- Startup-import guards (`tests/unit/ci/test_startup_imports.py`, `test_deferred_imports.py`).
  `phenotypic.util` now imports `CATEGORIES` at module level (`_measurement_outputs.py:14`).
  That module is stdlib-only, so the risk is low, but it belongs to the Task 10 regression.
