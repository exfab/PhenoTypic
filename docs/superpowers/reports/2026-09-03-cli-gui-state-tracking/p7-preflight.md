# P7 pre-flight — every claim in `phase-7-migrate-mode.md` against the tree

**Read-only.** Nothing outside this file was edited. No commands with side effects were run;
every verdict below comes from `grep`, `sed -n`, `awk` and `find` over the worktree
`/bigdata/exfab/anguy344/PhenoTypic/.worktrees/cli-gui-state-tracking`.

**Plan:** `docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/phase-7-migrate-mode.md`,
1607 lines (measured: `wc -l`), seven tasks (1, 2, 2b, 3, 4, 5, 6).

## What was measured

- **43** `file:line` citation instances extracted mechanically
  (`grep -noE '\S+\.(py|md|json|csv|toml):[0-9]+...'`), of which **39** are distinct
  (`phenotypicCLI.py:1661-1662`, `_cli_migrate_image.py:125`, `phenotypicCLI.py:602-612`
  and `_cli_recompile_tables.py:297` each appear twice).
- **4 more** citations the regex cannot see, because Task 6 writes them as a bare `(`:235`)`
  with the filename in a preceding sentence: `_cli/CLAUDE.md` `:175`, `:235`, `:257-261`,
  `:298`. **A citation form that a grep cannot enumerate is itself a finding** — it is why
  Task 6's citations have never been swept.
**All 43 distinct citations (39 + 4) were resolved and read.** Tallied by hand from
sections B and C below:

- **6** land on the identifier the plan attaches to them: `_cli_completion.py:132-135`,
  `_cli_migrate.py:852-880`, `_cli_process_single.py:122-171`,
  `phenotypicCLI.py:602-612`, `sdk_/ngff_.py:1737-1790`, `design.md:323-345`.
- **3** are multi-value citations where one component resolves and another does not:
  `_cli_migrate.py:611,1377`, `test_embedded_table_inversion.py:536,559,584`,
  `_cli_output_manager.py:325-357,606`.
- **34** do not resolve.

Separately, **12 claims** were checked without reference to their line numbers and hold
(section C's table).

## Classification legend

- **[WAS-WRONG]** wrong now and wrong before — never right.
- **[STALE-P6]** / **[STALE-P2..P5]** right when written; this change invalidated it.
- **[BLAST]** cites a file another live task is editing; re-verify after that task lands.

---

# A. Findings that change what a task *does*

## A1. Task 1 is already implemented. Its `Files:` block says "Create". [STALE-P1]

Plan `49-302`. `Files:` says *"Create: `src/phenotypic/_cli/_cli_schema_gate.py`"*, and Steps
1–3c are written as work to perform.

The tree:

| Plan says build | Tree has |
|---|---|
| `_cli_schema_gate.py` (create) | exists, 3624 bytes, `refuse_unconverted_schema` at `:54` |
| `requires_conversion` (build it here) | `sdk_/_schema_shape.py:338` — **a different module** |
| `ConversionVerdict`, no `BELOW_FLOOR` | `_schema_shape.py:189-205`, two members, `BELOW_FLOOR` absent ✓ |
| `STATE_SCHEMA_VERSION = 3` | `_schema_shape.py:65` ✓ |
| `tests/unit/cli/test_schema_gate.py` (new) | exists, 45184 bytes |
| 7 named tests | **6 of 7 present** (`test_a_converted_tree_is_accepted` absent; the parametrized dischargeability test subsumes it) |

The plan does say this in the *commit-message NOTE* at the end of Step 4 ("this task ships
in P1, not P7 (CAN-11)"). The note is 240 lines below the `Files:` block that contradicts it.
**Task 1 should be struck to a verification step**, not dispatched as implementation.

## A2. Task 2 tells the implementer to rename `stage2_done/` aside. That destroys a live staged-GPU run. [WAS-WRONG]

This is the one finding with a data-loss consequence, and the plan contradicts itself three
ways about it:

| Plan location | Says |
|---|---|
| `## What migrate converts`, `24-25` | **two** trees: `image_complete/` and `stage3_complete/` |
| Task 1 signal 2, `163-165` | *"**NOT `stage2_done/`** (U-9): that tree is current, not legacy, so firing on it would classify every modern GPU run CONVERT and strand it — an INV-DISCHARGEABLE violation"* |
| Task 2 Step 3, `384` | *"Enumerate the union of all **three** legacy trees"* |
| Task 2 commit msg, `436` | *"Enumerates the union of `image_complete/`, **`stage2_done/`** and `stage3_complete/`"* |
| Task 5 Step 1b, `1233` | **two** trees: rename `image_complete/` and `stage3_complete/` |

The tree settles it against Task 2. `sdk_/_io_constants.py:672-687`:

- `DIR_IMAGE_RECORDS` — *"One record per image, replacing `image_complete/` and
  `stage3_complete/` (spec §6.1)."* Two.
- `DIR_STAGE2_DONE` — *"**Retained, not collapsed** (U-9) … this tree survives the collapse
  with its file and its atomic `unlink` intact, so its segment is a durable layout fact with
  two readers — the token's path helper and the schema gate, **which must keep *not* firing
  on it***."*

And the shipped gate iterates exactly two: `_schema_shape.py:288` reads
`for segment in (DIR_IMAGE_COMPLETE, _DIR_STAGE3_COMPLETE):`.

Root `CLAUDE.md` makes the cost concrete: the Stage-2 token is *consumable* — Stage 3
"replays the raw array, measures, re-promotes the store, and **consumes the token** and then
the raw array". Renaming `stage2_done/` into `.phenotypic/legacy-v2/` mid-flight makes every
un-consumed Stage-2 result invisible to Stage 3, and Task 5 Step 1b puts that directory
where *"nothing reads it"*. **Strike `stage2_done/` from Task 2 Step 3 and its commit
message.**

## A3. Task 2's two marker tests assert stage sets the conversion table cannot produce. [WAS-WRONG]

Both are in Step 1, and both would be written red and then "fixed" by making the converter
wrong.

`test_three_markers_become_one_record` (`311-322`) plants `image_complete=True,
stage3_complete=True` — two markers — and asserts
`set(record["stages"]) == {"stage2", "stage3", "measured"}`. Nothing in the conversion table
maps anything to `stages.stage2`; under rows 1–2 the reachable set is `{"stage3",
"measured"}`. The test name says "three markers" while the fixture plants two.

`test_a_stage3_marker_with_no_image_complete_still_converts` (`350-360`) is internally
inconsistent in three places at once: the parameter is `stage3_complete=True`, the docstring
says *"Stage 2 finished and Stage 3 never ran"*, and the assertion is
`set(record["stages"]) == {"stage2"}`. Under the table, a `stage3_complete/` marker becomes
`stages.stage3`.

The stage keys themselves are real and closed — `sdk_/_image_record.py:51-54` gives
`stage1`/`stage2`/`stage3`/`measured` — so the fix is to name the intended one, not to add a
key.

## A4. Half of Task 2 may already ship, in a function the plan never mentions. [WAS-WRONG]

`migrate_legacy_stage3_markers` exists at `_cli/_cli_staged_resume.py:367` and writes
`STAGE_STAGE3` through `record_stage` (`:188-199`). Its own docstring (`:180-181`) says it
*"is called from the ordinary staged resume path at `phenotypicCLI.py:2552`, not only from
`--mode migrate`"* — which is also a **U-7 violation already on disk** (migration logic on a
normal run path), and U-7 is the ruling Task 2b Step 1 enforces by deleting the `--mode
full` dispatch of a *different* helper. Task 2 should say what it does with this one; as
written it will produce a second `stage3_complete/` converter.

## A5. Task 2b: "three tests exercise the helper" — one does. [WAS-WRONG]

Plan `702-703` names three tests to keep and retarget. Measured by grepping for the helper's
own name across `tests/`:

| Plan citation | Reality |
|---|---|
| `tests/unit/cli/test_cli_state_management.py:316` | exercises `_requires_legacy_success_migration` (the **sibling predicate**) at `:322,325,329`, not `_migrate_legacy_success_evidence` |
| `test_cli_completion_store.py:606` | the helper *is* called in this file — at **`:724`**. `:606` is inside an unrelated `"hdf"`-key allowlist test |
| `test_embedded_measurement_migration.py:312` | **no reference to either helper anywhere in the file.** `:312` is an import block in a migrate test |

"Keep all three; retarget them at the new call path" sends an implementer to retarget a test
that has nothing to retarget.

## A6. Task 2b's new-fixture rationale (MIG-15) cites `strip_completion_evidence`, not `make_markerless`. [WAS-WRONG]

Plan `707` and `735` both cite `tests/unit/sdk_/_migration_fixtures.py:437-449` / `:440-447`
for `make_markerless`. `make_markerless` is at **`:461`**; `:434-458` is
`strip_completion_evidence`. This is the exact failure mode the brief named — a valid
location carrying the wrong identifier.

**The claim itself holds.** `make_markerless` (`:461-472`) calls
`strip_completion_evidence`, then sets `state.config["success_markers_required"] = False` and
saves. It never touches `work_ids`. So MIG-15 is right that the fixture is *not* the floor
shape, and a new fixture is genuinely required.

## A7. Task 4's `master_carries_user_metadata` test passes a `Path` to a function that takes a `DataFrame`. [WAS-WRONG]

Plan `1078-1092`, `test_migrate_leaves_a_legacy_master_v1_shaped`:

```python
master = master_measurements_parquet_path(tmp_path)
assert master_carries_user_metadata(master) is True
```

`sdk_/_master_io.py:75` — `def master_carries_user_metadata(frame: "pl.DataFrame") -> bool:`.
The test needs a `pl.read_parquet(master)`. The plan's own paragraph calls this *"a positive
assertion about a value a real function returns"*, so the function was checked and the
signature was not.

(`read_master_measurements` is correctly reported as absent — zero hits in `src/`.)

## A8. Task 4's CAN-10(a) attaches the fan-out behaviour to the wrong call, and calls a comment a log. [WAS-WRONG]

Plan `989-993` cites `_embedded_measurement_tables.py:88-93` for the right-join and `:81-86`
for *"logging 'preserving duplicate-key fan-out'"*.

| Claim | Real location |
|---|---|
| `prepare_embedded_measurement_table` right-joins, metadata left, `maintain_order="right"` | **`:178`** def, **`:213-218`** join — claim ✓ |
| *"preserving duplicate-key fan-out"* | **`:154-158`** — and it is a **source comment**, not a log call, attached to a **different** function's `how="semi", maintain_order="left"` projection at `:159-164` |

`logger` in that module is used in exactly one place: `_restore_join_key_dtypes`. An
implementer told to find a fan-out *log statement* on the right-join will not find one.

CAN-10(b) survives intact, with the line moved: `_restore_join_key_dtypes` is at **`:27-44`**
(plan says `:22-39`), and its failure branch does warn and leave the column as-is
(**`:38-43`**, plan says `:31-38`).

## A9. Task 5 Step 1e: "three tests" is two, and the third is in a file the plan never names. [WAS-WRONG]

Plan `1381-1383`: *"Delete `_refuse_inverted_store` and its three tests
(`tests/unit/cli/test_embedded_table_inversion.py:536,559,584`)."*

Measured with `awk` over defs in `530..600`: **two** test defs, at `:536` and `:559`. Line
`:584` is a **comment block**, and it is the plan's own correction:

> ``_refuse_inverted_store``'s CALL SITE is proved in
> ``test_embedded_measurement_recompile.py`` by
> ``test_the_inverted_store_guard_runs_before_the_rewrite_transaction`` … **Delete the two
> tests above and that one together with the recompile repoint (P7 Task 5 Step 1e).**

So the correct instruction is *two tests here plus one in
`tests/unit/cli/test_embedded_measurement_recompile.py`*. Following the plan literally leaves
that third test importing a deleted symbol — a red suite pointing at the wrong file.

## A10. Task 5 Step 1e's gate test is at a different line, and `:474` is a different test. [WAS-WRONG]

Plan `1395-1396` names `test_every_mode_produces_a_byte_identical_master`
(`tests/unit/cli/test_finalize_run.py:474`). That function is at **`:292`**; `:474` is
`test_curation_re_keying_still_works_against_the_intrinsic_master`. Same class as A6.

## A11. Task 5 Step 1e's swap site is 97 lines from where the plan puts it. [STALE]

`recompile_embedded_measurement_table` is at `_cli_recompile_tables.py:357`, and its
`prepare_embedded_measurement_table` call — the one-line swap — is at **`:394`**. The plan
cites `:297` twice (`1360`, `1373`); `:297` is the tail of an unrelated fsync/clear block.
`_refuse_inverted_store` is at **`:176`**, not `:87` (P6's plan carries the same `:87`, so
this drifted after P6 was written).

**The load-bearing half of Step 1e is correct**, and worth saying because it is the claim the
step's cost estimate rests on: `prepare_image_tables` (`:108-111`) and
`prepare_embedded_measurement_table` (`:178-181`) do have byte-identical signatures
`(measurements: pd.DataFrame, metadata_csv: Path | None)`. The "one-line change" claim holds.
So does item 6's quotation of the docstring — `:188-193` names both `--mode migrate` and
`--mode recompile` verbatim.

## A12. Task 6 re-introduces the version floor U-6 withdrew. [WAS-WRONG — the withdrawn-row check the brief asked for]

The conversion table, row 9 (`33`):

> ~~anything below v0.17.3~~ | **No version floor (U-6).** `state.version` cannot express
> one; detection is by shape, and a pre-markers tree is supported however old.

Task 1 Step 3b (`121-131`) rules the same way twice and adds *"there is **no `BELOW_FLOOR`**
verdict"*. The tree agrees: `_schema_shape.py:192` — *"Two members, and there is deliberately
**no `BELOW_FLOOR`** (U-6)"*.

Task 6 Step 3 (`1553-1557`):

> **Migration floor is v0.17.3** (U-1). **Below it, migrate refuses with a version string and
> a pointer.**

That is the withdrawn behaviour, restated as a documentation deliverable — so P7 would ship a
register describing a refusal the code deliberately does not implement. It also contradicts
Step 3b's own reason (`"2.0.0"` spans both sides of the tag, so there is no version string to
refuse on).

## A13. Task 6's five `_cli/CLAUDE.md` citations: 0 of 5 resolve to the claimed content. [STALE]

| Plan | Claims the line is | Actually |
|---|---|---|
| `:251-254` | *"the root is written last … nothing writes into the store after publication"* | verification-cache `(size, mtime_ns)` fencing + a GPFS 0-of-200 measurement |
| `:235` | the heading `## Per-image completion markers` | mid-sentence on recompile/migrate rejection. That heading is at **`:287`** |
| `:257-261` | the `_migrate_legacy_success_evidence` paragraph | cache-soundness prose. The helper is not named there |
| `:298` | the heading `## Output layout & deliverables` | mid-sentence on `publish_image_success`. That heading is at **`:397`** |
| `:175` | the heading `## Legacy-tree migration` | a per-image-isolation bullet. That heading is at **`:178`** |

Two of these are *content* misses, not offsets: Task 6 Step 1's whole job is to correct a
specific false sentence, and it points at a paragraph that does not contain it. The sentence
does exist — `image_data_artifact`'s docstring at `_cli_completion.py:141` opens *"Because
the root `zarr.json` is written **last** by `promote_store`,"* — which is the same claim the
plan separately (and correctly) flags at `_cli_completion.py:132-135`. **The guide citation
should be re-derived; the source citation is the good one.**

## A14. Task 6 would document a rename that has not happened. [WAS-WRONG]

Plan `1470-1472`: *"`SUCCESS_MARKER_VERSION` is now `RECORD_VERSION`"*. Both exist:
`SUCCESS_MARKER_VERSION` at `sdk_/_io_constants.py:729`, exported at `sdk_/__init__.py:111`,
and **still written** by the HDF→Zarr migrator at `sdk_/_hdf_to_zarr.py:607,645`.
`RECORD_VERSION` is exported alongside it (`__init__.py:268`). This is an addition, not a
rename, and the surviving writer is inside the very migrator P7 Task 2b builds on.

## A15. The CAN-32 fix is half-landed: one row still names a step that does not exist. [WAS-WRONG]

The table's note (`36-38`) says *"Both now name a step."*

| Row | Names | Exists? |
|---|---|---|
| `master_measurements.csv` → deleted | **4, Step 0** | ✓ `1018` |
| `deliverables/metadata.canonical.csv` → emitted | **3, Step 4** | ✗ **Task 3 has Steps 1, 2, 3 only** |

And Task 3's only mention of the file is still an assertion inside
`test_the_metadata_snapshot_is_byte_unchanged_by_a_full_migrate` (`934`) — precisely the
shape CAN-32 described ("Task 3 asserted `metadata.canonical.csv` exists while no task built
it").

**Both rows are moot in the tree, which is the bigger finding:**

- `CANONICAL_METADATA_CSV_NAME = "metadata.canonical.csv"` at `sdk_/_hdf_to_zarr.py:458`,
  emitted by migrate, covered by `tests/unit/sdk_/test_metadata_canonical_view.py` (7
  assertions) and `tests/integration/cli/test_migrate_end_to_end.py:515`.
- `master_measurements.csv` was deleted by **D8** — `sdk_/_io_constants.py:317`:
  *"Parquet-only since D8. `master_measurements.csv` is gone."*
- `test_the_metadata_snapshot_is_byte_unchanged_by_a_full_migrate` **already exists**, at
  `tests/integration/cli/test_migrate_end_to_end.py:496` — not in
  `tests/unit/cli/test_migrate_state.py` where Task 3 Step 1 places it. Writing it there
  creates a second copy under one name.

## A16. Task 5 Step 1b's `clear_machine_state` claim is one release out of date. [STALE-P2]

Plan `1236-1237`: *"That function rmtree's **every** child of `.phenotypic/` except
`TERMINAL_FAILURES_JSONL` (`sdk_/_io_constants.py:1105-1116`)"*.

`_PRESERVED_ON_RESTART` is at **`:1259-1261`** and is
`frozenset({TERMINAL_FAILURES_JSONL, RESTART_EPOCH_JSON})` — **two** members;
`clear_machine_state` is at `:1264`, reading the set at `:1296`. The plan's own Task 6 table
(`1512`) states the second member correctly, so the two halves of the plan disagree.

The step's *instruction* is unaffected (add `legacy-v2/` to that set), and its cross-suite
warning is correct: `test_clear_machine_state_deletes_the_persisted_cache` exists at
`tests/unit/sdk_/test_verification_cache_disk.py:803`.

## A17. INV-DISCHARGEABLE's own test is `@pytest.mark.skip`, and no P7 step removes the mark. [WAS-WRONG]

`tests/unit/cli/test_schema_gate.py:1099-1106` carries:

```python
@pytest.mark.skip(reason=(
    "INV-DISCHARGEABLE's migrate half: `--mode migrate` does not yet convert "
    "`.phenotypic/` -- that is P7 Tasks 2, 2b and 3. … P7 Task 5 removes this "
    "mark; it is that phase's gate."
))
```

Task 5 Step 1d (`1334-1342`) removes the **two `xfail`** markers and says so explicitly
("with **both** `xfail` markers removed"). It never mentions this `skip`. Because `skip` is
not `strict`, nothing announces it — so P7 can close with the test the plan calls *"the test
that closes MIG-11, MIG-20, and the next shape nobody enumerated"* silently not running.

This is the same failure the plan diagnoses one paragraph earlier about Step 1d itself
(*"a tripwire that fires on the fix cannot report that the fix was never scheduled"*), one
level up.

---

# B. Citation drift that changes only where you look

Offsets, claim intact. Listed so the sweep is complete and auditable.

| Plan line | Citation | Real location | Note |
|---|---|---|---|
| `188`, `645` | `phenotypicCLI.py:1661-1662` (`_refuse_unmigrated_output` call, pre-`--restart`) | def `:402`, **call `:1676`** | `:1661` is the `--mode process` overlap check |
| `657` | `phenotypicCLI.py:560` (`_migrate_legacy_success_evidence`) | **`:571`** | sibling `_requires_legacy_success_migration` at `:555`, plan says `:544` |
| `465` | `phenotypicCLI.py:590-592` (`work_id_for_image` call) | **`:601-603`** | |
| `273`, `495` | `phenotypicCLI.py:602-612` (`process_only_layer` arm) | branch **`:612`**, artifact **`:613+`** | range ends exactly where the claim starts |
| `657` | `--mode full` dispatch `:2375-2378` | **`:2418-2419`** | |
| `462`, `519` | `_cli_migrate_image.py:125` (`_configured_work_id`) | **`:126`** | `_migration_work_id` **`:121`** (plan `:120-122`) |
| `691` | `_cli_migrate_image.py:207` (`_valid_migration_marker`) | **`:208`** | |
| `676` | `_cli_migrate_image.py:434` (`migrate_image_task`) | **`:456`** | |
| `514` | `_cli_migrate_image.py:567` (`publish_image_success`) | **`:589`** | |
| `513` | `_cli_migrate.py:1413` (`publish_image_success`) | **`:1456`** | |
| `682` | `_cli_migrate.py:567` (`_ensure_migration_processing_state`) | **`:569`** | synthetic mint **`:634`** (plan `:632`); stem comparison `:628-634` ✓ |
| `263` | `_cli_migrate.py:611,1377` | **`:611` ✓** (zarr glob); **`:1377` ✗** — an embedded-table match check, not an `.h5` walk |
| `678` | `_cli_migrate.py:852-880` (`_execute_migration_tasks`) | **`:875`** |
| `545` | `_cli_output_manager.py:325-357,606` (`_persist_pipeline_to_output_dir`) | **`:612`**; `:325-357` is a polars CSV dtype-inference comment |
| `570` | `_cli_process_single.py:122-171` (`_worker_work_identity`) | **`:122-176`** ✓ |
| `262` | `_cli_process_single.py:789,943` (`publish_image_success`) | **`:827`, `:981`** |
| — | `_cli_process_single.py:723-729` (identity cross-check `RuntimeError`) | **`:726-746`** |
| `564` | `_cli_failure_tracker.py:329-350` (`work_id_for_image`) | **`:310-342`** |
| `856` | `_cli_failure_tracker.py:353` (`append_terminal_failure`) | **`:344`** |
| `173` | `_cli_state_management.py:111-112` (`return None`) | **`:136-137`**; `migrate_legacy_machine_state` write **`:133`** (plan `:109`); `json.loads` **`:140`** (plan `:115`); unguarded `[VERSION]` **`:192`** (plan `:167`) |
| `900` | `_cli_state_management.py:121` (event-log re-aggregation) | **`:148-167`**; `:121` is `return state_file` in `save_processing_state` |
| `1215` | `_cli_completion.py:340-350` (metadata-migration receipts) | `refresh_success_markers_after_metadata_migration` **`:356`** |
| `1026` | `_measurement_tables.py:340-346` (descriptor docstring) | **`:836-841`**; def `:813` |
| — | `_measurement_tables.py:382` (`embedded_measurement_columns` `KeyError`) | **`:875-896`** |
| `1011` | `_measurement_tables.py:459-465` (`target.column`) | **`:272-275`**; `:459-465` is the *metadata*-table descriptor |
| `1463` | `sdk_/_measurement_tables.py:242` (`replace_embedded_measurement_table`) | **`:751`** |
| `260` | `_io_constants.py:2468-2482` (`BundleLayout.detect`) | **`:2661`**; `:2468` is `DashboardManifestKey.SLURM_INFO` |
| `1261` | `sdk_/ngff_.py:1737-1790` (`promote_store` move-aside) | def **`:1758`** |
| `702` | `test_cli_state_management.py:316` | **`:322,325,329`** |
| `1383` | `test_embedded_table_inversion.py:536,559` | **both ✓** |
| `32` | `design.md:323-345` (the `slurm_generation`/`lifecycle_epoch` withdrawal) | **✓** §5.1 at `:323`, the "collapse is achievable ZERO times" amendment at `:343+` |

**Claims verified correct in substance despite a wrong line** (so they need re-citation, not
re-thinking): `load_processing_state` does write via `migrate_legacy_machine_state` and does
subscript `VERSION` unguarded; `work_id_for_image` does recompute from a live
`ExecutionConfig` and never reads `state.config`; the descriptor docstring does say *"An
absent descriptor is a normal state, not a fault"*; `embedded_measurement_columns` does raise
`KeyError`; the descriptor does carry `target.column`; `_restore_join_key_dtypes` does warn
and leave the column.

---

# C. Claims that hold, including the expensive "nothing does Y" ones

Stated because the brief asked for the bucket to be explicit, and because each of these is a
precondition another task builds on.

| Plan | Claim | Verdict |
|---|---|---|
| `2b`, U-10 callout | *"`PROVENANCE_MIGRATED` has **zero production writers** as of P3"* | **✓** Every `src/` occurrence is a definition (`sdk_/_image_record.py:60`), an export, a docstring, or a **reader** comparison (`_image_record.py:125,184`; `_run_state.py:587,1374`) |
| `2b`, U-10 callout | *"`publish_image_success` has no `provenance` parameter"* | **✓** signature `_cli_completion.py:172-188`. `publish_image_record` has `provenance: str = PROVENANCE_FORWARD` (`_cli_image_record.py:92`) |
| `4` | `read_master_measurements` *"does not exist and is not coming"* | **✓** zero hits in `src/` |
| `5`, Step 1d | *"There is **no re-export** of this flag"* | **✓** `SCHEMA_GATE_ARMED` binds once, `_schema_shape.py:153`; every other hit reads it through the module or is prose |
| `5`, Step 1d | the two `xfail`s turn green on arming | **✓** `test_schema_gate.py:878`, `:929`, both `strict=True`, both naming Step 1d |
| `5`, Step 1d | `test_a_tree_this_build_wrote_needs_no_conversion` is the standing evidence | **✓** `tests/unit/cli/test_image_record.py:762` |
| `5`, Step 1b | the cross-suite cache test | **✓** `tests/unit/sdk_/test_verification_cache_disk.py:803` |
| `5`, Step 1e | *"Both take the identical signature"* | **✓** `_embedded_measurement_tables.py:108-111` vs `:178-181` |
| `5`, Step 1e item 6 | the docstring names only migrate + recompile | **✓** `:188-193` |
| `6` (a)/(c) | every named writer/deriver exists | **✓** `create_initial_state` `_cli_state_management.py:206`; `append_terminal_failure` `_cli_failure_tracker.py:344`; `bump_restart_epoch` `_cli_identity.py:386`; `mint_run_identity` `_cli_identity.py:204`; `persist_states` `sdk_/_verification_cache.py:485`; `finalization_input_object` `sdk_/_run_state.py:224`; `run_identity` `:274` |
| `6` (a) row 4 | `restart_epoch.json` is preserved by `clear_machine_state` | **✓** `_PRESERVED_ON_RESTART` `_io_constants.py:1259-1261` |
| `2b`, U-7 table | 12 digest fields, 7 `work_id` fields | **✓** `processing_configuration_digest_from_values` `_cli_failure_tracker.py:191-205` declares 12; `work_id_for_image` `:331-339` folds 7 |

One narrowing inside the last row: the plan says the digest function *"takes all twelve
fields as **required** keyword parameters"*. **Eleven are required; `drop_originals: bool =
False` has a default** (`:204`). The argument ("there is no 'absent' to express") survives,
but the sentence as written is falsifiable by reading the signature.

One soft internal inconsistency worth a line: U-7's table and prose say **five** of twelve
digest fields are absent (`❌` row: `detect_mode`, `process_only_layer`, `process_format`,
`overlay_alpha`, `drop_originals`), while Task 2b Step 2's test docstring (`757`) says
**seven**. Reconcilable if "seven" counts the two `⚠️ derivable by inference` rows as
unrecorded — but the plan should pick one number.

---

# D. In the P6 blast radius — re-verify after P6 lands, trust neither way

| Citation | File | Held by |
|---|---|---|
| `_cli_completion.py:132-135`, `:340-350` | `_cli/_cli_completion.py` | P6 Task 0 split this module (`67db3e3a`, `692d435d`, `7ca07d26`) and P6 Task 7 has more |
| `_run_state.py:587`, `:1374` (the `PROVENANCE_MIGRATED` readers) | `sdk_/_run_state.py` | P1/P6 Task 0 both edited it; P6 Task 7b may again |
| `_cli_state_management.py:133`, `:148-167`, `:192` | `_cli/_cli_state_management.py` | carries a P5-era comment at `:105-107` about the gate arming, so P5 moved it and P7 Task 5 Step 1d will again |
| Anything reaching `results_viewer/` | — | **P6-T1 is live in that tree right now.** No P7 citation lands there, which is the good news; the closing-checklist command `grep -c '' src/phenotypic/gui/results_viewer/_output_consistency.py` does, and it will answer differently before and after P6 Task 2 |

---

# E. The pattern, since the brief says it is worth more than any single fix

Three mechanisms produced all 34 misses, and each has a cheap countermeasure:

1. **Offsets, from edits above the cited line.** The bulk of the 34 (I did not tabulate the
   split, so no count is quoted). Unavoidable in a plan written against a moving tree; the
   fix is to cite `file::symbol` and let the reader grep, not `file:line`.
2. **Right location, wrong identifier.** At least eight: A6, A10, A13 (`:251-254` and
   `:257-261`), `_measurement_tables.py:459-465`, `_io_constants.py:2468-2482`,
   `_cli_state_management.py:121`, `_embedded_measurement_tables.py:88-93`,
   `test_cli_completion_store.py:606`. This is the class the brief flagged, and it is the
   dangerous one: the location check passes, so a reviewer verifying "does line N exist?"
   learns nothing. Only printing the line and reading it catches these.
3. **A count taken from a list in hand.** A2 ("three trees"), A5 ("three tests"), A9 ("three
   tests"), A16 ("except `TERMINAL_FAILURES_JSONL`"). Every one of them was refuted by
   enumerating the defining property instead — `for segment in (...)` in the gate, a grep for
   the helper's own name, an `awk` over `def test_` in a line range, the `frozenset` literal.
   **In all four cases the tree already stated the correct number in prose**, in the constant's
   docstring or in a comment two lines below the plan's citation.

A17 and A12 are a fourth, rarer kind: a decision recorded in one place and reversed or
forgotten in another, where nothing fails. Those are the two to fix before dispatch even if
every line number is left alone.
