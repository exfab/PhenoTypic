# P4 spec-adherence gate

**Question asked:** is every part of P4's plan actually implemented, and does what shipped
match what the spec says? Not "do the tests pass".

**Tree examined:** worktree `/bigdata/exfab/anguy344/PhenoTypic/.worktrees/cli-gui-state-tracking`,
branch `cli-gui-state-tracking`, at `6b3cf2c9`.

**Sources of truth used, in this precedence order:** `design.md` §0 Amendments →
`design.md` §6/§7/§8/D8 → `phase-4-finalize-run.md` → the tree.

**Method.** Every claim below was checked against the tree, not against the commit
messages or the plan's narrative. Where a commit message states a fact, that fact was
re-derived from the file it describes.

**Every total in this report was measured.** The two commands are given at the end.
The drift register's highest index is **42**, measured with the command its own header
prescribes — not with either of the two counts that header warns return 23.

---

## Verdict

P4 is substantially implemented, and the parts that shipped are of high quality: the
inversion, the promote-time metadata table, `finalize_run`, U-4, D8's deletion set, and
the three-entry-point routing are all present, wired to production code, and pinned by
tests that build real stores rather than hand-fed DataFrames.

**One blocker.** `--mode recompile` now hard-fails on any forward tree built with
`--metadata`. It is a deliberate, documented, in-phase decision — but it is documented
only in a commit message, it contradicts spec §7.4's model of recompile as a working
entry point, the fix is assigned to no phase, and the phase's own headline test for
§7.4 is constructed so that it cannot see it.

Everything else is a documentation or coverage gap, not a behaviour defect.

---

## BLOCKER — `--mode recompile` refuses every forward `--metadata` tree

### The mechanism

`recompile_embedded_measurement_table` still builds its payload with the **pre-inversion**
producer (`src/phenotypic/_cli/_cli_recompile_tables.py:297`,
`prepare_embedded_measurement_table`). Running that against an inverted store would
re-join metadata into `tables/measurements/table.parquet` and drop
`tables/metadata/pht-metadata.parquet` — silently. To stop that,
`_refuse_inverted_store` (`src/phenotypic/_cli/_cli_recompile_tables.py:87`) raises when
the store declares `tables.metadata`:

```
src/phenotypic/_cli/_cli_recompile_tables.py:124-131
    if isinstance(tables, dict) and METADATA_TABLE_GROUP in tables:
        raise RuntimeError("Cannot recompile an inverted store: ...")
```

It is called unconditionally from `_replace_and_republish_table`
(`src/phenotypic/_cli/_cli_recompile_tables.py:153`), which
`recompile_embedded_measurement_tables` calls for **every** authorized source
(`:363-370`), with no exception handling. And `--mode recompile` calls that function
unconditionally, before aggregation, at `src/phenotypic/phenotypicCLI.py:3792`.

A store gets `tables.metadata` exactly when `join_status == "joined"`
(`src/phenotypic/sdk_/_measurement_tables.py:359-362`). So:

| Forward tree | `--mode recompile` |
|---|---|
| built without `--metadata` | works |
| built with `--metadata` that shares a column with the measurements | **`RuntimeError` on the first store** |

That is a regression against pre-P4 behaviour, in a shipped mode, for the ordinary case.

### Why the phase's own gate does not see it

`test_every_mode_produces_a_byte_identical_master`
(`tests/unit/cli/test_finalize_run.py:474`) is the test that pins §7.4's "three modes,
one master". Its fixture states:

```
tests/unit/cli/test_finalize_run.py:437-439
    **No metadata snapshot**, deliberately. Post-D8 the master carries no user
    metadata at all, so a snapshot could only change the mirror; running
    without one keeps the comparison about the thing being compared.
```

The reasoning about the master is correct. The consequence is that the `recompile` arm
runs only in the one configuration `_refuse_inverted_store` does not refuse. The phase's
headline claim about recompile is therefore established on the only tree shape where
recompile still works.

### Why it has no owner

`_refuse_inverted_store` appears in **zero** plan, spec, or report documents — the only
matches for it anywhere under `docs/superpowers/` are in this report. Its retirement
condition is written as a code condition rather than a phase
(`src/phenotypic/_cli/_cli_recompile_tables.py:90-93`: *"delete it when
`recompile_embedded_measurement_tables` builds its payload with `prepare_image_tables`"*),
and no phase-5, -6 or -7 step schedules that repoint. Compare the other legacy arms this
phase added, which all carry *"DELETE WHEN: the schema gate is armed … (P7 Task 5 Step
1d)"* (`_cli_recompile_recovery.py:69-73`, `_cli_finalize_run.py:92-98`) — a trigger that
ties them to a scheduled event. This one has no such tie.

The plan **did** schedule the repoint. `phase-4-finalize-run.md:232` lists
`_cli_recompile_tables.py:102` as *"Hard-`isinstance`-checks `PreparedEmbeddedMeasurementTable`
… the exact type Task 1 replaces"*, and `:346` says *"→ `PreparedImageTables`. **This is
the crash**, and the only live one in this file"*. The type was never replaced (the
producer was retained for migrate, correctly), so the isinstance check never fired, and
the guard was added in its place. The guard is the right call for the loud-vs-silent
tradeoff; what is missing is a step that finishes the job.

### What I recommend

Not a fix in this report — but the debt needs an owner and a name. Either:

- a P5/P6/P7 step that repoints `recompile_embedded_measurement_table` onto
  `prepare_image_tables` and deletes the guard with its three tests
  (`tests/unit/cli/test_embedded_table_inversion.py:536,559,584`); or
- an explicit ruling that `--mode recompile` is unsupported on `--metadata` trees for the
  P4→P7 window, written where a user reads it (the mode's `--help` and
  `docs/source/tutorials/pages/cli_modes.md`), not only in a commit message.

**Verification command** (I have not run it; it should reproduce the refusal end to end):

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/integration/cli/test_promote_time_metadata.py -q -k a_real_run
# then, against a tree that run produced, drive the real CLI:
#   uv run python -m phenotypic --mode recompile --output <that tree>
# Expected today: RuntimeError("Cannot recompile an inverted store: ...")
```

---

## A — specified, not implemented

**None found, other than the blocker's spec consequence.**

Spec §7.4's table row for `recompile` (*"Per-image work: none | Then: `finalize_run`"*)
does not describe the shipped mode, which rewrites every embedded table first
(`phenotypicCLI.py:3792`). That divergence **predates P4** — the pre-inversion recompile
did the same — so it is not a P4 omission. It is recorded here because it is the reason
the blocker is reachable: the spec's model of recompile has no per-image write, so it
never considered what a per-image write does to an inverted store.

Everything else §6/§7/§8/D8 assigns to this phase is present:

| Spec obligation | Where it landed |
|---|---|
| §7.1 intrinsic stays, user metadata moves | `_cli/_embedded_measurement_tables.py:108-175` |
| §7.2 `pht-metadata.parquet`, absence when not joined | `sdk_/_measurement_tables.py:211-253`, `:143-151` |
| §7.2 self-describing Parquet KV triple | `sdk_/_measurement_tables.py:110-133` |
| §7.3 master un-joined, mirror joined | `_cli/_cli_finalize_run.py:388-430`, `_cli_output_manager.py:1149-1161` |
| §7.4 six steps, one path, three entry points | `_cli/_cli_finalize_run.py:297-442`; `_cli_output_manager.py:1471-1483`; `_cli_recompile_worker.py:796-829` |
| §7.5 INV-INPUTS + invalidate-on-success | `_cli_finalize_run.py:65-115`, `:209-289`, `:432-440` |
| §7.5 narrowing to the authorized arm | `_cli_finalize_run.py:92-106` (retirement condition present) |
| D-A: no path writes into a promoted store | `_core/_image_parts/_image_io_handler.py:1377-1409` (tables into the `.part`, root last) |
| D-A: store records its snapshot | `sdk_/_measurement_tables.py:304-338`, read by `sdk_/_run_state.py:652-667` |
| D8: parquet-only master, 3 required outputs | `_cli_completion.py:1019-1040`; `AGGREGATE_PROOF_VERSION = 2` at `sdk_/_io_constants.py:754` |
| U-4: `publication_id` cut, digest copied | `_cli_completion.py:1044-1063`, `:1169-1183`, `:1210-1219`, `:1257`, `:1273-1274`; `sdk_/_run_state.py:1088-1105`, `:1242-1280` |
| INV-PROVEN 2nd obligation (in-place branch) | in-place fast path deleted; both replacers share `_rewrite_store_tables` (`sdk_/_measurement_tables.py:632`) |

U-4's three tautology dispositions were all applied, each with a comment at its site
(`_cli_completion.py:1205`, `:1257`; `sdk_/_run_state.py:1088`, `:1098`), and
`_source_set_binding`'s compatibility arm kept its retirement condition
(`sdk_/_run_state.py:1256-1259`).

---

## B — planned, not done

### B-1 — the recompile producer repoint (root of the BLOCKER)

`phase-4-finalize-run.md:232`, `:346`. Covered above.

### B-2 — the ⛔ REQUIRED cost-inversion measurement was never reported

`phase-4-finalize-run.md:847-865` marks it *"REQUIRED BEFORE THE DESIGN IS COMMITTED"*
and closes with *"Report the numbers; every number in this plan must be one someone
measured."* It compares (a) N `pq.read_schema` calls against (b) N `pq.read_schema` + N
root-`zarr.json` reads.

**No numbers exist.** `grep -rn "cost inversion"` over the change's plans, spec and
reports returns exactly two hits — the plan's own heading (`:847`) and EXECUTION.md's
statement of the obligation (`EXECUTION.md:669`). No measurement, no disposition.

The gate is arguably moot: the plan's own escape clause said that if (b) is worse, the
answer is that `_consistent_embedded_join_keys` is retired by Task 3 anyway — and it was
(zero references survive outside comments and test docstrings). But *"measure whether
anything else still needs the per-store triple after that retirement"* was also part of
the instruction, and that question has no written answer either. This is a missing
disposition on a step the plan marked as a design gate.

### B-3 — Task 4 Step 5's phase gate was not run as specified

`phase-4-finalize-run.md:2338-2365` prescribes:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit tests/integration -q
```

and adds a paragraph (M4) explaining that `tests/integration` is in the gate *"not just
`tests/unit`"*, because Task 5's own test lives there and two of Step 4's dependents do
too. It requires reporting a **delta against the recorded baseline**, not a pass/fail.

`8a6d3ae0`'s commit message reports *"4588 passed, 0 failed across tests/unit/cli,
tests/unit/sdk_ and tests/gui/results_viewer"* — three named subtrees, not `tests/unit`,
and not `tests/integration`. `6b3cf2c9` reports `3 passed` for the one new integration
file. No full-suite run, no baseline delta.

The commit message is honest about its own scope (it also declines to claim a mypy delta
for the same reason), and register entry 42 records the class. But the gate as written
was not performed, and P4 is the phase whose plan says *"This is the first phase where
the full suite is warranted rather than a selection — the master's shape changed and it
is read almost everywhere."*

**Command to close it** (Slurm job — `run-phenotypic-test` + `slurm-job`, script at
`docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`):

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit tests/integration -q -p no:randomly
```

### B-4 — Task 4 Step 6's doc sweep is incomplete: five sites still name the master as CSV

`phase-4-finalize-run.md:2367-2372` asks for three doc surfaces. All three were updated
(`CLAUDE.md:412-423`, `_cli/CLAUDE.md`, `docs/source/…`), and `docs/source/` is now clean
— `grep -rn master_measurements docs/source/` returns ten hits, all `.parquet`.

But the sweep was driven by the phrase *"Master CSV"*, not by the filename. It caught
`_cli_interactive.py:150`, `_cli_sentinel.py:172,176`, `phenotypicCLI.py:60`, `:3020`.
It missed five sites in `src/` that name `master_measurements.csv` (or
`.{csv,parquet}`) as a **current fact** rather than as a D8-explanatory note:

| Site | Surface | Severity |
|---|---|---|
| `src/phenotypic/_cli/_cli_readme_generator.py:99` | the `## Output Structure` tree written into **`deliverables/README.md` of every run** — lists `master_measurements.csv` and does not list the parquet master at all | **user-visible artifact** |
| `src/phenotypic/phenotypicCLI.py:1416` | `--no-dataset-column` help text: *"Exclude 'Metadata_Dataset' column from master_measurements.csv"* — the `cli_reference.rst` copy of this exact sentence **was** fixed (`docs/source/api_reference/cli_reference.rst:185`), the `--help` string was not | **user-visible (`--help`)** |
| `src/phenotypic/_cli/_cli_interactive.py:127` | the wizard's printed output tree, **20 lines above the line the sweep did fix** (`:150`) | **user-visible (stdout)** |
| `src/phenotypic/phenotypicCLI.py:3736` | `_run_recompile_mode` docstring — the tutorial copy of this sentence **was** fixed (`cli_modes.md:178`) | internal |
| `src/phenotypic/_cli/_cli_output_manager.py:877` | `_apply_post_to_master` docstring: *"keeps `master_measurements.{csv,parquet}`"* | internal |

This is the same class the team lead flagged — a deletion's blast radius scoped by a
name — with the name being a *phrase* rather than a symbol. Register entry 42 diagnoses
the value-carrying variant of it; the prose variant is not recorded.

Separately, nine sites use the glob `master_measurements.*`
(`gui/CLAUDE.md:266`, `gui/FEATURES.md:351`, `gui/_config.py:32`,
`schema/_metadata_match.py:14`, `_cli/CLAUDE.md:507`, `_cli_output_manager.py:1289`,
`tests/_output_layout.py:4`, `tests/integration/cli/test_cli_metadata_deliverable.py:31`,
`.claude/skills/working-with-ome-zarr/SKILL.md:91`). Those are now **ambiguous rather
than false** — the glob matches exactly one file. Lower priority; listed so the count is
not mistaken for the count above.

---

## C — implemented, but differs from what was specified

### C-1 — `_invalidate_finalization_intermediates` targets a shard directory nothing writes

`src/phenotypic/_cli/_cli_finalize_run.py:255` deletes
`recompile_dir(progress) / DIR_RECOMPILE_SHARDS` — i.e.
`<progress>/recompile/measurement_shards`.

Recompile writes its shards to `task_manifest.parent / DIR_RECOMPILE_SHARDS`
(`src/phenotypic/_cli/_cli_recompile_worker.py:378-381`) and reads them from
`attempt_dir / DIR_RECOMPILE_SHARDS` (`:751`) — i.e.
`<progress>/recompile/attempts/<attempt_id>/measurement_shards`. **Nothing writes the
path `finalize_run` deletes.**

The test certifies the same wrong path. `_plant_stale_shard`
(`tests/unit/cli/test_finalize_run.py:327-338`) plants at
`recompile_dir(progress_dir(tmp_path)) / DIR_RECOMPILE_SHARDS / "shard_0.parquet"`, and
`test_finalize_run_invalidates_the_intermediates_on_success` (`:597-618`) asserts it is
gone. The fixture guard at `:609-611` establishes the file exists — so the test is not
vacuous in the standing-rule sense — but the location it establishes is not a location
production uses. Green, and it proves nothing about recompile's real shards.

**No data-loss risk**, and arguably no functional gap: spec §7.5 assigns shards to a
*different* guarantee (*"per-invocation scratch, emptied when fan-out begins"*), which is
P5's, and the invalidate-on-success list in §7.5 names only chunks, `analysis_full` and
`_dataset_aggregated`. So including shards in this list at all is extra. The problem is
that after P5 lands the clearing, a reader will believe there are two guards where there
is one, and the test says so.

Recommend either removing the shards entry with a comment naming P5 as the owner, or
repointing it to `recompile_dir(progress) / "attempts"` and repointing the fixture with
it — a behaviour change, so its own test and commit.

### C-2 — the Step 6b falsifier compares two v2 masters, and its comment says otherwise

`test_a_v1_metadata_free_master_is_indistinguishable_from_v2_and_that_is_harmless`
(`tests/unit/cli/test_finalize_run.py:1345`) is the designated falsifier for a HARD-STOP
ruling (no schema stamp). Its "v1" arm is:

```
tests/unit/cli/test_finalize_run.py:1367-1376
    # v1: the pre-inversion shape of the same metadata-free run -- the
    # PRE-INVERSION producer's output, written as the legacy finalizer wrote
    # it. ...
    v1.mkdir()
    _publish_successful_images(v1, snapshot=None)
    assert finalize_run(v1, dataset_names=[DATASET]) is not None
```

That is byte-for-byte the same call sequence as the v2 arm three lines above
(`:1363-1365`). `_publish_successful_images` goes through
`OutputManager.save_image_store` → `prepare_image_tables` (the file's own module
docstring says so at `:2-6`), which is the **post**-inversion producer. No pre-inversion
producer is invoked anywhere in the test.

The comment's *"written as the legacy finalizer wrote it"* is therefore false of the
code beneath it — a *never true* claim, in the test whose whole job is to settle a
question by measurement rather than inference.

The substantive weakness is inherited from the plan, not introduced here: with no
snapshot the two producers agree by construction (`prepare_embedded_measurement_table`
and `prepare_image_tables` share `_normalize_table_inputs` and both return the bare
baseline when `metadata_csv is None` —
`_cli/_embedded_measurement_tables.py:134-142`, `:196-203`). So even a *correctly*
built v1 arm would produce an identical frame and the outcomes would be equal for
reasons that have nothing to do with reader behaviour. The falsifier cannot fail.

Two honest options: state that limitation in the docstring and keep the test as a
regression pin, or give the v1 arm a genuinely joined master (built through
`prepare_embedded_measurement_table` with a snapshot) so the comparison has two distinct
inputs. Either needs its own commit. **What must not stand** is the current comment,
which asserts a construction the code does not perform.

### C-3 — the root `metadata_table` block carries the digest only; the plan says three keys

`phase-4-finalize-run.md:816` prescribes:

```json
"metadata_table": {"snapshot_sha256": "…", "join_keys": [...], "join_kind": "…"}
```

Shipped (`sdk_/_measurement_tables.py:337-339`) writes `{snapshot_sha256}` alone, with
the join provenance living on the metadata Parquet's own KV
(`sdk_/_measurement_tables.py:110-133`).

**The narrowing is correct** and is consistent with the plan's own ⚠ RULED block two
lines below (`:820`), which says the digest gets one home and *"the join provenance
belongs to the new **metadata** table's own Parquet KV"*. The JSON snippet at `:816` is
the stale half of a document that corrected itself and left the example behind — the
exact "accurate correction appended, document left self-contradictory" shape the plan's
own header calls out at `:30-38`. Reasoned in `06809fbc`'s message; **not** written back
into the plan.

Recommend amending `phase-4-finalize-run.md:816` to `{"snapshot_sha256": "…"}` so the
plan and the tree agree.

### C-4 — `finalize_run`'s step 6 is narrower than §7.4's

Spec §7.4 and the plan's docstring template both write step 6 as *"publish aggregate
proof → run proof"*. Shipped docstring (`_cli_finalize_run.py:318`) says *"publish the
aggregate proof"*, and the body publishes only that (`:435`).

Functionally correct: run-proof publication stayed at its pre-existing call sites
(`phenotypicCLI.py:2434`, `:3816`; `_cli_recompile_worker.py:690`;
`_cli_checkpoint_handler.py:354`, `:442`), so the chain is intact. Recording it because
the spec's step 6 and the code's step 6 are now different scopes, and nothing says so.

### C-5 — a forward-dated claim in a shipped docstring

`_cli_finalize_run.py:343-344`: *"`measurement_shards/` is emptied when fan-out begins,
so a prior run's shards can never be merged."* No code does that yet — the clearing is
P5's, per `design.md` §7.5's ⚠ CORRECTED block. Written in the present tense, in
production code, one phase early. Inherited verbatim from the plan's docstring template
(`phase-4-finalize-run.md:1247-1249`), so this is a plan defect that propagated rather
than an execution error. It is the register's own shape 1 (*"a pointer to a name that
does not exist yet"*), and it is not in the register.

### C-6 — test relocation, benign

`test_process_mode_skips_finalization_entirely` was specified under Task 4 Step 1 in
`tests/unit/cli/test_finalize_run.py` (`phase-4-finalize-run.md:2194`). It shipped in
`tests/integration/cli/test_promote_time_metadata.py:168`, driving the real CLI. That is
strictly stronger than what was asked, and it retains the standing-rule guard
(`:198-201`). No action.

---

## D — implemented, never specified

### D-1 — `_refuse_inverted_store` and its three tests

`src/phenotypic/_cli/_cli_recompile_tables.py:87-131`, `:153`;
`tests/unit/cli/test_embedded_table_inversion.py:536`, `:559`, `:584`.

**Judgement: it belongs, and the third test is the right instinct** — a guard defined and
never called reads as coverage, so `test_the_recompile_guard_is_wired_into_the_rewrite_path`
(`:584`) asserting the literal call-site string is a real check, not decoration. The
loud-over-silent tradeoff is correct and is exactly the criterion this change uses
elsewhere.

**What does not belong is that it is a behaviour change with no document.** See BLOCKER.

### D-2 — the `SCHEMA_GATE_ARMED` comment correction

`src/phenotypic/_cli/_cli_completion.py:240-252`.

**Verified accurate.** `SCHEMA_GATE_ARMED: bool = False` at
`src/phenotypic/sdk_/_schema_shape.py:153`, and three other comments correctly name P7
(`_cli_recompile_recovery.py:71`, `_cli_finalize_run.py:94`, `_cli_migrate.py:719`). The
old comment's *"which is why `SCHEMA_GATE_ARMED` flips in this same commit"* was **never
true** — it was written in P3 (`1cc6740c`) about a flag that had last been set in
`17f144ef`.

**Judgement: it belongs in P4.** It is comment-only, changes no behaviour, needs no test,
and P4 is the phase that made the false claim materially more misleading — it repointed
five production sites onto `_image_authority_shapes`' legacy arm, which is precisely the
code an armed gate would have made dead. Correcting it in a later phase would mean
shipping five new legacy arms whose justification the adjacent comment contradicts.
Recorded as register entry 41.

### D-3 — the `TestAggregateMeasurements` vacuity fix

`tests/unit/cli/test_cli_v2.py:2075-2082`, in
`test_aggregate_measurements_no_dataset_column`.

**Judgement: it belongs.** The class had to be touched anyway (D8 turned
`aggregate_measurements`' return into a Parquet path and eleven `pd.read_csv(result)`
sites broke). The added guard is four lines, changes no production code, is labelled
*"PRE-EXISTING — not something D8 removed"* at the site, and its sibling
`..._metadata_no_common_columns` already does it correctly — so this restores consistency
rather than inventing a convention. Leaving it out would have meant repointing an
assertion in the same commit that left it vacuous.

The decision not to give it a register entry is also right and is stated in entry 42:
*"it is a missing assertion, not a false claim, and inflating the register with those
would cost it the property that makes it worth reading."*

### D-4 — `_LEGACY_MASTER_MEASUREMENTS_CSV` in the metadata migrator

`src/phenotypic/sdk_/_metadata_migration.py:95-108`, used at `:1051` and `:1081`.

D8 deleted `BundleLayout.master_csv`, which those two sites iterated. The replacement is
a module-private literal with a full retirement condition (`:95-102`), keeping pre-D8
bundles migratable. The plan asked *"decide whether that is acceptable"* only for the GUI
schema cache (`phase-4-finalize-run.md:2299`); it did not name this decision.

**Judgement: it belongs**, and it is the right shape — the alternative (dropping the file
from discovery) would silently leave pre-D8 bundles un-migrated. The GUI half was decided
too, and documented at its site (`src/phenotypic/gui/_schema_cache.py:39-48`: master is
parquet-only, mirror keeps its CSV fallback). Neither decision is recorded in any plan or
spec document.

### D-5 — a fourth home of the false "nothing writes into a promoted store" claim

`docs/source/how_to/pages/zarr_storage.md`, corrected in `06809fbc`. The plan listed
three homes (`_cli/CLAUDE.md:251-254`, `sdk_/CLAUDE.md:134-138`, and the test name); this
was a fourth, and it named the guard by the old test id, so the rename would have left a
doc pointing at a test that no longer exists.

**Judgement: it belongs**, and finding it is the plan's own instruction followed properly
rather than scope creep.

### D-6 — `_cli/CLAUDE.md` corrected in P4 rather than P7 Task 6

`src/phenotypic/_cli/CLAUDE.md:307-330` now carries the corrected, narrower invariant.
Spec §0 says the file *"repeats the same false claim and is corrected in the same
change"*, so this is within the spec's own instruction; the plan's README merely
scheduled it for P7 Task 6. Correcting it in the commit that repairs the branch is what
the plan asks for at `phase-4-finalize-run.md:141-152`. No action.

---

## Cross-checks the lead asked for specifically

### "Look for anything else that reads or names the master as CSV"

Done, twice — once by symbol, once by filename.

**By symbol:** the D8 deletion set is complete. `MASTER_MEASUREMENTS_CSV`,
`master_measurements_csv_path` and `load_master_measurements` survive in exactly two
files: `tests/unit/sdk_/test_io_constants.py:193-195`, which asserts they are **absent**
from `sdk_`, and `sdk_/_metadata_migration.py:103`, which is D-4's deliberate literal.
`BundleLayout.master_csv` is gone. `gui/_config.py` and `gui/_schema_cache.py` are
converted with the fallback decision stated.

**By filename:** five live sites remain — B-4 above, three of them user-visible.

### The register

Highest index **42**, measured with the header's command. Entries 39-42 are P4's, and
they are accurate about what they measured. Three P4-specific items are **not** in the
register and would fit its four kinds:

| Would-be entry | Kind |
|---|---|
| `_cli_finalize_run.py:343-344`'s present-tense claim about P5's shard clearing (C-5) | *pointer to a name that does not exist yet* — the register's own shape 1 |
| `test_finalize_run.py:1367-1369`'s *"the PRE-INVERSION producer's output"* (C-2) | **never true** |
| `phase-4-finalize-run.md:816`'s three-key `metadata_table` example, contradicted by the ruling four lines below it (C-3) | **stale**, and specifically the self-contradiction shape the plan's own header names at `:30-38` |

Whether the prose half of the blast-radius class (B-4) warrants a fifth is a judgement
call I'd leave to you: entry 42 covers the value-carrying variant, and B-4 is the same
mechanism with a phrase instead of a symbol. If entry 42 is meant to be the general
statement, a sentence in it naming the prose variant is cheaper than a new entry.

### Two things I could not determine

- **Whether the cost-inversion measurement (B-2) was performed and simply not written
  down.** No artifact anywhere in the tree records it, and the retirement of
  `_consistent_embedded_join_keys` removed the reader it was about. *Cannot be
  determined* from the tree; the lead or the executing session would know.
- **Whether `--mode recompile` on a `--metadata` tree was ever exercised end to end
  during P4.** No test does it, and the commit messages do not say. The refusal is
  derivable from the code with certainty; whether anyone observed it is not.

---

## Commands used for the two totals in this report

```bash
# Drift register highest index (the header's own command)
cd docs/superpowers/reports/2026-09-03-cli-gui-state-tracking && \
  grep -oE '^(\| [0-9]+ \||### Entry [0-9]+)' document-drift.md \
  | grep -oE '[0-9]+' | sort -n | tail -1
# -> 42

# Sites in src/ naming the master as CSV
grep -rn "master_measurements\.csv\|master_measurements\.{csv" src/
# -> 12 hits, of which 7 are D8-explanatory notes and 5 are current-fact claims (B-4)
```
