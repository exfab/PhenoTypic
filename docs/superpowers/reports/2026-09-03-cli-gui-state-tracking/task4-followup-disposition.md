# P7 Task 4 follow-ups: disposition, and the `$SCRATCH` measurement

Companion to [`task4-projection-review.md`](task4-projection-review.md). That file records
what the reviewer found; this one records **what was done about each item**, and corrects
the one finding that turned out to be wrong.

Written 2026-09-11 as a handoff. Everything below is on branch
`fix/migrate-project-legacy-tables` (PR #221), in two commits:

| Commit | Subject |
|---|---|
| `92be762a` | `fix(finalize): project legacy embedded tables at read so migrated runs keep their measured rows` |
| `12238a6b` | `refactor(recompile): stop rewriting stores, and close the three Task 4 gaps` |

---

## ⚠️ FU-2 is wrong, and it is the finding most likely to be acted on

**FU-2 says `$SCRATCH` staging on the embedded arm is "pure overhead". It is not.
Staging stays on both arms. Do not remove it.**

The finding's *reasoning* is sound and its *conclusion* does not follow, which is why it
survived several readings — including mine. It was acted on in an earlier revision of this
branch and then reverted after measurement.

**The argument for removal.** `project_embedded_measurement_table` always reads each store's
descriptor from the store itself, never from the copy, so staging cannot avoid the per-store
GPFS round trip it exists to amortize; and each table is then read twice, once to copy and
once to project. Both halves are true.

**Why it still loses.** One bulk multithreaded copy of many small files beats reading them
individually by far more than the second read costs. GPFS small-file latency dominates, and
the copy converts the access pattern rather than eliminating it.

### The measurement

Full workload each time: all **6,529** embedded tables of the migrated Maresca run
(`ucr_029_e_d_Maresca/data/results/2026-08-11-ome-zarr`, 527 MB, 128,598 rows). One
configuration per **freshly allocated** node, two independent node pairs:

| | pair 1 | pair 2 | per table |
|---|---|---|---|
| **staged** | 323.9 s (i07) | 325.1 s (i08) | ~49.7 ms |
| **unstaged** | 397.3 s (i06) | 410.3 s (i09) | ~61.8 ms |

Staged is **~20% faster cold**, reproduced. The between-group gap (~75–85 s) is an order of
magnitude larger than the within-group spread (1.2 s and 13 s). Of the staged total, 156–161 s
is the copy itself; the aggregate phase still pays GPFS for every store's `zarr.json`, which is
exactly the round trip FU-2 correctly says staging cannot avoid.

**Warm, the two are a wash** (~46 s either way, with a 2.4–2.5 s copy). That is the number an
interleaved benchmark produces, and it is why the first attempt reported "staging is overhead,
+1.3 s". **A finalizer reading a just-written run is the cold case.**

### Two benchmark designs that produced void numbers

Recorded because both looked reasonable and neither announced its failure:

1. **Interleaved** (job 28307428) — four runs alternating staged/unstaged in one process. Run 1
   pulled all 527 MB into the page cache, so every later run was warm. Only one cold number
   exists in it (unstaged, 476.9 s); the "+1.3 s" verdict compares two warm runs.
2. **Split halves, swapped** (jobs 28308104/28308105) — each configuration gets its own cold
   half. Defeated twice over: the swapped job landed on the **same node** it had already
   warmed, and the halves are unequal (93,023 vs 35,575 rows), so per-table normalization does
   not rescue it.

**The rule that came out of it:** one configuration per freshly allocated node, over the whole
workload, repeated on a second independent node pair before believing the sign. Never
interleave configurations in one process; never reuse a node.

The numbers are also recorded at the call site, `_cli_finalize_run.py` (the `-- Stage to
$SCRATCH` comment above `_stage_to_scratch`), so the argument is answered where someone would
next make it.

### What FU-2 got right, and what is still open

- **"Leaks on the new raising paths"** — fixed. Cleanup is in a `try/finally`, and both arms
  now have a leak test: `test_a_projection_that_raises_still_removes_its_scratch_staging`
  (embedded) and `test_an_aggregation_that_raises_still_removes_its_scratch_staging` (legacy).
- **"Untested: a mis-keyed `read_paths` is invisible"** — **STILL OPEN, and it is the highest
  value item left.** Nothing asserts that each `read_path` is the staged copy *of its own*
  `table_path`. A swapped mapping would attribute one store's rows to another store's
  `filename`, the output would stay plausible, and no test would fail. FU-2's suggested test
  is the right one: set `SCRATCH`, wrap the projection to record `(table_path, read_path)`, and
  assert each `read_path` is byte-identical to its own `table_path`.

---

## Disposition of every item

### Must-fix (all three fixed in `92be762a`)

| | Item | Disposition |
|---|---|---|
| MF-1 | dtype advisory skipped across a shard boundary | Fixed; advisory runs at the shard merge too. Mutation-pinned. |
| MF-2 | every store excluded → fan-out publishes empty master + proof | Fixed; `_merge_measurement_shards` returns `None`, so no master overwrite, no mirror, no proof. |
| MF-3 | false K-invariance claim in a docstring | Sentence corrected. The behaviour itself is FU-8, still open. |

### Follow-ups

| | Item | Disposition |
|---|---|---|
| FU-1 | excluded store visible only in a log; run reads `incomplete` with no reason | **Done** (`12238a6b`). Deep verification records `declares_measurements` + `projectable_measurements` per image; `resolve_run_state` names stores where they disagree, and separately reports a certified-vs-verified shortfall **without diagnosing it** (the same shortfall is produced by an image finishing after the master was published). Recorded at verification time, not advisory time, so the shallow path the GUI polls emits the same advisories as the deep one. `VERIFICATION_CACHE_VERSION` → 2. |
| FU-2 | `$SCRATCH` staging is pure overhead | **Wrong — see above.** Staging kept. Leak half fixed; mis-keying half still open. |
| FU-3 | serial per-store reads; `SECONDS_PER_IMAGE_S2` measured on the old path | **Open.** Constant is 0.026 s/image; ~0.047 s/image measured with the join. |
| FU-4 | recompile shard with every store excluded raises | **Done** (`12238a6b`). Writes an empty shard, matching the forward fan-out. Aligned with MF-2's ruling. |
| FU-5 | statements this change makes false | **Done.** `cli_modes.md`, `_cli/CLAUDE.md`, root `CLAUDE.md`, `tracked_state.md` updated; the plan's Task 5 Step 1e marked ⛔ SUPERSEDED. |
| FU-6 | test-strength notes | **Partly done.** See "the fixture that was always green" below. |
| FU-7 | nothing asserts baseline migrate embeds carry no user metadata | **Open.** |
| FU-8 | underlying K-dependent widening (pre-existing) | **Open.** |
| FU-9 | confirm the dropped mirror rows are absent by name | **Done.** 14,784 dropped rows are metadata-coverage, not a join bug: 839 images absent from `metadata.csv`, plus edge grid rows 0/7. |

### The three "known gaps" the implementer self-reported

All closed in `12238a6b`: recompile shards now report their merged work ids to the finalizer as
`planned_work_ids` (only when *every* measurement status carries the key — an absent set is not
the claim an empty one makes); an all-excluded recompile shard writes empty; and an excluded
store is now surfaced through the run-state advisory rather than a log line alone.

---

## Two traps found while verifying, worth not re-learning

**A test that was always green is not evidence.** `test_recompile_writes_no_store_byte`
passed against the *pre-change* code, because `promote_recompile_table_transition`
short-circuits when the current table already equals the intended one
(`92be762a`:636-638) — the fixture published its snapshot before the stores, so the rewrite
had nothing to do. The fixture was rebuilt as `_publish_a_tree_the_old_rewrite_rewrote` to
produce migrate's real shape (stores first, snapshot after) and to assert its own premise.
Both it and `test_a_metadata_tree_recompiles_end_to_end` now **fail on the pre-change tree**.

**Deleting a function deletes what it carried.** Two behaviours lived inside the deleted
per-store rewrite and had to be re-established explicitly, not inherited:

- the unrecoverable-authority abort (now `_refuse_unrecoverable_recompile_authority`),
  reproducing the old control flow *including the arm that returns without asserting*, so no
  run the previous release completed starts aborting;
- the metadata snapshot fallback, which was computing the right value and handing it to the
  wrong consumer — with the master correctly un-joined, a bare `--mode recompile` was
  publishing a mirror with **no user metadata at all** until `effective_metadata` was routed
  to `aggregate_measurements`.

One deliberate narrowing: a marker-only legacy tree no longer reaches the authority assertion,
because `_standalone_marker_sources` went with the rewrite. Direction is safe (fewer aborts,
never more), and `SCHEMA_GATE_ARMED` is due to refuse that shape outright.

---

## State of the verification

- `tests/unit/cli`: **3047 passed, 2 skipped, 12 xfailed**. (`test_concurrent_process_appends_
  do_not_lose_records` fails under `-n 8` on an 8-core allocation and passes in isolation — the
  oversubscription casualty the `run-phenotypic-test` skill documents, not a regression.)
- ruff clean on every changed path; mypy 431 errors / 126 files against a pre-existing baseline
  of the same shape.
- The migration itself is **done and verified**: the hardlinked copy at
  `2026-08-11-ome-zarr` was re-migrated on `92be762a`, 0 failures, mirror correct. The original
  `2026-08-11/` was never written to.
- The FU-1 advisory was checked against that real run: 128 of 6,657 stores declare no
  projectable column list; **all 128** also have no measurement table and no `measurements`
  artifact in their record, so the conjunction is false and nothing fires. Zero false positives.

## If you need to re-measure staging

The benchmark scripts are at
`ucr_029_e_d_Maresca/scripts/migrate_2026-08-11/bench_scratch_onenode.{py,sbatch}`, but they
point at a frozen worktree (`.worktrees/prechange-recompile`) that has since been removed.
Create a fresh worktree detached at the SHA under test and repoint `FROZEN=` before running,
or the scripts measure nothing.

## What to pick up first

1. **The `read_paths` keying test** (FU-2's open half) — a silent row-misattribution hazard in
   code that is now confirmed staying.
2. FU-7, FU-8, FU-3 — all pre-existing, none blocking.
3. Cosmetic debt noted in passing: `measurement_columns` is spelled as a literal in six places
   and wants one constant; `_remap_to_scratch` has no `src/` caller (only a test imports it);
   `_cli_recompile_tables.py` is now a misleading filename, holding only the marker republisher.
4. A stale run proof can still read `complete` — recorded during this change, not caused by it.
