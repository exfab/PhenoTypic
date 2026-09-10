# Run-completion contract, identity schema, and measurement/metadata layout — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace nine cross-checked evidence sources for "is this run done?" with three
written authorities and one `resolve_run_state()` reader; collapse fourteen identity
tokens to six, three of them content-derived; make the completion predicate `O(N)` in
`stat()` instead of `O(N)` in full-file SHA-256; and invert the measurement/metadata
layout so embedded per-image tables carry measurements only and the metadata join moves
to one shared `finalize_run`.

**Architecture:** A new **read-only** module `sdk_/_run_state.py` owns `RunIdentity`,
`ImageState`, `RunState` and `resolve_run_state(output_dir, *, depth)`, backed by a
stat-keyed, **in-process** verification cache that can only ever cause *re*-verification.
Writers stay in `phenotypic._cli`. Per-image state collapses from three marker trees into
one `.phenotypic/progress/images/<ds>/<stem>.json` record with an open `stages` map.
Embedded per-image tables become pure measurements and each store's user metadata is
written **during its original promote**, so no artifact carrying a content proof is ever
mutated. `finalize_run` becomes the one aggregation + join + publish path for `full`,
`measure` and `recompile`, fanned out over SLURM array tasks and local `--njobs` with a
reserved `TASK_FINALIZE` trigger entry. Every legacy read path moves into `--mode migrate`;
every other mode refuses an unconverted tree.

**Tech Stack:** Python 3.11–3.12, `polars` (aggregation/mirror), `pyarrow` (embedded
tables), `zarr>=3.0` (NGFF 0.5 / Zarr v3 stores), `duckdb` (QC), Dash/Flask (GUI),
SLURM (`sbatch` arrays), `pytest` + `pytest-xdist`.

**Spec:** [`docs/superpowers/specs/2026-09-03-cli-gui-state-tracking/design.md`](../../specs/2026-09-03-cli-gui-state-tracking/design.md)
· companions [`audit.md`](../../specs/2026-09-03-cli-gui-state-tracking/audit.md),
[`DEFERRED.md`](../../specs/2026-09-03-cli-gui-state-tracking/DEFERRED.md)

**Decisions and open questions:** [`OPEN-QUESTIONS.md`](OPEN-QUESTIONS.md). **Read it
before Phase 0.** Three decisions taken by the user on 2026-09-03 depart from the spec as
written and are binding on every phase doc:

| | Decision | Effect |
|---|---|---|
| **D-A** | Per-store metadata is **written at promote time**, not backfilled into proven stores | Cuts §6.3 (hardlink re-promote), §6.4's generalisation, `stages.backfilled`, the backfill fan-out, spike S-1, and residual risk 4. Keeps §7's inversion in full. |
| **D-B** | The verification cache is **in process**; spike S-5 decides whether it ever moves to disk | No new tracked artifact unless measured to be needed. |
| **D-C** | `scientific_config_digest` **is** `processing_configuration_digest`, verbatim | §5.1 holds; §5.4's field list is corrected in P2. |

Together they remove the only mechanism in the design that mutates artifacts already
carrying a content proof, and the only new file the design would have added to
`.phenotypic/`.

**Branch:** `cli-gui-state-tracking` (worktree `.worktrees/cli-gui-state-tracking`).

---

## Global Constraints

Every task's requirements implicitly include this section. Values are copied verbatim
from the spec; where a task restates one, the value here is authoritative.

### The organising principle

> **Move state that is tracked to state that is checked.** Anything derivable from files
> on disk is derived, not written down and kept in sync.

A reviewer rejects any task that adds a written counter, a cached count, or a second
place a fact lives. The three exceptions are named in §4.1 and nowhere else.

### The three written authorities (spec §4.1)

| Authority | File | Why it cannot be derived |
|---|---|---|
| Accepted inventory | `work_ids` in `processing_state.json` | A directory listing is a different question from "what did this run accept". |
| Terminal failures | `.phenotypic/terminal_failures.jsonl` | A failure leaves no artifact. |
| Liveness & ownership | `slurm_lifecycle.json`, `slurm_jobs.jsonl`, `gui_launch_owner.json` | External-system and process facts. |

Everything else is either a **content proof** (per-image record, aggregate proof, run
proof — digest manifests over artifacts that already exist) or **diagnostics**.

### The four verdicts (spec §4.3)

`RunState.completion ∈ {"complete", "incomplete", "failed", "active"}`.
**`contradictory` does not exist.** Evaluated in exactly this order — see
[OPEN-QUESTIONS Q2](OPEN-QUESTIONS.md#q2-verdict-precedence-is-unspecified):

1. valid run proof covering the current inventory → `complete`
2. else a liveness authority reports work in flight for the current identity → `active`
3. else terminal-failure records exist with no superseding success proof → `failed`
4. else → `incomplete`

Half-migrated trees holding unconverted `.h5` files add a `RunState.advisories` entry.
**An advisory is never a gate.**

### The six identity tokens (spec §5.1)

`work_id` (content) · `processing_generation` (content) · `publication_id` (content) ·
`restart_epoch` (tracked counter) · `scheduler_epoch` (opaque) · `owner_generation`
(opaque). Nothing else reaches disk as identity. `record_revision`, registry `revision`
and `binding_generation` are in-memory CAS counters. Per-image `attempt_id` and the
event-log `generation` field are written but **never branched on**.

### Invariants (each is a test, and each must be proved able to fail)

- **INV-CACHE** — the verification cache can only cause re-verification, never a wrong
  `complete`. No code path yields a positive verdict from a cache entry alone. The
  invariant is about what a cache may *cause*, so it binds the in-process cache exactly as
  §9.1 wrote it for the on-disk one. (spec §9.1; D-B)
- **INV-INPUTS** — `finalize_run` step 1 selects exactly the marker-authorized embedded
  measurement tables. It never reads a prior master, chunk parquet, measurement shard,
  `analysis_full.parquet`, or `_dataset_aggregated.parquet` as an aggregation input.
  (spec §7.5)
- **INV-IMMUTABLE** — **no artifact carrying a content proof is ever mutated.** Under D-A
  there is no post-proof store write at all: a store's metadata table is written in its
  original `.part`, before the root `zarr.json`. The one pre-existing exception,
  `refresh_success_markers_after_metadata_migration` (`_cli_completion.py:305`), stays
  scoped to `--mode migrate` and keeps its `RuntimeError` for a marker-bound artifact that
  moved without a covering receipt. (spec §6.2, §13.2; D-A)
- **INV-DEGRADE** — unreadable ⇒ not complete, never complete. Every parse failure
  degrades toward `incomplete`. (spec §13.3)
- **INV-LAYER** — `sdk_/_run_state.py` imports nothing from `phenotypic._cli`, at module
  scope or inside a function, and exports **only readers**. Pinned by an import test.
  (see [OPEN-QUESTIONS Q4](OPEN-QUESTIONS.md#q4-sdk_-cannot-import-_cli-for-the-readers-it-must-host))

### Digest composition (spec §5.3) — four digests, four questions

| Digest | Question | Lives in |
|---|---|---|
| `inventory_digest` | Did the accepted **scope** change? | aggregate + run proofs |
| `source_set_digest` | Did the **succeeded subset** change? | aggregate proof |
| `scientific_config_digest` | Did the **pipeline** change? | both proofs |
| `finalization_input_digest` | Did the **join/QC inputs** change? | both proofs |

`inventory_digest` stays **out** of the generation digest (D7): generation fences
configuration, `inventory_digest` fences scope, and they change on different schedules.

`finalization_input_digest` is a **versioned object**, not a flat digest:

```json
{"schema_version": 1, "metadata_sha256": "…", "include_dataset_column": true, "no_qc": false}
```

Adding a field is a `schema_version` bump handled by the reader, never a second tree
migration.

### Layout rules that do not move

- **Marker-last publication, end to end.** Store root `zarr.json` last → per-image record
  after artifacts → aggregate proof after outputs → run proof after aggregate. Never
  reorder. **No store write may follow a per-image record publication.** (Under D-A there
  is no exception on any forward path; `--mode migrate`'s pre-existing receipt path is the
  only one in the tree and P7 keeps it scoped there.)
- **`deliverables/metadata.csv` is byte-exact input provenance and is never rewritten by
  any mode, `--mode migrate` included.** `--mode migrate` emits
  `deliverables/metadata.canonical.csv` alongside it. `metadata.original.csv` does not
  exist and must not be created. (project `CLAUDE.md`; spec D9/FLOW-4)
- **Never hand-join a path.** Resolve through the `phenotypic.sdk_` helpers
  (`progress_dir`, `results_dir`, `deliverables_dir`, `zarr_store_path`,
  `image_completion_marker_path`, …). Every new filename lands in
  `sdk_/_io_constants.py`, next to its siblings.
- **`TASK_FINALIZE` is a reserved trigger entry inside the array task list, never a
  parallel sidecar job**, and is counted when sizing chunks against `MaxArraySize` /
  `MaxSubmitJobs`. This is the existing `__PHENOTYPIC_CHECKPOINT__` /
  `__PHENOTYPIC_MANIFEST__` dispatch contract (`src/phenotypic/_cli/CLAUDE.md`).
- **`master_measurements.parquet` is parquet-only** after Phase 4 (D8).
  `master_measurements.csv`, `MASTER_MEASUREMENTS_CSV`, `master_measurements_csv_path()`,
  `BundleLayout.master_csv` and `load_master_measurements()` are deleted.
  The aggregate proof's `required_outputs` drops from four artifacts to three.
- **Vendored reference sources under `docs/superpowers/specs/*/refs/` are read-only.**
  Never lint or format them.

### Tooling

- `uv` is the sole runner. Never bare `python` or `pip`.
- `uv run ruff check --fix <explicit paths you changed>`. **Never bare `ruff check --fix`.**
- `uv run mypy src/phenotypic` at each phase gate.
- Any pytest run beyond a single test file goes through the **`run-phenotypic-test`**
  skill; the full `tests/unit` suite is ~65 minutes and is a Slurm job
  (**`slurm-job`** skill), not a foreground command. `QT_QPA_PLATFORM=offscreen` is
  mandatory; `-n auto` is wrong on HPCC (it reads node cores, not the allocation).

---

## Phase index

Each phase leaves a working tree and is independently reviewable. Phase docs are ordered
by dependency, not by file.

| Phase | Doc | Content | Depends on | Gate |
|---|---|---|---|---|
| **P0** | [phase-0-spike-gate.md](phase-0-spike-gate.md) | S-2, S-3, S-4, S-5 on real GPFS. **S-1 is cut by D-A.** | — | S-5 verdict decides P1's cache shape |
| **P1** | [phase-1-run-state-sdk.md](phase-1-run-state-sdk.md) | `sdk_/_run_state.py`: `RunIdentity`, `ImageState`, `RunState`, `resolve_run_state`, in-process cache + INV-CACHE. No consumers moved. | P0 (S-5) | INV-CACHE mutation suite green and proved able to fail |
| **P2** | [phase-2-identity-schema.md](phase-2-identity-schema.md) | Content-derived `processing_generation`, `restart_epoch`, `scheduler_epoch` collapse | P1 | stale-worker test |
| **P3** | [phase-3-per-image-record.md](phase-3-per-image-record.md) | One per-image record with `stages`; three marker trees deleted | P1, P2 | resume-equivalence matrix |
| **P4** | [phase-4-finalize-run.md](phase-4-finalize-run.md) | Embedded-table inversion, promote-time `pht-metadata.parquet`, `finalize_run`, local path only | P3 | INV-INPUTS, INV-IMMUTABLE |
| **P5** | [phase-5-fanout.md](phase-5-fanout.md) | SLURM array + `--njobs` fan-out of aggregation | P4, P0 (S-2, S-3) | partial-failure matrix |
| **P6** | [phase-6-consumer-migration.md](phase-6-consumer-migration.md) | Consumer migration and the ~1,400-line deletion | P1–P5 | full `tests/unit` + `tests/gui` suite on Slurm |
| **P7** | [phase-7-migrate-mode.md](phase-7-migrate-mode.md) | `--mode migrate` conversion; refusal in every other mode | P1–P6 | dry-run + rollback test |

**Phase gates are not optional.** After each phase: `uv run mypy src/phenotypic`,
`uv run ruff check --fix` on the changed paths, the phase's own test selection, and a
code review before the next phase starts.

---

## Why this is one change and not seven PRs

Decision **D9**, taken by the user after the review/bisect cost was stated. The three
problems are entangled: the completion predicate's cost is caused by the marker
cardinality, the marker cardinality is caused by the identity cardinality, and the
measurement/metadata inversion changes what the proofs certify. Splitting them means
shipping three intermediate contracts nobody wants to support.

The mitigation is §12's phases, reproduced above — **not** the absence of the risk.
Residual risk 3 in the spec stands: this is hard to bisect. Every phase therefore ends at
a commit that passes its own gate, so a bisect lands on a phase boundary rather than
mid-rewrite.

---

## What this plan deliberately does not do

Taken from [`DEFERRED.md`](../../specs/2026-09-03-cli-gui-state-tracking/DEFERRED.md).
A task that touches one of these is scope creep and a reviewer rejects it:

- **The SLURM observer's decision tree**, its 30-second reconciliation grace window, and
  `squeue`/`sacct` state ranking (D-1). Only its two `_cli_completion` call sites and its
  Stage-3 probe move — see P6 Task 6.
- Everything in DEFERRED's "unrelated churn" table (S10, S12, S14, S16, S17, S20, S22–S28)
  **except** the items §11.2 explicitly folds in, which are listed in P6 Task 8.
- Incremental aggregation (D-3), third-party Zarr chunk-tree probing (D-4),
  `BrowseCache.usage()` (D-5).

**One exception, recommended:** DEFERRED **D-2** (stale `gui_launch_owner.json` has no
repair path) is folded into **P6 Task 5**, because P6 rewrites
`_assert_output_claimable_locked` — the exact predicate that causes the dead-end — and
under the new verdict precedence a stale owner record masks `incomplete` as `active`.
This is called out in the phase doc as an explicit scope addition, not smuggled in.

---

## Spikes

Spike scripts live in [`spikes/`](spikes/) beside this plan, **not** in
`docs/superpowers/logic_validation_scripts/`. That directory's contract is that nothing
in it imports `phenotypic`, which is what makes it an independent witness; these spikes
must drive the shipped code and measure it on real stores, so they import `phenotypic`
and belong here (spec §10, project `CLAUDE.md`).

---

## Execution handoff

Two execution options:

1. **Subagent-driven (recommended)** — a fresh subagent per task, review between tasks.
   Dispatch through the **`orchestrate-subagent`** skill: subagents send commands with
   side effects back to the orchestrator to run, and the orchestrator returns output
   verbatim. Skipping that round-trip stalls the run silently.
2. **Inline execution** — batch the tasks in-session with checkpoints, via
   `superpowers:executing-plans`.

**P0 runs first either way.** S-5's verdict decides whether P1 Task 3 builds an in-process
cache alone or an in-process cache with an on-disk tier (D-B), so P1 does not begin until
S-5 has run. Unlike the spec's original S-1, no P0 result can now invalidate the design —
D-A removed the mechanism S-1 was gating.
# Decisions and open questions

Found while grounding the plan in the real tree at `c9d1fbfc`. Each carries the evidence,
the decision, and who took it.

**D-A, D-B and D-C were taken by the user on 2026-09-03** and are binding on the phase
docs. They are recorded here rather than folded silently into the plan because each one
*departs from the spec as written*, and a reader comparing the two needs to see why.

| | Question | Decision | Taken by |
|---|---|---|---|
| **D-A** | Per-store metadata: backfill into proven stores, or write at promote time? | **Write at promote time.** §6.3, §6.4's generalisation, `stages.backfilled`, the backfill fan-out and residual risk 4 are all **cut**. | user |
| **D-B** | Verification cache on disk, or in process? | **In process first.** A spike (S-5) at implementation time decides whether the on-disk variant is needed at all; prefer in-process if it works. | user |
| **D-C** | Q1 — `scientific_config_digest`'s field list | **Keep `work_id` unchanged.** §5.4's prose is the wrong half. | user |
| Q2 | Verdict precedence unspecified | Plan decides: `complete` > `active` > `failed` > `incomplete` | plan |
| Q3 | `ImageState` used but never defined | Plan defines it (P1 Task 2) | plan |
| Q4 | `sdk_` cannot import `_cli` | Plan resolves: plain-JSON state read, INV-LAYER test | plan |
| Q6 | Ten test files depend on `master_measurements_csv_path` | Sizing note for P4 | plan |

---

## D-A. Per-store metadata is written at promote time, not backfilled

**The spec's organising principle is "move state that is tracked to state that is
checked". §6.3/§6.4 move in the opposite direction**, and they are the largest source of
new machinery in the design:

- a hardlink re-promote protocol, itself spike-gated on S-1 (§6.3)
- a **new artifact class** — `.phenotypic/rewrites/<kind>-<digest>.json` certified
  receipts (§6.4)
- a `stages.backfilled` entry carrying its own `metadata_sha256`
- the backfill half of the SLURM fan-out (§8)
- a **new partial state**, which the spec itself records as residual risk 4:
  `deliverables/` complete and correct while `results/` is not yet self-describing
- a metadata edit re-promoting every store in the tree (§7.4)

All of it exists so that an already-promoted, already-proven store can gain
`tables/metadata/pht-metadata.parquet`. **The spec never names the consumer.** The GUI
reads the mirror; analysis reads the mirror; the master is intrinsic-identity-only by
design (§7.3). The benefit is a store that is self-describing to a third party — real, but
bought with the only mechanism in the change that mutates artifacts already carrying a
content proof.

### What is built instead

`pht-metadata.parquet` is written **during the store's original promote**, in the same
`.part` as `tables/measurements/table.parquet`, before the root `zarr.json`. The store's
existing marker-last publication protocol is untouched, and **no post-proof store mutation
exists at all** — §6.2's immutability constraint is *preserved* rather than renamed.

`deliverables/metadata.csv` is copied byte-for-byte before any per-image work starts, so
within one invocation the metadata a store is built against is already fixed.

### What is given up, and how it stays honest

A store built before `--metadata` was supplied (or against an older `metadata.csv`) keeps
the metadata it was built with. `metadata_sha256` does not participate in `work_id`
(correctly — it is a finalization input, §5.4), so a metadata edit does **not** invalidate
per-image proofs and does **not** rebuild stores. §7.4's guarantee therefore narrows:

> **Was:** a `metadata.csv` edit re-runs `finalize_run`, re-joining the mirror **and
> re-backfilling every store**, without touching a single image's measurement.
>
> **Becomes:** a `metadata.csv` edit re-runs `finalize_run`, re-joining the mirror.
> Stores keep the metadata snapshot they were built against; each store's
> `phenotypic.metadata.snapshot_sha256` records which one.

That divergence must be **derived and surfaced, never tracked**: `resolve_run_state` adds
one advisory when any store's recorded `snapshot_sha256` differs from the current
`metadata_sha256`. It is a `stat` + one attribute read per store on the deep path, it
reuses a value the store already carries, and — per §4.3 — **an advisory is never a gate**.
See P1 Task 5 and P4 Task 6.

### Consequences, phase by phase

| Spec section | Fate |
|---|---|
| §6.3 hardlink re-promote | **Cut.** Spike S-1 is no longer a gate and is dropped from P0. |
| §6.4 certified post-hoc rewrite protocol | **Generalisation cut.** The *existing* `refresh_success_markers_after_metadata_migration` (`_cli_completion.py:305`) stays exactly as it is, serving the one historical case it was written for; P7 keeps it for `--mode migrate`'s metadata-schema migration. INV-RECEIPT still holds — it is the behaviour that function already has. |
| §6.1 `stages.backfilled` | **Cut** from the record schema. `stages` stays an open map, so re-adding it later is additive. |
| §7.4 `finalize_run` step 6 | **Cut.** `finalize_run` is six steps, not seven. |
| §8 array task backfill half | **Cut.** Shard workers aggregate only. |
| §15.4 residual risk 4 | **Cut.** |
| §7.1–§7.3 measurement/metadata inversion | **Kept in full.** This half is clean subtraction and is the point of §7. |

---

## D-B. The verification cache starts in process, and a spike decides whether it moves to disk

Audit **S1** — the finding §9.1 responds to — proposed a **process-level** cache:

> Give `valid_image_success` a process-level cache keyed on the marker file's
> `(st_dev, st_ino, st_size, st_mtime_ns)` … pair it with the existing processing-inventory
> stat sweep rather than re-hashing.

§9.1 escalated that to `.phenotypic/verification_cache.json` — a new tracked artifact, in
the spec that exists to remove tracked artifacts. Every cadence the audit measured is a
**repeated call inside one long-lived process**: the observer's 2 s daemon tick, the
viewer's 5–10 s per-tab poll, `OutputRoot.discover`'s double read, `OutputMutationGuard`'s
double read. An in-memory cache serves all of them.

On-disk buys exactly one thing an in-memory cache cannot: **cold-start reuse across
processes** — a fresh GUI launch, the CLI deriving a resume worklist, each SLURM worker.
It costs an identity fence on disk, a corruption surface, the INV-CACHE mutation suite,
`clear_machine_state` coupling, and last-wins concurrent writes.

**Decision:** implement in-process. Run **spike S-5** at implementation time to measure
whether cold start actually matters at realistic `N`; only add the on-disk variant if it
does.

INV-CACHE and its mutation suite still apply to the in-process cache — the invariant is
about *what a cache may cause*, not about where it lives. The forged-file cases in the
suite become forged-dict cases; the corrupt-JSON cases move to S-5's on-disk variant if
one ships.

---

## D-C. `scientific_config_digest` is the existing digest, verbatim (was Q1)

### The evidence

Spec §5.4 says `scientific_config_digest` is "**not a new digest** … reused verbatim", then
lists its contents and claims they are "exactly as `work_id` does today", and separately:

> Fields that are finalization inputs rather than per-image configuration
> (`metadata_sha256`, `include_dataset_column`, `no_qc`) belong to
> `finalization_input_digest` and appear in **neither** `work_id` nor the generation.

The actual function, `processing_configuration_digest_from_values`
(`src/phenotypic/_cli/_cli_failure_tracker.py:200-243`), has this non-process branch:

```python
    else:
        payload.update(
            {
                "include_dataset_column": include_dataset_column,
                "overlay_alpha": overlay_alpha,
                "save_overlays": save_overlays,
            }
        )
```

folded into `work_id` at `_cli_failure_tracker.py:265`. So today:

- **`include_dataset_column` IS in `work_id`.** §5.4 says it is in neither.
- `overlay_alpha` and `save_overlays` are in `work_id` and appear nowhere in §5.4's list.
- `ext` and `process_format` are in the **process** branch only; §5.4 lists
  `process_format` unconditionally.

`validate_resume_compatibility` (`_cli_state_management.py:337-346`) guards the same
superset, so §5.4's appeal to "the fields `validate_resume_compatibility` already guards"
also under-lists. And §5.1 states `work_id` is "unchanged" — which §5.4 as written
contradicts.

### Decision

`scientific_config_digest := processing_configuration_digest(config)`, verbatim and
unchanged. §5.1 holds. §5.4's **argument** holds — it needs `generation ⊇ work_id`'s config
digest so the two can never disagree about what counts as scientific configuration, and
identity satisfies that maximally. §5.4's **field list** is the demonstrably wrong half and
is corrected in P2.

Consequence: `include_dataset_column` appears in both the generation and
`finalization_input_digest`. That is not incorrect — the two answer different questions —
but §5.3's "none is redundant" needs the footnote P2 adds. Flipping it still reprocesses
every image; removing the three fields from the per-image digest is a sound follow-up, and
it is a `work_id` change that deserves its own spec and its own migration rather than a
ride on this one.

---

## Q2. Verdict precedence

**Status: gap in the spec. Plan decides.**

Spec §4.3 defines four verdicts and asserts "there is exactly one path to each verdict",
but never orders them, and three of the four can hold at once on a real tree.

**Decision — first match wins, in this order:**

1. valid run proof covering the current inventory → `complete`
2. else a liveness authority reports work in flight **for the current identity** → `active`
3. else terminal-failure records exist with no superseding success proof → `failed`
4. else → `incomplete`

`complete` outranks `active` because a run proof covers the *current* inventory: a live
worker at that point is either fenced by `restart_epoch` (stale) or is a new invocation
that has already changed the inventory — in which case rule 1 does not fire. The ordering
is self-resolving rather than a tie-break.

`active` outranks `failed` so a failure journal entry from a previous attempt cannot mask
an attempt currently retrying it.

**This ordering is a test** (P1 Task 5), not a comment.

---

## Q3. `ImageState` is used but never defined

Spec §9 declares `images: Mapping[str, ImageState]` and never defines the type. P1 Task 2
defines it. With D-A, its `stages` map carries `stage1`/`stage2`/`stage3`/`measured` and
**not** `backfilled`.

---

## Q4. `sdk_` cannot import `_cli` for the readers it must host

Spec §11 moves `_cli_completion.py`'s readers into `sdk_/_run_state.py` and calls the
asymmetry "structural, not conventional". But `_cli_completion.py:14` imports
`phenotypic.sdk_` at module scope, and today's readers call `load_processing_state`
(`_cli_state_management.py:98`), which **replays the whole event log** on every load
(`:121`). `sdk_` already reaches into `_cli` from 16 sites, all lazily inside function
bodies.

**Decision:** `sdk_/_run_state.py` imports nothing from `phenotypic._cli`, at module scope
or inside a function, and reads `processing_state.json` as plain JSON. That is possible
precisely because §4.2 deletes `processing_state.datasets.{completed,failed,started}` and
demotes the event log out of the evidence set: what a verdict depends on is
`config.work_ids` and the digests, all literal JSON fields.

Pinned by **INV-LAYER**, an AST test in P1 Task 1. Without a test, "structural" is
convention with extra steps — the GUI's 25 private `_cli` imports across 9 modules are
what that looks like at scale.

---

## Q6. Ten test files depend on `master_measurements_csv_path`

Sizing note, not a contradiction. `grep -rl` gives 10 files in `tests/`, 6 in `src/`;
`MASTER_MEASUREMENTS_CSV` gives 4 and 1. D8 deletes all of them in P4.

`BundleLayout.detect` (`sdk_/_io_constants.py:2422`) keys discovery on
`master_measurements.parquet`, not the CSV, so bundle detection is unaffected — the risk is
confined to fixtures that write or assert the CSV.

---

## Still open, not blocking — raised for a later pass

Neither of these changes what P0–P7 build. Both are recorded so they are visibly
*deferred* rather than unnoticed.

### O-1. `scheduler_epoch` may be five names collapsing to one owner, not five tokens to one

§5.1 has `scheduler_epoch` "absorb" `slurm_generation`, staged `epoch`, `lifecycle_epoch`,
`execution_epoch` and recompile's `attempt_id`. Those five are written by four subsystems
(`_cli_slurm_lifecycle`, `_cli_staged_orchestration`, the recompile worker, the local
strategy) at four different times with four different lifetimes. Collapsing the *names*
without collapsing the *writers* gives one value with four owners, which is a coupling
increase dressed as a cardinality reduction.

P2 Task 4 therefore collapses them **only where a single writer already owns the
lifetime** — `slurm_generation` and `lifecycle_epoch` are the same value passed twice today
(the audit's own finding, §11.1: `_assert_worker_generation`'s
`slurm_generation != attempt_id` check is "one value passed twice, then asserted equal") —
and leaves staged `epoch` and recompile `attempt_id` as *diagnostic* fields written under
the collapsed name. If a later measurement shows four writers never actually race, the
rest of the collapse is a follow-up.

### O-2. `stages` is an open map with no name validation

§6.1 makes `stages` open so the schema can grow additively. Nothing then validates a stage
name, so `"stage_2"` reads as "stage 2 not done" and never errors. P3 Task 2 keeps the map
open but adds a module-level `KNOWN_STAGES` frozenset and emits a `RunState.advisory` for
an unrecognised key — surfacing the typo without closing the map.
# Phase 0 — Spike gate

**Depends on:** nothing. **Blocks:** P1 (via S-5), P5 (via S-2 and S-3).

**Spec:** §10, as amended by [D-A and D-B](OPEN-QUESTIONS.md).

**This phase writes no production code.** It produces four measurements and a written
verdict for each.

### What changed from spec §10

| Spike | Spec §10 | Here |
|---|---|---|
| **S-1** hardlink re-promote | Gates §6.3; a bad result cascades into §6.4 and §7.4 | **Cut.** D-A writes per-store metadata at promote time, so there is no re-promote to measure and no §6.3 to gate. |
| **S-2** shard sizing | Chunk-sizing formula; whether backfill shares a task | **Kept**, minus the backfill half — shard workers aggregate only. |
| **S-3** merge cost | Whether `TASK_FINALIZE` holds the merge in memory | **Kept unchanged.** |
| **S-4** backfill locality | Gates the §8 DAG | **Kept, reduced.** The question moves from "can a *post-hoc* worker project metadata locally?" to "does the *per-image* projection at promote time match what a global join would attribute?" — still worth asking, because duplicate-key fan-out and metadata-only rows are where a local projection quietly diverges. |
| **S-5** cache cold-start | *(new)* | **Added by D-B.** Decides whether the verification cache needs an on-disk tier at all. |

**No P0 result can now invalidate the design.** D-A removed the mechanism S-1 was gating.
S-5 chooses between two implementations of one task; S-2 and S-3 choose parameters.

**Files:**
- Create: `.../spikes/s2_shard_sizing.py`
- Create: `.../spikes/s3_merge_cost.py`
- Create: `.../spikes/s4_metadata_projection_locality.py`
- Create: `.../spikes/s5_cache_cold_start.py`
- Create: `.../spikes/RESULTS.md`
- Create: `.../spikes/run_spikes.sbatch`

**These scripts import `phenotypic` and therefore do NOT go in
`docs/superpowers/logic_validation_scripts/`.** That directory's contract — nothing in it
imports the code under test — is what makes it an independent witness, and the contract is
directory-wide, not per file (project `CLAUDE.md`; spec §10).

**Interfaces:**
- Consumes: nothing from this plan.
- Produces: `spikes/RESULTS.md`, whose verdict lines are cited by P1 Task 3 (cache shape),
  P4 Task 5 (metadata projection) and P5 Tasks 1 and 4 (shard count, merge strategy).

---

## Step 0: the fixture tree

Every spike needs a **real** PhenoTypic output tree on **GPFS** (`/bigdata` or `/rhome`),
not a tmpfs fixture — the point is metadata cost on the shared filesystem.

- [ ] **Step 0.1: Find or build one**

```bash
find /bigdata/exfab/anguy344 /rhome/anguy344/bigdata_exfab -maxdepth 4 \
     -type d -name 'zarr' -path '*/results/*' 2>/dev/null | head -20
```

Prefer a tree with **≥ 200 stores**. If none exists, build one:

```bash
uv run python -m phenotypic \
  --input <a real plate image directory> \
  --output /bigdata/exfab/anguy344/spike-fixture \
  --pipeline <a pipeline.json with a detector and MeasureShape> \
  --metadata <a real metadata.csv>
```

Record in `RESULTS.md`: the tree path, `N_stores`, total bytes, whether a `metadata.csv`
is present, and `df -T <path>` (must report `gpfs`).

---

## S-5 — Cache cold start (gates P1's cache shape) — **runs first**

**Measures:** the wall-clock of one **cold** deep verification at realistic `N`, and how
much a warm in-process cache saves on the second call. Decides whether the on-disk tier
D-B deferred is needed.

**Why this is the one that runs first:** it is the only P0 result that changes what P1
builds, and P1 is the phase everything else depends on.

- [ ] **Step 1: Write the spike**

`spikes/s5_cache_cold_start.py`:

```python
"""S-5: does the verification cache need an on-disk tier?

Decision D-B (OPEN-QUESTIONS): audit S1 proposed a PROCESS-LEVEL cache; spec §9.1
escalated it to a file. Every cadence the audit measured -- the observer's 2s tick,
the viewer's 5-10s poll, OutputRoot.discover's double read, OutputMutationGuard's
double read -- repeats inside ONE long-lived process, which an in-memory cache
serves completely. On-disk buys only cold-start reuse ACROSS processes.

This measures the thing that decision turns on: how expensive is a cold deep
verification, and how often would a process actually pay it?

Reports:
  cold_deep_seconds   -- one full verification from a cold process
  warm_stat_seconds   -- the same answer with every artifact already stat-able
  hash_bytes          -- total bytes hashed on the cold pass

Read it as: if cold_deep_seconds is small enough that a fresh GUI launch, a CLI
resume, or a SLURM worker start can absorb it, the in-process cache is sufficient
and no new tracked artifact ships.

Usage:
    uv run python .../spikes/s5_cache_cold_start.py <output_dir>
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path


def _marker_artifacts(output_dir: Path) -> list[tuple[Path, str]]:
    """Every (artifact, sha256) pair today's markers declare."""
    from phenotypic.sdk_ import progress_dir

    root = output_dir.resolve()
    pairs: list[tuple[Path, str]] = []
    for marker in (progress_dir(output_dir) / "image_complete").rglob("*.json"):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for descriptor in (payload.get("artifacts") or {}).values():
            rel = descriptor.get("path")
            digest = descriptor.get("sha256")
            if isinstance(rel, str) and isinstance(digest, str):
                pairs.append((root / rel, digest))
    return pairs


def main() -> int:
    output_dir = Path(sys.argv[1]).resolve()
    pairs = _marker_artifacts(output_dir)
    print(f"n_artifacts={len(pairs)}")

    # Cold: hash every declared artifact, exactly as valid_image_success does.
    hashed = 0
    t0 = time.perf_counter()
    for artifact, _digest in pairs:
        try:
            with artifact.open("rb") as handle:
                h = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    h.update(chunk)
                    hashed += len(chunk)
        except (OSError, IsADirectoryError):
            continue
    cold = time.perf_counter() - t0
    print(f"cold_deep_seconds={cold:.3f} hash_bytes={hashed} ({hashed / 1e9:.2f} GB)")

    # Warm: what a stat-tuple currency check costs instead.
    t0 = time.perf_counter()
    for artifact, _digest in pairs:
        try:
            st = artifact.stat()
            _ = (st.st_size, st.st_mtime_ns)
        except OSError:
            continue
    warm = time.perf_counter() - t0
    print(f"warm_stat_seconds={warm:.3f} speedup={cold / max(warm, 1e-9):.1f}x")

    # Extrapolate to 6,000 images at the observed per-artifact cost.
    per = cold / max(len(pairs), 1)
    print(f"projected_cold_seconds_at_6000_images={per * 6000 * 3:.1f}  "
          f"(3 artifacts/image: store root, measurements, overlay)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it on a compute node, on GPFS**

Compute work is a Slurm job — use the **`slurm-job`** skill:

```bash
srun -p short -c 4 --mem=16G -t 0:30:00 --pty bash
uv run python .../spikes/s5_cache_cold_start.py /bigdata/exfab/anguy344/spike-fixture
```

- [ ] **Step 3: Record the S-5 verdict**

Write exactly one of:

- **`S-5 IN-PROCESS SUFFICIENT`** — projected cold verification at N=6000 is under 30 s.
  A fresh GUI launch, a CLI resume, or a SLURM worker start absorbs it once; every
  subsequent call in that process is a stat sweep. **P1 Task 3 builds the in-process cache
  only. No new file ships.** This is the expected outcome and the one D-B prefers.
- **`S-5 ON-DISK TIER NEEDED`** — projected cold verification exceeds 30 s. P1 Task 3
  builds the in-process cache **plus** the `.phenotypic/verification_cache.json` tier from
  spec §9.1, with the full INV-CACHE mutation suite including the corrupt-JSON cases.
  Record the measured number that justifies the extra artifact **in the module docstring**,
  so a later reader can tell it was measured rather than assumed.

---

## S-2 — Shard sizing (gates P5's chunk formula)

**Measures:** per-image cost of one aggregation shard task's real work, at `K ∈ {1, 4, 16,
64}`, against the cluster's `MaxArraySize` / `MaxSubmitJobs`.

- [ ] **Step 4: Read the cluster's real limits**

```bash
scontrol show config | grep -E 'MaxArraySize|MaxJobCount|MaxSubmitJobs'
```

Record them. The plan assumes `MaxArraySize = 2500`, so the highest legal index is **2499**
(`--array=1-2500` is rejected) per the user's global `CLAUDE.md`. **If `scontrol` reports
something else, that value wins** and P5's formula uses it.

- [ ] **Step 5: Write and run `s2_shard_sizing.py`**

```python
"""S-2: how long does one aggregation shard task take?

Spec §8, §10, amended by D-A -- shard workers aggregate ONLY; the metadata backfill
they were also going to do is written at promote time instead.

Reports seconds per image and per shard for K in {1,4,16,64}, so P5 can size K from
a wall-clock target rather than a guess. It does not submit jobs; it times the
per-task body.

Usage:
    uv run python .../spikes/s2_shard_sizing.py <output_dir>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path


def _tables(output_dir: Path) -> list[Path]:
    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH, results_dir

    return [
        store / MEASUREMENT_TABLE_RELATIVE_PATH
        for store in sorted(results_dir(output_dir).glob("*/zarr/*.ome.zarr"))
        if (store / MEASUREMENT_TABLE_RELATIVE_PATH).is_file()
    ]


def _shard_body(tables: list[Path]) -> int:
    """Exactly what one array task's per-image loop will do after D-A."""
    import polars as pl

    rows = 0
    for table in tables:
        rows += pl.read_parquet(table).height
    return rows


def main() -> int:
    output_dir = Path(sys.argv[1]).resolve()
    tables = _tables(output_dir)
    print(f"n_tables={len(tables)}")

    for k in (1, 4, 16, 64):
        if k > len(tables):
            continue
        shard = tables[: max(len(tables) // k, 1)]
        t0 = time.perf_counter()
        rows = _shard_body(shard)
        elapsed = time.perf_counter() - t0
        print(
            f"K={k:3d} images={len(shard):4d} rows={rows:7d} "
            f"seconds={elapsed:7.3f} per_image={elapsed / len(shard):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Record the S-2 verdict**

Write the measured `seconds_per_image` and the resulting formula, instantiated:

```
K = clamp(ceil(N * seconds_per_image / 900), 1, min(MaxArraySize, MaxSubmitJobs) - 1)
```

The `- 1` is the reserved `TASK_FINALIZE` trigger entry — the project `CLAUDE.md` requires
every trigger entry to be counted when sizing chunks against `MaxArraySize`.

---

## S-3 — Merge cost (gates `TASK_FINALIZE`'s memory shape)

**Measures:** peak RSS and wall-clock merging `K` shard parquets versus a single-task
concat at `N ≈ 6000`, plus the streaming alternative.

- [ ] **Step 7: Write and run `s3_merge_cost.py`**

```python
"""S-3: can TASK_FINALIZE hold the shard merge in memory?

Spec §8, §10. Compares (a) polars concat of K shard parquets, (b) a single-task
concat of all N embedded tables, and (c) a streaming scan_parquet -> sink_parquet,
on peak RSS and wall-clock. A projected peak RSS above the finalizer's --mem means
P5 needs the streaming merge.

Usage:
    uv run python .../spikes/s3_merge_cost.py <output_dir> <scratch_dir> [K]
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> int:
    import polars as pl

    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH, results_dir

    output_dir = Path(sys.argv[1]).resolve()
    scratch = Path(sys.argv[2]).resolve()
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    scratch.mkdir(parents=True, exist_ok=True)

    tables = [
        s / MEASUREMENT_TABLE_RELATIVE_PATH
        for s in sorted(results_dir(output_dir).glob("*/zarr/*.ome.zarr"))
    ]
    tables = [t for t in tables if t.is_file()]
    print(f"n_tables={len(tables)} K={k} rss_start_mb={_peak_rss_mb():.1f}")

    t0 = time.perf_counter()
    whole = pl.concat([pl.read_parquet(t) for t in tables], how="diagonal_relaxed")
    t_direct = time.perf_counter() - t0
    print(
        f"direct_concat seconds={t_direct:.3f} rows={whole.height} "
        f"cols={whole.width} peak_rss_mb={_peak_rss_mb():.1f}"
    )

    shards: list[Path] = []
    step = max(len(tables) // k, 1)
    t0 = time.perf_counter()
    for i in range(k):
        chunk = tables[i * step : (i + 1) * step] if i < k - 1 else tables[i * step :]
        if not chunk:
            continue
        shard = scratch / f"shard_{i:04d}.parquet"
        pl.concat(
            [pl.read_parquet(t) for t in chunk], how="diagonal_relaxed"
        ).write_parquet(shard)
        shards.append(shard)
    t_shard = time.perf_counter() - t0

    t0 = time.perf_counter()
    merged = pl.concat([pl.read_parquet(s) for s in shards], how="diagonal_relaxed")
    t_merge = time.perf_counter() - t0
    print(
        f"shard_write seconds={t_shard:.3f}  merge seconds={t_merge:.3f} "
        f"rows={merged.height} peak_rss_mb={_peak_rss_mb():.1f}"
    )

    t0 = time.perf_counter()
    pl.scan_parquet([str(s) for s in shards]).sink_parquet(scratch / "streamed.parquet")
    t_stream = time.perf_counter() - t0
    print(f"streaming_merge seconds={t_stream:.3f} peak_rss_mb={_peak_rss_mb():.1f}")

    assert merged.height == whole.height, (
        f"shard merge lost rows: {merged.height} != {whole.height}"
    )
    print("row counts agree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 8: Record the S-3 verdict**

Extrapolate measured peak RSS to `N = 6000`. Write one of:

- **`S-3 IN-MEMORY`** — projected peak RSS under 32 GB. `TASK_FINALIZE` uses `pl.concat`;
  P5 Task 4 sets `--mem` to 2 × projected.
- **`S-3 STREAMING`** — projected peak RSS exceeds 32 GB, or `streaming_merge` is within
  1.5 × the in-memory merge. P5 Task 4 uses `pl.scan_parquet(...).sink_parquet(...)`.

---

## S-4 — Metadata projection locality (gates P4's promote-time write)

**Measures, reduced by D-A:** that the metadata rows one image's store should carry —
computed from that image's own measurements and `metadata.csv`, with no global frame —
equal the rows a global join would attribute to it, under **duplicate-key fan-out**,
**metadata-only rows**, and **partial keys**.

Under D-A this is asked at promote time, where the worker already holds the image's
measurements. That makes locality nearly structural — but "nearly" is why the spike still
runs: duplicate-key fan-out and metadata-only rows are exactly where a local projection
quietly diverges from a global one, and `prepare_embedded_measurement_table`
(`_embedded_measurement_tables.py:42`) already warns about both.

- [ ] **Step 9: Write `s4_metadata_projection_locality.py`**

```python
"""S-4: is the per-image metadata projection local?

Spec §8, §10, reduced by D-A. Asserts an EQUIVALENCE, not a duration: the metadata
rows a local projection selects for one image must equal the rows a global join
would attribute to that image, under fan-out, metadata-only rows, and partial keys.

If this fails, P4's promote-time write is wrong and the metadata table has to be
derived after the merge -- which under D-A means it cannot be written at promote
time at all, and D-A has to be revisited with the user.

Usage:
    uv run python .../spikes/s4_metadata_projection_locality.py <output_dir> <metadata.csv>
"""

from __future__ import annotations

import sys
from pathlib import Path


def _local_projection(table, metadata):
    """What the promote-time writer can compute from one image alone."""
    common = [c for c in metadata.columns if c in table.columns]
    if not common:
        return None
    keys = table.select(common).unique()
    return metadata.join(keys, on=common, how="semi").sort(common)


def _global_attribution(all_tables, metadata, index):
    """What a post-merge finalizer would attribute to image *index*."""
    import polars as pl

    table = all_tables[index]
    common = [c for c in metadata.columns if c in table.columns]
    if not common:
        return None
    merged = pl.concat(all_tables, how="diagonal_relaxed")
    joined = metadata.join(merged.select(common).unique(), on=common, how="semi")
    keys = table.select(common).unique()
    return joined.join(keys, on=common, how="semi").sort(common)


def main() -> int:
    import polars as pl

    from phenotypic.sdk_ import MEASUREMENT_TABLE_RELATIVE_PATH, results_dir

    output_dir = Path(sys.argv[1]).resolve()
    metadata_csv = Path(sys.argv[2]).resolve()

    tables = [
        pl.read_parquet(s / MEASUREMENT_TABLE_RELATIVE_PATH)
        for s in sorted(results_dir(output_dir).glob("*/zarr/*.ome.zarr"))
        if (s / MEASUREMENT_TABLE_RELATIVE_PATH).is_file()
    ]
    base = pl.read_csv(metadata_csv)
    print(f"n_tables={len(tables)} metadata_rows={base.height}")

    variants = {
        "clean": base,
        "fanout": pl.concat([base, base.head(1), base.head(1)]),
        "metadata_only": pl.concat(
            [
                base,
                base.head(1).with_columns(
                    pl.lit("__ABSENT__").alias(base.columns[0])
                ),
            ]
        ),
        "partial_keys": base.drop(base.columns[-1]) if base.width > 1 else base,
    }

    failures = 0
    for name, metadata in variants.items():
        for i in range(min(len(tables), 25)):
            local = _local_projection(tables[i], metadata)
            glob = _global_attribution(tables, metadata, i)
            if local is None and glob is None:
                continue
            if local is None or glob is None or not local.equals(glob):
                failures += 1
                print(f"MISMATCH variant={name} image={i}")
                break
        else:
            print(f"variant={name}: local == global for every image checked")

    print()
    print("S-4 PASS" if failures == 0 else f"S-4 FAIL ({failures} variants)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 10: Record the S-4 verdict**

- **`S-4 PASS`** — P4 Task 5 writes `pht-metadata.parquet` at promote time as planned.
- **`S-4 FAIL`** — **stop and report to the user.** A local projection that diverges from a
  global one means D-A's promote-time write cannot be correct, and the decision needs
  revisiting. Name the failing variant and one concrete diverging row.

---

## Running and recording

- [ ] **Step 11: Run S-2, S-3 and S-4 as one Slurm job**

`spikes/run_spikes.sbatch` — fill in and submit via the **`slurm-job`** skill. The
constraints from the user's global `CLAUDE.md` that bind here:

- `-p short` (max `2:00:00`), default account — **no `--account=` flag at all.** An empty
  value is an invalid account and `sbatch` rejects the whole submission.
- **Always set `--mem` and `--time`.** `DefMemPerCPU` is 1 GB/CPU and the default is the
  usual cause of a silent OOM kill. S-3 needs at least 64 GB.
- `--output` must be on shared storage (`/bigdata` or `/rhome`), **never**
  `/scratch/<user>/<jobid>` — that is node-local and per-job; a job landing on another node
  fails with `ExitCode 0:53` and no log file at all.
- `sbatch --parsable` prints the error and returns an **empty** id on rejection. Verify the
  captured id matches `^[0-9]+$` and surface the raw output on failure.
- Submission ≠ start. Check `scontrol show job <id> | grep -E 'StartTime|Reason'`.

- [ ] **Step 12: Write `RESULTS.md`**

For each of S-2…S-5: the fixture tree and its size, the raw numbers, the verdict line, and
the decision that verdict licenses. Then report S-5 to the user before starting P1.

- [ ] **Step 13: Commit**

```bash
git add docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/spikes/
git commit -m "spike(state): measure cache cold start, shard sizing, merge cost, projection locality

Spec §10 as amended by D-A and D-B. S-1 (hardlink re-promote) is cut: D-A writes
per-store metadata at promote time, so there is no re-promote to measure."
```
# Phase 1 — `sdk_/_run_state.py`: the one reader

**Depends on:** P0 (S-5 verdict). **Blocks:** P2–P7.

**Spec:** §4 (authority model), §5.2 (function surface), §9 (`RunState`), §9.1 (the
verification cache), §13 (error handling) — as amended by
[D-A, D-B and D-C](OPEN-QUESTIONS.md).

**What this phase does NOT do:** it moves **no consumers**. `_output_consistency.py`,
`RunRegistry`, the SLURM observer and `_snapshot_status.py` all keep working exactly as
they do today. Nothing is deleted. This phase adds a module and its tests, and nothing
else calls it yet. That is deliberate — it is the only phase whose correctness can be
established in isolation, and P6 depends on it being right.

**Read [`OPEN-QUESTIONS.md`](OPEN-QUESTIONS.md) before starting.** D-B decides the cache's
shape, and Q2/Q3/Q4 define the verdict precedence, the `ImageState` type, and the layering
rule this phase implements.

---

## File Structure

| File | Responsibility |
|---|---|
| **Create** `src/phenotypic/sdk_/_run_state.py` | The four public readers and the four state types. Imports nothing from `phenotypic._cli`. ~400 lines. |
| **Create** `src/phenotypic/sdk_/_verification_cache.py` | The bounded, in-process, identity-fenced verification cache and its currency rule. Separate file because INV-CACHE is tested against it directly and it must be trivially auditable. ~120 lines. |
| **Modify** `src/phenotypic/sdk_/_io_constants.py` | Add `DIR_IMAGE_RECORDS` and `image_record_path()`. |
| **Modify** `src/phenotypic/sdk_/__init__.py` | Export `RunIdentity`, `ImageState`, `RunState`, `RunDiagnostics`, `resolve_run_state`, `run_identity`, `assert_identity_current`, `clear_verification_cache`. **Not** `mint_run_identity` — that is a writer and lives CLI-side (P2). |
| **Create** `tests/unit/sdk_/test_run_state.py` | Verdict matrix, depth behaviour, advisories, INV-DEGRADE. |
| **Create** `tests/unit/sdk_/test_verification_cache.py` | INV-CACHE mutation suite. **The highest-value test in the change** (spec §14). |
| **Create** `tests/unit/sdk_/test_run_state_layering.py` | INV-LAYER. |

**Why two modules and not one:** the cache is the only part of this phase that can produce
a *wrong* answer rather than a slow one. Keeping it in its own file with its own test
module means a reviewer can read all of it at once, and means INV-CACHE's mutation tests
target a surface small enough to be exhaustive.

**No `verification_cache.json`, and no `VERIFICATION_CACHE_JSON` constant** — unless S-5
returned `ON-DISK TIER NEEDED`. See Task 3 Step 8.

---

## Interfaces

**Produces** (P2–P7 consume these exact signatures):

```python
# phenotypic.sdk_._run_state

@dataclass(frozen=True)
class RunIdentity:
    processing_generation: str      # content-derived from P2 onward
    restart_epoch: int
    scheduler_epoch: str | None
    owner_generation: str | None
    inventory_digest: str
    scientific_config_digest: str
    finalization_input_digest: str
    def digest(self) -> str: ...

@dataclass(frozen=True)
class ImageState:
    work_id: str
    dataset: str
    image_stem: str
    stages: Mapping[str, Mapping[str, object]]   # open map; §6.1 minus `backfilled` (D-A)
    verified: bool
    reason: str | None

@dataclass(frozen=True)
class RunDiagnostics:
    accepted: int
    verified: int
    failed: int
    manifest_completed: int | None
    manifest_total: int | None
    event_log_present: bool

@dataclass(frozen=True)
class RunState:
    completion: Literal["complete", "incomplete", "failed", "active"]
    identity: RunIdentity
    images: Mapping[str, ImageState]     # work_id -> ImageState
    advisories: tuple[str, ...]
    diagnostics: RunDiagnostics
    depth: Literal["shallow", "deep"]
    verified_at: datetime | None

def run_identity(output_dir: Path) -> RunIdentity | None: ...
def assert_identity_current(output_dir: Path, identity: RunIdentity) -> None: ...
def resolve_run_state(
    output_dir: Path, *, depth: Literal["shallow", "deep"] = "deep"
) -> RunState: ...
def finalization_input_object(output_dir: Path) -> dict[str, object]: ...
```

```python
# phenotypic.sdk_._verification_cache

@dataclass(frozen=True)
class CachedVerification:
    work_id: str
    verdict: bool
    stat_tuples: Mapping[str, tuple[int, int]]   # relative path -> (size, mtime_ns)

def cached_verification(
    output_dir: Path, identity_digest: str, work_id: str
) -> CachedVerification | None: ...
def remember_verification(
    output_dir: Path, identity_digest: str, entry: CachedVerification
) -> None: ...
def entry_is_still_current(output_dir: Path, entry: CachedVerification) -> bool: ...
def clear_verification_cache(output_dir: Path | None = None) -> None: ...
def verification_cache_size() -> int: ...   # test-only introspection
```

**Consumes:** nothing from this plan. From the existing tree:
`phenotypic.sdk_.resolve_processing_state_path`, `phenotypic_cache_dir`, `progress_dir`,
`image_completion_marker_path`, `run_completion_marker_path`,
`aggregate_publication_marker_path`, `STORE_ROOT_JSON`, `source_image_stem`.

---

## Task 1: Constants, package wiring, and INV-LAYER

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py`
- Create: `src/phenotypic/sdk_/_run_state.py` (stub)
- Create: `tests/unit/sdk_/test_run_state_layering.py`

- [ ] **Step 1: Write the failing layering test**

`tests/unit/sdk_/test_run_state_layering.py`:

```python
"""INV-LAYER: sdk_/_run_state.py never reaches into phenotypic._cli.

Spec §5.2 calls the read/write asymmetry "structural, not conventional": _run_state.py
exports only readers, so the GUI cannot reach a publish_* function. Structure that
nothing tests is convention with extra steps -- the GUI's 25 private phenotypic._cli
imports across 9 modules are what that looks like at scale (audit §7).

A LAZY import inside a function body is also a violation, not a loophole: it would
drag back load_processing_state's event-log replay, which spec §4.2 deletes. See
OPEN-QUESTIONS Q4. The AST walk catches both forms.
"""

from __future__ import annotations

import ast
from pathlib import Path

import phenotypic.sdk_._run_state as run_state
import phenotypic.sdk_._verification_cache as verification_cache

_MODULES = (Path(run_state.__file__), Path(verification_cache.__file__))


def test_neither_module_ever_names_the_cli_package():
    offenders: list[str] = []
    for source in _MODULES:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(("phenotypic._cli", "._cli")):
                    offenders.append(f"{source.name}:{node.lineno} from {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("phenotypic._cli"):
                        offenders.append(
                            f"{source.name}:{node.lineno} import {alias.name}"
                        )
    assert not offenders, (
        "INV-LAYER: the run-state readers must not import phenotypic._cli. "
        f"Found: {offenders}"
    )


def test_run_state_exports_no_writer():
    forbidden = ("publish", "write", "mint", "append", "save", "delete")
    exported = getattr(run_state, "__all__", None)
    assert exported is not None, "_run_state.py must declare __all__"
    bad = [
        name
        for name in exported
        if any(name.lower().startswith(prefix) for prefix in forbidden)
    ]
    assert not bad, f"_run_state.py exports writers: {bad}"
```

- [ ] **Step 2: Run it to see it fail**

Run: `uv run pytest tests/unit/sdk_/test_run_state_layering.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'phenotypic.sdk_._run_state'`.

- [ ] **Step 3: Add the constants**

In `src/phenotypic/sdk_/_io_constants.py`, beside `DIR_IMAGE_COMPLETE` (line 663):

```python
#: One record per image, replacing ``image_complete/``, ``stage2_done/`` and
#: ``stage3_complete/`` (spec §6.1). ``stage2_raw/`` stays a separate tree: it is
#: bulk replay data, not a record.
DIR_IMAGE_RECORDS: Final[str] = "images"
```

and beside `image_completion_marker_path` (line 1952):

```python
def image_record_path(output_dir: Path, dataset: str, image_stem: str) -> Path:
    """Return ``<output>/.phenotypic/progress/images/<dataset>/<stem>.json``."""
    return progress_dir(output_dir) / DIR_IMAGE_RECORDS / dataset / f"{image_stem}.json"
```

**Do not add `VERIFICATION_CACHE_JSON` or `verification_cache_path()`.** Under D-B the
cache is in process and has no path. Add them only under Task 3 Step 8, and only if S-5
said so.

- [ ] **Step 4: Create both module stubs with their `__all__`**

`src/phenotypic/sdk_/_run_state.py`:

```python
"""Read-only resolution of a run's completion state.

**Readers only.** Spec §5.2 makes the read/write asymmetry structural: every function
that *publishes* state stays in :mod:`phenotypic._cli`, so a GUI import of this module
cannot reach one. INV-LAYER (``tests/unit/sdk_/test_run_state_layering.py``) enforces
both halves -- no ``phenotypic._cli`` import, and no writer in ``__all__``.

This module reads ``processing_state.json`` as plain JSON and never replays the event
log. That is possible because spec §4.2 demotes the event log out of the evidence set
and deletes ``processing_state.datasets.{completed,failed,started}`` from the file:
what remains that a verdict depends on is ``config.work_ids`` and the digests, all
literal JSON fields. See OPEN-QUESTIONS Q4.
"""

from __future__ import annotations

__all__ = [
    "ImageState",
    "RunDiagnostics",
    "RunIdentity",
    "RunState",
    "assert_identity_current",
    "finalization_input_object",
    "resolve_run_state",
    "run_identity",
]
```

`src/phenotypic/sdk_/_verification_cache.py` — header only for now; the body lands in
Task 3.

- [ ] **Step 5: Run the layering test — it must pass**

Run: `uv run pytest tests/unit/sdk_/test_run_state_layering.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Prove the test can fail**

Temporarily add `from phenotypic._cli._cli_completion import valid_image_success` inside a
function body in `_run_state.py`, re-run, confirm
`test_neither_module_ever_names_the_cli_package` FAILS, then remove it. Repeat with a
module-scope import. **A test that has never been seen to fail is not evidence**, and the
lazy-import form is the one a future contributor will actually reach for.

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/sdk_/_io_constants.py src/phenotypic/sdk_/_run_state.py \
        src/phenotypic/sdk_/_verification_cache.py \
        tests/unit/sdk_/test_run_state_layering.py
git commit -m "feat(sdk): add the run-state module boundary and pin INV-LAYER

Spec §5.2. The modules are stubs; the test is the point -- 'structural, not
conventional' needs something that fails. Both the module-scope and the lazy
in-function import forms were confirmed to trip it."
```

---

## Task 2: The state types

**Files:**
- Modify: `src/phenotypic/sdk_/_run_state.py`
- Test: `tests/unit/sdk_/test_run_state.py`

Resolves [Q3](OPEN-QUESTIONS.md#q3-imagestate-is-used-but-never-defined).

- [ ] **Step 1: Write the failing test**

```python
import dataclasses


def test_the_demoted_sources_live_only_under_diagnostics():
    """Spec §9: a predicate reaching into state.diagnostics is visibly wrong.

    This does not stop someone writing `if state.diagnostics.verified ==
    state.diagnostics.accepted`, but it does pin WHERE the demoted evidence lives.
    manifest counts and the event log were evidence; §4.2 demoted them. If they
    reappear as top-level RunState fields, the demotion has been undone.
    """
    from phenotypic.sdk_ import RunDiagnostics, RunState

    top = {f.name for f in dataclasses.fields(RunState)}
    assert top == {
        "completion",
        "identity",
        "images",
        "advisories",
        "diagnostics",
        "depth",
        "verified_at",
    }
    diag = {f.name for f in dataclasses.fields(RunDiagnostics)}
    assert {"manifest_completed", "manifest_total", "event_log_present"} <= diag


def test_image_state_stages_carry_no_backfilled_key():
    """D-A: per-store metadata is written at promote time, so there is no
    backfill stage. `stages` stays an open map, so re-adding one later is
    additive -- but nothing in this phase may write or read that key."""
    from phenotypic.sdk_ import ImageState

    state = ImageState(
        work_id="w", dataset="d", image_stem="s",
        stages={"measured": {"at": "2026-09-03T00:00:00Z"}},
        verified=True, reason=None,
    )
    assert "backfilled" not in state.stages
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/sdk_/test_run_state.py -v`
Expected: FAIL — `ImportError: cannot import name 'RunState'`.

- [ ] **Step 3: Add the dataclasses**

Append to `_run_state.py` (imports at module top: `hashlib`, `json`, `dataclass`,
`datetime`, `Path`, `Literal`, `Mapping`):

```python
Completion = Literal["complete", "incomplete", "failed", "active"]
Depth = Literal["shallow", "deep"]


@dataclass(frozen=True)
class RunIdentity:
    """The six-token identity of one run configuration (spec §5.1).

    Three tokens are content-derived, so resume and fencing are emergent rather
    than bookkeeping: two invocations with the same inputs mint the same identity
    without either having read the other's state.
    """

    processing_generation: str
    restart_epoch: int
    scheduler_epoch: str | None
    owner_generation: str | None
    inventory_digest: str
    scientific_config_digest: str
    finalization_input_digest: str

    def digest(self) -> str:
        """Return a stable digest of the fencing-relevant tokens.

        ``scheduler_epoch`` and ``owner_generation`` are excluded: they are
        liveness facts, not configuration, and folding them in would discard the
        verification cache every time a job is submitted against unchanged work.
        """
        payload = {
            "processing_generation": self.processing_generation,
            "restart_epoch": self.restart_epoch,
            "inventory_digest": self.inventory_digest,
            "scientific_config_digest": self.scientific_config_digest,
            "finalization_input_digest": self.finalization_input_digest,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class ImageState:
    """One image's stages and whether its declared artifacts still match disk.

    ``stages`` is the open map from spec §6.1 -- ``stage1``/``stage2``/``stage3``/
    ``measured`` today, more later. Nothing here enumerates its keys; a caller
    asking "did stage 3 run?" reads ``"stage3" in state.stages``, which is what
    makes a future stage additive rather than a schema break.

    Under D-A there is no ``backfilled`` stage: per-store metadata is written in
    the store's original promote, so there is nothing to record having happened
    afterwards.
    """

    work_id: str
    dataset: str
    image_stem: str
    stages: Mapping[str, Mapping[str, object]]
    verified: bool
    reason: str | None = None


@dataclass(frozen=True)
class RunDiagnostics:
    """Counts. **Nothing branches on these** (spec §4.2, §9).

    Every field here was evidence once. ``manifest.json``'s counts and the event
    log's replay are kept because they are useful when a run looks wrong, and
    demoted because cross-checking derived counts against each other is what
    produced ``contradictory`` -- a state the user could not act on.
    """

    accepted: int
    verified: int
    failed: int
    manifest_completed: int | None = None
    manifest_total: int | None = None
    event_log_present: bool = False


@dataclass(frozen=True)
class RunState:
    """The single answer to "is this run done?" (spec §4.3, §9)."""

    completion: Completion
    identity: RunIdentity
    images: Mapping[str, ImageState]
    advisories: tuple[str, ...]
    diagnostics: RunDiagnostics
    depth: Depth
    verified_at: datetime | None = None
```

- [ ] **Step 4: Export from `sdk_/__init__.py`**

Add to the import block and to `__all__`, in alphabetical position: `ImageState`,
`RunDiagnostics`, `RunIdentity`, `RunState`, `assert_identity_current`,
`clear_verification_cache`, `finalization_input_object`, `resolve_run_state`,
`run_identity`.

- [ ] **Step 5: Run the test.** Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/sdk_/_run_state.py src/phenotypic/sdk_/__init__.py \
        tests/unit/sdk_/test_run_state.py
git commit -m "feat(sdk): define RunIdentity, ImageState, RunDiagnostics, RunState

Spec §9. ImageState is defined here because the spec uses it and never declares it
(OPEN-QUESTIONS Q3). stages carries no `backfilled` key (D-A)."
```

---

## Task 3: The in-process verification cache and INV-CACHE

**This is the highest-value task in the phase.** Spec §14 names its mutation tests as the
highest-value test in the whole change.

**Files:**
- Modify: `src/phenotypic/sdk_/_verification_cache.py`
- Test: `tests/unit/sdk_/test_verification_cache.py`

**Shape, per D-B:** a **bounded** module-level LRU keyed on
`(resolved output_dir, identity_digest, work_id)`. Bounded is not optional — audit §5,
S22 and S23 are all findings about unbounded module globals in this codebase
(`LocalRunner._instances`, `_terminal_job_cache`, `_LAST_DUMPED`), and shipping a fourth
while deleting the machinery that made the first three necessary would be indefensible.

- [ ] **Step 1: Write the INV-CACHE mutation suite first**

`tests/unit/sdk_/test_verification_cache.py`:

```python
"""INV-CACHE: the cache can only cause re-verification, never a wrong `complete`.

Spec §9.1 states the invariant and §14 calls these the highest-value tests in the
change. The current design's whole point is that it never trusts a cache, so the
correctness argument for introducing one has to be executable.

Each test corrupts the cache a different way and asserts the verdict never IMPROVES.
A cache that degrades to today's behaviour is correct; a cache that turns an
incomplete run into a complete one is the bug this file exists to prevent shipping.

D-B moved the cache in-process, so the "forge the file" cases here forge the dict.
The invariant is about what a cache may CAUSE, not where it lives, so it binds
identically. If S-5 added an on-disk tier, Step 8 adds the JSON-corruption cases.
"""

from __future__ import annotations

import pytest

from phenotypic.sdk_ import clear_verification_cache, resolve_run_state


@pytest.fixture(autouse=True)
def _isolate_cache():
    """A module-level cache is shared state; a leaked entry makes the next test lie."""
    clear_verification_cache()
    yield
    clear_verification_cache()


@pytest.fixture
def complete_run(tmp_path):
    from tests._output_layout import build_complete_run

    return build_complete_run(tmp_path)


@pytest.fixture
def incomplete_run(tmp_path):
    from tests._output_layout import build_incomplete_run

    return build_incomplete_run(tmp_path)


def test_a_forged_entry_cannot_manufacture_complete(incomplete_run):
    """The adversarial case: every entry claims verdict=True."""
    from phenotypic.sdk_ import run_identity
    from phenotypic.sdk_._verification_cache import (
        CachedVerification,
        remember_verification,
    )

    baseline = resolve_run_state(incomplete_run, depth="deep").completion
    assert baseline != "complete"

    identity = run_identity(incomplete_run)
    for work_id in resolve_run_state(incomplete_run, depth="deep").images:
        remember_verification(
            incomplete_run,
            identity.digest(),
            CachedVerification(work_id=work_id, verdict=True, stat_tuples={}),
        )

    after = resolve_run_state(incomplete_run, depth="shallow").completion
    assert after == baseline, (
        "a forged cache changed the verdict; a positive verdict must never come "
        "from a cache entry alone -- INV-CACHE"
    )


def test_a_stale_identity_never_matches(complete_run):
    from phenotypic.sdk_._verification_cache import cached_verification

    state = resolve_run_state(complete_run, depth="deep")
    work_id = next(iter(state.images))
    assert cached_verification(complete_run, state.identity.digest(), work_id)
    assert cached_verification(complete_run, "0" * 64, work_id) is None, (
        "an entry minted under a different identity was reused"
    )


def test_a_tampered_artifact_falls_through_even_with_a_warm_cache(complete_run):
    """The stat tuple is the currency check; content still decides."""
    resolve_run_state(complete_run, depth="deep")
    overlay = next(complete_run.rglob("overlays/**/*.png"), None)
    if overlay is None:
        pytest.skip("fixture has no overlay artifact")
    overlay.write_bytes(overlay.read_bytes() + b"tamper")

    assert resolve_run_state(complete_run, depth="shallow").completion != "complete"


def test_ctime_is_not_part_of_the_currency_check(complete_run):
    """Audit S3 / spec §9.1: ctime_ns moves on chmod, chown, hardlink and rsync -a,
    all routine on GPFS. size + mtime_ns already covers every write the publication
    contract makes, so a chmod must invalidate nothing."""
    warm = resolve_run_state(complete_run, depth="deep")
    for path in complete_run.rglob("*.png"):
        path.chmod(0o644)
    state = resolve_run_state(complete_run, depth="shallow")
    assert state.completion == warm.completion
    assert state.depth == "shallow", (
        "a chmod invalidated the cache -- ctime_ns has leaked into the currency "
        "check that audit S3 removed"
    )


def test_the_cache_is_bounded(complete_run):
    """Audit §5, S22, S23: this codebase's three known unbounded module globals are
    all findings. A fourth, added by the change that deletes the machinery which
    made them necessary, would not survive review."""
    from phenotypic.sdk_._verification_cache import (
        CachedVerification,
        _MAX_ENTRIES,
        remember_verification,
        verification_cache_size,
    )

    for i in range(_MAX_ENTRIES + 500):
        remember_verification(
            complete_run,
            "d" * 64,
            CachedVerification(work_id=f"w{i}", verdict=False, stat_tuples={}),
        )
    assert verification_cache_size() <= _MAX_ENTRIES


def test_eviction_is_lru_and_never_changes_an_answer(complete_run):
    """An evicted entry must produce a re-verification, not a different verdict."""
    from phenotypic.sdk_._verification_cache import (
        CachedVerification,
        _MAX_ENTRIES,
        remember_verification,
    )

    warm = resolve_run_state(complete_run, depth="deep")
    for i in range(_MAX_ENTRIES + 10):
        remember_verification(
            complete_run,
            "d" * 64,
            CachedVerification(work_id=f"filler{i}", verdict=True, stat_tuples={}),
        )
    evicted = resolve_run_state(complete_run, depth="shallow")
    assert evicted.completion == warm.completion
    assert evicted.depth == "deep", "eviction must escalate, not silently reuse"


def test_clear_scoped_to_one_output_does_not_clear_another(tmp_path):
    from tests._output_layout import build_complete_run

    a = build_complete_run(tmp_path / "a")
    b = build_complete_run(tmp_path / "b")
    resolve_run_state(a, depth="deep")
    resolve_run_state(b, depth="deep")
    clear_verification_cache(a)
    assert resolve_run_state(a, depth="shallow").depth == "deep"
    assert resolve_run_state(b, depth="shallow").depth == "shallow"
```

- [ ] **Step 2: Add the two fixture builders**

`tests/_output_layout.py` already holds `write_master` / `write_measurements_mirror` (used
by `tests/e2e/gui/test_heatmap_tab.py`). Add beside them:

```python
def build_complete_run(tmp_path: Path) -> Path:
    """Return an output tree whose deep verdict is `complete`.

    Deliberately minimal: two images in one dataset, each with a promoted store, an
    embedded measurement table and an overlay; a success marker for each; an
    aggregate proof; a run proof. Anything more makes a failing test hard to read.

    Built by calling the REAL publishers, never by hand-writing JSON: a fixture that
    hand-writes the format under test keeps passing after the format changes, which
    is the failure mode this whole plan is about. P3 swaps `publish_image_success`
    for the record writer and this function does not change.
    """
    from phenotypic._cli._cli_completion import (
        publish_aggregate_snapshot,
        publish_image_success,
    )

    output = tmp_path / "run"
    for stem in ("a", "b"):
        store = _promote_minimal_store(output, dataset="plate", stem=stem)
        overlay = _write_overlay(output, dataset="plate", stem=stem)
        publish_image_success(
            output,
            work_id=f"work-{stem}",
            dataset="plate",
            relative_image_path=f"{stem}.tif",
            image_stem=stem,
            mode="full",
            attempt_id=f"attempt-{stem}",
            lifecycle_epoch="local",       # `scheduler_epoch` from P2 Task 4 onward
            artifacts={"store": store, "overlay": overlay},
        )
    _write_processing_state(output, work_ids={"plate": {"a.tif": "work-a", "b.tif": "work-b"}})
    _write_master_and_mirror(output)       # tests._output_layout helpers, already present
    publish_aggregate_snapshot(output)
    _publish_run_completion(output)
    return output


def build_incomplete_run(tmp_path: Path) -> Path:
    """The same tree with the second image's success marker removed.

    Removing the MARKER rather than the artifacts is deliberate: it is the state a
    run killed between promoting a store and publishing its proof actually leaves,
    and the one the verdict ladder has to call `incomplete` rather than `complete`.
    """
    output = build_complete_run(tmp_path)
    image_completion_marker_path(output, "plate", "b").unlink()
    return output
```

`_promote_minimal_store`, `_write_overlay`, `_write_processing_state` and
`_publish_run_completion` are small local helpers in the same module; `_write_master_and_mirror`
wraps the existing `write_master` / `write_measurements_mirror` already in
`tests/_output_layout.py`.

- [ ] **Step 3: Run the suite to verify it fails**

Run: `uv run pytest tests/unit/sdk_/test_verification_cache.py -v`
Expected: FAIL — `ImportError: cannot import name 'clear_verification_cache'`.

- [ ] **Step 4: Implement the cache**

`src/phenotypic/sdk_/_verification_cache.py`:

```python
"""The in-process verification cache.

Audit **S1** -- the finding spec §9.1 responds to -- proposed a *process-level* cache
keyed on the marker file's stat tuple. §9.1 escalated that to a file on disk. Decision
D-B (OPEN-QUESTIONS) took it back: every cadence the audit measured is a repeated call
inside ONE long-lived process (the observer's 2 s tick, the viewer's 5-10 s poll,
``OutputRoot.discover``'s double read, ``OutputMutationGuard``'s double read), and an
in-memory cache serves all of them without adding a tracked artifact to a design whose
purpose is removing them.

INVARIANT (INV-CACHE) -- **the cache can only cause re-verification, never a wrong
`complete`.** No function here returns a verdict to a caller that has not deep-verified.
``entry_is_still_current`` answers one question: *may a previously deep-verified result
stand?* The caller supplies the verdict from its own deep pass, and a ``True`` here
merely licenses skipping that pass next time. A stale, evicted or forged entry therefore
degrades to today's behaviour and never past it.

``ctime_ns`` is deliberately absent from the stat tuple (audit S3): it moves on
``chmod``, ownership change, hardlink and ``rsync -a``, all routine on a shared
filesystem, and ``size`` + ``mtime_ns`` already covers every write the publication
contract makes.

**Bounded on purpose.** ``LocalRunner._instances``, ``_terminal_job_cache`` and
``_LAST_DUMPED`` are three unbounded module globals this codebase already carries as
audit findings (§5, S22, S23). This one evicts.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

#: One entry is ~3 stat tuples. 200k entries covers ~65k images -- an order of
#: magnitude above the largest run in the audit (10,000) -- at a few tens of MB.
_MAX_ENTRIES = 200_000

_CACHE: "OrderedDict[tuple[str, str, str], CachedVerification]" = OrderedDict()


@dataclass(frozen=True)
class CachedVerification:
    work_id: str
    verdict: bool
    stat_tuples: Mapping[str, tuple[int, int]]
```

Four rules the implementation must obey, each enforced by one of the tests above:

1. `cached_verification` returns `None` unless the `identity_digest` in the key matches
   **exactly**. There is no partial trust: an identity change discards every entry for
   that output.
2. `entry_is_still_current` returns `False` for an empty `stat_tuples` map, a missing
   file, an `OSError`, or any changed `(size, mtime_ns)`. It never raises.
3. `remember_verification` evicts the least-recently-used key when the cache exceeds
   `_MAX_ENTRIES`; `cached_verification` moves a hit to the end.
4. `clear_verification_cache(output_dir=None)` clears **that output's** keys, or all of
   them when `output_dir` is `None`. P2 wires it to `clear_machine_state`.

- [ ] **Step 5: Run the suite.** Expected: PASS (7 passed).

- [ ] **Step 6: Prove each test can fail (spec §14; project test-integrity rule)**

Reintroduce one at a time and confirm the named test fails:

| Bug to reintroduce | Test that must fail |
|---|---|
| `cached_verification` ignores `identity_digest` | `test_a_stale_identity_never_matches` |
| `resolve_run_state` returns the verdict from `entry.verdict` without re-stat | `test_a_forged_entry_cannot_manufacture_complete` |
| add `st_ctime_ns` to the stat tuple | `test_ctime_is_not_part_of_the_currency_check` |
| drop the eviction branch from `remember_verification` | `test_the_cache_is_bounded` |
| `clear_verification_cache` ignores `output_dir` and clears everything | `test_clear_scoped_to_one_output_does_not_clear_another` |

Record the five confirmations in the commit body. **A mutation not demonstrated is a
mutation not tested.**

- [ ] **Step 7: `mypy` and `ruff` on the two new modules**

```bash
uv run mypy src/phenotypic/sdk_/_verification_cache.py
uv run ruff check --fix src/phenotypic/sdk_/_verification_cache.py \
  tests/unit/sdk_/test_verification_cache.py
```

- [ ] **Step 8: ONLY IF S-5 returned `ON-DISK TIER NEEDED`**

Skip this step entirely otherwise — and if you skip it, say so in the commit body so a
later reader can tell the tier was *measured away*, not forgotten.

If S-5 said the tier is needed:

1. Add `VERIFICATION_CACHE_JSON` and `verification_cache_path()` to `_io_constants.py`,
   with a docstring naming the measured cold-start number that justifies them.
2. Add `load_verification_cache` / `store_verification_cache` to
   `_verification_cache.py`, backing the in-process LRU. The file carries a top-level
   `identity_digest`; a mismatch discards **the whole file**, not the mismatched entries.
3. `store_verification_cache` wraps `atomic_write_json` in `try/except OSError` and
   swallows — spec §9.1's "best-effort … must never turn an unwritable output into an
   error".
4. Add these cases to the mutation suite, each asserting the verdict never improves:
   `truncated` (`"{"`), `null`, `wrong-type` (`"[]"`), `binary-garbage`, `deleted`, and
   an `unwritable cache directory` case that asserts `resolve_run_state` returns
   `depth="deep"` rather than raising.
5. Prove each of those can fail too.

- [ ] **Step 9: Commit**

```bash
git add src/phenotypic/sdk_/_verification_cache.py \
        tests/unit/sdk_/test_verification_cache.py tests/_output_layout.py
git commit -m "feat(sdk): add the in-process verification cache and pin INV-CACHE

Spec §9.1, §14, as amended by D-B: audit S1 asked for a process-level cache and
that is what this is. S-5 measured cold start at <N>s, so no on-disk tier ships.

Each of the five mutations below was reintroduced and the named test confirmed to
fail:
  identity ignored          -> test_a_stale_identity_never_matches
  verdict straight from cache -> test_a_forged_entry_cannot_manufacture_complete
  ctime in the stat tuple   -> test_ctime_is_not_part_of_the_currency_check
  eviction removed          -> test_the_cache_is_bounded
  clear() unscoped          -> test_clear_scoped_to_one_output_does_not_clear_another"
```

---

## Task 4: `run_identity` and `assert_identity_current`

**Files:**
- Modify: `src/phenotypic/sdk_/_run_state.py`
- Test: `tests/unit/sdk_/test_run_state.py`

In this phase `run_identity` reads the tokens that **already exist** in
`processing_state.json`: `processing_generation` (still a `uuid4().hex` until P2),
`pipeline_sha256`, `metadata_sha256`, `include_dataset_column`, `no_qc`, `work_ids`.
`restart_epoch` defaults to `0` when absent — P2 introduces the writer. This is what keeps
P1 independently landable: the module works on today's trees.

- [ ] **Step 1: Write the failing tests**

```python
def test_run_identity_is_none_for_a_tree_with_no_processing_state(tmp_path):
    from phenotypic.sdk_ import run_identity

    assert run_identity(tmp_path) is None


def test_run_identity_reads_todays_state_file(complete_run):
    """P1 lands before P2, so it must work on a uuid4 processing_generation and a
    state file with no restart_epoch field at all."""
    from phenotypic.sdk_ import run_identity

    identity = run_identity(complete_run)
    assert identity is not None
    assert identity.restart_epoch == 0
    assert len(identity.inventory_digest) == 64
    assert identity.finalization_input_digest


def test_assert_identity_current_names_the_field_that_changed(complete_run):
    """D6: a config change still hard-errors with the SPECIFIC mismatch. A generic
    'identity changed' would make the content-derived generation a worse diagnostic
    than the uuid it replaces."""
    import dataclasses

    import pytest

    from phenotypic.sdk_ import assert_identity_current, run_identity

    identity = run_identity(complete_run)
    stale = dataclasses.replace(identity, inventory_digest="0" * 64)
    with pytest.raises(RuntimeError, match="inventory_digest"):
        assert_identity_current(complete_run, stale)


def test_finalization_input_digest_is_a_versioned_object(complete_run):
    """Spec §5.5: adding a field is a schema_version bump handled by the reader, not
    a second tree migration."""
    from phenotypic.sdk_ import finalization_input_object

    obj = finalization_input_object(complete_run)
    assert obj["schema_version"] == 1
    assert set(obj) == {
        "schema_version",
        "metadata_sha256",
        "include_dataset_column",
        "no_qc",
    }


def test_scheduler_epoch_and_owner_generation_are_not_in_the_digest(complete_run):
    """They are liveness facts, not configuration. Folding them in would discard the
    cache every time a job is submitted against unchanged work."""
    import dataclasses

    from phenotypic.sdk_ import run_identity

    identity = run_identity(complete_run)
    moved = dataclasses.replace(
        identity, scheduler_epoch="other", owner_generation="other"
    )
    assert moved.digest() == identity.digest()
```

- [ ] **Step 2: Run to verify failure.** Expected: `ImportError` / `AttributeError`.

- [ ] **Step 3: Implement**

```python
def _read_state_config(output_dir: Path) -> dict[str, object] | None:
    """Return ``processing_state.json``'s ``config`` block, or ``None``.

    Plain JSON, no event-log replay -- see the module docstring and OPEN-QUESTIONS
    Q4. Every failure returns ``None`` rather than raising (INV-DEGRADE).
    """
    from ._io_constants import resolve_processing_state_path

    try:
        raw = json.loads(
            resolve_processing_state_path(output_dir).read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError):
        return None
    config = raw.get("config") if isinstance(raw, dict) else None
    return config if isinstance(config, dict) else None


def finalization_input_object(output_dir: Path) -> dict[str, object]:
    """Return the versioned finalization-input object (spec §5.5)."""
    config = _read_state_config(output_dir) or {}
    return {
        "schema_version": 1,
        "metadata_sha256": config.get("metadata_sha256"),
        "include_dataset_column": config.get("include_dataset_column"),
        "no_qc": config.get("no_qc", False),
    }
```

`run_identity` composes those plus `_canonical_digest(config.get("work_ids", {}))`.
`assert_identity_current` compares field by field and raises
`RuntimeError(f"{field} changed: expected {a!r}, found {b!r}")` on the **first** mismatch.

**`_canonical_digest` currently lives in two places** — `_cli_completion.py:861` and
`_cli_failure_tracker.py`. INV-LAYER forbids importing either. Add a private third copy in
`_run_state.py` with a comment naming the other two, plus:

```python
def test_canonical_digest_agrees_across_modules():
    """Three copies is two too many; P6 Task 7 collapses them into one sdk_ helper.
    Until then, this is what stops them drifting -- a digest that disagrees with
    itself would invalidate every proof written by the other half of the code."""
    from phenotypic._cli._cli_completion import _canonical_digest as cli_completion
    from phenotypic._cli._cli_failure_tracker import _canonical_digest as cli_failure
    from phenotypic.sdk_._run_state import _canonical_digest as sdk

    probe = {"b": [1, 2, {"c": None}], "a": "é"}
    assert sdk(probe) == cli_completion(probe) == cli_failure(probe)
```

That test lives in `tests/unit/sdk_/test_run_state.py`, **not** in the layering test file —
it deliberately imports `_cli`, which is fine for a test and forbidden for the module.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/_run_state.py tests/unit/sdk_/test_run_state.py
git commit -m "feat(sdk): add run_identity and assert_identity_current

Spec §5.2, §5.5. Reads processing_state.json as plain JSON with no event-log replay
-- the property INV-LAYER protects. Third _canonical_digest copy is pinned against
the two CLI ones until P6 collapses them."
```

---

## Task 5: `resolve_run_state` — the deep path

**Files:**
- Modify: `src/phenotypic/sdk_/_run_state.py`
- Test: `tests/unit/sdk_/test_run_state.py`

Until P3 lands, the deep path reads **today's** `image_complete/<ds>/<stem>.json` markers
and maps them into `ImageState` with a single-key `stages = {"measured": {...}}`. P3
replaces the reader, not the caller. **Say so in a comment**, or the next reader will think
the single-key `stages` is the design.

- [ ] **Step 1: Write the failing verdict-matrix test**

```python
@pytest.mark.parametrize(
    "mutate,expected",
    [
        pytest.param(lambda d: None, "complete", id="untouched"),
        pytest.param(_remove_one_image_marker, "incomplete", id="missing-marker"),
        pytest.param(_remove_run_proof, "incomplete", id="no-run-proof"),
        pytest.param(_add_terminal_failure, "failed", id="terminal-failure"),
        pytest.param(_mark_slurm_lifecycle_active, "active", id="live-worker"),
        pytest.param(_corrupt_run_proof, "incomplete", id="unreadable-proof"),
        pytest.param(_corrupt_processing_state, "incomplete", id="unreadable-state"),
    ],
)
def test_the_verdict_matrix(complete_run, mutate, expected):
    mutate(complete_run)
    assert resolve_run_state(complete_run, depth="deep").completion == expected


def test_a_live_worker_does_not_mask_a_valid_run_proof(complete_run):
    """Q2: `complete` outranks `active`.

    A run proof covers the CURRENT inventory, so a live worker at that point is
    either fenced by restart_epoch or is a new invocation that has already changed
    the inventory -- in which case rule 1 does not fire and this is not the case
    being decided.
    """
    _mark_slurm_lifecycle_active(complete_run)
    assert resolve_run_state(complete_run, depth="deep").completion == "complete"


def test_an_active_run_outranks_a_stale_terminal_failure(incomplete_run):
    """Q2 rule 2 over rule 3: a failure from a previous attempt must not mask an
    attempt currently retrying it."""
    _add_terminal_failure(incomplete_run)
    _mark_slurm_lifecycle_active(incomplete_run)
    assert resolve_run_state(incomplete_run, depth="deep").completion == "active"


def test_an_unconverted_h5_is_an_advisory_and_never_a_gate(complete_run):
    """Spec §4.3: half-migrated trees contribute an advisory -- informational, not a
    gate. Today they reach `contradictory` and flag the whole output read-only for a
    reason the user cannot act on."""
    hdf = complete_run / "results" / "plate" / "hdf"
    hdf.mkdir(parents=True, exist_ok=True)
    (hdf / "legacy.h5").write_bytes(b"\x89HDF\r\n\x1a\n")
    state = resolve_run_state(complete_run, depth="deep")
    assert state.completion == "complete"
    assert any("migrate" in advisory for advisory in state.advisories)


def test_a_store_built_against_older_metadata_is_an_advisory(complete_run):
    """D-A: stores keep the metadata snapshot they were built against, and each
    store's phenotypic.metadata.snapshot_sha256 records which one. When that differs
    from the run's current metadata_sha256, say so -- derived from what the store
    already carries, never tracked, and never a gate."""
    _rewrite_metadata_csv(complete_run, b"Metadata_Well,Metadata_Strain\nA1,new\n")
    state = resolve_run_state(complete_run, depth="deep")
    assert state.completion == "complete"
    assert any("metadata" in advisory for advisory in state.advisories)


def test_an_empty_directory_is_incomplete_and_never_raises(tmp_path):
    """INV-DEGRADE. An unmanaged directory is not an error -- the GUI points at
    arbitrary paths and must get an answer, not a traceback."""
    state = resolve_run_state(tmp_path, depth="deep")
    assert state.completion == "incomplete"
    assert state.images == {}
```

- [ ] **Step 2: Run to verify failure.** Expected: `AttributeError: resolve_run_state`.

- [ ] **Step 3: Implement the deep path**

```python
def resolve_run_state(output_dir: Path, *, depth: Depth = "deep") -> RunState:
    """Resolve one run's completion state (spec §4.3, §9).

    Verdict precedence is total and ordered (OPEN-QUESTIONS Q2):
    ``complete`` > ``active`` > ``failed`` > ``incomplete``. First match wins.

    ``depth="shallow"`` re-stats the in-process cache's recorded tuples and falls
    through to ``deep`` for any image that is absent from the cache, moved, minted
    under a different identity, or unreadable. It **never** yields a positive verdict
    from a cache entry alone -- INV-CACHE.

    Args:
        output_dir: Run output root. May be any directory, including one this
            package has never written to.
        depth: ``"deep"`` re-verifies every declared artifact's content.
            ``"shallow"`` re-stats. See spec §9's caller/depth table.

    Returns:
        A :class:`RunState`. **Never raises** for an unreadable or absent tree --
        every parse failure degrades toward ``incomplete`` (INV-DEGRADE).
    """
```

Body, in order — each step is one of the four verdicts and nothing else:

1. `identity = run_identity(output_dir)`; on `None`, return an `incomplete` `RunState`
   with advisory `"no processing state"` and empty `images`.
2. Build `images` by walking `config["work_ids"]` — **the accepted-inventory authority**,
   never a directory listing. A `work_id` with no marker is an unverified `ImageState`,
   not an absent one; that is what makes "which images are missing?" answerable.
3. `completion` by the Q2 ladder. Rule 1 asks two things and no more: is there a valid run
   proof, and does its `inventory_digest` equal the current one.
4. `advisories`, each derived and each non-gating:
   - `datasets_needing_migration()` — the existing shared predicate — for unconverted `.h5`
   - any store whose `phenotypic.metadata.snapshot_sha256` ≠ the run's current
     `metadata_sha256` (D-A). One attribute read per store on the deep path, from a value
     the store already carries.
5. `diagnostics` from `manifest.json` and the event log's presence — read, recorded,
   **never branched on**.

- [ ] **Step 4: Run the tests.** Expected: PASS (13 passed).

- [ ] **Step 5: Prove the precedence tests can fail**

Swap ladder rules 1 and 2; confirm `test_a_live_worker_does_not_mask_a_valid_run_proof`
fails. Swap 2 and 3; confirm `test_an_active_run_outranks_a_stale_terminal_failure` fails.
Make the metadata advisory a gate (return `incomplete`); confirm
`test_a_store_built_against_older_metadata_is_an_advisory` fails. Restore all three.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/sdk_/_run_state.py tests/unit/sdk_/test_run_state.py
git commit -m "feat(sdk): resolve_run_state, deep path, with the Q2 verdict ladder

Spec §4.3, §9. Replaces ~23 classification rules with four ordered questions and
deletes `contradictory` as a reachable state. The D-A metadata-divergence advisory
is derived from each store's own snapshot_sha256 and is never a gate; all three
precedence tests were confirmed to fail when the ladder is reordered."
```

---

## Task 6: `resolve_run_state` — the shallow path

**Files:**
- Modify: `src/phenotypic/sdk_/_run_state.py`
- Test: `tests/unit/sdk_/test_run_state.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_shallow_after_deep_does_not_re_hash_artifacts(complete_run, monkeypatch):
    """Spec §9.2: adding 10 images to 6,000 should cost 6,000 stats and 10 deep
    verifications, not 6,000 re-hashes. On a 10,000-image run on GPFS, one badge
    refresh is currently ~10^4 marker reads and 2-3 x 10^4 file hashes. Per tab.
    Every five seconds."""
    import hashlib

    resolve_run_state(complete_run, depth="deep")   # warm

    calls = {"n": 0}
    real = hashlib.sha256

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(hashlib, "sha256", counting)
    state = resolve_run_state(complete_run, depth="shallow")
    assert state.completion == "complete"
    assert state.depth == "shallow"
    # Identity digests still hash a few small payloads; artifact CONTENT must not.
    assert calls["n"] <= 8, (
        f"shallow re-hashed artifacts ({calls['n']} sha256 calls); the whole point "
        "of §9.1 is that it re-stats instead"
    )


def test_a_new_image_escalates_the_whole_resolution(complete_run):
    resolve_run_state(complete_run, depth="deep")
    _add_third_image(complete_run)
    state = resolve_run_state(complete_run, depth="shallow")
    assert state.depth == "deep", "a cache miss must escalate"
    assert state.completion == "complete"


def test_shallow_with_a_cold_cache_equals_deep(complete_run):
    from phenotypic.sdk_ import clear_verification_cache

    clear_verification_cache()
    cold = resolve_run_state(complete_run, depth="shallow")
    deep = resolve_run_state(complete_run, depth="deep")
    assert cold.completion == deep.completion
    assert set(cold.images) == set(deep.images)
    assert cold.depth == "deep", "a cold shallow call is a deep call, and says so"
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

For each accepted `work_id`: if `cached_verification(output_dir, identity.digest(),
work_id)` returns an entry **and** `entry_is_still_current(output_dir, entry)`, reuse that
image's previous deep result. Otherwise mark the resolution escalated. If any image
escalated, re-run the deep verification **for the escalated images only**, remember the
results, and set `RunState.depth = "deep"`.

Setting `depth = "deep"` whenever *anything* escalated is deliberate: `depth` is what a
caller reads to know whether the answer is authoritative, and "mostly shallow" is not a
useful third value.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Run the Phase-1 test selection**

Via the **`run-phenotypic-test`** skill:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/sdk_/test_run_state.py \
  tests/unit/sdk_/test_verification_cache.py \
  tests/unit/sdk_/test_run_state_layering.py -p no:randomly -q
```

Record the count.

- [ ] **Step 6: Phase gate**

```bash
uv run mypy src/phenotypic
uv run ruff check --fix src/phenotypic/sdk_/_run_state.py \
  src/phenotypic/sdk_/_verification_cache.py src/phenotypic/sdk_/_io_constants.py \
  src/phenotypic/sdk_/__init__.py tests/unit/sdk_/ tests/_output_layout.py
```

Then the CLI + GUI regression selection, which must be **unchanged** — this phase moved no
consumers. Any new failure here means something was wired up that should not have been:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli tests/unit/gui -q
```

- [ ] **Step 7: Commit**

```bash
git add -A src/phenotypic/sdk_ tests/unit/sdk_
git commit -m "feat(sdk): shallow resolution via the in-process verification cache

Spec §9.1, §9.2. A steady-state badge refresh drops from ~10^4 artifact hashes to
~10^4 stats on the first tick and ~0 after, within one process. No consumer moved
in this phase."
```
# Phase 2 — Identity schema: fourteen tokens to six

**Depends on:** P1. **Blocks:** P3–P7.

**Spec:** §5 (identity schema), D3, D4, D5, D6, D7 — as amended by
[D-C](OPEN-QUESTIONS.md#d-c-scientific_config_digest-is-the-existing-digest-verbatim-was-q1)
and [O-1](OPEN-QUESTIONS.md#o-1-scheduler_epoch-may-be-five-names-collapsing-to-one-owner-not-five-tokens-to-one).

**Goal:** `processing_generation` stops being a `uuid4().hex` and becomes
`sha256(pipeline_sha256 ‖ scientific_config_digest ‖ restart_epoch)`; `restart_epoch`
becomes the one tracked counter the design admits to adding; `slurm_generation` and
`lifecycle_epoch` collapse into `scheduler_epoch` where a single writer already owns the
lifetime.

**Why content-derived matters (D3):** same inputs → same token, so resume and fencing
become **emergent** rather than bookkeeping. Two invocations with the same configuration
mint the same generation without either having read the other's state — which is what lets
a SLURM worker starting cold fence itself correctly against a run it has never seen.

---

## File Structure

| File | Responsibility |
|---|---|
| **Modify** `src/phenotypic/_cli/_cli_identity.py` *(new)* | `mint_run_identity(config, *, restart)`, `read_restart_epoch`, `bump_restart_epoch`. **CLI-side, because they write.** ~110 lines. |
| **Modify** `src/phenotypic/_cli/_cli_state_management.py:237` | `processing_generation` becomes content-derived; `restart_epoch` enters `config`. |
| **Modify** `src/phenotypic/sdk_/_io_constants.py:1081` | `clear_machine_state` **preserves** `restart_epoch.json`. |
| **Modify** `src/phenotypic/_cli/_cli_slurm_lifecycle.py:78` | `slurm_generation` → `scheduler_epoch`, one writer. |
| **Modify** `src/phenotypic/_cli/_cli_completion.py:163` | `publish_image_success` takes `scheduler_epoch`, not `lifecycle_epoch`. |
| **Modify** `docs/superpowers/specs/2026-09-03-cli-gui-state-tracking/design.md` §5.3–§5.4 | Correct the field list per D-C; add §5.3's redundancy footnote. |
| **Test** `tests/unit/cli/test_run_identity.py` *(new)* | Determinism, restart fencing, stale-worker. |

---

## Interfaces

**Produces:**

```python
# phenotypic._cli._cli_identity

def mint_run_identity(config: "ExecutionConfig", *, restart: bool) -> RunIdentity:
    """Mint the identity for a new or resumed invocation. **Writer.**"""

def read_restart_epoch(output_dir: Path) -> int:
    """Return the run's restart epoch, or 0 when absent. Never raises."""

def bump_restart_epoch(output_dir: Path) -> int:
    """Increment and persist the restart epoch. Returns the new value."""

def scientific_config_digest(config: "ExecutionConfig") -> str:
    """Return the per-image scientific configuration digest.

    D-C: this IS ``processing_configuration_digest`` -- the same function object,
    re-exported under the spec's name so §5.4's "one definition, two uses" is
    literally true rather than aspirationally true.
    """
```

**Consumes:** `phenotypic.sdk_.RunIdentity` (P1),
`phenotypic._cli._cli_failure_tracker.processing_configuration_digest`.

---

## Task 1: `restart_epoch` — the one tracked counter

**Files:**
- Create: `src/phenotypic/_cli/_cli_identity.py`
- Modify: `src/phenotypic/sdk_/_io_constants.py`
- Test: `tests/unit/cli/test_run_identity.py`

`clear_machine_state` (`_io_constants.py:1081`) currently deletes every child of
`.phenotypic/` except `terminal_failures.jsonl`. D4 requires `restart_epoch` to survive it
— otherwise a restart resets the counter and the stale-worker fence it exists for is gone.

- [ ] **Step 1: Write the failing test**

```python
def test_restart_epoch_survives_clear_machine_state(tmp_path):
    """D4: restart_epoch is THE one added tracked value, and it is worthless if a
    restart resets it -- the whole point is to distinguish 'deliberately fresh
    attempt' from 'same config again', which is exactly what a restart is."""
    from phenotypic._cli._cli_identity import bump_restart_epoch, read_restart_epoch
    from phenotypic.sdk_ import clear_machine_state

    (tmp_path / ".phenotypic").mkdir()
    assert read_restart_epoch(tmp_path) == 0
    assert bump_restart_epoch(tmp_path) == 1
    assert bump_restart_epoch(tmp_path) == 2

    clear_machine_state(tmp_path)
    assert read_restart_epoch(tmp_path) == 2, (
        "clear_machine_state destroyed the restart epoch; the fence it exists for "
        "cannot survive the operation it exists to fence"
    )


def test_reading_a_corrupt_restart_epoch_is_zero_not_an_error(tmp_path):
    """INV-DEGRADE. A restart must not be blocked by an unparseable counter."""
    from phenotypic._cli._cli_identity import read_restart_epoch

    cache = tmp_path / ".phenotypic"
    cache.mkdir()
    (cache / "restart_epoch.json").write_text("{not json", encoding="utf-8")
    assert read_restart_epoch(tmp_path) == 0
```

- [ ] **Step 2: Run to verify failure.** Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Add `RESTART_EPOCH_JSON: Final[str] = "restart_epoch.json"` and
`restart_epoch_path(output_dir)` to `_io_constants.py`, then extend
`clear_machine_state`'s preserve set:

```python
    _PRESERVED_ON_RESTART = frozenset({TERMINAL_FAILURES_JSONL, RESTART_EPOCH_JSON})
    ...
        for child in cache.iterdir():
            if child.name in _PRESERVED_ON_RESTART:
                continue
```

Update `clear_machine_state`'s docstring: it currently says it preserves "the append-only
`terminal_failures.jsonl` journal"; it now preserves that **and** the restart epoch, and
the docstring must say why — a counter that resets on the operation it fences is not a
fence.

`bump_restart_epoch` writes through `atomic_write_json`.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_identity.py src/phenotypic/sdk_/_io_constants.py \
        tests/unit/cli/test_run_identity.py
git commit -m "feat(cli): add restart_epoch, preserved across clear_machine_state

Spec §5.1 D4. One tracked integer, and the only one this design adds. It is
preserved by --restart on purpose: content-derived generations cannot tell a
deliberately fresh attempt from the same config again."
```

---

## Task 2: `scientific_config_digest` is the existing digest, and the spec is corrected

**Files:**
- Modify: `src/phenotypic/_cli/_cli_identity.py`
- Modify: `docs/superpowers/specs/2026-09-03-cli-gui-state-tracking/design.md`
- Test: `tests/unit/cli/test_run_identity.py`

Implements [D-C](OPEN-QUESTIONS.md#d-c-scientific_config_digest-is-the-existing-digest-verbatim-was-q1).

- [ ] **Step 1: Write the failing test**

```python
def test_scientific_config_digest_is_the_work_id_digest_itself(tmp_path):
    """D-C / spec §5.4: 'not a new digest ... reused verbatim'.

    §5.4's argument is that if the generation and work_id could disagree about what
    counts as scientific configuration, a change could invalidate per-image proofs
    without minting a new generation, or vice versa. Identity is the strongest form
    of agreement available, so this is an `is` check, not an equality check -- an
    equal-but-separate function would drift.
    """
    from phenotypic._cli._cli_failure_tracker import processing_configuration_digest
    from phenotypic._cli._cli_identity import scientific_config_digest

    assert scientific_config_digest is processing_configuration_digest
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement — one line, plus the docstring that explains it**

```python
# D-C: the spec calls this `scientific_config_digest`; the code has called it
# `processing_configuration_digest` since work_id was introduced. Re-export rather
# than wrap: §5.4's whole argument is that the generation and work_id must never
# disagree about what counts as scientific configuration, and identity is the only
# form of agreement that cannot drift.
#
# §5.4's prose ALSO claims include_dataset_column, overlay_alpha and save_overlays
# are excluded. They are not -- see _cli_failure_tracker.py:238. The prose is the
# wrong half (OPEN-QUESTIONS D-C); removing them from the per-image digest is a
# work_id change and belongs to its own spec with its own migration.
scientific_config_digest = processing_configuration_digest
```

- [ ] **Step 4: Correct the spec**

In `design.md` §5.4, replace the field list with the actual contents of
`processing_configuration_digest_from_values`, and add to §5.3, under the digest table:

> **Footnote (D-C).** `include_dataset_column` appears in both
> `scientific_config_digest` (via `work_id`) and `finalization_input_digest`. The two
> answer different questions and a field may be relevant to both, so "none is redundant"
> refers to the digests, not to their fields. Flipping `include_dataset_column` therefore
> still reprocesses every image; narrowing the per-image digest is a `work_id` change and
> deserves its own spec.

Mark the edit in the spec's own change log if it has one; otherwise state it in the commit
body. **Do not silently rewrite a spec section** — a reader comparing the plan to the spec
needs to see that the plan won an argument, not that the spec was always right.

- [ ] **Step 5: Run the test.** Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_cli/_cli_identity.py tests/unit/cli/test_run_identity.py \
        docs/superpowers/specs/2026-09-03-cli-gui-state-tracking/design.md
git commit -m "feat(cli): scientific_config_digest IS processing_configuration_digest

D-C. Also corrects design.md §5.4's field list, which claimed
include_dataset_column, overlay_alpha and save_overlays are excluded from work_id.
They are in it (_cli_failure_tracker.py:238), and §5.1 says work_id is unchanged --
the two could not both hold."
```

---

## Task 3: The content-derived `processing_generation`

**Files:**
- Modify: `src/phenotypic/_cli/_cli_identity.py`
- Modify: `src/phenotypic/_cli/_cli_state_management.py:237`
- Test: `tests/unit/cli/test_run_identity.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_the_same_config_mints_the_same_generation(tmp_path, execution_config):
    """D3: same inputs -> same token, so resume and fencing are emergent rather
    than bookkeeping. A SLURM worker starting cold can fence itself correctly
    against a run it has never read."""
    from phenotypic._cli._cli_identity import mint_run_identity

    a = mint_run_identity(execution_config, restart=False)
    b = mint_run_identity(execution_config, restart=False)
    assert a.processing_generation == b.processing_generation


def test_a_pipeline_edit_mints_a_new_generation(tmp_path, execution_config):
    from phenotypic._cli._cli_identity import mint_run_identity

    before = mint_run_identity(execution_config, restart=False)
    execution_config.pipeline_json.write_text(
        execution_config.pipeline_json.read_text() + "\n", encoding="utf-8"
    )
    after = mint_run_identity(execution_config, restart=False)
    assert after.processing_generation != before.processing_generation


def test_a_restart_mints_a_new_generation_for_identical_config(tmp_path, execution_config):
    """D4's reason for existing. Without restart_epoch the generation is a pure
    function of configuration, so a deliberately fresh attempt against unchanged
    config is indistinguishable from the run it replaces -- and a worker still
    holding the pre-restart generation would pass the fence."""
    from phenotypic._cli._cli_identity import mint_run_identity

    before = mint_run_identity(execution_config, restart=False)
    after = mint_run_identity(execution_config, restart=True)
    assert after.restart_epoch == before.restart_epoch + 1
    assert after.processing_generation != before.processing_generation


def test_a_new_image_does_NOT_mint_a_new_generation(tmp_path, execution_config):
    """D7: inventory_digest is deliberately OUT of the generation digest.

    Generation fences configuration; inventory_digest fences scope. Conflating them
    would make every new image under a rolling input look like a configuration
    change -- resetting live progress and fencing in-flight workers, which is the
    exact failure mode a 6,000-image rolling dataset produces daily."""
    from phenotypic._cli._cli_identity import mint_run_identity

    before = mint_run_identity(execution_config, restart=False)
    _add_image_to_input(execution_config)
    after = mint_run_identity(execution_config, restart=False)
    assert after.processing_generation == before.processing_generation
    assert after.inventory_digest != before.inventory_digest


def test_a_metadata_edit_does_NOT_mint_a_new_generation(tmp_path, execution_config):
    """§5.4/§7.4: a metadata edit changes finalization_input_digest only, so the
    next invocation re-runs finalize_run without touching a single image's
    measurement."""
    from phenotypic._cli._cli_identity import mint_run_identity

    before = mint_run_identity(execution_config, restart=False)
    execution_config.metadata_csv.write_text(
        "Metadata_Well,Metadata_Strain\nA1,new\n", encoding="utf-8"
    )
    after = mint_run_identity(execution_config, restart=False)
    assert after.processing_generation == before.processing_generation
    assert after.finalization_input_digest != before.finalization_input_digest
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

```python
def mint_run_identity(config: "ExecutionConfig", *, restart: bool) -> RunIdentity:
    """Mint the identity of a new or resumed invocation (spec §5.1, §5.4).

    ``processing_generation`` is ``sha256(pipeline_sha256 || scientific_config_digest
    || restart_epoch)``. **Writer** -- it can bump the restart epoch, which is why it
    lives in ``phenotypic._cli`` and not beside the readers in ``sdk_/_run_state.py``.

    ``inventory_digest`` is deliberately absent from the generation (D7): generation
    fences *configuration*, ``inventory_digest`` fences *scope*, and they change on
    different schedules. Folding them together makes every arrival under a rolling
    input look like a configuration change.

    Args:
        config: The invocation's execution configuration.
        restart: ``True`` for ``--restart``, which bumps and persists the epoch.

    Returns:
        A :class:`~phenotypic.sdk_.RunIdentity`.
    """
```

Then change `_cli_state_management.py:237` from `"processing_generation": uuid4().hex` to
the minted value, and add `"restart_epoch": identity.restart_epoch` to the same config
block. Both `create_initial_state` and every resume path must use the same mint.

- [ ] **Step 4: Run the tests.** Expected: PASS (5 passed).

- [ ] **Step 5: Regression — `--restart` still reuses surviving stores (D5)**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli -k 'restart or resume' -q
```

D5 is explicit: `--restart` keeps reusing surviving `results/` stores. The epoch fixes the
stale-worker hazard **without** turning `--restart` into `--overwrite`. If any of these
tests now show a restart reprocessing images it previously reused, the epoch has leaked
into `work_id` and the change is wrong.

- [ ] **Step 6: Commit**

```bash
git add -A src/phenotypic/_cli tests/unit/cli/test_run_identity.py
git commit -m "feat(cli): processing_generation becomes content-derived

Spec §5.1, §5.4, D3, D4, D7. sha256(pipeline || scientific_config || restart_epoch).
inventory_digest stays out (D7) so a rolling input's arrivals do not read as a
config change. --restart still reuses surviving stores (D5)."
```

---

## Task 4: The `scheduler_epoch` collapse — only where one writer owns the lifetime

**Files:**
- Modify: `src/phenotypic/_cli/_cli_slurm_lifecycle.py:78`
- Modify: `src/phenotypic/_cli/_cli_completion.py:163` (`publish_image_success`)
- Test: `tests/unit/cli/test_run_identity.py`

Implements [O-1](OPEN-QUESTIONS.md#o-1-scheduler_epoch-may-be-five-names-collapsing-to-one-owner-not-five-tokens-to-one).
**Read it before starting — this task is deliberately narrower than §5.1 asks for.**

§5.1 has `scheduler_epoch` absorb five tokens. Four subsystems write those five
(`_cli_slurm_lifecycle`, `_cli_staged_orchestration`, the recompile worker, the local
strategy) at four different times with four different lifetimes. Collapsing the *names*
without collapsing the *writers* gives one value with four owners — a coupling increase
dressed as a cardinality reduction.

**This task collapses only the pair that is already one value.** The audit found it
(§11.1): `_assert_worker_generation`'s `slurm_generation != attempt_id` check is "one value
passed twice, then asserted equal".

- [ ] **Step 1: Confirm the audit's claim against the code before acting on it**

```bash
grep -n '_assert_worker_generation' -A 25 src/phenotypic/_cli/*.py
```

Expected: the two compared values originate from the same source. **If they do not, stop
and report** — the collapse's justification is that finding, and a wrong finding makes this
task a behaviour change rather than a rename.

- [ ] **Step 2: Write the failing stale-worker test**

```python
def test_a_worker_holding_the_pre_restart_generation_is_fenced(tmp_path, execution_config):
    """Spec §14's stale-worker test. A worker that started before a --restart holds
    the old generation; its events must not be counted and its publications must be
    refused."""
    import pytest

    from phenotypic._cli._cli_completion import publish_image_success
    from phenotypic._cli._cli_identity import mint_run_identity

    stale = mint_run_identity(execution_config, restart=False)
    fresh = mint_run_identity(execution_config, restart=True)
    assert stale.processing_generation != fresh.processing_generation

    with pytest.raises(RuntimeError, match="stale"):
        publish_image_success(
            tmp_path,
            work_id="w",
            dataset="plate",
            relative_image_path="a.tif",
            image_stem="a",
            mode="full",
            attempt_id="attempt",
            scheduler_epoch=stale.processing_generation,
            artifacts={},
        )
```

- [ ] **Step 3: Run to verify failure.**

- [ ] **Step 4: Implement**

Rename `publish_image_success`'s `lifecycle_epoch` parameter to `scheduler_epoch` and
update all call sites. Keep the staged `epoch` and recompile `attempt_id` as **diagnostic**
fields written under the collapsed name but never compared — spec §5.1 already classifies
per-image `attempt_id` as "written, never branched on", so this is that rule applied
consistently rather than a new exception.

Delete `_assert_worker_generation`'s `slurm_generation != attempt_id` check (spec §11.1).

- [ ] **Step 5: Run the test and the SLURM lifecycle regression**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli -k 'lifecycle or slurm or staged' -q
```

- [ ] **Step 6: Phase gate**

```bash
uv run mypy src/phenotypic
uv run ruff check --fix src/phenotypic/_cli/_cli_identity.py \
  src/phenotypic/_cli/_cli_state_management.py src/phenotypic/_cli/_cli_slurm_lifecycle.py \
  src/phenotypic/_cli/_cli_completion.py src/phenotypic/sdk_/_io_constants.py \
  tests/unit/cli/test_run_identity.py
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli tests/unit/sdk_ -q
```

- [ ] **Step 7: Commit**

```bash
git add -A src/phenotypic/_cli tests/unit/cli
git commit -m "refactor(cli): collapse slurm_generation and lifecycle_epoch into scheduler_epoch

Spec §5.1, narrowed by OPEN-QUESTIONS O-1: only the pair the audit found to be one
value passed twice (§11.1) is collapsed. Staged epoch and recompile attempt_id are
written under the name as diagnostics and never compared -- four writers behind one
compared value would be a coupling increase, not a cardinality reduction."
```
# Phase 3 — One per-image record

**Depends on:** P1, P2. **Blocks:** P4–P7.

**Spec:** §6.1 (one record), §6.2 (store immutability) — as amended by
[D-A](OPEN-QUESTIONS.md#d-a-per-store-metadata-is-written-at-promote-time-not-backfilled)
and [O-2](OPEN-QUESTIONS.md#o-2-stages-is-an-open-map-with-no-name-validation).

**Goal:** `image_complete/`, `stage2_done/` and `stage3_complete/` — three parallel
`<ds>/<stem>.*` trees answering three sub-questions about the same image, spelled in three
different places — become one record at
`.phenotypic/progress/images/<dataset>/<stem>.json` with an open `stages` map. "Is this
image done?" becomes one JSON read instead of one read plus up to three `is_file()` probes
across three directory trees.

`stage2_raw/<ds>/<stem>.npy` **stays a separate file**. It is bulk replay data, not a
record, and the staged engine's Stage-3 replay reads it as an array.

### What D-A cuts from this phase

Spec §6.3's hardlink re-promote and §6.4's certified-rewrite protocol are **not built**.
Per-store metadata is written at promote time (P4 Task 5), so there is no post-proof store
mutation to certify. The pre-existing
`refresh_success_markers_after_metadata_migration` (`_cli_completion.py:305`) is
**untouched here** and stays scoped to `--mode migrate` in P7 — it serves one historical
case and keeps `RuntimeError` for an artifact that moved without a covering receipt, which
is INV-IMMUTABLE's exception and its only one.

`stages` therefore carries no `backfilled` key. The map stays open, so adding one later is
additive.

---

## File Structure

| File | Responsibility |
|---|---|
| **Create** `src/phenotypic/_cli/_cli_image_record.py` | `publish_image_record`, `read_image_record`, `record_stage`, `KNOWN_STAGES`. The single writer. ~220 lines. |
| **Modify** `src/phenotypic/_cli/_cli_completion.py` | `publish_image_success` / `valid_image_success` delegate to the record. |
| **Modify** `src/phenotypic/_cli/_cli_stage2_token.py` | `write_stage2_token` / `stage2_token_exists` / `delete_stage2_token` become `stages.stage2` operations. `stage2_raw` helpers unchanged. |
| **Modify** `src/phenotypic/_cli/_cli_staged_resume.py` | `stage3_completion_exists` / `write_stage3_completion_marker` / `remove_stage3_completion_marker` become `stages.stage3` operations. `classify_staged_image` reads one record. |
| **Modify** `src/phenotypic/sdk_/_run_state.py` | The deep path reads the record instead of the legacy marker. |
| **Delete** | `DIR_STAGE2_DONE` / `DIR_STAGE3_COMPLETE` path helpers, the inline `"stage3_complete"` literal at `_cli_staged_resume.py:141`, and `_STAGE2_DIR` at `_cli_stage2_token.py:42`. |
| **Test** `tests/unit/cli/test_image_record.py` *(new)* | Record schema, stage independence, O-2 advisory. |
| **Test** `tests/unit/cli/test_staged_resume_equivalence.py` *(new)* | The gate: resume decisions are unchanged. |

**The staged engine's resume logic is the risk in this phase**, not the record format.
`classify_staged_image` (`_cli_staged_resume.py:197`) decides, per image, whether to run
stage 1, 2 or 3, from four independent filesystem probes. Collapsing those into one read
must not change a single one of its decisions.

---

## Interfaces

**Produces:**

```python
# phenotypic._cli._cli_image_record

#: Stage names this build understands. `stages` stays an OPEN map (§6.1) -- an
#: unknown key is surfaced as a RunState advisory (O-2), never rejected.
KNOWN_STAGES: frozenset[str] = frozenset({"stage1", "stage2", "stage3", "measured"})

RECORD_VERSION: int = 1

def publish_image_record(
    output_dir: Path,
    *,
    work_id: str,
    dataset: str,
    image_stem: str,
    relative_image_path: str,
    mode: str,
    stages: Mapping[str, Mapping[str, object]],
    artifacts: Mapping[str, Path],
    attempt_id: str,
    scheduler_epoch: str,
    commit_guard: "CommitGuard | None" = None,
) -> Path: ...

def read_image_record(
    output_dir: Path, dataset: str, image_stem: str
) -> dict[str, object] | None: ...

def record_stage(
    output_dir: Path, dataset: str, image_stem: str, stage: str,
    payload: Mapping[str, object], *, commit_guard=None,
) -> Path: ...

def consume_stage(
    output_dir: Path, dataset: str, image_stem: str, stage: str
) -> bool: ...
```

**Consumes:** `phenotypic.sdk_.image_record_path` (P1),
`phenotypic._cli._cli_identity.scheduler_epoch` plumbing (P2).

---

## Task 1: The record writer and reader

**Files:**
- Create: `src/phenotypic/_cli/_cli_image_record.py`
- Test: `tests/unit/cli/test_image_record.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_the_record_is_one_file_carrying_every_stage(tmp_path):
    """Spec §6.1: 'Is this image done?' becomes ONE JSON read instead of one read
    plus up to three is_file() probes across three directory trees."""
    from phenotypic._cli._cli_image_record import publish_image_record, read_image_record
    from phenotypic.sdk_ import image_record_path

    store = tmp_path / "results" / "plate" / "zarr" / "a.ome.zarr"
    store.mkdir(parents=True)
    (store / "zarr.json").write_text("{}", encoding="utf-8")

    publish_image_record(
        tmp_path,
        work_id="w", dataset="plate", image_stem="a",
        relative_image_path="a.tif", mode="full",
        stages={
            "stage1": {"at": "2026-09-03T00:00:00Z"},
            "stage2": {"at": "2026-09-03T00:00:01Z", "objmap_shape": [1024, 1024]},
            "stage3": {"at": "2026-09-03T00:00:02Z"},
            "measured": {"at": "2026-09-03T00:00:03Z"},
        },
        artifacts={"store": store},
        attempt_id="attempt", scheduler_epoch="epoch",
    )

    assert image_record_path(tmp_path, "plate", "a").is_file()
    record = read_image_record(tmp_path, "plate", "a")
    assert set(record["stages"]) == {"stage1", "stage2", "stage3", "measured"}
    assert record["artifacts"]["store"]["kind"] == "store"


def test_stages_is_an_open_map(tmp_path):
    """§6.1: `stages` and `artifacts` are open maps -- that is what makes a future
    stage additive rather than a schema break."""
    from phenotypic._cli._cli_image_record import publish_image_record, read_image_record

    publish_image_record(
        tmp_path, work_id="w", dataset="plate", image_stem="a",
        relative_image_path="a.tif", mode="full",
        stages={"stage1": {"at": "t"}, "some_future_stage": {"at": "t"}},
        artifacts={}, attempt_id="x", scheduler_epoch="e",
    )
    assert "some_future_stage" in read_image_record(tmp_path, "plate", "a")["stages"]


def test_an_unknown_stage_becomes_an_advisory_not_a_failure(tmp_path):
    """O-2: an open map with no name validation means a typo like `stage_2` reads
    as 'stage 2 not done' and never errors. Surface it without closing the map."""
    from phenotypic._cli._cli_image_record import publish_image_record
    from phenotypic.sdk_ import resolve_run_state

    publish_image_record(
        tmp_path, work_id="w", dataset="plate", image_stem="a",
        relative_image_path="a.tif", mode="full",
        stages={"stage_2": {"at": "t"}},      # note the typo
        artifacts={}, attempt_id="x", scheduler_epoch="e",
    )
    state = resolve_run_state(tmp_path, depth="deep")
    assert any("stage_2" in advisory for advisory in state.advisories)


def test_recording_one_stage_leaves_the_others_untouched(tmp_path):
    """The three collapsed trees were independently writable and must stay so --
    Stage 2 and Stage 3 run in different jobs, on different nodes, minutes apart."""
    from phenotypic._cli._cli_image_record import read_image_record, record_stage

    record_stage(tmp_path, "plate", "a", "stage1", {"at": "t1"})
    record_stage(tmp_path, "plate", "a", "stage2", {"at": "t2"})
    record = read_image_record(tmp_path, "plate", "a")
    assert set(record["stages"]) == {"stage1", "stage2"}


def test_reading_a_corrupt_record_is_none_not_an_error(tmp_path):
    """INV-DEGRADE."""
    from phenotypic._cli._cli_image_record import read_image_record
    from phenotypic.sdk_ import image_record_path

    path = image_record_path(tmp_path, "plate", "a")
    path.parent.mkdir(parents=True)
    path.write_text("{truncated", encoding="utf-8")
    assert read_image_record(tmp_path, "plate", "a") is None
```

- [ ] **Step 2: Run to verify failure.** Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

The record, per §6.1 minus `backfilled`:

```json
{
  "version": 1,
  "work_id": "…", "dataset": "…", "image_stem": "…",
  "relative_image_path": "…", "mode": "full|process|measure",
  "stages": {
    "stage1":   {"at": "…"},
    "stage2":   {"at": "…", "objmap_shape": [1024, 1024], "detector_seconds": 1.23},
    "stage3":   {"at": "…"},
    "measured": {"at": "…"}
  },
  "artifacts": {
    "store":        {"kind": "store", "path": "…", "sha256": "<root zarr.json digest>"},
    "measurements": {"kind": "file",  "path": "…", "size": 12345, "sha256": "…"},
    "metadata":     {"kind": "file",  "path": "…", "size": 234,   "sha256": "…"},
    "overlay":      {"kind": "file",  "path": "…", "size": 67890, "sha256": "…"}
  },
  "attempt_id": "…", "scheduler_epoch": "…", "completed_at": "…"
}
```

`record_stage` is **read-modify-write under the existing `atomic_write_json` +
`pre_replace` revalidation**, exactly as `publish_image_success` does today
(`_cli_completion.py:163`). Two stages written concurrently for the same image is not a
real case — the staged engine runs them in different jobs, serialized by the stage-2 token
— but the read-modify-write must still not lose a key on a retry.

`consume_stage` replaces `delete_stage2_token`'s unlink: it removes one key from `stages`
and rewrites. **Consumption must be idempotent** — Stage 3 already tolerates a token that
another attempt consumed.

- [ ] **Step 4: Run the tests.** Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_image_record.py tests/unit/cli/test_image_record.py
git commit -m "feat(cli): one per-image record with an open stages map

Spec §6.1. No `backfilled` stage (D-A). An unrecognised stage name becomes a
RunState advisory rather than silently reading as not-done (O-2)."
```

---

## Task 2: Migrate `publish_image_success` and `valid_image_success` onto the record

**Files:**
- Modify: `src/phenotypic/_cli/_cli_completion.py:163`, `:255`
- Modify: `src/phenotypic/sdk_/_run_state.py`
- Test: `tests/unit/cli/test_image_record.py`, `tests/unit/sdk_/test_run_state.py`

- [ ] **Step 1: Write the failing test**

```python
def test_publish_image_success_writes_the_record_not_the_legacy_marker(tmp_path):
    from phenotypic.sdk_ import image_completion_marker_path, image_record_path

    _publish_a_successful_image(tmp_path, dataset="plate", stem="a")
    assert image_record_path(tmp_path, "plate", "a").is_file()
    assert not image_completion_marker_path(tmp_path, "plate", "a").exists(), (
        "the legacy image_complete/ marker is still being written; D1 is a clean "
        "break, not a dual write"
    )


def test_valid_image_success_still_rejects_a_tampered_artifact(tmp_path):
    """The artifact-digest contract is unchanged by the collapse. This is the
    property `_walk_current_success` has today and the one P6 will lean on."""
    from phenotypic._cli._cli_completion import valid_image_success

    store = _publish_a_successful_image(tmp_path, dataset="plate", stem="a")
    assert valid_image_success(tmp_path, dataset="plate", image_stem="a", work_id="w")
    (store / "zarr.json").write_text('{"tampered": true}', encoding="utf-8")
    assert not valid_image_success(
        tmp_path, dataset="plate", image_stem="a", work_id="w"
    )
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`publish_image_success` keeps its signature and its artifact-validation body — the marker
it writes is now the record, with `stages={"measured": {...}}` plus whatever the caller
passes. **D1 is a clean break: no dual write.** A tree with `image_complete/` and no
`images/` is a legacy tree, and P7's migrate converts it; every other mode refuses it.

`valid_image_success` reads the record. Its `SUCCESS_MARKER_VERSION` check becomes
`RECORD_VERSION`.

Update `sdk_/_run_state.py`'s deep path to read the record and populate `ImageState.stages`
from it — **and delete the P1 comment saying the single-key `stages` is temporary.**

- [ ] **Step 4: Run the tests plus the completion regression**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli tests/unit/sdk_ -q
```

- [ ] **Step 5: Commit**

```bash
git add -A src/phenotypic tests/unit
git commit -m "refactor(cli): publish_image_success writes the collapsed record

Spec §6.1, D1 -- clean break, no dual write. Artifact-digest validation is
unchanged; only where the descriptors live moved."
```

---

## Task 3: Stage 2 and Stage 3 become `stages` entries

**Files:**
- Modify: `src/phenotypic/_cli/_cli_stage2_token.py`
- Modify: `src/phenotypic/_cli/_cli_staged_resume.py`
- Test: `tests/unit/cli/test_staged_resume_equivalence.py`

**This is the risky task.** `classify_staged_image` (`_cli_staged_resume.py:197`) decides
per image whether to run stage 1, 2 or 3. Collapsing four filesystem probes into one read
must change none of its decisions.

- [ ] **Step 1: Write the equivalence gate BEFORE touching the staged engine**

`tests/unit/cli/test_staged_resume_equivalence.py`:

```python
"""The staged resume decisions must be identical after the marker collapse.

classify_staged_image reads four independent filesystem probes today: a valid stage-1
store, a stage-2 token, a retained stage-2 raw array, and a stage-3 completion marker.
Spec §6.1 collapses three of those into one record read. The record format is not the
risk -- the resume DECISIONS are, because a wrong one either reprocesses 6,000 images
or silently skips one.

This is a table test over every reachable combination, written against the CURRENT
behaviour and run before and after the change.
"""

import itertools

import pytest

_COMBOS = list(itertools.product([False, True], repeat=4))  # store, s2tok, s2raw, s3

#: Captured from the PRE-CHANGE behaviour in Step 2, as a literal table. Do not
#: derive these by reasoning about what the classifier should do -- the point is to
#: freeze what it DOES, so the collapse is provably behaviour-preserving.
_EXPECTED: dict[tuple[bool, bool, bool, bool], str] = {}   # filled in Step 2


@pytest.mark.parametrize("store,s2_token,s2_raw,s3_done", _COMBOS)
def test_classification_is_unchanged_by_the_collapse(
    tmp_path, store, s2_token, s2_raw, s3_done
):
    from phenotypic._cli._cli_staged_resume import classify_staged_image

    item = _plant(tmp_path, store=store, s2_token=s2_token, s2_raw=s2_raw, s3_done=s3_done)
    actual = classify_staged_image(
        tmp_path,
        dataset=item.dataset,
        image_stem=item.image_stem,
        work_id=item.work_id,
        image_path=item.image_path,
    )
    assert actual == _EXPECTED[(store, s2_token, s2_raw, s3_done)]
```

**Confirm `classify_staged_image`'s real signature before writing `_plant`:**

```bash
sed -n '197,230p' src/phenotypic/_cli/_cli_staged_resume.py
```

The keyword names above are the shape the function is expected to have; use whatever it
actually takes. `_plant` creates or omits each of the four artifacts and returns a small
record carrying the arguments the call needs.

- [ ] **Step 2: Populate `_EXPECTED` from the CURRENT code, before changing it**

Run the parametrized test against unmodified `main` with `_EXPECTED` empty, capture each
actual classification, and write those sixteen values into `_EXPECTED` **as a literal
table**. Then re-run: all sixteen pass. That table is now the contract.

Do **not** derive `_EXPECTED` by reasoning about what the classifier should do. The point is
to freeze what it *does*, so the collapse is provably behaviour-preserving. If one of the
sixteen looks wrong, record it in a comment and leave it — fixing a resume bug inside a
refactor makes both unreviewable.

- [ ] **Step 3: Collapse the two trees**

- `write_stage2_token` → `record_stage(..., "stage2", {...})`
- `stage2_token_exists` → `"stage2" in (read_image_record(...) or {}).get("stages", {})`
- `delete_stage2_token` → `consume_stage(..., "stage2")`
- `write_stage3_completion_marker` → `record_stage(..., "stage3", {...})`
- `stage3_completion_exists` → the same membership test on `"stage3"`
- `remove_stage3_completion_marker` → `consume_stage(..., "stage3")`

Keep the function names — the SLURM observer imports `stage3_completion_exists`
(`_slurm_observer.py`), and renaming it is P6's job, not this task's.

Delete `_STAGE2_DIR` (`_cli_stage2_token.py:42`), the inline `"stage3_complete"` literal
(`_cli_staged_resume.py:141`), and their path helpers. `stage2_raw_path`,
`write_stage2_raw`, `load_stage2_raw` and `delete_stage2_raw` are **unchanged**.

- [ ] **Step 4: Re-run the equivalence gate**

Run: `uv run pytest tests/unit/cli/test_staged_resume_equivalence.py -v`
Expected: all sixteen PASS, against the table captured in Step 2.

**If any combination changes, stop.** The collapse has altered a resume decision, and that
is the failure this task exists to prevent.

- [ ] **Step 5: Run the staged-engine regression**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli -k staged -q
```

- [ ] **Step 6: Phase gate**

```bash
uv run mypy src/phenotypic
uv run ruff check --fix src/phenotypic/_cli/_cli_image_record.py \
  src/phenotypic/_cli/_cli_stage2_token.py src/phenotypic/_cli/_cli_staged_resume.py \
  src/phenotypic/_cli/_cli_completion.py src/phenotypic/sdk_/_run_state.py \
  tests/unit/cli/
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli tests/unit/sdk_ -q
```

- [ ] **Step 7: Commit**

```bash
git add -A src/phenotypic/_cli tests/unit/cli
git commit -m "refactor(cli): stage2_done/ and stage3_complete/ become stages entries

Spec §6.1. Three parallel <ds>/<stem> trees, spelled in three places, become one
record. The sixteen-combination classify_staged_image table was captured from the
pre-change behaviour and is unchanged after -- the resume decisions are the risk
here, not the format."
```
# Phase 4 — Embedded-table inversion and `finalize_run`

**Depends on:** P3, P0 (S-4). **Blocks:** P5–P7.

**Spec:** §7 (measurement and metadata data flow), D8 — as amended by
[D-A](OPEN-QUESTIONS.md#d-a-per-store-metadata-is-written-at-promote-time-not-backfilled).

**Goal:** embedded per-image tables carry **measurements only**; each store's user metadata
is written as `tables/metadata/pht-metadata.parquet` **in the same `.part` as the
measurements**, before the root `zarr.json`; the metadata join moves to finalization; and
`finalize_run` becomes the one aggregation + join + publish path for `full`, `measure` and
`recompile`.

**Local path only.** SLURM and `--njobs` fan-out is P5.

### What D-A changes from spec §7

`finalize_run` is **six steps, not seven** — step 6 ("backfill `pht-metadata.parquet` per
store — certified re-promote") is cut. The metadata table is written at promote time
instead, so no artifact carrying a content proof is ever mutated (**INV-IMMUTABLE**).

§7.4's late-metadata guarantee narrows correspondingly, and the narrowing must be
documented where users read it, not only here:

> A `metadata.csv` edit changes `metadata_sha256`, invalidating
> `finalization_input_digest`, so the next invocation re-runs `finalize_run` — re-joining
> the mirror. **Stores keep the metadata snapshot they were built against**; each store's
> `phenotypic.metadata.snapshot_sha256` records which one, and `resolve_run_state` raises
> an advisory when they diverge (P1 Task 5).

---

## File Structure

| File | Responsibility |
|---|---|
| **Modify** `src/phenotypic/_cli/_embedded_measurement_tables.py:42` | `prepare_embedded_measurement_table` returns the **unjoined** baseline plus a separate metadata projection. |
| **Modify** `src/phenotypic/sdk_/_measurement_tables.py` | Write both tables into the `.part`; extend the root `tables` block. |
| **Create** `src/phenotypic/_cli/_cli_finalize_run.py` | `finalize_run(output_dir, …)` — the one path. ~260 lines. |
| **Modify** `src/phenotypic/_cli/_cli_output_manager.py:1351` | `_aggregate_measurements_unlocked` delegates aggregation to `finalize_run`. |
| **Modify** `src/phenotypic/_cli/_cli_recompile_worker.py:764` | `_run_post_master_steps` becomes a `finalize_run` call. |
| **Modify** `src/phenotypic/_cli/_cli_completion.py:868` | Aggregate proof's `required_outputs` drops `master_csv` (D8). |
| **Delete** | `MASTER_MEASUREMENTS_CSV`, `master_measurements_csv_path()`, `BundleLayout.master_csv`, `load_master_measurements()` (D8). |
| **Test** `tests/unit/cli/test_finalize_run.py` *(new)* | INV-INPUTS, the six steps, the three entry points. |
| **Test** `tests/unit/cli/test_embedded_table_inversion.py` *(new)* | Intrinsic/user metadata boundary; curation re-keying. |

---

## Interfaces

**Produces:**

```python
# phenotypic._cli._cli_finalize_run

def finalize_run(
    output_dir: Path,
    *,
    dataset_names: Sequence[str],
    pipeline: "ImagePipeline | None" = None,
    metadata_csv: Path | None = None,
    no_qc: bool = False,
    study_config: dict | None = None,
    shard_paths: Sequence[Path] | None = None,   # P5 supplies these; None = local concat
    commit_guard: "CommitGuard | None" = None,
) -> Path | None:
    """The one aggregation + join + publish path (spec §7.4)."""
```

```python
# phenotypic._cli._embedded_measurement_tables

@dataclass(frozen=True)
class PreparedImageTables:
    measurements: pd.DataFrame          # intrinsic identity only, NO user metadata
    metadata: pd.DataFrame | None       # user metadata rows + join keys, or None
    measurement_columns: tuple[str, ...]
    join_status: Literal["joined", "not_requested", "no_common_keys"]
    join_keys: tuple[str, ...]
    metadata_snapshot_sha256: str

def prepare_image_tables(
    measurements: pd.DataFrame, metadata_csv: Path | None
) -> PreparedImageTables: ...
```

**Consumes:** P3's `publish_image_record`; `phenotypic.sdk_.promote_store`,
`MEASUREMENT_TABLE_RELATIVE_PATH`.

---

## Task 1: Split the embedded table into measurements and metadata

**Files:**
- Modify: `src/phenotypic/_cli/_embedded_measurement_tables.py:42`
- Test: `tests/unit/cli/test_embedded_table_inversion.py`

**This is subtraction, not invention.** `prepare_embedded_measurement_table` already
computes `measurement_columns` from the baseline **before** joining
(`_embedded_measurement_tables.py:55`) and writes it as
`phenotypic.measurement_columns`. "Embedded table without user metadata" is exactly that
existing projection.

- [ ] **Step 1: Write the failing tests**

```python
def test_intrinsic_identity_stays_in_the_measurement_table(tmp_path):
    """Spec §7.1: a concatenated row that cannot say which image it came from is
    unusable. Metadata_ImageFile, Metadata_Dataset and the object label stay."""
    from phenotypic._cli._embedded_measurement_tables import prepare_image_tables

    prepared = prepare_image_tables(_measurements_with_metadata(), _metadata_csv(tmp_path))
    assert "Metadata_ImageFile" in prepared.measurements.columns
    assert "Metadata_Dataset" in prepared.measurements.columns


def test_user_metadata_leaves_the_measurement_table(tmp_path):
    """§7.3's contract change. Metadata_Strain came from --metadata, not from the
    image, so it belongs in pht-metadata.parquet."""
    from phenotypic._cli._embedded_measurement_tables import prepare_image_tables

    prepared = prepare_image_tables(_measurements_with_metadata(), _metadata_csv(tmp_path))
    assert "Metadata_Strain" not in prepared.measurements.columns
    assert "Metadata_Strain" in prepared.metadata.columns


def test_the_measurement_table_equals_the_pre_join_baseline_exactly(tmp_path):
    """The boundary already has a name: measurement_columns, computed from the
    baseline BEFORE joining (_embedded_measurement_tables.py:55). This asserts the
    new split IS that projection rather than a re-derivation of it."""
    from phenotypic._cli._embedded_measurement_tables import prepare_image_tables

    baseline = _measurements_with_metadata()
    prepared = prepare_image_tables(baseline, _metadata_csv(tmp_path))
    assert tuple(prepared.measurements.columns) == prepared.measurement_columns


def test_no_metadata_table_when_the_join_was_not_requested(tmp_path):
    """§7.2: absence is the honest signal."""
    from phenotypic._cli._embedded_measurement_tables import prepare_image_tables

    prepared = prepare_image_tables(_measurements_with_metadata(), None)
    assert prepared.metadata is None
    assert prepared.join_status == "not_requested"


def test_no_metadata_table_when_no_columns_are_in_common(tmp_path):
    from phenotypic._cli._embedded_measurement_tables import prepare_image_tables

    prepared = prepare_image_tables(
        _measurements_with_metadata(), _unrelated_metadata_csv(tmp_path)
    )
    assert prepared.metadata is None
    assert prepared.join_status == "no_common_keys"


def test_duplicate_metadata_keys_preserve_fan_out(tmp_path):
    """The behaviour prepare_embedded_measurement_table already warns about, and
    the one S-4 spiked. Losing it silently changes row counts in the mirror."""
    from phenotypic._cli._embedded_measurement_tables import prepare_image_tables

    prepared = prepare_image_tables(
        _measurements_with_metadata(), _metadata_csv_with_duplicate_keys(tmp_path)
    )
    assert len(prepared.metadata) == 3
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`prepare_image_tables` keeps `prepare_embedded_measurement_table`'s normalization and its
`prepare_metadata_join_keys` call, and then **stops before the right join**
(`_embedded_measurement_tables.py:88-95`). `measurements` is the baseline; `metadata` is
the semi-join of the metadata frame onto that image's distinct join keys.

**S-4's verdict licenses this.** If S-4 returned `FAIL`, stop and report — a local
projection that diverges from a global one means the promote-time write cannot be correct
and D-A needs revisiting with the user.

Keep `prepare_embedded_measurement_table` as a thin wrapper for one release **only if** a
caller outside this change needs it; grep first, and delete it if not.

- [ ] **Step 4: Run the tests.** Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_embedded_measurement_tables.py \
        tests/unit/cli/test_embedded_table_inversion.py
git commit -m "feat(cli): split the embedded table into measurements and metadata

Spec §7.1-7.2. Subtraction, not invention: measurement_columns already recorded
this boundary, computed from the baseline before the join."
```

---

## Task 2: Write both tables at promote time

**Files:**
- Modify: `src/phenotypic/sdk_/_measurement_tables.py`
- Test: `tests/unit/cli/test_embedded_table_inversion.py`, `tests/unit/sdk_/`

Implements **D-A** and **INV-IMMUTABLE**.

- [ ] **Step 1: Write the failing tests**

```python
def test_both_tables_land_in_the_same_part_before_the_root(tmp_path):
    """D-A / INV-IMMUTABLE. The root zarr.json is written last and is the record's
    content anchor (_cli_completion.py:41-47), so anything written after it is a
    mutation of a proven artifact. Writing metadata in the same .part is what makes
    the backfill unnecessary."""
    store = _build_store_with_metadata(tmp_path)
    assert (store / "tables" / "measurements" / "table.parquet").is_file()
    assert (store / "tables" / "metadata" / "pht-metadata.parquet").is_file()
    root = json.loads((store / "zarr.json").read_text())
    assert "metadata" in root["attributes"]["phenotypic"]["tables"]


def test_the_store_records_the_metadata_snapshot_it_was_built_against(tmp_path):
    """D-A: stores keep the metadata they were built with, and say which one. That
    is what lets resolve_run_state DERIVE the divergence advisory instead of
    tracking a backfill stage."""
    store = _build_store_with_metadata(tmp_path)
    root = json.loads((store / "zarr.json").read_text())
    assert root["attributes"]["phenotypic"]["metadata"]["snapshot_sha256"]


def test_nothing_writes_into_a_store_after_its_record_is_published(tmp_path):
    """INV-IMMUTABLE, as a property test rather than a convention.

    Publish a record, snapshot every mtime under the store, run finalize_run, and
    assert not one file moved. This is the test that would have caught the backfill
    if it had shipped."""
    from phenotypic._cli._cli_finalize_run import finalize_run

    store = _publish_a_successful_image(tmp_path, dataset="plate", stem="a")
    before = {p: p.stat().st_mtime_ns for p in sorted(store.rglob("*")) if p.is_file()}
    finalize_run(tmp_path, dataset_names=["plate"])
    after = {p: p.stat().st_mtime_ns for p in sorted(store.rglob("*")) if p.is_file()}
    assert before == after, (
        "finalize_run mutated a store that already carries a content proof; "
        "INV-IMMUTABLE forbids it and D-A removed the only mechanism that did"
    )
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

Extend the `.part` writer to emit `tables/metadata/pht-metadata.parquet` when
`prepared.metadata is not None`, before `OME/zarr.json` and the root. Extend the root's
`attributes.phenotypic.tables` with a `metadata` descriptor, and add
`attributes.phenotypic.metadata = {"snapshot_sha256": …, "join_keys": [...],
"join_kind": …}`.

The Parquet KV keys ride along unchanged (§7.2): `phenotypic.join.keys`,
`phenotypic.join.kind`, `phenotypic.metadata.snapshot_sha256`. The join is self-describing
from the file itself, which is the property that makes the store useful to a third party at
all.

**Order is load-bearing and is the whole of INV-IMMUTABLE:** chunks → both tables →
`OME/zarr.json` → root `zarr.json` → `promote_store`. Any other order and an interrupted
store can read as present.

- [ ] **Step 4: Run the tests plus the NGFF conformance suite**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/sdk_ tests/_ngff_conformance.py -q
```

The store gains a table, so its NGFF conformance must be re-checked — a non-conforming
store is one `napari` cannot open, which is half of why it is OME-Zarr.

- [ ] **Step 5: Commit**

```bash
git add -A src/phenotypic/sdk_ src/phenotypic/_cli tests/unit
git commit -m "feat(sdk): write pht-metadata.parquet in the store's original promote

D-A. No post-proof store mutation exists on any forward path, so §6.3's hardlink
re-promote and §6.4's receipt generalisation are both unnecessary. INV-IMMUTABLE is
pinned by a property test over every file's mtime across a finalize_run."
```

---

## Task 3: `finalize_run` — the one path

**Files:**
- Create: `src/phenotypic/_cli/_cli_finalize_run.py`
- Test: `tests/unit/cli/test_finalize_run.py`

The seam already exists and is already shared: `finalize_post_master_outputs`
(`_cli_output_manager.py:969`) is called by both the forward path (`:1526`) and the
recompile worker (`_cli_recompile_worker.py:802`), whose own comment says it is "matching
the forward CLI path". What is **not** shared is aggregation. This task widens the seam to
own it.

- [ ] **Step 1: Write INV-INPUTS first — the phase's gate**

```python
def test_finalize_run_ignores_every_stale_intermediate(tmp_path):
    """INV-INPUTS / spec §7.5. Plant a stale chunk parquet, a stale shard, a stale
    _dataset_aggregated.parquet, a stale analysis_full.parquet and a stale master,
    each containing a row that exists in NO embedded table. Assert the new master
    matches a concat of the embedded tables exactly.

    Those files are outputs and intermediates of a PREVIOUS finalization, not inputs
    to this one. Under a rolling input, reusing any of them silently omits images
    that arrived since the cache was built, or retains rows for an image whose
    content changed and therefore has a new work_id.
    """
    import polars as pl

    from phenotypic._cli._cli_finalize_run import finalize_run
    from phenotypic.sdk_ import master_measurements_parquet_path

    _publish_two_successful_images(tmp_path)
    poison = pl.DataFrame({"Metadata_ImageFile": ["GHOST.tif"], "Shape_Circularity": [0.0]})
    _plant_stale_chunk_parquet(tmp_path, poison)
    _plant_stale_shard(tmp_path, poison)
    _plant_stale_dataset_aggregate(tmp_path, poison)
    _plant_stale_analysis_full(tmp_path, poison)
    _plant_stale_master(tmp_path, poison)

    finalize_run(tmp_path, dataset_names=["plate"])

    master = pl.read_parquet(master_measurements_parquet_path(tmp_path))
    assert "GHOST.tif" not in master["Metadata_ImageFile"].to_list()
    assert master.equals(_concat_of_embedded_tables(tmp_path))


def test_finalize_run_invalidates_the_intermediates_on_success(tmp_path):
    """§7.5: so a later invocation cannot mistake them for inputs."""
    from phenotypic._cli._cli_finalize_run import finalize_run

    _publish_two_successful_images(tmp_path)
    chunk = _plant_stale_chunk_parquet(tmp_path, _poison())
    finalize_run(tmp_path, dataset_names=["plate"])
    assert not chunk.exists()


def test_the_master_carries_no_user_metadata(tmp_path):
    """§7.3's contract change, stated as a test.

    The one genuinely dangerous failure mode in §7 is code that filters the master
    on a user-metadata column: it returns EMPTY rather than erroring. The schema
    version P7 stamps is what makes an old reader fail loudly instead."""
    import polars as pl

    from phenotypic._cli._cli_finalize_run import finalize_run
    from phenotypic.sdk_ import (
        master_measurements_parquet_path,
        measurements_parquet_path,
    )

    _publish_two_successful_images(tmp_path, metadata=True)
    finalize_run(tmp_path, dataset_names=["plate"], metadata_csv=_metadata_csv(tmp_path))

    master = pl.read_parquet(master_measurements_parquet_path(tmp_path))
    mirror = pl.read_parquet(measurements_parquet_path(tmp_path))
    assert "Metadata_Strain" not in master.columns
    assert "Metadata_Strain" in mirror.columns


def test_curation_re_keying_still_works_against_the_intrinsic_master(tmp_path):
    """§7.3 names this as needing an explicit test rather than assumption.

    Curation deliberately reads the CLEAN master so labels survive for curated-out
    objects (_curation_labels.py:406). It keys on dataset / image / object-label --
    all intrinsic -- so it should be unaffected. Test it; do not assume it."""
    _publish_two_successful_images(tmp_path, metadata=True)
    _finalize_and_curate(tmp_path, curated_out=["a.tif::3"])
    assert _curated_label_survives(tmp_path, "a.tif::3")


def test_master_measurements_csv_is_gone(tmp_path):
    """D8: master is parquet-only. The un-joined master is no longer the file a
    human opens -- the mirror is."""
    from phenotypic._cli._cli_finalize_run import finalize_run

    _publish_two_successful_images(tmp_path)
    finalize_run(tmp_path, dataset_names=["plate"])
    assert not (tmp_path / "deliverables" / "master_measurements.csv").exists()
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement the six steps**

```python
def finalize_run(output_dir, *, dataset_names, pipeline=None, metadata_csv=None,
                 no_qc=False, study_config=None, shard_paths=None, commit_guard=None):
    """Aggregate, join, publish -- one path for `full`, `measure` and `recompile`.

    Six steps (spec §7.4, minus the backfill D-A cut):

    1. select marker-authorized embedded measurement tables
    2. concat  ->  master_measurements.parquet          (un-joined, D8: no CSV)
    3. join metadata + append metadata-only phantoms + apply post ops
    4. write  ->  deliverables/measurements.{parquet,csv}
    5. persist pipeline.json, analysis outputs, per-feature splits
    6. publish aggregate proof -> run proof

    INVARIANT (INV-INPUTS, §7.5) -- **step 1 selects exactly the marker-authorized
    embedded measurement tables.** It never reads a prior master, chunk parquet,
    measurement shard, ``analysis_full.parquet`` or ``_dataset_aggregated.parquet``
    as an aggregation input. Those are outputs and intermediates of a PREVIOUS
    finalization; under a rolling input, reusing one silently omits images that
    arrived since, or retains rows for an image whose content changed and therefore
    has a new ``work_id``. Master is a pure function of the currently authorized
    embedded tables -- which is the derivability property this whole design is for.

    ``shard_paths`` is P5's fan-out hook: when supplied, step 2 merges those instead
    of reading the tables directly. It does not weaken INV-INPUTS, because the shards
    were themselves produced from authorized embedded tables **in this invocation**,
    namespaced by ``scheduler_epoch`` so a prior run's shards can never be merged.
    """
```

Step 1 calls the existing `authorized_measurement_sources`
(`_cli_completion.py:768`) — already the right predicate, already marker-authorized. Step 3
onward is `finalize_post_master_outputs`, unchanged.

This **deletes recompile's separate master-merge** and collapses the `measurement_sources`
vs `metadata_join_keys` branch in `_run_post_master_steps`
(`_cli_recompile_worker.py:777-787`), which exists only because the two callers arrive with
differently-shaped inputs. After this task they arrive the same way.

- [ ] **Step 4: Run the tests.** Expected: PASS (5 passed).

- [ ] **Step 5: Prove INV-INPUTS can fail**

Add a `_dataset_aggregated.parquet` fast path to step 1 — the shape
`_aggregate_measurements_unlocked`'s docstring describes today ("Prefers pre-aggregated
`_dataset_aggregated.parquet` files when available"). Confirm
`test_finalize_run_ignores_every_stale_intermediate` fails, then remove it. **That fast
path is exactly the bug INV-INPUTS forbids, and it is in the current code.**

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_cli/_cli_finalize_run.py tests/unit/cli/test_finalize_run.py
git commit -m "feat(cli): finalize_run -- one aggregation and publication path

Spec §7.4, §7.5, six steps (D-A cut the backfill). INV-INPUTS was confirmed to fail
when the _dataset_aggregated.parquet fast path the current aggregator documents is
reintroduced."
```

---

## Task 4: Route all three entry points through `finalize_run`

**Files:**
- Modify: `src/phenotypic/_cli/_cli_output_manager.py:1351`, `:1545`
- Modify: `src/phenotypic/_cli/_cli_recompile_worker.py:764`
- Test: `tests/unit/cli/test_finalize_run.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("mode", ["full", "measure", "recompile"])
def test_every_mode_produces_a_byte_identical_master(tmp_path, mode):
    """§7.4: recompile becomes 'call finalize_run again', not a parallel
    implementation that must be kept in sync. Three modes, one master."""
    output = _run_mode(tmp_path, mode)
    assert _master_bytes(output) == _master_bytes(_run_mode(tmp_path / "ref", "full"))


def test_process_mode_skips_finalization_entirely(tmp_path):
    """§7.4's table: `process` writes one layer, no measurement, and
    process_only_layer already short-circuits the aggregate proof."""
    output = _run_mode(tmp_path, "process")
    assert not (output / "deliverables" / "master_measurements.parquet").exists()
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`_aggregate_measurements_unlocked` keeps its lock (`aggregate_measurements`'s
`.aggregate_publication.lock`, `_cli_output_manager.py:1552`) and delegates its body.
`_run_post_master_steps` becomes a `finalize_run` call, keeping its
`generation_publication_guard` wrapper.

- [ ] **Step 4: Delete the D8 surfaces**

`MASTER_MEASUREMENTS_CSV`, `master_measurements_csv_path()`, `BundleLayout.master_csv`,
`load_master_measurements()`, and the `master_csv` entry in the aggregate proof's
`required_outputs` (`_cli_completion.py:888`). The proof's `required_outputs` drops from
four artifacts to three.

Per [Q6](OPEN-QUESTIONS.md#q6-ten-test-files-depend-on-master_measurements_csv_path), ten
test files reference `master_measurements_csv_path`. Fix each: assert on the parquet, or on
`measurements.csv` where the test genuinely wanted a human-readable file.
`BundleLayout.detect` keys on `master_measurements.parquet` already, so bundle detection is
unaffected.

- [ ] **Step 5: Phase gate**

```bash
uv run mypy src/phenotypic
uv run ruff check --fix src/phenotypic/_cli/_cli_finalize_run.py \
  src/phenotypic/_cli/_cli_output_manager.py src/phenotypic/_cli/_cli_recompile_worker.py \
  src/phenotypic/_cli/_embedded_measurement_tables.py src/phenotypic/sdk_/ tests/unit/
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit -q
```

This is the first phase where the full `tests/unit` suite is warranted rather than a
selection — the master's shape changed and it is read almost everywhere. **The suite is
~65 minutes and is a Slurm job**: use the **`run-phenotypic-test`** and **`slurm-job`**
skills, with the committed script at
`docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`. Never
`-n auto` (it reads node cores, not the allocation) and never `-x` (it truncates a run that
then gets recorded as a baseline).

- [ ] **Step 6: Update the docs the contract change invalidates**

- `CLAUDE.md`'s "Output layout (`deliverables/`)" bullet: `master_measurements.*` is now
  `master_measurements.parquet`, un-joined and intrinsic-only.
- `src/phenotypic/_cli/CLAUDE.md`'s master-vs-mirror rules.
- `docs/source/how_to/pages/` wherever the master is described as metadata-joined.

- [ ] **Step 7: Commit**

```bash
git add -A src/phenotypic tests docs CLAUDE.md
git commit -m "refactor(cli): route full, measure and recompile through finalize_run

Spec §7.3, §7.4, D8. Deletes recompile's separate master-merge and the
measurement_sources/metadata_join_keys branch that existed only because the two
callers arrived with differently-shaped inputs. Master is parquet-only and carries
intrinsic identity only; the mirror carries the join."
```

---

## Task 5: Verify the promote-time metadata end to end

**Files:**
- Test: `tests/integration/` (a real single-image run)

- [ ] **Step 1: Run a real local run with `--metadata` and assert the store is self-describing**

```python
def test_a_real_run_leaves_stores_a_third_party_can_join(tmp_path):
    """D-A's whole justification: the store is self-describing WITHOUT any post-hoc
    rewrite. Read it back with plain pyarrow -- no phenotypic import in the assertion
    path -- and join it, the way a napari or QuPath user would."""
    import pyarrow.parquet as pq

    output = _run_full_pipeline(tmp_path, metadata=True)
    store = next(output.rglob("*.ome.zarr"))

    measurements = pq.read_table(store / "tables" / "measurements" / "table.parquet")
    metadata = pq.read_table(store / "tables" / "metadata" / "pht-metadata.parquet")
    keys = json.loads(metadata.schema.metadata[b"phenotypic.join.keys"])
    joined = measurements.to_pandas().merge(metadata.to_pandas(), on=keys, how="left")
    assert "Metadata_Strain" in joined.columns
```

- [ ] **Step 2: Run it.** Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/integration
git commit -m "test(cli): a promoted store is joinable by a third party with pyarrow alone

D-A. The assertion path imports no phenotypic code, which is the property that makes
'self-describing' mean something."
```
# Phase 5 — Fan-out: SLURM array and local `--njobs`

**Depends on:** P4, P0 (S-2, S-3). **Blocks:** P6, P7.

**Spec:** §8 (fan-out) — as amended by
[D-A](OPEN-QUESTIONS.md#d-a-per-store-metadata-is-written-at-promote-time-not-backfilled).

**Goal:** `finalize_run`'s aggregation fans out over SLURM array tasks and local `--njobs`,
with `TASK_FINALIZE` as a **reserved trigger entry inside the array task list** — never a
parallel sidecar job.

### What D-A changes from spec §8

§8's array task did two things per image: append to a measurement shard **and** project
metadata into a certified re-promote. The second half is gone — per-store metadata is
written at promote time (P4 Task 2). Shard workers **aggregate only**:

```
array task i ∈ [0, K):                          # aggregate
    for image in shard_i:
        read tables/measurements/table.parquet
        └─ append → measurement_shards/<scheduler_epoch>/shard_i.parquet

array task K (TASK_FINALIZE, dependent):        # reduce
    merge shard_*.parquet → master_measurements.parquet
    join + phantoms + post ops → measurements.{parquet,csv}
    pipeline.json, analysis outputs, per-feature splits
    publish aggregate proof → run proof
```

§8's "ordering and partial failure" narrows to **two** phases, not three: a run that
finishes aggregation has a valid aggregate proof and, once the finalizer publishes, a run
proof. There is no aggregated-not-backfilled state, because there is no backfill.

**The shape already exists.** Recompile has `TASK_MEASUREMENTS` (sharded by `shard_id`,
`_cli_recompile_slurm_scripts.py:146`), `TASK_OVERLAY` (`:339`) and `TASK_FINALIZE`
(`:198`). This phase promotes it to be universal rather than inventing it.

---

## File Structure

| File | Responsibility |
|---|---|
| **Create** `src/phenotypic/_cli/_cli_finalize_fanout.py` | Shard-count sizing, shard worker body, local process-pool driver. ~200 lines. |
| **Modify** `src/phenotypic/_cli/_cli_finalize_run.py` | Accept `shard_paths`; merge instead of concat when supplied. |
| **Modify** `src/phenotypic/_cli/_cli_slurm_array_scripts.py:30` | Add the finalize trigger beside `_CHECKPOINT_SENTINEL` and `_MANIFEST_SENTINEL`. |
| **Modify** `src/phenotypic/sdk_/_io_constants.py` | `measurement_shard_dir(output_dir, scheduler_epoch)`. |
| **Test** `tests/unit/cli/test_finalize_fanout.py` *(new)* | Sizing, epoch namespacing, partial-failure matrix. |
| **Test** `tests/unit/cli/test_array_auxiliary_routing.py` *(new)* | **The rule from `_cli/CLAUDE.md`: no standalone parallel job.** |

---

## Task 1: Shard sizing, counted against `MaxArraySize`

**Files:**
- Create: `src/phenotypic/_cli/_cli_finalize_fanout.py`
- Test: `tests/unit/cli/test_finalize_fanout.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_the_finalize_trigger_is_counted_against_the_array_bound():
    """Project CLAUDE.md: 'Count every trigger entry when sizing chunks against
    MaxArraySize.' A formula that sizes K to the bound and THEN appends the
    finalizer produces an array one index too long, which sbatch rejects at
    submission with a message that names neither the trigger nor the formula."""
    from phenotypic._cli._cli_finalize_fanout import shard_count

    k = shard_count(n_images=1_000_000, seconds_per_image=1.0, max_array_size=2500)
    assert k <= 2499, "K + TASK_FINALIZE must fit inside MaxArraySize"


def test_max_array_size_caps_the_index_not_the_task_count():
    """User's global CLAUDE.md: MaxArraySize (2500 here) caps the INDEX. The highest
    legal index is 2499 -- --array=0-2499 works, --array=1-2500 is rejected."""
    from phenotypic._cli._cli_finalize_fanout import array_spec, shard_count

    k = shard_count(n_images=1_000_000, seconds_per_image=1.0, max_array_size=2500)
    assert array_spec(k) == f"0-{k}"      # K shards + one finalizer index
    assert k < 2500


def test_shard_count_is_one_for_a_small_run():
    from phenotypic._cli._cli_finalize_fanout import shard_count

    assert shard_count(n_images=5, seconds_per_image=0.1, max_array_size=2500) == 1


def test_shards_are_namespaced_by_scheduler_epoch(tmp_path):
    """§7.5: measurement shards are per-invocation scratch, so a prior run's shards
    can never be merged. Recompile already does this
    (recompile/attempts/<attempt_id>/...); the pattern generalises."""
    from phenotypic.sdk_ import measurement_shard_dir

    a = measurement_shard_dir(tmp_path, "epoch-a")
    b = measurement_shard_dir(tmp_path, "epoch-b")
    assert a != b
    assert a.parent == b.parent
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

```python
#: Target wall-clock for one shard task. Chosen so a shard fits comfortably inside
#: `short`'s 2-hour cap with room for scheduler latency and a slow GPFS read, and so
#: a preempted `preempt` task loses at most this much work.
TARGET_TASK_SECONDS = 900


def shard_count(*, n_images: int, seconds_per_image: float, max_array_size: int) -> int:
    """Return K, the number of aggregation shard tasks.

    The ``- 1`` reserves the ``TASK_FINALIZE`` trigger entry's index. Project
    CLAUDE.md requires every trigger entry to be counted when sizing chunks against
    ``MaxArraySize``; the failure mode of not doing so is an sbatch rejection whose
    message names neither the trigger nor the formula.

    ``max_array_size`` caps the *index*, not the task count -- with the cluster's
    2500, the highest legal index is 2499.

    Args:
        n_images: Images to aggregate.
        seconds_per_image: Measured by spike S-2 against a real tree on GPFS.
        max_array_size: ``scontrol show config``'s ``MaxArraySize``, or
            ``MaxSubmitJobs`` when it is lower.

    Returns:
        K in ``[1, max_array_size - 1]``.
    """
    import math

    target = math.ceil(n_images * seconds_per_image / TARGET_TASK_SECONDS)
    return max(1, min(target, max_array_size - 1))
```

Use the `seconds_per_image` **S-2 measured**, recorded in `spikes/RESULTS.md`. Cite the
number and the fixture in the constant's docstring — a magic number nobody can trace back
to a measurement is a guess with better formatting.

- [ ] **Step 4: Run the tests.** Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_finalize_fanout.py \
        src/phenotypic/sdk_/_io_constants.py tests/unit/cli/test_finalize_fanout.py
git commit -m "feat(cli): shard sizing for aggregation fan-out

Spec §8. K reserves the TASK_FINALIZE index; seconds_per_image comes from S-2 on a
real GPFS tree, cited in the constant's docstring."
```

---

## Task 2: The shard worker and the reserved finalize trigger

**Files:**
- Modify: `src/phenotypic/_cli/_cli_finalize_fanout.py`
- Modify: `src/phenotypic/_cli/_cli_slurm_array_scripts.py:30`
- Test: `tests/unit/cli/test_array_auxiliary_routing.py`

**Read `src/phenotypic/_cli/CLAUDE.md`'s array-auxiliary-work contract before starting.**
It is the rule this task is most likely to break, and root `AGENTS.md` is a symlink to the
project `CLAUDE.md`, so it carries the same rule.

- [ ] **Step 1: Write the routing test first — it is the contract**

```python
"""No standalone parallel job beside an active ordinary array.

Project CLAUDE.md and _cli/CLAUDE.md: allocation and submission bounds are already
consumed by the array cohort. Ancillary work routes through reserved trigger entries
INSIDE the array task list, following the existing __PHENOTYPIC_CHECKPOINT__ and
__PHENOTYPIC_MANIFEST__ dispatch pattern (_cli_slurm_array_scripts.py:30-32).

A terminal `afterany` finalizer is NOT a parallel sidecar and is allowed.
"""


def test_finalization_submits_no_job_beside_the_array(tmp_path, fake_sbatch):
    from phenotypic._cli._cli_finalize_fanout import submit_aggregation

    submit_aggregation(tmp_path, dataset_names=["plate"], n_images=500)

    submissions = fake_sbatch.calls
    arrays = [c for c in submissions if "--array" in c.argv]
    assert len(arrays) == 1, f"expected exactly one array submission, got {submissions}"
    siblings = [c for c in submissions if c is not arrays[0] and not c.is_afterany]
    assert not siblings, f"a parallel sidecar job was submitted: {siblings}"


def test_the_finalize_entry_lives_inside_the_array_task_list(tmp_path):
    from phenotypic._cli._cli_finalize_fanout import build_task_list

    tasks = build_task_list(tmp_path, dataset_names=["plate"], n_images=500)
    assert tasks[-1]["task_type"] == "finalize"
    assert all(t["task_type"] == "measurements" for t in tasks[:-1])
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

Reuse recompile's task-type vocabulary verbatim — `TASK_MEASUREMENTS`, `TASK_FINALIZE`
(`_cli_recompile_slurm_scripts.py:51-53`) — rather than minting new names. Two vocabularies
for one dispatch pattern is the cardinality problem this whole change is about.

The shard worker body is one pass over its images: read
`tables/measurements/table.parquet`, append to
`measurement_shards/<scheduler_epoch>/shard_i.parquet`. Nothing else — no store write, no
metadata projection (D-A), no global frame.

`TASK_FINALIZE` calls `finalize_run(..., shard_paths=[...])`.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A src/phenotypic/_cli tests/unit/cli
git commit -m "feat(cli): shard worker plus the reserved TASK_FINALIZE trigger entry

Spec §8 and _cli/CLAUDE.md's array-auxiliary contract. Reuses recompile's task-type
vocabulary rather than minting a second one."
```

---

## Task 3: Local `--njobs` uses the same decomposition

**Files:**
- Modify: `src/phenotypic/_cli/_cli_finalize_fanout.py`
- Test: `tests/unit/cli/test_finalize_fanout.py`

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("njobs", [1, 2, 8])
def test_local_fanout_produces_a_byte_identical_master(tmp_path, njobs):
    """§8: 'Local --njobs uses the same decomposition with a process pool.' Same
    decomposition means same answer -- if the merge order can change the master's
    bytes, two runs of the same data disagree and the aggregate proof's
    source_set_digest becomes meaningless."""
    output = _run_and_finalize(tmp_path / str(njobs), njobs=njobs)
    assert _master_bytes(output) == _master_bytes(_run_and_finalize(tmp_path / "ref", njobs=1))
```

- [ ] **Step 2: Run to verify failure.** Expected: FAIL on shard-order nondeterminism.

- [ ] **Step 3: Implement**

Same `shard_count`, same shard worker, `concurrent.futures.ProcessPoolExecutor` instead of
an array. **Merge shards in sorted `shard_id` order**, and assign images to shards
deterministically by sorted `work_id` — otherwise the master's row order depends on
scheduling, and a re-run of identical inputs produces different bytes.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A src/phenotypic/_cli tests/unit/cli
git commit -m "feat(cli): local --njobs aggregation via the same decomposition

Spec §8. Deterministic shard assignment and merge order, so the master is
byte-identical across njobs -- otherwise source_set_digest certifies nothing."
```

---

## Task 4: The partial-failure matrix — the phase gate

**Files:**
- Test: `tests/unit/cli/test_finalize_fanout.py`

Spec §14's second named test. §8's partial-failure story narrows to two phases under D-A,
so the matrix is smaller than the spec's — and that is the point.

- [ ] **Step 1: Write the matrix**

```python
@pytest.mark.parametrize(
    "kill_after,expected_completion,expected_remaining",
    [
        ("nothing",            "complete",   set()),
        ("some_images",        "incomplete", {"measure"}),
        ("all_images",         "incomplete", {"aggregate"}),
        ("some_shards",        "incomplete", {"aggregate"}),
        ("all_shards",         "incomplete", {"finalize"}),
        ("master_written",     "incomplete", {"finalize"}),
    ],
)
def test_a_run_killed_mid_finalization_resumes_only_the_missing_phase(
    tmp_path, kill_after, expected_completion, expected_remaining
):
    """Spec §14's partial-failure matrix, narrowed by D-A to two phases.

    The aggregate proof asserts master + mirror; the run proof asserts everything.
    A run that finishes aggregation and dies before the run proof has a valid
    aggregate proof and no run proof -- a resumable state.
    """
    from phenotypic.sdk_ import resolve_run_state

    output = _run_until(tmp_path, kill_after)
    state = resolve_run_state(output, depth="deep")
    assert state.completion == expected_completion
    assert _remaining_phases(state) == expected_remaining


def test_a_prior_epochs_shards_are_never_merged(tmp_path):
    """§7.5: shards are per-invocation scratch, namespaced by scheduler_epoch. A
    prior run's shards being merged is the stale-cache hazard §7.5 exists to
    forbid, arriving through the fan-out instead of through the aggregator."""
    import polars as pl

    from phenotypic._cli._cli_finalize_run import finalize_run
    from phenotypic.sdk_ import master_measurements_parquet_path, measurement_shard_dir

    _publish_two_successful_images(tmp_path)
    stale = measurement_shard_dir(tmp_path, "old-epoch")
    stale.mkdir(parents=True)
    pl.DataFrame({"Metadata_ImageFile": ["GHOST.tif"]}).write_parquet(
        stale / "shard_0000.parquet"
    )

    finalize_run(tmp_path, dataset_names=["plate"])
    master = pl.read_parquet(master_measurements_parquet_path(tmp_path))
    assert "GHOST.tif" not in master["Metadata_ImageFile"].to_list()
```

- [ ] **Step 2: Run to verify failure, then implement until green.**

- [ ] **Step 3: Apply S-3's merge verdict**

If `spikes/RESULTS.md` says `S-3 STREAMING`, `TASK_FINALIZE` uses
`pl.scan_parquet([...]).sink_parquet(...)`. If it says `S-3 IN-MEMORY`, it uses
`pl.concat` and the finalizer's `--mem` is set to 2 × the projected peak RSS S-3 measured.
**Cite the measured number** wherever `--mem` is set.

- [ ] **Step 4: Phase gate — a real SLURM run**

The unit tests use a fake scheduler. That is necessary and not sufficient: the failure
modes this phase can produce are all scheduler-shaped. Submit one real fan-out on the
fixture tree via the **`slurm-job`** skill, and verify:

```bash
# submission ≠ start
scontrol show job <id> | grep -E 'StartTime|Reason'
# exactly one array, one dependent finalizer, no sidecar
sacct -j <id> --format=JobID,JobName,State,ExitCode
```

Confirm: exactly one array cohort plus one `afterany` finalizer; no third job; the master
matches a local `--njobs 1` run byte for byte.

- [ ] **Step 5: Commit**

```bash
git add -A src/phenotypic/_cli tests/unit/cli
git commit -m "test(cli): partial-failure matrix for fan-out aggregation

Spec §14, narrowed to two phases by D-A. Verified against a real SLURM submission:
one array cohort, one dependent finalizer, no sidecar, master byte-identical to the
local path."
```

---

## Task 5: The rolling-input matrix

**Files:**
- Test: `tests/integration/test_rolling_input.py` *(new)*

Spec §14's third named test, and the scenario the whole design is shaped around — the
audit's running example is a 6,000-image run that grows. It spans identity (P2),
per-image proofs (P3), `finalize_run` (P4) and fan-out (P5), which is why it lands here
rather than earlier.

**The property under test:** per-image proofs survive an arrival; only aggregate-level
proofs invalidate.

- [ ] **Step 1: Write the matrix**

```python
@pytest.mark.parametrize(
    "scenario",
    [
        "batch_added_mid_run",
        "batch_added_between_runs",
        "metadata_arrives_later",
        "unready_file_present",
    ],
)
def test_only_aggregate_proofs_invalidate_when_the_input_grows(tmp_path, scenario):
    """Spec §14's rolling-input matrix.

    §9.2: adding 10 images to a 6,000-image run today re-derives the worklist by
    validating 6,000 markers, each re-hashing its measurements parquet and overlay
    PNG. After this change the 6,000 unchanged images cost one stat() each and the
    10 arrivals are deep-verified.

    D7 is the identity half of the same property: a new image changes
    inventory_digest but NOT processing_generation, so live progress is not reset
    and in-flight workers are not fenced.
    """
    from phenotypic.sdk_ import resolve_run_state

    output = _run_to_completion(tmp_path, n_images=6)
    before = {w: s.stages for w, s in resolve_run_state(output, depth="deep").images.items()}

    _apply(scenario, output)
    _run_again(output)

    after = resolve_run_state(output, depth="deep")
    for work_id, stages in before.items():
        assert after.images[work_id].stages == stages, (
            f"{scenario} invalidated an existing image's proof; only aggregate-level "
            "proofs may invalidate when scope changes"
        )
    assert after.completion == "complete"


def test_an_unready_file_is_not_accepted_into_the_inventory(tmp_path):
    """A file still being written must not enter work_ids -- once accepted, its
    absence of a proof reads as `incomplete` forever."""
    from phenotypic.sdk_ import resolve_run_state

    output = _run_to_completion(tmp_path, n_images=6)
    _write_partial_image(output_input_dir(output) / "still-copying.tif")
    _run_again(output)
    assert resolve_run_state(output, depth="deep").completion == "complete"


def test_metadata_arriving_later_re_runs_finalize_and_nothing_else(tmp_path):
    """§7.4, as narrowed by D-A. A metadata edit changes finalization_input_digest,
    so the next invocation re-joins the mirror. Stores keep the snapshot they were
    built against and report the divergence as an advisory (P1 Task 5)."""
    import polars as pl

    from phenotypic.sdk_ import measurements_parquet_path, resolve_run_state

    output = _run_to_completion(tmp_path, n_images=6, metadata=False)
    store_mtimes = _store_mtimes(output)

    _add_metadata_csv(output)
    _run_again(output)

    mirror = pl.read_parquet(measurements_parquet_path(output))
    assert "Metadata_Strain" in mirror.columns
    assert _store_mtimes(output) == store_mtimes, "a metadata edit rewrote a store"
    assert any("metadata" in a for a in resolve_run_state(output, depth="deep").advisories)
```

- [ ] **Step 2: Run it against a real local run.** Expected: PASS.

- [ ] **Step 3: Prove the key assertion can fail**

Fold `inventory_digest` into the generation digest (undoing D7); confirm
`test_only_aggregate_proofs_invalidate_when_the_input_grows[batch_added_between_runs]`
fails because every existing proof was invalidated. Restore. **D7 is the decision this
test defends, and an undefended decision drifts.**

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_rolling_input.py
git commit -m "test: rolling-input matrix -- arrivals invalidate scope, not proofs

Spec §14, §9.2, D7. Confirmed to fail when inventory_digest is folded into the
generation digest, which is the mistake D7 exists to prevent."
```
# Phase 6 — Consumer migration and the deletions

**Depends on:** P1–P5. **Blocks:** P7.

**Spec:** §11 (consumer migration), §11.1 (~1,400 lines deleted), §11.2 (folded in).

**Goal:** every consumer of the nine evidence sources calls `resolve_run_state` instead,
and the machinery they used is deleted. This is the phase that pays for the previous five.

**The deletions are the deliverable.** A task here that migrates a consumer without
deleting what it replaced has not finished — the failure mode this whole change addresses
is nine sources that each closed a real hole and none of which was ever removed.

---

## Deletion ledger

Track these in the phase's final commit body. Spec §11.1 estimates ~1,400 lines.

| # | Delete | Where | Task |
|---|---|---|---|
| 1 | `classify_output_consistency`, `OutputCompletionEvidence`, `inspect_output_consistency`, `OutputConsistencyReport` | `gui/results_viewer/_output_consistency.py` (617 lines, whole file) | 2 |
| 2 | `RunRegistry._processing_state_conflict`, `_publication_evidence_conflict`, `_orchestration_state_conflict` | `gui/shell/_runs_registry.py:1087,1202,1264` | 4 |
| 3 | `_local_completion_evidence_conflict`'s 8-branch tree | `_runs_registry.py:591` | 5 |
| 4 | `_latest_event_states` | `_runs_registry.py:1172` | 4 |
| 5 | `_read_status_from_manifest`, `_manifest_is_complete` | `_runs_registry.py`, `_slurm_observer.py` | 4, 6 |
| 6 | `DashboardManifestKey.VERSION` — written as `3` at one site, read at **zero** | `_dashboard/_manifest_builder.py:766` | 7 |
| 7 | `sdk_/monitor_slurm_jobs.py` — zero importers in `src/` or `tests/` | whole file (241 lines) | 7 |
| 8 | `browse/_source_render.py`'s `browse_cache_base` / `cache_png_path` / `init_cache` / `wipe_cache` — zero production callers | `_source_render.py:35-38` | 7 |
| 9 | Eight zero-caller resolvers in `_io_constants` | `_io_constants.py:2107` | 7 |
| 10 | Every `_legacy_*` helper and `resolve_*` fallback on the hot path | across `_cli` | P7 (they **move into** migrate, not away) |

Items 1–9 land here. **Item 10 is P7's** — spec §11.1 says legacy paths move *into*
`--mode migrate`, and deleting them before migrate can read them would strand every
existing tree.

---

## Task 1: `_snapshot_status.py` — 101 lines to ~30

**Files:**
- Modify: `src/phenotypic/gui/_snapshot_status.py`
- Test: `tests/unit/gui/test_snapshot_status.py`

`snapshot_refresh_status` currently branches on `inspect_output_consistency` plus
`snapshot_is_current()` plus `refresh_state_is_current()` — the last of which full-content
SHA-256s seven files on every 5–10 s tick (`_output_root.py:559`).

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize(
    "completion,refresh_supported,expected_color",
    [
        ("complete",   True,  "success"),
        ("incomplete", True,  "warning"),
        ("failed",     True,  "danger"),
        ("active",     True,  "warning"),
        ("complete",   False, "success"),
    ],
)
def test_the_badge_is_a_pure_function_of_completion(
    completion, refresh_supported, expected_color, fake_output_root
):
    """§11: ~30 lines mapping `completion` -> badge, replacing two fingerprints and
    a full re-hash of 7 files per poll."""
    from phenotypic.gui._snapshot_status import snapshot_refresh_status

    fake_output_root.run_state = _state(completion=completion)
    _label, color, _disabled = snapshot_refresh_status(
        fake_output_root, refresh_supported=refresh_supported
    )
    assert color == expected_color


def test_no_badge_refresh_hashes_a_deliverable(fake_output_root, monkeypatch):
    """The 5-10s tick currently full-content SHA-256s measurements.parquet,
    measurements.csv, pipeline.json, curation_labels.parquet, custom_categories.json,
    qc.duckdb and review_state.json. Per tab."""
    import hashlib

    calls = {"n": 0}
    real = hashlib.sha256
    monkeypatch.setattr(
        hashlib, "sha256", lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), real(*a, **k))[1]
    )
    snapshot_refresh_status(fake_output_root, refresh_supported=True)
    assert calls["n"] == 0
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`snapshot_refresh_status` takes `resolve_run_state(output_root.layout.output_root,
depth="shallow")` and maps `completion` → `(label, color, disabled)`. Delete
`_completion_evidence_status` entirely.

**Fold in audit S2 while here** — it is the same function and the same tick. `OutputRoot`'s
frozen `consumed_state_fingerprint` (`_output_root.py:882`) and `CurationLabels`'
self-updating `_source_fingerprint` (`_curation_labels.py:760`) hash overlapping path sets
with different lifecycles, so **marking one colony makes the viewer report its own write as
external drift** — the badge flips to `"Changed on disk"` / `danger`. Exclude GUI-owned
mutable paths from the snapshot fingerprint, the way `snapshot_is_current()` already
deliberately does (`_output_root.py:545-548`). Add a test that a curation click leaves the
badge `success`.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/_snapshot_status.py src/phenotypic/gui/results_viewer/_output_root.py \
        tests/unit/gui/test_snapshot_status.py
git commit -m "refactor(gui): the snapshot badge becomes a map over RunState.completion

Spec §11. Also fixes audit S2: marking one colony no longer makes the viewer report
its own write as external drift, and the tick stops re-hashing seven deliverables."
```

---

## Task 2: Delete `_output_consistency.py`

**Files:**
- Delete: `src/phenotypic/gui/results_viewer/_output_consistency.py` (617 lines)
- Modify: every importer
- Test: `tests/unit/gui/`

- [ ] **Step 1: Find every importer**

```bash
grep -rn '_output_consistency\|inspect_output_consistency\|classify_output_consistency\|OutputConsistencyReport\|OutputCompletionEvidence' src/ tests/
```

Expected from the pre-flight count: 4 in `src/`, 3 in `tests/`.

- [ ] **Step 2: Write the failing test that `contradictory` is unreachable**

```python
def test_contradictory_is_not_a_state_any_more():
    """Spec §4.3: `contradictory` is DELETED as a state.

    It exists today only because derived counts are cross-checked against each
    other. Once counts stop being evidence, two authorities cannot disagree: there
    is exactly one path to each verdict. This test is the thing that stops it coming
    back -- it was the source of 'run flagged read-only for a reason the user cannot
    act on', which is the user-visible bug this whole change is for.
    """
    import typing

    from phenotypic.sdk_._run_state import Completion

    assert set(typing.get_args(Completion)) == {
        "complete", "incomplete", "failed", "active"
    }


def test_no_module_still_imports_the_deleted_classifier():
    import subprocess

    hits = subprocess.run(
        ["grep", "-rn", "_output_consistency", "src/"],
        capture_output=True, text=True,
    ).stdout
    assert not hits, f"dangling importers of the deleted classifier:\n{hits}"
```

- [ ] **Step 3: Migrate the callers, then delete the file**

- `_snapshot_status.py` — done in Task 1.
- `OutputRoot.discover` (`_output_root.py:178`) — `resolve_run_state(depth="deep")`,
  **one** call replacing the current double read.
- `OutputMutationGuard` — `resolve_run_state(depth="deep")`; the mutation gate becomes
  `state.completion != "active"` rather than `report.is_read_only`.
- The processing-inventory cache's `cache_reusable` (`report.state == "coherent"`) becomes
  `state.completion == "complete"`.

Then `git rm src/phenotypic/gui/results_viewer/_output_consistency.py`.

**`core_readable` needs care.** `OutputConsistencyReport.core_readable`
(`_output_consistency.py:106`) is what the live-run test gate asks — a memory note in this
project records `skipif` must ask `core_readable`, not `is_dir`. Its replacement is
`state.completion in {"complete", "incomplete"}` **and** a valid aggregate proof. Grep for
`core_readable` in `tests/` and migrate each site deliberately; a gate that silently starts
returning `True` runs tests against a half-written tree.

- [ ] **Step 4: Run the GUI suite**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui tests/gui -q
```

`tests/gui` **is** in `testpaths`; `tests/e2e` is not and needs `PLAYWRIGHT=1`.

- [ ] **Step 5: Commit**

```bash
git rm src/phenotypic/gui/results_viewer/_output_consistency.py
git add -A src tests
git commit -m "refactor(gui): delete _output_consistency.py -- 617 lines, 9 sources, 23 rules

Spec §4.3, §11. `contradictory` is gone as a reachable state, and a test now stops
it coming back. Callers use resolve_run_state(depth=...)."
```

---

## Task 3: `OutputRoot` currency — one shallow verification

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_output_root.py:542,559,882`
- Test: `tests/unit/gui/results_viewer/`

- [ ] **Step 1: Write the failing test**

```python
def test_one_currency_check_replaces_two(fake_output_root):
    """§11: snapshot_is_current() + refresh_state_is_current() -> one shallow
    verification. Two overlapping fingerprints with different lifecycles is audit
    S2, and the fix is one owner, not two better-synchronised ones."""
    from phenotypic.gui.results_viewer._output_root import OutputRoot

    assert not hasattr(OutputRoot, "refresh_state_is_current")
    assert not hasattr(OutputRoot, "consumed_state_fingerprint")


def test_a_chmod_does_not_report_changed_on_disk(bound_output_root):
    """Audit S3, at the consumer. _inventory_is_current compares st_ctime_ns
    (_processing_inventory.py:462), which moves on chmod, chown, hardlink and
    rsync -a -- all routine on a shared HPC filesystem, and each one makes the whole
    binding report 'Changed on disk'."""
    for path in bound_output_root.root.rglob("*.parquet"):
        path.chmod(0o644)
    assert bound_output_root.is_current()
```

- [ ] **Step 2: Implement**

`snapshot_is_current()` delegates to `resolve_run_state(depth="shallow")`.
`refresh_state_is_current()` and `consumed_state_fingerprint` are deleted. Drop
`st_ctime_ns` from `_inventory_is_current` (`_processing_inventory.py:462`) — audit S3.

**Confirm no test depends on ctime-sensitivity before dropping it**
(`grep -rn 'ctime' tests/`), as audit S3 asks.

- [ ] **Step 3: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui tests/gui -q
git add -A src/phenotypic/gui tests
git commit -m "refactor(gui): one currency check, and drop ctime from the inventory sweep

Spec §11, audit S2 and S3. A chmod on GPFS no longer makes a binding report
'Changed on disk'."
```

---

## Task 4: `RunRegistry` claimability — 248 lines to one call

**Files:**
- Modify: `src/phenotypic/gui/shell/_runs_registry.py:1087,1172,1202,1264`
- Test: `tests/unit/gui/shell/`

- [ ] **Step 1: Write the failing tests**

```python
def test_claimability_is_one_resolve_call(fake_registry):
    """§11: three conflict predicates -> one resolve_run_state call."""
    from phenotypic.gui.shell import _runs_registry as reg

    for gone in (
        "_processing_state_conflict",
        "_publication_evidence_conflict",
        "_orchestration_state_conflict",
        "_latest_event_states",
        "_read_status_from_manifest",
    ):
        assert not hasattr(reg.RunRegistry, gone) and not hasattr(reg, gone), gone


def test_the_event_log_is_replayed_at_most_once(bound_registry, monkeypatch):
    """Audit S5 / §11: _latest_event_states reimplements aggregate_state_from_events
    with different semantics (stage demotion, no inventory fence). Two parsers of one
    append-only log will drift."""
    calls = {"n": 0}
    _count_event_log_reads(monkeypatch, calls)
    bound_registry.claimability("some-output")
    assert calls["n"] <= 1
```

- [ ] **Step 2: Implement**

Replace all three conflict predicates with `resolve_run_state(output_dir, depth="shallow")`
and a `completion`-based decision. Delete `_latest_event_states` and
`_read_status_from_manifest`.

**Where does the stage-demotion rule go?** Audit S5 proposes folding it into the CLI
aggregator as an option. Under §4.2 the event log is no longer evidence, so the demotion
has no consumer — verify that with a grep before deleting rather than after, and if
something still reads it, fold it in rather than dropping it silently.

- [ ] **Step 3: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui tests/gui -q
git add -A src/phenotypic/gui tests
git commit -m "refactor(gui): RunRegistry claimability becomes one resolve_run_state call

Spec §11, audit S5. Three conflict predicates and the second event-log replay are
deleted; one append-only log now has one parser."
```

---

## Task 5: `RunRegistry` local exit, plus DEFERRED D-2

**Files:**
- Modify: `src/phenotypic/gui/shell/_runs_registry.py:591,1058,1306`
- Test: `tests/unit/gui/shell/`

**This task includes an explicit scope addition.** DEFERRED **D-2** (stale
`gui_launch_owner.json` has no repair path) is folded in here, for two reasons stated in
the README: this task rewrites `_assert_output_claimable_locked`, the exact predicate that
causes the dead-end; and under the Q2 verdict ladder a stale owner record now masks
`incomplete` as `active`. DEFERRED itself recommends P6.

- [ ] **Step 1: Write the failing tests**

```python
def test_the_eight_branch_refusal_tree_becomes_advisories(fake_registry):
    """§11: '8-branch refusal tree -> resolve_run_state(deep); refusals become
    advisories.' A refusal the user cannot act on is the bug; an advisory they can
    read is the fix."""
    state = fake_registry.local_exit_state("some-output")
    assert state.completion in {"complete", "incomplete", "failed", "active"}
    assert isinstance(state.advisories, tuple)


def test_a_sigkilled_gui_does_not_lock_the_output_forever(tmp_path):
    """DEFERRED D-2 / audit S7 [verified]: nothing in the codebase ever deletes or
    repairs gui_launch_owner.json. A SIGKILLed GUI leaves status: 'running';
    rehydrate_from_sandbox downgrades it IN MEMORY ONLY (_runs_registry.py:773,
    persist=False), and _assert_output_claimable_locked then refuses the output
    forever, with no UI affordance to clear it.

    The record already stores pid and started_at. Use them."""
    _write_owner_record(tmp_path, status="running", pid=_a_dead_pid(), started_at="2020-01-01")
    registry = _registry(tmp_path)
    registry.assert_output_claimable(tmp_path)   # must not raise


def test_a_live_owner_still_refuses_the_claim(tmp_path):
    """The liveness check must not become a rubber stamp -- an owner whose process
    is alive still owns the output."""
    import os

    import pytest

    _write_owner_record(tmp_path, status="running", pid=os.getpid(), started_at="2026-09-03")
    with pytest.raises(RuntimeError):
        _registry(tmp_path).assert_output_claimable(tmp_path)
```

- [ ] **Step 2: Implement**

`_local_completion_evidence_conflict`'s eight refusal strings become advisories on the
`RunState`. `_assert_output_claimable_locked` gains a liveness check: an owner record whose
`status` is non-terminal but whose `pid` is not alive is downgraded **and persisted** — the
`persist=False` at `_runs_registry.py:773` is precisely what makes today's downgrade
useless.

Use `os.kill(pid, 0)` guarded for `ProcessLookupError` / `PermissionError`; a `pid` that
has been recycled is a real but bounded risk, and `started_at` bounds it further — treat a
record older than the boot time as dead regardless.

- [ ] **Step 3: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui tests/gui -q
git add -A src/phenotypic/gui tests
git commit -m "refactor(gui): local-exit refusals become advisories; repair a stale owner record

Spec §11 plus DEFERRED D-2, folded in deliberately: this task rewrites the exact
predicate that caused the permanent dead-end, and under the Q2 ladder a stale owner
record masks incomplete as active. The record already stored pid and started_at;
nothing read them."
```

---

## Task 6: SLURM observer — call sites only

**Files:**
- Modify: `src/phenotypic/gui/run_console/_slurm_observer.py:536,909,1312`
- Test: `tests/unit/gui/run_console/`

**Scope discipline is the whole of this task.** Spec §2.2 and DEFERRED D-1 put the
observer's decision tree, its 30-second reconciliation grace window and its `squeue`/`sacct`
state ranking **out of scope**. Only ~185 of its lines are filesystem-derived; the rest is
scheduler domain, it is the least testable code in the GUI, and its failure mode ("run
stuck in `reconciling`") is directly user-visible.

**Change exactly three things:**

1. the two `_cli_completion` call sites (`current_success_counts`, `valid_run_completion`) →
   one `resolve_run_state(depth="shallow")`
2. `_all_stage3_markers_exist` → reads `stages.stage3` from the P3 record
3. `_manifest_is_complete` → deleted

- [ ] **Step 1: Write the failing test**

```python
def test_the_observer_tick_does_not_hash_anything(fake_observer, monkeypatch):
    """Audit §4: the 2-second daemon tick currently runs valid_run_completion ->
    current_run_is_complete -> current_success_counts -> _walk_current_success, which
    calls valid_image_success once per image, each re-hashing the embedded
    measurements parquet AND the overlay PNG. Then current_aggregate_is_current walks
    it all AGAIN. On a 10,000-image run that is ~2-3 x 10^4 file hashes every two
    seconds."""
    import hashlib

    calls = {"n": 0}
    _count_sha256(monkeypatch, calls)
    fake_observer.tick()
    assert calls["n"] <= 8


def test_the_decision_tree_is_untouched():
    """Spec §2.2, DEFERRED D-1. This test exists to make scope creep fail CI rather
    than fail review."""
    import inspect

    from phenotypic.gui.run_console._slurm_observer import SlurmLifecycleObserver

    source = inspect.getsource(SlurmLifecycleObserver._observe_record)
    assert "resolve_run_state" in source
    assert "_manifest_is_complete" not in source
    # The grace window and squeue/sacct ranking stay exactly as they are.
    assert "GRACE" in source or "grace" in source
```

- [ ] **Step 2: Implement, run, commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/gui/run_console tests/gui -q
git add -A src/phenotypic/gui tests
git commit -m "refactor(gui): the observer tick asks resolve_run_state once

Spec §11, scoped by §2.2 and DEFERRED D-1: two call sites and the Stage-3 probe
move; the decision tree, grace window and scheduler polling are untouched. The 2s
tick stops walking every image twice and hashing every artifact."
```

---

## Task 7: The pure deletions, and §11.2's fold-ins

**Files:** as listed in the deletion ledger, items 6–9.

Each of these is **evidence-backed** — the audit verified the caller counts by hand. Re-run
each grep before deleting; a claim from 2026-09-03 is not a claim about the tree you are
editing.

- [ ] **Step 1: Verify each claim, then delete**

```bash
grep -rn 'monitor_slurm_jobs'        src/ tests/    # expect 0 outside the file itself
grep -rn 'DashboardManifestKey.VERSION' src/ tests/ # expect 1 write, 0 reads
grep -rn 'browse_cache_base\|cache_png_path\|init_cache\|wipe_cache' src/ tests/
grep -rn 'read_run_manifest\|load_master_measurements\|resolve_best_pipeline_path\|resolve_qc_dir\|recompile_status_dir\|chunk_parquet_path\|checkpoint_lock_path\|chunk_manifest_path' src/ tests/
```

**If a grep disagrees with the audit, stop and record it** — either the tree moved or the
audit was wrong, and both matter more than the deletion.

Two of the eight `_io_constants` resolvers "claim in their docstrings to replace inline
blocks that still exist" (S21). For those, route the inline block through the helper and
keep it, or delete both. Deleting the helper while the inline block survives is the worse
of the three outcomes.

- [ ] **Step 2: Fold in §11.2 — inside files already being rewritten**

Only these. Everything else in DEFERRED's churn table stays deferred:

- Hand-joined `.phenotypic/aggregate_publication.json` in the GUI → use
  `aggregate_publication_marker_path()` (audit S8, `_output_consistency.py:380` — the file
  Task 2 deleted, so this is now wherever its caller moved).
- The ~17 shadow state filenames into `_io_constants` (S9). Two of them —
  `staged_orchestration.json` and `staged_finalization_complete.json` — are **double-spelled
  across the CLI/GUI boundary**; those two are the ones that matter.
- `DIR_PROGRESS` at the two literal sites in `phenotypicCLI.py:839,841`, in the same file
  that already imports and uses it at `:943` (S15).
- The recompile `task_manifest.json` and `job_metadata.json` naive writers made atomic
  (S11) — these **are** polled by concurrently launched SLURM workers, including the
  unlocked write at `phenotypicCLI.py:3385`.

- [ ] **Step 3: Collapse the three `_canonical_digest` copies**

P1 Task 4 left three, pinned by a test. Now that `_run_state.py` exists and `sdk_` is the
shared layer, move it to `sdk_/_io_constants.py` (or a small `sdk_/_digests.py`) and have
all three import it. Delete the agreement test — it exists only while there are copies to
disagree.

- [ ] **Step 4: Phase gate — the full suite**

```bash
uv run mypy src/phenotypic
uv run ruff check --fix <every path this phase touched>
```

Then the full `tests/unit` **and** `tests/gui` suites, as a Slurm job via the
**`run-phenotypic-test`** and **`slurm-job`** skills. This is the phase most likely to break
something distant.

Compare against the recorded baseline: four failures are known pre-existing, three of which
fail only on compute nodes. **A fifth failure is this phase's**, not the baseline's.

- [ ] **Step 5: Commit with the deletion ledger**

```bash
git add -A
git commit -m "refactor: delete the nine-source completion machinery

Spec §11.1, §11.2. Net -<N> lines.

Deleted:
  _output_consistency.py                        -617
  RunRegistry three claimability predicates     -<n>
  _local_completion_evidence_conflict tree      -<n>
  _latest_event_states, _read_status_from_manifest, _manifest_is_complete  -<n>
  sdk_/monitor_slurm_jobs.py (0 importers)      -241
  browse/_source_render.py dead cache API       -<n>
  eight zero-caller _io_constants resolvers     -<n>
  DashboardManifestKey.VERSION (1 write, 0 reads) -<n>

Folded in (§11.2): aggregate_publication_marker_path at the GUI site; 17 shadow
filenames into _io_constants, including the two double-spelled across the CLI/GUI
boundary; DIR_PROGRESS at phenotypicCLI.py:839,841; the four naive control-manifest
writers made atomic.

Every caller count was re-grepped before deletion, not taken from the audit."
```
# Phase 7 — `--mode migrate` conversion, and refusal everywhere else

**Depends on:** P1–P6. **Blocks:** nothing.

**Spec:** D1 (clean break), §11.1 (legacy paths move *into* migrate), §15.1 (residual risk).

**Goal:** every existing tree can be converted, and every mode that is not `migrate`
refuses an unconverted one with a pointer to the command that fixes it.

> **§15.1: "The migrate step is the riskiest part of this design."** It rewrites machine
> state across the whole tree and, unlike the rest of the change, **cannot be rolled back by
> reverting code.** It needs the receipt/rollback discipline the existing metadata migration
> has, plus its own dry-run mode.

That risk is materially lower than the spec assessed, because
[D-A](OPEN-QUESTIONS.md#d-a-per-store-metadata-is-written-at-promote-time-not-backfilled)
removed the store rewrite: migrate converts `.phenotypic/` — markers, state, identity — and
**does not touch a single byte inside any `results/**/*.ome.zarr`**. A converted tree's
stores are the stores it already had.

---

## What migrate converts

| From | To | Task |
|---|---|---|
| `image_complete/<ds>/<stem>.json` (v2 marker) | `images/<ds>/<stem>.json` record, `stages={"measured": …}` | 2 |
| `stage2_done/<ds>/<stem>.json` | `stages.stage2` in the same record | 2 |
| `stage3_complete/<ds>/<stem>.json` | `stages.stage3` in the same record | 2 |
| `processing_generation: <uuid4>` | content-derived generation; `restart_epoch: 0` | 3 |
| `processing_state.datasets.{completed,failed,started}` | **deleted from the file** (§4.2) | 3 |
| `slurm_generation` / `lifecycle_epoch` | `scheduler_epoch` | 3 |
| joined embedded tables | **left alone** — see Task 4 | 4 |
| `master_measurements.csv` | deleted; master is parquet-only (D8) | 4 |
| legacy `.h5` per-image files | unchanged — the existing OME-Zarr migration already owns this | — |

**What migrate does NOT do:** re-mint `work_id` (D-C keeps the digest unchanged, so every
existing `work_id` stays valid), rewrite `deliverables/metadata.csv` (project `CLAUDE.md`,
spec D9/FLOW-4 — there is **no exception, including migrate**), or write into any store.

---

## Task 1: Detection and refusal

**Files:**
- Create: `src/phenotypic/_cli/_cli_schema_gate.py`
- Modify: `src/phenotypic/phenotypicCLI.py`
- Test: `tests/unit/cli/test_schema_gate.py` *(new)*

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("mode", ["full", "measure", "recompile", "process"])
def test_every_writing_mode_refuses_an_unconverted_tree(tmp_path, mode):
    """D1: clean break. New code reads only the consolidated schema, so a mode that
    silently half-read a legacy tree would produce a run whose proofs certify
    nothing."""
    import pytest

    _build_legacy_tree(tmp_path)
    with pytest.raises(SystemExit) as exc:
        _invoke_cli(mode=mode, output=tmp_path)
    assert "--mode migrate" in str(exc.value), (
        "the refusal must name the command that fixes it; a refusal the user cannot "
        "act on is the bug class this whole change exists to remove"
    )


def test_a_converted_tree_is_accepted(tmp_path):
    _build_legacy_tree(tmp_path)
    _invoke_cli(mode="migrate", output=tmp_path)
    _invoke_cli(mode="full", output=tmp_path)      # must not raise


def test_a_fresh_output_directory_is_not_an_unconverted_tree(tmp_path):
    """An empty directory has no schema to be wrong about. Refusing one would make
    every new run start with an error."""
    _invoke_cli(mode="full", output=tmp_path / "brand-new")


def test_the_gui_reports_rather_than_refuses(tmp_path):
    """§4.3: a half-migrated tree is an ADVISORY, not a gate. The GUI is a reader;
    refusing to display a legacy tree would be a regression from today, where it
    displays one."""
    from phenotypic.sdk_ import resolve_run_state

    _build_legacy_tree(tmp_path)
    state = resolve_run_state(tmp_path, depth="deep")
    assert any("migrate" in advisory for advisory in state.advisories)
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

```python
STATE_SCHEMA_VERSION = 3


def requires_conversion(output_dir: Path) -> str | None:
    """Return why *output_dir* needs `--mode migrate`, or ``None``.

    Detection is by **presence of the old shape**, never by absence of the new one:
    an empty directory and a half-written new run both lack ``images/``, and only one
    of them is a legacy tree. The signals, in order of certainty:

    1. ``.phenotypic/progress/image_complete/`` exists
    2. ``.phenotypic/progress/stage2_done/`` or ``stage3_complete/`` exists
    3. ``processing_state.json`` carries ``datasets.completed`` (deleted by §4.2)
    4. ``processing_state.json`` has no ``restart_epoch`` **and** has ``work_ids``

    Returns:
        A message naming the specific evidence and the command, or ``None``.
    """
```

Wire it into `phenotypicCLI.py` for `full`, `measure`, `recompile` and `process`. **Not**
for the GUI or any reader — spec §4.3 makes a half-migrated tree an advisory.

- [ ] **Step 4: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli/test_schema_gate.py -v
git add src/phenotypic/_cli/_cli_schema_gate.py src/phenotypic/phenotypicCLI.py \
        tests/unit/cli/test_schema_gate.py
git commit -m "feat(cli): refuse an unconverted tree in every writing mode

D1. Detection is by presence of the old shape, not absence of the new -- an empty
directory is not a legacy tree. Readers get an advisory, not a refusal (§4.3)."
```

---

## Task 2: Convert the per-image markers

**Files:**
- Modify: `src/phenotypic/_cli/_cli_migrate.py`
- Create: `src/phenotypic/_cli/_cli_migrate_state.py`
- Test: `tests/unit/cli/test_migrate_state.py` *(new)*

- [ ] **Step 1: Write the failing tests**

```python
def test_three_markers_become_one_record(tmp_path):
    from phenotypic._cli._cli_migrate_state import convert_per_image_markers
    from phenotypic.sdk_ import image_record_path

    _plant_legacy_markers(tmp_path, dataset="plate", stem="a",
                          image_complete=True, stage2_done=True, stage3_complete=True)
    convert_per_image_markers(tmp_path)

    record = json.loads(image_record_path(tmp_path, "plate", "a").read_text())
    assert set(record["stages"]) == {"stage2", "stage3", "measured"}
    assert record["work_id"] == "w"


def test_artifact_descriptors_survive_conversion_byte_for_byte(tmp_path):
    """The descriptors are the content proof. Re-deriving them during migration
    would certify whatever is on disk NOW, including a corrupted artifact -- which
    turns migrate from a format change into a laundering step."""
    _plant_legacy_markers(tmp_path, dataset="plate", stem="a", image_complete=True)
    before = json.loads(_legacy_marker_path(tmp_path, "plate", "a").read_text())["artifacts"]

    from phenotypic._cli._cli_migrate_state import convert_per_image_markers

    convert_per_image_markers(tmp_path)
    after = json.loads(image_record_path(tmp_path, "plate", "a").read_text())["artifacts"]
    assert after == before


def test_conversion_is_idempotent(tmp_path):
    """Re-running after an interruption is the recovery procedure -- the CLI's
    documented migrate contract."""
    from phenotypic._cli._cli_migrate_state import convert_per_image_markers

    _plant_legacy_markers(tmp_path, dataset="plate", stem="a", image_complete=True)
    convert_per_image_markers(tmp_path)
    first = image_record_path(tmp_path, "plate", "a").read_bytes()
    convert_per_image_markers(tmp_path)
    assert image_record_path(tmp_path, "plate", "a").read_bytes() == first


def test_a_stage2_token_with_no_image_complete_still_converts(tmp_path):
    """Stage 2 finished and Stage 3 never ran -- a real interrupted-run state, and
    the one a naive 'iterate image_complete/' conversion drops on the floor."""
    from phenotypic._cli._cli_migrate_state import convert_per_image_markers
    from phenotypic.sdk_ import image_record_path

    _plant_legacy_markers(tmp_path, dataset="plate", stem="a",
                          image_complete=False, stage2_done=True)
    convert_per_image_markers(tmp_path)
    record = json.loads(image_record_path(tmp_path, "plate", "a").read_text())
    assert set(record["stages"]) == {"stage2"}


def test_the_legacy_trees_are_removed_only_after_every_record_is_written(tmp_path):
    """Marker-last, applied to the migration itself. A conversion that deletes as it
    goes and then dies leaves a tree that is neither shape."""
    from phenotypic._cli._cli_migrate_state import convert_per_image_markers

    _plant_legacy_markers(tmp_path, dataset="plate", stem="a", image_complete=True)
    _plant_legacy_markers(tmp_path, dataset="plate", stem="b", image_complete=True)
    with _fail_after_n_records(1):
        with pytest.raises(RuntimeError):
            convert_per_image_markers(tmp_path)
    assert _legacy_marker_path(tmp_path, "plate", "a").exists(), (
        "the legacy tree was removed before conversion completed"
    )
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

Enumerate the union of all three legacy trees, not just `image_complete/` — an image with
a stage-2 token and no completion marker is a real interrupted state. Write every record
first, then remove the three legacy trees. **Copy `artifacts` verbatim**; never re-derive.

- [ ] **Step 4: Run and commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli/test_migrate_state.py -v
git add -A src/phenotypic/_cli tests/unit/cli
git commit -m "feat(cli): migrate converts the three marker trees into one record

Spec §6.1, §11.1. Artifact descriptors are copied verbatim -- re-deriving them would
certify whatever is on disk now, turning migrate into a laundering step."
```

---

## Task 3: Convert `processing_state.json`

**Files:**
- Modify: `src/phenotypic/_cli/_cli_migrate_state.py`
- Test: `tests/unit/cli/test_migrate_state.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_the_uuid_generation_becomes_content_derived(tmp_path):
    from phenotypic._cli._cli_migrate_state import convert_processing_state

    _plant_legacy_state(tmp_path, processing_generation="deadbeef" * 4)
    convert_processing_state(tmp_path)
    config = _state_config(tmp_path)
    assert config["restart_epoch"] == 0
    assert config["processing_generation"] != "deadbeef" * 4
    assert config["processing_generation"] == _expected_generation(config)


def test_the_derived_dataset_counts_are_removed(tmp_path):
    """§4.2: processing_state.datasets.{completed,failed,started} is DELETED from
    the file -- it was already re-aggregated from the event log on every load
    (_cli_state_management.py:121), a cache of a cache."""
    from phenotypic._cli._cli_migrate_state import convert_processing_state

    _plant_legacy_state(tmp_path)
    convert_processing_state(tmp_path)
    for dataset in _state(tmp_path)["datasets"].values():
        assert not {"completed", "failed", "started"} & set(dataset)


def test_work_ids_are_untouched(tmp_path):
    """D-C keeps processing_configuration_digest unchanged, so every existing
    work_id stays valid. Re-minting them would invalidate every marker migrate just
    converted, and a tree half-migrated across that boundary is unrecoverable
    without the original config."""
    from phenotypic._cli._cli_migrate_state import convert_processing_state

    _plant_legacy_state(tmp_path)
    before = _state_config(tmp_path)["work_ids"]
    convert_processing_state(tmp_path)
    assert _state_config(tmp_path)["work_ids"] == before


def test_the_metadata_snapshot_is_byte_unchanged_by_a_full_migrate(tmp_path):
    """Project CLAUDE.md, spec D9/FLOW-4. There is NO exception, including migrate.

    An earlier draft of this rule carved one out -- migrate would rewrite
    deliverables/metadata.csv with canonical headers after copying the original to
    metadata.original.csv. That was WITHDRAWN and never implemented: a snapshot that
    is sometimes rewritten is not provenance, and 'the original is recoverable over
    there' is a weaker guarantee than 'the bytes you supplied are still the bytes on
    disk'. metadata.original.csv does not exist and must not be created."""
    _plant_legacy_tree(tmp_path, metadata=b"Well,Strain\r\nA1,BY4741\r\n")
    before = (tmp_path / "deliverables" / "metadata.csv").read_bytes()
    _invoke_cli(mode="migrate", output=tmp_path)
    assert (tmp_path / "deliverables" / "metadata.csv").read_bytes() == before
    assert (tmp_path / "deliverables" / "metadata.canonical.csv").is_file()
    assert not (tmp_path / "deliverables" / "metadata.original.csv").exists()
```

- [ ] **Step 2: Run to verify failure, implement, re-run.**

- [ ] **Step 3: Commit**

```bash
git add -A src/phenotypic/_cli tests/unit/cli
git commit -m "feat(cli): migrate converts processing_state.json to the v3 schema

Spec §4.2, §5.1. work_ids are untouched (D-C); metadata.csv is byte-identical, with
the canonical view emitted alongside it."
```

---

## Task 4: The embedded-table question, and the master's schema stamp

**Files:**
- Modify: `src/phenotypic/_cli/_cli_migrate_state.py`
- Test: `tests/unit/cli/test_migrate_state.py`

Existing trees have embedded tables that are **already metadata-joined** — the shape P4
inverted. Two options, and the plan takes the cheaper one:

**Decision: migrate leaves embedded tables alone and stamps the master.**

Rewriting every store's embedded table would reintroduce exactly the post-proof store
mutation D-A removed — the hardlink re-promote, the receipts, all of it — for trees that
already have a correct master. Instead, `finalize_run`'s step 1 projects each embedded
table onto its recorded `phenotypic.measurement_columns` before concatenating. That
projection is free, it is the same boundary P4 uses, and the column list is already written
into every existing store.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_legacy_joined_embedded_table_projects_to_measurements_only(tmp_path):
    """The projection uses the store's own recorded measurement_columns, so a
    pre-inversion store aggregates to the same master a post-inversion one does --
    with no store rewrite."""
    import polars as pl

    from phenotypic._cli._cli_finalize_run import finalize_run
    from phenotypic.sdk_ import master_measurements_parquet_path

    _build_legacy_tree_with_joined_embedded_tables(tmp_path)
    _invoke_cli(mode="migrate", output=tmp_path)
    finalize_run(tmp_path, dataset_names=["plate"])

    master = pl.read_parquet(master_measurements_parquet_path(tmp_path))
    assert "Metadata_Strain" not in master.columns
    assert "Metadata_ImageFile" in master.columns


def test_migrate_writes_no_byte_into_any_store(tmp_path):
    """D-A, applied to migrate. This is what made §15.1's 'riskiest part' claim
    smaller: migrate converts .phenotypic/ and nothing else."""
    _build_legacy_tree_with_joined_embedded_tables(tmp_path)
    stores = sorted((tmp_path / "results").rglob("*.ome.zarr"))
    before = {
        p: p.stat().st_mtime_ns
        for store in stores for p in sorted(store.rglob("*")) if p.is_file()
    }
    _invoke_cli(mode="migrate", output=tmp_path)
    after = {
        p: p.stat().st_mtime_ns
        for store in stores for p in sorted(store.rglob("*")) if p.is_file()
    }
    assert before == after


def test_the_master_carries_a_schema_version_an_old_reader_fails_on(tmp_path):
    """§7.3: 'anything filtering master on a user-metadata column would return EMPTY
    rather than error. This is the one genuinely dangerous failure mode in §7, and
    it is why the migrate step must tag the master with a schema version so an old
    reader fails loudly.'"""
    import pyarrow.parquet as pq

    from phenotypic.sdk_ import master_measurements_parquet_path

    _build_legacy_tree(tmp_path)
    _invoke_cli(mode="migrate", output=tmp_path)
    schema = pq.read_schema(master_measurements_parquet_path(tmp_path))
    assert schema.metadata[b"phenotypic.master_schema_version"] == b"2"
```

- [ ] **Step 2: Implement, run, commit**

```bash
git add -A src/phenotypic tests/unit/cli
git commit -m "feat(cli): project legacy embedded tables instead of rewriting them

D-A applied to migrate: not one byte is written into any store. finalize_run
projects each table onto its own recorded measurement_columns, so a pre-inversion
store aggregates identically. The master is stamped master_schema_version=2 so a
reader that filters on a user-metadata column fails loudly instead of silently
returning nothing (§7.3)."
```

---

## Task 5: Dry-run and rollback — the phase gate

**Files:**
- Modify: `src/phenotypic/_cli/_cli_migrate_state.py`, `phenotypicCLI.py`
- Test: `tests/unit/cli/test_migrate_state.py`, `tests/integration/`

§15.1 requires "the receipt/rollback discipline the existing metadata migration has, plus
its own dry-run mode".

- [ ] **Step 1: Write the failing tests**

```python
def test_dry_run_reports_and_writes_nothing(tmp_path):
    _build_legacy_tree(tmp_path)
    before = _tree_fingerprint(tmp_path)
    report = _invoke_cli(mode="migrate", output=tmp_path, dry_run=True)
    assert _tree_fingerprint(tmp_path) == before
    assert "2 records" in report and "processing_state.json" in report


def test_an_interrupted_migration_is_resumable_not_corrupt(tmp_path):
    """Re-running after an interruption is the recovery procedure. The tree between
    attempts must be readable by SOMETHING -- either the old shape or the new one,
    never neither."""
    _build_legacy_tree(tmp_path)
    with _fail_midway():
        with pytest.raises(RuntimeError):
            _invoke_cli(mode="migrate", output=tmp_path)
    assert _tree_is_readable_by_one_schema(tmp_path)
    _invoke_cli(mode="migrate", output=tmp_path)
    _invoke_cli(mode="full", output=tmp_path)


def test_the_pre_existing_metadata_receipt_path_still_raises_on_uncertified_drift(tmp_path):
    """INV-IMMUTABLE's one exception, kept scoped. D-A cut §6.4's GENERALISATION,
    not refresh_success_markers_after_metadata_migration itself -- it serves a real
    historical case and it keeps its RuntimeError."""
    import pytest

    from phenotypic._cli._cli_completion import (
        refresh_success_markers_after_metadata_migration,
    )

    _build_tree_with_uncertified_artifact_drift(tmp_path)
    with pytest.raises(RuntimeError):
        refresh_success_markers_after_metadata_migration(tmp_path)
```

- [ ] **Step 2: Implement, run.**

- [ ] **Step 3: Phase gate — a real tree**

Run `--mode migrate` against a **real legacy output tree on GPFS**, not a fixture, via the
**`slurm-job`** skill. Then:

```bash
uv run python -m phenotypic --mode full --output <migrated> ...   # must resume, not restart
uv run phenotypic-gui --root <migrated>                            # must bind and display
```

Confirm the migrated run resumes without reprocessing a single image — if it reprocesses,
`work_id` changed somewhere and D-C was violated.

- [ ] **Step 4: Full suite and docs**

```bash
uv run mypy src/phenotypic
uv run ruff check --fix <every path this phase touched>
```

Full `tests/unit` + `tests/gui` as a Slurm job, compared against the recorded baseline of
four known pre-existing failures.

Update:
- `CLAUDE.md`'s `--mode migrate` bullet — it now also converts run-state schema v2 → v3
- `docs/source/how_to/pages/migrate_ome_zarr.md`
- `src/phenotypic/_cli/CLAUDE.md`'s master-vs-mirror rules, if P4 left anything

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(cli): migrate gains a dry run and resumable conversion

Spec §15.1. Verified end to end on a real legacy tree on GPFS: migrate, then a full
run that resumes without reprocessing an image, then a GUI bind. The pre-existing
metadata receipt path keeps its RuntimeError for uncertified artifact drift --
INV-IMMUTABLE's one exception, still scoped to migrate."
```

---

## Closing the change

- [ ] **Final: confirm the headline claims, with numbers**

Do not write any of these from the spec. Measure each:

```bash
git diff --stat main...HEAD | tail -1                  # net lines
grep -c '' src/phenotypic/gui/results_viewer/_output_consistency.py 2>/dev/null || echo "deleted"
```

- **9 sources → 3 authorities** — `grep -rn 'resolve_run_state' src/ | wc -l` versus the
  count of files that previously read a completion source.
- **14 tokens → 6** — list the six and grep that nothing else reaches disk as identity.
- **~1,400 lines deleted** — the real `git diff --stat`, not the estimate.
- **The completion predicate's cost** — re-run spike S-5 against a migrated tree and quote
  the before/after.

State the measured numbers in the PR body. A claim in a PR that came from the spec rather
than from the tree is the same class of error this change spent seven phases removing.
