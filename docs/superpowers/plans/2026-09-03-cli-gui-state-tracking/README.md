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
