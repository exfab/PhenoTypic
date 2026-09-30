# The SLURM finalizer as a chain of jobs

Date: 2026-09-30. Status: implemented on `claude/amazing-hopper-f0f93f`.

## Problem

A large SLURM run's finalizer exceeded the walltime of the one job it ran in.

P5 (`docs/superpowers/plans/2026-09-03-cli-gui-state-tracking/phase-5-fanout.md`)
had split only the embedded-table read. The ordinary finalizer was one `0-K`
array whose indices `0..K-1` wrote aggregation shards and whose index K
(`_cli_checkpoint_handler._run_finalize`) did everything else under the same
`--time` as a per-image worker:

1. waited up to 600 s for the event log;
2. waited up to 3600 s for its own shards, which the scheduler starts
   alongside it, so the shards' wall-clock was charged to index K;
3. built and wrote the master, joined metadata, applied post ops, wrote the
   mirror, rendered every plot, fitted the named analysis, rebuilt QC, and
   wrote the per-feature and per-category splits;
4. published the aggregate proof, rebuilt the display manifest (a per-image
   verification pass) and the dashboard, and published the completion marker.

`shard_count` gives K = 1 below about 34,600 images, so in practice the split
was one shard task beside one monolithic finalizer. Two other SLURM finalizers
had no split at all:

- The **staged GPU** finalizer was a one-task job that read every store
  directly.
- The **recompile** finalizer was one `TASK_FINALIZE` array entry, and it is
  also the recovery path: re-running an ordinary run whose finalizer died goes
  through `_handle_recompile_slurm`.

## Design

`_cli_finalize_chain.py` runs the finalization as dependent jobs, each with its
own walltime and each `afterany` on the one before:

| Job | Tasks | Work |
|---|---|---|
| `prepare` | 1 | Wait for (ordinary) or reconcile (staged) images, then submit the rest of the chain. |
| `shards` | K | The unchanged P5 shard worker. Omitted on recompile, whose measurement tasks are its shards. |
| `master` | 1 | `publish_master_and_mirror`: master, metadata join, post ops, mirror, REMBI, `pipeline.json`. Writes `handoff.json`. |
| `outputs` | 2 | `publish_finalization_outputs`. Task 0 is `analysis` (measurement plots, analysis fit, analysis plots). Task 1 is `tables` (splits, error re-emit). |
| `qc` | 1 | The `qc` group. It runs after `outputs` because a QC plot may read a named analysis table. |
| `publish` | 1 | Aggregate proof, then the manifest, dashboard, staged report and README, and the completion decision; or recompile's tail. |

The accepted cost is one Python start-up per job (user ruling, 2026-09-30).

### Submission

The slot that submitted the single finalizer now submits `prepare`: the
drip-feed `finalizer_script` on ordinary runs and recompile, and the staged
controller's `"finalizer"` token. `prepare` submits everything else through
`submit_with_lifecycle`, which is ledgered, runs once per `finalize-<stage>`
token, and is fenced by the lifecycle generation. Because `prepare` runs only
after the image arrays are terminal, no chain job is queued beside an active
ordinary array, which is what the project's scheduler-sidecar rule requires.

On a staged run, `prepare` also sets the orchestration's `active_job_id` to
the `publish` job and retargets the pending recovery controller to it.
Without that step, the controller would run as soon as `prepare` exited and
mark the epoch `failed` while the chain was still queued.

### Between jobs

In-memory frames do not cross a job boundary. `outputs` and `qc` read the
master and the mirror back from disk. `master` records the SHA-256 of both in
`handoff.json`, and every later job refuses bytes that differ. `publish`
checks again under `.aggregate_publication.lock` before it writes the proof.

### Failure

Each task writes `status/<stage>_<i>.json`. A task killed at its walltime
writes none, and a missing status is read as a failure. `publish` always runs.
If any task failed, it publishes neither proof, closes the lifecycle (ordinary:
deactivated; staged: `failed`; recompile: finalizer status `failed`), and exits
non-zero with a message that names the task and its log. The run then reads
`incomplete`. Re-running the same command recovers, through recompile, which
uses the same chain (user ruling: an output-job failure marks the run
incomplete rather than complete with a warning).

### Resources

Every chain job uses the run's `--slurm` profile, and the CPU profile on a
staged run. No separate finalizer profile was added (user ruling). What
changes is that each job gets that walltime to itself.

## Decisions and their limits

- **Which step dominates is not measured.** No log from the job that timed
  out was available. The split isolates every step the code shows can scale
  with the image count. If one job still exceeds the walltime on its own, the
  most likely candidates are `master` at very large N and the verification
  passes in `publish`. That would need further splitting or less work, which
  is out of this change's scope.
- **`--wait` on an ordinary run still aggregates in the submitting process as
  well.** This predates the change. It remains redundant rather than
  divergent, because both writers produce the same bytes. The handoff digest
  now depends on that sameness: a divergent rewrite would leave the run
  incomplete rather than certify the wrong bytes.
- **`_run_finalize` is kept.** It is the same three helpers composed in one
  job. A run submitted before the chain existed calls it from a finalizer
  script that is already written.

## Verification

- `tests/unit/cli/test_finalize_chain.py` drives each stage against a real
  tree. It covers submission order and dependencies, a finished chain that
  certifies the run, byte identity with one in-process `finalize_run`, a
  killed output task, a missing shard, a mirror rewritten between jobs, a
  cancelled generation, and the staged hand-off. Two deliberate breakages were
  each caught.
- `tests/integration/cli/test_finalize_chain_fake_slurm.py` runs the real CLI
  against a fake scheduler that honours dependencies (`tests/_fakes/fake_slurm.py`).
  It covers an ordinary full run, a recompile with `--wait`, and a staged GPU
  run with `--wait`. The staged test fails when the controller hand-off is
  removed.
- `tests/unit/cli/test_array_auxiliary_routing.py` pins `prepare` as the
  chain's only submission site.
