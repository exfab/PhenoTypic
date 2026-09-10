# Standalone Optuna Journal Backend Design

## Purpose

Distributed Optuna tuning must work without an MCP or service-layer owner. A
Slurm submission writes one shared journal on GPFS, publishes the same durable
deliverables as a local run, and can be safely finalized again after an
interruption. The existing `ome-zarr-merged` image loading and shared Slurm
lifecycle are authoritative.

## User Interface

The installed entry point is `phenotypic-tune`; `python -m phenotypic.tune`
remains equivalent, and a bare tuning-spec positional still normalizes to the
`run` subcommand.

`phenotypic-tune run` accepts `--slurm` with no value or repeatable
`--slurm KEY=VALUE` occurrences. Legacy tune Slurm flags remain compatible.
Explicit key/value occurrences override legacy values. Merging is by rendered
SBATCH directive identity, so aliases cannot emit duplicate directives.

Storage precedence is CLI, tuning spec, environment, then mode default. Local
Optuna defaults to a run-local SQLite database. Slurm Optuna defaults to an
absolute run-local `journal://` URL. Generated Journal URLs use the exact
`?v=1` query marker and canonical percent encoding for path data. A marked URL
is accepted only when decoding once and re-encoding produces its exact original
path spelling; malformed escapes, percent-encoded `/` or `\\` separators, and
alternate encodings fail closed. Windows path separators normalize to `/`
before encoding. Canonical generation rejects a literal POSIX backslash
filename because it would become a separator on a Windows worker; literal
percent text such as `%5C` remains representable as `%255C`. Unmarked legacy
URLs preserve raw percent text literally, while retaining the historic
rejection of every raw query or fragment. Any query
other than the exact version marker is invalid.
Slurm plus explicit SQLite, password-bearing URLs, and screen plus Slurm are
rejected before run artifacts are created.

`phenotypic-tune finalize OUTPUT [--force]` republishes an existing distributed
run. It never creates a missing journal/SQLite file or initializes an empty
external RDB schema. A non-creating RDB open requires Optuna's complete current
table catalog plus populated version and Alembic authority, and disables table
creation in the Optuna constructor. Finalization exits nonzero when the trial
budget is incomplete or no valid terminal winner exists.

## Journal and Winner Semantics

The Optuna journal backend is selected by URL scheme and uses shared-filesystem
locking suitable for GPFS. Retriable open/write failures use bounded retry.
Recovery repairs a torn terminal tail before appending; it never joins a partial
record to a new record. A read-only finalizer refuses missing or corrupt storage
rather than constructing an empty study.

Terminal trial accounting is explicit. `RUNNING` and `WAITING` trials left by
live or dead workers neither consume the completed budget nor become winners.
`FAILED` trials remain diagnostic-only and do not consume that budget. A
terminal `PRUNED` trial consumes budget and remains exported terminal history,
including when it carries a finite partial-fidelity objective. Only finite
scalar `COMPLETE` trials, and multi-objective `COMPLETE` trials whose entire
vector contains exactly the ordered scorer-declared
`objective_names(spec.scorer)` axes, unique by exact spelling and Unicode
casefold, with finite values, may participate in any
published scalar winner,
per-axis winner, Pareto front, knee-point selection, or per-axis importance
model. Duplicate or empty axis names, fewer than two axes on a scorer marked
multi-objective, and unsafe filename components fail closed before study
creation or publication. Safe components reject absolute, drive, UNC, dot, and
traversal semantics; Windows-reserved characters and device names; trailing
dots or spaces; NUL; and every C0/C1 control character. Direct
`OptunaStrategy` and `OptunaStudyStore` construction use the same ordered,
unique-axis validator as scorer inference. A COMPLETE multi-objective result
must supply exactly those keys before user attributes, native state, or
persistent storage can mutate. Native Optuna vectors are always serialized by
this scorer-authoritative order; result-dictionary insertion order is never
semantic.
Partial-fidelity scores are not comparable with complete evaluations. If
a multi-objective study has no such
Pareto candidate, finalization fails closed before publishing winner artifacts;
study identity never falls back to the scalar projection. The expected terminal
budget is recorded in the run metadata and checked before publication.


`TuningEngine` supplies authoritative axes through a backward-compatible
`StrategyConfig.build_with_objectives` hook. Its default delegates to the
historic `build(space, store, *, directions=...)` signature, so third-party
subclasses do not need to accept a new keyword.
## Distributed Ownership

A Slurm run claims one lifecycle generation before writing its resolved spec,
split, run marker, or constructing/mutating the shared study. A rejected
concurrent rerun leaves the incumbent generation and every shared artifact and
study unchanged. Tune worker, finalizer, and dispatcher scripts use
generation-scoped directories keyed by a fixed-format SHA-256 digest, so every
accepted generation is contained beneath the script root and distinct raw
generation strings cannot alias. A stale submitter can only prepare its own
scripts before lifecycle-fenced submission. The tune worker array is routed
dispatcher. One terminal `afterany` finalizer is submitted with the worker
compute profile and interpreter but no array directive. It is not a parallel
scheduler sidecar.

Automatic publication follows this state transition:

1. Acquire the exact-generation publication guard.
2. Reject a stale or superseded generation without changing the current owner.
3. Open the recorded backend without create semantics.
4. Validate terminal budget and winner, then publish trials, best parameters,
   best pipeline, screening, Pareto, and held-out generalization artifacts.
5. Require successful best-parameter publication and close the same generation
   while the guard is still held.

An owned submission or finalization failure records that generation failed and
inactive, then exits nonzero. Failure handling is generation-fenced so a stale
job cannot fail a successor.

Manual publication first reads the run marker and requires the current,
supported study-name constant. A missing marker, missing or legacy study name,
or mismatched study name fails before cancellation, lifecycle mutation,
lifecycle-lock creation/open, storage open, or publication output mutation.
Only after that read-only identity preflight does publication apply generation
fencing and hold the lifecycle lock from ownership validation through all
writes. Without `--force`, any active generation is refused. With `--force`, the
command cancels the recorded generation and requires a quiescent result with no
unresolved tokens; force never skips that proof. A generation-bearing marker
requires readable lifecycle authority for exactly that generation before
cancellation and again before publication. Missing, corrupt, or mismatched
authority fails without artifact mutation. A generation-less marker with the
current study name retains its compatibility path. A second successful
publication produces byte-identical artifacts.

## Image and GUI Integration

Tune input discovery retains the target branch's canonical file and
`.ome.zarr` directory discovery. Both the submitter and each worker load stores
through `load_image_from_store(..., fallback="GridImage")`; journal changes do
not replace or duplicate that path.

The tune Monitor performs bounded storage reads. Parameter-importance/fANOVA
work is outside the storage polling deadline so expensive analysis cannot turn
a healthy journal into a permanent timeout/backlog state. User-facing GUI copy
commands use `uv run phenotypic-tune`; internal subprocesses may use the module
entry point for interpreter fidelity.

## Failure and Safety Contract

- Missing, corrupt, or incomplete storage fails closed.
- Every backend's non-creating open validates backing storage before invoking an
  Optuna constructor that could initialize it.
- Automatic and manual publication are mutually excluded by the lifecycle lock.
- Forced recovery publishes only after scheduler quiescence is proven.
- A generation-bearing marker cannot publish without matching lifecycle authority.
A current run marker must also carry exactly the supported study identity;
missing, legacy, or mismatched names fail before storage is opened.
- Stale finalizers cannot close, fail, or publish over a newer generation.
- Submission failure after lifecycle initialization terminalizes that generation.
- OME-Zarr inputs are read-only; the move does not alter stores or migration.
- No `_services`, MCP/FastMCP code, service tests, or MCP design artifacts are
  part of this feature.

## Verification Contract

Unit tests cover URL dispatch and precedence, journal recovery, terminal winner
selection, Slurm argument merging, lifecycle ownership, publication idempotence,
Monitor timeouts, and real OME-Zarr loading. Full regression runs as a massively
parallel Slurm array. Separate acceptance jobs prove four-node GPFS journal
concurrency and the native `phenotypic-tune` worker-array plus terminal-finalizer
flow.

No executable numeric-validation script is required: this design introduces no
load-bearing numerical claim, and the required Optuna/PhenoTypic concurrency
harness is an integration test rather than a stdlib/NumPy/SciPy invariant.
