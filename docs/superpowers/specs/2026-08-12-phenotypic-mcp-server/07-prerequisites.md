# PhenoTypic MCP Server — §7 Prerequisites and Rollout

Status: **draft, reviewed once, revised — P1 is significantly larger than first estimated**
Date: 2026-08-12

Five changes land before the MCP server is fully useful. P1 is a substantial
engine change to `phenotypic.tune` needing its own spec; P2–P5 are smaller.

---

## P1 — NFS-safe `JournalStorage` backend for distributed tune

### The problem

A SLURM tune fleet needs storage every worker can write. Today both options are
unsatisfactory for an agent-driven workflow:

| Today | Problem |
|---|---|
| `sqlite:///<output>/.pht-tune-cache/study.db` (the default) | Documented-unsafe across nodes — `tune_distributed_hpcc.md:8-12`: "SQLite-WAL is unsafe on the network filesystems HPCC clusters use (NFS, Lustre) … several SLURM array workers writing the same `study.db` will corrupt it or lose trials." |
| `postgresql+psycopg://…` | Requires a user-space Postgres server as its own sbatch job, a `pgdata` allocation, `~/.pgpass`, and a `createdb` per study. Real operational weight for an agent trying four pipelines. |

**H1 — `--slurm` does not check which of the two you got.**
`_validate_slurm_request` (`tune/_tune_cli/_run.py:664`) validates strategy, spec
path, images dir, and worker count — never the storage backend. Verified by
grepping `sqlite`/`storage_url` across `tune/`, `_execution/`, and `gui/tune/`:
no path anywhere cross-checks scheme against `--slurm`. So `--slurm` with the
default SQLite URL submits straight into the documented corruption case.

**H2 — the study name is a hard constant.** `_STUDY_NAME = "tune_cost_v1"`
(`_run.py:73`), used by store, fleet, and worker with no parameterization by
output directory. The isolation unit for concurrent studies is therefore the
**storage URL**, not the study name — two studies on one Postgres database
silently attach to the same Optuna study and pool trials. And
`tune_distributed_hpcc.md:122` instructs `export PHENOTYPIC_TUNE_STORAGE_URL=…`,
which is a resolution fallback: with it set, *every* subsequent study merges.
That is exactly the agent fan-out case.

### The fix

Add Optuna's `JournalStorage` over a symlink-locked file backend, and make it
the default for distributed runs.

```python
from optuna.storages.journal import JournalFileBackend, JournalFileSymlinkLock

lock = JournalFileSymlinkLock(str(journal_path))
storage = optuna.storages.JournalStorage(JournalFileBackend(str(journal_path), lock_obj=lock))
```

`JournalFileSymlinkLock` is the NFS-safe lock (symlink creation is atomic on
NFS; `JournalFileOpenLock` relies on `O_EXCL` semantics NFS does not reliably
provide).

**Default routing becomes:**

| Invocation | Storage |
|---|---|
| local, no `--slurm` | `sqlite:///<output>/.pht-tune-cache/study.db` — unchanged |
| `--slurm`, no explicit URL | `journal:///<output>/.pht-tune-cache/journal.log` — **new default** |
| explicit `postgresql+psycopg://…` | `RDBStorage` — unchanged, still right for very large fleets |
| explicit `sqlite://…` **with** `--slurm` | **hard error** — closes H1 |

Because the journal default derives from `output_dir`, **each study gets its own
journal file by construction** — the isolation local SQLite already has, now
extended to the distributed case. H2 stops being reachable by default. The MCP
server adds two belt-and-braces guards regardless (§4.3): it always passes
`--storage-url` explicitly, and refuses two live studies resolving to the same
URL.

### Blast radius — larger than first estimated

An earlier draft claimed the strategy layer's duck-typing made this "a small
blast radius". **That was wrong.** Optuna's string resolver wraps *any* storage
string in `RDBStorage`; there is no hook for a pseudo-scheme. Verified against
the pinned optuna 4.9.0:

```
optuna.storages.RDBStorage("journal:///tmp/x/journal.log")
  -> NoSuchModuleError: Can't load plugin: sqlalchemy.dialects:journal
optuna.storages.get_storage("journal:///tmp/x/journal.log")
  -> NoSuchModuleError: Can't load plugin: sqlalchemy.dialects:journal
```

So a **scheme-dispatch resolver is mandatory**, threaded through five
construction sites, not "a branch alongside RDBStorage":

| Site | Role |
|---|---|
| `tune/_tune_cli/_run.py:475` | Engine opens the store |
| `tune/_tune_cli/_run.py:785` | SLURM pre-create — the first thing a `--slurm` run does under the new default |
| `tune/_tune_cli/_worker.py:50` | Every SLURM worker — the entire point of P1 |
| `tune/_study/_optuna_store.py:106-109` and `gui/tune/_callbacks.py:871` | GUI Monitor, `create=False` |
| `tune/strategy/_optuna.py:239` | Fallback when the store exposes no `.study` |

Four further consequences, none of them in the first draft:

**B1 — the transient-retry predicate does not cover journal failures.**
`retry_on_transient_db_error` (`strategy/_optuna_support.py:260-319`) catches
only `sqlalchemy.exc.OperationalError`. Every `ask`/`tell`/user-attr call in the
loop (`strategy/_optuna.py:289,400,407,414,426,432`) depends on it for exactly
the condition P1 introduces: transient contention on a shared, jittery store. A
journal-backend transient — `OSError`/`PermissionError` from a flaky NFS,
`UpdateFinishedTrialError`, a `RuntimeError` from lock release — matches nothing
and crashes the worker on first occurrence. The predicate must become
backend-aware before P1 lands.

**B2 — a "read-only" Monitor open has a write side effect.**
`JournalFileBackend.__init__` does
`if not os.path.exists(file_path): open(file_path, "ab").close()` — verified.
So merely constructing the backend creates the file. A Monitor poll against a
not-yet-started study would touch a phantom `journal.log` into
`.pht-tune-cache/`, contradicting `OptunaStudyStore`'s documented `create=False`
contract ("load only an existing study"). It also loses the clean
`FileNotFoundError` UX SQLite gets, because `_open_live_study`'s pre-check
(`gui/tune/_callbacks.py:864`) is scheme-gated to sqlite and silently no-ops for
anything else. Both need an explicit existence check before construction.

**B3 — the Monitor's timeout bounding assumes file storage cannot network-hang.**
`_CONNECT_TIMEOUT_BACKENDS = {"postgresql"}` (`gui/tune/_callbacks.py:116-118`),
with the comment that SQLite is local and never network-hangs. P1's premise is
putting `journal.log` on the same NFS/Lustre mount the fleet writes. An
`os.path.exists()` or `open()` against a stale mount (default `hard` mounts,
a routine HPC failure) blocks at the syscall level indefinitely, and Python
cannot cancel a thread stuck there. `_LIVE_OPEN_POOL` is `max_workers=1`
(`:128-130`) — so one wedged probe poisons **every** subsequent Monitor poll for
**every** study for the life of the process. Either bound the file probe with an
OS-level killable mechanism or document and accept the risk explicitly.

**B4 — `journal.log` never compacts.** optuna 4.9.0 ships no compaction for the
file backend. A long study (hundreds of trials × full `pheno_params` /
`pheno_terms` payloads) grows monotonically, unlike Postgres which an operator
can vacuum. The P1 plan should state expected sizes for realistic MCP fan-out.

### What genuinely does not change

The strategy layer's heartbeat probes *are* storage-agnostic by duck-typing:

- `_heartbeat_interval` (`strategy/_optuna.py:57`) probes
  `getattr(storage, "get_heartbeat_interval", None)`, returns `None` when
  absent → `_start_heartbeat` returns `None` → no thread starts.
- `_record_heartbeat` (`:64`) returns early when `record_heartbeat` is absent.
- `_fail_stale_trials` wraps `optuna.storages.fail_stale_trials` in try/except.

Verified: `JournalStorage` has neither method; `RDBStorage` has both.

### Evidence, and the honest limit of it

`docs/superpowers/logic_validation_scripts/2026-08-12-phenotypic-mcp-server/optuna_journal_storage.py`
re-derives five claims directly from Optuna (never importing `phenotypic`).

```
ok   [C1]  optuna 4.9.0: JournalStorage + JournalFileSymlinkLock present
ok   [C2a] 4 processes x 15 trials — 60 trials persisted intact
ok   [C2b] negative control ran; see DISCRIMINATION note
ok   [C3]  journal: no heartbeat; RDB: interval=60 — stale reclamation LOST
ok   [C4]  fail_stale_trials is a safe no-op on a journal-backed study
ok   [C5]  journal:// rejected by RDBStorage AND get_storage; backend __init__ CREATES the file
```

**C2b is the important one, and on a local filesystem it reports
`DISCRIMINATION: NONE`.** An earlier version of this script asserted C2a alone
and was reported as passing. Mutation testing killed it: replacing the lock with
a no-op *also* passed, 60/60 trials, no loss. The reason is that
`JournalFileBackend.append_logs` does `open(path,"ab")` → one `write()` →
`fsync()`, and POSIX guarantees `O_APPEND` atomicity on a local filesystem
regardless of any application-level lock. On APFS/ext4 the test measures OS
write atomicity, not the lock — it could not fail, and a test that cannot fail is
worthless (project test-integrity rule).

The script now runs the negative control explicitly and says so.

**L1 — the gating step.** Before P1 is implemented, run:

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-08-12-phenotypic-mcp-server/optuna_journal_storage.py \
    --dir /path/on/the/cluster/shared/mount --require-discrimination
```

`--require-discrimination` exits non-zero unless the no-op-lock control
**actually loses trials there**. Only a passing-vs-failing pair on the *same*
filesystem is evidence that `JournalFileSymlinkLock` does any work on that
mount. Until that passes, Postgres remains the supported distributed path and
the journal default stays disabled — this is a gate, not a footnote.

### L2 — no heartbeat means no stale-trial reclamation

A worker killed by walltime or OOM leaves its trial `RUNNING` forever; under
`RDBStorage`, `fail_stale_trials` would reclaim it after the grace period.

Bounded, not fatal: `OptunaStrategy.is_exhausted()`
(`strategy/_optuna.py:441-456`) counts only `COMPLETE` and `PRUNED`, so orphaned
`RUNNING` trials do not stall budget drain — the fleet **reaches, and may
modestly overshoot,** its `n_trials`. (The overshoot is pre-existing and
identical under `RDBStorage`: `TuningEngine.optimize` (`_engine.py:109-116`)
does check-then-act on `is_exhausted()`, a TOCTOU race that concurrent workers
can overrun by up to N−1 trials. `is_exhausted`'s own docstring calls this
tolerated.) The cost is zombie rows and a slightly optimistic in-flight count.
Where that matters, Postgres remains available.

### Scope

P1 is an engine change to `phenotypic.tune`, not MCP work. It needs its own
spec, an implementation plan covering the five sites plus B1–B4, tests
(concurrent-worker, plus a mutation check that reintroducing SLURM+SQLite fails
the guard), and a rewrite of `tune_distributed_hpcc.md`. **The MCP server's
distributed-tune path is blocked on it; nothing else is.**

---

## P2 — Promote the Dash-free tier to `phenotypic/_services/`

Nine module moves plus one real split (§1.4). Watch items:

- **`gui/tune/_space.py` must be split** — it has `import dash` at `:33-34`.
- **`runs.py` is a three-file job.** `rehydrate_from_sandbox` → `classify()` →
  `gui/shell/_classifier.py:34` → `from phenotypic.gui.builder._directory_browser
  import IMAGE_EXTS` → `_directory_browser.py:20-21` imports Dash. `IMAGE_EXTS`
  is a bare suffix set; relocate it to `sdk_/_io_constants.py`.
- **`RunConsoleState` moves with `to_argv`**, or `_services/argv.py` imports back
  up into `gui/`.
- `gui/_operation_registry.py` holds a module singleton (`_REGISTRY`); the shim
  must re-export the *same* object, not construct a second.
- `discover()` is lazy today and must stay lazy.

Guarded by the import-purity test in §6.5, which must be shown to fail when an
`import dash` is reintroduced.

**Plus one extraction that is not a move.** `deploy_plan` (§5.3) must render an
sbatch preview without touching the run's output directory, but
`generate_array_job_script` (`_cli/_cli_slurm_array_scripts.py:116-368`) does
`mkdir` + `write_text` + `chmod` under `output_dir`. `SlurmArrayScriptSpec.render()`
is already pure; the ~150 lines that *build* the spec are entangled with the
write. Extract `build_array_script_spec(...) -> SlurmArrayScriptSpec` (no I/O)
and have both the real generator and `deploy_plan` call it.

## P3 — Catalog reconciliation and the JSON descriptor

Two pieces:

1. **One enumeration list.** `OperationRegistry.discover()` walks eight modules;
   `_find_class_in_phenotypic` resolves those plus `prefab`, `tune`,
   `tune.score`, `tune.strategy`, `detect.nn`. Reconcile to a shared constant so
   the agent can reach NN/GPU detectors and prefabs (§3.1).
2. **A JSON descriptor projection** over `OperationInfo`/`ParamInfo` — plus the
   `header_scheme()`-dispatching column derivation for `produces_columns` and
   `catalog_measurements`, which is *not* the ~40-line job first estimated
   (`TEXTURE.get_headers()` raises `TypeError` without a `scale`; see §3.1).
3. **The `phenotypic/subset/` subpackage** — `SubsetSelector` ABC plus
   `RandomSubsetSelector`, `MetadataGroupSubsetSelector`, and the
   `EmbeddingSubsetSelector` placeholder (§10.3). Must be added to
   `_find_class_in_phenotypic`'s submodule list in the same change that
   reconciles it with `OperationRegistry.discover()`, so selectors resolve and
   serialize by bare class name like every other extensible class.
4. **A directory-level digest helper.** `campaign_status.comparable` (§8.3) must
   detect two arms tuned against different image sets, and nothing today can:
   `bytes_fingerprint`, `file_fingerprint` (`sdk_/_io_constants.py:154,166`) and
   `pipeline_content_digest` are all single-file, and `TuningSpec` records no
   dataset at all. A stable digest over sorted `(relative path, size, mtime_ns)`
   is sufficient and cheap. Until it lands, `comparable` reports dataset
   comparability as **unknown** rather than assuming it.

## P4 — Close the `--screen` + `--slurm` silent no-op

`run_tuning` branches into `_submit_slurm_fleet` and returns **before** reaching
its `if screen:` block, and `_worker.py`'s `run_worker` constructs
`TuningEngine(...).optimize(...)` with no `ScreeningController` at all. So
`--screen --slurm` today silently drops screening — no error, no warning, the
full unscreened space runs on the fleet.

This must become an explicit error before the MCP server exposes screening at
all, since an agent requesting screening + SLURM compute would otherwise get
silently different behaviour than it asked for. It also settles OQ-4.1: the
question is not merely whether to expose `--screen`, but that the combination is
broken today.

---

## P5 — Give the tune CLI a `--slurm key=value` surface

`python -m phenotypic` accepts free-form repeated `--slurm key=value`
(`phenotypicCLI.py:795`). `python -m phenotypic.tune` accepts only four discrete
flags — `--slurm-partition`, `--slurm-mem`, `--slurm-time`,
`--slurm-constraint` (`tune/__main__.py:104-125`) — and `_submit_slurm_fleet`
(`_run.py:797-805`) builds its `slurm_args` from those alone.

So **`account`, `qos`, `cpus_per_task`, and `gpus_per_node` cannot reach a tune
fleet at all.** On a cluster where `account` is mandatory every tune submission
is rejected; where it is optional, the work is silently billed to the default
account. Both engines already funnel into the same
`format_sbatch_directives` (`sdk_/slurm/_sbatch.py:102`), which handles arbitrary
keys — the narrowing is purely in the tune CLI's argument surface.

Adding `--slurm key=value` to the tune CLI (keeping the four existing flags as
sugar) collapses one profile to both paths and makes §5.2.1's expressibility
check vestigial. Small, self-contained, and independent of P1–P4.

Until it lands, the MCP server refuses inexpressible profiles rather than
dropping keys (`profile_not_expressible`), and `paths` in the profile config
declares which surfaces a profile is valid for.

---

## Rollout order

```
P2 (_services promotion)  ──┐
P3 (catalog + descriptor) ──┼──> MCP v1: catalog + pipeline + probe + campaigns
P4 (--screen guard)       ──┤        + local tune + deploy (W0/W1/W3)
                            │
L1 (negative-control run   ─┘
    on the real cluster)
      └──> P1 (JournalStorage: 5 sites + B1–B4)  ──> distributed tune (W2 on SLURM)
```

MCP v1 ships without P1. Building pipelines, probing them, planning campaigns,
tuning **locally**, and deploying to SLURM all work with today's engines. Only
**distributed tuning** waits — and until then a SLURM tune request returns
`distributed_storage_unavailable` naming Postgres as the supported path, rather
than silently submitting into H1.
