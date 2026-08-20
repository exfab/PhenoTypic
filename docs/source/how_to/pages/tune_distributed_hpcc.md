# Distributed Tuning on HPCC Clusters

Run a PhenoTypic tune study across many SLURM compute nodes that share one
optimization study.

## Storage: what a fleet shares, and why it is not SQLite

A local single-node run keeps its Optuna study in a SQLite database
(`.pht-tune-cache/study.db`) using WAL mode. That is fine when one process owns
the file, but SQLite-WAL is unsafe on the network filesystems HPCC clusters use
(NFS, Lustre): file locking across nodes is unreliable, so several SLURM array
workers writing the same `study.db` will corrupt it or lose trials.

So a fleet needs different storage. There are two supported answers, and
`--slurm` picks the first for you:

| Invocation | Storage |
|---|---|
| no `--slurm` | `sqlite:///<output>/.pht-tune-cache/study.db` |
| `--slurm`, no explicit URL | `journal:///<output>/.pht-tune-cache/journal.log` — **the default** |
| any explicit `postgresql+psycopg://…` | that server |
| an explicit `sqlite://…` **with** `--slurm` | **refused**, with an error |

That last row matters: nothing used to cross-check the backend against
`--slurm`, so a SQLite URL — typed, or inherited from
`$PHENOTYPIC_TUNE_STORAGE_URL` — submitted straight into the corruption case
above. It is now a hard error naming the two alternatives.

### The journal backend (default)

`journal.log` is Optuna's `JournalStorage`: an append-only log guarded by a
symlink lock, which is atomic on NFS where the `O_EXCL` semantics a lock file
would need are not. **Nothing to stand up** — no server, no `pgdata`, no
`createdb`, no `~/.pgpass`. Just add `--slurm`.

Because the path derives from `-o`, every run gets its own journal, so two
concurrent studies cannot silently pool trials into one another (they otherwise
share a hardcoded study name).

Two limits to know:

- **No stale-trial reclamation — permanently, not briefly.** The journal
  backend has no heartbeat, so `optuna.storages.fail_stale_trials()` has
  nothing to act on: against a journal study it returns cleanly and changes
  nothing, with no error or log line to say so. A worker killed by walltime
  or OOM therefore leaves its trial marked `RUNNING` for the life of the
  study, and nothing ever reclaims it. Postgres transitions the same trial to
  `FAIL` once its grace period lapses — the failure is self-healing there and
  standing here, on exactly the path (SLURM walltime kills) that creates it.
  At tens of minutes per evaluation, a killed worker is the *normal* end
  state of a long fleet run, so expect these rows rather than treating them
  as rare.

  The damage is contained but real. Selection and the budget both read only
  terminal (`COMPLETE`/`PRUNED`/`FAIL`) trials, so the fleet still drains its
  `--n-trials` and a resultless trial can never be picked as the winner. What
  the zombies do cost you is the *raw* trial count: they accumulate in the
  study file, inflate the trial count the GUI Monitor shows, and inflate the
  "still in flight" figure `finalize` warns with — a figure that, on this
  backend, cannot be distinguished from live workers and so may never reach
  zero. Where an accurate in-flight count matters, use Postgres.
- **The log only grows** — but not fast enough to matter for a campaign.
  Optuna ships no compaction for the file backend and there is no `VACUUM`, so
  `journal.log` grows monotonically at roughly **6 KB (about 20 log records) per
  trial**: a 200-trial arm is a **1.2 MB** file, and 2,000 trials is 12 MB
  (measured on optuna 4.9.0 over the real ask/tell path; the rate is pinned by
  `test_journal_growth_per_trial_stays_near_the_measured_rate`).

  What scales with the file is not disk but *replay*: every worker start and
  every Monitor poll re-reads the whole log — 0.07 s at 200 trials, 1.0 s at
  2,000 — and the per-trial write cost rises from ~30 ms to ~90 ms across that
  range, against a ~30-minute evaluation. Nothing needs doing at campaign scale.

  The size only becomes real if you **reuse one output directory across many
  campaigns**; past 64 MiB every open logs a warning saying so. The remedy is a
  fresh `-o` directory (or Postgres), never editing the log: records are
  addressed by byte offset, so rewriting the file shorter would leave every live
  worker and Monitor reading from the middle of a record.

Where either matters, use Postgres.

### Postgres (still supported)

PhenoTypic is backend-agnostic: it takes a generic, user-provided storage URL
and hands it to Optuna's RDB storage, so any reachable PostgreSQL server works.
It remains the right choice for the two limits above, and for an existing
shared server you already run. It is real operational weight for a single study
— hence no longer the default.

The rest of this section covers that setup; skip it if you are using the
default journal backend.

## Standing up a Postgres server

PhenoTypic does not manage a server for you; it only needs a URL that resolves
to a running PostgreSQL instance the workers can reach. One common pattern on an
HPCC is a user-space PostgreSQL server submitted as its own `sbatch` job: the
job initializes a data directory on shared storage (e.g. `pgdata/` under your
`/bigdata` allocation), starts `postgres` on a non-default port such as `54399`,
and writes its host:port address to a file other jobs can read. Any tooling that
gets you a reachable server is fine — this is one example, not a requirement.

Once the server is up, create a database for the study:

```bash
createdb -h <pg-host> -p 54399 -U $USER tune_study
```

## Connecting

The connection target is given to `--storage-url` (or the
`$PHENOTYPIC_TUNE_STORAGE_URL` environment variable). PhenoTypic uses the
password-less psycopg3 scheme

```
postgresql+psycopg://USER@HOST:PORT/DB
```

The password is **never** part of the URL, the command line, or the generated
worker script. libpq resolves it from `~/.pgpass` (or the `$PGPASSWORD`
environment variable), the same as any other PostgreSQL client. Add one line to
`~/.pgpass` (mode `600`) in `host:port:database:username:password` format:

```bash
chmod 600 ~/.pgpass
# host:port:database:username:password
echo "<pg-host>:54399:*:$USER:<secret>" >> ~/.pgpass
```

There are three equivalent ways to point at the server; all three already work
through the `psycopg → SQLAlchemy → Optuna` stack with no extra configuration in
PhenoTypic, and all three read the password from `~/.pgpass`.

The first is the full password-less URL, naming the driver, user, host, port,
and database directly:

```bash
python -m phenotypic.tune run spec.json -i ./plates -o ./out \
    --strategy tpe --n-trials 200 \
    --storage-url "postgresql+psycopg://$USER@<pg-host>:54399/tune_study" \
    --slurm
```

The second is a named service. Define the connection target once in
`~/.pg_service.conf` so you never retype host/port/dbname:

```ini
[tune]
host=<pg-host>
port=54399
dbname=tune_study
user=<your-user>
```

then reference it with a service-only URL:

```bash
python -m phenotypic.tune run spec.json -i ./plates -o ./out \
    --strategy tpe --n-trials 200 \
    --storage-url "postgresql+psycopg://?service=tune" \
    --slurm
```

The third uses the standard `PG*` environment variables for the target and a
bare URL:

```bash
export PGHOST=<pg-host>
export PGPORT=54399
export PGDATABASE=tune_study
export PGUSER=$USER
python -m phenotypic.tune run spec.json -i ./plates -o ./out \
    --strategy tpe --n-trials 200 \
    --storage-url "postgresql+psycopg://" \
    --slurm
```

A target is always required because `--storage-url` names the server to reach,
while `~/.pgpass` only supplies the password (it is keyed by host/port/db/user
and does not, on its own, tell libpq which server to contact). A future
convenience flag — `--pg-service NAME` to build the `?service=NAME` URL for you
— is **not yet implemented**; until then the `?service=tune` form above gives
the same result.

## Launching the fleet

Add `--slurm` to submit a worker fleet instead of running in-process. Every
worker opens the same study and drains the one trial budget set by
`--n-trials`, so the budget is shared across the fleet rather than multiplied by
the number of workers.

With no storage flags at all, the study is the run's own `journal.log`:

```bash
python -m phenotypic.tune run spec.json -i ./plates -o ./out \
    --strategy tpe --n-trials 200 --slurm
```

To share a Postgres server instead, name it — on the command line, or once in
the environment:

```bash
export PHENOTYPIC_TUNE_STORAGE_URL="postgresql+psycopg://$USER@<pg-host>:54399/tune_study"
python -m phenotypic.tune run spec.json -i ./plates -o ./out \
    --strategy tpe --n-trials 200 --slurm
```

Note that this variable applies to *every* subsequent run in the shell, and two
runs on one server attach to the same study and pool their trials. Prefer
`--storage-url` per run when you are tuning more than one pipeline.

The submitting process pre-creates the study before any worker starts. A cold
Postgres database has no Optuna tables, so without this step the workers would
race to create the schema and all but one would crash on a duplicate-key error;
the journal backend is materialized up front for the same reason. Either way
every worker finds an existing study and only reads and appends trials.

The workers reload the resolved spec written to
`deliverables/tuning_spec.json`, not the raw input spec, so any `--strategy`,
`--n-trials`, or held-out overrides given at submission are honored on every
node.

### Sizing the fleet

The fleet's shape is set by four `--slurm`-only flags; all are optional and
fall back to sensible defaults:

- `--n-workers N` — how many SLURM array workers to submit. Unset, it defaults
  to `min(8, n_trials)` (or 4 when no trial budget is known). Because the
  budget is shared, adding workers shortens wall-clock without spending more
  trials.
- `--slurm-partition NAME` — the partition for the array. Unset, the
  `#SBATCH --partition` directive is omitted and the cluster default applies.
- `--slurm-mem MEM` — per-worker memory (e.g. `8G`).
- `--slurm-time HMS` — per-worker wall-clock limit (e.g. `04:00:00`).

```bash
python -m phenotypic.tune run spec.json -i ./plates -o ./out \
    --strategy tpe --n-trials 200 --slurm \
    --n-workers 16 --slurm-partition batch --slurm-mem 8G --slurm-time 04:00:00 \
    --storage-url "postgresql+psycopg://$USER@<pg-host>:54399/tune_study"
```

### Any other `#SBATCH` directive: `--slurm key=value`

The four flags above are sugar over a general passthrough. `--slurm` is
repeatable and each occurrence may carry one free-form `KEY=VALUE`, exactly as
on `python -m phenotypic`; every key becomes one `#SBATCH` directive, with a
leading `slurm_` stripped and remaining underscores turned into dashes
(`slurm_cpus_per_task=8` → `#SBATCH --cpus-per-task=8`).

This is the only way to reach keys the four flags cannot express — most
importantly **`account`**, which UCR HPCC *requires* for the `exfab` and
`preempt` partitions. Without it a tune fleet cannot reach the GPU node or the
preempt pool at all, and elsewhere the work is silently billed to your default
account.

```bash
# A GPU fleet on exfab: --account is mandatory there.
python -m phenotypic.tune run spec.json -i ./plates -o ./out \
    --strategy tpe --n-trials 200 --n-workers 4 \
    --slurm slurm_partition=exfab \
    --slurm slurm_account=exfab \
    --slurm slurm_cpus_per_task=8 \
    --slurm slurm_mem=64G \
    --slurm slurm_gpus_per_node=1 \
    --storage-url "postgresql+psycopg://$USER@<pg-host>:54399/tune_study"
```

Notes:

- The **presence** of `--slurm` is what requests submission, so a bare `--slurm`
  still means "submit with cluster defaults", and a run carrying only
  `--slurm key=value` pairs is submitted too. There is no separate on/off flag.
- Because a bare `--slurm` accepts an optional value, it swallows the next token
  when that token is not an option. Write it *after* the spec path.
- When both spellings are given, the explicit pair wins:
  `--slurm-partition batch --slurm partition=epyc` submits to `epyc`.
  `partition=` and `slurm_partition=` are the same key.

## Resume

A tune run resumes by re-pointing `-o` at the same output directory. The shared
study still holds every trial the fleet recorded, and the run's
`trials.parquet` plus the machine state under `.pht-tune-cache/` let a new
invocation pick up where the previous one stopped. If the local `study.db` (or a
relocated split) was moved, it is read back from `.pht-tune-cache/`. For a
distributed study, resuming means submitting the fleet again against the same
`--storage-url` — the workers re-attach to the existing study rather than
starting fresh.

```bash
python -m phenotypic.tune run spec.json -i ./plates -o ./out \
    --strategy tpe --n-trials 400 \
    --storage-url "postgresql+psycopg://$USER@<pg-host>:54399/tune_study" \
    --slurm
```

## Local vs. distributed (when to use which)

For a single node, omit `--slurm` and `--storage-url`. The run uses the local
SQLite-WAL study at `.pht-tune-cache/study.db`, which is the safe and simplest
default when one process owns the database.

For a SLURM array spanning multiple nodes on a shared (NFS/Lustre) filesystem,
use a Postgres `--storage-url` with `--slurm`. SQLite-WAL is unsafe there, so the
shared relational database is what lets the workers cooperate on one study.
