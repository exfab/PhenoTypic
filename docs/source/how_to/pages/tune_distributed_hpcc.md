# Distributed Tuning on HPCC Clusters

Run a PhenoTypic tune study across many SLURM compute nodes that share one
optimization study.

## Why Postgres for distributed studies

A local single-node run keeps its Optuna study in a SQLite database
(`.pht-tune-cache/study.db`) using WAL mode. That is fine when one process owns
the file, but SQLite-WAL is unsafe on the network filesystems HPCC clusters use
(NFS, Lustre): file locking across nodes is unreliable, so several SLURM array
workers writing the same `study.db` will corrupt it or lose trials.

A distributed study therefore needs a shared relational database that every
worker can reach over the network. PhenoTypic is backend-agnostic — it does not
ship or depend on any particular database server. It takes a generic,
user-provided storage URL and hands it straight to Optuna's RDB storage, so any
reachable PostgreSQL server works. The rest of this guide uses Postgres because
it is the standard choice for an Optuna study shared by a SLURM array.

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
worker opens the same study at the shared `--storage-url` (or
`$PHENOTYPIC_TUNE_STORAGE_URL`) and drains the one trial budget set by
`--n-trials`, so the budget is shared across the fleet rather than multiplied by
the number of workers:

```bash
export PHENOTYPIC_TUNE_STORAGE_URL="postgresql+psycopg://$USER@<pg-host>:54399/tune_study"
python -m phenotypic.tune run spec.json -i ./plates -o ./out \
    --strategy tpe --n-trials 200 --slurm
```

The submitting process pre-creates the study (and its RDB schema) before any
worker starts. A cold Postgres database has no Optuna tables, so without this
step the workers would race to create the schema and all but one would crash on
a duplicate-key error. Materializing the study once up front means every worker
finds an existing study and only reads and appends trials.

The workers reload the resolved spec written to
`deliverables/tuning_spec.json`, not the raw input spec, so any `--strategy`,
`--n-trials`, or held-out overrides given at submission are honored on every
node.

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
