# Choosing the polars CPU build (lts-cpu vs stock)

`phenotypic` uses [polars](https://pola.rs) for measurement compilation
(aggregating marker-authorized embedded Parquet tables into the master tables,
writing the CSV/Parquet deliverables, and the per-feature splits). Polars ships in **two
interchangeable PyPI distributions** that both provide the same `import polars`
API and differ only in the CPU instruction set they target:

| Distribution | Baseline instruction set | Runs on… | Speed |
|---|---|---|---|
| **`polars-lts-cpu`** (shipped by default) | x86-64 baseline (SSE) | **every** CPU, incl. pre-AVX2 | full write speed; parquet *decode* ~4–5× slower |
| `polars` (stock) | x86-64-v3 (**AVX2** required) | AVX2-capable CPUs only | fastest |

The stock `polars` wheel bakes AVX2 into its baseline **with no runtime
fallback**, so it aborts with `Illegal instruction (core dumped)` (SIGILL) on
CPUs without AVX2. The UCR HPCC has such nodes — `abu_dhabi` (`c01–30`, AMD
Piledriver) and `ivy` (`h01–06`, Intel Ivy Bridge) — so **`phenotypic` installs
`polars-lts-cpu` by default** to run everywhere out of the box. (numpy/scipy use
*runtime* SIMD dispatch and are unaffected on those nodes.)

If your machine has AVX2 (any AMD Zen / EPYC `rome`/`milan`/`genoa`/`ryzen`, or
Intel `broadwell`/`cascade`/`sapphire` and newer), you can swap in the faster
stock build.

## Check whether your CPU has AVX2

```bash
grep -qw avx2 /proc/cpuinfo && echo "AVX2: yes (stock polars OK)" || echo "AVX2: no (keep polars-lts-cpu)"
```

## Install the fast stock build with uv

The two distributions own the same `polars/` import directory and **cannot be
installed at the same time**, so switching is an uninstall + install:

```bash
uv sync                           # installs the default polars-lts-cpu
uv pip uninstall polars-lts-cpu
uv pip install "polars>=1.0.0"    # the fast stock build (same `import polars`)
```

Verify the swap:

```bash
uv run python -c "import polars; print(polars.__version__)"
```

> **Note:** a later `uv sync` re-pins the environment to `polars-lts-cpu` (it
> enforces the project's locked dependency), so re-run the two `uv pip` lines
> after any sync, or use a dedicated environment as below.

## Recommended HPCC pattern: two environments, routed by CPU

On a heterogeneous cluster, keep the default lts-cpu environment (works
everywhere) and an extra AVX2-only environment, then pick one at job start:

```bash
# One-time setup
uv sync                                   # .venv  -> polars-lts-cpu (all nodes)
uv venv .venv-fast                        # second env for AVX2 nodes
VIRTUAL_ENV=.venv-fast uv pip install -e .
VIRTUAL_ENV=.venv-fast uv pip uninstall polars-lts-cpu
VIRTUAL_ENV=.venv-fast uv pip install "polars>=1.0.0"
```

```bash
# In your SLURM job script, before running phenotypic:
if grep -qw avx2 /proc/cpuinfo; then
    source .venv-fast/bin/activate     # AVX2 node -> fast stock polars
else
    source .venv/bin/activate          # pre-AVX2 node -> polars-lts-cpu
fi
python -m phenotypic ...
```

Modern nodes then run the fast build while pre-AVX2 nodes fall back
automatically — no node names hardcoded.

### Alternative: pin jobs to AVX2 nodes

If you would rather run a single stock-`polars` environment everywhere, keep
SLURM off the pre-AVX2 nodes instead. The CLI forwards `--slurm key=value` as
`#SBATCH --key=value`, so either constrain to AVX2 microarchitectures:

```bash
python -m phenotypic ... --slurm constraint="milan|genoa|rome|ryzen|broadwell|cascade|sapphire"
```

or exclude the pre-AVX2 nodes:

```bash
python -m phenotypic ... --slurm exclude=c[01-30],h[01-06]
```

## Performance context

The lts-cpu penalty is **concentrated in Parquet decode** (the SIMD-heavy read
step); CSV/Parquet *writing* — which dominates compilation cost — is essentially
unaffected. Even on lts-cpu, polars compilation remains far faster than a
pandas-based equivalent. See `diagnostics/polars_vs_duckdb/REPORT.md` in the
source tree for the full benchmark.
