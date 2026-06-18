# Diagnostic: Polars vs DuckDB+pandas for measurement compilation

**Date:** 2026-06-17 · **Branch:** `worktree-polars-diagnostic` ·
**Harness:** `diagnostics/polars_vs_duckdb/` ·
**Status:** complete — awaiting review before any repo changes.

## 1. Question

Is keeping **polars** worth it for the measurement-compilation path
(`aggregate_measurements` / `finalize_post_master_outputs` / chunk writer),
versus a **duckdb + pandas** stack? Motivation: the default polars wheel bakes
AVX2 into its baseline (no runtime fallback) and **SIGILLs on the ~1/3 of HPCC
compute nodes with pre-AVX2 CPUs**. Keep polars only if the speedup justifies
that fragility.

## 2. TL;DR verdict

**Keep polars.** For this workload it is **~5–12× faster** end-to-end than
duckdb+pandas, and the gap is dominated by CSV/Parquet writing where polars'
multithreaded writers beat pandas by **4–23×**. Swapping to duckdb+pandas would
make every compile (and every mid-run checkpoint) markedly slower while **not
actually fixing the old-node problem** — numpy/scipy are flagged with the same
SIGILL risk in the project CLAUDE.md, and dropping polars doesn't touch them.

The old-node issue is a **deployment** problem, best solved by (a) pinning SLURM
to AVX2-capable partitions (already the documented mitigation) and/or (b)
standardizing on the official **`polars-lts-cpu`** drop-in wheel, which runs on
the old nodes, keeps the API, and keeps write speed (the lts penalty is confined
to parquet *decode*, not writes).

**Bonus finding (separate optimization, flagged for review):** the repo's
current forward path is *suboptimal* — it routes the concat through DuckDB, which
is the **slowest single step measured (7.3 s, 2.7 GB)**. A pure-polars concat
producing identical output is **6.7× faster (1.1 s) and 4× lighter (0.7 GB)**.

## 3. Method

- **Node:** AMD EPYC 9554 (AVX-512), SLURM allocation 12 cores / 32 GB.
- **Isolation:** corpus staged to node-local `/tmp` (warm) so the engine
  comparison is not dominated by GPFS metadata variance (production stages to
  `$SCRATCH` for the same reason). Each variant runs in a **fresh subprocess**
  (clean peak-RSS via `ru_maxrss`, no warm-state bias); **1 warmup + 7 measured**
  repeats; **median** reported. All engines capped to **12 threads**.
- **Versions:** polars 1.38.1, duckdb 1.5.1, pandas 2.3.3, pyarrow 23.0.1,
  numpy 2.3.5. lts venv: `polars-lts-cpu` 1.33.1.
- **Equivalence checked:** the polars and pandas pipelines were verified to
  produce **byte-identical master output** (356,302 × 30, `frame_equal=True`,
  numeric checksums match) — the speed comparison is on like-for-like results.

### Dataset (clean, per-image only)

Per your instruction, a clean copy with **batch-aggregated files removed** was
used (`/tmp/pl_diag_sample_clean`, durable artifact):

- **7,908** per-image parquets, 127 MB total (mean 16.5 KB).
- **3** `_dataset_aggregated.parquet` files **excluded** (Runs 4, 5, 11) — this
  is the "compile from per-image scratch" worst case, not the optimistic path
  that shortcuts through pre-aggregated files.
- Aggregates to a **356,302 row × 30 col** master (~96 MB in-mem; 32 MB Parquet
  zstd; **153 MB CSV**).

## 4. Results

### 4a. Concatenation — read 7,908 parquets → one frame (median of 7)

| strategy | time | peak RSS | notes |
|---|---:|---:|---|
| `concat_duckdb_polars` — **repo forward path** | **7,348 ms** | 2,734 MB | DuckDB UNION + `filename` LEFT JOIN → arrow → polars |
| `concat_duckdb_pandas` | 4,674 ms | 2,468 MB | same SQL → arrow → pandas |
| `concat_pyarrow_pandas` | 2,760 ms | 1,089 MB | pyarrow dataset → pandas |
| `concat_polars_scan` | 1,817 ms | 701 MB | lazy `scan_parquet` + concat (chunk-writer style) |
| **`concat_polars_native`** | **1,095 ms** | **682 MB** | `pl.read_parquet(…, include_file_paths)` — same columns as repo |
| `concat_polars_read` | 655 ms | 369 MB | raw multi-file read (no dataset labels) |

> The repo's DuckDB concat is the **slowest and heaviest** option. The cost is
> the string-keyed `filename` LEFT JOIN over 356 k rows used to attach
> `Metadata_Dataset` (per-image files carry no dataset column — confirmed).
> Pure polars yields identical columns **6.7× faster, 4× lighter**.

### 4b. End-to-end compile — concat + derive + write master.{parquet,csv} + per-feature split (median of 7)

| pipeline | time | peak RSS | vs best |
|---|---:|---:|---:|
| **`macro_polars_native`** — pure polars, no duckdb | **1,414 ms** | 960 MB | 1.0× |
| `macro_pandas_duckwrite` — duckdb + pandas, DuckDB `COPY` writes | 7,218 ms | 2,470 MB | 5.1× |
| `macro_polars` — **repo**: duckdb concat + polars writes | 7,703 ms | 2,736 MB | 5.4× |
| `macro_pandas` — duckdb + pandas writers | **17,666 ms** | 2,471 MB | 12.5× |

### 4c. Per-stage write/convert shootout — on the real master (348,521 × 30), median of 7

| operation | polars | pandas | duckdb | pandas/polars |
|---|---:|---:|---:|---:|
| write Parquet (zstd L3) | **224 ms** | 954 ms | 448 ms | **4.3×** |
| write **CSV** (153 MB) | **262 ms** | 6,014 ms | 709 ms | **23×** |
| per-feature split (select + csv + parquet) | **337 ms** | 6,412 ms | — | **19×** |
| convert polars→pandas | 685 ms | — | — | (round-trip tax) |
| convert pandas→polars | 790 ms | — | — | (round-trip tax) |

> The decisive factor is **CSV writing**: pandas `to_csv` is single-threaded and
> ~23× slower than polars `write_csv`. The compile writes several CSVs
> (`master_measurements.csv`, `measurements.csv`, per-feature CSVs) on every run
> and at every mid-run checkpoint, so this cost recurs.

### 4d. `polars-lts-cpu` (old-node-safe build) vs default polars — same node (median of 5)

| operation | default | lts-cpu | lts/default |
|---|---:|---:|---:|
| concat (read 7,908) | 1,152 ms | 5,353 ms | 4.65× |
| **end-to-end compile** | 1,414 ms | 6,072 ms | 4.29× |
| write CSV | 264 ms | 250 ms | **0.95× (no penalty)** |
| write Parquet | 240 ms | 345 ms | 1.44× |
| per-feature split | 291 ms | 479 ms | 1.65× |

> The lts penalty is **confined to parquet decode** (the SIMD-heavy concat).
> **Writes are essentially unaffected** (CSV identical). Even at its worst,
> lts-cpu pure-polars compile (**6.1 s**) is *faster than the repo's current path*
> (7.7 s) and ~3× faster than naive duckdb+pandas (17.7 s). On modern nodes,
> default polars (1.4 s) is ~12× faster than naive pandas.

## 5. Why "drop polars" does not solve the stated problem

- **Default polars** compiles AVX2 into its baseline with no runtime fallback →
  hard SIGILL on pre-AVX2 CPUs. This is the real failure. `polars-lts-cpu` is the
  official baseline-ISA wheel that fixes exactly this.
- **numpy/scipy** use *runtime* SIMD dispatch (SSE2 baseline + detected AVX), so
  they normally degrade gracefully — **but** the project CLAUDE.md explicitly
  documents observed SIGILL of "modern numpy/scipy wheels" on pre-AVX nodes.
  Since the whole pipeline (and pandas itself) depends on numpy/scipy, removing
  polars would **not** make a job natively safe on those nodes.
- Therefore the instruction-set fragility is a **fleet/deployment** matter, not a
  reason to give up a 5–12× compile speedup.

## 6. Recommendation

1. **Keep polars.** The speedup is large and recurring; the alternative is slower
   *and* doesn't fix the old nodes.
2. **Handle old nodes at deploy time:**
   - Preferred: pin SLURM aggregation/finalize jobs to AVX2-capable partitions
     (`--constraint`/partition), as already noted in CLAUDE.md.
   - If pre-AVX2 nodes must run: standardize the environment on `polars-lts-cpu`
     (and confirm numpy/scipy baseline builds). Keeps the API and write speed;
     ~4–5× slower parquet decode on modern nodes, still ahead of the alternatives.
3. **Separate optimization (flag for review, not part of this ask):** replace the
   DuckDB concat in `aggregate_measurements` / `_cli_duckdb_agg.duckdb_aggregate`
   with `pl.read_parquet(paths, include_file_paths=…).rechunk()` + vectorized
   `Metadata_Dataset`/`Metadata_ImageFile` derivation. Identical output, **6.7×
   faster, 4× less memory**, and removes DuckDB from the hot path. (Whether to
   drop the DuckDB dependency entirely is a separate question — it is also used in
   the recompile worker.)

## 7. Caveats / threats to validity

- **Warm node-local storage.** Production reads from GPFS; a cold read of 7,908
  files was observed at ~70 s and is **engine-independent** (identical for all
  engines) — it would dominate wall-clock but does not change the comparison.
- **Worst-case file count.** Benchmarked per-image-only (batch-aggregated files
  excluded, per request). When `_dataset_aggregated.parquet` exist, the forward
  path reads far fewer files and the concat is cheaper for *all* engines.
- **One feature family.** This dataset splits into a single per-feature file
  (`SymmetricZones`). More feature producers → more CSV writes → polars'
  advantage **compounds**.
- **pandas already in the loop.** post/analysis/QC round-trip through pandas
  (~0.7 s each) regardless of engine choice; that cost is not a differentiator.
- **lts venv version skew.** `polars-lts-cpu` 1.33.1 vs default 1.38.1 (closest
  available); the lts↔default delta is dominated by the instruction-set baseline,
  with a minor version component.
- **CSV float formatting** differs cosmetically between polars and pandas;
  **Parquet content is verified identical**.

## 8. Reproduction

```bash
# Full engine benchmark (stages corpus, 7 repeats, equivalence check)
uv run python diagnostics/polars_vs_duckdb/run_diagnostic.py \
  --dataset /rhome/anguy344/shared_exfab/projects/ucr_010_i_d_neurospora/data/results/2026-05-11 \
  --repeat 7 --threads 12

# polars-lts-cpu vs default polars (after building /tmp/pl_lts_venv)
uv run python diagnostics/polars_vs_duckdb/run_lts_compare.py \
  --corpus /tmp/pl_diag_sample_clean/results \
  --master-parquet <dataset>/master_measurements.parquet \
  --lts-python /tmp/pl_lts_venv/bin/python --repeat 5
```

Raw numbers: `results.json`, `results_lts.json`. Clean per-image-only sample:
`/tmp/pl_diag_sample_clean`.
