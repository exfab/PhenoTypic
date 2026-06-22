"""Orchestrate the polars-vs-duckdb+pandas measurement-compilation diagnostic.

What it does
------------
1. Stages the real per-image parquet corpus to node-local ``/tmp`` once, so the
   engine comparison is not dominated by GPFS metadata variance (production
   stages to ``$SCRATCH`` for the same reason -- see ``_stage_to_scratch``).
2. Runs every benchmark variant ``--repeat`` times, each in a *fresh*
   subprocess (clean peak-RSS, no warm-state bias), via ``bench_runner.py``.
3. Verifies the polars and duckdb+pandas macro pipelines produce numerically
   equivalent master Parquet output (a speed comparison is meaningless if the
   results differ).
4. Aggregates min / median / mean / std and writes ``results.json`` +
   prints a human-readable summary.

Usage::

    uv run python diagnostics/polars_vs_duckdb/run_diagnostic.py \
        --dataset /path/to/results/2026-05-11 \
        --repeat 7 --threads 12
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "bench_runner.py"

CONCAT_VARIANTS = [
    "concat_duckdb_polars",   # current forward path (repo): duckdb -> polars
    "concat_duckdb_pandas",   # proposed: identical duckdb SQL -> pandas
    "concat_polars_native",   # optimal pure-polars (same columns, no duckdb)
    "concat_polars_read",     # raw pure-polars multi-file read (no labels)
    "concat_polars_scan",     # pure-polars lazy scan+concat (chunk writer)
    "concat_pyarrow_pandas",  # pure pyarrow dataset -> pandas
]
MACRO_VARIANTS = [
    "macro_polars",            # current behaviour (duckdb concat + polars writes)
    "macro_polars_native",     # pure polars, no duckdb at all
    "macro_pandas",            # duckdb+pandas drop-in (pandas writers)
    "macro_pandas_duckwrite",  # duckdb+pandas, DuckDB COPY for master writes
]
MICRO_VARIANTS = [
    "wpq_polars", "wpq_pandas_pyarrow", "wpq_duckdb",
    "wcsv_polars", "wcsv_pandas", "wcsv_duckdb",
    "conv_to_pandas", "conv_from_pandas",
    "split_polars", "split_pandas",
]


def _stage_corpus(dataset: Path, stage_root: Path) -> Path:
    """Copy per-image parquets to node-local storage, preserving ds/measurements."""
    results = dataset / "results"
    corpus = stage_root / "corpus"
    if corpus.exists():
        shutil.rmtree(corpus)
    n = 0
    total = 0
    for ds_dir in sorted(results.iterdir()):
        meas = ds_dir / "measurements"
        if not meas.is_dir():
            continue
        dest = corpus / ds_dir.name / "measurements"
        dest.mkdir(parents=True, exist_ok=True)
        for pq in meas.glob("*.parquet"):
            if pq.name.startswith("_"):
                continue
            shutil.copy2(pq, dest / pq.name)
            n += 1
            total += pq.stat().st_size
    print(f"  staged {n} parquet files ({total/1048576:.1f} MB) -> {corpus}")
    return corpus


def _run_variant(family: str, variant: str, *, corpus: Path | None,
                 master_pq: Path | None, out: Path, threads: int) -> dict:
    cmd = [
        sys.executable, str(RUNNER),
        "--family", family, "--variant", variant,
        "--out", str(out), "--threads", str(threads),
    ]
    if corpus is not None:
        cmd += ["--corpus", str(corpus)]
    if master_pq is not None:
        cmd += ["--master-parquet", str(master_pq)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"    !! {variant} FAILED:\n{proc.stderr[-2000:]}", file=sys.stderr)
        return {}
    # runner prints exactly one JSON line last
    line = proc.stdout.strip().splitlines()[-1]
    return json.loads(line)


def _bench(family: str, variants: list[str], repeat: int, *,
           corpus: Path | None, master_pq: Path | None,
           stage_root: Path, threads: int) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {}
    for variant in variants:
        out = stage_root / "out" / variant
        runs: list[dict] = []
        # 1 warmup (discarded) + `repeat` measured
        for i in range(repeat + 1):
            if out.exists():
                shutil.rmtree(out)
            rec = _run_variant(family, variant, corpus=corpus, master_pq=master_pq,
                               out=out, threads=threads)
            if rec and i > 0:
                runs.append(rec)
        if runs:
            results[variant] = runs
            walls = [r["wall_total_s"] for r in runs]
            rss = [r["peak_rss_mb"] for r in runs]
            print(f"  {variant:24s} median={statistics.median(walls)*1000:8.1f} ms"
                  f"  min={min(walls)*1000:8.1f} ms  peakRSS={max(rss):6.0f} MB"
                  f"  (n={len(runs)})")
        else:
            print(f"  {variant:24s} NO DATA")
    return results


def _verify_equivalence(stage_root: Path, threads: int, corpus: Path) -> dict:
    """Run macro_polars and macro_pandas once and compare master Parquet output."""
    import polars as pl

    verdicts = {}
    frames = {}
    for variant in ("macro_polars", "macro_pandas"):
        out = stage_root / "verify" / variant
        if out.exists():
            shutil.rmtree(out)
        rec = _run_variant("macro", variant, corpus=corpus, master_pq=None,
                           out=out, threads=threads)
        if not rec:
            return {"status": "error", "detail": f"{variant} did not run"}
        frames[variant] = pl.read_parquet(str(out / "master.parquet"))

    a = frames["macro_polars"]
    b = frames["macro_pandas"]
    verdicts["polars_shape"] = list(a.shape)
    verdicts["pandas_shape"] = list(b.shape)
    verdicts["same_shape"] = a.shape == b.shape
    verdicts["same_columns"] = sorted(a.columns) == sorted(b.columns)

    if verdicts["same_shape"] and verdicts["same_columns"]:
        # align column order, sort by a stable key, compare numerically
        common = a.columns
        b2 = b.select(common)
        sort_keys = [c for c in ("Metadata_Dataset", "Metadata_ImageName",
                                 "Metadata_ImageFile", "ObjectLabel") if c in common]
        a_s = a.sort(sort_keys)
        b_s = b2.sort(sort_keys)
        try:
            verdicts["frame_equal"] = a_s.equals(b_s)
        except Exception as e:
            verdicts["frame_equal"] = f"compare-error: {e}"
        # numeric checksum on float columns as a softer equality signal
        num_cols = [c for c, t in zip(a.columns, a.dtypes) if t in (pl.Float64, pl.Int64)]
        verdicts["numeric_checksum_match"] = bool(
            abs(a.select(num_cols).sum().to_numpy().sum()
                - b2.select(num_cols).sum().to_numpy().sum()) < 1e-3
        )
    verdicts["status"] = "ok"
    return verdicts


def _summary_stats(runs: list[dict]) -> dict:
    walls = [r["wall_total_s"] for r in runs]
    rss = [r["peak_rss_mb"] for r in runs]
    stage_keys = runs[0]["stages"].keys()
    stage_stats = {}
    for k in stage_keys:
        vals = [r["stages"].get(k, float("nan")) for r in runs]
        vals = [v for v in vals if v == v]
        if vals:
            stage_stats[k] = {"median_s": statistics.median(vals), "min_s": min(vals)}
    return {
        "n": len(runs),
        "wall_min_s": min(walls),
        "wall_median_s": statistics.median(walls),
        "wall_mean_s": statistics.mean(walls),
        "wall_std_s": statistics.pstdev(walls) if len(walls) > 1 else 0.0,
        "peak_rss_mb_max": max(rss),
        "peak_rss_mb_median": statistics.median(rss),
        "stages": stage_stats,
        "rows": runs[0].get("rows"),
        "cols": runs[0].get("cols"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--repeat", type=int, default=7)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--stage-root", type=Path,
                    default=Path("/tmp") / "pl_diag")
    ap.add_argument("--results-json", type=Path, default=HERE / "results.json")
    args = ap.parse_args()

    args.stage_root.mkdir(parents=True, exist_ok=True)
    print(f"== Polars vs DuckDB+pandas diagnostic ==")
    print(f"dataset : {args.dataset}")
    print(f"repeat  : {args.repeat}  threads: {args.threads}")
    print(f"stage   : {args.stage_root}\n")

    print("[1/5] Staging corpus to node-local storage ...")
    corpus = _stage_corpus(args.dataset, args.stage_root)
    master_pq = args.dataset / "master_measurements.parquet"

    t0 = time.time()
    print("\n[2/5] CONCAT shootout (read 7.9k parquets -> single frame):")
    concat = _bench("concat", CONCAT_VARIANTS, args.repeat, corpus=corpus,
                    master_pq=None, stage_root=args.stage_root, threads=args.threads)

    print("\n[3/5] MACRO end-to-end compile (concat+derive+write master+split):")
    macro = _bench("macro", MACRO_VARIANTS, args.repeat, corpus=corpus,
                   master_pq=None, stage_root=args.stage_root, threads=args.threads)

    print("\n[4/5] MICRO per-stage shootouts (on the real master parquet):")
    micro = _bench("micro", MICRO_VARIANTS, args.repeat, corpus=None,
                   master_pq=master_pq, stage_root=args.stage_root, threads=args.threads)

    print("\n[5/5] Verifying polars vs pandas output equivalence ...")
    equiv = _verify_equivalence(args.stage_root, args.threads, corpus)
    print(f"  {equiv}")

    out = {
        "meta": {
            "dataset": str(args.dataset),
            "repeat": args.repeat,
            "threads": args.threads,
            "elapsed_orchestration_s": time.time() - t0,
        },
        "equivalence": equiv,
        "concat": {k: _summary_stats(v) for k, v in concat.items()},
        "macro": {k: _summary_stats(v) for k, v in macro.items()},
        "micro": {k: _summary_stats(v) for k, v in micro.items()},
    }
    args.results_json.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.results_json}")


if __name__ == "__main__":
    main()
