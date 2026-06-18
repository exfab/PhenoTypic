"""Compare default polars vs polars-lts-cpu (the no-AVX2 build) on the same node.

The instruction-set problem is specifically that the default polars wheel uses
AVX2/AVX512 and SIGILLs on pre-AVX2 nodes. ``polars-lts-cpu`` is the official
drop-in wheel compiled for a baseline instruction set, so it runs everywhere.
This harness quantifies the *cost of that compatibility* by running identical
pure-polars variants under both interpreters on this (modern) node.

Run AFTER the main benchmark so the two never contend for the 12 cores::

    uv run python diagnostics/polars_vs_duckdb/run_lts_compare.py \
        --corpus /tmp/pl_diag_sample_clean/results \
        --master-parquet <dataset>/master_measurements.parquet \
        --lts-python /tmp/pl_lts_venv/bin/python --repeat 5
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "bench_runner.py"

# Only pure-polars variants (no duckdb / no phenotypic import) so the lts venv
# -- which has just polars-lts-cpu + pandas + pyarrow -- can run them.
CONCAT = ["concat_polars_native", "concat_polars_read", "concat_polars_scan"]
MACRO = ["macro_polars_native"]
MICRO = ["wpq_polars", "wcsv_polars", "split_polars"]


def _run(python: str, family: str, variant: str, *, corpus, master_pq, out, threads):
    cmd = [python, str(RUNNER), "--family", family, "--variant", variant,
           "--out", str(out), "--threads", str(threads)]
    if corpus:
        cmd += ["--corpus", str(corpus)]
    if master_pq:
        cmd += ["--master-parquet", str(master_pq)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"    !! {variant} FAILED under {python}:\n{p.stderr[-1500:]}", file=sys.stderr)
        return None
    return json.loads(p.stdout.strip().splitlines()[-1])


def _median_ms(python, family, variant, *, corpus, master_pq, stage_root, threads, repeat):
    out = stage_root / "lts_out" / variant
    walls = []
    for i in range(repeat + 1):
        if out.exists():
            shutil.rmtree(out)
        rec = _run(python, family, variant, corpus=corpus, master_pq=master_pq,
                   out=out, threads=threads)
        if rec and i > 0:
            walls.append(rec["wall_total_s"])
    return statistics.median(walls) * 1000 if walls else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--master-parquet", type=Path, required=True)
    ap.add_argument("--default-python", default=sys.executable)
    ap.add_argument("--lts-python", required=True)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--stage-root", type=Path, default=Path("/tmp/pl_diag"))
    ap.add_argument("--results-json", type=Path, default=HERE / "results_lts.json")
    args = ap.parse_args()

    jobs = [("concat", v, args.corpus, None) for v in CONCAT]
    jobs += [("macro", v, args.corpus, None) for v in MACRO]
    jobs += [("micro", v, None, args.master_parquet) for v in MICRO]

    print(f"{'variant':24s} {'default ms':>12s} {'lts-cpu ms':>12s} {'lts/default':>12s}")
    out = {}
    for family, variant, corpus, master_pq in jobs:
        d = _median_ms(args.default_python, family, variant, corpus=corpus,
                       master_pq=master_pq, stage_root=args.stage_root,
                       threads=args.threads, repeat=args.repeat)
        l = _median_ms(args.lts_python, family, variant, corpus=corpus,
                       master_pq=master_pq, stage_root=args.stage_root,
                       threads=args.threads, repeat=args.repeat)
        ratio = l / d if d and d == d else float("nan")
        out[variant] = {"default_ms": d, "lts_ms": l, "ratio": ratio}
        print(f"{variant:24s} {d:12.1f} {l:12.1f} {ratio:12.2f}x")

    args.results_json.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.results_json}")


if __name__ == "__main__":
    main()
