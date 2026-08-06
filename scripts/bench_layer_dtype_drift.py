"""Benchmark harness: per-measure drift / timing / dtypes + HDF5 layer sizes.

Records each measurement's output, timing and the saved HDF5 layer breakdown for
a detected synthetic plate, tagged so two runs can be diffed. To produce a
float64 baseline for an A/B without checking out old code, force the layer dtype
before running (see ``scripts/bench_fungi_pipeline_drift.py`` for the in-process
toggle pattern via ``ImageData.FLOAT_LAYER_DTYPE``).

    uv run python scripts/bench_layer_dtype_drift.py --tag float32
    uv run python scripts/bench_layer_dtype_drift.py --compare baseline float32
    # custom output dir:
    uv run python scripts/bench_layer_dtype_drift.py --tag float32 --out /path/to/dir
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

OUT = Path(tempfile.gettempdir()) / "phenotypic_bench_layer_dtype"


def _build_image():
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector

    img = load_synth_yeast_plate()
    OtsuDetector().apply(img)
    return img


def _measure_ops():
    from phenotypic.measure import (
        MeasureBounds, MeasureColor, MeasureIntensity, MeasureShape,
        MeasureSize, MeasureTexture, MeasureSymZones,
    )

    return {
        "bounds"         : MeasureBounds(),
        "size"           : MeasureSize(),
        "intensity"      : MeasureIntensity(),
        "shape"          : MeasureShape(),
        "texture"        : MeasureTexture(),
        "color"          : MeasureColor(),
        "symmetric_zones": MeasureSymZones(),
    }


def run(tag: str):
    import phenotypic  # noqa

    img = _build_image()
    OUT.mkdir(parents=True, exist_ok=True)
    tagdir = OUT / tag
    tagdir.mkdir(exist_ok=True)

    summary = {
        "tag"             : tag,
        "gray_dtype"      : str(img.gray[:].dtype),
        "detect_mat_dtype": str(img.detect_mat[:].dtype),
        "objmap_dtype"    : str(img.objmap[:].dtype),
        "num_objects"     : int(img.num_objects),
        "measures"        : {},
    }

    for name, op in _measure_ops().items():
        try:
            t0 = time.perf_counter()
            # average a few runs for stable timing
            for _ in range(3):
                df = op.measure(img)
            dt = (time.perf_counter() - t0) / 3
        except Exception as e:  # noqa
            summary["measures"][name] = {"error": f"{type(e).__name__}: {e}"}
            continue

        df = df.reset_index()
        df.to_parquet(tagdir / f"{name}.parquet")
        numeric = df.select_dtypes(include=[np.number])
        summary["measures"][name] = {
            "seconds"       : round(dt, 5),
            "rows"          : int(df.shape[0]),
            "cols"          : list(df.columns),
            "n_numeric_cols": int(numeric.shape[1]),
            "colsum"        : {c: float(np.nansum(numeric[c].values.astype(np.float64)))
                               for c in numeric.columns},
        }

    # whole-file HDF size
    import h5py

    hp = tagdir / "image.h5"
    img.save2hdf5(hp)
    summary["hdf5_total_bytes"] = os.path.getsize(hp)
    layer_sizes = {}
    with h5py.File(hp, "r") as f:
        def visit(n, o):
            if isinstance(o, h5py.Dataset):
                layer_sizes[n] = o.id.get_storage_size()

        f.visititems(visit)
    summary["hdf5_layers"] = layer_sizes

    (tagdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"[{tag}] gray={summary['gray_dtype']} detect_mat={summary['detect_mat_dtype']} "
        f"objs={summary['num_objects']}  hdf5={summary['hdf5_total_bytes'] / 1024:.1f} KiB")
    for n, m in summary["measures"].items():
        if "error" in m:
            print(f"  {n:18s} ERROR: {m['error']}")
        else:
            print(f"  {n:18s} {m['seconds'] * 1000:8.2f} ms  rows={m['rows']:4d} "
                  f"numcols={m['n_numeric_cols']}")


def compare(tag_a: str, tag_b: str):
    a = json.loads((OUT / tag_a / "summary.json").read_text())
    b = json.loads((OUT / tag_b / "summary.json").read_text())
    print(f"\n=== {tag_a} vs {tag_b} ===")
    print(f"gray dtype       : {a['gray_dtype']:>10s} -> {b['gray_dtype']}")
    print(f"detect_mat dtype : {a['detect_mat_dtype']:>10s} -> {b['detect_mat_dtype']}")
    print(f"num_objects      : {a['num_objects']} -> {b['num_objects']}"
          f"  {'OK' if a['num_objects'] == b['num_objects'] else 'MISMATCH!'}")
    ta, tb = a["hdf5_total_bytes"], b["hdf5_total_bytes"]
    print(
        f"hdf5 total       : {ta / 1024:8.1f} -> {tb / 1024:8.1f} KiB  ({100 * (1 - tb / ta):+.1f}%)")
    for layer in sorted(a.get("hdf5_layers", {})):
        la = a["hdf5_layers"][layer];
        lb = b["hdf5_layers"].get(layer, 0)
        if la:
            print(
                f"  {layer:22s} {la / 1024:8.1f} -> {lb / 1024:8.1f} KiB ({100 * (1 - lb / la):+.1f}%)")

    print("\n--- measurement value drift (max |rel| per measure) & timing ---")
    print(
        f"{'measure':18s} {'maxrel':>12s} {'maxabs':>12s} {'t_a(ms)':>9s} {'t_b(ms)':>9s} {'speedup':>8s}")
    for name in a["measures"]:
        ma = a["measures"][name];
        mb = b["measures"].get(name, {})
        if "error" in ma or "error" in mb:
            print(f"{name:18s}  (error a={'error' in ma} b={'error' in mb})")
            continue
        try:
            da = pd.read_parquet(OUT / tag_a / f"{name}.parquet")
            db = pd.read_parquet(OUT / tag_b / f"{name}.parquet")
        except FileNotFoundError:
            continue
        na = da.select_dtypes(include=[np.number]).astype(np.float64)
        nb = db.select_dtypes(include=[np.number]).astype(np.float64)
        cols = [c for c in na.columns if c in nb.columns]
        max_rel = 0.0;
        max_abs = 0.0;
        worst_col = ""
        for c in cols:
            va = na[c].values;
            vb = nb[c].values
            if va.shape != vb.shape:
                max_rel = float("nan");
                break
            absdiff = np.abs(va - vb)
            denom = np.maximum(np.abs(va), 1e-12)
            rel = np.nanmax(absdiff / denom) if absdiff.size else 0.0
            ab = np.nanmax(absdiff) if absdiff.size else 0.0
            if rel > max_rel:
                max_rel = float(rel);
                worst_col = c
            max_abs = max(max_abs, float(ab))
        ta_ms = ma["seconds"] * 1000;
        tb_ms = mb["seconds"] * 1000
        speed = ta_ms / tb_ms if tb_ms else float("nan")
        print(f"{name:18s} {max_rel:12.3e} {max_abs:12.3e} {ta_ms:9.2f} {tb_ms:9.2f} "
              f"{speed:7.2f}x  worst={worst_col}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument(
            "--out", type=Path, default=OUT,
            help="Output directory for per-tag artifacts (default: system temp).",
    )
    args = ap.parse_args()
    OUT = args.out
    if args.compare:
        compare(*args.compare)
    elif args.tag:
        run(args.tag)
    else:
        ap.error("need --tag or --compare")
