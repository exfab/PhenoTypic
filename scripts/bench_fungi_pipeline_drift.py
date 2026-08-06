"""A/B drift harness for the FilamentousFungiPipeline (tutorial 10).

Runs the exact tutorial pipeline on load_fungi_plate() under float64 (old) and
float32 (new) by toggling ImageData.FLOAT_LAYER_DTYPE, then reports segmentation
drift (objmap) and measurement drift. Also runs float64 twice as a determinism
control so any float32 drift is attributable to dtype, not RNG.

    uv run python scripts/bench_fungi_pipeline_drift.py
"""
from __future__ import annotations

import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np

from phenotypic._core._image_parts._image_data_manager import ImageData


def _run(dtype):
    """Run the tutorial fungi pipeline + measurements under the given layer dtype."""
    ImageData.FLOAT_LAYER_DTYPE = dtype

    from phenotypic.data import load_fungi_plate
    from phenotypic.prefab import FilamentousFungiPipeline
    from phenotypic.detect import ManualGridPointDetector
    from phenotypic.measure import (
        MeasureSymZones, MeasureSize, MeasureIntensity,
        MeasureShape, MeasureBounds, MeasureColor, MeasureTexture,
    )

    plate = load_fungi_plate()
    pipe = FilamentousFungiPipeline(
            inoculum_detector=ManualGridPointDetector(
                    coord1=(230, 240), coord2=(630, 640), shape="disk", width=100,
            ),
            ignore_borders=False,
    )
    t0 = time.perf_counter()
    plate = pipe.apply(plate)
    elapsed = time.perf_counter() - t0

    objmap = np.asarray(plate.objmap[:]).copy()
    meas = {}
    ops = {
        "symmetric_zones": MeasureSymZones(),  # the tutorial measurement
        "size"           : MeasureSize(),
        "intensity"      : MeasureIntensity(),
        "shape"          : MeasureShape(),
        "bounds"         : MeasureBounds(),
        "color"          : MeasureColor(),
        "texture"        : MeasureTexture(),
    }
    for name, op in ops.items():
        try:
            meas[name] = op.measure(plate).reset_index()
        except Exception as e:  # noqa
            meas[name] = e
    return {
        "gray_dtype"  : str(plate.gray[:].dtype),
        "detect_dtype": str(plate.detect_mat[:].dtype),
        "objmap"      : objmap,
        "num_objects" : int(plate.num_objects),
        "elapsed"     : elapsed,
        "meas"        : meas,
    }


def _seg_drift(a, b):
    """Segmentation drift between two label maps."""
    fa = a > 0
    fb = b > 0
    inter = int(np.count_nonzero(fa & fb))
    union = int(np.count_nonzero(fa | fb))
    iou = inter / union if union else 1.0
    mismatch = int(np.count_nonzero(fa != fb))
    exact_label = float(np.mean(a == b))  # only meaningful if labels align
    # per-object IoU: for each label in A, best-overlap label in B
    obj_ious = []
    for lab in np.unique(a[a > 0]):
        amask = a == lab
        overlap_labels = b[amask]
        overlap_labels = overlap_labels[overlap_labels > 0]
        if overlap_labels.size == 0:
            obj_ious.append(0.0)
            continue
        best = np.bincount(overlap_labels).argmax()
        bmask = b == best
        i = int(np.count_nonzero(amask & bmask))
        u = int(np.count_nonzero(amask | bmask))
        obj_ious.append(i / u if u else 0.0)
    obj_ious = np.array(obj_ious) if obj_ious else np.array([1.0])
    return {
        "fg_iou"              : iou,
        "fg_pixels_a"         : int(fa.sum()),
        "fg_pixels_b"         : int(fb.sum()),
        "fg_pixel_delta"      : int(fb.sum()) - int(fa.sum()),
        "mismatch_pixels"     : mismatch,
        "total_pixels"        : int(a.size),
        "exact_label_frac"    : exact_label,
        "obj_iou_min"         : float(obj_ious.min()),
        "obj_iou_mean"        : float(obj_ious.mean()),
        "n_obj_iou_below_0.99": int(np.count_nonzero(obj_ious < 0.99)),
    }


def _meas_drift(ma, mb):
    out = {}
    for name in ma:
        da, db = ma[name], mb[name]
        if isinstance(da, Exception) or isinstance(db, Exception):
            out[
                name] = f"err(a={isinstance(da, Exception)},b={isinstance(db, Exception)})"
            continue
        na = da.select_dtypes(include=[np.number]).astype(np.float64)
        nb = db.select_dtypes(include=[np.number]).astype(np.float64)
        cols = [c for c in na.columns if c in nb.columns]
        max_rel = 0.0;
        max_abs = 0.0;
        worst = ""
        shape_mismatch = na.shape[0] != nb.shape[0]
        for c in cols:
            va, vb = na[c].values, nb[c].values
            if va.shape != vb.shape:
                shape_mismatch = True
                continue
            ad = np.abs(va - vb)
            rel = np.nanmax(ad / np.maximum(np.abs(va), 1e-12)) if ad.size else 0.0
            if rel > max_rel:
                max_rel, worst = float(rel), c
            max_abs = max(max_abs, float(np.nanmax(ad)) if ad.size else 0.0)
        out[name] = {
            "rows_a"        : int(na.shape[0]), "rows_b": int(nb.shape[0]),
            "shape_mismatch": shape_mismatch,
            "max_rel"       : max_rel, "max_abs": max_abs, "worst_col": worst,
        }
    return out


if __name__ == "__main__":
    print("Running FilamentousFungiPipeline under float64 (baseline)...")
    f64 = _run(np.float64)
    print("Running float64 again (determinism control)...")
    f64b = _run(np.float64)
    print("Running under float32 (new contract)...")
    f32 = _run(np.float32)

    print("\n=== dtypes / object counts / timing ===")
    for tag, r in (("float64", f64), ("float64#2", f64b), ("float32", f32)):
        print(f"  {tag:9s} gray={r['gray_dtype']:7s} detect={r['detect_dtype']:7s} "
              f"num_objects={r['num_objects']:4d}  pipeline={r['elapsed']:.2f}s")

    print("\n=== SEGMENTATION: float64 vs float64#2 (determinism control) ===")
    ctrl = _seg_drift(f64["objmap"], f64b["objmap"])
    for k, v in ctrl.items():
        print(f"  {k:22s} {v}")

    print("\n=== SEGMENTATION: float64 vs float32 (DTYPE DRIFT) ===")
    drift = _seg_drift(f64["objmap"], f32["objmap"])
    for k, v in drift.items():
        print(f"  {k:22s} {v}")

    print("\n=== MEASUREMENT DRIFT: float64 vs float32 ===")
    md = _meas_drift(f64["meas"], f32["meas"])
    print(
        f"  {'measure':18s} {'rows_a':>6s} {'rows_b':>6s} {'shapeMM':>8s} {'max_rel':>11s} {'max_abs':>11s}  worst")
    for name, m in md.items():
        if isinstance(m, str):
            print(f"  {name:18s} {m}")
        else:
            print(
                f"  {name:18s} {m['rows_a']:6d} {m['rows_b']:6d} {str(m['shape_mismatch']):>8s} "
                f"{m['max_rel']:11.3e} {m['max_abs']:11.3e}  {m['worst_col']}")
