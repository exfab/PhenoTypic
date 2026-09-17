"""Compare the staged run (arm A) with the single-pass reference (arm B).

    python compare_arms.py <staged-output> <reference-dir>

For every image the reference recorded:

1. objmap: identical? If not, how many pixels differ, and the object counts.
2. measurements: the columns both arms have, rows matched on the object label,
   and each numeric column's largest absolute difference.

Exits non-zero if any image is missing from the staged run, any objmap differs,
or any shared numeric column differs by more than ``TOLERANCE``. GPU inference
is not promised bit-identical across batch composition, so a non-zero report
prints the magnitudes -- read them before concluding anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from phenotypic import GridImage
from phenotypic.schema import OBJECT
from phenotypic.sdk_._io_constants import zarr_store_path
from phenotypic.sdk_.ngff_ import MEASUREMENT_TABLE_RELATIVE_PATH

TOLERANCE = 1e-9
LABEL = str(OBJECT.LABEL)


def compare_measurements(staged: pd.DataFrame, reference: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    only_ref = sorted(set(reference.columns) - set(staged.columns))
    only_staged = sorted(set(staged.columns) - set(reference.columns))
    if only_ref:
        print(f"      columns only in reference: {only_ref}")
    if only_staged:
        print(f"      columns only in staged:    {only_staged}")
    if len(staged) != len(reference):
        problems.append(f"row count staged={len(staged)} reference={len(reference)}")
        return problems
    s = staged.sort_values(LABEL).reset_index(drop=True)
    r = reference.sort_values(LABEL).reset_index(drop=True)
    if not (s[LABEL].to_numpy() == r[LABEL].to_numpy()).all():
        problems.append("object labels differ")
        return problems
    shared = [c for c in r.columns if c in s.columns]
    worst = (0.0, None)
    for col in shared:
        a, b = s[col], r[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            x = a.to_numpy(dtype=float)
            y = b.to_numpy(dtype=float)
            both_nan = np.isnan(x) & np.isnan(y)
            if (np.isnan(x) != np.isnan(y)).any():
                problems.append(f"{col}: NaN pattern differs")
                continue
            diff = float(np.max(np.abs(x[~both_nan] - y[~both_nan]), initial=0.0))
            if diff > worst[0]:
                worst = (diff, col)
            if diff > TOLERANCE:
                problems.append(f"{col}: max |diff| {diff:.3g}")
        elif not (a.astype(str).to_numpy() == b.astype(str).to_numpy()).all():
            problems.append(f"{col}: values differ")
    print(
        f"      {len(shared)} shared columns, {len(r)} rows; "
        f"largest numeric diff {worst[0]:.3g} ({worst[1]})"
    )
    return problems


def compare_arms(staged_root: Path, reference_root: Path) -> int:
    failures = 0
    recorded = sorted(reference_root.glob("*/*.objmap.npy"))
    if not recorded:
        print(f"no reference images under {reference_root}")
        return 1
    for ref_objmap in recorded:
        dataset = ref_objmap.parent.name
        stem = ref_objmap.name.removesuffix(".objmap.npy")
        store = zarr_store_path(staged_root, dataset, stem)
        print(f"{dataset}/{stem}")
        if not (store / "zarr.json").is_file():
            print("   MISSING staged store")
            failures += 1
            continue
        expected = np.load(ref_objmap)
        actual = GridImage.load_zarr(store).objmap[:]
        if actual.shape != expected.shape:
            print(f"   OBJMAP shape staged={actual.shape} reference={expected.shape}")
            failures += 1
            continue
        differing = int(np.count_nonzero(actual != expected))
        n_actual = len(np.unique(actual[actual > 0]))
        n_expected = len(np.unique(expected[expected > 0]))
        verdict = "identical" if differing == 0 else f"DIFFERS in {differing} px"
        print(f"   objmap {verdict}; objects staged={n_actual} reference={n_expected}")
        staged_table = pd.read_parquet(store / MEASUREMENT_TABLE_RELATIVE_PATH)
        reference_table = pd.read_parquet(
            ref_objmap.with_name(f"{stem}.measurements.parquet")
        )
        problems = compare_measurements(staged_table, reference_table)
        for problem in problems:
            print(f"   MEASUREMENT {problem}")
        # One count per IMAGE: an image whose objmap and table both differ is
        # one differing image, not two.
        failures += bool(differing or problems)
    print(f"\n{len(recorded)} images compared, {failures} with differences")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(compare_arms(Path(sys.argv[1]), Path(sys.argv[2])))
