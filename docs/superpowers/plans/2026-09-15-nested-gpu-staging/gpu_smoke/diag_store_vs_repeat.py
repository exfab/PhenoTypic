"""Separate a store round-trip effect from non-repeatable preprocessing.

    python diag_store_vs_repeat.py <pipeline> <image.tiff> <scratch-dir>

A. crop only:            in memory  vs  save2zarr -> load_zarr      (gray, detect_mat)
B. Stage-1 ops, twice:   run 1      vs  run 2                       (repeatability)
C. Stage-1 ops:          in memory  vs  save2zarr -> load_zarr      (round trip)

Also prints, per layer, what the store actually holds on disk (dtype), so a
lossy write is visible rather than inferred.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from phenotypic import GridImage, ImagePipeline
from phenotypic._cli._cli_pipeline_split import split_pipeline_at_gpu


def compare(label: str, a: GridImage, b: GridImage) -> None:
    for layer in ("rgb", "gray", "detect_mat"):
        x, y = getattr(a, layer)[:], getattr(b, layer)[:]
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        print(f"{label:<28} {layer:<10} {x.dtype}/{y.dtype} "
              f"differing={int(np.count_nonzero(d))} max|diff|={d.max():.3g}")


def roundtrip(image: GridImage, path: Path) -> GridImage:
    image.save2zarr(path)
    for layer in ("rgb", "gray", "detect_mat"):
        meta = path / layer / "0" / "zarr.json"
        if meta.is_file():
            info = json.loads(meta.read_text())
            print(f"   on disk {layer}/0: {info.get('data_type')} {info.get('shape')}")
        else:
            print(f"   on disk {layer}/0: ABSENT")
    return GridImage.load_zarr(path)


def separate(pipeline_path: Path, tiff: Path, scratch: Path) -> None:
    scratch.mkdir(parents=True, exist_ok=False)
    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipeline_path))
    ops = dict(plan.pre_pipeline.get_ops())
    crop = ImagePipeline(ops={k: v for k, v in ops.items() if k == "CropImage"})

    def read() -> GridImage:
        return GridImage.imread(tiff, bit_depth=16, nrows=8, ncols=12)

    a = read()
    crop.apply(a, inplace=True)
    compare("A crop: mem vs store", a, roundtrip(a, scratch / "crop.ome.zarr"))

    b1 = read()
    plan.pre_pipeline.apply(b1, inplace=True)
    b2 = read()
    plan.pre_pipeline.apply(b2, inplace=True)
    compare("B stage1: run1 vs run2", b1, b2)

    compare("C stage1: mem vs store", b1, roundtrip(b1, scratch / "stage1.ome.zarr"))


if __name__ == "__main__":
    separate(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
