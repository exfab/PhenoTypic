"""Does the Stage-1 store hand Stage 2 the same pixels a single pass would see?

    python diag_stage1_roundtrip.py <pipeline> <image.tiff> <dataset> <staged-output>

Re-runs Stage 1's operations in memory on a freshly read image, loads the
staged run's store for the same image, and compares rgb / gray / detect_mat
exactly -- then compares the uint8 array Sam2's ``_preprocess`` would build
from each ``input_layer``, which is what the model actually sees. CPU only.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

from phenotypic import GridImage, ImagePipeline
from phenotypic._cli._cli_pipeline_split import split_pipeline_at_gpu
from phenotypic.sdk_._io_constants import zarr_store_path
from phenotypic.sdk_._operation_tree import get_at_path


def describe(name: str, a: np.ndarray, b: np.ndarray) -> None:
    same_shape = a.shape == b.shape
    line = f"{name:<12} mem {a.dtype}{a.shape} | store {b.dtype}{b.shape}"
    if not same_shape:
        print(line + "  SHAPE DIFFERS")
        return
    diff = a.astype(np.float64) - b.astype(np.float64)
    n = int(np.count_nonzero(diff))
    print(
        f"{line}  differing={n} max|diff|={np.abs(diff).max():.3g} "
        f"mem[min,max]=[{a.min():.6g},{a.max():.6g}] "
        f"store[min,max]=[{b.min():.6g},{b.max():.6g}]"
    )


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def compare_stage1(pipeline_path: Path, tiff: Path, dataset: str, staged: Path) -> None:
    pipeline = ImagePipeline.from_json(pipeline_path)
    plan = split_pipeline_at_gpu(pipeline)
    detector = get_at_path(pipeline, plan.gpu_path)
    print(f"gpu_path={plan.gpu_path} input_layer={detector.input_layer} "
          f"scaling={detector.input_scaling}")
    print("stage1 ops:", [type(op).__name__ for _, op in plan.pre_pipeline.get_ops().items()])
    print("stage2_prefix:", plan.stage2_prefix)

    fresh = GridImage.imread(tiff, bit_depth=16, nrows=8, ncols=12)
    plan.pre_pipeline.apply(fresh, inplace=True)
    stored = GridImage.load_zarr(zarr_store_path(staged, dataset, tiff.stem))

    for layer in ("rgb", "gray", "detect_mat"):
        describe(layer, getattr(fresh, layer)[:], getattr(stored, layer)[:])

    layer = detector.input_layer
    mem_u8 = detector._preprocess(getattr(fresh, layer)[:])
    store_u8 = detector._preprocess(getattr(stored, layer)[:])
    describe(f"u8({layer})", mem_u8, store_u8)
    print(f"sha mem={digest(mem_u8)} store={digest(store_u8)}")


if __name__ == "__main__":
    compare_stage1(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]))
