"""Is Sam2 repeatable, and does a replayed mask reproduce the single pass?

    python diag_sam2_repeatability.py <pipeline> <image.tiff> <dataset> \
        <staged-output> <reference-dir>

On one image, with one resident model:

1. raw result on the in-memory Stage-1 image, twice   -> A1, A2  (repeatability)
2. raw result on the staged store's input layer        -> C       (store input)
3. the post-detector pipeline replaying A1 on the in-memory image, versus the
   single-pass objmap already recorded under <reference-dir>  (downstream path)
4. a full single pass (pipeline.apply) on a fresh image, versus that recorded
   objmap                                               (run-to-run, end to end)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from phenotypic import GridImage, ImagePipeline
from phenotypic._cli._cli_pipeline_split import split_pipeline_at_gpu
from phenotypic._cli._cli_replay_detector import build_replay_pipeline
from phenotypic.sdk_._io_constants import zarr_store_path
from phenotypic.sdk_._operation_tree import get_at_path


def raw(detector, array: np.ndarray) -> np.ndarray:
    return detector._infer_batch(detector._collate([detector._preprocess(array)]))[0]


def report(name: str, a: np.ndarray, b: np.ndarray) -> None:
    n = int(np.count_nonzero(a != b)) if a.shape == b.shape else -1
    labels = (len(np.unique(a)) - 1, len(np.unique(b)) - 1)
    fg = int(np.count_nonzero((a > 0) != (b > 0))) if a.shape == b.shape else -1
    print(f"{name:<44} label-differing px={n}  fg-differing px={fg}  labels={labels}")


def check_repeatability(
    pipeline_path: Path, tiff: Path, dataset: str, staged: Path, reference: Path
) -> None:
    pipeline = ImagePipeline.from_json(pipeline_path)
    plan = split_pipeline_at_gpu(pipeline)
    detector = plan.gpu_detector
    detector._ensure_model_loaded()
    layer = detector.input_layer

    image = GridImage.imread(tiff, bit_depth=16, nrows=8, ncols=12)
    plan.pre_pipeline.apply(image, inplace=True)
    stored = GridImage.load_zarr(zarr_store_path(staged, dataset, tiff.stem))

    a1 = raw(detector, getattr(image, layer)[:])
    a2 = raw(detector, getattr(image, layer)[:])
    c = raw(detector, getattr(stored, layer)[:])
    report("raw A1 vs A2 (same input, twice)", a1, a2)
    report("raw A1 vs C (memory vs store input)", a1, c)

    expected = np.load(reference / dataset / f"{tiff.stem}.objmap.npy")
    replayed = image.copy()
    build_replay_pipeline(plan, a1).apply(replayed, inplace=True)
    report("replay(A1) objmap vs recorded single pass", replayed.objmap[:], expected)

    fresh = GridImage.imread(tiff, bit_depth=16, nrows=8, ncols=12)
    pipeline.apply(fresh, inplace=True)
    report("new single pass vs recorded single pass", fresh.objmap[:], expected)
    report("new single pass vs replay(A1)", fresh.objmap[:], replayed.objmap[:])
    print("gpu_path", plan.gpu_path, "same object:",
          get_at_path(pipeline, plan.gpu_path) is not None)


if __name__ == "__main__":
    check_repeatability(
        Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]),
        Path(sys.argv[5]),
    )
