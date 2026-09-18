"""Given ONE preprocessed image, is replaying Stage 2 the same as running live?

    python diag_replay_equivalence.py <pipeline> <image.tiff>

Stage-1 operations run once. From that single image:

  live    = post_pipeline applied as-is (the real Sam2 runs inside the composite)
  live2   = the same again                         (post-detector repeatability)
  replay  = post_pipeline with a ReplayDetector holding Sam2's raw result for
            that same image                         (what Stage 3 does)

live vs replay isolates the staging mechanism from anything upstream of it.
Measurements are compared too, since Stage 3 measures the replayed image.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from phenotypic import GridImage, ImagePipeline
from phenotypic._cli._cli_pipeline_split import split_pipeline_at_gpu
from phenotypic._cli._cli_replay_detector import build_replay_pipeline
from phenotypic.schema import OBJECT


def raw(detector, array: np.ndarray) -> np.ndarray:
    return detector._infer_batch(detector._collate([detector._preprocess(array)]))[0]


def objmap_report(name: str, a: np.ndarray, b: np.ndarray) -> None:
    print(f"{name:<22} objmap differing px={int(np.count_nonzero(a != b))} "
          f"labels={(len(np.unique(a)) - 1, len(np.unique(b)) - 1)}")


def frame_report(name: str, a: pd.DataFrame, b: pd.DataFrame) -> None:
    key = str(OBJECT.LABEL)
    a = a.sort_values(key).reset_index(drop=True)
    b = b.sort_values(key).reset_index(drop=True)
    worst, col = 0.0, None
    for c in a.columns:
        if c in b.columns and pd.api.types.is_numeric_dtype(a[c]):
            d = np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)), initial=0.0)
            if d > worst:
                worst, col = float(d), c
    print(f"{name:<22} measurements rows={len(a)}/{len(b)} largest diff={worst:.3g} ({col})")


def check_equivalence(pipeline_path: Path, tiff: Path) -> None:
    pipeline = ImagePipeline.from_json(pipeline_path)
    plan = split_pipeline_at_gpu(pipeline)
    plan.gpu_detector._ensure_model_loaded()

    staged_input = GridImage.imread(tiff, bit_depth=16, nrows=8, ncols=12)
    plan.pre_pipeline.apply(staged_input, inplace=True)

    live = staged_input.copy()
    plan.post_pipeline.apply(live, inplace=True)
    live_frame = plan.post_pipeline.measure(live, apply_post=False)

    live2 = staged_input.copy()
    plan.post_pipeline.apply(live2, inplace=True)

    result = raw(plan.gpu_detector, getattr(staged_input, plan.gpu_detector.input_layer)[:])
    replay_pipeline = build_replay_pipeline(plan, result)
    replayed = staged_input.copy()
    replay_pipeline.apply(replayed, inplace=True)
    replay_frame = replay_pipeline.measure(replayed, apply_post=False)

    objmap_report("live vs live2", live.objmap[:], live2.objmap[:])
    objmap_report("live vs replay", live.objmap[:], replayed.objmap[:])
    frame_report("live vs replay", live_frame, replay_frame)


if __name__ == "__main__":
    check_equivalence(Path(sys.argv[1]), Path(sys.argv[2]))
