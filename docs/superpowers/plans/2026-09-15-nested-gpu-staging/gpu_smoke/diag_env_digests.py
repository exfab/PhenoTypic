"""Print digests that should be identical across jobs if the code is reproducible.

    python diag_env_digests.py cpu <pipeline> <image.tiff>
    python diag_env_digests.py gpu <pipeline> <fixed-input.ome.zarr> <out.npy>

cpu: node, CPU model and thread count, then sha256 of gray after the crop and
     of gray / detect_mat after all Stage-1 operations. Run on several node
     types: equal digests mean the CPU steps are reproducible across nodes.
gpu: Sam2's raw result on a FIXED input read from a store (so the CPU steps
     are out of the picture), saved to <out.npy> and digested. Run as separate
     jobs: equal digests mean SAM2 is reproducible across processes.
"""

from __future__ import annotations

import hashlib
import os
import platform
import sys
from pathlib import Path

import numpy as np

from phenotypic import GridImage, ImagePipeline
from phenotypic._cli._cli_pipeline_split import split_pipeline_at_gpu


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def digest_cpu(pipeline_path: Path, tiff: Path) -> None:
    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipeline_path))
    ops = dict(plan.pre_pipeline.get_ops())
    crop = ImagePipeline(ops={"CropImage": ops["CropImage"]})
    print(f"node={platform.node()} cpu={cpu_model()!r} "
          f"affinity={len(os.sched_getaffinity(0))} OMP={os.environ.get('OMP_NUM_THREADS')}")
    image = GridImage.imread(tiff, bit_depth=16, nrows=8, ncols=12)
    print(f"gray@read      {sha(image.gray[:])}")
    crop.apply(image, inplace=True)
    print(f"gray@crop      {sha(image.gray[:])}")
    full = GridImage.imread(tiff, bit_depth=16, nrows=8, ncols=12)
    plan.pre_pipeline.apply(full, inplace=True)
    print(f"gray@stage1    {sha(full.gray[:])}")
    print(f"detect@stage1  {sha(full.detect_mat[:])}")


def digest_gpu(pipeline_path: Path, store: Path, out: Path) -> None:
    import torch

    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipeline_path))
    detector = plan.gpu_detector
    detector._ensure_model_loaded()
    array = getattr(GridImage.load_zarr(store), detector.input_layer)[:]
    print(f"node={platform.node()} gpu={torch.cuda.get_device_name(0)} "
          f"cudnn.benchmark={torch.backends.cudnn.benchmark} "
          f"deterministic={torch.backends.cudnn.deterministic}")
    print(f"input          {sha(array)}")
    result = detector._infer_batch(detector._collate([detector._preprocess(array)]))[0]
    np.save(out, result)
    print(f"raw            {sha(result)}  labels={len(np.unique(result)) - 1}")


if __name__ == "__main__":
    if sys.argv[1] == "cpu":
        digest_cpu(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        digest_gpu(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
