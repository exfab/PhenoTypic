"""Arm B: run a pipeline in ONE pass per image and record what it produced.

Mirrors the CLI worker's compute (`_cli_process_single.py`): the same
`imread` arguments and `apply_and_measure(image, inplace=True,
apply_post=False)`, without the CLI around it. The CLI is avoided on purpose:
on this branch a local CLI run of a nested-GPU pipeline is itself staged.

    python single_pass_reference.py <pipeline> <input-root> <out-dir>

Writes, per image, ``<out>/<dataset>/<stem>.objmap.npy`` and
``<out>/<dataset>/<stem>.measurements.parquet``. The model loads once and is
reused across images, as a resident model would be.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from phenotypic import GridImage, ImagePipeline


def record_single_pass(pipeline_path: Path, input_root: Path, out: Path) -> int:
    pipeline = ImagePipeline.from_json(pipeline_path)
    images = sorted(input_root.glob("*/*.tiff"))
    if not images:
        print(f"no images under {input_root}")
        return 1
    failures = 0
    for path in images:
        dataset = path.parent.name
        dest = out / dataset
        dest.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        try:
            image = GridImage.imread(path, bit_depth=16, nrows=8, ncols=12)
            frame = pipeline.apply_and_measure(
                image, inplace=True, apply_post=False
            )
        except Exception as exc:  # noqa: BLE001 -- record and continue
            failures += 1
            print(f"FAIL {dataset}/{path.stem}: {type(exc).__name__}: {exc}")
            continue
        np.save(dest / f"{path.stem}.objmap.npy", image.objmap[:])
        frame.to_parquet(dest / f"{path.stem}.measurements.parquet")
        print(
            f"ok   {dataset}/{path.stem}: {image.num_objects} objects, "
            f"{len(frame)} rows, {time.perf_counter() - started:.1f}s"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(
        record_single_pass(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
    )
