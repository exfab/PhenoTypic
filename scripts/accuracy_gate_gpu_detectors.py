"""Measure objmask IoU against a synthetic plate with known ground truth.

Decides whether the costly defaults ship: the DINO 1:1 processor policy
(FssDino, Insid3) and Sam2's crop_n_layers. The code changes land regardless;
only the defaults are gated.

Spec: docs/superpowers/specs/2026-07-08-gpu-detect-fixes/

Insid3 needs the gated DINOv3 weights:
    PHENOTYPIC_ACCEPT_MODEL_LICENSE=dinov3 uv run python scripts/accuracy_gate_gpu_detectors.py
"""

from __future__ import annotations

import contextlib
import time

import numpy as np

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate


def tiled_plate(rows: int = 3, cols: int = 4) -> tuple[Image, np.ndarray]:
    """Replicate synth_plate into a plate larger than SAM2's 1024 encoder.

    Colony diameters stay 32-44 px; only the plate grows, so the detectors face
    real downsampling instead of upsampling. Labels are offset per block so the
    ground truth stays instance-correct.

    Returns:
        ``(Image, truth_objmap)`` — the image carries only RGB.
    """
    src = load_synth_yeast_plate()
    rgb = np.asarray(src.rgb[:])
    om = np.asarray(src.objmap[:])
    h, w = om.shape
    n = int(om.max())

    big_om = np.zeros((h * rows, w * cols), dtype=np.uint16)
    k = 0
    for r in range(rows):
        for c in range(cols):
            blk = om.astype(np.uint16).copy()
            blk[blk > 0] += k
            big_om[r * h : (r + 1) * h, c * w : (c + 1) * w] = blk
            k += n
    return Image(np.tile(rgb, (rows, cols, 1))), big_om


def mask_iou(pred: np.ndarray, truth: np.ndarray) -> float:
    """Foreground IoU of two boolean masks."""
    pred, truth = pred.astype(bool), truth.astype(bool)
    union = (pred | truth).sum()
    return float((pred & truth).sum() / union) if union else 0.0


@contextlib.contextmanager
def legacy_processor_policy():
    """Restore the pre-fix 224 classification preset for the baseline run."""
    import phenotypic.detect.nn._dino_support as ds

    saved = dict(ds.NATIVE_PROCESSOR_KWARGS)
    ds.NATIVE_PROCESSOR_KWARGS.clear()
    try:
        yield
    finally:
        ds.NATIVE_PROCESSOR_KWARGS.update(saved)


def evaluate(detector, label: str, *, legacy: bool = False) -> float:
    image, truth_om = tiled_plate()
    truth = truth_om > 0
    ctx = legacy_processor_policy() if legacy else contextlib.nullcontext()
    t0 = time.perf_counter()
    with ctx:
        # apply() defaults to inplace=False and returns a copy; reading back
        # from `image` would measure the empty input objmap.
        result = detector.apply(image)
    elapsed = time.perf_counter() - t0
    pred = np.asarray(result.objmask[:])
    iou = mask_iou(pred, truth)
    print(
        f"{label:<40} IoU {iou:.4f}  "
        f"objects {result.num_objects:>5} / {int(truth_om.max())}  "
        f"{elapsed:7.1f}s"
    )
    return iou


if __name__ == "__main__":
    from phenotypic.detect.nn import FssDinoDetector, Insid3Detector, Sam2Detector

    print("plate: 1800x3200, 1152 ground-truth colonies\n")

    evaluate(
        FssDinoDetector(dino_version=2, dino_size="small", device="auto"),
        "FssDino  224 preset (pre-fix)",
        legacy=True,
    )
    evaluate(
        FssDinoDetector(dino_version=2, dino_size="small", device="auto"),
        "FssDino  1:1 native (post-fix)",
    )

    evaluate(
        Insid3Detector(dino_size="small", device="auto"),
        "Insid3   224 preset (pre-fix)",
        legacy=True,
    )
    evaluate(
        Insid3Detector(dino_size="small", device="auto"),
        "Insid3   1:1 native (post-fix)",
    )

    # Sam2's crop_n_layers is the one costly default with no measurement behind
    # it — it ships on the strength of SAM2's four documented defenses (edge
    # rejection, crop overlap, full-image fallback, resolution-preferring NMS),
    # not on evidence. SAM2 is ungated and installed, so measure it.
    evaluate(
        Sam2Detector(model_size="tiny", crop_n_layers=0, device="auto"),
        "Sam2     crop_n_layers=0 (old default)",
    )
    evaluate(
        Sam2Detector(model_size="tiny", crop_n_layers=1, device="auto"),
        "Sam2     crop_n_layers=1 (new default)",
    )
