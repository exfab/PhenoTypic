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
from dataclasses import dataclass
from typing import Literal

import numpy as np

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate


@dataclass(frozen=True)
class Evaluation:
    """One detector result used by the accuracy and equivalence gates."""

    iou: float
    objects: int
    truth_objects: int
    objmap: np.ndarray
    elapsed: float


def tiled_plate(
    rows: int = 3,
    cols: int = 4,
    *,
    encoding: Literal["uint8", "uint16_nonfull"] = "uint8",
) -> tuple[Image, np.ndarray]:
    """Replicate synth_plate into a plate larger than SAM2's 1024 encoder.

    Colony diameters stay 32-44 px; only the plate grows, so the detectors face
    real downsampling instead of upsampling. Labels are offset per block so the
    ground truth stays instance-correct.

    Returns:
        ``(Image, truth_objmap)`` — the image carries only RGB.
    """
    src = load_synth_yeast_plate()
    rgb = np.asarray(src.rgb[:])
    if encoding == "uint16_nonfull":
        # Exercise the policy choice with a 16-bit acquisition whose brightest
        # sample does not fill the dtype range. This is lossless for uint8.
        rgb = rgb.astype(np.uint16) * 128
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


def evaluate(
    detector,
    label: str,
    *,
    legacy: bool = False,
    encoding: Literal["uint8", "uint16_nonfull"] = "uint8",
) -> Evaluation:
    image, truth_om = tiled_plate(encoding=encoding)
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
    objects = result.num_objects
    truth_objects = int(truth_om.max())
    print(
        f"{label:<40} IoU {iou:.4f}  "
        f"objects {objects:>5} / {truth_objects}  "
        f"{elapsed:7.1f}s"
    )
    return Evaluation(
        iou=iou,
        objects=objects,
        truth_objects=truth_objects,
        objmap=np.asarray(result.objmap[:]),
        elapsed=elapsed,
    )


def canonical_objmap(objmap: np.ndarray) -> np.ndarray:
    """Relabel objects by top-left extent so label-order changes are ignored."""
    records: list[tuple[int, int, int, int]] = []
    for label in np.unique(objmap):
        if label == 0:
            continue
        ys, xs = np.nonzero(objmap == label)
        records.append((int(ys.min()), int(xs.min()), int(ys.size), int(label)))
    records.sort()
    canonical = np.zeros(objmap.shape, dtype=np.uint16)
    for new_label, record in enumerate(records, start=1):
        old_label = record[-1]
        canonical[objmap == old_label] = new_label
    return canonical


def assert_batch_equivalence(reference: Evaluation, candidate: Evaluation) -> None:
    """Require resource-only batching changes to preserve exact segmentation."""
    np.testing.assert_array_equal(reference.objmap > 0, candidate.objmap > 0)
    assert reference.objects == candidate.objects
    np.testing.assert_array_equal(
        canonical_objmap(reference.objmap), canonical_objmap(candidate.objmap)
    )


def assert_selected_scaling_not_worse(
    selected: Evaluation, alternative: Evaluation
) -> None:
    """Require the selected default to match or beat the alternative policy."""
    assert selected.iou >= alternative.iou
    selected_error = abs(selected.objects - selected.truth_objects)
    alternative_error = abs(alternative.objects - alternative.truth_objects)
    assert selected_error <= alternative_error


def assert_scaling_fixture_distinguishes_policies() -> None:
    """Guard against accidentally benchmarking the uint8 passthrough path."""
    from phenotypic.detect.nn import Sam2Detector

    image, _ = tiled_plate(rows=1, cols=1, encoding="uint16_nonfull")
    rgb = np.asarray(image.rgb[:])
    dtype_input = Sam2Detector(input_scaling="dtype_range")._preprocess(rgb)
    legacy_input = Sam2Detector(input_scaling="image_max")._preprocess(rgb)
    assert rgb.dtype == np.uint16
    assert int(rgb.max()) < np.iinfo(np.uint16).max
    assert not np.array_equal(dtype_input, legacy_input)


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

    sam2_batch_64 = evaluate(
        Sam2Detector(
            model_size="tiny",
            crop_n_layers=1,
            points_per_batch=64,
            input_scaling="dtype_range",
            device="auto",
        ),
        "Sam2     points_per_batch=64",
    )
    sam2_batch_8 = evaluate(
        Sam2Detector(
            model_size="tiny",
            crop_n_layers=1,
            points_per_batch=8,
            input_scaling="dtype_range",
            device="auto",
        ),
        "Sam2     points_per_batch=8",
    )
    assert_batch_equivalence(sam2_batch_64, sam2_batch_8)

    assert_scaling_fixture_distinguishes_policies()
    sam2_dtype_range = evaluate(
        Sam2Detector(
            model_size="tiny",
            crop_n_layers=1,
            points_per_batch=8,
            input_scaling="dtype_range",
            device="auto",
        ),
        "Sam2     uint16 dtype_range",
        encoding="uint16_nonfull",
    )
    sam2_image_max = evaluate(
        Sam2Detector(
            model_size="tiny",
            crop_n_layers=1,
            points_per_batch=8,
            input_scaling="image_max",
            device="auto",
        ),
        "Sam2     uint16 image_max",
        encoding="uint16_nonfull",
    )
    # The labeled uint16 fixture favors image_max (IoU 0.8709 versus 0.8626,
    # 1079 versus 1051 objects against 1152 truth objects), so retain the
    # compatibility policy as the default while keeping both options public.
    assert_selected_scaling_not_worse(sam2_image_max, sam2_dtype_range)
