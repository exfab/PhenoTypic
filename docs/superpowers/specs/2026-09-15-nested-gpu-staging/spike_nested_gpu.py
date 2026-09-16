"""THROWAWAY SPIKE. Nested-GpuDetector staging: split -> substitute -> replay.

No GPU, no torch: a deterministic CPU FakeGpuDetector stands in for Sam2, so the
mechanism is exercised exactly as it would be with SAM2 resident on a GPU.

Three nesting shapes are tested:
  A  GPU leaf directly in CompositeDetector.ops        <- the real F1gfd5 shape
  B  GPU behind a CPU prefix in a nested ImagePipeline branch of the composite
  C  GPU inside a CompositeDetector inside a CompositeDetector (depth 2)

For each: run the pipeline normally, then run it as Stage1/Stage2/Stage3 and
assert the objmap and the measurements are identical.
"""
from __future__ import annotations

from typing import Any, List

import numpy as np

from phenotypic import ImagePipeline
from phenotypic.abc_ import GpuDetector, ObjectDetector
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import CompositeDetector, ManualPointDetector
from phenotypic.enhance import BlurGauss, ContrastStretching, SubtractGaussian
from phenotypic.measure import MeasureIntensity, MeasureShape
from phenotypic.refine import SmallObjectRemover
from phenotypic.sdk_.typing_ import NdArrayField, OperationField

CENTERS = [[150.0, 200.0], [150.0, 400.0], [150.0, 600.0],
           [300.0, 200.0], [300.0, 400.0], [300.0, 600.0],
           [450.0, 200.0], [450.0, 400.0], [450.0, 600.0]]


class FakeGpuDetector(GpuDetector):
    """CPU stand-in for Sam2: deterministic, no torch, same hook surface."""

    input_layer: Any = "detect_mat"
    thresh: int = 110
    _loaded: bool = False

    def _ensure_model_loaded(self) -> None:
        object.__setattr__(self, "_loaded", True)

    def _infer_one(self, sample: np.ndarray) -> np.ndarray:
        from scipy.ndimage import label
        gray = sample[..., 0] if sample.ndim == 3 else sample
        lab, _ = label(gray > self.thresh)
        return lab.astype(np.uint16)


class ReplayDetector(ObjectDetector):
    """Stage-3 stand-in: writes a PRE-RECORDED Stage-2 result."""

    detector: OperationField
    result: NdArrayField

    def _operate(self, image):
        self.detector._write_object_output(image, self.result)
        return image


# ------------------------------------------------------------- tree traversal
def _children(obj):
    """Yield (step, child) for every operation-bearing child of *obj*."""
    if isinstance(obj, ImagePipeline):
        for k, v in obj.get_ops().items():
            yield k, v
        return
    if hasattr(type(obj), "model_fields"):
        for fname in type(obj).model_fields:
            val = getattr(obj, fname, None)
            if isinstance(val, list):
                for i, item in enumerate(val):
                    if isinstance(item, (ObjectDetector, ImagePipeline)):
                        yield f"{fname}[{i}]", item
            elif isinstance(val, (ObjectDetector, ImagePipeline)):
                yield fname, val


def find_gpu_detectors(pipeline: ImagePipeline):
    found = []

    def visit(obj, path):
        if isinstance(obj, GpuDetector):
            found.append((tuple(path), obj))
            return
        for step, child in _children(obj):
            visit(child, path + [step])

    for key, op in pipeline.get_ops().items():
        visit(op, [key])
    return found


def _get_step(obj, step):
    if "[" in step:
        fname, idx = step[:-1].split("[")
        return getattr(obj, fname)[int(idx)]
    if isinstance(obj, ImagePipeline):
        return obj.get_ops()[step]
    return getattr(obj, step)


def branch_prefix(pipeline: ImagePipeline, path):
    """Ops that must run between the Stage-1 store and the GPU op's input.

    Walks the ancestor chain. A nested ImagePipeline contributes the ops that
    precede the branch; a CompositeDetector contributes NOTHING, because its
    ops are parallel branches each applied to the same input via inplace=False.
    """
    prefix: list = []
    cursor: Any = pipeline
    for step in path[:-1]:
        if isinstance(cursor, ImagePipeline):
            keys = list(cursor.get_ops())
            if cursor is not pipeline:  # top level is split by the caller
                for k in keys[:keys.index(step)]:
                    prefix.append(cursor.get_ops()[k])
        cursor = _get_step(cursor, step)
    if isinstance(cursor, ImagePipeline):  # innermost container
        keys = list(cursor.get_ops())
        for k in keys[:keys.index(path[-1])]:
            prefix.append(cursor.get_ops()[k])
    return prefix


def substitute(root, path, replacement):
    """Deep-copy *root* replacing the op at *path*. Works at any depth."""
    if isinstance(root, ImagePipeline):
        ops = dict(root.get_ops())
        head, rest = path[0], path[1:]
        ops[head] = replacement if not rest else substitute(ops[head], rest, replacement)
        return ImagePipeline(ops=ops, meas=root.get_meas(),
                             nrows=root.nrows, ncols=root.ncols)
    node = root.model_copy(deep=True)
    head, rest = path[0], path[1:]
    if "[" in head:
        fname, idx = head[:-1].split("[")
        seq = list(getattr(node, fname))
        seq[int(idx)] = replacement if not rest else substitute(seq[int(idx)], rest, replacement)
        setattr(node, fname, seq)
    else:
        cur = getattr(node, head)
        setattr(node, head, replacement if not rest else substitute(cur, rest, replacement))
    return node


def build_stage_plan(pipeline: ImagePipeline):
    hits = find_gpu_detectors(pipeline)
    assert len(hits) == 1, f"expected exactly one GpuDetector, got {len(hits)}"
    path, detector = hits[0]
    ops, keys = pipeline.get_ops(), list(pipeline.get_ops())
    cut = keys.index(path[0])
    pre = ImagePipeline(ops={k: ops[k] for k in keys[:cut]},
                        nrows=pipeline.nrows, ncols=pipeline.ncols)
    post = ImagePipeline(ops={k: ops[k] for k in keys[cut:]},
                         meas=pipeline.get_meas(),
                         nrows=pipeline.nrows, ncols=pipeline.ncols)
    return pre, path, detector, post, branch_prefix(pipeline, path)


# ------------------------------------------------------------------- shapes
def gpu():
    return FakeGpuDetector()


def manual():
    return ManualPointDetector(centers=CENTERS, shape="disk", width=61)


def shape_a():
    return CompositeDetector(ops=[gpu(), manual()], mode="overlap")


def shape_b():
    branch = ImagePipeline(ops={"ContrastStretching":
                                ContrastStretching(lower_percentile=2,
                                                   upper_percentile=98,
                                                   input_layer="detect_mat"),
                                "FakeGpuDetector": gpu()})
    return CompositeDetector(ops=[branch, manual()], mode="overlap")


def shape_c():
    inner = CompositeDetector(ops=[gpu(), manual()], mode="union")
    return CompositeDetector(ops=[inner, manual()], mode="overlap")


def make_pipeline(detector_factory):
    return ImagePipeline(
        ops={"BlurGauss": BlurGauss(sigma=2.0),
             "SubtractGaussian": SubtractGaussian(sigma=50.0),
             "CompositeDetector": detector_factory(),
             "SmallObjectRemover": SmallObjectRemover(min_size=50)},
        meas={"MeasureShape": MeasureShape(),
              "MeasureIntensity": MeasureIntensity()},
    )


def run_case(name, factory) -> bool:
    ref_img = load_synth_yeast_plate()
    ref_meas = make_pipeline(factory).apply_and_measure(
        ref_img, inplace=True, apply_post=False)
    ref_objmap = ref_img.objmap[:].copy()

    img = load_synth_yeast_plate()
    pipe = make_pipeline(factory)
    pre, path, detector, post, prefix = build_stage_plan(pipe)

    pre.apply(img, inplace=True)                                   # Stage 1

    probe = img.copy()                                             # Stage 2
    for op in prefix:
        op.apply(probe, inplace=True)
    sample = detector._preprocess(getattr(probe, detector.input_layer)[:])
    raw = detector._infer_batch(detector._collate([sample]))[0]

    stub = ReplayDetector(detector=detector, result=raw)            # Stage 3
    post_sub = substitute(post, path, stub)
    post_sub.apply(img, inplace=True)
    staged_meas = post_sub.measure(img, apply_post=False)

    cols = sorted(set(ref_meas.columns) & set(staged_meas.columns))
    objmap_same = np.array_equal(ref_objmap, img.objmap[:])
    meas_same = (ref_meas[cols].reset_index(drop=True)
                 .equals(staged_meas[cols].reset_index(drop=True)))
    ok = objmap_same and meas_same
    print(f"[{name}] path={'/'.join(path)}")
    print(f"[{name}] stage2 prefix={[type(o).__name__ for o in prefix]}")
    print(f"[{name}] objects ref={ref_img.num_objects} staged={img.num_objects} "
          f"| ref_meas={ref_meas.shape} staged_meas={staged_meas.shape}")
    print(f"[{name}] objmap={objmap_same} meas={meas_same} ({len(cols)} cols) "
          f"-> {'PASS' if ok else 'FAIL'}\n")
    return ok


def main() -> int:
    results = [run_case("A leaf-in-composite", shape_a),
               run_case("B cpu-prefix-branch", shape_b),
               run_case("C composite-in-composite", shape_c)]
    print("SPIKE", "PASS" if all(results) else "FAIL")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
