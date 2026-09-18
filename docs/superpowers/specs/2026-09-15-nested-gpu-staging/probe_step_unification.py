"""THROWAWAY: does adding step-descent make walker path == provenance step path?

Simulates the proposed change by subclassing CompositeDetector so each child
apply runs inside `pipeline_step(f"ops[{i}]")`, then compares the walker's
gpu_path against the step path the journal actually records.
"""
import numpy as np

from phenotypic import ImagePipeline
from phenotypic._core._provenance import pipeline_step
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import CompositeDetector
import spike_nested_gpu as S


class SteppedComposite(CompositeDetector):
    """CompositeDetector with the proposed per-branch step push."""

    def _operate(self, image):
        objmaps = []
        for i, det in enumerate(self.ops):
            if det is None:
                continue
            with pipeline_step(f"ops[{i}]"):           # <-- the proposed change
                if isinstance(det, ImagePipeline):
                    out = det.apply(image, inplace=False, reset=False)
                else:
                    out = det.apply(image, inplace=False)
            objmaps.append(out.objmap[:].astype(bool))
        if self.mode == "union":
            combined = np.logical_or.reduce(objmaps)
        elif self.mode == "intersection":
            combined = np.logical_and.reduce(objmaps)
        else:
            combined = objmaps[0]
            for m in objmaps[1:]:
                combined = CompositeDetector._filter_mask_by_overlap_bidirectional(
                    combined, m, min_overlap_ratio=self.min_overlap_ratio)
        image.objmask[:] = combined > 0
        return image


def stepped(factory):
    """Rebuild a shape with SteppedComposite in place of CompositeDetector."""
    comp = factory()
    ops = [stepped_op(o) for o in comp.ops]
    return SteppedComposite(ops=ops, mode=comp.mode,
                            min_overlap_ratio=comp.min_overlap_ratio)


def stepped_op(o):
    if isinstance(o, CompositeDetector):
        return SteppedComposite(ops=[stepped_op(x) for x in o.ops], mode=o.mode,
                                min_overlap_ratio=o.min_overlap_ratio)
    return o


for name, factory in [("A leaf-in-composite", S.shape_a),
                      ("B cpu-prefix-branch", S.shape_b),
                      ("C composite-in-composite", S.shape_c)]:
    pipe = S.make_pipeline(lambda f=factory: stepped(f))
    walker_path, _ = S.find_gpu_detectors(pipe)[0]

    img = load_synth_yeast_plate()
    pipe.apply_and_measure(img, inplace=True, apply_post=False)
    journal_path = None
    for app in img._metadata.provenance_journal.get("applications", []):
        for op in app.get("operations", []):
            if op["operation_class"].endswith("FakeGpuDetector"):
                journal_path = op.get("pipeline_step_path")
    match = list(walker_path) == list(journal_path or [])
    print(f"--- {name}")
    print(f"    walker gpu_path : {list(walker_path)}")
    print(f"    journal step    : {journal_path}")
    print(f"    IDENTICAL       : {match}\n")
