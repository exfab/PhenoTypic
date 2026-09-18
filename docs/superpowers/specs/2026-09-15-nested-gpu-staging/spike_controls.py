"""THROWAWAY SPIKE part 2: negative control + provenance comparison."""
from __future__ import annotations
import json
import numpy as np

import spike_nested_gpu as S
from phenotypic.data import load_synth_yeast_plate


def staged_run(factory, corrupt=False):
    img = load_synth_yeast_plate()
    pipe = S.make_pipeline(factory)
    pre, path, detector, post, prefix = S.build_stage_plan(pipe)
    pre.apply(img, inplace=True)
    probe = img.copy()
    for op in prefix:
        op.apply(probe, inplace=True)
    sample = detector._preprocess(getattr(probe, detector.input_layer)[:])
    raw = detector._infer_batch(detector._collate([sample]))[0]
    if corrupt:
        raw = np.roll(raw, 7, axis=0)          # a 7-px shift: a broken replay
    stub = S.ReplayDetector(detector=detector, result=raw)
    S.substitute(post, path, stub).apply(img, inplace=True)
    return img


def main() -> int:
    ref = load_synth_yeast_plate()
    S.make_pipeline(S.shape_a).apply_and_measure(ref, inplace=True, apply_post=False)
    ref_objmap = ref.objmap[:].copy()

    clean = staged_run(S.shape_a)
    dirty = staged_run(S.shape_a, corrupt=True)

    clean_same = np.array_equal(ref_objmap, clean.objmap[:])
    dirty_same = np.array_equal(ref_objmap, dirty.objmap[:])
    print(f"[control] clean replay matches reference : {clean_same}  (want True)")
    print(f"[control] corrupted replay matches ref    : {dirty_same}  (want False)")
    print(f"[control] corrupted objects={dirty.num_objects} vs ref={ref.num_objects}")

    # ---- provenance: does a nested op get its own journal entry, and with what path?
    def ops_of(image):
        j = image._metadata.provenance_journal
        out = []
        for app in j.get("applications", []):
            for op in app.get("operations", []):
                out.append((op["operation_class"], op.get("pipeline_step_path")))
        return out

    print("\n[provenance] single-pass journal:")
    for cls, path in ops_of(ref):
        print(f"   {cls:<24} path={path}")
    print("[provenance] staged journal:")
    for cls, path in ops_of(clean):
        print(f"   {cls:<24} path={path}")

    ok = clean_same and not dirty_same
    print("\nCONTROLS", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
