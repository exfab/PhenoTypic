"""THROWAWAY: does running the residual chain in process mode hit the depth-0 trap?"""
from phenotypic.data import load_synth_yeast_plate
from phenotypic._core._provenance import (
    _application_owner_depth, provenance_application, set_provenance_status,
)
import spike_nested_gpu as S

img = load_synth_yeast_plate()
pipe = S.make_pipeline(S.shape_a)
pre, path, det, post, prefix = S.build_stage_plan(pipe)

# --- emulate Stage 1: apply pre-ops, then mark the application "staged"
with provenance_application(img, kind="programmatic"):
    for op in pre.get_ops().values():
        op.apply(img, inplace=True)
# Stage 1 marks the application "staged" AFTER the block closes, then saves.
set_provenance_status(img, "staged")
status = img._metadata.provenance_journal["applications"][-1]["status"]
print(f"Stage-1 trailing application status: {status!r}")

# --- emulate process mode: residual chain at CLI owner-depth 0, no store write
sample = det._preprocess(getattr(img, det.input_layer)[:])
raw = det._infer_batch(det._collate([sample]))[0]
stub = S.ReplayDetector(detector=det, result=raw)
residual = S.substitute(post, path, stub)

token = _application_owner_depth.set(0)          # CLI stage context
try:
    residual.apply(img, inplace=True)
    print("residual chain at depth 0: OK  (no provenance handling needed)")
except Exception as exc:
    print(f"residual chain at depth 0: {type(exc).__name__}: {exc}")
finally:
    _application_owner_depth.reset(token)
