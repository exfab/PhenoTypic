"""THROWAWAY: run the REAL F1gfd5 pipeline through the prototype walker/splitter."""
from pathlib import Path
from phenotypic import ImagePipeline
from phenotypic.abc_ import GpuDetector, ObjectDetector
import spike_nested_gpu as S

p = Path('/bigdata/exfab/anguy344/projects/ucr_033_e_d_Linzer_Ganoderma/config/F1gfd5.json.pht-pipe')
pipe = ImagePipeline.from_json(p)

hits = S.find_gpu_detectors(pipe)
print("ops-slot GPU detectors:", [('/'.join(h[0]), type(h[1]).__name__) for h in hits] if hits else None)

pre, path, det, post, prefix = S.build_stage_plan(pipe)
print("  gpu path      :", '/'.join(path))
print("  stage1 ops    :", list(pre.get_ops()))
print("  stage2 detector:", type(det).__name__, "input_layer=", det.input_layer)
print("  stage2 prefix :", [type(o).__name__ for o in prefix], "<- empty means: read the store layer directly")
print("  stage3 ops    :", list(post.get_ops()))
print("  stage3 meas   :", list(post.get_meas()))

# --- bounds probe: do OTHER slots hide operations the walker must also check?
print("\nnested ops in the MEAS slot (not covered by an ops-only walk):")
for mkey, m in pipe.get_meas().items():
    for step, child in S._children(m):
        kind = "GpuDetector" if isinstance(child, GpuDetector) else type(child).__name__
        print(f"   {mkey}.{step} -> {kind}")
