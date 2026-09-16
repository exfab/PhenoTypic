"""THROWAWAY: what pipeline_step_path does each nesting shape actually produce?"""
from phenotypic.data import load_synth_yeast_plate
import spike_nested_gpu as S

for name, factory in [("A leaf-in-composite", S.shape_a),
                      ("B cpu-prefix-branch", S.shape_b),
                      ("C composite-in-composite", S.shape_c)]:
    img = load_synth_yeast_plate()
    S.make_pipeline(factory).apply_and_measure(img, inplace=True, apply_post=False)
    print(f"--- {name}")
    for app in img._metadata.provenance_journal.get("applications", []):
        for op in app.get("operations", []):
            cls = op["operation_class"].rsplit(".", 1)[-1]
            print(f"    {cls:<22} {op.get('pipeline_step_path')}")
