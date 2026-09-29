import logging, io, polars as pl, pandas as pd
from phenotypic import ImagePipeline
from phenotypic.measure import MeasureSize
from phenotypic.detect import OtsuDetector
from phenotypic.post import AppendString, ExpandMetadata
from phenotypic._cli._cli_output_manager import _apply_post_to_master
buf = io.StringIO(); h = logging.StreamHandler(buf); h.setLevel(logging.DEBUG)
h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
lg = logging.getLogger("phenotypic._cli._cli_output_manager"); lg.addHandler(h); lg.setLevel(logging.DEBUG); lg.propagate = False
master = pl.DataFrame({"Metadata_ImageName": ["a", "a", "b"], "Object_Label": [1, 2, 1], "Size_Area": [10, 20, 30]})
cases = {
  "AppendString(column='Metadata_DoesNotExist')": AppendString(column="Metadata_DoesNotExist", value="_x"),
  "ExpandMetadata(column='Metadata_DoesNotExist', labels=['A','B'])": ExpandMetadata(column="Metadata_DoesNotExist", labels=["A","B"]),
  "AppendString(column='DoesNotExist')  [bare label]": AppendString(column="DoesNotExist", value="_x"),
}
for label, op in cases.items():
    print("\n==", label)
    try:
        op.apply(master.to_pandas()); print("  direct op.apply: no error")
    except Exception as e:
        print("  direct op.apply RAISED", type(e).__name__, ":", str(e)[:200])
    pipe = ImagePipeline(ops={"det": OtsuDetector()}, meas={"size": MeasureSize()}, post={"p": op})
    buf.truncate(0); buf.seek(0)
    try:
        out = _apply_post_to_master(master, pipe)
        print("  _apply_post_to_master returned; identical to master:", out.equals(master), "| is same object:", out is master)
    except Exception as e:
        print("  _apply_post_to_master RAISED", type(e).__name__, e)
    lines = [l for l in buf.getvalue().splitlines() if l.startswith(("WARNING", "ERROR", "DEBUG"))]
    for l in lines: print("  LOG:", l[:200])
    tb_last = [l for l in buf.getvalue().splitlines() if l and not l.startswith(" ")][-1:]
    print("  exc_info last line:", tb_last)
