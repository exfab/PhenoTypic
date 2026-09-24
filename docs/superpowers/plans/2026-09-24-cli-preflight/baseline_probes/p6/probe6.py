import numpy as np, traceback
from phenotypic import Image, ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureColor
def tail(e, n=6):
    tb = traceback.format_exc().strip().splitlines(); return "\n    ".join(tb[-n:])
gray = load_synth_yeast_plate().gray[:].copy()
print("gray dtype/range:", gray.dtype, float(gray.min()), float(gray.max()))
im = Image(gray, name="grayplate"); print("rgb empty:", im.rgb.isempty(), "| bit_depth:", im.bit_depth)
OtsuDetector().apply(im, inplace=True) if 'inplace' in OtsuDetector.apply.__code__.co_varnames else None
im = OtsuDetector().apply(im); print("num_objects after Otsu:", im.num_objects)
print("\n== MeasureColor().measure(gray image)")
try:
    df = MeasureColor().measure(im); print("  OK shape", df.shape, "cols:", list(df.columns)[:8], "... nan frac:", float(df.select_dtypes('number').isna().mean().mean()))
except Exception as e:
    print("  RAISED", type(e).__name__, ":", str(e)[:300]); print("   ", tail(e))
print("\n== pipeline Otsu + MeasureColor apply_and_measure on fresh gray Image")
try:
    df = ImagePipeline(ops={'det': OtsuDetector()}, meas={'color': MeasureColor()}).apply_and_measure(Image(gray, name='g2'))
    print("  OK shape", df.shape)
except Exception as e:
    print("  RAISED", type(e).__name__, ":", str(e)[:400]); print("   ", tail(e, 4))
print("\n== CalibrateColorRpcc construction")
from phenotypic.correction import CalibrateColorRpcc
from phenotypic.correction._color_correction._checker_roi import CheckerRoi
for kw in ({}, {'rois': []}, {'rois': [CheckerRoi(row=(0, 50), col=(0, 50))]}):
    try:
        op = CalibrateColorRpcc(**kw); print("  construct", {k: (len(v) if isinstance(v, list) else v) for k,v in kw.items()}, "-> OK")
    except Exception as e:
        print("  construct", kw.keys(), "-> RAISED", type(e).__name__, str(e).splitlines()[0][:200], "|", " ".join(str(e).splitlines()[1:3])[:200])
print("\n== CalibrateColorRpcc(rois=[CheckerRoi]).apply(gray image) (no checker present)")
try:
    op = CalibrateColorRpcc(rois=[CheckerRoi(row=(0, 50), col=(0, 50))])
    out = op.apply(Image(gray, name='g3')); print("  OK returned; qc:", op.qc[:2])
except Exception as e:
    print("  RAISED", type(e).__name__, ":", str(e)[:400]); print("   ", tail(e, 4))
