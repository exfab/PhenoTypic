import traceback, warnings, logging
logging.basicConfig(level=logging.WARNING)
import numpy as np
from phenotypic import Image, ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.measure import MeasureSize
from phenotypic import enhance
g = load_synth_yeast_plate()
rgb = g.rgb[:].copy()
print("synth GridImage num_objects (pre-populated objmap):", g.num_objects)
Enh = enhance.BlurGauss
print("enhancer used:", Enh.__name__ if Enh else None)

def fresh():
    im = Image(rgb, name="plain"); return im
def with_objmap():
    im = Image(rgb, name="plain_obj"); im.objmap[:] = g.objmap[:]; return im

cases = {
 "enh-only ops + MeasureSize, fresh Image (0 objects)": (lambda: ImagePipeline(ops={'enh': Enh()}, meas={'size': MeasureSize()}), fresh),
 "empty ops + MeasureSize, fresh Image": (lambda: ImagePipeline(ops={}, meas={'size': MeasureSize()}), fresh),
 "empty ops + MeasureSize, Image w/ copied objmap": (lambda: ImagePipeline(ops={}, meas={'size': MeasureSize()}), with_objmap),
 "enh-only ops + empty meas, fresh Image": (lambda: ImagePipeline(ops={'enh': Enh()}, meas={}), fresh),
 "empty ops + empty meas, fresh Image": (lambda: ImagePipeline(ops={}, meas={}), fresh),
}
for label, (mkp, mki) in cases.items():
    print("\n====", label)
    try:
        p = mkp(); im = mki(); print("  num_objects before:", im.num_objects)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = p.apply_and_measure(im)
        print("  RESULT type:", type(df).__name__, "shape:", getattr(df,'shape',None))
        print("  columns:", list(df.columns)[:12])
        for x in w: print("  WARNING:", x.category.__name__, str(x.message)[:200])
    except Exception as e:
        tb = traceback.format_exc().strip().splitlines()
        print("  RAISED", type(e).__name__, ":", str(e)[:400]); print("  tb tail:", *tb[-8:], sep="\n    ")
