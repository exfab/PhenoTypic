import importlib.util, sys, numpy as np, traceback
from pathlib import Path
print("rawpy installed:", importlib.util.find_spec("rawpy") is not None)
import skimage.io
from phenotypic import Image
import phenotypic._core._image_parts._image_io_handler as H
from phenotypic.sdk_.constants_ import IO
print("'.dng' in ACCEPTED:", '.dng' in IO.ACCEPTED_FILE_EXTENSIONS, "| '.nef' in ACCEPTED:", '.nef' in IO.ACCEPTED_FILE_EXTENSIONS)
print("handler module rawpy:", H.rawpy)
calls = []
orig = skimage.io.imread
def rec(*a, **k):
    calls.append(("skimage.io.imread", str(k.get('fname', a[0] if a else None))))
    return np.zeros((8,8,3), np.uint8)
skimage.io.imread = rec
raw_calls = []
if H.rawpy is not None:
    orig_raw = H.rawpy.imread
    def rrec(*a, **k): raw_calls.append(a); raise RuntimeError("rawpy.imread reached")
    H.rawpy.imread = rrec
d = Path(sys.argv[1])
for ext in ('.dng', '.nef', '.NEF', '.cr2'):
    p = d / f"empty{ext}"; p.write_bytes(b"")
    calls.clear(); raw_calls.clear()
    try:
        im = Image.imread(p); res = f"returned Image shape {im.rgb[:].shape}"
    except Exception as e:
        res = f"RAISED {type(e).__name__}: {str(e)[:150]}"
    print(f"{ext}: skimage calls={calls} rawpy calls={len(raw_calls)} -> {res}")
skimage.io.imread = orig
# Tiny valid TIFF saved with .dng suffix, un-patched skimage
import tifffile
p = d / "tiffy.dng"
a = (np.arange(16*16*3) % 255).astype(np.uint16).reshape(16,16,3) * 200
tifffile.imwrite(p, a, photometric='rgb')
print("direct skimage.io.imread on TIFF-as-.dng:", end=" ")
try:
    r = orig(p); print(type(r).__name__, r.shape, r.dtype, "equal:", np.array_equal(r, a))
except Exception as e:
    print("RAISED", type(e).__name__, str(e)[:200])
try:
    im = Image.imread(p); print("Image.imread(TIFF-as-.dng):", im.rgb[:].shape, im.rgb[:].dtype, "bit_depth", im.bit_depth)
except Exception as e:
    print("Image.imread(TIFF-as-.dng) RAISED", type(e).__name__, str(e)[:300])
