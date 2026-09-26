import numpy as np, sys
from phenotypic import Image
gray = (np.random.default_rng(0).random((32,32))*200).astype(np.uint8)
rgb = np.stack([gray, gray//2, gray//3], -1)
# Path A: imread with detect_mode kwarg
try:
    im = Image.imread(sys.argv[1], detect_mode='red'); print("A imread(gray png, detect_mode='red') OK, detect_mode:", im._data.detect_mode)
except Exception as e:
    print("A imread(..., detect_mode='red') RAISED", type(e).__name__, ":", e)
# Path B: the only reachable silent-fallback site: set_image(2-D) on image already in RGB mode
im = Image(rgb); im.set_detect_mode('red'); print("B before: detect_mode", im._data.detect_mode)
try:
    im.set_image(gray)
    print("B set_image(gray) OK; detect_mode now:", im._data.detect_mode, "| rgb empty:", im.rgb.isempty(),
          "| detect_mat == gray layer:", np.allclose(im.detect_mat[:], im.gray[:]))
except Exception as e:
    print("B set_image RAISED", type(e).__name__, e)
