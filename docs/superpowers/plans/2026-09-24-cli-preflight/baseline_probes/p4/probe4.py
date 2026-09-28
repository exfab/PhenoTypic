import numpy as np, traceback
from phenotypic import Image
from phenotypic._core._image_parts.detection_modes import available_modes, get_detection_mode
modes = available_modes()
print("modes:", {m: get_detection_mode(m).requires_rgb for m in modes})
rgb_modes = [m for m in modes if get_detection_mode(m).requires_rgb]
gray = (np.random.default_rng(0).random((32,32))*200).astype(np.uint8); gray[8:16,8:16]=250
for mode in rgb_modes:
    print(f"\n== mode {mode!r}")
    try:
        im = Image(gray, detect_mode=mode)
        dm = im.detect_mat[:]
        print("  Image(gray, detect_mode=...) OK; stored detect_mode:", getattr(im, 'detect_mode', None),
              "| detect_mat equals gray layer:", np.allclose(dm, im.gray[:]), "| rgb empty:", im.rgb.isempty())
    except Exception as e:
        print("  Image(...) RAISED", type(e).__name__, e)
    try:
        im2 = Image(gray); im2.set_detect_mode(mode); print("  set_detect_mode OK (unexpected)")
    except Exception as e:
        print("  set_detect_mode RAISED", type(e).__name__, ":", e)
