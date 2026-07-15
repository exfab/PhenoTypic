# tests/unit/enhance/test_focus_edge_two_k_phase.py
import numpy as np
import pytest
from pydantic import ValidationError

from phenotypic import Image
from phenotypic.enhance import FocusEdgeTwoKPhase


def _image_from(arr: np.ndarray) -> Image:
    """Wrap a float [0,1] array in an Image by broadcasting to uint8 RGB (codebase idiom)."""
    rgb = np.repeat((arr[..., None] * 255).astype(np.uint8), 3, axis=2)
    return Image(rgb)


def _plate():
    # a solid bright inoculum disk (its interior yields NO edges -> hole preserved) plus a
    # radiating branch line (an edge -> nonzero gated response)
    img = np.full((80, 80), 0.4, dtype=np.float32)
    yy, xx = np.ogrid[:80, :80]
    core = ((yy - 40) ** 2 + (xx - 40) ** 2) < 12 ** 2
    img[core] = 0.95                                   # solid core
    img[39:41, 40:74] = 0.9                            # a branch tendril
    return _image_from(img)


def test_defaults():
    e = FocusEdgeTwoKPhase()
    assert e.k_strict == 6.0 and e.k_loose == 4.5
    assert e.seed_thresh == "otsu" and e.cand_thresh == "triangle"
    assert e.n_orient == 8 and e.min_wavelength == 5.0


def test_writes_gated_response_into_detect_mat():
    im = _plate()
    out = FocusEdgeTwoKPhase().apply(im, inplace=False)
    dm = np.asarray(out.detect_mat[:])
    assert dm.shape == (80, 80)
    assert dm.max() > 0
    assert dm.min() == 0.0            # agar / background gated to zero


def test_center_hole_preserved():
    im = _plate()
    out = FocusEdgeTwoKPhase().apply(im, inplace=False)
    dm = np.asarray(out.detect_mat[:])
    assert dm[40, 40] == 0.0          # solid inoculum core stays a hole (edge detector)


@pytest.mark.parametrize("bad", [dict(k_strict=-1.0), dict(min_wavelength=1.0), dict(n_orient=0)])
def test_parameter_validation(bad):
    with pytest.raises(ValidationError):
        FocusEdgeTwoKPhase(**bad)
