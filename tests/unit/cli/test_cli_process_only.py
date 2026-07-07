import numpy as np
import tifffile

from phenotypic._cli._cli_process_only import (
    process_only_output_path,
    write_process_only_layer,
)
from phenotypic.data import load_synth_yeast_plate


def test_output_path_mirrors_one_level(tmp_path):
    out = tmp_path / "out"
    root = tmp_path / "in"
    img = root / "day1" / "plateA.tif"
    assert (
        process_only_output_path(out, img, root, "detect_mat")
        == out / "day1" / "plateA.tiff"
    )
    assert (
        process_only_output_path(out, img, root, "objmap")
        == out / "day1" / "plateA.png"
    )


def test_output_path_flat_and_single_file(tmp_path):
    out = tmp_path / "out"
    root = tmp_path / "in"
    assert (
        process_only_output_path(out, root / "a.tif", root, "rgb")
        == out / "a.tiff"
    )
    # single-file input: input_root is the file's parent
    f = tmp_path / "solo.tif"
    assert (
        process_only_output_path(out, f, f.parent, "gray")
        == tmp_path / "out" / "solo.tiff"
    )


def test_write_rgb_is_uint8_for_8bit_source(tmp_path):
    img = load_synth_yeast_plate()  # 8-bit source; rgb uint8
    p = tmp_path / "rgb.tiff"
    write_process_only_layer(img, "rgb", p)
    arr = tifffile.imread(p)
    assert arr.dtype == np.uint8 and arr.ndim == 3


def test_write_detect_mat_preserves_float_precision(tmp_path):
    img = load_synth_yeast_plate()  # detect_mat is float in [0,1]
    p = tmp_path / "dm.tiff"
    write_process_only_layer(img, "detect_mat", p)
    arr = tifffile.imread(p)
    # imsave writes a float TIFF — full precision, not quantized
    assert np.issubdtype(arr.dtype, np.floating)


def test_write_objmap_is_uint16_png(tmp_path):
    import cv2

    img = load_synth_yeast_plate()
    p = tmp_path / "om.png"
    write_process_only_layer(img, "objmap", p)
    arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    assert arr.dtype == np.uint16  # raw labels, 16-bit regardless of source


def test_objmap_without_objects_warns_and_writes_empty(tmp_path, recwarn):
    img = load_synth_yeast_plate()
    img.reset()  # no detection -> empty objmap
    p = tmp_path / "om.png"
    write_process_only_layer(img, "objmap", p)
    assert p.is_file()
    assert any("object map" in str(w.message).lower() for w in recwarn.list)
