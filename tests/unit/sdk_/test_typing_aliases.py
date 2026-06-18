from typing import get_args

from phenotypic.tools_.typing_ import DinoSize, DinoVersion, ProcessOnlyLayer


def test_process_only_layer_values():
    assert set(get_args(ProcessOnlyLayer)) == {"rgb", "gray", "detect_mat", "objmap"}


def test_dino_version_alias_values():
    assert set(get_args(DinoVersion)) == {2, 3}


def test_dino_size_alias_values():
    assert set(get_args(DinoSize)) == {"small", "base", "large"}


def test_process_only_layers_are_image_accessors():
    from phenotypic.data import load_synth_yeast_plate

    img = load_synth_yeast_plate()
    for layer in get_args(ProcessOnlyLayer):
        assert hasattr(img, layer), f"{layer} is not an Image accessor"
