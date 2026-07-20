from typing import get_args

from phenotypic.sdk_.typing_ import (
    BoundaryMode,
    DinoSize,
    DinoVersion,
    FilamentousFungiReconnectStrategy,
    FilFinderOutput,
    FilFinderPruneCriteria,
    InputLayer,
    NormOut,
    ProcessOnlyLayer,
)


def test_boundary_mode_values():
    assert set(get_args(BoundaryMode)) == {
        "reflect",
        "constant",
        "nearest",
        "mirror",
        "wrap",
    }


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


def test_input_layer_values():
    assert set(get_args(InputLayer)) == {"detect_mat", "rgb"}


def test_norm_out_values():
    # Optional[Literal[...]] -> (Literal[...], NoneType)
    literal, none_type = get_args(NormOut)
    assert set(get_args(literal)) == {"clip", "rescale"}
    assert none_type is type(None)


def test_filfinder_closed_set_values_and_field_annotations():
    """The A10 aliases exactly match the frozen wrapper fields."""
    from phenotypic.detect import FilFinderDetector

    assert get_args(FilFinderOutput) == (
        "mask",
        "skeleton",
        "longest_path",
    )
    assert get_args(FilFinderPruneCriteria) == (
        "all",
        "intensity",
        "length",
    )
    assert (
        FilFinderDetector.model_fields["output"].annotation
        == FilFinderOutput
    )
    assert (
        FilFinderDetector.model_fields["prune_criteria"].annotation
        == FilFinderPruneCriteria
    )


def test_filamentous_fungi_reconnect_strategy_values_and_field_annotation():
    """The S01 alias exactly matches the integrated detector field."""
    from phenotypic.detect import FilamentousFungiDetector

    assert get_args(FilamentousFungiReconnectStrategy) == (
        "dijkstra",
        "app2_gwdt",
    )
    assert (
        FilamentousFungiDetector.model_fields[
            "reconnect_strategy"
        ].annotation
        == FilamentousFungiReconnectStrategy
    )
