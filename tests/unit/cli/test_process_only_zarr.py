"""--mode process writes a single-series store carrying its own provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic import Image, ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_ import ngff_
from phenotypic.sdk_.ngff_ import PhenotypicAttr
from phenotypic._cli._cli_process_only import (
    process_only_output_path,
    process_single_apply_only_core,
    write_process_only_layer,
)


@pytest.fixture
def source_image(tmp_path: Path) -> Path:
    root = tmp_path / "in"
    root.mkdir()
    path = root / "IMG_4471.tiff"
    Image(load_synth_yeast_plate()).rgb.imsave(filepath=path)
    return path


@pytest.fixture
def pipeline_file(tmp_path: Path) -> Path:
    """Named `.pht-pipe` because `to_json` RENAMES anything else.

    `ImagePipeline().to_json(tmp / "preprocess.json")` writes
    `preprocess.json.pht-pipe` and returns None -- verified by execution -- so
    a fixture that returns the path it passed in returns a path that does not
    exist, and every test using it dies in `from_json`.
    """
    nested = tmp_path / "config" / "deep"
    nested.mkdir(parents=True)
    path = nested / "preprocess.json.pht-pipe"
    ImagePipeline().to_json(path)
    assert path.is_file()
    return path


def _block(store: Path) -> dict:
    payload = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    return payload["attributes"][PhenotypicAttr.ROOT]


def test_output_path_is_a_store_for_zarr(tmp_path: Path) -> None:
    out = process_only_output_path(
        tmp_path / "out", tmp_path / "in" / "a" / "p01.tiff",
        tmp_path / "in", "rgb", fmt="zarr",
    )
    assert out == tmp_path / "out" / "a" / f"p01{ngff_.STORE_SUFFIX}"


def test_output_path_is_unchanged_for_tiff(tmp_path: Path) -> None:
    assert process_only_output_path(
        tmp_path / "out", tmp_path / "in" / "p01.tiff", tmp_path / "in", "rgb",
    ).name == "p01.tiff"
    assert process_only_output_path(
        tmp_path / "out", tmp_path / "in" / "p01.tiff", tmp_path / "in", "objmap",
    ).name == "p01.png"


def test_a_store_input_does_not_double_its_suffix(tmp_path: Path) -> None:
    """`Path("p01.ome.zarr").stem` is `"p01.ome"` -> `p01.ome.ome.zarr`.

    Spec 7.3. A tree of stores is valid input, so this is the ordinary case
    for the second run of the loop, not an exotic one -- and the wrong name is
    a plausible-looking one that nothing raises on.
    """
    store_in = tmp_path / "in" / f"p01{ngff_.STORE_SUFFIX}"
    assert process_only_output_path(
        tmp_path / "out", store_in, tmp_path / "in", "rgb", fmt="zarr",
    ).name == f"p01{ngff_.STORE_SUFFIX}"
    assert process_only_output_path(
        tmp_path / "out", store_in, tmp_path / "in", "detect_mat", fmt="tiff",
    ).name == "p01.tiff"


def test_writer_emits_only_the_requested_series(tmp_path: Path) -> None:
    img = Image(load_synth_yeast_plate())
    out = tmp_path / "p01.ome.zarr"
    write_process_only_layer(img, "rgb", out, fmt="zarr")
    assert (out / "rgb").is_dir()
    assert not (out / "gray").exists()
    assert not (out / "detect_mat").exists()
    assert not (out / "rgb" / "labels").exists()
    assert _block(out)[PhenotypicAttr.SERIES] == {"rgb": "rgb"}


def test_writer_omits_image_class(tmp_path: Path) -> None:
    img = Image(load_synth_yeast_plate())
    out = tmp_path / "p01.ome.zarr"
    write_process_only_layer(img, "rgb", out, fmt="zarr")
    assert PhenotypicAttr.IMAGE_CLASS not in _block(out)


@pytest.mark.parametrize("layer", ["detect_mat", "objmap"])
def test_the_writer_refuses_a_layer_with_no_store_form(
    tmp_path: Path, layer: str
) -> None:
    """Belt to the CLI's braces. `_save_store` would raise anyway for
    detect_mat -- `primary_series` accepts only rgb/gray -- but with a message
    about internal series naming rather than about what the user asked for.
    """
    img = Image(load_synth_yeast_plate())
    with pytest.raises(ValueError, match=layer):
        write_process_only_layer(img, layer, tmp_path / "x.ome.zarr", fmt="zarr")


def test_a_single_series_rgb_store_is_twelve_files(tmp_path: Path) -> None:
    """Spec 1.1. Guards against an accidental extra series or level.

    The `4 + 2 * levels` shorthand holds only while every pyramid level fits
    inside ONE shard -- true up to a 4096-pixel level-0 edge, and true for the
    600x800 synthetic plate. Above that a level contributes more than one shard
    file; the committed validation script (Task 11) carries the general form.
    """
    img = Image(load_synth_yeast_plate())
    out = tmp_path / "p01.ome.zarr"
    write_process_only_layer(img, "rgb", out, fmt="zarr")
    levels = ngff_.pyramid_level_count(*img.rgb[:].shape[:2])
    files = [p for p in out.rglob("*") if p.is_file()]
    # root + OME/zarr.json + OME xml + series zarr.json + 2 per level
    assert len(files) == 4 + 2 * levels


def test_core_records_the_pipeline_basename_not_its_path(
    tmp_path: Path, source_image: Path, pipeline_file: Path
) -> None:
    """A published store must not carry cluster filesystem layout."""
    out = tmp_path / "out"
    process_single_apply_only_core(
        pipeline_path=pipeline_file,
        image_path=source_image,
        input_root=source_image.parent,
        output_dir=out,
        image_type="Image",
        layer="rgb",
        read_kwargs={},
        process_format="zarr",
    )
    store = out / f"IMG_4471{ngff_.STORE_SUFFIX}"
    journal = _block(store)[PhenotypicAttr.PROVENANCE]
    assert journal["pipeline"]["source_path"] == "preprocess.json.pht-pipe"
    assert "/" not in journal["pipeline"]["source_path"]
    assert len(journal["pipeline"]["sha256"]) == 64


def test_provenance_init_runs_before_apply_not_after(
    tmp_path: Path, source_image: Path, pipeline_file: Path
) -> None:
    """`initialize_cli_provenance` resets the journal (_provenance.py:294).

    Called after `pipeline.apply()` it would discard `operations[]` -- and the
    store would still have a `pipeline` key, so the store looks fine and the
    operations are simply gone. The pipeline here has no operations, so what
    this pins is that BOTH keys survive: an empty list, not a missing one.
    """
    out = tmp_path / "out"
    process_single_apply_only_core(
        pipeline_path=pipeline_file,
        image_path=source_image,
        input_root=source_image.parent,
        output_dir=out,
        image_type="Image",
        layer="rgb",
        read_kwargs={},
        process_format="zarr",
    )
    journal = _block(out / f"IMG_4471{ngff_.STORE_SUFFIX}")[
        PhenotypicAttr.PROVENANCE
    ]
    assert journal["pipeline"] is not None
    assert journal["operations"] == []


def test_the_store_round_trips_through_imread(
    tmp_path: Path, source_image: Path, pipeline_file: Path
) -> None:
    """The loop closes: what process mode writes, imread reads."""
    out = tmp_path / "out"
    process_single_apply_only_core(
        pipeline_path=pipeline_file,
        image_path=source_image,
        input_root=source_image.parent,
        output_dir=out,
        image_type="Image",
        layer="rgb",
        read_kwargs={},
        process_format="zarr",
    )
    store = out / f"IMG_4471{ngff_.STORE_SUFFIX}"
    assert Image.imread(store).name == "IMG_4471"


def test_tiff_output_is_unchanged(
    tmp_path: Path, source_image: Path, pipeline_file: Path
) -> None:
    """The default path, and the AutoConvertRaw contract, must not move."""
    out = tmp_path / "out"
    process_single_apply_only_core(
        pipeline_path=pipeline_file,
        image_path=source_image,
        input_root=source_image.parent,
        output_dir=out,
        image_type="Image",
        layer="rgb",
        read_kwargs={},
        process_format="tiff",
    )
    assert (out / "IMG_4471.tiff").is_file()
