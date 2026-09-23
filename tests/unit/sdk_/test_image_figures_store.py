"""figures/<run>/ rides the root-last transaction (spec §1, §1a, §3 step 2)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

from phenotypic import Image
from phenotypic._cli._embedded_measurement_tables import prepare_image_tables
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_ import ngff_
from phenotypic.sdk_._image_figures import (
    FigureRun,
    StoredFigureBinding,
    StoredFigureFile,
    StoredFigurePage,
    StoredFigures,
    read_image_figures_descriptor,
)
from phenotypic.sdk_._measurement_tables import replace_image_tables


RUN = FigureRun(date="2026-09-22", pipeline_sha256="ab" * 32)
LATER = FigureRun(date="2026-10-03", pipeline_sha256="cd" * 32)


def _figures(
    tag: bytes = b"one", binding: str = "sym", run: FigureRun = RUN
) -> StoredFigures:
    page = StoredFigurePage("default", None, "plotly", {}, (
        StoredFigureFile("plotly-json", "application/vnd.plotly.v1+json",
                         "default.plotly.json", tag),
    ))
    return StoredFigures(run, (StoredFigureBinding(binding, "X", binding, (page,)),), ())


def _at(relative: str, run: FigureRun = RUN) -> str:
    return f"figures/{run.run_id}/{relative}"


def _runs(store) -> dict:
    return read_image_figures_descriptor(store)["runs"]


def _snapshot(store, run: FigureRun):
    """Bytes and inode of every file in one run folder."""
    folder = store / "figures" / run.run_id
    return {
        p.relative_to(store).as_posix(): (p.read_bytes(), p.stat().st_ino)
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def _tables():
    return prepare_image_tables(pd.DataFrame({"Object_Label": [1]}), None)


@pytest.fixture(scope="module")
def plate() -> Image:
    return Image(load_synth_yeast_plate())


def test_save2zarr_writes_figures_and_an_independent_reader_opens_them(tmp_path, plate):
    import zarr

    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures())
    run = _runs(store)[RUN.run_id]
    assert run["bindings"]["sym"]["pages"][0]["files"][0]["path"] == _at(
        "sym/default.plotly.json"
    )
    root = zarr.open_group(str(store), mode="r")
    assert isinstance(root["figures"], zarr.Group)
    assert isinstance(root[f"figures/{RUN.run_id}/sym"], zarr.Group)
    ome = json.loads((store / "OME" / "zarr.json").read_text(encoding="utf-8"))
    assert "figures" not in ome["attributes"]["ome"]["series"]
    assert "figures" not in (store / "OME" / "METADATA.ome.xml").read_text(
        encoding="utf-8"
    )


def test_no_figures_means_no_key_and_no_group(tmp_path, plate):
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    assert read_image_figures_descriptor(store) is None
    assert not (store / "figures").exists()


def test_process_writer_carries_figures_inside_the_consolidated_store(tmp_path, plate):
    from phenotypic._cli._cli_process_only import write_process_only_layer

    out = tmp_path / "p.ome.zarr"
    write_process_only_layer(plate, "rgb", out, fmt="zarr", figures=_figures())
    assert (out / _at("sym/default.plotly.json")).read_bytes() == b"one"
    root = json.loads((out / "zarr.json").read_text(encoding="utf-8"))
    assert _at("sym") in root["consolidated_metadata"]["metadata"]


def test_a_measure_rewrite_adds_its_run_and_keeps_the_others(tmp_path, plate):
    """Two runs, one store (spec §1a): the earlier folder stays, hard-linked."""
    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures(b"old", "gone"))
    earlier = _snapshot(store, RUN)
    earlier_entry = _runs(store)[RUN.run_id]
    pixel = next(
        p for p in (store / "rgb" / "0").rglob("*") if p.is_file() and p.name != "zarr.json"
    )
    pixel_inode = pixel.stat().st_ino
    replace_image_tables(
        store, _tables(), objmap_target=ngff_.objmap_path("rgb"),
        figures=_figures(b"new", "kept", LATER),
    )
    assert _snapshot(store, RUN) == earlier
    assert _runs(store)[RUN.run_id] == earlier_entry
    assert (store / _at("kept/default.plotly.json", LATER)).read_bytes() == b"new"
    assert set(_runs(store)) == {RUN.run_id, LATER.run_id}
    assert pixel.stat().st_ino == pixel_inode


def test_the_same_run_id_replaces_only_that_folder(tmp_path, plate):
    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures(b"later", "x", LATER))
    replace_image_tables(
        store, _tables(), objmap_target=ngff_.objmap_path("rgb"),
        figures=_figures(b"old", "gone"),
    )
    other = _snapshot(store, LATER)
    replace_image_tables(
        store, _tables(), objmap_target=ngff_.objmap_path("rgb"),
        figures=_figures(b"new", "kept"),
    )
    assert not (store / _at("gone")).exists()
    assert (store / _at("kept/default.plotly.json")).read_bytes() == b"new"
    assert list(_runs(store)[RUN.run_id]["bindings"]) == ["kept"]
    assert _snapshot(store, LATER) == other


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="holds a descriptor across a directory rename, which Windows refuses",
)
def test_a_same_name_rebuild_never_writes_through_into_the_live_store(tmp_path, plate):
    """Spec §5: the part's copies are hard links; the new bytes must be new files."""
    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures(b"old-bytes", "sym"))
    held = os.open(store / _at("sym/default.plotly.json"), os.O_RDONLY)
    try:
        replace_image_tables(
            store, _tables(), objmap_target=ngff_.objmap_path("rgb"),
            figures=_figures(b"new-bytes", "sym"),
        )
        os.lseek(held, 0, os.SEEK_SET)
        assert os.read(held, 32) == b"old-bytes"
    finally:
        os.close(held)
    assert (store / _at("sym/default.plotly.json")).read_bytes() == b"new-bytes"


def test_no_new_run_touches_no_figure(tmp_path, plate):
    """`figures=None` -- the default, and a pipeline with no image binding --
    adds no run and removes none (spec §1a; KEEP_FIGURES is retired)."""
    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures())
    files_before = _snapshot(store, RUN)
    descriptor_before = read_image_figures_descriptor(store)
    replace_image_tables(store, _tables(), objmap_target=ngff_.objmap_path("rgb"))
    replace_image_tables(
        store, _tables(), objmap_target=ngff_.objmap_path("rgb"), figures=None
    )
    assert _snapshot(store, RUN) == files_before
    assert read_image_figures_descriptor(store) == descriptor_before


def test_a_save2zarr_over_an_existing_store_keeps_the_other_runs(tmp_path, plate):
    """A `save2zarr` over an existing store -- what `--restart`, a re-derived
    process run and Stage 3 over Stage 1 each do -- carries every other run
    (spec §1a). `--overwrite` deletes the output tree before the run, so it
    never reaches this path and is not covered here."""
    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures(b"first", "a"))
    first = {k: v[0] for k, v in _snapshot(store, RUN).items()}
    first_entry = _runs(store)[RUN.run_id]
    plate.save2zarr(store, figures=_figures(b"second", "b", LATER))
    assert {k: v[0] for k, v in _snapshot(store, RUN).items()} == first
    assert _runs(store)[RUN.run_id] == first_entry
    assert set(_runs(store)) == {RUN.run_id, LATER.run_id}
    plate.save2zarr(store)
    assert set(_runs(store)) == {RUN.run_id, LATER.run_id}
    plate.save2zarr(store, figures=_figures(b"third", "c", LATER))
    assert not (store / _at("b", LATER)).exists()
    assert (store / _at("c/default.plotly.json", LATER)).read_bytes() == b"third"
    assert {k: v[0] for k, v in _snapshot(store, RUN).items()} == first


def test_a_re_derived_process_store_carries_another_days_run(tmp_path, plate):
    """MINOR-12: the consolidated process writer, over a store holding another
    day's run, carries it and consolidates both runs' groups (spec §1a)."""
    import zarr

    from phenotypic._cli._cli_process_only import write_process_only_layer

    out = tmp_path / "p.ome.zarr"
    write_process_only_layer(plate, "rgb", out, fmt="zarr", figures=_figures(b"old", "a"))
    earlier = {k: v[0] for k, v in _snapshot(out, RUN).items()}
    earlier_entry = _runs(out)[RUN.run_id]
    write_process_only_layer(
        plate, "rgb", out, fmt="zarr", figures=_figures(b"new", "b", LATER)
    )
    assert {k: v[0] for k, v in _snapshot(out, RUN).items()} == earlier
    assert _runs(out)[RUN.run_id] == earlier_entry
    assert set(_runs(out)) == {RUN.run_id, LATER.run_id}
    listed = json.loads((out / "zarr.json").read_text(encoding="utf-8"))[
        "consolidated_metadata"
    ]["metadata"]
    for group in (f"figures/{RUN.run_id}", _at("a"), f"figures/{LATER.run_id}", _at("b", LATER)):
        assert group in listed
    root = zarr.open_group(str(out), mode="r")
    assert isinstance(root[_at("a")], zarr.Group)
    assert isinstance(root[_at("b", LATER)], zarr.Group)


def test_the_writer_never_writes_through_a_hard_link_it_finds(tmp_path):
    """MINOR-6: a file left in the part -- a hard link into the live store that
    a clear missed -- is replaced by a new file, never written into."""
    from phenotypic.sdk_._image_figures import write_image_figures

    live = tmp_path / "live.plotly.json"
    live.write_bytes(b"published")
    part = tmp_path / "p.ome.zarr.part"
    planted = part / _at("sym/default.plotly.json")
    planted.parent.mkdir(parents=True)
    try:
        os.link(live, planted)
    except OSError:
        pytest.skip("this filesystem refuses hard links")
    write_image_figures(part, _figures(b"new"))
    assert live.read_bytes() == b"published"
    assert planted.read_bytes() == b"new"


def _relabel_as_newer(store) -> dict:
    """Give *store* a figures layout this writer does not know: a newer
    ``schema_version`` and a file only that layout names."""
    root_path = store / "zarr.json"
    root = json.loads(root_path.read_text(encoding="utf-8"))
    root["attributes"]["phenotypic"]["figures"]["schema_version"] = 2
    root_path.write_text(json.dumps(root), encoding="utf-8")
    (store / "figures" / "newer.bin").write_bytes(b"newer layout")
    return read_image_figures_descriptor(store)


def _figures_tree(store) -> dict:
    return {
        p.relative_to(store).as_posix(): (p.read_bytes(), p.stat().st_ino)
        for p in sorted((store / "figures").rglob("*"))
        if p.is_file()
    }


def _without_inodes(tree: dict) -> dict:
    return {path: data for path, (data, _inode) in tree.items()}


def test_a_save_over_an_unknown_figures_schema_carries_it_untouched(tmp_path, plate):
    """MINOR-7, full save: the newer layout is carried whole -- files linked,
    descriptor as it is -- and gains no run, not even this run's id."""
    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures(b"old"))
    newer = _relabel_as_newer(store)
    before = _figures_tree(store)
    plate.save2zarr(store, figures=_figures(b"new"))
    assert read_image_figures_descriptor(store) == newer
    after = _figures_tree(store)
    assert _without_inodes(after) == _without_inodes(before)
    if sys.platform != "win32":
        assert after == before


def test_a_measure_rewrite_leaves_an_unknown_figures_schema_untouched(tmp_path, plate):
    """MINOR-7, measure rewrite: same behaviour as the full save."""
    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures(b"old"))
    newer = _relabel_as_newer(store)
    before = _without_inodes(_figures_tree(store))
    replace_image_tables(
        store, _tables(), objmap_target=ngff_.objmap_path("rgb"), figures=_figures(b"new")
    )
    assert read_image_figures_descriptor(store) == newer
    assert _without_inodes(_figures_tree(store)) == before


def test_the_descriptor_reader_answers_none_for_a_foreign_root(tmp_path):
    """MINOR-2: a third-party store has no `phenotypic` block at all."""
    store = tmp_path / "foreign.ome.zarr"
    store.mkdir()
    (store / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {"ome": {}}}),
        encoding="utf-8",
    )
    assert read_image_figures_descriptor(store) is None
    with pytest.raises(FileNotFoundError):
        read_image_figures_descriptor(tmp_path / "missing.ome.zarr")


def test_figure_files_are_written_through_long_path(tmp_path, monkeypatch):
    """MINOR-6: a page key near MAX_PATH must not fail the store on Windows."""
    from phenotypic.sdk_._image_figures import write_image_figures

    seen = []
    real = ngff_.long_path

    def _spy(path):
        seen.append(Path(path).name)
        return real(path)

    monkeypatch.setattr(ngff_, "long_path", _spy)
    part = tmp_path / "p.ome.zarr.part"
    part.mkdir()
    write_image_figures(part, _figures())
    assert "default.plotly.json" in seen
    assert (part / _at("sym/default.plotly.json")).read_bytes() == b"one"


def test_measure_rebuild_writes_figures_before_the_root_is_promoted(
    tmp_path, plate, monkeypatch
):
    """Spec §5: judged at the promote, not from the final tree."""
    import hashlib

    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    real_promote = ngff_.promote_store
    seen = {}

    def _spy(part, final, **kwargs):
        root = json.loads((Path(part) / "zarr.json").read_text(encoding="utf-8"))
        run = root["attributes"]["phenotypic"]["figures"]["runs"][RUN.run_id]
        entry = run["bindings"]["sym"]["pages"][0]["files"][0]
        data = (Path(part) / entry["path"]).read_bytes()
        seen["match"] = hashlib.sha256(data).hexdigest() == entry["sha256"]
        return real_promote(part, final, **kwargs)

    monkeypatch.setattr(ngff_, "promote_store", _spy)
    replace_image_tables(
        store, _tables(), objmap_target=ngff_.objmap_path("rgb"), figures=_figures()
    )
    assert seen == {"match": True}


def test_migrates_table_replace_leaves_figures_untouched(tmp_path, plate):
    from phenotypic.sdk_._measurement_tables import replace_embedded_measurement_table

    store = plate.save2zarr(tmp_path / "p.ome.zarr", figures=_figures())
    before = read_image_figures_descriptor(store)
    replace_embedded_measurement_table(
        store, _tables().measurements_payload(), objmap_target=ngff_.objmap_path("rgb")
    )
    assert read_image_figures_descriptor(store) == before
    assert (store / _at("sym/default.plotly.json")).read_bytes() == b"one"
