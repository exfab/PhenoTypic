"""figures/<run>/ groups + attributes.phenotypic.figures descriptor (spec §1, §1a)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from phenotypic.sdk_ import ngff_
from phenotypic.sdk_._image_figures import (
    FigureRun,
    StoredFigureBinding,
    StoredFigureFailure,
    StoredFigureFile,
    StoredFigurePage,
    StoredFigures,
    apply_image_figures_attributes,
    carry_figure_runs,
    read_figure_run,
    read_image_figures_descriptor,
    split_figure_file_path,
    write_image_figures,
)

_JSON = b'{"data": []}'
_PNG = b"\x89PNG fake"
_SHA = "3f9a1c2b7e04" + "0" * 52
_RUN = FigureRun(date="2026-09-22", pipeline_sha256=_SHA)
_OTHER = FigureRun(date="2026-10-03", pipeline_sha256="a07bc5e91d22" + "1" * 52)


def _stored(run: FigureRun = _RUN, *, unavailable=()) -> StoredFigures:
    page = StoredFigurePage(
        key="default", label=None, backend="plotly", metadata={"plate": 1},
        files=(
            StoredFigureFile("plotly-json", "application/vnd.plotly.v1+json",
                             "default.plotly.json", _JSON),
            StoredFigureFile("png", "image/png", "default.png", _PNG),
        ),
    )
    return StoredFigures(
        run=run,
        bindings=(StoredFigureBinding("sym", "MeasureSymZones", "sym", (page,)),),
        failed=(StoredFigureFailure("orient", None, None, "RuntimeError: boom"),),
        unavailable=unavailable,
    )


def _store_with(tmp_path: Path, *runs: StoredFigures) -> Path:
    store = tmp_path / "p.ome.zarr"
    store.mkdir(parents=True)
    phenotypic: dict = {}
    for run in runs:
        apply_image_figures_attributes(phenotypic, write_image_figures(store, run))
    (store / "zarr.json").write_text(json.dumps(
        {"zarr_format": 3, "node_type": "group", "attributes": {"phenotypic": phenotypic}}
    ), encoding="utf-8")
    return store


def test_the_run_id_is_the_date_and_twelve_hex_of_the_pipeline_sha():
    assert _RUN.run_id == "2026-09-22-3f9a1c2b7e04"


@pytest.mark.parametrize(
    "date", ["2026-9-22", "20260922", "22-09-2026", "2026-02-30", "", None]
)
def test_a_malformed_run_date_is_refused(date):
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        FigureRun(date=date, pipeline_sha256=_SHA)


@pytest.mark.parametrize("sha", ["abc", _SHA.upper(), _SHA + "0", "../" + _SHA[3:]])
def test_a_malformed_pipeline_sha_is_refused(sha):
    with pytest.raises(ValueError, match="sha256"):
        FigureRun(date="2026-09-22", pipeline_sha256=sha)


def test_writer_lays_out_one_run_folder_and_a_hash_bound_descriptor(tmp_path: Path):
    fragment = write_image_figures(tmp_path, _stored(unavailable=("cal",)))
    group = json.loads((tmp_path / "figures" / "zarr.json").read_text(encoding="utf-8"))
    assert group == {"zarr_format": 3, "node_type": "group", "attributes": {}}
    for level in (_RUN.run_id, f"{_RUN.run_id}/sym"):
        document = (tmp_path / "figures" / level / "zarr.json").read_text(encoding="utf-8")
        assert json.loads(document) == group
    assert (tmp_path / f"figures/{_RUN.run_id}/sym/default.plotly.json").read_bytes() == _JSON

    descriptor = fragment[ngff_.PhenotypicAttr.FIGURES]
    assert descriptor["schema_version"] == 2
    [(run_id, run)] = descriptor["runs"].items()
    assert run_id == _RUN.run_id
    assert (run["date"], run["pipeline_sha256"]) == ("2026-09-22", _SHA)
    assert run["unavailable"] == ["cal"]
    assert run["bindings"]["sym"]["class"] == "MeasureSymZones"
    page = run["bindings"]["sym"]["pages"][0]
    assert page["metadata"] == {"plate": 1}
    assert [f["format"] for f in page["files"]] == ["plotly-json", "png"]
    for entry in page["files"]:
        assert split_figure_file_path(entry["path"]) == (
            _RUN.run_id, "sym", None, Path(entry["path"]).name
        )
        data = (tmp_path / entry["path"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
    assert run["failed"] == [
        {"binding": "orient", "page": None, "plot": None, "format": None,
         "error": "RuntimeError: boom"}
    ]


def test_the_initial_call_is_in_the_run_entry_but_not_the_run_id(tmp_path: Path):
    """Spec §1a: the CLI call whose run last wrote this folder."""
    from dataclasses import replace

    called = replace(_RUN, initiated_at_utc="2026-09-22T23:59:58.123Z", initiated_pid=4242)
    assert called.run_id == _RUN.run_id
    run = write_image_figures(tmp_path, _stored(called))["figures"]["runs"][_RUN.run_id]
    assert (run["initiated_at_utc"], run["initiated_pid"]) == (
        "2026-09-22T23:59:58.123Z", 4242
    )


def test_a_run_without_the_call_omits_both_fields(tmp_path: Path):
    """Process-mode stores: same-day byte identity (spec §1a)."""
    run = write_image_figures(tmp_path, _stored())["figures"]["runs"][_RUN.run_id]
    assert "initiated_at_utc" not in run and "initiated_pid" not in run


def test_unavailable_is_always_present(tmp_path: Path):
    run = write_image_figures(tmp_path, _stored())["figures"]["runs"][_RUN.run_id]
    assert run["unavailable"] == []


def test_all_failed_writes_the_run_with_empty_bindings(tmp_path: Path):
    stored = StoredFigures(run=_RUN, bindings=(), failed=_stored().failed)
    run = write_image_figures(tmp_path, stored)["figures"]["runs"][_RUN.run_id]
    assert run["bindings"] == {}
    assert (tmp_path / "figures" / _RUN.run_id / "zarr.json").is_file()


def test_apply_merges_a_run_and_keeps_every_other():
    phenotypic: dict = {}
    first = {"figures": {"schema_version": 1, "runs": {"a": {"n": 1}, "b": {"n": 2}}}}
    apply_image_figures_attributes(phenotypic, first)
    apply_image_figures_attributes(
        phenotypic, {"figures": {"schema_version": 1, "runs": {"b": {"n": 3}}}}
    )
    assert phenotypic["figures"]["runs"] == {"a": {"n": 1}, "b": {"n": 3}}


def test_apply_never_merges_with_an_unknown_schema():
    """MINOR-7: never relabelled, never dropped (spec §1a)."""
    newer = {"schema_version": 3, "layout": {"x": 1}}
    phenotypic: dict = {"figures": dict(newer)}
    apply_image_figures_attributes(
        phenotypic, {"figures": {"schema_version": 1, "runs": {"b": {"n": 3}}}}
    )
    assert phenotypic == {"figures": newer}
    # Carried whole into a root that has none: set as it is.
    fresh: dict = {}
    apply_image_figures_attributes(fresh, {"figures": dict(newer)})
    assert fresh == {"figures": newer}


def test_apply_none_changes_nothing():
    """`figures=None` means "no new run", never removal (spec §1a)."""
    phenotypic: dict = {"figures": {"schema_version": 1, "runs": {"a": {}}}}
    apply_image_figures_attributes(phenotypic, None)
    assert phenotypic == {"figures": {"schema_version": 1, "runs": {"a": {}}}}


def test_reader_returns_none_for_a_pre_feature_store(tmp_path: Path):
    (tmp_path / "zarr.json").write_text(json.dumps(
        {"zarr_format": 3, "node_type": "group",
         "attributes": {"phenotypic": {"store_schema_version": 3}}}
    ), encoding="utf-8")
    assert read_image_figures_descriptor(tmp_path) is None
    assert read_figure_run(tmp_path, _RUN.run_id) is None


def test_read_figure_run_picks_one_run_and_refuses_an_unknown_schema(tmp_path: Path):
    store = _store_with(tmp_path, _stored(), _stored(_OTHER))
    assert read_figure_run(store, _OTHER.run_id)["date"] == "2026-10-03"
    assert read_figure_run(store, "2026-01-01-000000000000") is None
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    root["attributes"]["phenotypic"]["figures"]["schema_version"] = 3
    (store / "zarr.json").write_text(json.dumps(root), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version 3"):
        read_figure_run(store, _RUN.run_id)


def test_latest_run_date_is_the_most_recent_run_of_this_pipeline():
    """Measure mode reuses it (spec §1a, revision 14)."""
    from phenotypic.sdk_._image_figures import latest_run_date

    run = lambda date, sha: {"date": date, "pipeline_sha256": sha}  # noqa: E731
    descriptor = {"runs": {
        "a": run("2026-09-22", _SHA),
        "b": run("2026-10-03", _SHA),
        "c": run("2026-12-01", "f" * 64),
        "d": run("2026-09-30", _SHA),
    }}
    assert latest_run_date(descriptor, _SHA) == "2026-10-03"
    assert latest_run_date(descriptor, "e" * 64) is None
    assert latest_run_date(None, _SHA) is None


def test_latest_run_date_ignores_a_malformed_date_of_the_same_pipeline():
    """``"garbage"`` sorts after every ISO date, so it would otherwise win."""
    from phenotypic.sdk_._image_figures import latest_run_date

    descriptor = {"runs": {
        "a": {"date": "2026-09-22", "pipeline_sha256": _SHA},
        "b": {"date": "garbage", "pipeline_sha256": _SHA},
        "c": {"date": "2026-02-30", "pipeline_sha256": _SHA},
    }}
    assert latest_run_date(descriptor, _SHA) == "2026-09-22"


def test_carry_links_every_other_run_byte_for_byte(tmp_path: Path):
    store = _store_with(tmp_path / "old", _stored(), _stored(_OTHER))
    part = tmp_path / "new.part"
    part.mkdir()
    fragment = carry_figure_runs(store, part, exclude=_RUN.run_id)
    runs = fragment["figures"]["runs"]
    assert list(runs) == [_OTHER.run_id]
    assert runs[_OTHER.run_id] == read_figure_run(store, _OTHER.run_id)
    assert not (part / "figures" / _RUN.run_id).exists()
    for entry in runs[_OTHER.run_id]["bindings"]["sym"]["pages"][0]["files"]:
        assert (part / entry["path"]).read_bytes() == (store / entry["path"]).read_bytes()
        # A link, not a copy, where the platform allows one.
        if sys.platform != "win32":
            assert (part / entry["path"]).stat().st_ino == (store / entry["path"]).stat().st_ino
    for level in ("", _OTHER.run_id, f"{_OTHER.run_id}/sym"):
        assert (part / "figures" / level / "zarr.json").is_file()


def test_carry_from_nothing_carries_nothing(tmp_path: Path):
    part = tmp_path / "new.part"
    part.mkdir()
    assert carry_figure_runs(tmp_path / "absent.ome.zarr", part) is None
    lone = _store_with(tmp_path / "lone", _stored())
    assert carry_figure_runs(lone, part, exclude=_RUN.run_id) is None
    assert not (part / "figures").exists()


def test_a_corrupt_file_is_carried_as_it_is_never_dropped(tmp_path: Path, caplog):
    """Never wiped (spec §1a): the sha256 still exposes it to a verifier."""
    store = _store_with(tmp_path / "old", _stored(_OTHER))
    corrupt = store / f"figures/{_OTHER.run_id}/sym/default.png"
    corrupt.write_bytes(b"corrupt")
    part = tmp_path / "new.part"
    part.mkdir()
    runs = carry_figure_runs(store, part)["figures"]["runs"]
    assert runs[_OTHER.run_id] == read_figure_run(store, _OTHER.run_id)
    assert (part / f"figures/{_OTHER.run_id}/sym/default.png").read_bytes() == b"corrupt"
    assert "does not match its recorded sha256" in caplog.text


# --- Descriptor version 2: plot folders (spec 2026-09-30 §2) ---


def _stored_v2(run: FigureRun = _RUN) -> StoredFigures:
    tiles = tuple(
        StoredFigurePage(
            key=f"roi_{i}", label=None, backend="mpl", metadata={},
            files=(StoredFigureFile("png", "image/png", f"roi_{i}.png", _PNG + bytes([i])),),
            plot="tiles", directory="tiles",
        )
        for i in range(2)
    )
    delta = StoredFigurePage(
        key="delta_e", label=None, backend="mpl", metadata={},
        files=(StoredFigureFile("png", "image/png", "delta_e.png", _PNG + b"d"),),
        plot="delta_e", directory="delta_e",
    )
    return StoredFigures(
        run=run,
        bindings=(StoredFigureBinding("cal", "CalibrateColorRpcc", "cal", (*tiles, delta)),),
        failed=(StoredFigureFailure("cal", "roi_2", "png", "OSError: x", plot="tiles"),),
    )


def test_writer_lays_out_plot_folders_with_group_documents(tmp_path: Path):
    fragment = write_image_figures(tmp_path, _stored_v2())
    group = {"zarr_format": 3, "node_type": "group", "attributes": {}}
    for level in ("cal", "cal/tiles", "cal/delta_e"):
        document = tmp_path / "figures" / _RUN.run_id / level / "zarr.json"
        assert json.loads(document.read_text(encoding="utf-8")) == group
    assert (tmp_path / f"figures/{_RUN.run_id}/cal/tiles/roi_1.png").read_bytes() == _PNG + b"\x01"
    descriptor = fragment[ngff_.PhenotypicAttr.FIGURES]
    assert descriptor["schema_version"] == 2
    run = descriptor["runs"][_RUN.run_id]
    pages = run["bindings"]["cal"]["pages"]
    assert [(p["plot"], p["key"]) for p in pages] == [
        ("tiles", "roi_0"), ("tiles", "roi_1"), ("delta_e", "delta_e"),
    ]
    assert pages[0]["files"][0]["path"] == f"figures/{_RUN.run_id}/cal/tiles/roi_0.png"
    assert run["failed"] == [
        {"binding": "cal", "page": "roi_2", "plot": "tiles", "format": "png", "error": "OSError: x"}
    ]


def test_a_flat_page_is_written_flat_with_a_null_plot(tmp_path: Path):
    fragment = write_image_figures(tmp_path, _stored())
    page = fragment["figures"]["runs"][_RUN.run_id]["bindings"]["sym"]["pages"][0]
    assert page["plot"] is None
    assert page["files"][1]["path"] == f"figures/{_RUN.run_id}/sym/default.png"


@pytest.mark.parametrize("path, expected", [
    ("figures/r/b/f.png", ("r", "b", None, "f.png")),
    ("figures/r/b/p/f.png", ("r", "b", "p", "f.png")),
])
def test_split_accepts_both_layouts(path, expected):
    assert split_figure_file_path(path) == expected


@pytest.mark.parametrize("path", [
    "figures/r/f.png", "figures/r/b/p/q/f.png", "figures/r/../p/f.png",
    "figures/r/b/./f.png", "/figures/r/b/f.png", "tables/r/b/f.png",
])
def test_split_refuses_every_other_shape(path):
    with pytest.raises(ValueError):
        split_figure_file_path(path)


def test_figure_file_path_round_trips_both_layouts():
    from phenotypic.sdk_._image_figures import figure_file_path

    for plot in (None, "tiles"):
        path = figure_file_path("r", "b", "f.png", plot)
        assert split_figure_file_path(path) == ("r", "b", plot, "f.png")


def test_a_v1_store_gains_a_v2_run_and_keeps_its_v1_run_byte_for_byte(tmp_path: Path):
    store = _store_with(tmp_path, _stored(_OTHER))
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    root["attributes"]["phenotypic"]["figures"]["schema_version"] = 1   # as 0.19 wrote it
    # The true version 1 shape: 0.19 never wrote "plot".
    v1_run = root["attributes"]["phenotypic"]["figures"]["runs"][_OTHER.run_id]
    for binding in v1_run["bindings"].values():
        for page in binding["pages"]:
            page.pop("plot")
    for failure in v1_run["failed"]:
        failure.pop("plot")
    (store / "zarr.json").write_text(json.dumps(root), encoding="utf-8")
    old_entry = json.loads(json.dumps(v1_run))
    old_bytes = (store / f"figures/{_OTHER.run_id}/sym/default.png").read_bytes()

    part = tmp_path / "p.ome.zarr.part"
    part.mkdir()
    carried = carry_figure_runs(store, part, exclude=_RUN.run_id)
    phenotypic = {"figures": root["attributes"]["phenotypic"]["figures"]}
    apply_image_figures_attributes(phenotypic, carried)
    apply_image_figures_attributes(phenotypic, write_image_figures(part, _stored_v2()))

    assert phenotypic["figures"]["schema_version"] == 2
    assert phenotypic["figures"]["runs"][_OTHER.run_id] == old_entry
    assert (part / f"figures/{_OTHER.run_id}/sym/default.png").read_bytes() == old_bytes
    assert (part / f"figures/{_RUN.run_id}/cal/tiles/roi_0.png").is_file()


def test_read_figure_run_reads_v1_and_v2(tmp_path: Path):
    store = _store_with(tmp_path, _stored())
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    for version in (1, 2):
        root["attributes"]["phenotypic"]["figures"]["schema_version"] = version
        (store / "zarr.json").write_text(json.dumps(root), encoding="utf-8")
        assert read_figure_run(store, _RUN.run_id)["date"] == "2026-09-22"


def test_a_v1_only_writer_adds_no_run_to_a_v2_store(monkeypatch):
    """Review Focus 3: an older PhenoTypic leaves a v2 descriptor alone."""
    v2 = {"schema_version": 2, "runs": {"a": {"n": 1}}}
    phenotypic = {"figures": json.loads(json.dumps(v2))}
    monkeypatch.setattr(ngff_, "FIGURES_SCHEMA_VERSION", 1)
    monkeypatch.setattr(ngff_, "READABLE_FIGURES_SCHEMA_VERSIONS", frozenset({1}))
    apply_image_figures_attributes(
        phenotypic, {"figures": {"schema_version": 1, "runs": {"b": {"n": 2}}}}
    )
    assert phenotypic == {"figures": v2}


def test_a_boolean_schema_version_is_not_a_known_one():
    from phenotypic.sdk_._image_figures import known_figures_schema

    assert known_figures_schema({"schema_version": True}) is False


def test_carry_links_a_v2_run_into_its_plot_folders(tmp_path: Path):
    store = _store_with(tmp_path, _stored_v2(_OTHER))
    part = tmp_path / "p.ome.zarr.part"
    part.mkdir()
    carry_figure_runs(store, part, exclude=_RUN.run_id)
    for name in ("tiles/roi_0.png", "tiles/roi_1.png", "delta_e/delta_e.png"):
        carried = part / f"figures/{_OTHER.run_id}/cal/{name}"
        assert carried.read_bytes() == (store / f"figures/{_OTHER.run_id}/cal/{name}").read_bytes()
    assert (part / f"figures/{_OTHER.run_id}/cal/tiles/zarr.json").is_file()
