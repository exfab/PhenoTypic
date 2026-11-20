"""The example time series loaders: each kind of data is fetched on demand.

These tests never touch the network. They build a small stand-in for each
release asset, register its real SHA256 in place of the release's, and point the
registry at a local folder, so pooch's own hash check and unzip run for real.
"""
from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

import pytest

from phenotypic.data import _remote_registry as registry
from phenotypic.data import (
    EXAMPLE_PLATES,
    load_example_images,
    load_example_measurements,
    load_example_metadata,
)

KEY = ["Metadata_ImageName", "Grid_RowNum", "Grid_ColNum"]


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def local_release(tmp_path, monkeypatch):
    """Stand-in assets served from a local folder; records every fetched name."""
    served = tmp_path / "release"
    served.mkdir()
    (served / "example-metadata.tsv").write_text(
        "\t".join(KEY + ["Metadata_Strain"]) + "\n"
        "plate1_t000\t0\t0\tBLANK-1\n"
        "plate1_t000\t0\t1\tSYN-83\n",
        encoding="utf-8",
    )
    (served / "example-measurements.tsv").write_text(
        "\t".join(KEY + ["Object_Label", "Size_Area"]) + "\n"
        "plate1_t000\t0\t1\t1\t1234\n",
        encoding="utf-8",
    )
    for plate in EXAMPLE_PLATES:
        with zipfile.ZipFile(served / f"plate{plate}.zip", "w") as zf:
            for hours in (0, 18):
                zf.writestr(f"plate{plate}_t{hours:03d}.ome.zarr/zarr.json", "{}")
    fetched: list[str] = []
    real_fetch = registry._EXAMPLES.fetch

    def copy_from_served(url, output_file, _pooch):
        # pooch speaks http(s)/ftp/sftp only; copy the stand-in asset instead.
        Path(output_file).write_bytes((served / url.rsplit("/", 1)[-1]).read_bytes())

    def fetch(name, *args, **kwargs):
        fetched.append(name)
        return real_fetch(name, *args, downloader=copy_from_served, **kwargs)

    monkeypatch.setattr(registry._EXAMPLES, "path", tmp_path / "cache")
    monkeypatch.setattr(registry._EXAMPLES, "base_url", "https://example.invalid/data-v2/")
    monkeypatch.setattr(
        registry._EXAMPLES, "registry", {p.name: _sha(p) for p in served.iterdir()}
    )
    monkeypatch.setattr(registry._EXAMPLES, "fetch", fetch)
    return fetched, served


def test_the_registry_pins_every_asset_by_sha256():
    assert set(registry._EXAMPLES.registry) == {
        "plate1.zip", "plate2.zip", "plate3.zip",
        "example-metadata.tsv", "example-measurements.tsv",
    }
    for digest in registry._EXAMPLES.registry.values():
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
    assert registry._EXAMPLES.base_url.endswith("/releases/download/data-v2/")


def test_metadata_fetches_only_the_metadata_table(local_release):
    fetched, _ = local_release
    meta = load_example_metadata()
    assert meta.columns.tolist() == KEY + ["Metadata_Strain"]
    assert fetched == ["example-metadata.tsv"]


def test_measurements_fetch_only_the_measurement_table_and_join_on_image_and_well(local_release):
    fetched, _ = local_release
    m = load_example_measurements()
    assert fetched == ["example-measurements.tsv"]
    joined = m.merge(load_example_metadata(), on=KEY)
    assert joined.loc[0, "Metadata_Strain"] == "SYN-83"


def test_images_fetch_only_the_plates_asked_for(local_release):
    fetched, _ = local_release
    stores = load_example_images(plates=[2])
    assert fetched == ["plate2.zip"]
    assert [s.name for s in stores] == ["plate2_t000.ome.zarr", "plate2_t018.ome.zarr"]
    assert all(s.is_dir() for s in stores)


def test_all_plates_by_default_sorted_by_name(local_release):
    fetched, _ = local_release
    names = [s.name for s in load_example_images()]
    assert names == sorted(names) and len(names) == 6
    assert sorted(fetched) == ["plate1.zip", "plate2.zip", "plate3.zip"]


def test_an_unknown_plate_is_refused_before_any_download(local_release):
    fetched, _ = local_release
    with pytest.raises(ValueError, match=r"unknown example plate\(s\) \[4\]"):
        load_example_images(plates=[1, 4])
    assert fetched == []


def test_a_second_call_reuses_the_cache(local_release):
    _, served = local_release
    load_example_images(plates=[1])
    (served / "plate1.zip").unlink()       # the next call must not need the source
    assert [s.name for s in load_example_images(plates=[1])] == [
        "plate1_t000.ome.zarr", "plate1_t018.ome.zarr",
    ]
