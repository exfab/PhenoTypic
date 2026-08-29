import numpy as np
import pytest
from PIL import Image as PILImage

from phenotypic.gui.browse._source_probe import SourceProbeError, probe_source


def test_probe_reads_headers_and_builds_revision_identity(tmp_path):
    source = tmp_path / "dataset" / "plate.png"
    source.parent.mkdir()
    PILImage.fromarray(np.zeros((12, 20, 3), dtype=np.uint8)).save(source)

    revision = probe_source(source, sandbox_root=tmp_path)

    assert revision.relative_path == "dataset/plate.png"
    assert (revision.width, revision.height) == (20, 12)
    assert revision.size_bytes == source.stat().st_size
    assert len(revision.cache_key) == 64
    assert revision.matches_disk()


def test_revision_changes_when_source_changes(tmp_path):
    source = tmp_path / "plate.png"
    PILImage.new("RGB", (2, 2), "red").save(source)
    first = probe_source(source, sandbox_root=tmp_path)
    PILImage.new("RGB", (3, 3), "blue").save(source)
    second = probe_source(source, sandbox_root=tmp_path)

    assert first.cache_key != second.cache_key
    assert not first.matches_disk()


def test_dzi_parameters_are_part_of_cache_identity(tmp_path):
    source = tmp_path / "plate.png"
    PILImage.new("RGB", (2, 2)).save(source)

    default = probe_source(source, sandbox_root=tmp_path)
    changed = probe_source(source, sandbox_root=tmp_path, tile_size=512)

    assert default.source_id == changed.source_id
    assert default.cache_key != changed.cache_key


def test_store_revision_changes_when_nested_member_changes(tmp_path):
    store = tmp_path / "p01.ome.zarr"
    chunk = store / "rgb" / "c" / "0"
    chunk.parent.mkdir(parents=True)
    chunk.write_bytes(b"first")
    (store / "zarr.json").write_text("{}", encoding="utf-8")

    first = probe_source(store, sandbox_root=tmp_path)
    chunk.write_bytes(b"second-version")
    second = probe_source(store, sandbox_root=tmp_path)

    assert first.cache_key != second.cache_key
    assert not first.matches_disk()


def test_store_probe_rejects_nested_symlink(tmp_path):
    store = tmp_path / "p01.ome.zarr"
    store.mkdir()
    outside = tmp_path / "outside"
    outside.write_bytes(b"secret")
    link = store / "chunk"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):  # pragma: no cover - platform guard
        pytest.skip("symlinks not supported on this platform/filesystem")

    with pytest.raises(SourceProbeError):
        probe_source(store, sandbox_root=tmp_path)
