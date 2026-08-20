"""``OutputRoot`` resolves per-image stores instead of per-image HDFs.

Two failure shapes this module pins, both of which a naive port produces
silently rather than loudly (OPEN-QUESTIONS D4 / D5):

* Enumerating store **directories** instead of each store's root
  ``zarr.json`` reduces the enumeration to a constant --
  ``_cancellable_paths_fingerprint`` emits one sentinel byte for a directory
  and does not recurse (``_output_root.py:832-834``), and the processing
  inventory's ``read_only_bounded`` scan drops directory metadata entirely
  (``_processing_inventory.py:371-379``).
* ``_image_source_token`` hashes ``st_dev``/``st_ino``/``st_size``/
  ``st_mtime_ns``/``st_ctime_ns``, none of which move when a nested chunk is
  rewritten -- so a store-directory port goes blind to republishes.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from phenotypic.gui.results_viewer._output_root import OutputRoot
from phenotypic.schema import IMAGE
from phenotypic.sdk_ import zarr_store_path

from tests._output_layout import write_complete_manifest, write_master


def _discover(root: Path) -> OutputRoot:
    """Discover with a test-owned cache OUTSIDE the selected output."""
    source = Path(root).resolve()
    return OutputRoot.discover(
        source, cache_root=source.parent / ".test-phenotypic-viewer-cache"
    )


def _seed(root: Path, stems: list[str]) -> None:
    """Seed a minimal store-backed output: real master, then one store per stem."""
    write_master(
        root,
        pl.DataFrame(
            {
                "Metadata_Dataset": ["ds"] * len(stems),
                str(IMAGE.IMAGE_NAME): list(stems),
                "Size_Area": [100.0] * len(stems),
            }
        ),
    )
    write_complete_manifest(root, total_images=len(stems))
    (root / "results" / "ds" / "measurements").mkdir(parents=True, exist_ok=True)
    for stem in stems:
        store = zarr_store_path(root, "ds", stem)
        (store / "gray" / "0").mkdir(parents=True)
        (store / "gray" / "0" / "c.0.0").write_bytes(b"chunk")
        (store / "zarr.json").write_text("{}", encoding="utf-8")


def test_store_path_resolves_a_directory(tmp_path: Path) -> None:
    _seed(tmp_path, ["a"])
    root = _discover(tmp_path)
    assert root.store_path("ds", "a") == zarr_store_path(tmp_path, "ds", "a")


def test_store_path_is_none_when_absent(tmp_path: Path) -> None:
    _seed(tmp_path, ["a"])
    assert _discover(tmp_path).store_path("ds", "missing") is None


def test_store_path_ignores_a_plain_file_named_like_a_store(
    tmp_path: Path,
) -> None:
    """``is_dir``, not ``exists``: a store is a directory, never a file."""
    _seed(tmp_path, ["a"])
    decoy = zarr_store_path(tmp_path, "ds", "decoy")
    decoy.parent.mkdir(parents=True, exist_ok=True)
    decoy.write_bytes(b"not a store")
    assert _discover(tmp_path).store_path("ds", "decoy") is None


def test_store_path_finds_every_store(tmp_path: Path) -> None:
    _seed(tmp_path, ["a", "b", "c"])
    root = _discover(tmp_path)
    assert all(root.store_path("ds", stem) is not None for stem in "abc")


def test_has_image_source_follows_the_store(tmp_path: Path) -> None:
    """The picker's per-image gate must ask about the store, not an HDF."""
    _seed(tmp_path, ["a"])
    root = _discover(tmp_path)
    assert root.has_image_source("ds", "a") is True
    assert root.has_image_source("ds", "missing") is False


def test_discovery_never_lists_a_directory_inside_a_store(
    tmp_path: Path, monkeypatch
) -> None:
    """The COST of discovery, asserted on directory listings (user ruling).

    ``_scan_processing_inventory`` used an unbounded ``results_root.rglob``,
    which descends into every store: a 4000x3000 plate's store holds 58
    entries where the ``.h5`` it replaced was 1, so 10k images cost ~580,000
    stat calls instead of ~10,000, on every viewer open.

    Asserted on ``os.scandir`` rather than on the entry list, because the
    two are NOT equivalent: a recursive walk that filters its results back
    down to the store roots produces the identical inventory while doing all
    the work the bound exists to avoid. That mutation survived a result-set
    assertion once already in this phase.
    """
    import os

    _seed(tmp_path, ["a", "b"])
    stores = [zarr_store_path(tmp_path, "ds", stem) for stem in ("a", "b")]
    listed: list[str] = []
    real_scandir = os.scandir

    def _counting(path=".", *args, **kwargs):
        listed.append(os.fspath(path))
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", _counting)
    _discover(tmp_path)

    inside = [
        seen
        for seen in listed
        for store in stores
        if Path(seen) == store or store in Path(seen).parents
    ]
    assert inside == [], inside


def test_discovery_still_walks_the_rest_of_the_results_tree(
    tmp_path: Path,
) -> None:
    """Only stores are pruned. Everything else keeps its exhaustive walk.

    ``measurements/*.parquet`` are per-image processing products too, and
    nothing about them changed -- pruning them as well would trade a real
    cost problem for a real correctness one.
    """
    from phenotypic.gui.results_viewer._processing_inventory import (
        ProcessingInventoryEntry,
    )

    _seed(tmp_path, ["a"])
    measurement = tmp_path / "results" / "ds" / "measurements" / "a.parquet"
    measurement.write_bytes(b"parquet-ish")

    entries = _discover(tmp_path).processing_inventory.entries
    recorded = {entry.relative_path for entry in entries}
    assert "results/ds/measurements/a.parquet" in recorded
    assert "results/ds/zarr/a.ome.zarr/zarr.json" in recorded
    assert not any(
        path.startswith("results/ds/zarr/a.ome.zarr/gray") for path in recorded
    ), sorted(recorded)
    assert all(isinstance(entry, ProcessingInventoryEntry) for entry in entries)


def test_processing_inventory_goes_stale_after_a_store_republish(
    tmp_path: Path,
) -> None:
    """The property FLOW-11 wanted, asserted against the real producer.

    Ledger FLOW-11 named ``_processing_snapshot_paths`` as the mechanism; it
    never had a production caller and Phase 6 deleted it. The property it
    wanted is the right one either way: after a store republish the captured
    inventory must stop verifying, or a viewer bound across a run never
    notices new pixels.
    """
    from phenotypic.gui.results_viewer._discovery_contracts import (
        OutputDiscoveryCancellation,
    )
    from phenotypic.gui.results_viewer._processing_inventory import (
        inventory_is_current,
    )

    _seed(tmp_path, ["a"])
    source = tmp_path.resolve()
    inventory = _discover(tmp_path).processing_inventory
    assert inventory_is_current(
        inventory,
        source_root=source,
        cancellation=OutputDiscoveryCancellation(),
        progress=None,
    )

    (zarr_store_path(tmp_path, "ds", "a") / "zarr.json").write_text(
        '{"republished": 1}', encoding="utf-8"
    )
    assert not inventory_is_current(
        inventory,
        source_root=source,
        cancellation=OutputDiscoveryCancellation(),
        progress=None,
    )


def test_processing_fingerprint_changes_when_a_store_changes(tmp_path: Path) -> None:
    """The end-to-end property D5 exists to protect.

    Note the mechanism was never ``_processing_snapshot_paths`` -- that
    helper had no production caller and Phase 6 deleted it.
    ``source_fingerprint`` is ``ProcessingInventory.fingerprint``, built by
    ``_scan_processing_inventory``, whose results walk is now bounded to each
    store's root. This asserts the property from the outside, so it holds
    across that change rather than describing one enumeration.
    """
    _seed(tmp_path, ["a"])
    before = _discover(tmp_path).source_fingerprint
    (zarr_store_path(tmp_path, "ds", "a") / "zarr.json").write_text(
        '{"changed": true}', encoding="utf-8"
    )
    assert _discover(tmp_path).source_fingerprint != before


def test_image_source_token_changes_when_a_store_changes(tmp_path: Path) -> None:
    """It is a staleness fingerprint, not a report label (D4)."""
    from phenotypic.gui.results_viewer._output_root import _image_source_token
    from phenotypic.sdk_ import BundleLayout

    _seed(tmp_path, ["a"])
    layout = BundleLayout.detect(tmp_path)
    before = _image_source_token(layout, "ds", "a", has_overlay=False)
    (zarr_store_path(tmp_path, "ds", "a") / "zarr.json").write_text(
        '{"changed": true}', encoding="utf-8"
    )
    assert _image_source_token(layout, "ds", "a", has_overlay=False) != before


def test_image_source_token_ignores_a_chunk_rewrite_only_via_the_root(
    tmp_path: Path,
) -> None:
    """Keying on the store DIRECTORY would freeze the token permanently.

    The directory's own ``st_mtime_ns`` does not move when a nested chunk is
    rewritten, so this is the mutation the D4 fix has to survive: the token
    must key on a real file whose stat moves on every publish.
    """
    from phenotypic.gui.results_viewer._output_root import _image_source_token
    from phenotypic.sdk_ import BundleLayout

    _seed(tmp_path, ["a"])
    layout = BundleLayout.detect(tmp_path)
    store = zarr_store_path(tmp_path, "ds", "a")
    before = _image_source_token(layout, "ds", "a", has_overlay=False)
    # A directory-keyed token is a constant of the path: prove the port did
    # not merely relabel by showing the store dir's stat is unchanged while
    # the token still moves.
    dir_stat_before = store.stat().st_mtime_ns
    (store / "zarr.json").write_text('{"republished": 1}', encoding="utf-8")
    assert store.stat().st_mtime_ns == dir_stat_before
    assert _image_source_token(layout, "ds", "a", has_overlay=False) != before


def test_the_bound_token_tracks_the_store_too(tmp_path: Path) -> None:
    """The public surface the token actually reaches the viewer through."""
    _seed(tmp_path, ["a"])
    before = _discover(tmp_path).bound_image_source_token("ds", "a")
    (zarr_store_path(tmp_path, "ds", "a") / "zarr.json").write_text(
        '{"changed": true}', encoding="utf-8"
    )
    assert _discover(tmp_path).bound_image_source_token("ds", "a") != before
