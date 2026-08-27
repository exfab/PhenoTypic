"""One resolver for every client-controlled path tail inside a store.

The symlink test is the one that matters and the one an earlier draft
failed: the restriction must bind the RESOLVED path, not the URL segments.
Checking the unresolved head lets a symlink inside a readable root escape it
while still passing containment.

The store here is a plain directory tree rather than a written
``*.ome.zarr``: every property under test is filesystem resolution, and
``save2zarr`` would add ~2 s per module for pixels nothing reads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from werkzeug.exceptions import BadRequest, NotFound

from phenotypic.gui._shared.tiles import resolve_within_root

ROOTS = frozenset({"OME", "rgb", "gray", "detect_mat"})


@pytest.fixture
def tmp_store(tmp_path: Path) -> Path:
    """A store-shaped directory tree with one chunk and a root ``zarr.json``."""
    store = tmp_path / "plate.ome.zarr"
    (store / "rgb" / "0").mkdir(parents=True)
    (store / "rgb" / "0" / "c.0.0.0").write_bytes(b"chunk")
    (store / "OME").mkdir()
    (store / "OME" / "METADATA.ome.xml").write_bytes(b"<OME/>")
    (store / ngff_root_json()).write_bytes(b"{}")
    return store


def ngff_root_json() -> str:
    """The store's root metadata filename, read from the backend contract."""
    from phenotypic.sdk_ import ngff_

    return ngff_.STORE_ROOT_JSON


def test_resolves_a_file_inside_an_allowed_root(tmp_store: Path) -> None:
    got = resolve_within_root(tmp_store, "rgb/0/c.0.0.0", allowed_roots=ROOTS)
    assert got == (tmp_store / "rgb" / "0" / "c.0.0.0").resolve()


def test_the_root_zarr_json_is_exempt_at_depth_one(tmp_store: Path) -> None:
    """The client bootstraps from it, and it belongs to no series.

    It is exempt only as a whole first component -- ``allowed_roots`` never
    names it, so a depth-1 exemption is the only thing that serves it.
    """
    got = resolve_within_root(
        tmp_store, ngff_root_json(), allowed_roots=ROOTS
    )
    assert got == (tmp_store / ngff_root_json()).resolve()


def test_rejects_a_disallowed_root(tmp_store: Path) -> None:
    target = tmp_store / "tables" / "measurements" / "table.parquet"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"secret")
    with pytest.raises(NotFound):
        resolve_within_root(
            tmp_store, "tables/measurements/table.parquet", allowed_roots=ROOTS
        )


def test_a_symlink_into_a_disallowed_root_is_rejected(tmp_store: Path) -> None:
    """The escape an unresolved head check misses.

    ``rgb/sneak`` passes a head check on segments[0] and resolves to a path
    still INSIDE the store, so containment passes too. Only testing the
    resolved path's first component catches it.
    """
    target = tmp_store / "tables" / "measurements" / "table.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"secret")
    (tmp_store / "rgb" / "sneak").symlink_to(target)

    # Premise: the escape really does pass the two weaker checks, so the
    # test fails loudly rather than silently if the fixture stops modelling
    # the attack.
    assert (tmp_store / "rgb" / "sneak").resolve().is_relative_to(
        tmp_store.resolve()
    )
    assert "rgb" in ROOTS

    with pytest.raises((NotFound, BadRequest)):
        resolve_within_root(tmp_store, "rgb/sneak", allowed_roots=ROOTS)


def test_a_symlink_out_of_the_store_is_rejected(tmp_store: Path) -> None:
    """Containment still binds: an escaping symlink is a BadRequest."""
    outside = tmp_store.parent / "outside.txt"
    outside.write_bytes(b"secret")
    (tmp_store / "rgb" / "escape").symlink_to(outside)
    with pytest.raises((NotFound, BadRequest)):
        resolve_within_root(tmp_store, "rgb/escape", allowed_roots=ROOTS)


def test_label_group_under_a_series_resolves(tmp_store: Path) -> None:
    """Only the FIRST resolved component is restricted, by design.

    Restricting every component would block ``labels``, ``objmap`` and every
    level index, killing the label layer.
    """
    p = tmp_store / "rgb" / "labels" / "objmap" / "0"
    p.mkdir(parents=True, exist_ok=True)
    (p / "c.0.0").write_bytes(b"x")
    assert resolve_within_root(
        tmp_store, "rgb/labels/objmap/0/c.0.0", allowed_roots=ROOTS
    )


@pytest.mark.parametrize(
    "tail",
    [
        "../../../../etc/passwd",
        "rgb/../../../../etc/passwd",
        "rgb/0/%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "rgb/./../../zarr.json",
        "rgb/0/.hidden",
        "",
    ],
)
def test_rejects_traversal_in_any_segment(tmp_store: Path, tail: str) -> None:
    with pytest.raises((BadRequest, NotFound)):
        resolve_within_root(tmp_store, tail, allowed_roots=ROOTS)


def test_an_empty_allow_list_rejects_everything(tmp_store: Path) -> None:
    """Fail-closed, and the reason ``allowed_roots`` has no permissive value."""
    with pytest.raises(NotFound):
        resolve_within_root(
            tmp_store, "rgb/0/c.0.0.0", allowed_roots=frozenset()
        )


def test_a_directory_is_not_a_file(tmp_store: Path) -> None:
    """``send_file`` on a directory would be a 500, so reject it here."""
    with pytest.raises(NotFound):
        resolve_within_root(tmp_store, "rgb/0", allowed_roots=ROOTS)


def test_a_missing_chunk_is_404_not_500(tmp_store: Path) -> None:
    """Sparse stores are normal: zarr omits a chunk equal to ``fill_value``."""
    with pytest.raises(NotFound):
        resolve_within_root(tmp_store, "rgb/0/c.9.9.9", allowed_roots=ROOTS)


def test_a_vanished_root_is_404_not_500(tmp_store: Path) -> None:
    """A promote mid-request renames the whole store directory.

    That is the routine path -- it is the event the generation token exists
    to handle -- so it must not surface as an unhandled exception.
    """
    missing = tmp_store.parent / "gone.ome.zarr"
    with pytest.raises(NotFound):
        resolve_within_root(missing, "rgb/0/c.0.0.0", allowed_roots=ROOTS)
