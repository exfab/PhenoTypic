"""valid_staged_store mirrors valid_staged_hdf case for case."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from phenotypic.sdk_ import ngff_


def _write_store(
    root: Path,
    *,
    shapes: dict[str, tuple[int, ...]],
    series: list[str],
    with_root: bool = True,
    store_schema_version: int | None = 3,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    primary = ngff_.primary_series(series) if series else "gray"
    members = {name: name for name in series}
    labels = {"objmap": ngff_.objmap_path(primary)}
    for name, path in [*members.items(), *labels.items()]:
        if name not in shapes:
            continue
        array = np.zeros(shapes[name], dtype=np.uint16)
        if 0 in array.shape:
            # A zero-extent array is legal in Zarr v3, but it cannot be built
            # through `array_create_kwargs`: `chunk_shape_for` clamps the chunk
            # to the level extent, so the chunk is 0 too, and the sharding
            # codec's divisibility check then evaluates `shard % chunk` with a
            # zero divisor (ZeroDivisionError, zarr/codecs/sharding.py:318).
            # Sharding is dropped for this fixture only; what lands on disk is
            # exactly the degenerate store `valid_staged_store` must reject.
            zarr.create_array(
                store=str(root / path / "0"),
                shape=array.shape,
                dtype=array.dtype,
                chunks=tuple(max(1, extent) for extent in array.shape),
                dimension_names=list(ngff_.axes_for(name)),
            )
            continue
        zarr.create_array(
            store=str(root / path / "0"),
            **ngff_.array_create_kwargs(array.shape, array.dtype, name),
        )
    if with_root:
        block = {
            ngff_.PhenotypicAttr.SERIES: members,
            ngff_.PhenotypicAttr.LABELS: labels,
        }
        if store_schema_version is not None:
            block[ngff_.PhenotypicAttr.STORE_SCHEMA_VERSION] = store_schema_version
        (root / "zarr.json").write_text(
            json.dumps(
                {
                    "zarr_format": 3,
                    "node_type": "group",
                    "attributes": {"ome": {"version": "0.5"}, "phenotypic": block},
                }
            ),
            encoding="utf-8",
        )
    return root


def test_complete_store_is_valid(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (64, 48), "detect_mat": (64, 48), "objmap": (64, 48)},
        series=["gray", "detect_mat"],
    )
    assert ngff_.valid_staged_store(store) is True


def test_missing_store_is_invalid(tmp_path: Path) -> None:
    assert ngff_.valid_staged_store(tmp_path / "absent.ome.zarr") is False


def test_missing_root_zarr_json_is_invalid(tmp_path: Path) -> None:
    """Interrupted after chunks, before the root: reads as absent, by design."""
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (64, 48), "detect_mat": (64, 48), "objmap": (64, 48)},
        series=["gray", "detect_mat"],
        with_root=False,
    )
    assert ngff_.valid_staged_store(store) is False


def test_root_without_store_schema_version_is_invalid(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (64, 48), "detect_mat": (64, 48), "objmap": (64, 48)},
        series=["gray", "detect_mat"],
        store_schema_version=None,
    )
    assert ngff_.valid_staged_store(store) is False


def test_a_future_store_schema_version_is_invalid(tmp_path: Path) -> None:
    """User ruling 2026-08-19: the gate is by VALUE, not presence.

    Kills the `presence-only` mutant. A store written by a future v4 must not
    be read under v3 semantics -- and a `not in block` check accepts it
    silently, which is the whole reason the ruling was made. This is the only
    test that distinguishes the two; `..._without_store_schema_version_...`
    above passes under either implementation.
    """
    store = _write_store(
        tmp_path / "future.ome.zarr",
        shapes={"gray": (64, 48), "detect_mat": (64, 48), "objmap": (64, 48)},
        series=["gray", "detect_mat"],
        store_schema_version=ngff_.STORE_SCHEMA_VERSION + 1,
    )
    assert ngff_.valid_staged_store(store) is False


def test_the_current_store_schema_version_is_valid(tmp_path: Path) -> None:
    """The companion: proves the gate DISCRIMINATES rather than always failing."""
    store = _write_store(
        tmp_path / "current.ome.zarr",
        shapes={"gray": (64, 48), "detect_mat": (64, 48), "objmap": (64, 48)},
        series=["gray", "detect_mat"],
        store_schema_version=ngff_.STORE_SCHEMA_VERSION,
    )
    assert ngff_.valid_staged_store(store) is True


def test_missing_objmap_is_invalid(tmp_path: Path) -> None:
    """Stage 1 writes a zeros objmap, so its absence means an incomplete write."""
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (64, 48), "detect_mat": (64, 48)},
        series=["gray", "detect_mat"],
    )
    assert ngff_.valid_staged_store(store) is False


def test_missing_detect_mat_is_invalid(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (64, 48), "objmap": (64, 48)},
        series=["gray", "detect_mat"],
    )
    assert ngff_.valid_staged_store(store) is False


def test_disagreeing_extents_are_invalid(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (64, 48), "detect_mat": (64, 47), "objmap": (64, 48)},
        series=["gray", "detect_mat"],
    )
    assert ngff_.valid_staged_store(store) is False


def test_zero_extent_is_invalid(tmp_path: Path) -> None:
    """A zero-size Zarr array is legal; it must not pass validity."""
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (0, 48), "detect_mat": (0, 48), "objmap": (0, 48)},
        series=["gray", "detect_mat"],
    )
    assert ngff_.valid_staged_store(store) is False


def test_rgb_store_attaches_labels_under_rgb(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={
            "rgb": (3, 64, 48),
            "gray": (64, 48),
            "detect_mat": (64, 48),
            "objmap": (64, 48),
        },
        series=["rgb", "gray", "detect_mat"],
    )
    assert ngff_.valid_staged_store(store) is True
    block = ngff_.read_phenotypic_attributes(store)
    assert block[ngff_.PhenotypicAttr.LABELS]["objmap"] == "rgb/labels/objmap"


def test_corrupt_root_json_is_invalid_not_raising(tmp_path: Path) -> None:
    store = tmp_path / "a.ome.zarr"
    store.mkdir()
    (store / "zarr.json").write_text("{not json", encoding="utf-8")
    assert ngff_.valid_staged_store(store) is False


def test_a_file_where_a_store_should_be_is_invalid(tmp_path: Path) -> None:
    path = tmp_path / "a.ome.zarr"
    path.write_bytes(b"not a directory")
    assert ngff_.valid_staged_store(path) is False


def test_a_malformed_array_metadata_is_invalid(tmp_path: Path) -> None:
    """A reachable zarr error, not a monkeypatched one.

    Replaces an earlier `test_zarr_errors_are_caught_not_propagated`, which
    could not fail: `BaseZarrError` subclasses `ValueError`, so the assertion
    held with or without it in the tuple.
    """
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (64, 48), "detect_mat": (64, 48), "objmap": (64, 48)},
        series=["gray", "detect_mat"],
    )
    (store / "gray" / "0" / "zarr.json").write_text(
        '{"zarr_format": 3, "node_type": "array"}', encoding="utf-8"
    )
    assert ngff_.valid_staged_store(store) is False


def test_a_label_less_store_does_not_raise(tmp_path: Path) -> None:
    """The `labels` key is OMITTED for a label-less store (Task 1.3, ledger C3).

    `valid_staged_store` is a validity predicate: resume classification and
    migration both call it to decide what to do next, so a KeyError escaping
    here is a crash in production, not a rejected store. Indexing the key
    instead of `.get`-ing it is the mutation this pins.
    """
    store = tmp_path / "a.ome.zarr"
    store.mkdir()
    array = np.zeros((64, 48), dtype=np.uint16)
    zarr.create_array(
        store=str(store / "gray" / "0"),
        **ngff_.array_create_kwargs(array.shape, array.dtype, "gray"),
    )
    (store / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "ome": {"version": "0.5"},
                    "phenotypic": {
                        ngff_.PhenotypicAttr.STORE_SCHEMA_VERSION: 3,
                        ngff_.PhenotypicAttr.SERIES: {"gray": "gray"},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    assert ngff_.valid_staged_store(store) is True


def test_an_empty_member_list_is_invalid(tmp_path: Path) -> None:
    """A root that parses but names no arrays is not a store Stage 2 can use."""
    store = tmp_path / "a.ome.zarr"
    store.mkdir()
    (store / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {
                    "phenotypic": {
                        ngff_.PhenotypicAttr.STORE_SCHEMA_VERSION: 3,
                        ngff_.PhenotypicAttr.SERIES: {},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    assert ngff_.valid_staged_store(store) is False


def test_store_level0_shape_is_none_for_an_absent_member(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "a.ome.zarr",
        shapes={"gray": (64, 48), "detect_mat": (64, 48), "objmap": (64, 48)},
        series=["gray", "detect_mat"],
    )
    assert ngff_.store_level0_shape(store, "gray") == (64, 48)
    assert ngff_.store_level0_shape(store, "rgb") is None


@pytest.mark.parametrize("member", ["gray", "detect_mat"])
def test_every_named_member_is_checked_not_just_the_first(
    tmp_path: Path, member: str
) -> None:
    """Guards a `members[:1]` shortcut: dropping ANY member must invalidate."""
    shapes = {"gray": (64, 48), "detect_mat": (64, 48), "objmap": (64, 48)}
    del shapes[member]
    store = _write_store(
        tmp_path / "a.ome.zarr", shapes=shapes, series=["gray", "detect_mat"]
    )
    assert ngff_.valid_staged_store(store) is False


@pytest.mark.parametrize(
    ("block", "why"),
    [
        (
            {
                ngff_.PhenotypicAttr.STORE_SCHEMA_VERSION: 3,
                ngff_.PhenotypicAttr.SERIES: ["gray"],
            },
            "series as a list rather than a mapping",
        ),
        (
            {
                ngff_.PhenotypicAttr.STORE_SCHEMA_VERSION: 3,
                ngff_.PhenotypicAttr.SERIES: {"gray": "gray"},
                ngff_.PhenotypicAttr.LABELS: ["objmap"],
            },
            "labels as a list rather than a mapping",
        ),
        (["not", "a", "mapping"], "the whole phenotypic block as a list"),
    ],
)
def test_a_non_mapping_phenotypic_block_is_invalid_not_a_crash(
    tmp_path: Path, block: object, why: str
) -> None:
    """A REAL defect, not a test gap (D1).

    The docstring insists this predicate "must RETURN FALSE on a store it does
    not accept, never raise", but the except tuple was `(OSError, KeyError,
    TypeError, ValueError)` -- and `["gray"].values()` / `["objmap"].values()` /
    `[...].get(...)` all raise `AttributeError`, which escaped. Resume
    classification and migration both call this to decide what to do next, so a
    store written by another tool or a future schema crashed production rather
    than being rejected. The realistic input is the first case.
    """
    store = tmp_path / "a.ome.zarr"
    store.mkdir()
    array = np.zeros((64, 48), dtype=np.uint16)
    zarr.create_array(
        store=str(store / "gray" / "0"),
        **ngff_.array_create_kwargs(array.shape, array.dtype, "gray"),
    )
    (store / "zarr.json").write_text(
        json.dumps(
            {
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {"ome": {"version": "0.5"}, "phenotypic": block},
            }
        ),
        encoding="utf-8",
    )
    assert ngff_.valid_staged_store(store) is False, why
