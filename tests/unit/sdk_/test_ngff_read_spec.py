"""The imread projection rule: explicit, ordered, and it refuses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from phenotypic.sdk_ import ngff_


def _axes(*names: str) -> list[dict[str, str]]:
    kind = {"t": "time", "c": "channel", "z": "space", "y": "space", "x": "space"}
    return [{"name": n, "type": kind[n]} for n in names]


# --- the pure projector -----------------------------------------------------


def test_2d_passes_through_unprojected() -> None:
    index, is_rgb = ngff_.project_ngff_axes(_axes("y", "x"), (40, 30))
    assert index == (slice(None), slice(None))
    assert is_rgb is False


def test_three_channels_are_rgb() -> None:
    index, is_rgb = ngff_.project_ngff_axes(_axes("c", "y", "x"), (3, 40, 30))
    assert is_rgb is True
    assert index == (slice(None), slice(None), slice(None))


def test_singleton_axes_are_squeezed() -> None:
    index, is_rgb = ngff_.project_ngff_axes(
        _axes("t", "c", "z", "y", "x"), (1, 3, 1, 40, 30)
    )
    assert index == (0, slice(None), 0, slice(None), slice(None))
    assert is_rgb is True


def test_single_channel_squeezes_to_2d() -> None:
    index, is_rgb = ngff_.project_ngff_axes(_axes("c", "y", "x"), (1, 40, 30))
    assert index == (0, slice(None), slice(None))
    assert is_rgb is False


def test_a_real_time_axis_is_refused() -> None:
    """The message names the axis TYPE, not just whatever the store called it.

    A store may name its axes anything -- `_pick` is handed both, and the type
    is the half a reader can act on. An earlier draft asserted `match="time"`
    against a message that formatted only the name, `'t'`, and would have
    passed only by accident on a store that happened to use that letter.
    """
    with pytest.raises(ValueError, match="time axis 't'"):
        ngff_.project_ngff_axes(_axes("t", "y", "x"), (10, 40, 30))


def test_the_refusal_names_the_override_that_would_read_it() -> None:
    with pytest.raises(ValueError, match=r"t=<index>"):
        ngff_.project_ngff_axes(_axes("t", "y", "x"), (10, 40, 30))


def test_an_oddly_named_time_axis_is_still_named_by_type() -> None:
    """NGFF constrains `type`, not `name`. The type is what we can rely on."""
    axes = [
        {"name": "frame", "type": "time"},
        {"name": "row", "type": "space"},
        {"name": "col", "type": "space"},
    ]
    with pytest.raises(ValueError, match="time axis 'frame'"):
        ngff_.project_ngff_axes(axes, (10, 40, 30))


def test_a_real_time_axis_is_readable_with_an_explicit_index() -> None:
    index, _ = ngff_.project_ngff_axes(_axes("t", "y", "x"), (10, 40, 30), t=4)
    assert index == (4, slice(None), slice(None))


def test_a_real_z_axis_is_refused() -> None:
    with pytest.raises(ValueError, match="space axis 'z'"):
        ngff_.project_ngff_axes(_axes("z", "y", "x"), (12, 40, 30))


def test_five_channels_are_refused() -> None:
    with pytest.raises(ValueError, match="channel axis"):
        ngff_.project_ngff_axes(_axes("c", "y", "x"), (5, 40, 30))


def test_five_channels_are_readable_with_an_explicit_index() -> None:
    index, is_rgb = ngff_.project_ngff_axes(_axes("c", "y", "x"), (5, 40, 30), c=2)
    assert index == (2, slice(None), slice(None))
    assert is_rgb is False


def test_an_explicit_c_overrides_a_three_channel_store() -> None:
    """`c=` means "this one channel", even where RGB was available.

    The override is the caller saying they know better; silently returning RGB
    because the count happened to be 3 would ignore an explicit instruction.
    """
    index, is_rgb = ngff_.project_ngff_axes(_axes("c", "y", "x"), (3, 40, 30), c=0)
    assert index == (0, slice(None), slice(None))
    assert is_rgb is False


def test_two_channels_are_refused_rather_than_guessed() -> None:
    """2 is neither a grayscale nor an RGB triple. Refuse."""
    with pytest.raises(ValueError, match="channel axis"):
        ngff_.project_ngff_axes(_axes("c", "y", "x"), (2, 40, 30))


def test_an_out_of_range_override_is_refused() -> None:
    with pytest.raises(ValueError, match="out of range"):
        ngff_.project_ngff_axes(_axes("t", "y", "x"), (10, 40, 30), t=99)


def test_axes_and_shape_must_agree_in_length() -> None:
    with pytest.raises(ValueError, match="axes/shape mismatch"):
        ngff_.project_ngff_axes(_axes("y", "x"), (3, 40, 30))


# --- the store resolver -----------------------------------------------------


def _write_store(
    root: Path,
    *,
    series: dict[str, tuple[tuple[int, ...], list[dict[str, str]]]],
    series_list: list[str] | None = None,
    phenotypic: dict | None = None,
    extra_root_ome: dict | None = None,
) -> Path:
    """Build a minimal but conformant multi-series NGFF store."""
    group = zarr.create_group(store=str(root), zarr_format=3)
    root_ome: dict = {"version": "0.5", "bioformats2raw.layout": 3}
    root_ome.update(extra_root_ome or {})
    group.attrs["ome"] = root_ome
    if phenotypic is not None:
        group.attrs["phenotypic"] = phenotypic

    if series_list is not None:
        ome_group = group.create_group("OME")
        ome_group.attrs["ome"] = {"version": "0.5", "series": series_list}

    rng = np.random.default_rng(0)
    for name, (shape, axes) in series.items():
        sub = group.create_group(name)
        arr = sub.create_array(
            "0",
            shape=shape,
            chunks=shape,
            dtype="uint16",
            dimension_names=[a["name"] for a in axes],
        )
        arr[:] = rng.integers(1, 4096, size=shape, dtype=np.uint16)
        sub.attrs["ome"] = {
            "version": "0.5",
            "multiscales": [
                {
                    "name": name,
                    "axes": axes,
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": [1.0] * len(shape)}
                            ],
                        }
                    ],
                }
            ],
        }
    return root


def test_resolver_reads_the_first_declared_series(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "s.ome.zarr",
        series={
            "rgb": ((3, 8, 6), _axes("c", "y", "x")),
            "gray": ((8, 6), _axes("y", "x")),
        },
        series_list=["rgb", "gray"],
    )
    spec = ngff_.read_ngff_image_spec(store)
    assert spec.series == "rgb"
    assert spec.array.shape == (8, 6, 3)  # transposed to HWC
    assert spec.bit_depth == 16  # inferred from uint16


def test_resolver_honours_an_explicit_series(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "s.ome.zarr",
        series={
            "rgb": ((3, 8, 6), _axes("c", "y", "x")),
            "gray": ((8, 6), _axes("y", "x")),
        },
        series_list=["rgb", "gray"],
    )
    spec = ngff_.read_ngff_image_spec(store, series="gray")
    assert spec.series == "gray"
    assert spec.array.shape == (8, 6)


def _hostile_series_path(
    tmp_path: Path, store: Path, attack: str
) -> str:
    outside = _write_store(
        tmp_path / "outside.ome.zarr",
        series={"gray": ((8, 6), _axes("y", "x"))},
        series_list=["gray"],
    )
    if attack == "absolute":
        return (outside / "gray").as_posix()
    if attack == "traversal":
        return "../outside.ome.zarr/gray"
    link = store / "escape"
    link.symlink_to(outside / "gray", target_is_directory=True)
    return "escape"


@pytest.mark.parametrize("attack", ["absolute", "traversal", "symlink"])
def test_explicit_series_cannot_escape_store_boundary(
    tmp_path: Path, attack: str
) -> None:
    store = _write_store(
        tmp_path / "s.ome.zarr",
        series={"gray": ((8, 6), _axes("y", "x"))},
        series_list=["gray"],
    )
    hostile = _hostile_series_path(tmp_path, store, attack)

    with pytest.raises(ValueError, match="series path"):
        ngff_.read_ngff_image_spec(store, series=hostile)


@pytest.mark.parametrize("attack", ["absolute", "traversal", "symlink"])
def test_declared_series_cannot_escape_store_boundary(
    tmp_path: Path, attack: str
) -> None:
    store = _write_store(
        tmp_path / "s.ome.zarr",
        series={"gray": ((8, 6), _axes("y", "x"))},
        series_list=["gray"],
    )
    hostile = _hostile_series_path(tmp_path, store, attack)
    ome_json = store / "OME" / "zarr.json"
    payload = json.loads(ome_json.read_text(encoding="utf-8"))
    payload["attributes"]["ome"]["series"] = [hostile]
    ome_json.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="series path"):
        ngff_.read_ngff_image_spec(store)


def test_explicit_series_rejects_backslashes(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "s.ome.zarr",
        series={r"bad\series": ((8, 6), _axes("y", "x"))},
    )

    with pytest.raises(ValueError, match="series path"):
        ngff_.read_ngff_image_spec(store, series=r"bad\series")


def test_explicit_series_rejects_a_dangling_symlink(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "s.ome.zarr",
        series={"gray": ((8, 6), _axes("y", "x"))},
    )
    (store / "dangling").symlink_to(
        tmp_path / "missing-series", target_is_directory=True
    )

    with pytest.raises(ValueError, match="series path"):
        ngff_.read_ngff_image_spec(store, series="dangling")


def _set_first_dataset_path(store: Path, series: str, path: str) -> None:
    group_json = store / series / "zarr.json"
    payload = json.loads(group_json.read_text(encoding="utf-8"))
    payload["attributes"]["ome"]["multiscales"][0]["datasets"][0][
        "path"
    ] = path
    group_json.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("attack", ["absolute", "traversal", "symlink"])
def test_dataset_path_cannot_escape_selected_series_boundary(
    tmp_path: Path, attack: str
) -> None:
    store = _write_store(
        tmp_path / "s.ome.zarr",
        series={"gray": ((8, 6), _axes("y", "x"))},
        series_list=["gray"],
    )
    outside = zarr.create_array(
        store=str(tmp_path / "outside-array"),
        shape=(8, 6),
        chunks=(8, 6),
        dtype="uint16",
        zarr_format=3,
        dimension_names=["y", "x"],
    )
    outside[:] = np.ones((8, 6), dtype=np.uint16)
    if attack == "absolute":
        hostile = (tmp_path / "outside-array").as_posix()
    elif attack == "traversal":
        hostile = "../../outside-array"
    else:
        (store / "gray" / "escape").symlink_to(
            tmp_path / "outside-array", target_is_directory=True
        )
        hostile = "escape"
    _set_first_dataset_path(store, "gray", hostile)

    with pytest.raises(ValueError, match="dataset path"):
        ngff_.read_ngff_image_spec(store)


def test_resolver_falls_back_to_group_zero_without_a_series_list(
    tmp_path: Path,
) -> None:
    """NGFF 2.2.3: no series attribute means consecutively numbered groups."""
    store = _write_store(
        tmp_path / "s.ome.zarr",
        series={"0": ((8, 6), _axes("y", "x"))},
    )
    assert ngff_.read_ngff_image_spec(store).series == "0"


def test_group_zero_fallback_requires_bioformats_layout_marker(
    tmp_path: Path,
) -> None:
    store = _write_store(
        tmp_path / "s.ome.zarr",
        series={"0": ((8, 6), _axes("y", "x"))},
    )
    root_json = store / "zarr.json"
    payload = json.loads(root_json.read_text(encoding="utf-8"))
    del payload["attributes"]["ome"]["bioformats2raw.layout"]
    root_json.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="bioformats2raw.layout"):
        ngff_.read_ngff_image_spec(store)


def test_resolver_refuses_an_hcs_plate(tmp_path: Path) -> None:
    """The plate check runs BEFORE the declared-series list, and the fixture
    has to carry both to pin that.

    A `bioformats2raw` plate carries a root `ome.plate` AND an `OME/zarr.json`
    series list of its well fields. Built with `series_list=None`, this test
    passed under either ordering -- and under the series-first ordering the
    real store returns one well field instead of refusing, contradicting the
    spec's own "HCS plate -> raises" row.
    """
    store = _write_store(
        tmp_path / "p.ome.zarr",
        series={"A/1/0": ((8, 6), _axes("y", "x"))},
        series_list=["A/1/0"],
        extra_root_ome={
            "plate": {
                "name": "plate1",
                "rows": [{"name": "A"}],
                "columns": [{"name": "1"}],
                "wells": [{"path": "A/1", "rowIndex": 0, "columnIndex": 0}],
            }
        },
    )
    # The series-first ordering would find this and read it as one image.
    assert (store / "A" / "1" / "0" / "zarr.json").is_file()
    with pytest.raises(ValueError, match="plate"):
        ngff_.read_ngff_image_spec(store)


def test_resolver_reads_a_store_with_no_phenotypic_block(tmp_path: Path) -> None:
    """Case C. require_readable_store must never be reached from here."""
    store = _write_store(
        tmp_path / "foreign.ome.zarr",
        series={"0": ((8, 6), _axes("y", "x"))},
    )
    spec = ngff_.read_ngff_image_spec(store)
    assert spec.phenotypic == {}
    assert spec.array.shape == (8, 6)


def test_resolver_reads_a_future_store_version(tmp_path: Path) -> None:
    """A newer store's NGFF geometry is still NGFF (spec 4.6)."""
    store = _write_store(
        tmp_path / "future.ome.zarr",
        series={"0": ((8, 6), _axes("y", "x"))},
        phenotypic={"store_schema_version": 999},
    )
    assert ngff_.read_ngff_image_spec(store).array.shape == (8, 6)


def test_resolver_prefers_stored_bit_depth_over_dtype(tmp_path: Path) -> None:
    """From metadata.protected -- `phenotypic.bit_depth` is not a real key.

    No writer emits `phenotypic.bit_depth`: `build_phenotypic_attributes`
    (ngff_.py:540-586) emits store_schema_version, phenotypic_version,
    image_class, series, pyramid, detect_mode, illuminant, gamma, metadata,
    and the optional provenance/labels/work_id/grid -- nothing else. Bit depth
    lives in metadata.protected[Metadata_BitDepth], which is where
    `_load_from_store` reads it (_image_io_handler.py:1406). An earlier draft
    read the non-existent key, which would have silently dropped bit depth on
    every float round trip -- the one case dtype inference cannot rescue.
    """
    from phenotypic.schema import IMAGE

    store = _write_store(
        tmp_path / "s.ome.zarr",
        series={"0": ((8, 6), _axes("y", "x"))},
        phenotypic={
            "store_schema_version": 3,
            "metadata": {"protected": {IMAGE.BIT_DEPTH: 12}},
        },
    )
    assert ngff_.read_ngff_image_spec(store).bit_depth == 12


def test_resolver_infers_bit_depth_from_dtype_when_unstored(
    tmp_path: Path,
) -> None:
    """Case C: a third-party store has no protected section at all."""
    store = _write_store(
        tmp_path / "foreign.ome.zarr",
        series={"0": ((8, 6), _axes("y", "x"))},
    )
    assert ngff_.read_ngff_image_spec(store).bit_depth == 16  # uint16


def test_resolver_refuses_a_non_image_directory(tmp_path: Path) -> None:
    empty = tmp_path / "nothing.ome.zarr"
    empty.mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        ngff_.read_ngff_image_spec(empty)


# --- overrides on a size-1 axis (the range check precedes the shortcut) -----


def test_an_out_of_range_c_is_refused_on_a_size_one_channel_axis() -> None:
    """`if size == 1: return 0` used to precede the range check, so `c=7` on a
    1-channel store silently read channel 0 -- contradicting the projector's
    own rule that an explicit override is an instruction, not a hint."""
    with pytest.raises(ValueError, match="out of range"):
        ngff_.project_ngff_axes(_axes("c", "y", "x"), (1, 8, 6), c=7)


def test_an_out_of_range_t_is_refused_on_a_size_one_time_axis() -> None:
    with pytest.raises(ValueError, match="out of range"):
        ngff_.project_ngff_axes(_axes("t", "y", "x"), (1, 8, 6), t=3)


def test_an_out_of_range_z_is_refused_on_a_size_one_space_axis() -> None:
    with pytest.raises(ValueError, match="out of range"):
        ngff_.project_ngff_axes(_axes("z", "y", "x"), (1, 8, 6), z=3)


def test_an_in_range_override_still_squeezes_a_size_one_axis() -> None:
    """The range check must not break the squeeze it now precedes."""
    index, _ = ngff_.project_ngff_axes(_axes("t", "y", "x"), (1, 8, 6), t=0)
    assert index == (0, slice(None), slice(None))


# --- axis-shape guards ------------------------------------------------------


def test_too_few_space_axes_are_refused() -> None:
    """The docstring calls this a TOTAL mapping; that holds only for NGFF's
    own 2-or-3 space axes."""
    axes = [{"name": "c", "type": "channel"}, {"name": "y", "type": "space"}]
    with pytest.raises(ValueError, match="space axes"):
        ngff_.project_ngff_axes(axes, (1, 8))


def test_too_many_space_axes_are_refused() -> None:
    axes = [{"name": n, "type": "space"} for n in ("w", "z", "y", "x")]
    with pytest.raises(ValueError, match="space axes"):
        ngff_.project_ngff_axes(axes, (2, 2, 8, 6))


def test_an_untyped_axis_gets_its_own_message() -> None:
    """Not `_pick`'s, which would advertise a flag named `(no override)`."""
    axes = [
        {"name": "phase", "type": "wavelength"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]
    with pytest.raises(ValueError) as excinfo:
        ngff_.project_ngff_axes(axes, (4, 8, 6))
    message = str(excinfo.value)
    assert "'phase'" in message
    assert "wavelength" in message
    assert "no override" not in message


def test_a_size_one_untyped_axis_is_still_squeezed() -> None:
    axes = [
        {"name": "phase", "type": "wavelength"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]
    index, _ = ngff_.project_ngff_axes(axes, (1, 8, 6))
    assert index == (0, slice(None), slice(None))


# --- a full 5-D store, read through the store path (spec 9) -----------------


def _write_5d_store(root: Path, *, shape: tuple[int, ...]) -> Path:
    """A single-series `tczyx` store -- `_write_store` with the axes named.

    The projection assertions read the raw array back off disk rather than
    trusting a seed, so nothing here depends on which values were written.
    """
    return _write_store(
        root,
        series={"0": (shape, _axes("t", "c", "z", "y", "x"))},
        series_list=["0"],
    )


def test_a_5d_tczyx_store_reads_the_named_timepoint_and_plane(
    tmp_path: Path,
) -> None:
    """Spec 9 lists three 5-D rows; none was covered end to end through a store.

    Shape alone would pass for the wrong `t` -- so the values are compared
    against the raw array, projected by hand.
    """
    store = _write_5d_store(tmp_path / "movie.ome.zarr", shape=(4, 3, 5, 8, 6))
    raw = np.asarray(zarr.open_array(store=str(store / "0" / "0"), mode="r")[:])

    spec = ngff_.read_ngff_image_spec(store, t=2, z=3)
    assert spec.array.shape == (8, 6, 3)
    assert np.array_equal(spec.array, np.moveaxis(raw[2, :, 3, :, :], 0, -1))

    # A different timepoint really is different data, so the index is read.
    other = ngff_.read_ngff_image_spec(store, t=0, z=3)
    assert not np.array_equal(other.array, spec.array)


def test_a_5d_store_refuses_without_the_indices(tmp_path: Path) -> None:
    store = _write_5d_store(tmp_path / "movie.ome.zarr", shape=(4, 3, 5, 8, 6))
    with pytest.raises(ValueError, match="time axis"):
        ngff_.read_ngff_image_spec(store)


def test_a_5d_store_with_one_odd_channel_needs_c(tmp_path: Path) -> None:
    store = _write_5d_store(tmp_path / "movie.ome.zarr", shape=(1, 5, 1, 8, 6))
    with pytest.raises(ValueError, match="channel axis"):
        ngff_.read_ngff_image_spec(store)
    spec = ngff_.read_ngff_image_spec(store, c=4)
    assert spec.array.shape == (8, 6)


# --- spec 4.1 step 2: the root group is itself the image --------------------


def _write_root_multiscale_store(root: Path, *, levels: int = 1) -> Path:
    """No OME/ series list and no group '0' -- multiscales sits at the root."""
    group = zarr.create_group(store=str(root), zarr_format=3)
    axes = _axes("y", "x")
    rng = np.random.default_rng(11)
    datasets = []
    shape = (16, 12)
    for level in range(levels):
        level_shape = (shape[0] >> level, shape[1] >> level)
        arr = group.create_array(
            str(level),
            shape=level_shape,
            chunks=level_shape,
            dtype="uint16",
            dimension_names=["y", "x"],
        )
        arr[:] = rng.integers(1, 4096, size=level_shape, dtype=np.uint16)
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [float(1 << level)] * 2}
                ],
            }
        )
    group.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [{"name": "root", "axes": axes, "datasets": datasets}],
    }
    return root


def test_resolver_reads_multiscales_at_the_root(tmp_path: Path) -> None:
    """Spec 4.1 step 2. `_resolve_series_path` returns "" and the group path
    is the store root itself -- the one branch where the join is skipped."""
    store = _write_root_multiscale_store(tmp_path / "flat.ome.zarr")
    spec = ngff_.read_ngff_image_spec(store)
    assert spec.series == ""
    assert spec.array.shape == (16, 12)


# --- pyramid levels ---------------------------------------------------------


def test_an_explicit_level_reads_the_downsampled_array(tmp_path: Path) -> None:
    store = _write_root_multiscale_store(tmp_path / "pyr.ome.zarr", levels=3)
    assert ngff_.read_ngff_image_spec(store, level=0).array.shape == (16, 12)
    assert ngff_.read_ngff_image_spec(store, level=2).array.shape == (4, 3)
    assert ngff_.read_ngff_image_spec(store, level=2).level == 2


def test_an_out_of_range_level_is_refused(tmp_path: Path) -> None:
    store = _write_root_multiscale_store(tmp_path / "pyr.ome.zarr", levels=3)
    with pytest.raises(ValueError, match="level 5 is out of range"):
        ngff_.read_ngff_image_spec(store, level=5)


def test_a_negative_level_is_refused(tmp_path: Path) -> None:
    store = _write_root_multiscale_store(tmp_path / "pyr.ome.zarr", levels=3)
    with pytest.raises(ValueError, match="out of range"):
        ngff_.read_ngff_image_spec(store, level=-1)


# --- an unknown series= is a caller error, not a missing store --------------


def test_an_unknown_series_names_what_the_store_declares(tmp_path: Path) -> None:
    """It raised a bare FileNotFoundError on an internal path, which reads as
    "the store is gone" -- the codebase's signal for an interrupted write."""
    store = _write_store(
        tmp_path / "s.ome.zarr",
        series={
            "rgb": ((3, 8, 6), _axes("c", "y", "x")),
            "gray": ((8, 6), _axes("y", "x")),
        },
        series_list=["rgb", "gray"],
    )
    with pytest.raises(ValueError) as excinfo:
        ngff_.read_ngff_image_spec(store, series="detect_mat")
    message = str(excinfo.value)
    assert "'detect_mat'" in message
    assert "'rgb'" in message and "'gray'" in message


def test_an_unknown_series_on_a_phenotypic_store_lists_its_series(
    tmp_path: Path,
) -> None:
    """With no OME/ series list, the fallback source is `phenotypic.series`."""
    store = _write_store(
        tmp_path / "s.ome.zarr",
        series={"gray": ((8, 6), _axes("y", "x"))},
        phenotypic={"store_schema_version": 3, "series": {"gray": "gray"}},
        extra_root_ome={"multiscales": []},
    )
    with pytest.raises(ValueError, match="'gray'"):
        ngff_.read_ngff_image_spec(store, series="rgb")


# --- NGFF 0.4 / Zarr v2 is refused by name, not by FileNotFoundError --------


def test_a_zarr_v2_store_is_refused_by_name(tmp_path: Path) -> None:
    """`bioformats2raw`'s default output and QuPath's export are 0.4/v2 today
    (spec 3.1 case C). A v2 group has `.zgroup`, not `zarr.json`, so it
    surfaced as the FileNotFoundError that means "interrupted write"."""
    import json

    store = tmp_path / "legacy.ome.zarr"
    store.mkdir()
    (store / ".zgroup").write_text(json.dumps({"zarr_format": 2}), encoding="utf-8")
    (store / ".zattrs").write_text(
        json.dumps({"multiscales": [{"version": "0.4", "datasets": []}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError) as excinfo:
        ngff_.read_ngff_image_spec(store)
    message = str(excinfo.value)
    assert ".zgroup" in message
    assert "v3" in message or "0.5" in message


def test_an_absent_store_is_still_a_file_not_found_error(tmp_path: Path) -> None:
    """The v2 branch must not swallow the interrupted-write signal."""
    store = tmp_path / "half_written.ome.zarr"
    store.mkdir()
    with pytest.raises(FileNotFoundError):
        ngff_.read_ngff_image_spec(store)
