"""The imread projection rule: explicit, ordered, and it refuses."""

from __future__ import annotations

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


def test_resolver_falls_back_to_group_zero_without_a_series_list(
    tmp_path: Path,
) -> None:
    """NGFF 2.2.3: no series attribute means consecutively numbered groups."""
    store = _write_store(
        tmp_path / "s.ome.zarr",
        series={"0": ((8, 6), _axes("y", "x"))},
    )
    assert ngff_.read_ngff_image_spec(store).series == "0"


def test_resolver_refuses_an_hcs_plate(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "p.ome.zarr",
        series={"A": ((8, 6), _axes("y", "x"))},
        extra_root_ome={"plate": {"name": "plate1", "wells": []}},
    )
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
