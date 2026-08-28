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
