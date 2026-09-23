"""A zone figure does not depend on whether measure() just ran (spec §4)."""
from __future__ import annotations

import json

import pytest

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureOrientationZones, MeasureSymZones
from phenotypic.plotting._pipeline._store_formats import serialize_store_format


def _detected() -> Image:
    image = Image(load_synth_yeast_plate())
    OtsuDetector().apply(image, inplace=True)
    return image


def _first_difference(left, right, path="$"):
    """The JSON path of the first structural difference, or None."""
    if type(left) is not type(right):
        return path
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            found = _first_difference(left.get(key), right.get(key), f"{path}.{key}")
            if found:
                return found
    elif isinstance(left, list):
        if len(left) != len(right):
            return f"{path}[len]"
        for index, (a, b) in enumerate(zip(left, right)):
            found = _first_difference(a, b, f"{path}[{index}]")
            if found:
                return found
    elif left != right:
        return path
    return None


@pytest.mark.parametrize("measurer_cls", [MeasureSymZones, MeasureOrientationZones])
def test_cache_hit_and_recompute_render_identical_bytes(measurer_cls, tmp_path):
    """Full mode renders from the cache; process/measure mode recomputes.

    Spec §4 makes a mismatch a provider bug, not a tolerance, so the
    assertion is exact and its message names the first differing JSON path.
    """
    image = _detected()
    measurer = measurer_cls()
    measurer.measure(image)
    hit = measurer.inspect(image, for_save=True)

    reloaded = Image.load_zarr(image.save2zarr(tmp_path / "p.ome.zarr"))
    recomputed = measurer_cls().inspect(reloaded, for_save=True)

    def encode(fig):
        return serialize_store_format(
            "plotly-json", fig, binding_id="b", page_key="default"
        )

    left, right = encode(hit), encode(recomputed)
    assert left == right, _first_difference(json.loads(left), json.loads(right))
