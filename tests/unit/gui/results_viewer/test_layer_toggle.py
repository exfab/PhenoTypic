"""Unit tests for the colony-view pixel-layer toggle (Batch B2, Task 8).

Pins three contracts:

- :func:`build_layer_toggle` is gated on ``output_root.has_results`` — hidden
  (returns ``None``) for a standalone deliverables bundle, shown for a full
  CLI run with per-image ``results/`` HDFs.
- :func:`_normalize_layer_value` (the toggle→store callback body, extracted to
  a module-level helper so the wiring is unit-testable) coerces any value to a
  valid ``LayerName``, defaulting to ``"rgb"``.
- :func:`_colony_crop_url` (the store→URL leg) threads the active layer onto
  each tile crop ``<img src>`` as the ``&layer=`` query-param the crop route
  consumes.
"""

from __future__ import annotations

from typing import Iterator

from phenotypic.gui.results_viewer import _ids as ids
from phenotypic.gui.results_viewer.colony_view._callbacks import (
    _normalize_layer_value,
)
from phenotypic.gui.results_viewer.colony_view._grid import _colony_crop_url
from phenotypic.gui.results_viewer.colony_view._layout import build_layer_toggle


class _FakeRoot:
    """Minimal stand-in exposing only the ``has_results`` flag the toggle reads."""

    def __init__(self, has_results: bool) -> None:
        self.has_results = has_results


def _walk(component: object) -> Iterator[object]:
    """Yield ``component`` and every descendant component, depth-first."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk(child)
    else:
        yield from _walk(children)


# ---------------------------------------------------------------------------
# build_layer_toggle — has_results gate
# ---------------------------------------------------------------------------


def test_layer_toggle_hidden_in_standalone() -> None:
    """A standalone bundle (no ``results/``) hides the toggle entirely."""
    assert build_layer_toggle(_FakeRoot(has_results=False)) is None


def test_layer_toggle_shown_in_full_run() -> None:
    """A full run renders the segmented control defaulting to ``rgb``."""
    component = build_layer_toggle(_FakeRoot(has_results=True))
    assert component is not None

    # The control carries the LAYER_TOGGLE id, defaults to rgb, and offers
    # exactly the three displayable layers.
    radios = [
        node
        for node in _walk(component)
        if getattr(node, "id", None) == ids.LAYER_TOGGLE
    ]
    assert len(radios) == 1
    toggle = radios[0]
    assert toggle.value == "rgb"
    assert {opt["value"] for opt in toggle.options} == {
        "rgb",
        "detect_mat",
        "objmap",
    }


# ---------------------------------------------------------------------------
# _normalize_layer_value — toggle→store callback body
# ---------------------------------------------------------------------------


def test_normalize_layer_value_passes_through_valid_layers() -> None:
    """Each valid ``LayerName`` member round-trips unchanged."""
    for layer in ("rgb", "detect_mat", "objmap"):
        assert _normalize_layer_value(layer) == layer


def test_normalize_layer_value_defaults_invalid_to_rgb() -> None:
    """``None`` / unknown values collapse to the safe ``rgb`` default."""
    assert _normalize_layer_value(None) == "rgb"
    assert _normalize_layer_value("gray") == "rgb"
    assert _normalize_layer_value(42) == "rgb"


# ---------------------------------------------------------------------------
# _colony_crop_url — store→URL leg
# ---------------------------------------------------------------------------


def test_colony_crop_url_threads_layer_query_param() -> None:
    """The crop URL carries the selected layer as ``&layer=<layer>``."""
    url = _colony_crop_url("d1", "img-1", 7, 64, dim_alpha=0.0, layer="objmap")
    assert "?size=64" in url
    assert "&layer=objmap" in url


def test_colony_crop_url_defaults_to_rgb_layer() -> None:
    """Omitting ``layer`` defaults the URL to the finished ``rgb`` plate."""
    url = _colony_crop_url("d1", "img-1", 7, 64)
    assert "&layer=rgb" in url
