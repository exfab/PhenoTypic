"""Phase 3 proof: ``MeasureSymmetricZones.inspect`` routed onto ``@figure``.

Verifies the pydantic operation gains the figure protocol (themed ``inspect``,
a ``base_layer`` select Control, an ipywidgets ``.dash()``, control-driven
recompute) WITHOUT changing the CLI ``inspect`` contract (``for_save`` flatten)
or its pydantic schema / serialization.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from phenotypic import Image
from phenotypic.measure import MeasureSymmetricZones
from phenotypic.measure._measure_symmetric_zones import BASE_LAYER
from phenotypic.sdk_ import CONFIG_SUFFIX_OPERATION, ensure_typed_json_suffix
from phenotypic.sdk_.viz.figures._theme import OKABE_ITO


def _circular_colony_image() -> Image:
    """A single concentric colony (bright core + halo) with a preset objmap."""
    shape = (120, 120)
    cr, cc = 60, 60
    gray = np.full(shape, 220, dtype=np.uint8)
    objmap = np.zeros(shape, dtype=np.int32)
    rr, ccx = np.ogrid[: shape[0], : shape[1]]
    dist_sq = (rr - cr) ** 2 + (ccx - cc) ** 2
    gray[dist_sq < 35**2] = 120
    gray[dist_sq < 18**2] = 40
    objmap[dist_sq < 35**2] = 1
    image = Image(np.stack([gray, gray, gray], axis=-1))
    image.objmap[:] = objmap
    return image


@pytest.fixture
def measured():
    # function-scoped: each test gets a fresh op so the per-instance diagnostic
    # cache (mutated by inspect/measure) never leaks across tests.
    image = _circular_colony_image()
    op = MeasureSymmetricZones()
    op.measure(image)  # populates the diagnostic cache
    return op, image


def test_inspect_returns_themed_figure(measured):
    op, image = measured
    fig = op.inspect(image)
    assert isinstance(fig, go.Figure)
    # @figure applied the house theme → merged template carries the colorway
    assert tuple(fig.layout.template.layout.colorway) == OKABE_ITO


def test_inspect_hides_object_labels(measured):
    op, image = measured
    fig = op.inspect(image)
    annotation_text = {str(ann.text) for ann in fig.layout.annotations}
    object_labels = {str(prop.label) for prop in image.objects.props}
    assert annotation_text.isdisjoint(object_labels)


def test_inspect_for_save_flattens_legendonly(measured):
    op, image = measured
    # non-vacuous: the interactive figure DOES hide overlay layers behind the legend
    interactive = op.inspect(image, for_save=False)
    assert any(getattr(t, "visible", None) == "legendonly" for t in interactive.data)
    # for_save flattens them so the static raster shows every layer
    saved = op.inspect(image, for_save=True)
    assert all(getattr(t, "visible", True) != "legendonly" for t in saved.data)


def test_iter_figures_primary_and_control(measured):
    op, _ = measured
    specs = op.iter_figures()
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "inspect"
    assert spec.primary is True
    assert spec.controls == {"base_layer": BASE_LAYER}
    assert spec.wants_subject is True  # `image` is the subject


def test_dash_returns_ipywidget(measured):
    op, image = measured
    widgets = pytest.importorskip("ipywidgets")
    dashboard = op.dash(image)
    assert isinstance(dashboard, widgets.Widget)


def test_base_layer_drives_recompute(measured):
    op, image = measured
    bound = op.figures(image)
    spec = op.iter_figures()[0]
    f_gray = bound.render(spec, base_layer="gray")
    f_rgb = bound.render(spec, base_layer="rgb")
    f_gray_again = bound.render(spec, base_layer="gray")
    assert f_gray is f_gray_again  # cached for the same control value
    assert f_gray is not f_rgb  # recomputed for a different value


def test_inspect_rejects_invalid_base_layer(measured):
    op, image = measured

    with pytest.raises(ValueError, match="base_layer"):
        op.inspect(image, base_layer="bad")


def test_pydantic_schema_and_serialization_unchanged():
    fields = set(MeasureSymmetricZones.model_fields)
    props = set(MeasureSymmetricZones.model_json_schema()["properties"])
    # the mixin and its Control add nothing pydantic collects
    assert "base_layer" not in fields and "base_layer" not in props
    op = MeasureSymmetricZones()
    again = MeasureSymmetricZones.from_json(op.to_json())
    assert isinstance(again, MeasureSymmetricZones)
    assert again.model_dump() == op.model_dump()


def test_operation_to_json_file_uses_typed_suffix(tmp_path):
    op = MeasureSymmetricZones()
    filepath = tmp_path / "measure_symmetric_zones.json"
    typed_filepath = ensure_typed_json_suffix(filepath, CONFIG_SUFFIX_OPERATION)

    op.to_json(filepath)

    assert not filepath.exists()
    assert typed_filepath.exists()
    again = MeasureSymmetricZones.from_json(typed_filepath)
    assert isinstance(again, MeasureSymmetricZones)
    assert again.model_dump() == op.model_dump()


def test_inspect_for_save_png_export_if_chrome(measured, tmp_path):
    """The kaleido PNG path that ``--save-inspect`` uses (skipped without Chrome)."""
    op, image = measured
    fig = op.inspect(image, for_save=True)
    out = tmp_path / "inspect.png"
    try:
        fig.write_image(str(out), format="png", scale=2)
    except Exception as exc:  # kaleido>=1 needs Chrome; absent on this host
        pytest.skip(f"kaleido PNG export unavailable: {exc}")
    assert out.exists() and out.stat().st_size > 0
