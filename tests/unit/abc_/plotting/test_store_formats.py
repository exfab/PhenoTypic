"""@figure(store=...) -- the closed format set, defaults and validation (spec §2)."""
from __future__ import annotations

import pytest

from phenotypic.abc_.plotting import PhtPlot, figure
from phenotypic.abc_.plotting._store_formats import (
    STORE_FORMATS,
    default_store_formats,
)


def test_the_format_table_is_the_closed_set_with_its_media_types():
    assert {n: (i.extension, i.media_type) for n, i in STORE_FORMATS.items()} == {
        "plotly-json": (".plotly.json", "application/vnd.plotly.v1+json"),
        "png": (".png", "image/png"),
    }


@pytest.mark.parametrize(
    ("backend", "expected"),
    [("plotly", ("plotly-json",)), ("mpl", ("png",))],
)
def test_an_omitted_store_takes_the_backend_default(backend, expected):
    class Plot(PhtPlot):
        @figure(title="t", backend=backend, primary=True)
        def draw(self, image):
            raise AssertionError("never rendered")

    assert Plot._class_primary_spec().store == expected
    assert default_store_formats(backend) == expected


def test_a_declared_store_is_kept_in_declared_order():
    class Plot(PhtPlot):
        @figure(title="t", backend="plotly", primary=True, store=("png", "plotly-json"))
        def draw(self, image):
            raise AssertionError

    assert Plot._class_primary_spec().store == ("png", "plotly-json")


@pytest.mark.parametrize(
    ("backend", "store", "match"),
    [
        ("mpl", ("plotly-json",), "cannot produce"),
        ("plotly", ("svg",), "unknown"),
        ("plotly", ("png", "png"), "duplicate"),
        ("plotly", (), "at least one"),
        ("plotly", "png", "tuple"),
    ],
)
def test_an_invalid_store_is_refused_at_class_definition(backend, store, match):
    with pytest.raises(TypeError, match=match):

        class Plot(PhtPlot):  # noqa: F841 - the definition itself must raise
            @figure(title="t", backend=backend, primary=True, store=store)
            def draw(self, image):
                raise AssertionError
