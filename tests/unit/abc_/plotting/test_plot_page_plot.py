"""PlotPage.plot names a page's folder (spec 2026-09-30 §1)."""
from __future__ import annotations

import pytest

from phenotypic.abc_.plotting import PlotOutput, PlotPage


def test_plot_defaults_to_the_key():
    page = PlotPage(key="delta_e", figure=object())
    assert page.plot is None
    assert page.plot_name == "delta_e"


def test_plot_names_the_folder_when_given():
    page = PlotPage(key="roi_0", plot="tiles", figure=object())
    assert (page.plot_name, page.key) == ("tiles", "roi_0")


@pytest.mark.parametrize("plot", ["", 3, "a/b", "/tiles", "tiles/"])
def test_an_empty_non_string_or_slashed_plot_is_refused(plot):
    with pytest.raises(ValueError, match="plot"):
        PlotPage(key="k", plot=plot, figure=object())


def test_the_same_key_in_two_plots_is_allowed():
    output = PlotOutput(pages=(
        PlotPage(key="roi_0", plot="tiles", figure=object()),
        PlotPage(key="roi_0", plot="masks", figure=object()),
    ))
    assert [(p.plot_name, p.key) for p in output.pages] == [("tiles", "roi_0"), ("masks", "roi_0")]


def test_the_same_key_in_one_plot_is_refused():
    with pytest.raises(ValueError, match="duplicate page keys"):
        PlotOutput(pages=(
            PlotPage(key="roi_0", plot="tiles", figure=object()),
            PlotPage(key="roi_0", plot="tiles", figure=object()),
        ))


def test_two_bare_pages_with_one_key_are_still_refused():
    with pytest.raises(ValueError, match="duplicate page keys"):
        PlotOutput(pages=(PlotPage(key="a", figure=object()), PlotPage(key="a", figure=object())))


def test_a_bare_page_and_a_plotted_page_with_one_key_coexist():
    # (plot "a", key "a") and (plot "tiles", key "a") are different pages.
    PlotOutput(pages=(PlotPage(key="a", figure=object()),
                      PlotPage(key="a", plot="tiles", figure=object())))
