"""Browser-driven E2E tests for the results-viewer Scatter tab.

The tab shipped with nine unit-test modules and no browser coverage, so
the chain a user actually exercises -- click a point, resolve it to a
colony, fetch its crop over an HTTP route -- had never been driven end to
end. Every link in it is server-side and none of it is reachable from a
unit test: ``clickData`` is produced by Plotly, the crop is a Flask
route, and the Style steppers are pattern-matching ids whose callback
only exists on a live app.

**Plotly state is read through ``_fullLayout`` and ``data``, not the
DOM.** The traces are ``Scattergl``, which paints to a WebGL canvas and
mounts no per-point elements -- a selector-based assertion here would
find nothing and could only ever be written to pass. For the same reason
a point is "clicked" by dispatching Plotly's own ``plotly_click`` with
the point's carried ``customdata``, which is what the real click handler
receives.

Tests gated by ``PLAYWRIGHT=1`` via the module-level skip in
``conftest.py``.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Iterator

import polars as pl
import pytest
from playwright.sync_api import Page

from phenotypic.schema import CULTURE, EXPERIMENT, IMAGE, SHAPE
from tests._output_layout import write_master, write_measurements_mirror
from tests.e2e.gui.conftest import (
    _build_sandbox,
    _start_live_server,
    bind_results_output,
    publish_coherent_terminal_evidence,
)

_OUTPUT_NAME = "CliOutputExample"
_IMAGES = ("plate_001.tif", "plate_002.tif")
_DATASET_COLUMN = str(EXPERIMENT.DATASET)
_TIME_COLUMN = str(CULTURE.TIME)
_AREA_COLUMN = str(SHAPE.AREA)
_STRAIN_COLUMN = "Metadata_Strain"
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)

#: Read one Plotly property off the mounted graph. Returned rather than
#: asserted in JS so a failure reports the value, not just ``False``.
_FIGURE_STATE_JS = """
() => {
    const gd = document.querySelector('#scatter-graph .js-plotly-plot');
    if (!gd || !gd._fullLayout) return null;
    return {
        axisSize: gd.layout.font ? gd.layout.font.size : null,
        markerSize: gd.data.length ? gd.data[0].marker.size : null,
        markerOpacity: gd.data.length ? gd.data[0].marker.opacity : null,
        traces: gd.data.length,
        points: gd.data.reduce((n, t) => n + (t.x ? t.x.length : 0), 0),
        graphHeight: document.getElementById('scatter-graph').style.height,
    };
}
"""

#: Fire Plotly's own click event for the first drawn point, carrying the
#: ``customdata`` the real handler resolves against. Returns the index so
#: a test can report which colony it asked for.
_CLICK_FIRST_POINT_JS = """
() => {
    const gd = document.querySelector('#scatter-graph .js-plotly-plot');
    if (!gd || !gd.data.length) return null;
    for (const trace of gd.data) {
        if (!trace.customdata || !trace.customdata.length) continue;
        const carried = trace.customdata[0];
        gd.dispatchEvent(new CustomEvent('plotly_click', {}));
        if (typeof gd.emit === 'function') {
            gd.emit('plotly_click', {
                points: [{
                    customdata: carried,
                    x: trace.x[0],
                    y: trace.y[0],
                    curveNumber: 0,
                    pointNumber: 0,
                }],
            });
        }
        return Array.isArray(carried) ? carried[0] : carried;
    }
    return null;
}
"""


def _master_df() -> pl.DataFrame:
    """Two images over two strains, with a time column and an area.

    Two strains so the section pager has somewhere to step, and two time
    points so an X axis is not a constant.
    """
    rows: list[dict[str, object]] = []
    label = 0
    for image_index, image in enumerate(_IMAGES):
        for strain in ("BY4741", "S288C"):
            for _ in range(3):
                label += 1
                rows.append(
                    {
                        _DATASET_COLUMN: "ds1",
                        str(IMAGE.IMAGE_NAME): image,
                        _TIME_COLUMN: float(image_index),
                        "Object_Label": label,
                        _STRAIN_COLUMN: strain,
                        "Grid_RowNum": 1,
                        "Grid_ColNum": label,
                        "Bbox_MinRR": 10,
                        "Bbox_MaxRR": 310,
                        "Bbox_MinCC": 10,
                        "Bbox_MaxCC": 310,
                        "Bbox_CenterRR": 160,
                        "Bbox_CenterCC": 160,
                        _AREA_COLUMN: float(100 + label),
                    }
                )
    return pl.DataFrame(rows)


@pytest.fixture()
def scatter_server(tmp_path: Path) -> Iterator[tuple[str, str]]:
    """A live hub over a run the Scatter tab can plot.

    Yields ``(hub_url, output_rel)``.
    """
    sandbox = _build_sandbox(tmp_path)
    cli_out = sandbox / "results" / _OUTPUT_NAME
    df = _master_df()
    write_master(cli_out, df)
    write_measurements_mirror(cli_out, df)
    (cli_out / "results" / "ds1" / "measurements").mkdir(
        parents=True, exist_ok=True
    )
    overlays = cli_out / "deliverables" / "overlays" / "ds1"
    overlays.mkdir(parents=True, exist_ok=True)
    for image in _IMAGES:
        (overlays / f"{image}.png").write_bytes(_TINY_PNG)
        (overlays / f"{Path(image).stem}.png").write_bytes(_TINY_PNG)
    publish_coherent_terminal_evidence(cli_out, total_images=len(_IMAGES))

    for hub_url in _start_live_server(sandbox):
        yield hub_url, f"results/{_OUTPUT_NAME}"


def _open_scatter(page: Page, hub_url: str, output_rel: str) -> None:
    """Bind the run, open the Scatter tab and wait for a drawn figure."""
    bind_results_output(page, hub_url, output_rel)
    page.goto(hub_url + "/results/")
    page.wait_for_selector("a.nav-link", timeout=20_000)
    page.locator("a.nav-link", has_text="Scatter").first.click()
    page.wait_for_selector("#scatter-graph", state="visible", timeout=20_000)
    # Bind axes explicitly: the tab's default Y is the first numeric
    # measurement, which for this frame is not guaranteed to be the one
    # with values in it.
    page.evaluate(
        """([x, y, section]) => {
            const dc = window.dash_clientside;
            dc.set_props('scatter-x-col', {value: x});
            dc.set_props('scatter-y-col', {value: y});
            dc.set_props('scatter-section-col', {value: section});
        }""",
        [_TIME_COLUMN, _AREA_COLUMN, _STRAIN_COLUMN],
    )
    page.wait_for_function(
        "() => { const gd = document.querySelector('#scatter-graph "
        ".js-plotly-plot'); return gd && gd.data && gd.data.length > 0 "
        "&& gd.data.some(t => t.x && t.x.length); }",
        timeout=30_000,
    )


def _open_style_section(page: Page) -> None:
    """Open the settings popover and expand its Style accordion item."""
    page.click("#scatter-config-toggle")
    page.wait_for_selector("#scatter-section-col", state="visible", timeout=10_000)
    page.locator('button.accordion-button:has-text("Style")').first.click()
    page.wait_for_selector(
        '[id*="scatter-style-readout"]', state="visible", timeout=10_000
    )


def test_the_scatter_tab_renders_a_figure_with_points(
    page: Page, scatter_server
) -> None:
    """The tab mounts and actually draws, on a real browser.

    An empty figure and a drawn one are the same DOM, so the assertion
    is on the point count Plotly reports rather than on the canvas
    existing.
    """
    hub_url, output_rel = scatter_server
    _open_scatter(page, hub_url, output_rel)

    state = page.evaluate(_FIGURE_STATE_JS)
    assert state is not None, "the Plotly graph never initialised"
    assert state["points"] > 0, "the figure mounted but drew no points"


def test_a_style_stepper_changes_the_rendered_figure(
    page: Page, scatter_server
) -> None:
    """A real click on a pattern-matching button moves the real figure.

    The unit tests drive ``build_render_state`` directly, so they prove
    the store is read; they cannot prove the button is wired to it. That
    wiring is an ``ALL``-keyed callback over a JSON id, which exists only
    on a live app.
    """
    hub_url, output_rel = scatter_server
    _open_scatter(page, hub_url, output_rel)
    before = page.evaluate(_FIGURE_STATE_JS)

    _open_style_section(page)
    plus = page.locator(
        '[id*=\'"field":"axis"\'][id*=\'"dir":1\']'
    ).first
    plus.click()
    page.wait_for_function(
        "(prev) => { const gd = document.querySelector('#scatter-graph "
        ".js-plotly-plot'); return gd && gd.layout.font "
        "&& gd.layout.font.size !== prev; }",
        arg=before["axisSize"],
        timeout=20_000,
    )

    after = page.evaluate(_FIGURE_STATE_JS)
    assert after["axisSize"] == before["axisSize"] + 1
    readout = page.locator('[id*="scatter-style-readout"][id*="axis"]').first
    assert readout.inner_text().strip() == str(after["axisSize"])


def test_facet_height_grows_the_graph_rather_than_the_facets(
    page: Page, scatter_server
) -> None:
    """The control that replaced the fixed ``72vh``.

    Asserted on the graph element's own height, because that is what
    scrolls; a Plotly-internal height would not tell us the page grew.
    """
    hub_url, output_rel = scatter_server
    _open_scatter(page, hub_url, output_rel)
    before = page.evaluate(_FIGURE_STATE_JS)["graphHeight"]

    _open_style_section(page)
    page.locator(
        '[id*=\'"field":"facet_height"\'][id*=\'"dir":1\']'
    ).first.click()
    page.wait_for_function(
        "(prev) => document.getElementById('scatter-graph').style.height "
        "!== prev",
        arg=before,
        timeout=20_000,
    )

    after = page.evaluate(_FIGURE_STATE_JS)["graphHeight"]
    assert int(after.removesuffix("px")) > int(before.removesuffix("px"))


def test_the_page_size_preset_reveals_and_drives_the_inch_inputs(
    page: Page, scatter_server
) -> None:
    """Spec section 11's control, exercised on a live app.

    The Custom row's visibility is a callback Output, so a unit test on
    :func:`page_size_payload` proves the arithmetic and nothing about
    whether the row is actually shown.
    """
    hub_url, output_rel = scatter_server
    _open_scatter(page, hub_url, output_rel)
    page.click("#scatter-config-toggle")
    page.wait_for_selector("#scatter-section-col", state="visible", timeout=10_000)
    page.locator('button.accordion-button:has-text("Export")').first.click()
    page.wait_for_selector("#scatter-page-preset", state="visible", timeout=10_000)

    page.evaluate(
        "() => window.dash_clientside.set_props("
        "'scatter-page-preset', {value: 'A4 landscape'})"
    )
    page.wait_for_function(
        "() => document.getElementById('scatter-page-width').value === '11.69'",
        timeout=20_000,
    )
    assert (
        page.evaluate(
            "() => getComputedStyle(document.getElementById("
            "'scatter-page-custom-row')).display"
        )
        == "none"
    )

    page.evaluate(
        "() => window.dash_clientside.set_props("
        "'scatter-page-preset', {value: 'custom'})"
    )
    page.wait_for_function(
        "() => getComputedStyle(document.getElementById("
        "'scatter-page-custom-row')).display !== 'none'",
        timeout=20_000,
    )


def test_clicking_a_point_opens_its_colony_in_the_inspector(
    page: Page, scatter_server
) -> None:
    """The chain no test covered: click -> resolve -> crop over HTTP.

    Every link is server-side. The click carries an int index into
    ``master_df``, a callback resolves it to ``(dataset, stem, label)``,
    and the crop arrives from the ``scatter-crops`` Flask route -- none
    of which a unit test reaches.
    """
    hub_url, output_rel = scatter_server
    _open_scatter(page, hub_url, output_rel)

    carried = page.evaluate(_CLICK_FIRST_POINT_JS)
    assert carried is not None, "no point carried a customdata index to click"

    page.wait_for_selector(
        "#scatter-inspector.show", state="visible", timeout=20_000
    )
    title = page.locator("#scatter-inspector-title").inner_text().strip()
    assert "ds1" in title, f"inspector did not name the colony: {title!r}"
    assert "label" in title

    page.wait_for_function(
        "() => { const img = document.getElementById("
        "'scatter-inspector-crop'); return img && img.getAttribute('src') "
        "&& img.getAttribute('src').includes('scatter-crops'); }",
        timeout=20_000,
    )
