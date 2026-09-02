"""Browser-driven E2E tests for the results-viewer Heatmap tab.

Seven tests covering every shipping Heatmap affordance enumerated in
the spec at
``docs/superpowers/specs/2026-05-12-qc-analysis-and-gui-design.md``
lines 1222-1227, plus one race-condition edge (``test_heatmap_renders
_qc_augmented_frame_not_stale``). Each test docstring summarises the
row in FEATURES.md that references the function name.

The tests share a function-scoped sandbox helper that builds a real
``master_measurements.parquet`` with grid columns (``Grid_RowNum`` /
``Grid_ColNum``), canonical image/time metadata, and a representative
measurement column. Specific tests override the fixture
shape when they need a different frame (e.g. no grid columns for the
empty-state test, or multi-row-per-cell for the aggregator semantics
test).

Tests gated by ``PLAYWRIGHT=1`` via the module-level skip in
``conftest.py``.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Callable, Iterator

import polars as pl
import pytest
from playwright.sync_api import Page

from phenotypic.gui._design import OI_VERMILION
from phenotypic.schema import CULTURE, EXPERIMENT, IMAGE
from phenotypic.sdk_ import pipeline_json_path
from tests._output_layout import write_master, write_measurements_mirror
from tests.e2e.gui.conftest import (
    _build_sandbox,
    _start_live_server,
    bind_results_output,
    publish_coherent_terminal_evidence,
)


# Module-level marker: skipped on CI via ``-m "not ci_flaky"`` in the
# gui-e2e workflow. Locally these tests pass reliably (verified: 24/24
# across three full-file runs on macOS aarch64); on GHA ubuntu-latest
# shared runners the Dash callback chain + Plotly first-render budget
# stochastically exceeds the 15s ``wait_for_function`` poll. See
# ``tests/CLAUDE.md`` for the convention and re-validation workflow.
pytestmark = [
    pytest.mark.ci_flaky,
    pytest.mark.skip(
        reason=(
            "QC/Heatmap/Error are unmounted by "
            "docs/superpowers/specs/2026-08-26-gui-simplification-removals "
            "(spec section 3). These tests are the acceptance suite for the "
            "overhauled tabs; delete this marker when the surface returns."
        )
    ),
]


# ---------------------------------------------------------------------------
# Sandbox helpers (kept in sync with test_qc_tab.py — duplicated to keep
# each test module self-contained without importing test helpers across
# files, which conftest auto-discovery does not handle cleanly).
# ---------------------------------------------------------------------------


_OUTPUT_NAME = "CliOutputExample"

_IMAGES = ("plate_001.tif", "plate_002.tif")
_DATASET_COLUMN = str(EXPERIMENT.DATASET)
_TIME_COLUMN = str(CULTURE.TIME)
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII="
)


def _default_master_df() -> pl.DataFrame:
    """Build a 2-image, 2x3 grid frame with one time-point per row.

    Single time-point per row means the time slider stays hidden by
    default. The ``test_time_slider_visibility`` test overrides this
    builder for the variants that need multi-time-point frames.
    """
    rows: list[dict[str, object]] = []
    label = 0
    for image in _IMAGES:
        for r in range(1, 3):
            for c in range(1, 4):
                label += 1
                rows.append(
                    {
                        _DATASET_COLUMN: "ds1",
                        str(IMAGE.IMAGE_NAME): image,
                        _TIME_COLUMN: 0.0,
                        "Object_Label": label,
                        "Grid_RowNum": r,
                        "Grid_ColNum": c,
                        "Bbox_MinRR": 10,
                        "Bbox_MaxRR": 310,
                        "Bbox_MinCC": 10,
                        "Bbox_MaxCC": 310,
                        "Bbox_CenterRR": 160,
                        "Bbox_CenterCC": 160,
                        "Size_Area": float(100 + r * 10 + c),
                    }
                )
    return pl.DataFrame(rows)


def _seed_master_df_in_output(sandbox: Path, df: pl.DataFrame) -> Path:
    """Write a master parquet (and post-applied mirror) into the sandbox.

    Returns the absolute CLI output directory.
    """
    cli_out = sandbox / "results" / _OUTPUT_NAME
    write_master(cli_out, df)
    write_measurements_mirror(cli_out, df)
    (cli_out / "results" / "ds1" / "measurements").mkdir(parents=True, exist_ok=True)
    overlays = cli_out / "deliverables" / "overlays" / "ds1"
    overlays.mkdir(parents=True, exist_ok=True)
    for image in _IMAGES:
        (overlays / f"{image}.png").write_bytes(_TINY_PNG)
        (overlays / f"{Path(image).stem}.png").write_bytes(_TINY_PNG)
    publish_coherent_terminal_evidence(cli_out, total_images=len(_IMAGES))
    return cli_out


def _seed_qc_recipe(output_dir: Path, payload: dict | str) -> Path:
    """Seed the canonical typed pipeline config with the viewer's QC recipe."""
    target = pipeline_json_path(output_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        target.write_text(payload, encoding="utf-8")
    else:
        target.write_text(
            json.dumps({"qc": payload.get("checks", [])}, indent=2),
            encoding="utf-8",
        )
    return target


def _hand_off_viewer(page: Page, hub_url: str, output_rel: str) -> None:
    """POST ``output_rel`` to the viewer-handoff endpoint via the page."""
    bind_results_output(page, hub_url, output_rel)


def _dismiss_qc_modal_if_open(page: Page) -> None:
    """Dismiss the QC add-check modal if Dash opened it on initial layout.

    The QC tab's ``_on_modal_open`` callback uses a pattern-matching
    ``Input({"type": "qc-card-edit", "index": ALL}, "n_clicks")`` which
    in Dash 4 fires once at boot when there's already a matching card.
    With ``prevent_initial_call=True`` the callback is meant to skip
    that fire, but pattern-matching ``ALL`` triggers still slip through
    when the pattern matches more than zero elements at mount time.
    Robust: poll for up to ~3 s, pressing Escape each time the modal
    is still open. This handles late re-opens from concurrent
    pattern-matching fires after the first dismiss.
    """
    import time as _time

    deadline = _time.monotonic() + 3.0
    page.wait_for_timeout(300)  # let any initial callback settle
    while _time.monotonic() < deadline:
        has_open = page.evaluate(
            "() => document.querySelectorAll('.modal.show').length > 0"
        )
        if not has_open:
            return
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)
    # Final state check — if still open, raise to surface the issue.
    has_open = page.evaluate(
        "() => document.querySelectorAll('.modal.show').length > 0"
    )
    assert not has_open, "QC modal failed to dismiss within 3 s"


def _navigate_to_heatmap_tab(page: Page, hub_url: str) -> None:
    """Navigate to /results/ and switch to the Heatmap tab."""
    page.goto(hub_url + "/results/")
    # The figure is the load-bearing anchor — its mount confirms the
    # Heatmap tab body was attached.
    page.wait_for_selector("#heatmap-figure", state="attached", timeout=15_000)
    _dismiss_qc_modal_if_open(page)
    heatmap_tab = page.locator("a.nav-link", has_text="Heatmap").first
    heatmap_tab.click()


def _se_entry(
    *,
    instance_id: str,
    on: str = "Size_Area",
    groupby: tuple[str, ...] = (str(IMAGE.IMAGE_NAME),),
    warn_threshold: float = 0.10,
    fail_threshold: float = 0.20,
    enabled: bool = True,
) -> dict:
    """Build a ReplicateAgreement recipe entry."""
    return {
        "instance_id": instance_id,
        "class": "ReplicateAgreement",
        "enabled": enabled,
        "params": {
            "on": on,
            "groupby": list(groupby),
            "time_label": _TIME_COLUMN,
            "warn_threshold": warn_threshold,
            "fail_threshold": fail_threshold,
            "min_replicates": 2,
        },
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_factory(request: pytest.FixtureRequest) -> Callable[[], pl.DataFrame]:
    """Allow individual tests to swap the master frame builder."""
    return getattr(request, "param", _default_master_df)


@pytest.fixture
def fake_sandbox(
    tmp_path: Path,
    df_factory: Callable[[], pl.DataFrame],
) -> Path:
    """Function-scoped sandbox seeded with the parametrised frame.

    The default factory ``_default_master_df`` is used unless a test
    parametrises ``df_factory`` indirectly.
    """
    sandbox = _build_sandbox(tmp_path)
    _seed_master_df_in_output(sandbox, df_factory())
    return sandbox


@pytest.fixture
def live_server(fake_sandbox: Path) -> Iterator[str]:
    yield from _start_live_server(fake_sandbox)


@pytest.fixture
def hub_url(live_server: str) -> str:
    return live_server


@pytest.fixture
def output_dir(fake_sandbox: Path) -> Path:
    return fake_sandbox / "results" / _OUTPUT_NAME


@pytest.fixture
def output_rel() -> str:
    return f"results/{_OUTPUT_NAME}"


def _open_dash_dropdown(page: Page, dropdown_id: str) -> None:
    """Open a Dash ``dcc.Dropdown`` via keyboard.

    The current Dash dropdown (Dash 4 + dash-bootstrap-components 2)
    renders as a Radix UI button. Plain ``page.click()`` does not
    always trigger the Radix open handler when the picker sits below
    the fold or when sibling tab panels overlap in the DOM; using
    keyboard ``Enter`` after a focus is the most reliable cross-page
    opener and is what Radix's docs document for keyboard users.
    """
    locator = page.locator(f"#{dropdown_id}")
    locator.scroll_into_view_if_needed()
    locator.focus()
    page.keyboard.press("Enter")
    page.wait_for_selector(
        '[role="listbox"] [role="option"]', state="attached", timeout=5_000
    )


def _dash_dropdown_options(page: Page, dropdown_id: str) -> list[str]:
    """Read the option text list from a Dash ``dcc.Dropdown``."""
    _open_dash_dropdown(page, dropdown_id)
    options = page.evaluate(
        """
        (id) => {
            const trigger = document.getElementById(id);
            if (!trigger) return [];
            // Radix mounts the listbox via aria-controls; resolve it.
            const controlledId = trigger.getAttribute('aria-controls');
            let lb = controlledId && document.getElementById(controlledId);
            if (!lb) {
                lb = document.querySelector('[role="listbox"]');
            }
            if (!lb) return [];
            return Array.from(lb.querySelectorAll('[role="option"]'))
                .map(o => (o.textContent || '').trim());
        }
        """,
        dropdown_id,
    )
    page.keyboard.press("Escape")
    return options


def _dash_dropdown_value(page: Page, dropdown_id: str) -> str | None:
    """Return the rendered text of a Dash dropdown's currently selected value."""
    return page.evaluate(
        f"""
        () => {{
            const ip = document.getElementById('{dropdown_id}');
            if (!ip) return null;
            const val = ip.querySelector('.dash-dropdown-value');
            return val ? val.textContent.trim() : null;
        }}
        """
    )


def _dash_dropdown_pick(page: Page, dropdown_id: str, label_text: str) -> None:
    """Open a Dash dropdown and click the option whose text matches ``label_text``."""
    _open_dash_dropdown(page, dropdown_id)
    page.locator(
        '[role="listbox"] [role="option"]', has_text=label_text
    ).first.click()


def _first_card_picker_value(page: Page) -> str | None:
    """Return the rendered value of the first Plate card image picker."""
    return page.evaluate(
        """() => {
            const picker = Array.from(document.querySelectorAll('[id]')).find((el) =>
                el.id.includes('"type":"card-picker"')
            );
            if (!picker) return null;
            const value = picker.querySelector('.dash-dropdown-value');
            if (!value) return null;
            const text = value.textContent.trim();
            return text === 'Select image...' ? null : text;
        }"""
    )


def _click_first_card_nav(page: Page, direction: str) -> None:
    """Click the first Plate card picker navigation button."""
    selector_type = "card-picker-next" if direction == "next" else "card-picker-prev"
    selector = f'button[id*="\\"type\\":\\"{selector_type}\\""]'
    button = page.locator(selector).first
    button.wait_for(state="attached", timeout=10_000)
    button.click()


def _wait_for_first_card_picker_options(page: Page) -> None:
    """Wait until the first Plate card picker exposes both seeded images."""
    picker = page.locator(
        '[id*="\\"type\\":\\"card-picker\\""].dash-dropdown'
    ).first
    picker.wait_for(state="attached", timeout=10_000)
    picker.focus()
    page.keyboard.press("Enter")
    page.wait_for_function(
        "() => document.querySelectorAll('[role=\"listbox\"] [role=\"option\"]').length >= 2",
        timeout=10_000,
    )
    page.keyboard.press("Escape")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_plate_card_image_icon_navigation(
    page: Page,
    hub_url: str,
    output_rel: str,
) -> None:
    """Plate card navigation survives reload and uses icon-only buttons."""
    _hand_off_viewer(page, hub_url, output_rel)

    # Binding navigates to Results. Verify the initial card first, then reload
    # the same browser session. The session-backed card list must rebuild the
    # new DOM instead of being mistaken for an already-rendered server state.
    _wait_for_first_card_picker_options(page)
    assert page.locator(
        '[id*="\\"type\\":\\"card-picker\\""].dash-dropdown'
    ).count() == 1
    page.reload()
    _wait_for_first_card_picker_options(page)
    page.wait_for_selector(
        'button[id*="\\"type\\":\\"card-picker-next\\""]:not([disabled])',
        timeout=15_000,
    )

    assert _first_card_picker_value(page) is None
    _click_first_card_nav(page, "next")
    page.wait_for_function(
        "() => {"
        "  const picker = Array.from(document.querySelectorAll('[id]')).find((el) => "
        "    el.id.includes('\"type\":\"card-picker\"'));"
        "  const value = picker && picker.querySelector('.dash-dropdown-value');"
        "  return value && value.textContent.trim() === 'ds1 / plate_001.tif';"
        "}",
        timeout=10_000,
    )
    assert _first_card_picker_value(page) == "ds1 / plate_001.tif"

    _click_first_card_nav(page, "next")
    page.wait_for_function(
        "() => {"
        "  const picker = Array.from(document.querySelectorAll('[id]')).find((el) => "
        "    el.id.includes('\"type\":\"card-picker\"'));"
        "  const value = picker && picker.querySelector('.dash-dropdown-value');"
        "  return value && value.textContent.trim() === 'ds1 / plate_002.tif';"
        "}",
        timeout=10_000,
    )
    assert _first_card_picker_value(page) == "ds1 / plate_002.tif"

    _click_first_card_nav(page, "previous")
    page.wait_for_function(
        "() => {"
        "  const picker = Array.from(document.querySelectorAll('[id]')).find((el) => "
        "    el.id.includes('\"type\":\"card-picker\"'));"
        "  const value = picker && picker.querySelector('.dash-dropdown-value');"
        "  return value && value.textContent.trim() === 'ds1 / plate_001.tif';"
        "}",
        timeout=10_000,
    )


def test_color_picker_lists_measurements_and_qc_severities(
    page: Page,
    hub_url: str,
    output_rel: str,
    output_dir: Path,
) -> None:
    """Color picker options = measurement columns ∪ ``QC_*_Metric``.

    Spec line 1222. After a ReplicateAgreement check is configured the
    augmented frame should expose ``QC_SE_Metric`` and the dropdown
    should include it. Removing the check should contract the list.
    """
    instance_id = "qc-SE-color000"
    _seed_qc_recipe(
        output_dir,
        {
            "version": 1,
            "checks": [_se_entry(instance_id=instance_id)],
        },
    )
    _hand_off_viewer(page, hub_url, output_rel)
    _navigate_to_heatmap_tab(page, hub_url)
    page.wait_for_selector("#heatmap-color-picker", timeout=10_000)

    # Activate the QC tab so the card-body refresh callback writes
    # CFG_QC_AUGMENTED_FRAME with QC_SE_Metric, then switch back to
    # the Heatmap tab and read the color picker options.
    page.locator("a.nav-link", has_text="QC").first.click()
    page.wait_for_function(
        "() => {"
        "  const s = document.querySelector('[id*=\"qc-card-summary\"]');"
        "  return s && (s.textContent || '').includes('groups:');"
        "}",
        timeout=15_000,
    )
    page.locator("a.nav-link", has_text="Heatmap").first.click()
    page.wait_for_timeout(1_500)

    options = _dash_dropdown_options(page, "heatmap-color-picker")
    assert "Size_Area" in options, (
        f"Color picker missing the seed measurement column; got: {options!r}"
    )
    # Per spec line 800: "The color-picker option list also reads from
    # the cached augmented frame: union of measurements columns plus any
    # QC_*_Metric columns present in the augmented frame." With a
    # ReplicateAgreement configured, QC_SE_Metric must appear.
    assert "QC_SE_Metric" in options, (
        "Heatmap color picker missing QC_SE_Metric even though a "
        "ReplicateAgreement check is configured. Wave D's "
        "`_refresh_heatmap_controls` callback in "
        "src/phenotypic/gui/results_viewer/_heatmap_tab/_callbacks.py "
        "subscribes only to STORE_QC_RECIPE_REVISION and "
        "STORE_REMOVED_KEYS; it does NOT subscribe to "
        "STORE_QC_AUGMENTED_REVISION, so the picker option list is "
        "not refreshed when the QC writer populates "
        "CFG_QC_AUGMENTED_FRAME with new QC_*_Metric columns. "
        f"Got options: {options!r}"
    )


def test_aggregator_semantics(
    page: Page,
    hub_url: str,
    output_rel: str,
    output_dir: Path,
    fake_sandbox: Path,
) -> None:
    """Switching ``mean`` -> ``max`` changes a cell's aggregated value.

    Spec line 1223. We rewrite the master frame so two rows share
    ``(Grid_RowNum=2, Grid_ColNum=3, time=4)`` with different
    ``Size_Area`` values. Mean and max should disagree.
    """
    # Two rows sharing a bin so the aggregator's choice matters.
    rows = []
    label = 0
    for image in _IMAGES:
        for r in range(1, 3):
            for c in range(1, 4):
                label += 1
                rows.append(
                    {
                        _DATASET_COLUMN: "ds1",
                        str(IMAGE.IMAGE_NAME): image,
                        _TIME_COLUMN: 4.0,
                        "Object_Label": label,
                        "Grid_RowNum": r,
                        "Grid_ColNum": c,
                        "Size_Area": 100.0,
                    }
                )
    # Append a second row at (image_0, r=2, c=3) with a much higher
    # value so mean vs. max disagree on that bin.
    label += 1
    rows.append(
        {
            _DATASET_COLUMN: "ds1",
            str(IMAGE.IMAGE_NAME): _IMAGES[0],
            _TIME_COLUMN: 4.0,
            "Object_Label": label,
            "Grid_RowNum": 2,
            "Grid_ColNum": 3,
            "Size_Area": 200.0,
        }
    )
    df = pl.DataFrame(rows)
    _seed_master_df_in_output(fake_sandbox, df)

    _hand_off_viewer(page, hub_url, output_rel)
    _navigate_to_heatmap_tab(page, hub_url)
    page.wait_for_selector("#heatmap-aggregator-picker", timeout=10_000)

    # The color-picker defaults to the canonical dataset column (first option),
    # which is a string column that won't aggregate. Switch to
    # ``Size_Area`` so the heatmap holds numeric cells.
    _dash_dropdown_pick(page, "heatmap-color-picker", "Size_Area")
    page.wait_for_timeout(1_000)

    def _heatmap_z_for(row: int, col: int) -> float | None:
        """Read the value of cell (row, col) from the heatmap trace.

        Plotly 6 packs the ``z`` array on ``fig.data[0]`` as a typed-
        array wrapper with ``{dtype, bdata, shape, _inputArray}`` — the
        public 2-D array lives at ``fig._fullData[0].z``. We read from
        ``_fullData`` so the raw numeric values are visible to JS.
        """
        return page.evaluate(
            """
            ({r, c}) => {
                const fig = document.querySelector('#heatmap-figure .js-plotly-plot');
                if (!fig || !fig._fullData || !fig._fullData.length) return null;
                const trace = fig._fullData[0];
                if (!trace || !trace.z) return null;
                const xs = Array.from(trace.x || []);
                const ys = Array.from(trace.y || []);
                let yIdx = -1;
                for (let i = 0; i < ys.length; i++) {
                    if (Number(ys[i]) === r) { yIdx = i; break; }
                }
                let xIdx = -1;
                for (let j = 0; j < xs.length; j++) {
                    if (Number(xs[j]) === c) { xIdx = j; break; }
                }
                if (yIdx < 0 || xIdx < 0) return null;
                const row = trace.z[yIdx];
                if (!row) return null;
                const v = row[xIdx];
                return v === null || v === undefined ? null : Number(v);
            }
            """,
            {"r": row, "c": col},
        )

    # Wait for the initial figure render.
    page.wait_for_function(
        "() => {"
        "  const fig = document.querySelector('#heatmap-figure .js-plotly-plot');"
        "  return fig && fig._fullData && fig._fullData.length > 0;"
        "}",
        timeout=15_000,
    )

    # Mean of {100, 200} = 150.
    mean_value = _heatmap_z_for(2, 3)
    # Switch aggregator to max via the Dash dropdown and wait for the
    # render callback to re-fire.
    _dash_dropdown_pick(page, "heatmap-aggregator-picker", "Max")
    page.wait_for_timeout(1_000)
    max_value = _heatmap_z_for(2, 3)
    assert mean_value is not None and max_value is not None
    assert abs(max_value - 200.0) < 0.5, f"max should be ~200; got {max_value!r}"
    assert mean_value != max_value, (
        f"mean ({mean_value!r}) should differ from max ({max_value!r})"
    )


def test_image_picker(
    page: Page,
    hub_url: str,
    output_rel: str,
    output_dir: Path,
) -> None:
    """Image picker switches the rendered grid contents.

    Spec line 1224. The two test images carry the same value matrix,
    so we differentiate via row/column counts and the hover-template
    image-file echo.
    """
    _hand_off_viewer(page, hub_url, output_rel)
    _navigate_to_heatmap_tab(page, hub_url)
    page.wait_for_selector("#heatmap-image-picker", timeout=10_000)
    page.wait_for_function(
        "() => {"
        "  const fig = document.querySelector('#heatmap-figure .js-plotly-plot');"
        "  return fig && fig._fullData && fig._fullData.length > 0;"
        "}",
        timeout=15_000,
    )

    initial = _dash_dropdown_value(page, "heatmap-image-picker")
    assert initial in _IMAGES, f"unexpected initial image: {initial!r}"

    # Switch to the other image via the icon-only next button, then back
    # via the previous button.
    other = _IMAGES[1] if initial == _IMAGES[0] else _IMAGES[0]
    page.locator("#heatmap-image-next").click()
    page.wait_for_function(
        "(expected) => {"
        "  const ip = document.getElementById('heatmap-image-picker');"
        "  const val = ip && ip.querySelector('.dash-dropdown-value');"
        "  return val && val.textContent.trim() === expected;"
        "}",
        arg=other,
        timeout=10_000,
    )
    assert _dash_dropdown_value(page, "heatmap-image-picker") == other

    page.locator("#heatmap-image-prev").click()
    page.wait_for_function(
        "(expected) => {"
        "  const ip = document.getElementById('heatmap-image-picker');"
        "  const val = ip && ip.querySelector('.dash-dropdown-value');"
        "  return val && val.textContent.trim() === expected;"
        "}",
        arg=initial,
        timeout=10_000,
    )
    assert _dash_dropdown_value(page, "heatmap-image-picker") == initial


def test_colony_tile_size_icon_stepper(
    page: Page,
    hub_url: str,
    output_rel: str,
) -> None:
    """Colony tile-size ``−``/``+`` buttons update readout and tile width."""
    _hand_off_viewer(page, hub_url, output_rel)
    page.goto(hub_url + "/results/")
    page.locator("a.nav-link", has_text="Colony").first.click()
    page.wait_for_selector("#colony-tile-size-readout", timeout=10_000)
    page.wait_for_selector(".colony-cell-img", timeout=15_000)

    initial_readout = page.locator("#colony-tile-size-readout").text_content()
    assert initial_readout == "150 px"
    initial_width = page.evaluate(
        "() => document.querySelector('.colony-cell-img').getBoundingClientRect().width"
    )

    page.locator("#colony-tile-size-plus").click()
    page.wait_for_function(
        "() => {"
        "  const readout = document.querySelector('#colony-tile-size-readout');"
        "  const tile = document.querySelector('.colony-cell-img');"
        "  return readout && tile"
        "    && readout.textContent === '166 px'"
        "    && tile.getBoundingClientRect().width === 166;"
        "}",
        timeout=10_000,
    )
    larger_width = page.evaluate(
        "() => document.querySelector('.colony-cell-img').getBoundingClientRect().width"
    )
    assert larger_width > initial_width

    page.locator("#colony-tile-size-minus").click()
    page.wait_for_function(
        "() => {"
        "  const readout = document.querySelector('#colony-tile-size-readout');"
        "  const tile = document.querySelector('.colony-cell-img');"
        "  return readout && tile"
        "    && readout.textContent === '150 px'"
        "    && tile.getBoundingClientRect().width === 150;"
        "}",
        timeout=10_000,
    )
    restored_width = page.evaluate(
        "() => document.querySelector('.colony-cell-img').getBoundingClientRect().width"
    )
    assert restored_width == initial_width


@pytest.mark.parametrize(
    "df_factory,visible_expected,caption_expected",
    [
        # Single-tp hidden: all rows share a single canonical time value.
        pytest.param(
            _default_master_df,
            False,
            "",
            id="single-tp-hidden",
        ),
        # Multi-tp visible with no caption.
        pytest.param(
            lambda: pl.DataFrame(
                [
                    {
                            _DATASET_COLUMN: "ds1",
                            str(IMAGE.IMAGE_NAME): _IMAGES[0],
                            _TIME_COLUMN: t,
                        "Object_Label": idx,
                        "Grid_RowNum": 1,
                        "Grid_ColNum": 1,
                        "Size_Area": 100.0 + t,
                    }
                    for idx, t in enumerate([0.0, 1.0, 2.0, 3.0])
                ]
            ),
            True,
            "",
            id="multi-tp-visible",
        ),
    ],
    indirect=["df_factory"],
)
def test_time_slider_visibility(
    page: Page,
    hub_url: str,
    output_rel: str,
    visible_expected: bool,
    caption_expected: str,
) -> None:
    """Time slider visibility tracks the master frame's time column.

    Spec line 1225. Two parametrised sub-cases: single-time-point hidden
    and multi-time-point visible. The remaining two spec sub-cases
    (column absent; all-NaN non-numeric) are exercised by the unit
    tests on ``_build_time_slider_state``; here we use the parametrised
    E2E cases to verify the wiring through Dash callbacks.
    """
    _hand_off_viewer(page, hub_url, output_rel)
    _navigate_to_heatmap_tab(page, hub_url)
    page.wait_for_selector("#heatmap-time-slider-wrapper", state="attached")
    # Allow the controls-refresh callback to settle.
    page.wait_for_timeout(1_500)
    display = page.evaluate(
        "() => getComputedStyle(document.getElementById('heatmap-time-slider-wrapper')).display"
    )
    if visible_expected:
        assert display != "none", "Time slider should be visible"
    else:
        assert display == "none", f"Time slider should be hidden; got {display!r}"

    # When the slider is visible, the non-numeric caption should match
    # the expected payload (empty for a fully numeric time column
    # values; populated when some rows fail coercion).
    caption = (
        page.locator("#heatmap-time-non-numeric-caption").text_content() or ""
    ).strip()
    if visible_expected:
        assert caption == caption_expected, (
            f"caption mismatch: expected {caption_expected!r}, got {caption!r}"
        )


def test_removed_cells_visually_distinct(
    page: Page,
    hub_url: str,
    output_rel: str,
    output_dir: Path,
) -> None:
    """Curated cells render a muted ``x`` marker overlay.

    Spec line 1226. We push curation through the QC tab's
    ``Mark all flagged for removal`` button — this updates
    ``STORE_REMOVED_KEYS``, which the Heatmap render callback
    subscribes to and which causes the overlay traces to be
    emitted for every matched ``(image_file, Object_Label)``.

    A naive approach of pre-trimming the on-disk
    ``measurements.parquet`` mirror does not work: ``OutputRoot``
    prefers the mirror over the master archive, so the in-memory
    master frame equals the trimmed frame and the curation diff is
    zero. Driving the curation via the QC card matches the user-
    facing path.
    """
    # Seed a Count check whose metadata expects more colonies than
    # the master frame contains, so every row gets flagged.
    rows = []
    label = 0
    for image in _IMAGES:
        for _ in range(100):
            label += 1
            rows.append({str(IMAGE.IMAGE_NAME): image, "Object_Label": label})
    csv_path = output_dir / "count_metadata.csv"
    pl.DataFrame(rows).write_csv(csv_path)
    instance_id = "qc-Count-overly"
    _seed_qc_recipe(
        output_dir,
        {
            "version": 1,
            "checks": [
                {
                    "instance_id": instance_id,
                    "class": "ExpectedVsDetectedCount",
                    "enabled": True,
                    "params": {
                        "metadata": str(csv_path),
                        "groupby": [str(IMAGE.IMAGE_NAME)],
                        "on": "Object_Label",
                    },
                },
            ],
        },
    )

    _hand_off_viewer(page, hub_url, output_rel)
    # Activate the QC tab so the card-body refresh runs and the check
    # caches its analyze output (required by ``flagged_keys``).
    page.goto(hub_url + "/results/")
    page.wait_for_selector("#qc-cards-container", state="attached", timeout=15_000)
    _dismiss_qc_modal_if_open(page)
    page.locator("a.nav-link", has_text="QC").first.click()
    summary_selector = f'[id*="qc-card-summary"][id*="{instance_id}"]'
    page.wait_for_function(
        f"""() => {{
            const el = document.querySelector('{summary_selector}');
            return el && (el.textContent || '').includes('groups:');
        }}""",
        timeout=15_000,
    )
    # Click Mark-flagged-for-removal to push every flagged row into
    # ``STORE_REMOVED_KEYS``.
    mark_selector = f'[id*="qc-card-mark-flag"][id*="{instance_id}"]'
    _dismiss_qc_modal_if_open(page)
    page.locator(mark_selector).click(force=True)
    # Yield so the store updates propagate.
    page.wait_for_timeout(800)

    # Switch to the Heatmap tab and pick a numeric color column.
    page.locator("a.nav-link", has_text="Heatmap").first.click()
    page.wait_for_selector("#heatmap-figure", timeout=10_000)
    page.wait_for_function(
        "() => {"
        "  const fig = document.querySelector('#heatmap-figure .js-plotly-plot');"
        "  return fig && fig._fullData && fig._fullData.length > 0;"
        "}",
        timeout=15_000,
    )
    _dash_dropdown_pick(page, "heatmap-color-picker", "Size_Area")
    page.wait_for_timeout(1_500)

    # With curation applied, the figure builds 3 traces: the primary
    # heatmap, a zero-opacity overlay heatmap for hover, and a
    # scatter trace of muted ``x`` markers.
    trace_count = page.evaluate(
        "() => document.querySelector('#heatmap-figure .js-plotly-plot')._fullData.length"
    )
    assert trace_count == 3, (
        f"Expected 3 traces (primary heatmap + zero-opacity overlay + "
        f"scatter x-markers); got {trace_count}"
    )
    scatter_color = page.evaluate(
        """
        () => {
            const fig = document.querySelector('#heatmap-figure .js-plotly-plot');
            const traces = fig._fullData || [];
            for (const trace of traces) {
                if (trace.type === 'scatter' && trace.mode && trace.mode.includes('markers')) {
                    return trace.marker && trace.marker.color;
                }
            }
            return null;
        }
        """
    )
    assert scatter_color is not None, "Scatter overlay trace missing"
    # Removed/excluded cells use the shared vermilion failure color.
    assert scatter_color.lower() == OI_VERMILION.lower(), (
        f"Removed-cell marker color expected OI_VERMILION ({OI_VERMILION}); "
        f"got {scatter_color!r}"
    )


# A frame WITHOUT grid columns so the heatmap renders the empty-state
# annotation rather than a pivot.
def _no_grid_df_factory() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                _DATASET_COLUMN: "ds1",
                str(IMAGE.IMAGE_NAME): _IMAGES[0],
                _TIME_COLUMN: 0.0,
                "Object_Label": idx,
                "Size_Area": 100.0 + idx,
            }
            for idx in range(6)
        ]
    )


@pytest.mark.parametrize(
    "df_factory",
    [pytest.param(_no_grid_df_factory, id="no-grid-cols")],
    indirect=True,
)
def test_empty_state_when_no_grid(
    page: Page,
    hub_url: str,
    output_rel: str,
) -> None:
    """Heatmap renders an empty-state annotation when grid columns are missing.

    Spec line 1227. The figure builder returns a placeholder figure
    with an explanatory annotation rather than raising. The annotation
    text must mention the missing grid columns.
    """
    _hand_off_viewer(page, hub_url, output_rel)
    _navigate_to_heatmap_tab(page, hub_url)
    page.wait_for_selector("#heatmap-figure", timeout=10_000)
    # Allow the controls-refresh + render callbacks to fire.
    page.wait_for_timeout(1_500)

    annotation = page.evaluate(
        """
        () => {
            const fig = document.querySelector('#heatmap-figure .js-plotly-plot');
            const layout = (fig && (fig._fullLayout || fig.layout)) || null;
            if (!layout || !layout.annotations) return null;
            return Array.from(layout.annotations)
                .map(a => a.text || '')
                .join(' | ');
        }
        """
    )
    if annotation is None:
        # The figure may not be fully painted; surface a clearer error.
        pytest.fail(
            "heatmap-figure did not expose layout.annotations; the "
            "figure builder likely did not run. Check controls-refresh "
            "callback availability."
        )
    assert "Grid" in annotation or "grid" in annotation, (
        f"Expected empty-state annotation mentioning grid; got: {annotation!r}"
    )


def test_heatmap_renders_qc_augmented_frame_not_stale(
    page: Page,
    hub_url: str,
    output_rel: str,
    output_dir: Path,
) -> None:
    """Heatmap honours STORE_QC_AUGMENTED_REVISION ordering.

    Race-condition edge from spec lines 775-798. With a configured
    ReplicateAgreement check and the user driving curation via the QC
    card's Mark-flagged-for-removal button, switching the color picker
    to ``QC_SE_Metric`` should reflect the post-curation augmented
    frame — not a stale pre-curation read.

    Driving curation through Mark-flagged-for-removal also bumps
    ``STORE_REMOVED_KEYS``, which triggers the heatmap controls-refresh
    callback so the picker option list re-emits with any newly-present
    ``QC_*_Metric`` columns.
    """
    # Seed a Count check whose metadata expects more colonies than the
    # master frame contains; every group is flagged so Mark-flagged-for-
    # removal pushes keys into STORE_REMOVED_KEYS.
    rows = []
    label = 0
    for image in _IMAGES:
        for _ in range(100):
            label += 1
            rows.append({str(IMAGE.IMAGE_NAME): image, "Object_Label": label})
    csv_path = output_dir / "count_metadata.csv"
    pl.DataFrame(rows).write_csv(csv_path)

    count_id = "qc-Count-stale0"
    se_id = "qc-SE-stalecheck"
    _seed_qc_recipe(
        output_dir,
        {
            "version": 1,
            "checks": [
                {
                    "instance_id": count_id,
                    "class": "ExpectedVsDetectedCount",
                    "enabled": True,
                    "params": {
                        "metadata": str(csv_path),
                        "groupby": [str(IMAGE.IMAGE_NAME)],
                        "on": "Object_Label",
                    },
                },
                _se_entry(instance_id=se_id),
            ],
        },
    )
    _hand_off_viewer(page, hub_url, output_rel)
    page.goto(hub_url + "/results/")
    page.wait_for_selector("#qc-cards-container", state="attached", timeout=15_000)
    _dismiss_qc_modal_if_open(page)
    page.locator("a.nav-link", has_text="QC").first.click()
    page.wait_for_function(
        f"""() => {{
            const el = document.querySelector('[id*=\"qc-card-summary\"][id*=\"{count_id}\"]');
            return el && (el.textContent || '').includes('groups:');
        }}""",
        timeout=15_000,
    )
    # Push curation via Mark-flagged-for-removal — bumps STORE_REMOVED_KEYS,
    # which triggers controls-refresh, which re-reads the augmented frame.
    _dismiss_qc_modal_if_open(page)
    page.locator(f'[id*="qc-card-mark-flag"][id*="{count_id}"]').click(force=True)
    page.wait_for_timeout(1_500)

    # Switch to Heatmap and verify the augmented frame is fresh.
    page.locator("a.nav-link", has_text="Heatmap").first.click()
    page.wait_for_selector("#heatmap-color-picker", timeout=10_000)
    page.wait_for_timeout(2_500)
    options = _dash_dropdown_options(page, "heatmap-color-picker")
    assert "QC_SE_Metric" in options, (
        "Heatmap color picker missing QC_SE_Metric after Mark-flagged-"
        "for-removal triggered STORE_REMOVED_KEYS update. The Wave D "
        "controls-refresh callback should re-fire when the QC writer "
        "(re-)populates CFG_QC_AUGMENTED_FRAME. "
        f"Got options: {options!r}"
    )

    # Switch the color picker to QC_SE_Metric and confirm the figure
    # renders without falling into the stale-frame empty-state.
    _dash_dropdown_pick(page, "heatmap-color-picker", "QC_SE_Metric")
    page.wait_for_timeout(1_500)

    has_data = page.evaluate(
        """
        () => {
            const fig = document.querySelector('#heatmap-figure .js-plotly-plot');
            if (!fig || !fig._fullData || fig._fullData.length === 0) return false;
            return fig._fullData[0].z !== undefined;
        }
        """
    )
    assert has_data, (
        "Heatmap did not render data after switching to QC_SE_Metric; "
        "augmented frame may have been read stale."
    )
