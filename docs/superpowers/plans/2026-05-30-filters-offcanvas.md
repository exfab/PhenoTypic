# Filters Offcanvas Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Relocate the results-viewer's always-on left filter sidebar into a right-docked, on-demand `dbc.Offcanvas` toggled from the top bar, so every tab renders full-width by default while filtering stays one click away.

**Architecture:** The existing `_filter_panel.layout()` tree (a self-contained `dbc.Card`) is moved verbatim into a top-level `dbc.Offcanvas(placement="end")`. The two-column body collapses to a single full-width container. A new `_filter_offcanvas.py` module holds three pure, unit-tested helpers (`next_offcanvas_state`, `active_filter_count`, `badge_children`/`badge_style`) plus `register_filter_offcanvas_callbacks(app)` (toggle + live count badge). The filter spec store, filter semantics, re-hydration, and curation are unchanged — only the panel's location and how it opens change.

**Tech Stack:** Python 3.12, Dash + dash-bootstrap-components (`>=2.0.4`, provides `dbc.Offcanvas`), Polars, pytest (+ Playwright `ci_flaky` for E2E), `uv` runner.

**Approved spec:** `docs/superpowers/specs/2026-05-30-filters-offcanvas-design.md`

**Branch:** `feature/smart-qc-gui` (shared — the user is making parallel changes). Commit steps use **explicit path-scoped** `git add <paths>` per the shared-index protocol; verify `git diff --cached --name-only` before each commit.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/phenotypic/gui/results_viewer/_filter_offcanvas.py` | Pure toggle/count helpers + `register_filter_offcanvas_callbacks` | **Create** |
| `src/phenotypic/gui/results_viewer/_ids.py` | Add `OFFCANVAS_FILTER_ID`, `BTN_FILTERS_TOGGLE`, `FILTER_TOGGLE_BADGE_ID` + `__all__` | Modify |
| `src/phenotypic/gui/results_viewer/_layout.py` | Top-bar Filters toggle in `_build_header`; full-width body + top-level offcanvas in `build_app_layout` | Modify |
| `src/phenotypic/gui/results_viewer/_callbacks.py` | Dispatch `register_filter_offcanvas_callbacks(app)` | Modify |
| `src/phenotypic/gui/results_viewer/_filter_panel.py` | Bulk-paste popover `placement="bottom"` → `"left"` | Modify |
| `src/phenotypic/gui/FEATURES.md` | Rows for toggle button, count badge, offcanvas | Modify |
| `scripts/capture_gui_tutorial_screenshots.py` | Add an offcanvas-open shot to the existing `view_results` flow | Modify |
| `tests/unit/gui/results_viewer/test_filter_offcanvas.py` | Unit tests for the pure helpers + callback registration | **Create** |
| `tests/integration/gui/test_filter_offcanvas_layout.py` | Layout structure: offcanvas hosts the panel; no `lg` columns | **Create** |
| `tests/e2e/gui/test_filter_offcanvas.py` | `ci_flaky` browser: toggle, filter narrows, badge, close | **Create** |

---

## Task 1: Pure offcanvas + badge logic

**Files:**
- Create: `src/phenotypic/gui/results_viewer/_filter_offcanvas.py`
- Test: `tests/unit/gui/results_viewer/test_filter_offcanvas.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/gui/results_viewer/test_filter_offcanvas.py`:

```python
"""Unit tests for the filter-offcanvas pure helpers + callback wiring."""

from __future__ import annotations

import dash
import pytest

from phenotypic.gui.results_viewer import _ids as ids
from phenotypic.gui.results_viewer._filter_offcanvas import (
    active_filter_count,
    badge_children,
    badge_style,
    next_offcanvas_state,
    register_filter_offcanvas_callbacks,
)


class TestNextOffcanvasState:
    def test_falsy_clicks_leave_state_unchanged(self) -> None:
        assert next_offcanvas_state(None, False) is False
        assert next_offcanvas_state(None, True) is True
        assert next_offcanvas_state(0, True) is True

    def test_click_toggles_state(self) -> None:
        assert next_offcanvas_state(1, False) is True
        assert next_offcanvas_state(2, True) is False

    def test_none_is_open_treated_as_closed(self) -> None:
        assert next_offcanvas_state(1, None) is True


class TestActiveFilterCount:
    def test_empty_or_none_is_zero(self) -> None:
        assert active_filter_count([]) == 0
        assert active_filter_count(None) == 0

    def test_counts_only_rows_with_a_column(self) -> None:
        spec = [
            {"id": "a", "column": "Metadata_Dataset", "values": ["WT"]},
            {"id": "b", "column": "", "values": []},  # unconfigured row
            {"id": "c", "column": "Grid_RowNum", "values": []},  # column, no values
        ]
        assert active_filter_count(spec) == 2

    def test_ignores_malformed_entries(self) -> None:
        assert active_filter_count(["junk", {"values": []}, {"column": "  "}]) == 0


class TestBadge:
    def test_children_blank_at_zero(self) -> None:
        assert badge_children(0) == ""
        assert badge_children(3) == "3"

    def test_style_hides_at_zero(self) -> None:
        assert badge_style(0) == {"display": "none"}
        assert badge_style(2) == {"display": "inline-block"}


def test_register_adds_toggle_and_badge_callbacks() -> None:
    app = dash.Dash(__name__)
    register_filter_offcanvas_callbacks(app)
    outputs = set(app.callback_map.keys())
    assert any(ids.OFFCANVAS_FILTER_ID in key and "is_open" in key for key in outputs)
    assert any(
        ids.FILTER_TOGGLE_BADGE_ID in key and "children" in key for key in outputs
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/gui/results_viewer/test_filter_offcanvas.py -q`
Expected: FAIL — `ModuleNotFoundError: ... _filter_offcanvas` (and `AttributeError` for the new ids, added in Task 2).

- [ ] **Step 3: Create the module**

Create `src/phenotypic/gui/results_viewer/_filter_offcanvas.py`:

```python
"""Right-docked filter offcanvas: toggle + active-filter count badge.

The filter panel itself (rows, match-count chip, bulk-paste) lives in
:mod:`._filter_panel` and is mounted *inside* a top-level
``dbc.Offcanvas`` by :mod:`._layout`. This module owns only the two
behaviors that surround that panel:

1. **Toggle** — the top-bar "Filters" button flips the offcanvas
   ``is_open``. dbc's own backdrop / ✕ close the offcanvas internally, so
   the toggle reads the current ``is_open`` as State and inverts it.
2. **Count badge** — a small badge on the toggle button shows how many
   *configured* filter rows are active (rows with a column chosen), so the
   user sees filtering state without opening the panel. The panel keeps its
   own "N images match" chip (result size); this badge is the applied-count.

The logic is split into pure, importable helpers so it is unit-testable
without booting Dash (mirrors the smart-QC ``worklist_row_metric_update``
pattern).
"""

from __future__ import annotations

from typing import Any

import dash
from dash import Input, Output, State

from phenotypic.gui.results_viewer import _ids as ids


def next_offcanvas_state(n_clicks: int | None, is_open: bool | None) -> bool:
    """Return the offcanvas ``is_open`` after a toggle-button click.

    A real click (truthy ``n_clicks``) inverts the current state; a falsy
    ``n_clicks`` (initial mount / no click) leaves it unchanged.
    """
    if not n_clicks:
        return bool(is_open)
    return not bool(is_open)


def active_filter_count(spec: Any) -> int:
    """Count configured filter rows (rows whose ``column`` is set).

    Mirrors the spec-store row shape produced by ``_filter_panel`` —
    ``{"id", "column", "values"}`` — and tolerates malformed payloads
    (non-list, non-dict entries, missing/blank columns) by ignoring them.
    """
    if not isinstance(spec, list):
        return 0
    count = 0
    for row in spec:
        if not isinstance(row, dict):
            continue
        if str(row.get("column", "") or "").strip():
            count += 1
    return count


def badge_children(count: int) -> str:
    """Badge text: blank at 0 (so an empty badge can be hidden), else the count."""
    return "" if count <= 0 else str(count)


def badge_style(count: int) -> dict[str, str]:
    """Badge style: hidden at 0, inline otherwise (no stray empty pill)."""
    return {"display": "none"} if count <= 0 else {"display": "inline-block"}


def register_filter_offcanvas_callbacks(app: dash.Dash) -> None:
    """Wire the Filters toggle and the active-filter count badge."""

    @app.callback(
        Output(ids.OFFCANVAS_FILTER_ID, "is_open"),
        Input(ids.BTN_FILTERS_TOGGLE, "n_clicks"),
        State(ids.OFFCANVAS_FILTER_ID, "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_filter_offcanvas(n_clicks: int | None, is_open: bool | None) -> bool:
        """Flip the offcanvas open/closed on a toggle-button click."""
        return next_offcanvas_state(n_clicks, is_open)

    @app.callback(
        Output(ids.FILTER_TOGGLE_BADGE_ID, "children"),
        Output(ids.FILTER_TOGGLE_BADGE_ID, "style"),
        Input(ids.STORE_FILTER_SPEC, "data"),
    )
    def _update_filter_badge(spec: Any) -> tuple[str, dict[str, str]]:
        """Reflect the active-filter count on the toggle button badge."""
        count = active_filter_count(spec)
        return badge_children(count), badge_style(count)


__all__ = [
    "next_offcanvas_state",
    "active_filter_count",
    "badge_children",
    "badge_style",
    "register_filter_offcanvas_callbacks",
]
```

- [ ] **Step 4: Run test to verify pure-helper tests pass**

Run: `uv run pytest tests/unit/gui/results_viewer/test_filter_offcanvas.py -q -k "NextOffcanvas or ActiveFilter or Badge"`
Expected: PASS (the `test_register_*` test still fails until Task 2 adds the ids).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_filter_offcanvas.py tests/unit/gui/results_viewer/test_filter_offcanvas.py
git diff --cached --name-only   # confirm only these two files
git commit -m "feat(gui): filter-offcanvas pure helpers (toggle + active-count badge)"
```

---

## Task 2: New component IDs

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_ids.py`
- Test: `tests/unit/gui/results_viewer/test_filter_offcanvas.py` (the `test_register_*` test from Task 1)

- [ ] **Step 1: Add the id constants**

In `src/phenotypic/gui/results_viewer/_ids.py`, in the "Static buttons" section (after `BTN_LOCK_VIEWS_TOGGLE`, around L74), add:

```python
#: Top-bar button that opens/closes the right-docked filter offcanvas.
BTN_FILTERS_TOGGLE = "btn-filters-toggle"

#: Count badge on the Filters toggle showing the number of active filter rows.
FILTER_TOGGLE_BADGE_ID = "filter-toggle-badge"
```

In the "Static layout anchors" section (after `FILTER_MATCH_COUNT_ID`, around L97), add:

```python
#: Right-docked ``dbc.Offcanvas`` hosting the filter panel; its ``is_open``
#: is driven by :data:`BTN_FILTERS_TOGGLE`.
OFFCANVAS_FILTER_ID = "filter-offcanvas"
```

- [ ] **Step 2: Export them in `__all__`**

In the `__all__` list (around L633), add `"BTN_FILTERS_TOGGLE"`, `"FILTER_TOGGLE_BADGE_ID"`, and `"OFFCANVAS_FILTER_ID"` (e.g. right after `"BTN_LOCK_VIEWS_TOGGLE"` and `"FILTER_MATCH_COUNT_ID"` respectively):

```python
    "BTN_LOCK_VIEWS_TOGGLE",
    "BTN_FILTERS_TOGGLE",
    "FILTER_TOGGLE_BADGE_ID",
    "OFFCANVAS_FILTER_ID",
```

- [ ] **Step 3: Run the registration test to verify it passes**

Run: `uv run pytest tests/unit/gui/results_viewer/test_filter_offcanvas.py -q`
Expected: PASS (all tests, including `test_register_adds_toggle_and_badge_callbacks`).

- [ ] **Step 4: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_ids.py
git diff --cached --name-only
git commit -m "feat(gui): ids for filter offcanvas + toggle + badge"
```

---

## Task 3: Mount the offcanvas, full-width body, top-bar toggle

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_layout.py`
- Modify: `src/phenotypic/gui/results_viewer/_callbacks.py`
- Test: `tests/integration/gui/test_filter_offcanvas_layout.py`

- [ ] **Step 1: Write the failing layout test**

Create `tests/integration/gui/test_filter_offcanvas_layout.py`:

```python
"""Layout structure: the filter panel lives in a right offcanvas, tabs full-width."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import polars as pl
import pytest

from phenotypic.gui.results_viewer import _ids as ids
from phenotypic.gui.results_viewer._filtered_state import FilteredMeasurements
from phenotypic.gui.results_viewer._layout import build_app_layout
from phenotypic.gui.results_viewer._output_root import OutputRoot


def _seed_output(tmp_path: Path) -> Path:
    """Write a minimal CLI output dir the OutputRoot can discover."""
    out = tmp_path / "results" / "Example"
    out.mkdir(parents=True)
    master = pl.DataFrame(
        {
            "Metadata_Dataset": ["ds1", "ds1"],
            "Metadata_ImageFile": ["a.tif", "b.tif"],
            "Object_Label": [1, 2],
            "Size_Area": [100.0, 200.0],
        }
    )
    master.write_parquet(out / "master_measurements.parquet")
    master.write_parquet(out / "measurements.parquet")
    return out


@pytest.fixture
def output_root(tmp_path: Path) -> OutputRoot:
    return OutputRoot.discover(_seed_output(tmp_path))


def _walk(component: Any) -> Iterator[Any]:
    """Yield a component and all of its descendants."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children") or hasattr(child, "id"):
            yield from _walk(child)


def _ids_in(component: Any) -> set[Any]:
    out: set[Any] = set()
    for node in _walk(component):
        node_id = getattr(node, "id", None)
        if node_id is not None:
            out.add(node_id if isinstance(node_id, str) else str(node_id))
    return out


def test_offcanvas_hosts_filter_panel_and_toggle_present(
    output_root: OutputRoot,
) -> None:
    filtered = FilteredMeasurements.load(Path(output_root.root), output_root.master_df)
    layout = build_app_layout(output_root, filtered)

    # The offcanvas exists and contains the filter panel's row container + add button.
    offcanvas = next(
        n for n in _walk(layout) if getattr(n, "id", None) == ids.OFFCANVAS_FILTER_ID
    )
    assert offcanvas._type == "Offcanvas"
    assert getattr(offcanvas, "placement", None) == "end"
    assert getattr(offcanvas, "is_open", None) is False
    inner = _ids_in(offcanvas)
    assert ids.FILTER_ROWS_CONTAINER_ID in inner
    assert ids.BTN_ADD_FILTER_ROW in inner
    assert ids.FILTER_MATCH_COUNT_ID in inner

    # The top-bar toggle + badge are present in the overall tree.
    all_ids = _ids_in(layout)
    assert ids.BTN_FILTERS_TOGGLE in all_ids
    assert ids.FILTER_TOGGLE_BADGE_ID in all_ids


def test_body_has_no_lg_sidebar_columns(output_root: OutputRoot) -> None:
    filtered = FilteredMeasurements.load(Path(output_root.root), output_root.master_df)
    layout = build_app_layout(output_root, filtered)
    # No dbc.Col with an lg=3/lg=9 split should remain (full-width content).
    for node in _walk(layout):
        if getattr(node, "_type", None) == "Col":
            assert getattr(node, "lg", None) not in (3, 9)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/integration/gui/test_filter_offcanvas_layout.py -q`
Expected: FAIL — `StopIteration`/assertion (no offcanvas yet; `lg=3`/`lg=9` columns still present).

- [ ] **Step 3: Add the toggle button to the header**

In `src/phenotypic/gui/results_viewer/_layout.py`, add `OFFCANVAS_FILTER_ID` is not needed here, but the button is. First extend the design-token import (top of file, the `from phenotypic.gui._design import (...)` block) to include `COLOR_SURFACE` if not already imported — it is already imported in `_filter_panel` but `_layout` imports `COLOR_BG, COLOR_BLUE, COLOR_GOLD, COLOR_MUTED, COLOR_NAVY, COLOR_SURFACE, FONT_SIZE_LABEL`; `COLOR_SURFACE` is already present, so no import change is required.

In `_build_header`, replace the `lock_switch` definition and `top_row` (currently L114–148) with a version that adds the Filters toggle to the right cluster:

```python
    lock_switch = dbc.Switch(
        id=ids.BTN_LOCK_VIEWS_TOGGLE,
        label="Lock views",
        value=False,
        className="ms-2 mb-0",
    )

    filters_toggle = dbc.Button(
        [
            "Filters",
            dbc.Badge(
                "",
                id=ids.FILTER_TOGGLE_BADGE_ID,
                color="primary",
                pill=True,
                className="ms-2",
                style={"display": "none"},
            ),
        ],
        id=ids.BTN_FILTERS_TOGGLE,
        color="secondary",
        outline=True,
        size="sm",
        n_clicks=0,
        className="ms-2",
    )

    logo = html.Img(
        src=f"{url_prefix}{SHARED_LOGO_PATH}",
        alt="PhenoTypic",
        className="results-viewer-header__logo",
    )

    title = html.H4(
        "Results Viewer",
        className="mb-0 me-3 results-viewer-header-title",
        style={"color": _NAVY},
    )

    subtitle = html.Div(
        str(output_root.root),
        className="text-muted small results-viewer-header-subtitle",
        style={"marginTop": "0.1rem"},
    )

    top_row = html.Div(
        [
            logo,
            title,
            pipeline_chip,
            html.Div(style={"flex": "1 1 auto"}),  # spacer
            filters_toggle,
            lock_switch,
        ],
        className="d-flex align-items-center",
    )
```

- [ ] **Step 4: Replace the two-column body with full-width + mount the offcanvas**

In `build_app_layout` (`_layout.py`), replace the `body = dbc.Row([...])` block (L436–455) with a single full-width container:

```python
    body = html.Div(
        tabs,
        className="results-viewer-body",
        style={
            "background": _BG,
            "minHeight": "calc(100vh - 7rem)",
        },
    )

    filter_offcanvas = dbc.Offcanvas(
        sidebar,
        id=ids.OFFCANVAS_FILTER_ID,
        title="Filter",
        placement="end",
        is_open=False,
        scrollable=True,
        backdrop=True,
    )
```

Then update the return tuple (L457–461) to mount the offcanvas:

```python
    return html.Div(
        [stores, header, banner, body, filter_offcanvas],
        id="results-viewer-root",
        style={"background": _BG, "minHeight": "100vh"},
    )
```

(The `sidebar = _filter_panel.layout(output_root)` line at L396 is unchanged — the same tree is now the offcanvas body.)

- [ ] **Step 5: Dispatch the new callbacks**

In `src/phenotypic/gui/results_viewer/_callbacks.py`, import the new module and register it. Update the import block (L57–62):

```python
from phenotypic.gui.results_viewer import (
    _filter_offcanvas,
    _filter_panel,
    _ids as ids,
    _layout,
    _viewer_card,
)
```

And in `register_callbacks`, after the `_filter_panel.register_callbacks(...)` line (L93), add:

```python
    _filter_panel.register_callbacks(app, output_root, filtered_state)
    _filter_offcanvas.register_filter_offcanvas_callbacks(app)
```

- [ ] **Step 6: Run the layout test to verify it passes**

Run: `uv run pytest tests/integration/gui/test_filter_offcanvas_layout.py -q`
Expected: PASS.

- [ ] **Step 7: Lint + type-check the changed files**

Run: `uv run ruff check --fix src/phenotypic/gui/results_viewer/_layout.py src/phenotypic/gui/results_viewer/_callbacks.py src/phenotypic/gui/results_viewer/_filter_offcanvas.py`
Run: `uv run mypy src/phenotypic/gui/results_viewer/_filter_offcanvas.py src/phenotypic/gui/results_viewer/_layout.py 2>&1 | grep -E "_filter_offcanvas.py|_layout.py:" || echo "no new errors in changed files"`
Expected: ruff clean; no NEW mypy errors attributable to the changed files (pre-existing residuals elsewhere are fine).

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_layout.py src/phenotypic/gui/results_viewer/_callbacks.py tests/integration/gui/test_filter_offcanvas_layout.py
git diff --cached --name-only
git commit -m "feat(gui): move filter panel into right offcanvas; full-width tabs"
```

---

## Task 4: Bulk-paste popover opens inward

**Files:**
- Modify: `src/phenotypic/gui/results_viewer/_filter_panel.py:349`
- Test: `tests/unit/gui/results_viewer/test_filter_offcanvas.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/unit/gui/results_viewer/test_filter_offcanvas.py`:

```python
def test_bulk_paste_popover_opens_left() -> None:
    """The per-row bulk-paste popover opens leftward so it stays on-screen
    inside the right-docked offcanvas."""
    from phenotypic.gui.results_viewer._filter_panel import _render_filter_row

    row = _render_filter_row("idx1", "Metadata_Dataset", ["WT"], [])
    popovers = [
        n
        for n in _iter_components(row)
        if getattr(n, "_type", None) == "Popover"
    ]
    assert popovers, "expected a bulk-paste popover in the rendered row"
    assert all(getattr(p, "placement", None) == "left" for p in popovers)


def _iter_components(component):
    """Yield a component and all descendants (local helper)."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "children") or hasattr(child, "id"):
            yield from _iter_components(child)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/gui/results_viewer/test_filter_offcanvas.py::test_bulk_paste_popover_opens_left -q`
Expected: FAIL — popover `placement` is currently `"bottom"`.

- [ ] **Step 3: Change the placement**

In `src/phenotypic/gui/results_viewer/_filter_panel.py`, in `_render_filter_row`, the `paste_popover = dbc.Popover(...)` (L319–352): change

```python
        placement="bottom",
```

to

```python
        placement="left",
```

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run pytest tests/unit/gui/results_viewer/test_filter_offcanvas.py::test_bulk_paste_popover_opens_left -q`
Expected: PASS.

- [ ] **Step 5: Regression — the filter data layer is untouched**

Run: `uv run pytest tests/gui/results_viewer/test_filter_state.py tests/gui/results_viewer/test_filtered_state.py -q`
Expected: PASS (unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/gui/results_viewer/_filter_panel.py tests/unit/gui/results_viewer/test_filter_offcanvas.py
git diff --cached --name-only
git commit -m "fix(gui): bulk-paste popover opens left in right-docked filter panel"
```

---

## Task 5: End-to-end browser proof

**Files:**
- Create: `tests/e2e/gui/test_filter_offcanvas.py`

- [ ] **Step 1: Write the E2E test**

Create `tests/e2e/gui/test_filter_offcanvas.py` (mirrors the harness in `tests/e2e/gui/test_qc_review_splitter.py`):

```python
"""Browser E2E: the filter panel lives in a right offcanvas and still filters.

``ci_flaky``-gated like the other browser E2E (single-threaded Werkzeug dev
server + Dash callback-chain timing flakes on shared CI runners; the SUT is
correct — see ``tests/CLAUDE.md``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import polars as pl
import pytest
from PIL import Image as PILImage
from playwright.sync_api import Page

from tests.e2e.gui.conftest import _build_sandbox, _start_live_server

pytestmark = pytest.mark.ci_flaky

_OUTPUT_NAME = "FilterOffcanvasExample"
_IMAGES = ("plate_001.tif", "plate_002.tif")


def _seed_output(sandbox: Path) -> None:
    out = sandbox / "results" / _OUTPUT_NAME
    rows = []
    label = 0
    for image in _IMAGES:
        dataset = "ds1" if image == _IMAGES[0] else "ds2"
        for _ in range(4):
            label += 1
            rows.append(
                {
                    "Metadata_Dataset": dataset,
                    "Metadata_ImageFile": image,
                    "Object_Label": label,
                    "Bbox_CenterRR": 50,
                    "Bbox_CenterCC": 50,
                    "Bbox_MinRR": 40,
                    "Bbox_MaxRR": 60,
                    "Bbox_MinCC": 40,
                    "Bbox_MaxCC": 60,
                    "Size_Area": float(100 + label),
                }
            )
    master = pl.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    master.write_parquet(out / "master_measurements.parquet")
    master.write_parquet(out / "measurements.parquet")
    overlays = out / "results" / "ds1" / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    for image in _IMAGES:
        stem = Path(image).stem
        PILImage.new("RGB", (120, 120), (200, 0, 0)).save(overlays / f"{stem}.png")


@pytest.fixture
def fake_sandbox(tmp_path: Path) -> Path:
    sandbox = _build_sandbox(tmp_path)
    _seed_output(sandbox)
    return sandbox


@pytest.fixture
def hub_url(fake_sandbox: Path) -> Iterator[str]:
    yield from _start_live_server(fake_sandbox)


def _open_viewer(page: Page, hub_url: str) -> None:
    output_rel = f"results/{_OUTPUT_NAME}"
    page.goto(hub_url + "/")
    page.wait_for_load_state("networkidle")
    resp = page.evaluate(
        """async (path) => {
            const r = await fetch('/sandbox/api/viewer/output-root', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({path: path}),
            });
            return r.status;
        }""",
        output_rel,
    )
    assert resp == 200, f"viewer hand-off failed: {resp}"
    page.goto(hub_url + "/results/")
    page.wait_for_selector("#btn-filters-toggle", state="attached", timeout=15_000)


def _offcanvas_visible(page: Page) -> bool:
    return page.evaluate(
        "() => { const o = document.querySelector('#filter-offcanvas');"
        " return !!o && getComputedStyle(o).visibility !== 'hidden'"
        " && o.getBoundingClientRect().width > 0; }"
    )


def test_offcanvas_toggle_and_filter(page: Page, hub_url: str) -> None:
    _open_viewer(page, hub_url)

    # Boots closed.
    page.wait_for_timeout(300)
    assert not _offcanvas_visible(page), "offcanvas should boot closed"

    # Open via the toggle.
    page.locator("#btn-filters-toggle").click()
    page.wait_for_selector("#filter-rows-container", state="visible", timeout=10_000)
    assert _offcanvas_visible(page)

    # Add a filter row, choose the Dataset column + one value.
    page.locator("#btn-add-filter-row").click()
    column_dd = page.locator(".filter-row .Select-control, .filter-row [class*=control]").first
    column_dd.click()
    page.get_by_text("Metadata_Dataset", exact=True).first.click()
    page.wait_for_timeout(300)

    # Badge reflects one active filter.
    badge_text = page.locator("#filter-toggle-badge").inner_text()
    assert badge_text.strip() == "1", f"badge should show 1 active filter, got {badge_text!r}"

    # Pick a single dataset value → results narrow (match-count chip updates).
    value_dd = page.locator(".filter-row [class*=control]").nth(1)
    value_dd.click()
    page.get_by_text("ds1", exact=True).first.click()
    page.wait_for_timeout(400)
    chip = page.locator("#filter-match-count").inner_text()
    assert "images match" in chip

    # Close via backdrop click.
    page.locator(".offcanvas-backdrop").click()
    page.wait_for_timeout(300)
    assert not _offcanvas_visible(page)
```

- [ ] **Step 2: Run it locally**

Run: `uv run pytest tests/e2e/gui/test_filter_offcanvas.py -q -m ci_flaky`
Expected: PASS locally (deselected on CI). If the react-select selectors flake, adjust the `.filter-row` control locators — the assertions on `#filter-toggle-badge` and `#filter-match-count` are the contract.

- [ ] **Step 3: Commit**

```bash
git add tests/e2e/gui/test_filter_offcanvas.py
git diff --cached --name-only
git commit -m "test(gui): ci_flaky E2E for filter offcanvas toggle + narrowing"
```

---

## Task 6: FEATURES.md ledger

**Files:**
- Modify: `src/phenotypic/gui/FEATURES.md`

- [ ] **Step 1: Read the table format**

Run: `sed -n '1,30p' src/phenotypic/gui/FEATURES.md`
Confirm columns: `Feature | Element | Expected behaviour | Status | Test layer | Test ref` and the status vocabulary (`✅ shipping`).

- [ ] **Step 2: Add three rows**

In the Results Viewer section of `src/phenotypic/gui/FEATURES.md`, add (match the existing column order exactly):

```markdown
| Filters toggle           | `#btn-filters-toggle` (top bar)                 | Opens/closes the right-docked filter offcanvas; offcanvas boots closed so all tabs are full-width by default. | ✅ shipping | unit        | tests/unit/gui/results_viewer/test_filter_offcanvas.py::TestNextOffcanvasState::test_click_toggles_state |
| Active-filter badge      | `#filter-toggle-badge`                          | Badge on the Filters toggle shows the count of configured filter rows; hidden when none are active. | ✅ shipping | unit        | tests/unit/gui/results_viewer/test_filter_offcanvas.py::TestActiveFilterCount::test_counts_only_rows_with_a_column |
| Filter offcanvas         | `#filter-offcanvas` (right slide-in)            | Hosts the filter panel (rows, match-count chip, bulk-paste) unchanged; filtering still narrows Plate + Colony. | ✅ shipping | integration | tests/integration/gui/test_filter_offcanvas_layout.py::test_offcanvas_hosts_filter_panel_and_toggle_present |
```

- [ ] **Step 3: Validate the gate**

Run: `uv run python scripts/check_features_md.py`
Expected: exit 0 (syntax OK; all three `Test ref`s resolve to real tests).

- [ ] **Step 4: Commit**

```bash
git add src/phenotypic/gui/FEATURES.md
git diff --cached --name-only
git commit -m "docs(gui): FEATURES rows for filter offcanvas toggle + badge"
```

---

## Task 7: Tutorial screenshots

**Files:**
- Modify: `scripts/capture_gui_tutorial_screenshots.py`
- Regenerated PNGs under `docs/source/_static/gui_images/`

- [ ] **Step 1: Locate the view_results capture flow**

Run: `grep -n "_capture_view_results\|capture_standalone_viewer_screenshots\|02_viewer_loaded\|03_measurement_table" scripts/capture_gui_tutorial_screenshots.py`
Identify the standalone-viewer capture function and the point after the viewer has loaded.

- [ ] **Step 2: Add an offcanvas-open shot**

In `capture_standalone_viewer_screenshots()`, after the `02_viewer_loaded.png` capture, add a step that clicks the Filters toggle and captures the open panel (use the existing `page`/save-path helpers in that function; follow the local naming convention, e.g. `view_results/04_filter_offcanvas.png`):

```python
    page.click("#btn-filters-toggle")
    page.wait_for_selector("#filter-rows-container", state="visible", timeout=10_000)
    page.wait_for_timeout(400)
    _save(page, "view_results", "04_filter_offcanvas.png")  # use this module's save helper
    page.click("#btn-filters-toggle")  # close again before later shots
    page.wait_for_timeout(300)
```

(Replace `_save(...)` with whatever screenshot helper the surrounding code already uses; do not introduce a new save mechanism. No `WORKFLOWS.md` row is added — the existing `view_results` workflow already covers this.)

- [ ] **Step 3: Regenerate the full screenshot set**

Run: `uv run python scripts/capture_gui_tutorial_screenshots.py`
Expected: completes; `view_results/02`,`03`, the new `04_filter_offcanvas.png`, and `heatmap_exploration/*` reflect the full-width layout.

- [ ] **Step 4: Validate the workflows gate**

Run: `uv run python scripts/check_workflows_md.py`
Expected: exit 0 (no new workflow row required; existing round-trip intact).

- [ ] **Step 5: Commit the full regenerated set**

```bash
git add scripts/capture_gui_tutorial_screenshots.py docs/source/_static/gui_images/
git status --short          # expect view_results/*, heatmap_exploration/*, font-render collateral
git commit -m "docs(gui): refresh tutorial screenshots for filter offcanvas (full set)"
```

(Per CLAUDE.md: commit the **full** regenerated set; do not cherry-pick or revert the font-render collateral.)

---

## Final Verification

- [ ] Logic + regression:
  `uv run pytest tests/unit/gui/results_viewer/test_filter_offcanvas.py tests/integration/gui/test_filter_offcanvas_layout.py tests/gui/results_viewer/test_filter_state.py tests/gui/results_viewer/test_filtered_state.py -q`
  Expected: all green.
- [ ] Broader viewer regression (no collateral breakage):
  `uv run pytest tests/unit/gui/results_viewer tests/gui/results_viewer -q`
  Expected: all green.
- [ ] E2E (local): `uv run pytest tests/e2e/gui/test_filter_offcanvas.py -q -m ci_flaky` — PASS.
- [ ] Lint/types: `uv run ruff check --fix` on changed files; `uv run mypy` shows no NEW errors on the changed files.
- [ ] Gates: `uv run python scripts/check_features_md.py` and `uv run python scripts/check_workflows_md.py` both exit 0.
- [ ] Manual: `uv run phenotypic-gui --root <output>` →
  - every tab (Plate, Colony, QC, Heatmap) is full-width; no left sidebar;
  - `Filters` button (with hidden-at-0 badge) sits in the top-bar right cluster;
  - click → panel slides in from the **right** with a dimming backdrop;
  - add a filter + choose a column → badge increments; choose a value → Plate picker / Colony grid narrow and the "N images match" chip updates inside the panel;
  - the bulk-paste popover opens leftward and stays fully on-screen;
  - backdrop / ✕ closes the panel; reopening shows the same rows (re-hydrated from `STORE_FILTER_SPEC`);
  - Heatmap + QC behavior unchanged.
- [ ] Acceptance (spec): full-width default on all tabs; filters in ≤1 click; active-filter count visible without opening; no filter/re-hydration regression; ledgers + filter tests green.

---

## Self-Review notes

- **Spec coverage:** mechanism/offcanvas-right (Task 3), boot-closed/no-store (Task 3, `is_open=False`), both-badges (Task 1/3 badge + untouched `FILTER_MATCH_COUNT_ID` chip), uniform-all-tabs (Task 3, single top-level offcanvas), bulk-paste left (Task 4), FEATURES.md (Task 6), screenshots/no-WORKFLOWS-row (Task 7), tab filtering preserved (Task 3 integration + Task 5 e2e; data layer regression in Task 4). All covered.
- **Type consistency:** helper names (`next_offcanvas_state`, `active_filter_count`, `badge_children`, `badge_style`, `register_filter_offcanvas_callbacks`) and ids (`OFFCANVAS_FILTER_ID`, `BTN_FILTERS_TOGGLE`, `FILTER_TOGGLE_BADGE_ID`) are used identically across Tasks 1–6.
- **Known adjustment point:** the E2E (Task 5) react-select option selection is the only flaky surface; the badge + match-count assertions are the durable contract.
```