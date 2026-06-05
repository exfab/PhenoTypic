# Migrate & Refactor Plotting: Panel → Plotly (Dash + ipywidgets)

**Date:** 2026-06-04
**Branch/worktree:** `migrate-and-refactor-plotting`
**Status:** Design — pending review
**Companion:** [`DEFERRED.md`](./DEFERRED.md) — visualization surfaces that adopt
this protocol later.

---

## 1. Goal

Remove **Panel** (and its transitive HoloViz deps `param`, `bokeh`,
`jupyter-bokeh`), and replace every Panel-powered dashboard with a **shared
Plotly figure layer** plus two thin shells:

- **Notebook** — an `ipywidgets` shell (Jupyter-inline).
- **GUI** — a `Dash` shell (the existing web app).

In doing so, establish a **single reusable visualization protocol**
(`FigureProvider` + `@figure`) that works on both plain helper classes **and
pydantic `ImageOperation`s**, unifying the existing `.dash()` (interactive
plotly figure) and `.inspect()` (saveable diagnostic figure) conventions.

### Why Panel can go cleanly

- The production GUI (`src/phenotypic/gui/`) is **already** Dash/Plotly/Flask.
- Every Panel usage merely wraps **matplotlib** in `pn.pane.Matplotlib`; there
  is no Bokeh/HoloViews-native rendering. Panel is only a *shell*.
- `.dash() -> go.Figure` is already a **repo-wide convention** (`image.gray.dash()`,
  `check.dash()`, `analysis_node.dash()`, `model_fitter.dash()`).
- `plotly` + `kaleido` are already **core** deps; `ipywidgets` is already in
  the `gui` extra; `dash` is core. The shared layer needs **no new deps**.

---

## 2. The reframe: two kinds of "interactive"

This is the insight that lets one protocol serve `.dash()`, `.inspect()`, and
the slider dashboards:

1. **Plotly-native** — zoom, hover, legend-toggle layers. Needs **no shell**,
   just a `go.Figure`. This is what `.dash()` and `inspect()` already deliver
   (`inspect(for_save=True)` flattens legend-only layers for the static PNG).
2. **Recompute-on-change** — controls (sliders/dropdowns) that re-run
   computation. The **only** case that needs an ipywidgets/Dash recompute shell.

Therefore: **`Control`s are the only thing that requires a shell.** A `@figure`
with no `Control`s is just a `go.Figure`; a provider whose figures declare
`Control`s gets the shell.

| Surface | figures | controls? | `.dash()` returns |
|---|---|---|---|
| `image.gray.dash()` | 1 | no | `go.Figure` *(unchanged)* |
| `detector.inspect(image)` | 1 | no | `go.Figure` *(unchanged `--save-inspect`)* |
| grid-finder, detect_modes | many | **no** | composed subplot `go.Figure` (one `dcc.Graph`) |
| diagnostics, color-correction | many | **yes** | ipywidgets dashboard / Dash page |

---

## 3. Locked decisions

| # | Decision |
|---|----------|
| D1 | Plotly for both surfaces; builders return `go.Figure`, shared 1:1. |
| D2 | `FigureProvider` ABC mixin + `@figure` decorator. Plot methods live **on the class they visualize** (operation or plotter helper). |
| D3 | `Control`s are typed objects bound to method kwargs **by identity** — no global name namespace, no `sigma`-vs-`sigma` collisions. |
| D4 | `@figure` auto-applies the house Plotly template (prestyled). |
| D5 | Migrate **all four** Panel dashboards; remove Panel at the end. |
| D6 | Diagnostics renderer: **dual** — keep matplotlib `DiagnosticsPlotter` for static `image.plot.diagnostics()`/`--save-inspect`; add parallel Plotly `@figure` methods for the interactive dashboard. |
| D7 | Notebook API folds into `image.plot`. Interactive entry = the **`image.plot.dash.<name>()`** sub-namespace (`.dash` per the repo's "plotly view" convention). `image.panel` + `ImagePanelHandler` removed. |
| D8 | **Controls are the only thing that needs a shell.** Control-free providers (grid-finder, detect_modes) render as a composed subplot `go.Figure`, preserving the `.dash() -> go.Figure` contract and embedding in the GUI as one `dcc.Graph`. |
| D9 | `FigureProvider` is **subject-aware** and **pydantic-safe**: operations pass the subject at call time (`op.dash(image)`), helpers hold it; no transient state is added to a pydantic model (per-render cache lives on the bound view / existing `_opcache`). |
| D10 | `.inspect(subject=None, *, for_save=False)` → the **primary** saveable `go.Figure` (existing CLI contract). `.dash(subject=None)` → the interactive view (composed figure or ipywidgets dashboard). |
| D11 | Color-correction → a dedicated `ColorCorrectionReport(FigureProvider)` helper (mirrors `DiagnosticsPlotter`/`GridFitReport`); keeps `ColorCheckerProfile` a lean pydantic model. `corrector.dashboard()`/`profile.dashboard()` keep their names. |
| D12 | Section show/hide toggles become **native collapsible cards** in each adapter, **not** `Control`s. `Control` stays reserved for inputs that recompute a figure. |
| D13 | Operation/`inspect()` unification scope: **additive abstraction + one proof.** Convert `MeasureSymmetricZones.inspect` onto `@figure` (surfacing `base_layer` as a select `Control`, exercising the pydantic+control+shell path). All other surfaces → [`DEFERRED.md`](./DEFERRED.md). Hand-written `inspect()`/`dash()` methods keep working untouched. |

---

## 4. Architecture / layers

```
Image / Operation (the subject)
      │
      ▼  compute → plain data        (viz/data — mostly exists:
ImageMetricsCalculator, _run_timed_pipeline   ImageMetricsCalculator, TypedDicts)
      │
      ▼  data → go.Figure            (@figure methods ON the owning class/helper,
@figure-decorated methods             plotly-only, auto-styled; may take a subject)
      │
      ▼  introspected by             (abc_/: Control, FigureSpec, FigureProvider,
FigureProvider.iter_figures()          @figure)
      │
      ├── no Control anywhere ──► composed go.Figure  (.dash()/.inspect() direct)
      │
      └── Control present ──┬──────────────────┬────────────
                            ▼                  ▼
                  ipywidgets adapter      Dash adapter
                  viz/notebook/           gui/.../<dash>/
                            ▼                  ▼
                     Jupyter cell        Browser (dcc.Graph)
```

**Import rule** (lightweight import test enforces it):

```
viz/data/        → numpy, scipy          (no plotly, no UI)
@figure methods  → plotly + viz/data     (ON the owning class/helper;
   + viz/figures/_theme.py                 no dash, ipywidgets, panel)
abc_ (contract)  → stdlib dataclasses only
viz/notebook/    → + ipywidgets          ┐ only here may a UI toolkit
gui/.../adapter  → + dash                ┘ be imported (controls case only)
```

`@figure` builders are **methods on the class they visualize**
(`DiagnosticsPlotter`, `GridFitReport`, `DetectModesPlotter`,
`ColorCorrectionReport`, `MeasureSymmetricZones`). `viz/figures/` holds only the
shared `_theme.py`.

---

## 5. Contracts

### 5.1 `Control`

```python
@dataclass(frozen=True)
class Control:
    """Renderer-neutral input, bound to a figure method's kwarg BY IDENTITY.
    Same instance referenced by several @figure methods → one shared widget.
    Distinct instances (even same label) → independent widgets. No global
    name namespace."""
    label: str
    kind: Literal["float", "select", "bool", "text"]
    default: Any
    bounds: tuple[float, float] | None = None    # required: float
    step: float | None = None                    # optional: float
    options: tuple[Any, ...] | None = None        # required: select
    help: str | None = None
    # __post_init__ validates kind-specific requirements at construction.
```

| `kind` | notebook | Dash | value |
|---|---|---|---|
| `float` | `FloatSlider` | `dcc.Slider` | `float` |
| `select` | `Dropdown` | `dcc.Dropdown` | member of `options` |
| `bool` | `Checkbox` | `dbc.Switch` | `bool` |
| `text` | `Text` | `dcc.Input` | `str` |

### 5.2 `FigureSpec`

```python
@dataclass(frozen=True)
class FigureSpec:
    title: str                          # author
    section: str                        # author — flat grouping tag (collapsible card)
    controls: dict[str, Control]        # author — {method-kwarg: Control}
    description: PanelDescription | None # author — existing interpretive block
    primary: bool                       # author — marks the .inspect() figure
    name: str                           # decorator — method.__name__
    method: Callable[..., go.Figure]    # decorator — wrapped, auto-styled
    wants_subject: bool                 # decorator — signature has a subject param
    order: int                          # decorator — definition-order index
```

### 5.3 `@figure` + `FigureProvider`

```python
def figure(*, title, section="default", controls=None, description=None, primary=False):
    """Mark a method as a figure; auto-applies the house style. The method
    returns a raw go.Figure, accepts each control as a keyword arg, and MAY
    accept a subject as its first positional param (detected from the
    signature). `controls` keys are verified against the signature at import."""
    ...

class FigureProvider:
    """Mixin: turns @figure methods into .inspect()/.dash()/a dashboard.
    Methods only — no fields, no __init__, no instance state → pydantic-safe."""

    def _figure_subject(self):
        """Subject for subject-taking @figure methods. Helpers override to
        return held state; operations leave None (subject passed at call)."""
        return None

    def iter_figures(self) -> list[FigureSpec]: ...          # sorted by .order

    def inspect(self, subject=None, *, for_save=False, **overrides) -> go.Figure:
        """Primary saveable go.Figure (the primary=True figure, or the sole one).
        IS the existing inspect() contract; --save-inspect passes for_save=True."""
        ...

    def dash(self, subject=None):
        """Interactive view. No Controls anywhere → a composed go.Figure
        (preserves the repo-wide .dash()->go.Figure contract). Controls present
        → the ipywidgets dashboard."""
        ...

    def figures(self, subject=None) -> "BoundFigures":
        """Bind a subject → transient renderable the GUI Dash adapter consumes.
        Per-render cache lives HERE, not on the provider (pydantic stays clean)."""
        ...
```

**Correctness traps in the contract:** `iter_figures()` sorts by `order`
(`dir()` is alphabetical); adapter closures bind loop vars as default args
(`c=ctrl`, `_spec=spec`); Dash seeds figures (`prevent_initial_call=False`).

---

## 6. Subject binding (pydantic operations vs helpers)

Two `@figure` signature shapes, detected by the decorator (`wants_subject`):

```python
# pydantic OPERATION — subject passed at call (matches inspect(self, image=None, ...))
class MeasureSymmetricZones(MeasureFeatures):          # pydantic BaseModel
    @figure(title="Symmetric-radius overlay", primary=True,
            controls={"base_layer": BASE_LAYER})       # select control → shell path
    def inspect(self, image=None, *, base_layer="gray", for_save=False) -> go.Figure:
        img = image if image is not None else self._cached_image()   # existing _opcache
        ...
    # .inspect(image)  → static primary figure (for_save flattens layers) — unchanged CLI
    # .dash(image)     → ipywidgets dashboard with a base_layer dropdown

# HELPER — holds its subject
class DiagnosticsPlotter(BasePlotter, FigureProvider):
    def _figure_subject(self): return self._root_image
    @figure(title="Ridge Response", section="structure",
            controls={"sigma": SIGMA, "method": METHOD})
    def plot_ridge(self, *, sigma, method) -> go.Figure:   # no subject param → reads self
        ...
```

**Pydantic-safety:** `FigureProvider` adds only methods (no annotations → no
fields; `model_json_schema()` unchanged). `Control`s are decorator args /
module constants, never class annotations. Transient cache lives on
`BoundFigures` / the existing `_opcache` per-instance cache (`image=None`
reuses the last-measured image) — **nothing transient on the model**. May sit
on `BaseOperation` so any operation can light up `inspect()`/`dash()` by
declaring `@figure` methods; existing hand-written `inspect()`/`dash()` methods
returning `go.Figure` keep working.

---

## 7. Adapters

- **Notebook** (`viz/notebook/_adapter.py`) — invoked only when controls exist.
  Dedups controls **by `id()`**; a control change re-renders exactly the figures
  referencing that instance; each called with its own local kwargs (+ injected
  subject when `wants_subject`). Sections → collapsible `Accordion` cards (D12).
- **Dash** (`gui/.../<dash>/_adapter.py`) — `layout()` + `register_callbacks()`
  consuming `iter_figures()`; same `id()`-dedup; `dbc.Accordion` sections.
- **Control-free** providers skip both shells: `dash()` composes the figures
  into one subplot `go.Figure`.

Both shells funnel into `getattr(provider, spec.name)(*subject?, **kwargs)`.

---

## 8. Theme

`viz/figures/_theme.py` centralizes today's scattered tokens (`_OI_NAVY`,
`_dashboard_rcparams()`, DM Sans) into one Plotly `Template` registered as
`"phenotypic"`, applied by `apply_theme()` inside `@figure`. Aligned with
`DESIGN.md`.

---

## 9. Per-dashboard migration

| Dashboard | Owner | Controls | Shell? | Notes |
|---|---|---|---|---|
| diagnostics | `DiagnosticsPlotter` (+`FigureProvider`) | sigma, ridge_method, bg_sigma, scales | yes | **Dual** (D6): keep matplotlib `_plot_*`/static `diagnostics()`; add ~11 Plotly `@figure` methods reusing `ImageMetricsCalculator`/`PanelDescription`. |
| detect_modes | new `DetectModesPlotter` | none | no | One faceted `go.Figure` (subplot per detection mode). Plotly-only. |
| grid-finder | new transient `GridFitReport` | none | no | 6 `@figure` methods → composed subplot figure; markdown summary → `go.Table`. `finder.dashboard(image)` → `report.dash()`. |
| color-correction | `ColorCorrectionReport` (was `ColorCorrectionDashboard`) | ROI selector (select) | yes | Drops `param.Parameterized`; `show_*` toggles → collapsible cards (D12). |
| **proof:** symmetric-zones inspect | `MeasureSymmetricZones` | base_layer (select) | yes | The D13 pydantic proof. |

---

## 10. Access / API changes

### Removed
`image.panel` accessor; `ImagePanelHandler` MRO layer (relink
`ImagePlotHandler → ImageColorSpace`); `accessors/_panel_accessor.py`;
`_core/_image_parts/panel_accessor/`; `tools_/panel_.py`;
`tools_/register/_dashboard_registry.py`. Update `_core/CLAUDE.md` (drop
`image.panel`, fix MRO diagram).

### Notebook entry (D7)
- Static, unchanged: `image.plot.diagnostics()` → `(fig, metrics)` matplotlib.
- Interactive: `image.plot.dash.<name>()` — a small dispatch sub-accessor under
  `plot`, mirroring the `@register_plotter` registry, returning each provider's
  `.dash()`:
  - `image.plot.dash.diagnostics()` (ipywidgets dashboard — has controls)
  - `image.plot.dash.detect_modes()` (composed faceted `go.Figure` — control-free)

### Operations
`finder.dashboard(image)` / `corrector.dashboard()` / `profile.dashboard()` keep
names, return ipywidgets (or a composed figure when control-free).
`MeasureSymmetricZones.inspect(image)` / `.dash(image)` per §6.

---

## 11. Dependencies

Remove from the `gui` extra: `panel`, `param`, `bokeh`, `jupyter-bokeh`. No
additions. `uv lock` after.

---

## 12. Testing

- **Figure builders:** unit-test trace types/counts/key data; deterministic, no browser.
- **`@figure`/`FigureProvider`:** `iter_figures()` order; `id()`-dedup of controls;
  signature/`controls`-keys validation; `wants_subject` detection; a decorated
  method called directly returns a prestyled figure; `inspect()` picks the
  primary; pydantic `model_json_schema()` unaffected by the mixin.
- **Notebook adapter:** extract the render loop into a module-level helper; unit-test
  value→kwarg mapping + dependency-driven re-render set.
- **Dash adapter:** live browser (Playwright MCP) — callback wiring only fires on
  `/_dash-update-component`. Add to GUI E2E.
- **Proof:** `MeasureSymmetricZones.inspect(for_save=True)` still flattens layers and
  `--save-inspect` still writes a PNG (kaleido); `.dash(image)` base_layer dropdown
  recomputes.
- **Doctests** runnable via `load_synth_yeast_plate()`.

---

## 13. Docs / CI

- Delete `phenotypic.tools_.panel_*.rst` (7); update `visualization_plotting.rst`,
  `component_registry.md`.
- Update 2 notebooks referencing `image.panel`; the image-quality explanation page.
- GUI ledgers (CI-gated): `FEATURES.md` for the four Dash surfaces; `WORKFLOWS.md`
  + `_capture_<id>` + tutorial page + regenerated screenshots for new flows.

---

## 14. Non-goals

- Converting `image.plot.*` simple raster displays (`rgb/gray.show()`) to Plotly.
- Converting `napari` viewers.
- Converting the 7 existing `.dash()` methods or adding `inspect()` to other
  operations now → [`DEFERRED.md`](./DEFERRED.md).
- A declarative layout/section-nesting engine (flat `section` only).

---

## 15. Sequencing

1. **Foundation:** `abc_/` `Control`/`FigureSpec`/`@figure`/`FigureProvider`
   (subject-aware, `.inspect()`/`.dash()`/`figures()`); `viz/figures/_theme.py`;
   `viz/notebook/` adapter. Unit tests.
2. **Diagnostics (pilot, dual, controls):** Plotly `@figure` on `DiagnosticsPlotter`;
   `image.plot.dash.diagnostics()`; notebook render. Hardest interactive case →
   validates controls + identity model.
3. **Pydantic proof:** `MeasureSymmetricZones.inspect` onto `@figure` with the
   `base_layer` control; verify `--save-inspect` + `.dash(image)`.
4. **Dash adapter + GUI diagnostics page:** same provider in Dash; FEATURES/WORKFLOWS.
5. **Remaining dashboards:** detect_modes (faceted, control-free), grid-finder
   (`GridFitReport`, control-free), color-correction (`ColorCorrectionReport`).
6. **Remove Panel:** delete accessor/handler/packages/`panel_.py`/dashboard registry;
   relink MRO; drop deps; docs + notebooks; `uv lock`.
7. **Verify:** full suite + GUI E2E + screenshot regen + a code-simplifier pass.

---

## 16. Remaining open questions

None blocking. Minor, resolved with stated defaults:
- `image.plot.dash` is a dispatch sub-accessor mirroring `@register_plotter`.
- `ColorCorrectionReport` is the home for color-correction figures (D11).
- The pydantic proof uses `MeasureSymmetricZones` because it is the **only**
  existing `inspect()` and already returns plotly with a parameter that becomes
  a `Control`.
