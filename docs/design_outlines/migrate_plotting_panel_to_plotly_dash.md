# Migrate & Refactor Plotting: Panel → Plotly + Dash/ipywidgets

**Date:** 2026-06-04
**Branch/worktree:** `migrate-and-refactor-plotting`
**Status:** Design — pending review

---

## 1. Goal

Remove **Panel** (and its transitive HoloViz deps `param`, `bokeh`,
`jupyter-bokeh`) as a dependency, and re-home every interactive dashboard it
currently powers onto a **shared Plotly figure layer** consumed by two thin
shells:

- **Notebook** — an `ipywidgets` shell (Jupyter-inline).
- **GUI** — a `Dash` shell (the existing web app, already Dash/Flask/Plotly).

The same `data -> go.Figure` builder code is reused **1:1** across both shells;
only the control/layout shell diverges. Plot logic lives **on the class it
visualizes** (or its plotter helper), surfaced through an ABC mixin +
decorator, never in a free-floating registry of lambdas.

### Why Panel can go cleanly

- The production GUI (`src/phenotypic/gui/`) is **already** Dash/Plotly/Flask —
  zero Panel imports there.
- Every current Panel usage merely wraps **matplotlib** figures in
  `pn.pane.Matplotlib`; there is no Bokeh/HoloViews-native rendering anywhere.
  Panel is only a *shell* (layout + widgets + reactivity + Jupyter display).
- `plotly` is already a **core** dependency (`pyproject.toml`), and `kaleido`
  (static export) is too. `ipywidgets` is already in the `gui` extra; `dash`
  is core. So the shared layer needs **no new dependencies**.

---

## 2. Locked decisions

| # | Decision |
|---|----------|
| D1 | **Renderer:** Plotly for both surfaces. Plot builders return `go.Figure`, shared 1:1 between the ipywidgets and Dash shells. |
| D2 | **Mechanism:** an ABC mixin `FigureProvider` + a `@figure` method decorator. Plot methods stay on the data-owning class or its plotter helper. |
| D3 | **Controls:** typed `Control` objects bound to a method's keyword args. Binding is by **object identity**, not by string name — sharing one widget = referencing the same instance; there is **no global control namespace**, so a `sigma` in one figure can never collide with a `sigma` in another. |
| D4 | **Auto-styling:** `@figure` applies the house Plotly template automatically (the "prestyled" path). Authors return a raw `go.Figure`. |
| D5 | **Scope:** migrate **all four** Panel dashboards in this effort — `diagnostics`, `detect_modes`, grid-finder, color-correction — and remove Panel at the end. |
| D6 | **Diagnostics renderer reconciliation:** **dual.** Keep the existing matplotlib `DiagnosticsPlotter` backing the static `image.plot.diagnostics()` / CLI `--save-inspect` path; **add** parallel Plotly `@figure` methods for the interactive dashboard. |
| D7 | **Notebook API:** fold the image-level dashboards into `image.plot.*` (see §6). The `image.panel` accessor and `ImagePanelHandler` MRO layer are removed. |

### Defaulted (stated for the record; flag in review if wrong)

- `FigureProvider`, `@figure`, `Control`, `FigureSpec` live under `abc_/`
  (next to the operation ABCs); the Plotly theme lives in
  `viz/figures/_theme.py` aligned with `DESIGN.md`.
- **Two entry points**, not a unified `render(backend=)`: notebook =
  `provider.to_ipywidget()`; GUI = a Dash adapter module consuming
  `provider.iter_figures()` (it does `layout()` + `register_callbacks()`,
  which does not collapse into one call).
- `detect_modes`' variable-N thumbnails = **one** `@figure` returning a
  faceted `go.Figure` (subplot grid, one cell per detection mode), not N panels.
- Static export / `--save-inspect` uses Plotly + `kaleido` (`fig.write_image`).
- Simple raster displays (`rgb/gray/objmap.show()`) stay matplotlib — untouched.

---

## 3. Architecture

Four layers. Everything left of `go.Figure` is shared and Panel-free; only the
final hand-off splits by surface.

```
Image / Operation
      │
      ▼  compute → plain data        (viz/data — mostly already exists:
ImageMetricsCalculator, _run_timed_pipeline   ImageMetricsCalculator, TypedDicts)
      │
      ▼  data → go.Figure            (@figure methods ON the owning class/helper,
plot_* methods (Plotly, auto-styled)  plotly-only, never import dash/ipywidgets)
      │
      ▼  introspected by             (abc_/: Control, FigureSpec, FigureProvider)
FigureProvider.iter_figures()
      │
      ├───────────────┬──────────────────────
      ▼               ▼
ipywidgets adapter   Dash adapter           (the ONLY two modules that know a
viz/notebook/        gui/.../<dash>/         UI toolkit exists)
      ▼               ▼
 Jupyter cell      Browser (dcc.Graph)
```

**Import rule (enforced by review / a lightweight import test):**

```
viz/data/        → numpy, scipy          (no plotly, no UI)
@figure methods  → plotly + viz/data     (live ON the owning class/helper;
   + viz/figures/_theme.py                 no dash, ipywidgets, panel)
abc_ (spec)      → stdlib dataclasses only
viz/notebook/    → + ipywidgets          ┐ only here may a UI toolkit
gui/.../adapter  → + dash                ┘ be imported
```

> **Module layout:** plot builders are `@figure`-decorated **methods on the
> class they visualize** (`DiagnosticsPlotter`, `GridFitReport`,
> `DetectModesPlotter`, the color-correction provider). `viz/figures/` holds
> only the shared `_theme.py` (and any genuinely standalone shared builders);
> it is **not** a registry of all plots. `abc_/` holds the `Control` /
> `FigureSpec` / `FigureProvider` / `@figure` contract; `viz/notebook/` and the
> GUI Dash module hold the two adapters.

### 3.1 `Control`

```python
@dataclass(frozen=True)
class Control:
    """A renderer-neutral input descriptor, bound to a figure method's kwarg.

    IDENTITY is the contract: reference the SAME instance from several @figure
    methods to share ONE widget across them. Distinct instances — even with the
    same label — render independently. There is no global name namespace.
    """
    label: str                                   # widget caption (display only)
    kind: Literal["float", "select", "bool", "text"]
    default: Any                                 # type must match kind; initial value
    bounds: tuple[float, float] | None = None    # required for float
    step: float | None = None                    # optional float increment
    options: tuple[Any, ...] | None = None       # required for select
    help: str | None = None                      # optional tooltip / doc text

    def __post_init__(self) -> None:
        # validate: bounds present for float, options non-empty for select,
        # default consistent with kind/options. Fail at construction.
```

`kind` → widget/value mapping:

| `kind` | notebook | Dash | value type |
|--------|----------|------|------------|
| `float`  | `FloatSlider` | `dcc.Slider` | `float` |
| `select` | `Dropdown`    | `dcc.Dropdown` | member of `options` |
| `bool`   | `Checkbox`    | `dbc.Switch` | `bool` |
| `text`   | `Text`        | `dcc.Input` | `str` |

These four cover every widget the current Panel dashboards use
(`FloatSlider`, `Select`, `Boolean`, `TextInput`).

### 3.2 `FigureSpec`

Author supplies `title`/`section`/`controls`/`description`; the decorator fills
`name`/`method`.

```python
@dataclass(frozen=True)
class FigureSpec:
    title: str                          # author — card header + figure title
    section: str                        # author — flat grouping tag (cards/columns)
    controls: dict[str, Control]        # author — {method-kwarg-name: Control}
    description: PanelDescription | None # author — existing interpretive block
    name: str                           # decorator — method.__name__
    method: Callable[..., go.Figure]    # decorator — wrapped, auto-styled callable
    order: int                          # decorator — creation index (def order)
```

`section` is a **flat string**, not a layout tree — deliberately short of a
"full dashboard spec." Genuine custom layout drops to the escape hatch
(call `iter_figures()` and arrange by hand).

### 3.3 `@figure` decorator + `FigureProvider` mixin

```python
def figure(*, title, section="default", controls=None, description=None):
    """Mark a method as a dashboard figure; auto-applies the house style.

    The method returns a raw go.Figure and accepts each control as a keyword
    arg (keys of `controls`). Called directly it still returns a *prestyled*
    figure, so it doubles as a standalone plot. `controls` keys are verified
    against the method signature at definition time (loud failure on mismatch).
    """
    def deco(method):
        @functools.wraps(method)
        def wrapper(self, **kwargs) -> go.Figure:
            return apply_theme(method(self, **kwargs))          # prestyled
        wrapper.__figure_spec__ = FigureSpec(
            title=title, section=section, controls=controls or {},
            description=description, name=method.__name__, method=wrapper,
            order=_next_figure_index(),                          # def-order, not dir()
        )
        return wrapper
    return deco


class FigureProvider:
    """Mixin: turns @figure methods into a dashboard / ipywidget."""
    def iter_figures(self) -> list[FigureSpec]:
        specs = [s for n in dir(type(self))
                 if (s := getattr(getattr(type(self), n, None), "__figure_spec__", None))]
        return sorted(specs, key=lambda s: s.order)             # definition order

    def to_ipywidget(self):
        from phenotypic.viz.notebook import render_provider
        return render_provider(self)
```

**Correctness traps captured in the contract:**

- `dir()` is alphabetical → `iter_figures()` sorts by `order` (a monotonic
  decorator counter) to preserve **definition order**.
- Notebook `.observe` closures and Dash callback closures must bind loop vars
  as default args (`c=ctrl`, `_panel=p`) or every closure captures the last
  value (a known prior bug class in this repo).
- Dash adapter seeds figures (`prevent_initial_call=False` or layout-time
  render) so graphs are not blank until first interaction.

### 3.4 Notebook adapter (collision-proofing)

```python
# phenotypic/viz/notebook/_adapter.py
def render_provider(provider: FigureProvider):
    specs = provider.iter_figures()
    widgets = {}                                   # id(Control) -> (Control, widget)
    for spec in specs:
        for ctrl in spec.controls.values():
            widgets.setdefault(id(ctrl), (ctrl, _to_ipywidget(ctrl)))   # dedup by IDENTITY
    outputs = {s.name: W.Output() for s in specs}

    def render(spec):
        kwargs = {kw: widgets[id(c)][1].value for kw, c in spec.controls.items()}
        fig = getattr(provider, spec.name)(**kwargs)            # bound, already styled
        with outputs[spec.name]:
            outputs[spec.name].clear_output(wait=True); fig.show()

    for _id, (ctrl, w) in widgets.items():
        w.observe(lambda _e, c=ctrl: [render(s) for s in specs
                                      if c in s.controls.values()], "value")
    for s in specs: render(s)
    return _stack_by_section(specs, widgets, outputs)
```

The dedup key is `id(ctrl)` — that is the entire fix for the param-namespace
problem. A control change re-renders exactly the figures referencing that
instance, each called with its own local kwargs.

### 3.5 Dash adapter

```python
# gui/.../<dashboard>/_adapter.py
def layout(provider) -> html.Div:
    specs = provider.iter_figures()
    controls = _dedup_by_identity(specs)                       # one component per Control
    return html.Div([_sidebar(controls), _graphs(specs)])      # dcc.Graph per spec

def register_callbacks(app, provider, ...):
    for spec in provider.iter_figures():
        inputs = [Input(_cid(c), "value") for c in spec.controls.values()]
        @app.callback(Output(_gid(spec), "figure"), *inputs, prevent_initial_call=False)
        def _cb(*vals, _spec=spec):
            kwargs = dict(zip(_spec.controls, vals))
            return getattr(provider, _spec.name)(**kwargs)     # SAME shared call
```

Both adapters funnel into `getattr(provider, spec.name)(**kwargs)`.

---

## 4. The Plotly theme

Centralizes today's scattered style tokens (`_OI_NAVY`, `_dashboard_rcparams()`,
DM Sans) into one Plotly template, aligned with `DESIGN.md`.

```python
# phenotypic/viz/figures/_theme.py
OI_NAVY, OI_VERMILION, OI_GREY = "#0b1f3a", "#d55e00", "#8892a4"
PHENO_TEMPLATE = go.layout.Template(layout=dict(
    font=dict(family="DM Sans, sans-serif", color="#2e3a4e"),
    colorway=[OI_NAVY, OI_VERMILION, "#009e73", "#cc79a7"],
    margin=dict(l=50, r=20, t=40, b=40), plot_bgcolor="white"))
pio.templates["phenotypic"] = PHENO_TEMPLATE

def apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(template="phenotypic")
    return fig
```

---

## 5. Per-dashboard migration

| Dashboard | Owner class / helper | Controls | Notes |
|-----------|----------------------|----------|-------|
| **diagnostics** | `DiagnosticsPlotter` (under `image.plot`) gains `FigureProvider` + Plotly `@figure` methods | `sigma`, `ridge_method`, `background_sigma`, `ridge_scales`, section toggles | **Dual** (D6): keep its matplotlib `_plot_*` + static `diagnostics()`; add ~11 Plotly `@figure` methods reusing `ImageMetricsCalculator` + `PanelDescription`. |
| **detect_modes** | new `DetectModesPlotter` (FigureProvider, registered via `@register_plotter`) | none | One faceted `go.Figure` (subplot per detection mode) + an optional selector for the large view. No prior matplotlib static plotter — Plotly-only. |
| **grid-finder** | new transient `GridFitReport` (FigureProvider) holding one `_run_timed_pipeline` result | none (static) | `AutoGridFinder.dashboard(image)` builds the report and returns `report.to_ipywidget()`. The 6 `pn.pane.Matplotlib` panels → 6 `@figure` methods; the markdown summary → `go.Table`. |
| **color-correction** | `ColorCorrectionDashboard` reworked as a FigureProvider (or methods on `ColorCheckerProfile`) | `show_pipeline/segmentation/delta_e/patches` (bool), ROI selector (select) | `ColorCorrector.dashboard()` / `profile.dashboard()` keep their names, now return ipywidgets. |

Grid-finder and color-correction dashboards hang off operation classes (not
`Image`), so their public entry methods (`finder.dashboard()`,
`corrector.dashboard()`, `profile.dashboard()`) **keep their names** and just
change what they return (ipywidgets instead of a Panel layout). The GUI imports
the same providers into Dash adapters.

---

## 6. Access / API changes

### Removed
- `image.panel` accessor; `ImagePanelHandler` MRO layer; `accessors/_panel_accessor.py`;
  `_core/_image_parts/panel_accessor/` package; `tools_/panel_.py`;
  `tools_/register/_dashboard_registry.py` (the `@register_dashboard` mechanism —
  superseded by `@figure` + the plotter registry).
- MRO relinks: `ImagePlotHandler → ImageColorSpace` directly (drop the panel layer).
- Update `_core/CLAUDE.md` accessor list + MRO diagram (remove `image.panel`).

### Notebook entry (D7 — fold into `image.plot`)
- **Static, unchanged:** `image.plot.diagnostics()` → `(fig, metrics)` matplotlib (dual).
- **Interactive:** add a `dashboard` sub-namespace on the plot accessor that
  returns ipywidgets via the providers' `to_ipywidget()`:
  - `image.plot.dashboard.diagnostics()`
  - `image.plot.dashboard.detect_modes()`
  - (optionally `image.plot.detect_modes()` static faceted figure — same builder)

  > **Micro-OQ for review:** `image.plot.dashboard.<name>()` (a small parallel
  > dispatch accessor mirroring the plotter registry) vs. suffixed methods
  > `image.plot.diagnostics_dashboard()`. Recommendation: the `dashboard`
  > sub-namespace — it groups cleanly and scales to detect_modes without
  > polluting the static method namespace.

---

## 7. Dependency changes (`pyproject.toml`)

Remove from the `gui` extra: `panel`, `param`, `bokeh`, `jupyter-bokeh`
(the `# Panel GUI components` block). Keep `ipywidgets` (now the notebook shell),
`napari`/`jupyter`/`pyvips` (unrelated). No additions — `plotly`, `kaleido`,
`dash`, `dash-bootstrap-components` already present.

---

## 8. Testing strategy

- **Figure builders** (pure `data -> go.Figure`): unit-test trace types, counts,
  and key data (e.g. histogram has N bins, ridge has the optimal-scale vline).
  Cheap and deterministic; no browser.
- **`FigureProvider` / `@figure`:** unit-test `iter_figures()` order, the
  identity-dedup of controls, signature/`controls`-keys validation, and that a
  decorated method called directly returns a prestyled figure.
- **Notebook adapter:** extract the render loop into a module-level helper and
  unit-test value→kwarg mapping and the dependency-driven re-render set, per the
  repo lesson to make callback bodies unit-testable.
- **Dash adapter:** drive in a live browser (Playwright MCP) — wiring bugs only
  fire on `/_dash-update-component`. Add to the GUI E2E suite.
- **Doctests:** every `@figure` method's docstring example runnable via
  `load_synth_yeast_plate()` (project rule).
- **Regression:** the existing matplotlib `image.plot.diagnostics()` tests stay
  green (dual renderer untouched).

---

## 9. Docs / CI impacts

- Delete `docs/.../phenotypic.tools_.panel_*.rst` (7 files); update
  `visualization_plotting.rst`, `component_registry.md`.
- Update the 2 notebooks referencing `image.panel`
  (`09_diagnosing_image_quality.ipynb`, `assess_image_quality.ipynb`) and the
  `image_quality_noise_contrast_structure.md` explanation.
- **GUI ledgers (CI-gated):** any `src/phenotypic/gui/` change requires
  `FEATURES.md` updates; new end-to-end flows require `WORKFLOWS.md` +
  `_capture_<id>` + a tutorial page + regenerated screenshots
  (`scripts/capture_gui_tutorial_screenshots.py`, commit the full set).
- Add the four Dash dashboard surfaces to `FEATURES.md`.

---

## 10. Non-goals

- Converting `image.plot.*` simple raster displays to Plotly (stay matplotlib).
- Converting `napari` viewers.
- Adding new diagnostics/metrics — pure render/shell migration.
- A declarative layout/section-nesting engine (explicitly out; flat `section`).

---

## 11. Sequencing (phases)

1. **Foundation:** `abc_/` `Control`/`FigureSpec`/`@figure`/`FigureProvider`
   + `viz/figures/_theme.py` + notebook adapter `viz/notebook/`. Unit tests.
2. **Diagnostics (pilot, dual):** Plotly `@figure` methods on `DiagnosticsPlotter`;
   `image.plot.dashboard.diagnostics()`; notebook render. Validates both the
   mixin and the shared-control identity model on the hardest interactive case.
3. **Dash adapter + GUI diagnostics page:** prove the same providers render in
   Dash; FEATURES.md/WORKFLOWS.md round-trip.
4. **Remaining three:** detect_modes (faceted), grid-finder (`GridFitReport`),
   color-correction.
5. **Remove Panel:** delete `image.panel`/handler/packages/`panel_.py`/dashboard
   registry; relink MRO; drop deps from `pyproject`; docs + notebooks; `uv lock`.
6. **Verification:** full test suite + GUI E2E + screenshot regen + a code
   simplifier pass.

---

## 12. Remaining open questions

1. **§6 micro-OQ:** `image.plot.dashboard.<name>()` sub-namespace vs.
   `image.plot.<name>_dashboard()` suffixed methods. (Rec: sub-namespace.)
2. **Color-correction home:** keep the `ColorCorrectionDashboard` class as the
   FigureProvider, or move the `@figure` methods onto `ColorCheckerProfile`
   directly? (Rec: a dedicated provider helper, mirroring `DiagnosticsPlotter`,
   to keep the pydantic profile model lean.)
