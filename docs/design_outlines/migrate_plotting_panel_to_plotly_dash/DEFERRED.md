# Deferred: visualization surfaces to converge onto `FigureProvider` later

This migration ([`design.md`](./design.md)) builds the reusable
`FigureProvider` + `@figure` protocol, migrates the four Panel dashboards, and
converts **one** operation `inspect()` as a pydantic proof
(`MeasureSymmetricZones`). Everything below **keeps working untouched** today —
these are *opt-in convergence candidates* for a later pass, not breakage.

The protocol is **additive**: a hand-written `.dash()`/`.inspect()` method that
returns a `go.Figure` is fully compatible. Converging a surface means
re-expressing its figure(s) as `@figure` methods so it gains uniform
`.inspect()`/`.dash()`/dashboard behavior and (where useful) `Control`s.

---

## A. Existing `.dash() -> go.Figure` methods (7)

Single-figure plotly views already following the repo convention. Converging
them is pure cleanup (uniform styling via `@figure`, optional `Control`s); none
are blocked or broken.

| File | Class / surface | Notes |
|---|---|---|
| `analysis/_replicate_agreement.py:212` | `ReplicateAgreement.dash()` | analysis node figure |
| `analysis/_expected_vs_detected.py:432` | `ExpectedVsDetectedCount.dash()` | QC/analysis node figure |
| `analysis/abc_/_set_analyzer.py:86` | `SetAnalyzer.dash()` (base) | base for analysis nodes; converge at the ABC |
| `analysis/abc_/_model_fitter.py:576` | `ModelFitter.dash()` (base) | growth-curve fitters |
| `_core/_image_parts/_image_handler.py:738` | `Image.dash()` | delegates to `rgb/gray.dash()` |
| `_core/.../accessor_abstracts/_multichannel_accessor.py:189` | `*.rgb.dash()` | channel accessor |
| `_core/.../accessor_abstracts/_single_channel_accessor.py:93` | `*.gray/detect_mat.dash()` | channel accessor |

Consumed by the GUI as `node.dash(**plot_kwargs)` (`gui/analysis/_render.py`,
`gui/results_viewer/_qc_tab/_callbacks.py`) — any conversion must preserve the
`-> go.Figure` return so those call sites keep working.

## B. Operations without an `inspect()` (add later)

Only `MeasureSymmetricZones` currently implements `inspect()` (the proof).
Other operation families are candidates to grow an `@figure inspect`:

- **Detectors** (`detect/`) — threshold/mask preview overlays.
- **Enhancers** (`enhance/`) — before/after `detect_mat` panels.
- **Refiners** (`refine/`) — mask-edit diffs.
- **Other measurers** (`measure/`) — per-feature diagnostic overlays.
- **Grid finders** (`grid/`) — beyond `AutoGridFinder.dashboard`.

## C. Other notes

- The `_opcache` per-instance cache (`tools_/_opcache.py`) is the established
  pattern for `inspect(image=None)` to reuse the last-measured image; reuse it
  when converging operations rather than adding model state.
- Section show/hide toggles are intentionally **not** `Control`s — they map to
  native collapsible cards (design.md D12). Apply the same rule when converging.

---

## Revisit criteria

Pick these up when: (a) a surface needs `Control`-driven recompute it can't get
from plotly-native interactivity, (b) a GUI page wants the multi-figure
dashboard form, or (c) a focused cleanup pass consolidates styling. Track each
conversion as its own small change; do **not** bundle with unrelated work.

---

## D. Scope reductions recorded post-review

These were shipped intentionally but were not originally logged here; the
review (REVIEW.md) flagged the design/code divergence. Each is additive to
converge later, none is a regression in numeric output.

- **Diagnostics `scales` control (design.md §9).** Listed as a diagnostics
  control, but `ridge_scales` is list-valued and does not map to any
  `Control` kind (float/select/bool/text). Shipped without it; the figure
  uses a fixed scale set. Converge if a multi-select `Control` kind is added.
- **Color-correction ROI selector (design.md §9, D11).** `ColorCorrectionReport`
  ships control-free (`.dash()` → composed `go.Figure`) rather than with an
  ROI `select` Control + shell. The ipywidgets path exists
  (`FigureProvider.dash` → `build_notebook_dashboard`) so this is a pure
  add-a-Control convergence when ROI-by-ROI recompute is wanted.
- **Grid axis-occupancy image-pitch overlay (grid `_grid_fit_report.py`).**
  `fig_axis_occupancy` shows fitted per-cell counts only; the old
  `_plot_axis_occupancy` overlaid image-pitch counts when fit and image-pitch
  disagreed. The disagreement is still surfaced numerically in the summary
  `go.Table` (`fit_occupied` vs `ip_occupied`); restoring the visual overlay
  needs `ip_counts`/`agree` plumbed into the per-axis stats dict.
- **Dash web-GUI figure adapter (was `gui/_figure_dashboard.py`).** Removed as
  a deferred stub. To restore: build a controls panel + `dbc.Accordion` of
  per-figure `dcc.Graph`s from `provider.iter_figures()` / `figures(subject)`,
  one Dash callback per control-bearing figure (controls deduped by identity,
  figures seeded on load). Mirror `gui/analysis/_render.py` (renders
  `node.dash() -> go.Figure` into `dcc.Graph`). Protocol:
  `phenotypic.abc_._figure_provider`; design: design.md §7.
