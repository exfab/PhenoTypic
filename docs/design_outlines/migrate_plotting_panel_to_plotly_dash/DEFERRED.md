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
