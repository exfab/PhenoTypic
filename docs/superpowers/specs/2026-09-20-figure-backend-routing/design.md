# Explicit figure backends and non-silent plot failures

- **Date:** 2026-09-20
- **Branch:** `fix/figure-backend-routing` (worktree `.claude/worktrees/figure-backend-routing`, off `origin/main` @ `01a1def1`)
- **Status:** design approved, awaiting implementation plan
- **Origin:** the source trace at `docs/superpowers/artifacts/2026-09-20-pipeline-figure-storage/`
  and its independent audit at
  `docs/superpowers/reports/2026-09-20-pipeline-figure-storage/claim-verification.md`

## Objective

Make the rendering backend of a pipeline figure an explicit declaration rather than
an inference, and make a plot that fails to publish impossible to mistake for one
that succeeded.

These are one change, not two. The bug that started this — a matplotlib figure
returned from an `@figure` method dying inside a Plotly-only theming call — is only
dangerous because the failure it produces is swallowed on the ordinary CLI path and
leaves behind a manifest asserting success. Fixing the routing without fixing the
reporting would leave the second half of the trap in place.

## Non-goals

- **Removing backend sniffing from `FigureAdapter`.** It serves figures that never
  passed through `@figure` (an overridden `inspect()`, a direct writer call) — a
  legitimate, already-tested path. Recorded in `DEFERRED.md`.
- **A GUI builder surface for authoring plot bindings.** Filed as a feature request
  rather than built here.
- **Changing what a plot *is*.** No new lifecycle, no change to binding
  normalization, serialization, or the publication transaction.
- **Installing Chrome.** This spec makes its absence a recorded, announced
  capability limit rather than a failure — Plotly output still publishes, as HTML.
  It does not provision Chrome.
- **HTML output for matplotlib.** matplotlib has no native HTML export and mpld3 is
  not a dependency worth taking here. Matplotlib bindings publish PNG only.

## Background

`@figure` (`abc_/plotting/_pht_plot.py:131`) decorates a method as a figure builder
and records a `FigureSpec` on it. Its wrapper at `:192` calls
`apply_theme(fn(*args, **kwargs))` unconditionally, and `apply_theme`
(`sdk_/viz/figures/_theme.py:227`) sets `fig.layout.template` with no backend
guard. A matplotlib figure therefore dies with
`AttributeError: 'Figure' object has no attribute 'layout'` — an error that names
neither the decorator nor the backend.

The publication layer below the decorator does support both backends:
`FigureAdapter` (`plotting/_pipeline/_adapter.py`) dispatches on module string, and
a PhenoTypic matplotlib theme already exists — `phenotypic_rc()` /
`phenotypic_mpl_context()` in `sdk_/viz/figures/_mpl_theme.py`, exported from the
same package as `apply_theme` and carrying the `DESIGN.md` rcParams block.

So matplotlib support is real, and reachable today only by overriding `inspect()`
to bypass the decorator. Verified: all **27** `@figure` methods under `src/` return
Plotly, and the only matplotlib coverage
(`tests/unit/plotting/test_output_adapter.py`) constructs figures by hand and
pushes them straight through `publish_plot_output` / `to_dash_component`, never
through `@figure`.

**The two themes apply at different times.** `apply_theme` post-processes a
returned Plotly figure. `phenotypic_mpl_context` is an `rc_context` whose params
must be live *while the figure is being built*. Any implementation that treats
them symmetrically produces an unthemed matplotlib figure that looks like it
worked.

## Findings this spec closes

| # | Finding | Evidence | Section |
|---|---|---|---|
| F1 | `@figure` assumes Plotly; a matplotlib return raises an error naming neither the decorator nor the backend | `_pht_plot.py:192`, `_theme.py:227` | §1 |
| F2 | Missing Chrome makes every Plotly PNG export fail, discovered per-figure and late, leaving a plot directory that looks published but is empty | `plotly.io._kaleido` → `ChromeNotFoundError`; reproduced, 0.59 s to fail | §2 |
| F3 | `strict=True` on the staged GPU worker but not the two ordinary CLI sites: the same broken plot fails one run loudly and leaves the other green | `_cli_staged_workers.py:526` vs `_cli_process_single.py:348,450` | §3 |
| F4 | `emit_qc` dereferences `binding` in an `except` handler that can run before `binding` is assigned | `_coordinator.py:226` (try), `:239` (assign), `:261` (read) | §3 |
| F5 | Every page failing still writes a durable `manifest.json` asserting zero pages | `_writer.py:149` (`continue`) vs `:160-177` (unconditional write) | §3 |
| F6 | `custom_plotter.md` states the output directory is `plots/<ClassName>/` unconditionally | doc vs `_bindings.py:165-171` | §4 |

F4 and F5 came from the independent audit, not the original trace. F6 is not drift:
`git log` shows the doc line and `_bindings.py` last changed in the same commit
(`71737850`), so the doc was never accurate.

## §1 — `backend` on `@figure`

### Signature

```python
def figure(
    *,
    title: str,
    backend: Literal["plotly", "mpl"],      # required, no default
    section: str = "default",
    controls: dict[str, Control] | None = None,
    description: Any = None,
    primary: bool = False,
) -> ...
```

`FigureSpec` gains `backend: Literal["plotly", "mpl"]`.

**`backend` is required.** Omitting it is a `TypeError` at class-definition time.
This breaks every provider written against the current signature, in this repo and
outside it — accepted deliberately: a default would preserve exactly the
silent-default behaviour this change exists to remove, and the break is loud,
immediate, and one word to fix. (The example in `custom_plotter.md` is *not* among
the breakages; it overrides `inspect()` and never uses the decorator, which §4
treats as its own problem.)

### Wrapper

```python
@functools.wraps(fn)
def wrapper(*args, **kwargs):
    if backend == "plotly":
        from phenotypic.sdk_.viz.figures._theme import apply_theme
        fig = fn(*args, **kwargs)
        _require_backend(fig, "plotly", fn)
        return apply_theme(fig)

    from phenotypic.sdk_.viz.figures._mpl_theme import phenotypic_mpl_context
    with phenotypic_mpl_context():
        fig = fn(*args, **kwargs)
    _require_backend(fig, "mpl", fn)
    return fig
```

The asymmetry is the point, per Background: Plotly themes the result, matplotlib
themes the construction. The matplotlib branch applies no post-pass.

Both theme imports stay inside the wrapper body. `_pht_plot.py` documents itself as
importing only the standard library at runtime (`:3-6`) and that must hold — it is
enforced by `tests/unit/ci/test_deferred_imports.py`.

### Mismatch is an error, not a fallback

`_require_backend(figure, declared, fn)` raises `TypeError` naming the method, the
declared backend, and the actual type:

```
@figure('count_figure'): declared backend 'plotly' but the method returned
matplotlib.figure.Figure. Declare backend="mpl", or return a
plotly.graph_objects.Figure.
```

No coercion, no sniffed fallback. An unrecognised type raises the same error with
the actual module path, which also replaces the current failure mode for a method
that returns `None` or a numpy array.

### One shared predicate, not a sixth copy

The check needs the same two predicates `FigureAdapter._is_plotly` /
`_is_matplotlib` already implement. `_pht_plot.py` cannot import them:
`plotting/_pipeline/_adapter.py` is a private runtime module and `abc_/` importing
from it inverts the layering, besides breaking the stdlib-only rule.

Introduce `figure_backend_of(fig) -> Literal["plotly", "mpl"] | None` in
`abc_/plotting/_output.py` — already backend-neutral, already stdlib-only, already
the home of the `FigureLike` alias. `_require_backend` and `FigureAdapter`
(`_is_plotly`, `_is_matplotlib`, `backend_name`) both delegate to it. This removes
the duplication rather than adding to it, and keeps the one definition of "what
backend is this figure" in the one module both layers may import.

### `report()` with matplotlib figures

`_compose_control_free_figure` (`_pht_plot.py:439`) calls `make_subplots` and
iterates `rendered.data` — Plotly-only, and it would fail obscurely on a
matplotlib figure.

**Rule:** if any visible spec declares `backend="mpl"`, `PhtPlot.report()` raises
`TypeError` pointing the caller at `inspect()` or a plot-specific `report()`
override. Mixed providers raise for the same reason. This is a stated limitation,
not a silent one; composing matplotlib figures is out of scope.

`inspect()` is unaffected — it renders a single spec and returns whatever that
spec's backend produces.

## §2 — Two renderings, and a Chrome capability check

Plotly figures serialize to standalone HTML with no Kaleido and no Chrome in the
path. That changes what F2 is: a machine without Chrome is not a machine that
cannot publish plots, it is a machine that cannot publish *rasters*. So the fix is
not a louder failure, it is a second rendering that always succeeds.

**Plotly bindings publish HTML always, and PNG additionally when Chrome is
available. Matplotlib bindings publish PNG only** — matplotlib has no native HTML
export, and mpld3 is a dependency not worth taking for this.

### Why HTML is not a downgrade

Nothing consumes the published plot files programmatically. The only three
references in the tree are the GUI's own *writer* (`_gui/_plot_refresh.py:132`) and
two documentation strings describing the tree to the user
(`_cli_interactive.py:132`, `_cli_readme_generator.py:106`). `deliverables/plots/`
is a human-facing artifact, and for a human reading a 1536-colony scatter an
interactive figure with zoom, pan and hover is better than a flat raster. PNG was
never chosen over HTML; it is what `FigureAdapter` happened to implement.

### One shared `plotly.min.js`, hoisted

`include_plotlyjs` accepts a string that is emitted verbatim as the script `src`.
Measured behaviour:

| `include_plotlyjs` | Page size | Side effects |
|---|---|---|
| `True` | 4.9 MB | self-contained, unusable per image |
| `"cdn"` | 49 KB | needs network when viewed |
| `"directory"` | 49 KB | writes `plotly.min.js` beside the page, `src="plotly.min.js"` |
| `"../../plotly.min.js"` | 49 KB | writes nothing; `src` emitted verbatim |

`"directory"` is wrong here: a multi-page `PlotImage` gets one directory per image,
so it would write 4.8 MB of identical JS per image — gigabytes on a real run.

**Write the bundle once** at `<output>/deliverables/plots/plotly.min.js` from
`plotly.offline.get_plotlyjs()` (4,847,452 chars, verified), and give each page a
`src` computed with `os.path.relpath(js_path, page_dir)`. The three layouts resolve
to `../`, `../../` and `../../../` respectively, but the relpath call means no
layout knowledge is hard-coded.

Total JS cost per run: 4.8 MB, independent of image count and page count.

Write the bundle under the same `exclusive_path_lock` used for publication, skipping
it when already present with the expected size — concurrent SLURM workers otherwise
race on first creation. (Plotly's own `"directory"` mode does not rewrite an
existing file, which is the behaviour being reproduced.)

### The capability check

`chrome_available() -> bool` in `plotting/_pipeline/`, memoised per process:
`plotly.io.to_image(go.Figure(), format="png", width=8, height=8)` returning `True`
on success and `False` on `RuntimeError` / `ChromeNotFoundError`. Measured: **0.59 s
to fail** when Chrome is absent. The success-path cost is **not measured here** —
Chrome is not installed on this cluster — and the plan must measure it before the
probe is placed on a hot path; a first-launch cost of a few seconds multiplied
across a 2500-task array is not free.

Because of that, the probe runs in two places with different strategies:

- **Submitting process** — probed eagerly inside
  `_cli_validation.validate_pipeline` (`_cli_validation.py:21`), once, so the user
  learns *before* a long run whether they will get rasters.
- **Workers** — probed lazily. The first PNG write attempt *is* the probe: on the
  Chrome error it memoises `False`, records the reason once, and every later page
  in that process skips PNG without retrying. A working environment then pays
  nothing at all, because the first real figure was going to be rendered anyway.

`preflight_plot_backends` survives, but its only remaining hard failure is a
declared `mpl` backend with matplotlib not importable. Missing Chrome no longer
fails validation.

### Announcement

Once per process, at `WARNING`:

```
Chrome is not available; 3 Plotly plots will publish HTML only, without PNG:
sym, PlotDiagnostics, LogGrowthModel.
Install it for raster output with:  plotly_get_chrome
```

Naming the binding ids keeps it actionable. The manifest carries the same fact
durably (§3) so the announcement is not the only record — a log line nobody reads
is exactly the silence this spec exists to remove.

## §3 — Failures that look like failures

### F3 — one policy, both paths

Remove `strict=True` from `_cli_staged_workers.py:526`. All three `emit_image` call
sites are then uniformly best-effort, which is the correct policy for output that
is decorative relative to measurements: one bad figure must not kill a ten-hour
run that produced good data.

This is only safe because §2 has already moved the systemic cause — an unusable
backend — upstream of the run. What remains at emit time is a genuinely per-figure
accident.

### F4 — `emit_qc`'s handler

`emit_qc` opens its `try` at `_coordinator.py:226` and assigns `binding` at `:239`,
but the handler reads `binding.id` at `:261`. An exception in the prelude
(`:227-238`) produces, on the first iteration, an `UnboundLocalError` raised *from
inside the exception handler* — which replaces the original exception and escapes
`emit_qc`, making the one aggregate path that is documented best-effort not
best-effort at all. On a later iteration it is quieter and worse: `binding` still
holds the previous value, so the warning names the wrong plot.

**Fix:** narrow the `try` to begin at the `model_copy` (`:239`). The prelude is
pure dict and attribute access whose failure is a programming error, not a plot
failure; swallowing it was never intentional. The remaining handler can only run
with `binding` bound.

### F5 — the zero-page manifest

`publish_plot_output` skips `pages.append` on a per-page failure (`_writer.py:149`)
and writes the manifest unconditionally outside the loop (`:160-177`). Every page
failing therefore yields a durable `manifest.json` with `"pages": []` — a file that
asserts nothing was produced, with no record of why.

**Fix:** add a `"failed"` array beside `"pages"`, one entry per page that did not
publish: `{"key", "label", "error"}` where `error` is
`f"{type(exc).__name__}: {exc}"`.

A consumer looking at an empty plot directory now learns the reason from the
manifest without cross-referencing anything.

### Manifest shape, at `schema_version: 2`

§2 makes a page potentially two files, so the per-page `"file"` string becomes a
`"files"` map. That changes a field's meaning, so the version bumps:

```json
{
  "schema_version": 2,
  "plot_id": "sym",
  "class": "MeasureSymZones",
  "renderers": {"html": "available", "png": "unavailable: chrome not found"},
  "pages": [
    {"key": "count", "label": "Colony count", "backend": "plotly",
     "files": {"html": "Colony-count.html", "png": "Colony-count.png"},
     "metadata": {}}
  ],
  "failed": []
}
```

- `"renderers"` records the capability verdict **per published directory**, which
  is what makes a PNG-less plot directory self-explaining rather than merely empty.
  A matplotlib binding records `{"png": "available"}` and no `html` key.
- `"files"` omits `png` when Chrome was unavailable, and omits `html` for a
  matplotlib backend. A page with an empty `"files"` is a failure and belongs in
  `"failed"` instead.

The version bump is nearly free: **no production code reads the plot manifest.**
Its only consumers are `publish_plot_output`'s own return value and
`tests/unit/plotting/test_output_adapter.py`, which reads `manifest["pages"]`
(`:80`, `:106`) and the persisted `"backend"` set (`:86`). Those assertions are
updated in this change; nothing else in the tree touches it.

### The durable failure record

Every swallowed failure appends one JSON line to
`<output>/deliverables/plots/.failures.jsonl`:

```json
{"ts": "2026-09-20T18:04:11Z", "binding_id": "sym",
 "plot_class": "MeasureSymZones", "lifecycle": "image",
 "dataset": "plate_a", "image_stem": "A01",
 "error": "TypeError: ... "}
```

- Written from all five `except` blocks in `_coordinator.py` (`:107, :178, :260,
  :305, :329`) through one helper, plus the per-page failures in `_writer.py:140`.
- Path resolved by a new `plot_failures_jsonl_path(output_dir)` helper in
  `sdk_/_io_constants.py`. The project rule against hand-joined output names is not
  negotiable for a new artifact.
- `dataset` / `image_stem` present only for the image lifecycle.
- Concurrent SLURM workers serialise on `exclusive_path_lock`, consistent with
  `publish_plot_output`'s own directory lock. The file is opened `"a"` and one line
  written per acquisition.
- **Recording must never raise.** A failure in the failure recorder is caught and
  logged; it cannot escalate a best-effort plot failure into a run failure.

The record is the thing that makes "best-effort" honest: a green run with missing
plots now carries the reason on disk instead of only in a log the user does not
read.

## §4 — Call sites and documentation

**In scope, and sequenced last.** Nothing in this section is deferred. It lands as
explicit post-implementation phases of the implementation plan, after §1–§3 are
merged and green, for one reason: annotating 27 sites against a decorator whose
signature is still moving means annotating them twice. The required `backend`
argument must be final before the mechanical pass starts.

Staffing, to be carried into the plan:

| Work | Agent | Why |
|---|---|---|
| The 27 `@figure` annotations | Sonnet | Mechanical and verifiable — every site is Plotly today, and the guard is that the suite still imports. A site that would need `mpl` is a finding to escalate, not to annotate. |
| `abc_/CLAUDE.md` + doc sync | Sonnet | Follows the settled signature. |
| `docs/source/extending/pages/custom_plotter.md` | Opus | Not mechanical. It carries the three-case binding-id rule that the audit found stated wrongly, a runnable example that must import under the new signature, and the `report()` limitation — the page readers learn this system from. |

### The 27 declarations

Annotate every existing `@figure` with `backend="plotly"`:

| File | Sites |
|---|---|
| `_core/_image_parts/plot_accessor/_diagnostics_plotter.py` | 12 |
| `grid/_grid_fit_report.py` | 6 |
| `correction/_color_correction/_color_correction_report.py` | 4 |
| `measure/_measure_orientation_zones.py` | 3 |
| `_core/_image_parts/plot_accessor/_detect_modes_plotter.py` | 1 |
| `measure/_measure_symzones.py` | 1 |

All 27 are Plotly today; none changes behaviour. Any site whose annotation would
have to be `mpl` is a bug found by this change and must be reported, not silently
annotated to match.

Counts measured, not inherited: `grep -c '@figure('` returns 28 across `src/`, but
the 28th is an error-message string inside the decorator itself
(`_pht_plot.py:174`), not a decoration.

**A further 11 sites live in the test suite** and break identically:
`tests/unit/abc_/plotting/test_pht_plot.py` (9) and
`tests/unit/viz/test_notebook_adapter.py` (2). They are annotated in the same pass
— except where a test's subject *is* the missing argument, which is one of the new
tests in §5.

### `docs/source/extending/pages/custom_plotter.md`

- **The example does not break, and that is itself the problem.** Its
  `PlotColonyArea` overrides `inspect()` and never uses `@figure` (`:20-25`), so
  the page teaches the one authoring path that bypasses the decorator — and
  therefore bypasses theming, the backend declaration, and the preflight. Add a
  `@figure(backend="plotly")` example as the primary form, and keep the
  `inspect()` override as the documented escape hatch for multi-page and
  matplotlib output, labelled as such.
- Replace the `plots/<ClassName>/` claim with the actual rule, all three cases:
  - a slot-owned object with a key → the **slot key** (`meas={"sym": …}` →
    `plots/sym/`);
  - the singleton `model` slot, where `ref.key` is `None` by construction
    (`_image_pipeline_core.py:704`, and `PipelineObjectRef` forbids a key on
    `model`) → the **class name**;
  - an inline plot → the **class name**.
- State that `PlotOutput` pages land under
  `plots/<id>/<dataset>/<stem>-<hash>/`, and a single `"default"` page under
  `plots/<id>/<dataset>/<stem>-<hash>.{html,png}`.
- **Document the two renderings**: a Plotly binding publishes interactive `.html`
  always and `.png` when Chrome is available; a matplotlib binding publishes
  `.png` only. Say where `plotly.min.js` lives and that it is one file per run,
  so a user copying a plot directory elsewhere knows to bring it.
- Document `.failures.jsonl`, the manifest's `"renderers"` and `"failed"` keys,
  and the `report()` limitation on matplotlib providers.

### Other documentation

- `abc_/CLAUDE.md` — `@figure` convention gains the required `backend`.
- Root `CLAUDE.md` — no change; `plots` is not described there.

## §5 — Testing

New tests, each pinned to the finding it guards:

| Test | Guards |
|---|---|
| `@figure(title=...)` without `backend` raises `TypeError` | F1 — the break is intentional |
| `mpl` figure from a `backend="plotly"` method raises `TypeError` naming both | F1 |
| Plotly figure from a `backend="mpl"` method raises `TypeError` naming both | F1 |
| A `backend="mpl"` method asserts `matplotlib.rcParams["axes.prop_cycle"]` is themed **inside its own body**, and the returned figure is not post-processed | §1 — the only test that catches treating the two themes symmetrically |
| `report()` raises on an all-`mpl` and on a mixed provider | §1 |
| `figure_backend_of` agrees with `FigureAdapter.backend_name` across Plotly, matplotlib, and an unsupported object | §1 dedupe |
| A Plotly binding publishes `.html` with **no Chrome available**, and the run is green | F2 — the core inversion |
| With Chrome unavailable, the manifest records `renderers.png` as unavailable and `files` carries `html` only | F2 + §3 |
| An `mpl` binding publishes `.png` only, and records no `html` renderer | §2 matrix |
| `plotly.min.js` is written **once** per run at `deliverables/plots/`, and each page's `src` resolves to it from its own depth (aggregate, single-page image, multi-page image) | §2 — the gigabyte trap |
| A second page written into the same directory does not rewrite the bundle | §2 concurrency |
| `chrome_available()` probes once per process across repeated calls | §2 cost |
| Worker-side lazy probe: after the first Chrome failure, later pages skip PNG without re-attempting | §2 cost |
| `preflight_plot_backends` still raises for a declared `mpl` backend with matplotlib absent, and no longer raises for missing Chrome | §2 |
| `emit_qc` with a prelude that raises: remaining QC bindings still emit, and no `UnboundLocalError` escapes | F4 |
| A raising figure leaves the run green **and** writes exactly one `.failures.jsonl` line with the right binding id | F3 + record |
| A failure inside the recorder itself does not propagate | record |
| All pages failing yields a manifest with `"pages": []` and a populated `"failed"` | F5 |

Regression: `tests/unit/plotting/`, `tests/unit/abc_/plotting/`,
`tests/unit/gui/test_plot_refresh.py`, plus the modules holding the 27 annotated
sites. Use the **`run-phenotypic-test`** skill; the full suite is a Slurm job, not
an inline run.

## Blast radius

- **Breaking:** `@figure` without `backend` stops working. In-repo: 27 sites under
  `src/` and 11 under `tests/`, all updated here. Out of repo: any custom provider
  that uses the decorator. The published example is *not* affected — it overrides
  `inspect()` — which is exactly why §4 rewrites it. This is the accepted cost of
  the required-argument decision.
- **Behavioural:** a Plotly binding now publishes `.html` in addition to `.png`, and
  publishes `.html` *instead of* `.png` where Chrome is unavailable — where it
  previously produced nothing and said so only in a log. No run that succeeded
  before now fails; runs that silently produced empty plot directories now produce
  usable ones.
- **Output-shape:** `deliverables/plots/` gains `.html` pages and one
  `plotly.min.js` (4.8 MB) per run. Anything downstream that globs `plots/**/*.png`
  and assumes that is everything will now be seeing a subset — no such consumer
  exists in-tree, but a user's own script might.
- **Format:** the plot `manifest.json` moves to `schema_version: 2`; `"file"` becomes
  `"files"`. One test reads it; nothing in `src/` does.
- **Additive:** `.failures.jsonl`, the manifest `"failed"` and `"renderers"` keys,
  `figure_backend_of`, `chrome_available`, `plot_failures_jsonl_path`.
- **Unchanged:** binding normalization, serialization, the publication transaction,
  `FigureAdapter` dispatch, every lifecycle, every existing output path.

## Decisions on record

| Decision | Choice | Rationale |
|---|---|---|
| `backend` default | Required, no default | A default reproduces the silent-default behaviour being removed. The break is loud and one word to fix. |
| How far the declaration reaches | Decorator-local | `FigureAdapter` also serves figures that never met the decorator — a legitimate, tested path. Strict-everywhere deferred. |
| Failed-plot policy | Per-figure soft, recorded | A single bad figure is an accident and must not kill a long run; the durable record is what stops "soft" from meaning "silent". |
| Plotly output format | HTML always, PNG when Chrome is available | HTML needs no Chrome, so a missing browser stops being a failure. Nothing consumes these files programmatically, and an interactive figure is the better artifact for the human who does. |
| Matplotlib output format | PNG only | No native HTML export; mpld3 is not worth adding for this. |
| Where the format choice lives | Publication layer, not `@figure` | `backend` is a property of the figure — which library built it. Format is a property of the publication. Putting format on the decorator would fix a deployment choice at authoring time. |
| `plotly.min.js` placement | One hoisted copy per run | `include_plotlyjs="directory"` writes 4.8 MB per directory, and a multi-page image plot gets a directory per image — gigabytes. The string form takes a relative `src` and writes nothing. |
| Manifest `"failed"` + `"renderers"`, at v2 | In | Zero production readers, so the version bump is free; and a plot directory that explains its own missing PNGs is the whole point of §3. |
| GUI builder authoring | Out, filed as an issue | Largest item here and independent of everything else. |

## Out of scope

- `DEFERRED.md` in this folder — the strict-everywhere path for `FigureAdapter`.
- GitHub feature request — a builder surface for authoring plot bindings. Note the
  gap is specifically *authoring*: `_gui/analysis/_recipe_state.py` already reads,
  merges and re-writes plot-binding nodes into `pipeline.json` on every recipe save
  (`:69, 183-208, 597-660, 760-783`).
- Corrections to the source-trace artifact under
  `docs/superpowers/artifacts/2026-09-20-pipeline-figure-storage/` — five factual
  errors identified by the audit. Documentation of prior work, not part of this
  change.
