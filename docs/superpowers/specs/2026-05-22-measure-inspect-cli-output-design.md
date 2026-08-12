# Saving MeasureFeatures `.inspect()` Figures from the CLI / GUI

**Date:** 2026-05-22
**Status:** Spec (pre-implementation)
**Scope:** `phenotypic` CLI and GUI run console
**Related code:** `src/phenotypic/measure/_measure_symmetric_zones.py`,
`src/phenotypic/_cli/`, `src/phenotypic/abc_/_measure_features.py`,
`src/phenotypic/gui/run_console/`

---

## 1. Problem

`MeasureSymZones.inspect()` returns a plotly figure that overlays the geometry behind
every measured number (core radius, symmetric radius, zone disks, mask envelope,
centroids, objmap polygons). Today this figure is only reachable in an interactive
Jupyter session against the in-memory measurer instance — a CLI batch run produces the
measurement parquet but no way to verify *which pixels were measured for which object*
without re-running the whole pipeline interactively.

We want the CLI to optionally save a static PNG of `.inspect()` for every processed
image, mirroring the way `--save-overlays` already saves detection overlays. The
mechanism must extend to any future `MeasureFeatures` subclass that grows its own
`inspect()` method without re-touching the CLI.

---

## 2. Goals

- Add `--save-inspect` (opt-in, off by default) to the `phenotypic` CLI.
- For each image processed by a forward run *or* a `--measure` HDF rerun, save one PNG
  per measurer that implements `.inspect()`.
- Output path: `results/<dataset>/inspect/<measurer-step-name>/<image-stem>.png`.
- Static PNG must show **every** diagnostic overlay layer — no legend-only toggles, no
  widget-gated content.
- Define a duck-typed protocol so the next `MeasureFeatures` subclass that wants this
  only has to implement one method.
- Expose the flag in the GUI run console as a checkbox; propagate it to the spawned CLI
  subprocess.
- `phenotypic.sweep` is **out of scope** for this PR.

## 3. Non-goals

- Adding `.inspect()` to other existing `MeasureFeatures` subclasses (`MeasureSize`,
  `MeasureShape`, etc.). Those land in later PRs.
- Retrofitting `AutoGridFinder.inspect()` (which currently returns a `panel.Column`)
  into this protocol. AutoGridFinder will eventually expose a separate savable-figure
  method; that's a follow-up.
- Saving the interactive HTML form of plotly figures. PNG only this iteration.
- Aggregating per-dataset summary inspect figures. Inspect output is strictly per-image.

---

## 4. Design

### 4.1 The protocol (duck-typed)

A `MeasureFeatures` subclass opts in by implementing:

```python
def inspect(
    self,
    image: Image | None = None,
    *,
    for_save: bool = False,
    **kwargs,
) -> matplotlib.figure.Figure | plotly.graph_objects.Figure:
    ...
```

**Contract:**

- Returns either a `matplotlib.figure.Figure` or a `plotly.graph_objects.Figure`. Any
  other return type is logged at WARNING and skipped by the saver.
- When `for_save=True`, the returned figure must render meaningfully as a static raster:
  every diagnostic trace/artist visible without user interaction (no
  `visible="legendonly"`, no collapsed subplot panels).
- When `for_save=False` (default), the figure may use interactive affordances. This is
  the Jupyter-friendly default.
- The implementation should reuse the diagnostic cache populated by the
  immediately-preceding `measure()` call rather than recomputing the radial/intensity
  pipelines.

No ABC inheritance is required — `OutputManager.save_inspect` dispatches on
`hasattr(measurer, "inspect")` and on the returned figure's type.

### 4.2 Cache liveness invariant

Saved inspect avoids a full re-compute of the diagnostic pipeline by reusing the
per-instance cache populated during `_operate()`. For `MeasureSymZones` this is
`__cache_image`, `__cache_props`, and `__cache_intermediates` (the dict of
`_SymmetryIntermediates` keyed by object label).

The CLI must call `save_inspect(measurer, image, …)` **immediately** after
`pipeline.apply_and_measure(image)` (or `pipeline.measure(image)` on the `--measure`
rerun path) **for the same `image` instance**. The `inspect()` implementation checks
`self.__cache_image is image` to decide cache hit vs recompute, so this invariant is
enforced at runtime — if a future caller batches multiple images through the same
measurer before saving inspect, the cache only holds the last image and earlier images
quietly recompute.

A short comment in the save block documents the precondition, citing this section.

### 4.3 Pipeline accessor

Add `ImagePipeline.iter_measurers() -> Iterator[tuple[str, MeasureFeatures]]` to
`_image_pipeline_core.py`. Yields `(step_key, measurer_instance)` from the same dict
`_build_measurement_run_order()` walks. The CLI consumes this instead of reading
`pipeline._meas` directly, removing the private-attribute coupling.

### 4.4 Output layout

```
<output_dir>/
└── results/
    └── <dataset>/
        ├── measurements/
        ├── hdf/
        ├── overlays/                       # gated by --save-overlays (default on)
        └── inspect/                        # gated by --save-inspect (default off)
            └── <measurer-step-key>/
                ├── <image-stem-1>.png
                ├── <image-stem-2>.png
                └── ...
```

The measurer step key is whatever the pipeline assigned (typically the class name, with
`_1`, `_2` suffixes for duplicates). Two `MeasureSymZones` configured differently in one
pipeline land in distinct subdirs.

A new `DIR_INSPECT: Final[str] = "inspect"` lives in `phenotypic.sdk_._io_constants` and
is re-exported via `phenotypic.sdk_.__init__` and `phenotypic.gui._config` (the latter
for GUI imports per the CLAUDE.md convention).

### 4.5 OutputManager

Naming: the existing `OutputManager` uses plural for the gating attribute (
`save_overlays: bool`) and singular for the per-image method (`save_overlay(...)`). We
mirror that: **`save_inspects` (attr) + `save_inspect(...)` (method)**. The plural
attribute name is slightly awkward but avoids the attribute-shadowing trap that would
arise from naming both `save_inspect`.

Add to `_cli_output_manager.py`:

```python
class OutputManager:
    save_inspects: bool = False   # new field; default off

    def __init__(..., save_inspects: bool = False, ...):
        ...
        self.save_inspects = save_inspects

    def create_structure(self, datasets):
        ...
        for dataset in datasets:
            ...
            if self.save_inspects:
                (dataset_dir / DIR_INSPECT).mkdir(exist_ok=True)
                # per-measurer subdirs are created lazily by save_inspect()

    def save_inspect(
        self,
        measurer: MeasureFeatures,
        image: Image,
        dataset_name: str,
        image_stem: str,
        *,
        measurer_key: str,
    ) -> Path | None:
        """Render measurer.inspect(image, for_save=True) and write a PNG.

        Returns the written path on success, or None on a logged failure.
        """
```

The `save_inspect` method:

1. Builds path
   `results_dir / dataset_name / DIR_INSPECT / measurer_key / f"{image_stem}.png"` and
   creates the leaf directory if missing.
2. Calls `fig = measurer.inspect(image, for_save=True)`.
3. Prepends `image_stem` to the figure's title (matplotlib `fig.suptitle` / plotly
   `fig.layout.title.text`) so the file is self-describing when opened standalone.
4. Type-dispatches the write into the existing `_atomic_write(path, writer)` helper:
    - `matplotlib.figure.Figure` →
      `writer = lambda p: (fig.savefig(p, dpi=150, bbox_inches="tight"), plt.close(fig))`.
    - `plotly.graph_objects.Figure` → `writer = lambda p: fig.write_image(p, scale=2)` (
      requires `kaleido`).
    - Anything else → log WARNING with the class name and return None.
5. `_atomic_write` handles the `.tmp` → final rename so a killed worker never leaves a
   half-written file.
6. Any exception from `measurer.inspect(...)` or from the writer is caught, logged at
   WARNING with `dataset/image_stem/measurer_key`, and `None` is returned — same
   discipline as `_save_layer_safely`.

### 4.6 CLI wiring

In `_cli_process_single.py`:

```python
@click.option(
    "--save-inspect",
    is_flag=True,
    default=False,
    help="Save MeasureFeatures.inspect() figures as PNGs under "
         "results/<dataset>/inspect/<measurer>/<image-stem>.png. "
         "Off by default; opt in when debugging or verifying measurements.",
)
def process_single_image_command(..., save_inspect: bool, ...):
    ...
    output_manager = OutputManager(..., save_inspects=save_inspect, ...)
```

After the existing measurement/overlay save block:

```python
if output_manager.save_inspects:
    for key, measurer in pipeline.iter_measurers():
        if hasattr(measurer, "inspect"):
            output_manager.save_inspect(
                measurer, image, dataset_name, image_stem,
                measurer_key=key,
            )
```

The same block goes in `process_single_hdf_measure_core` so `--measure` reruns also
regenerate inspect figures. (Overlays are skipped on rerun because detection is
unchanged; inspect is regenerated because measurement was just re-run.)

Add `save_inspect: bool = False` to `_cli_types.py`'s run config dataclass.

In `_cli_slurm_scripts.py` and `_cli_slurm_array_scripts.py`, propagate the flag into
generated `sbatch` command strings when set (mirrors how `--save-overlays` is forwarded
today).

In `_cli_recompile_worker.py`, accept `save_inspect` in the worker config and pass
through to its `OutputManager`.

### 4.7 `MeasureSymZones.inspect()` modification

Change the signature:

```python
def inspect(
    self,
    image: Image | None = None,
    base_layer: Literal["rgb", "gray", "detect_mat"] = "gray",
    *,
    for_save: bool = False,
):
```

After the figure is built (just before `return fig`), add:

```python
if for_save:
    for trace in fig.data:
        if getattr(trace, "visible", True) == "legendonly":
            trace.visible = True
```

Four lines. No duplication of layer-building code.

**Note on the contract for future implementations:** `for_save` is intentionally
generic — what it *means* depends on how the class encodes "this layer is hidden by
default." For plotly figures using `visible="legendonly"` (the convention here), the
cheap walk above suffices. For a matplotlib implementation that hides subplots behind
`set_visible(False)` or collapses a `gridspec`, the class is expected to do the
equivalent reveal. The protocol's binding requirement is the *outcome* — every
diagnostic artist renders without user interaction — not the mechanism.

### 4.8 GUI run console

The run console builds the CLI argv via `to_argv` in `_state.py` and shells out via
`_runner.py`. Wiring:

1. **State** — add `save_inspect: bool = False` to `RunConsoleState` in `_state.py`.
2. **Argv builder** — in `to_argv`, append `"--save-inspect"` when `state.save_inspect`
   is truthy.
3. **Form** — add a checkbox to `_form.py` under the existing "Output" section (label: "
   Save inspect figures (debug)"). Wire it to the state via the existing callbacks
   pattern in `_callbacks.py`.
4. **FEATURES.md gate** — add a row to `src/phenotypic/gui/FEATURES.md` declaring the
   new toggle (CI-gated by `features-md-gate`).
5. **No new workflow** — this is a single affordance, not a new end-to-end flow, so
   `WORKFLOWS.md` and tutorial screenshots are not touched.

The GUI wiring is delegated to a subagent during implementation per project workflow
convention.

### 4.9 ABC documentation

Add a new subsection to the `MeasureFeatures` class docstring in
`src/phenotypic/abc_/_measure_features.py`, placed after the existing "How it works (for
developers)" block:

```
Optional: Saveable Inspect Output (CLI auto-discovery):
    Implementing :meth:`inspect` on a subclass is optional. If a subclass
    defines an ``inspect(self, image=None, *, for_save=False, **kwargs)``
    method, the ``phenotypic`` CLI's ``--save-inspect`` flag will
    automatically render and save its output for every processed image
    to ``results/<dataset>/inspect/<step-name>/<image-stem>.png``.

    Contract:
        - Return ``matplotlib.figure.Figure`` or
          ``plotly.graph_objects.Figure``. Any other type is logged at
          WARNING and skipped by the saver.
        - When ``for_save=True``, ensure every diagnostic trace/artist
          is visible without user interaction. The CLI flattens the
          figure to a static PNG; legend-only / collapsed layers are
          invisible in the artifact.
        - Reuse cached intermediates from the immediately-preceding
          ``measure()`` call; do not recompute the full pipeline if
          avoidable.

    Reference implementation:
        :class:`phenotypic.measure.MeasureSymZones`.
```

Add a one-line cross-reference in `src/phenotypic/abc_/CLAUDE.md` under "Implementation
Rules" pointing to the docstring section.

### 4.10 Dependencies

Add `kaleido` to runtime dependencies in `pyproject.toml`. Required for
`plotly.graph_objects.Figure.write_image(...)`. Roughly 80MB installed. Cross-platform;
no system-package or browser dependency.

---

## 5. Test plan

1. **`tests/unit/measure/test_measure_symmetric_zones.py`** — new test: load synthetic
   plate, run pipeline, call `inspector.inspect(image, for_save=True)`, assert no trace
   has `visible == "legendonly"`.
2. **`tests/unit/_cli/test_output_manager_inspect.py`** (new) — unit-test
   `OutputManager.save_inspect` with:
    - A stub measurer returning an mpl Figure → asserts file at expected path is a valid
      PNG.
    - A stub measurer returning a plotly Figure → asserts file at expected path is a
      valid PNG.
    - A stub measurer returning a `str` → asserts no file written, WARNING logged.
    - A stub measurer whose `inspect()` raises → asserts no file written, WARNING
      logged, no exception propagates.
3. **`tests/unit/cli/test_save_inspect.py`** (new) — full CLI integration: invoke
   `phenotypic` on `load_synth_yeast_plate()` fixture with `--save-inspect`, assert PNG
   exists at `results/<ds>/inspect/MeasureSymZones/<stem>.png` and is non-empty. Run
   again without the flag, assert no inspect directory.
4. **`tests/unit/cli/test_hdf_remeasure_inspect.py`** (new) — invoke `--measure` rerun
   on an existing HDF with `--save-inspect`, assert PNG is regenerated.
5. **`tests/unit/_core/test_image_pipeline.py`** — add test for new `iter_measurers()`
   accessor returning expected `(key, instance)` pairs in declared order.
6. **`tests/unit/gui/run_console/test_state.py`** — assert `to_argv` emits
   `--save-inspect` when `save_inspect=True` and omits it otherwise.

CI: existing GUI checks (`features-md-gate`) verify the `FEATURES.md` row.

---

## 6. Risks and mitigations

- **kaleido footprint.** ~80MB. Already approved by the user. Documented in the CLI
  `--help` text and in `docs/source/cli.rst`.
- **Cache liveness drift.** A future change that batches images through a measurer
  before saving inspect would silently fall through to the recompute path (slow but
  correct). Mitigated by inline comment + integration test.
- **`pipeline._meas` private access.** Replaced with public `iter_measurers()` accessor.
  No more private breach.
- **Plotly trace visibility quirks.** Forcing `visible=True` on `Scattergl` traces with
  custom legend groupings is a no-op or worse if traces are intended invisible.
  `MeasureSymZones` only uses `visible="legendonly"` as the "show on toggle" idiom, so
  the cheap walk is sufficient today. Future classes are documented to follow the same
  idiom in the ABC docstring.
- **Atomic writes.** PNG saves go through `path.with_suffix(".png.tmp")` →
  `path.replace(final_path)` to avoid leaving half-written files when a worker is killed
  mid-render.
- **--measure rerun + missing pipeline.** The HDF rerun path loads the pipeline from
  `pipeline.json`. If the pipeline's measurer set changed since the original run, the
  new measurers' inspect output gets written, the missing ones aren't — that's the
  desired behavior (re-measure with the new pipeline).

---

## 7. Implementation phases (for writing-plans)

Per the user's [multi-surface cadence preference](feedback_implementation_phases.md):

1. **Phase 1 — Library + protocol.**
    - `ImagePipeline.iter_measurers()`, `DIR_INSPECT` constant,
      `MeasureSymZones.inspect(..., for_save=...)`, ABC docstring update.
    - Tests: 1, 5.
    - End-of-phase: spawn code-simplifier.
2. **Phase 2 — CLI.**
    - `OutputManager.save_inspect`, `--save-inspect` flag, wiring in
      `process_single_image_core` and `process_single_hdf_measure_core`, SLURM script
      forwarding, recompile worker, `kaleido` dep.
    - Tests: 2, 3, 4.
    - End-of-phase: spawn code-simplifier.
3. **Phase 3 — GUI (subagent).**
    - `RunConsoleState.save_inspect`, form checkbox, `to_argv` emission, `FEATURES.md`
      row.
    - Tests: 6.
    - End-of-phase: spawn code-simplifier.
4. **Phase 4 — Final pass.** One more code-simplifier across all touched files.

---

## 8. Out of scope (follow-ups)

- `.inspect()` on `MeasureSize`, `MeasureShape`, `MeasureIntensity`, `MeasureColor`,
  `MeasureBounds`, `MeasureRadialExpansion`, etc.
- `AutoGridFinder.inspect()` retrofit (rename current panel-based method to
  `.dashboard()`, add new mpl/plotly `.inspect()`).
- `phenotypic.sweep` exposing `--save-inspect`.
- Saving plotly figures as interactive HTML alongside PNG.
- A results-viewer panel that surfaces saved inspect PNGs in the GUI.
