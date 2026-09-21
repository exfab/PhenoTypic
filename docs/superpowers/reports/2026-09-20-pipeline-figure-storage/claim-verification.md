# Claim verification — `pipeline-plot-bindings.html`

Subject: `docs/superpowers/artifacts/2026-09-20-pipeline-figure-storage/pipeline-plot-bindings.html`
Tree: `/bigdata/exfab/anguy344/PhenoTypic`, branch `fix/aged-out-job-active-check`, HEAD `91491bfd`
Reviewer: independent claim audit. Analysis only — no source file was modified.
Evidence: source reading at the stated HEAD, the existing test suite, and one
runtime probe (`pkgutil.walk_packages` over the whole `phenotypic` namespace plus
six targeted checks) executed by the orchestrator under `uv run`, exit 0.

---

## Verdict

The artifact is safe to rely on for its central architectural claims, and its
citation hygiene is unusually good: of the ~35 `file:line` references I resolved,
all but one land on the construct they are cited for, and several (`_bindings.py:149-153`,
`_bindings.py:173-199`, `_cli_pipeline_split.py:53-63`, `_coordinator.py:393-402`)
are line-exact. A runtime probe settled the dynamic half: the `ops`-slot binding
reproduces character for character, the matplotlib `AttributeError` is the exact
string quoted, and the manifest's 381 bytes are right. All four "load-bearing"
mechanisms — `plots` as the pydantic field
(`exclude=True`), identity-not-equality reference resolution, the exactly-one-lifecycle
gate, and the `strict` asymmetry between the staged and ordinary CLI paths — survive
scrutiny unchanged. Five things need correction before the page is treated as a
reference: the shipped-provider inventory is **incomplete** (two classes missing,
one of them a `PlotImage`); the "default id is `ref.key`" rule is **wrong for the
`model` slot**, which is where three of the shipped providers bind; `DiagnosticsPlotter`
is described as "the `image.plot` accessor", an API that **no longer exists**; the
Kaleido finding's "publishes nothing" is **false on the multi-page and aggregate
paths**, which still write an empty manifest; and the anatomy table contradicts the
rest of the page on the flat filename. None of these overturn a verdict card. The
provenance section is also *understated* — the suite already pins the `PlotAnalysis`
and `PlotQc` dispatch paths, `AnalysisRegistry` resolution, and the `publication_guard`
compare-and-set that the page lists as "not exercised."

---

## Errors

### E1 — The shipped-provider inventory has two omissions, and the method explains one of them

> "Every plot provider that ships today … Enumerated at runtime by walking
> `PhtPlot.__subclasses__()` after importing `detect`, `enhance`, `measure`, `refine`,
> `grid`, `correction`, `post`, `analysis`, `plotting` and `qc`."

I rebuilt the inventory two ways: a grep closure over `src/` (every `class X(...)`
whose bases name a plot contract, then the subclasses of each hit — two levels;
level 3 is empty), and a runtime `pkgutil.walk_packages` over the entire
`phenotypic` namespace followed by a recursive `PhtPlot.__subclasses__()` walk.
Both agree, and the runtime walk imported every module with **zero** import
failures, so it is exhaustive.

The walk returns **23** classes. Remove the four lifecycle mixins themselves
(`PlotImage`, `PlotMeas`, `PlotAnalysis`, `PlotQc`, all in `abc_/plotting/_lifecycle.py`)
and the two abstract/private `PlotAnalysis` bases that exist only to be subclassed
(`ModelFitter` at `analysis/abc_/_model_fitter.py:41`, `_LinearSoftplusBase` at
`analysis/abc_/_linear_softplus_base.py:40`), and **17** providers remain — not the
15 in the table. Missing:

`src/phenotypic/measure/_measure_orientation_zones.py:3686`

```python
class _OrientationZonesReport(PlotImage):
    """Stateless image consumer composing the orientation diagnostic."""
```

This is a **`PlotImage`** — the same lifecycle category the artifact's `ImageOperation`
argument rests on — declared at module scope in a package the enumeration says it
imported, and the runtime walk lists it (`_OrientationZonesReport … PlotImage`). A
recursive `__subclasses__()` walk after importing `measure` would have found it. It is
private and not a `BaseModel` (so it could never serialize as an inline binding),
which may have been an unstated editorial filter, but the table's heading is "Every
plot provider that ships today" and it already lists three non-bindable classes.

`src/phenotypic/_core/_image_parts/plot_accessor/_detect_modes_plotter.py:23`

```python
class DetectModesPlotter(BasePlotter, PhtPlot):
```

No lifecycle, so not bindable — it belongs in the same block as `DiagnosticsPlotter`.
This one is **exactly the blind spot the method has**, and the probe demonstrates it:
`DetectModesPlotter` is imported only lazily, inside method bodies at
`src/phenotypic/plotting/_image_plots.py:129` and `:138`, so importing the ten listed
subpackages never loads its module and the class never appears in `__subclasses__()`.
Force every module to load and it appears. `DiagnosticsPlotter` survived the artifact's
walk only because `_image_plots.py:17-19` imports it at decoration time, i.e. at
`phenotypic.plotting` import — an accident of that one module, not of the method.

**Correction:** add both rows. The two substantive verdicts the table supports are
unaffected — neither new class is an `ImageOperation`, and neither is a
`MeasureFeatures`, so "no shipped `ImageOperation` appears here at all" still holds
(independently confirmed: the closure contains no `ImageOperation` subclass).

### E2 — The default binding id is *not* `ref.key` for the `model` slot

> "For a pipeline-owned object the default id is `ref.key` (`_bindings.py:165-171`)"
> — and, in the diagram, "binding id (class name; a slot key when ref'd)".

`src/phenotypic/plotting/_pipeline/_bindings.py:165-170`:

```python
            ref = _identity_ref(raw, registry)
            default_id = (
                ref.key
                if ref is not None and ref.key is not None
                else type(raw).__name__
            )
```

The `model` slot is registered with a **`None` key** — `_image_pipeline_core.py:704`
does `registry[("model", None)] = self.model`, and `PipelineObjectRef._validate_key`
(`_bindings.py:31-32`) *forbids* a key on `model`. So a `ModelFitter` passed bare to
`plots=` falls through to `type(raw).__name__`. A `LogGrowthModel` bound on the model
slot publishes to `plots/LogGrowthModel/`, which is what the how-to doc says, not
`plots/model/`.

Confirmed at runtime. Binding the same object three ways gives
`meas:'sym'  model:'LinearLagModel'  inline:'PlotMeasTimeSeries'`, with the model
binding reporting `ref: slot='model' key=None`. The model-slot case produces the
*inline* answer despite being a reference.

This matters twice over: it is one of six slots, but it covers **three of the shipped
providers** (`LogGrowthModel`, `LinearLagModel`, `LinearCapAndLagModel` — the inventory
rows that read "binds on the `model` slot"), and it is a counterexample to the very
sharp-edge the page files as finding #2. It is misleading rather than merely
incomplete, because the rule is stated as the *contrast* against the doc
("the default id is `ref.key`… not `plots/MeasureSymZones/`") — and for a bound model
the doc's `<ClassName>` is exactly right. The finding still holds for
`ops`/`meas`/`post`/`filters`/`qc`; it just is not universal for pipeline-owned objects.

**Correction:** "the default id is `ref.key` when the ref has one — every slot except
`model`, whose ref is keyless, so a bound model falls back to the class name."

### E3 — `DiagnosticsPlotter` is not "the `image.plot` accessor"; that accessor was removed

> `DiagnosticsPlotter` | no lifecycle | "**Not bindable.** The `image.plot` accessor"

`src/phenotypic/_core/_image_parts/plot_accessor/__init__.py`:

```
"""Legacy plotter implementation sources pending standalone plot migration.

This package intentionally has no public exports. ``Image.plot`` and the dynamic
plotter registry were removed as part of the plotting API hard cutover.
"""
```

and `src/phenotypic/plotting/_image_plots.py:1`:

```python
"""Standalone replacements for the removed ``Image.plot`` diagnostics."""
```

`Image` exposes no `plot` member (no `def plot` and no occurrence of the string
`plot` anywhere in `src/phenotypic/_core/_image.py`). `DiagnosticsPlotter` is a
legacy implementation source that `PlotDiagnostics` constructs short-lived and
delegates to (`_image_plots.py:44-50`). A reader following the table would look for
an API that is gone.

**Correction:** "legacy implementation source behind `PlotDiagnostics`; the
`Image.plot` accessor it once served has been removed."

### E4 — "publishes *nothing*" is false on the multi-page and aggregate paths

> "a Plotly plot configured on an ordinary CLI run here publishes *nothing* and
> reports *success*."

True for the single-page image path, where the write is `FigureAdapter.save_png` →
`os.replace` inline (`_coordinator.py:370-380`) and the whole thing is swallowed by
`emit_image`'s handler. **Not** true elsewhere. `_writer.py:140-149` catches a page
failure *per page* and `continue`s, and then `_writer.py:160-177` writes the manifest
**unconditionally**:

```python
    manifest = {
        "schema_version": 1,
        "plot_id": plot_id,
        "class": plot_class or plot_id,
        "pages": pages,
    }
```

With every page failing, `pages` is `[]` and the run still publishes
`<dir>/manifest.json` (plus the `.publication.lock` created at `_writer.py:84`).
That is arguably a worse outcome than nothing — a durable artifact asserting the
plot has zero pages — and it is what a multi-page `PlotImage`, and every
`PlotMeas` / `PlotAnalysis` / `PlotQc`, would leave behind.

**Correction:** "…publishes no PNG and reports success; multi-page and aggregate
plots still publish a `manifest.json` listing zero pages."

### E5 — The anatomy table contradicts the rest of the page on the flat filename

> "that single-default case is the only one that gets the flat `<stem>.png`
> filename on the image path."

`_coordinator.py:371` is `destination = base / f"{output_stem}.png"`, and
`output_stem` comes from `_image_output_stem` (`:393-402`), which is
`f"{safe_path_component(image_stem)}-{digest}"`. There is no code path that
produces a bare `<stem>.png`. The page states this correctly twice elsewhere
(the tree annotation and the single-page-shortcut section), so this is an internal
contradiction, but it is the row a reader hits first.

Pinned by `tests/unit/plotting/test_coordinator.py:98-100`:

```python
    written = list((plots_dir(tmp_path) / "image" / "dataset").glob("*.png"))
    assert len(written) == 1
    assert written[0].name.startswith("plate-1-")
```

---

## Imprecise but not wrong

- **The `PlotOutput` normalization is cited to the wrong file.** The anatomy row
  points at `abc_/plotting/_output.py` and says a bare figure is normalized to a
  page keyed `"default"`. That file holds only the `PlotPage`/`PlotOutput`
  dataclasses; the normalization is `normalize_plot_output` in
  `src/phenotypic/plotting/_pipeline/_output.py:10-17`. The row's "Where" cell is
  right for the contracts, wrong for the behaviour described in the same cell.

- **"beside its dataset folder."** The flat single-page PNG is written *inside* the
  dataset folder: `base = plots_base / safe_path_component(binding.id) /
  safe_path_component(dataset)` (`_coordinator.py:355-359`), then
  `base / f"{output_stem}.png"`. It takes the place the per-image directory would
  have occupied, not a sibling of the dataset directory.

- **The id-validator rule reads as banning the `.` character.** "No `/`, `\`, `.`,
  `..`, no empty." `_bindings.py:97` is
  `if "/" in value or "\\" in value or value in {".", ".."}` — `/` and `\` are banned
  as *characters*, `.` and `..` only as the *entire* value. Confirmed at runtime:
  `id='a.b'` **accepted**, `id='a b'` **accepted**, `id='.'` and `id='..'` rejected.

- **The manifest block is not byte-for-byte what lands on disk.** `_writer.py:170`
  uses `json.dumps(manifest, indent=2, sort_keys=True)`. Reproducing the two-page
  publication gives a file whose top-level key order is
  `['class', 'pages', 'plot_id', 'schema_version']` and whose page objects are
  `['backend', 'file', 'key', 'label', 'metadata']`; the artifact's block shows both
  in a hand-readable order. Every key and value is correct; only the ordering is
  presentational, and the byte count is right (see Confirmed). The provenance
  section's verbatim claim covers "console lines and the file tree", so nothing is
  technically overclaimed — but the block sits among material that is verbatim.

- **The rules-table caption miscounts itself.** "**Two of these were confirmed by
  triggering them** — the lifecycle count (both directions), the duplicate id, and
  the post-sanitization collision" enumerates three (four if the two lifecycle
  directions count separately).

- **The `PlotQc` call-site row is incomplete.** It lists
  `_cli_output_manager.py:1272` and `_gui/_plot_refresh.py:113`, omitting the three
  `emit_dependent_qc` sites at `_gui/_plot_refresh.py:46, 58, 92`. The diagram names
  `emit_dependent_qc`; the table does not account for it.

- **`_image_pipeline_core.py:690-708`** — `_plot_object_registry` is `691-708`; 690
  is the closing line of `get_plots`. Off by one; harmless.

- **Quoted message clipped.** "ids collide after filesystem sanitization: [['a b','a-b']]"
  — the real prefix is `"plot binding ids collide after filesystem sanitization: "`
  (`_bindings.py:246`), and Python's list repr puts a space after the comma.

- **"The GUI *consumes* bindings … it just does not author them" understates a third
  role.** The authoring claim is correct and the builder grep is exactly as reported —
  `grep -rn "plots" src/phenotypic/_gui/builder/ --include=*.py` returns **zero hits**,
  and a broader case-insensitive `plot` sweep of that directory returns only two
  matplotlib-colormap lines in `_image_renderer.py:169,171`. But the analysis sub-app
  *reads, merges and re-writes* plot-binding nodes into `pipeline.json` on every GUI
  recipe save: `src/phenotypic/_gui/analysis/_recipe_state.py:69` (`"plots"` in
  `_SERIALIZED_PIPELINE_KEYS`), `:183-208` (`_plot_validation_node`), `:281-283`,
  `:597-660` (per-binding envelope merge including the `input` field), `:760-783`
  (opaque-node preservation), `:815-820`. Nothing there *creates* a binding, so the
  finding stands — but "consumes" is too weak for machinery that round-trips them
  through a user-facing save. `_gui/analysis/_plot_controls.py` is unrelated: it
  builds session-scoped widgets for analyzer `show`/`inspect` *method* arguments
  that "never serialize into `pipeline.json`" (its own docstring).

- **"doc drift" is the wrong diagnosis, though the finding is right.** `git log --follow`
  shows `docs/source/extending/pages/custom_plotter.md` and
  `src/phenotypic/plotting/_pipeline/_bindings.py` were **last touched in the same
  commit**, `71737850` ("Clean up plotting package exports", 2026-08-03). The doc's
  unconditional `deliverables/plots/<ClassName>/` was written alongside the `ref.key`
  default; it never described the shipped behaviour rather than falling behind it.
  Your reading of the doc is otherwise fair: line 38 is unconditional, and line 35's
  "Plotly and Matplotlib figures are both accepted by the CLI publisher" is a claim
  about the *publisher*, which the page correctly says is true. Worth noting the
  doc's own example (lines 20-29) overrides `inspect()` rather than using `@figure`,
  which is precisely the pattern where matplotlib does work — so the doc is not as
  misleading on that point as the juxtaposition implies.

- **The recipe snippet omits `from phenotypic import ImagePipeline`.** It imports
  `go`, `PlotImage`, `figure` and `OtsuDetector`, then uses `ImagePipeline`
  unqualified.

---

## Unverified

- **The two PNG byte sizes and the digest.** `9326 B` / `8472 B` and
  `7e9f968494db` are still unchecked; PNG size depends on the figure content, which
  the artifact's script defined and I did not reproduce. The digest *formula* is
  confirmed (`_coordinator.py:400-402` is `sha256(dataset + b"\0" + stem)[:12]`). To
  settle: re-run that script and diff. (`381 B` for the manifest **is** confirmed —
  see Confirmed.)

- **The console transcripts.** The `JSON top-level keys: [...]` line and the two
  terminal blocks. Consistent with the source (`pipe_cfgs` is the serialized alias
  per `_image_pipeline_core.py:202-205`; `qc`/`nrows`/`ncols` are omitted when empty
  per `_serializable_pipeline.py:179-194`), and the `ops`-slot block is independently
  reproduced (see Confirmed), but I did not re-run the five scripts end to end.

- **The Kaleido/Chrome failure.** The environment is consistent with the claim —
  `kaleido 1.2.0` and `choreographer 1.2.1` are installed in `.venv`, `kaleido>=1.2.0`
  is pinned at `pyproject.toml:70`, no `google-chrome`/`chromium`/`chrome` is on
  `PATH`, `~/.cache/kaleido` and `~/.local/share/kaleido` do not exist, and
  `.venv/bin/plotly_get_chrome` is present — but I did not trigger `write_image`.

- **Nothing else.** The whole-tree inventory, the matplotlib raise, the model-slot
  default id, the `ops`-slot binding, the manifest layout and the id-validator
  boundaries were all settled by the runtime probe and have moved out of this
  section.

---

## Confirmed

Each of the following resolves to the cited construct and says what the artifact
says it says.

**The two verdicts.** `plots` is the pydantic field —
`_image_pipeline_core.py:222` `plots: List[Any] = Field(default_factory=list, exclude=True)`.
No `figures` field, kwarg or alias exists anywhere in `src/phenotypic`; the only
`figures` is `PhtPlot.figures()` at `_pht_plot.py:435-437`, returning `BoundFigures`.
The naming verdict is stronger than the page claims: `ImagePipeline(figures=[...])`
does not merely go nowhere, it **raises** `ValidationError: Extra inputs are not
permitted [type=extra_forbidden]`. `ImagePipelineCore(BaseOperation, LazyWidgetMixin)`
(`_image_pipeline_core.py:143`) inherits `extra="forbid"` from
`abc_/_base_operation.py:175-179`, and pydantic merges `model_config` across the MRO,
so the core's own `ConfigDict(validate_by_name=True)` (`:187`) adds to it rather than
replacing it. There is no silent-drop failure mode here.
`MeasureFeatures` is `class MeasureFeatures(BaseOperation, ABC)`
(`abc_/_measure_features.py:49`) and `ImageOperation` is
`class ImageOperation(BaseOperation, LazyWidgetMixin, ABC)`
(`abc_/_image_operation.py:19`) — sibling ABCs, neither a subclass of the other.
`MeasureSymZones` (`measure/_measure_symzones.py:88`) and `MeasureOrientationZones`
(`measure/_measure_orientation_zones.py:947`) both mix `CanonicalZoneMeasure` (a
`MeasureFeatures`) with `PlotImage`. No shipped `ImageOperation` carries a lifecycle.

**Identity, not equality.** `_bindings.py:119-121` carries exactly the quoted
rationale; `_identity_ref` at `:357` does `if candidate is plot` at `:362`. Pinned
by `tests/unit/plotting/test_pipeline_bindings.py:70-80`
(`test_equal_but_distinct_inline_plot_is_not_made_a_reference`) and `:47-57`.

**The anatomy.** `PhtPlot` at `_pht_plot.py:265` is fieldless and constructor-free.
`figure` at `:131`; the wrapper at `:188-192` returns `apply_theme(fn(*args, **kwargs))`
and `FigureSpec` (`:194-205`) records title/section/controls/primary/order.
`PlotImage` is the only lifecycle setting `_weakly_bind_subject = True`
(`_lifecycle.py:13`). `PlotOutput`/`PlotPage` live in `abc_/plotting/_output.py`.

**All ten validation rules** raise where and as described, inside the pipeline's
`model_validator(mode="after")` `_resolve_plot_bindings` (`_image_pipeline_core.py:328-344`),
which runs after every ordinary slot. `_bindings.py:161-164`, `:207-212`, `:149-153`,
`:91-101`, `:227-230`, `:243-248`, `:214-223`, `:173-199`, `:258-269`, and the nested
refusal at `_image_pipeline_core.py:332-336` with the exact string "nested plot
execution is not supported". The rejection message quoted in the transcript
("plots entries must inherit PhtPlot or be PlotBinding instances; got OtsuDetector")
matches `_bindings.py:162-163` word for word.

**Serialization.** `serialize_plot_binding` at `_bindings.py:252`; the ref branch
emits `{"id", "ref"}` with no copy of the object (`:255-256`), the inline branch
`{"id", "inline": {module, qualname, params}}` (`:270-274`). The "restores refs last"
claim is `_serializable_pipeline.py:393-398`, whose comment is verbatim the artifact's
sentence. Pinned by `test_measurer_reference_round_trip_preserves_shared_identity`
and `test_model_reference_round_trip_preserves_shared_identity`.

**The registry.** `_plot_object_registry` (`_image_pipeline_core.py:691-708`) contains
`ops`, `meas`, `post`, `filters`, `model` (keyless) and `qc` (keyed by
`entry.instance_id`).

**Every emit call site.** `emit_image(..., strict: bool = False)` at
`_coordinator.py:79-86`. There are exactly three `emit_image` call sites in `src/`:
`_cli_process_single.py:348-354` (in `process_single_image_core`, the forward path),
`_cli_process_single.py:450-456` (in `process_single_store_measure_core`, re-measure),
and `_cli_staged_workers.py:526-533`, which is the **only** one passing `strict=True`
(`:532`). No wrapper sets it elsewhere. `_cli_output_manager.py:1221/1241/1272` and
`_gui/_plot_refresh.py:38/40/53/87/113` are all exact. Every aggregate emit is
unconditionally best-effort (`_coordinator.py:177-181, 259-263, 304-309, 328-332`) —
none takes a `strict` parameter. The `PlotMeas` subject really is the post-applied
mirror as pandas: `measurements_pd = post_df.to_pandas()` at
`_cli_output_manager.py:1216`. `QcPlotSubject`'s `qc_database` really is DuckDB —
`layout.qc_duckdb` at `_plot_refresh.py:117`, `qc_duckdb_path(output_dir)` at
`_cli_output_manager.py:1277`. Pinned by
`tests/unit/plotting/test_coordinator.py:103-114` (strict) and
`tests/unit/gui/test_plot_refresh.py`.

**The staged-GPU guard.** `_cli_pipeline_split.py:53-63`, message quoted exactly.
Exercised by `tests/unit/cli/test_cli_pipeline_split.py:53-61`, which builds
`class _PreGpuPlot(BlurGauss, PlotImage)` and binds it on `ops` — i.e. the
`ImageOperation`-on-the-`ops`-slot path really is exercised by tests, exactly as the
page says.

**The `ops`-slot demo reproduces exactly.** Rebuilding the artifact's
`class PlottyDetector(OtsuDetector, PlotImage)` with a `@figure` method and passing the
same object to `ops={"detect": op}` and `plots=[op]` yields
`id='detect'`, `ref=PipelineObjectRef(slot='ops', key='detect')`, identity
`binding.plot is pipeline.get_ops()["detect"]` → `True`, and JSON
`[{'id': 'detect', 'ref': {'slot': 'ops', 'key': 'detect'}}]` — character for
character what the artifact's second terminal block reports. Nothing special was
needed, as the page says.

**The published tree and the manifest size.** Re-running a two-page publication with
labels "Colony count" and "Half" writes exactly
`['.publication.lock', 'Colony-count.png', 'Half.png', 'manifest.json']`, and
`manifest.json` is **381 bytes** — the figure the artifact prints. The file listing
and that byte count are confirmed; only the two PNG sizes are not.

**Publication mechanics.** `publish_plot_output` at `_writer.py:54`; lock
(`:84`, `.publication.lock`), per-page temp + `os.replace` (`:129-135`), manifest
written last and replaced atomically (`:166-174`). A failing page is logged and
omitted while siblings continue (`:140-149`) — pinned by
`tests/unit/plotting/test_output_adapter.py:112`. `safe_path_component` (`:32-51`)
maps `"ds 1"` → `"ds-1"` and preserves `plate_01`, matching the tree. The
single-page `key == "default"` shortcut is `_coordinator.py:360-380`, and
`normalize_plot_output` producing that key is `plotting/_pipeline/_output.py:10-17`.

**Sharp edge #1 (matplotlib through `@figure`), confirmed both ways.** A
`BaseModel, PlotImage` whose `@figure` method returns `plt.figure()` raises, on
`inspect()`, exactly `AttributeError: 'Figure' object has no attribute 'layout'` —
the string the artifact quotes. The source agrees:
`_pht_plot.py:192` calls `apply_theme` unconditionally; `apply_theme`
(`sdk_/viz/figures/_theme.py:195-228`) sets `fig.layout.template` at `:227` with no
backend guard. `matplotlib/figure.py` in this venv defines no `layout` attribute,
property or `__getattr__`, and neither does `matplotlib/artist.py` — so the access
is a plain `AttributeError`, matching the quoted text. No code path pre-empts it:
`inspect()` (`:382-399`) and `_render_spec` (`:349-359`) both resolve the method with
`getattr(self, spec.name)`, which is the wrapper. And **no shipped provider returns
matplotlib from a decorated method** — every one of the 49 `@figure` methods in
`src/` is annotated `-> go.Figure` or builds one with `plotly.graph_objects`.
`DiagnosticsPlotter.diagnostics()` (`_diagnostics_plotter.py:1354`), which *does*
return matplotlib, is **not** decorated. The providers that override `inspect()`
outright (`ModelFitter` at `analysis/abc_/_model_fitter.py:887`, the three QC checks,
the two `plotting/` meas models, `PlotDetectModes`) bypass the wrapper, which is the
escape hatch the artifact describes.

**`PlotBinding` / `PipelineObjectRef` are private.** Both are exported from
`phenotypic.plotting._pipeline.__all__` (`:63-64`); `phenotypic.plotting.__all__`
exports only the four ready-to-use models.

---

## Provenance section is understating what is already pinned

The "Not exercised" block says the `PlotAnalysis` and `PlotQc` dispatch paths, the
`AnalysisRegistry` resolution, the `publication_guard` compare-and-set and the
QC-recipe reference resolution are "traced from source only". Read as "not
demonstrated *in this artifact's five scripts*" that is accurate, but it will be
read as "unpinned", and it is not:

- `tests/unit/plotting/test_coordinator.py:193, 213, 237, 258, 289` pin
  `PlotAnalysis` dispatch including per-dispatch input resolution and the
  fitted-state reuse path the lifecycle table describes.
- `tests/unit/plotting/test_coordinator.py:310, 335, 352, 386` pin `PlotQc`
  dispatch, `emit_dependent_qc` filtering, and — directly — the QC-recipe reference
  resolving by `instance_id` even when the output id is custom.
- `tests/unit/plotting/test_analysis_registry.py` has ten tests over
  `AnalysisRegistry` resolution: manifest-selected Parquet, per-refresh checksum
  revalidation, interrupted-publication recovery, legacy fallback ordering.
- The `publication_guard` compare-and-set is pinned end-to-end by
  `tests/gui/results_viewer/test_mutation_guard.py:553`
  (`test_real_plot_writer_rechecks_after_render_and_preserves_generation`), which
  drives the real `publish_plot_output` with a late-failing guard, asserts
  `PlotPublicationBlocked`, and asserts the output tree is byte-unchanged apart from
  the deliberately perturbed file.
- `tests/unit/plotting/test_pipeline_bindings.py:138-208` pin the QC-reference
  capability and lifecycle-exclusivity rules.

Nothing in any of these contradicts the artifact.

---

## Separate — a possible bug in PhenoTypic (not an artifact error; not fixed)

`src/phenotypic/plotting/_pipeline/_coordinator.py:225-263`, `PlotCoordinator.emit_qc`:

```python
        for configured in self._pipeline.get_plots():
            try:
                ref = configured.ref
                ...
                if not isinstance(plot, PlotQc):
                    continue
                binding = configured.model_copy(update={"plot": plot})   # :239
                ...
            except Exception:  # noqa: BLE001 - plot output is best-effort
                logger.warning(
                    "Plot %s failed during QC inspect", binding.id,      # :261
                    exc_info=True,
                )
```

`binding` is first assigned at `:239`, but the handler at `:259-263` dereferences it.
Anything raising in `:227-237` — the `assert` at `:231`, a `module.check` property,
an unhashable `module_key` — is caught and then the handler touches `binding`. On the
first iteration that is an `UnboundLocalError` *inside the except block*, which
replaces the original exception and escapes `emit_qc` entirely (defeating the
best-effort intent). On a later iteration it silently logs the **previous** binding's
id. `emit_dependent_qc` (`:283-309`) does not have this shape, because there the loop
variable *is* the binding. Suggested shape: log `configured.id` (always bound) rather
than `binding.id`.
