# Quality-control analysis operations + GUI integration — design

**Date:** 2026-05-12
**Branch:** `builder-gui-redesign`
**Status:** Design

---

## Summary

Introduce a `QualityCheck` family of analysis operations that flag data-quality
issues in arrayed-colony measurement frames, plus two new tabs in the results
viewer:

- a **QC tab** where the user configures, runs, and acts on quality checks; and
- a **Heatmap tab** that renders plate-overlay heatmaps over any measurement or
  QC severity column.

QC checks recompute live as the user curates colonies in the existing Colony
tab — no polling, no manual refresh — because the QC tab lives inside the
results-viewer Dash app and subscribes to the same `STORE_REMOVED_KEYS` store
that already drives curation.

The v1 ships two concrete checks:

- **`ExpectedVsDetectedCount`** — compares detected colony count per group
  against the row count of a provided metadata frame.
- **`ReplicateAgreement`** — flags `(group, time)` bins whose standard error
  across replicates exceeds a threshold.

Per-colony pixel-level segmentation metrics (Solidity, Boundary Sharpness,
etc.) are intentionally deferred to a follow-up iteration.

---

## Motivation

Today, after a CLI run the user lands in the results viewer with `Plate` and
`Colony` tabs. Both surface the measurement frame, but neither answers two
quality-control questions the user repeatedly needs to ask:

1. **Did the pipeline detect what it should have detected?** A metadata file
   often specifies a 96-well or 384-well plate layout. The user wants to know
   immediately when a plate is missing colonies.
2. **Are my replicates agreeing?** When the same condition is replicated
   across the plate or across plates, growth measurements at matched
   time points should agree within a tolerable standard error. Disagreement
   often points to a bad image, a wrong metadata join, or a real biological
   outlier worth excluding.

Additionally, the user wants to **see spatial patterns** in any
measurement — edge effects, contamination blobs, batch shifts — that scalar
summary tables and per-colony cards can't surface. A plate-overlay heatmap is
the obvious complement.

The whole point of a curation loop ("remove this bad colony, see metrics
improve") falls apart if QC metrics don't recompute live. Hence the decision
to put both new tabs inside the results-viewer Dash app rather than the
`/analysis/` sub-app — see the architectural-decisions section below.

---

## Scope

### In scope (v1)

- New `QualityCheck(SetAnalyzer)` ABC in `phenotypic.analysis.abc_`.
- Two concrete subclasses: `ExpectedVsDetectedCount`, `ReplicateAgreement`.
- New `QUALITY_COUNT` and `QUALITY_SE` MeasurementInfo classes (one
  per concrete check; no generic `QUALITY_CHECK` enum — see
  rationale in the MeasurementInfo section).
- QC tab inside the results viewer with configurable check cards, live
  recompute on curation, "mark flagged for removal" hand-off into the
  existing `STORE_REMOVED_KEYS` curation store.
- Heatmap tab inside the results viewer with measurement-or-QC-severity
  color picker, image picker, time slider, aggregator picker.
- Sidecar persistence (`<output>/.viewer_cache/qc_recipe.json`) for the QC
  recipe.
- Optional "Export QC report" button that writes `qc.parquet` +
  `qc_summary.json` under the output root.
- Shared-state refactor: move `_schema_cache.py` from `gui/analysis/` up
  to `gui/` so both Dash apps can mount it.
- `FEATURES.md` and `WORKFLOWS.md` ledger rows, with screenshot capture
  functions and tutorial pages.

### Out of scope (deferred)

- Pixel-level segmentation-quality `MeasureFeatures` subclasses (Solidity,
  Boundary Sharpness, Edge-touching fraction, Saturated-pixel fraction,
  etc.). Their column-emission seam is preserved — they'd land as ordinary
  `MeasureFeatures` subclasses in `phenotypic.measure/` and immediately
  flow into `measurements.parquet` like any other measurement — but the
  specific list and implementation are a separate iteration.
- CLI auto-emission of `qc.parquet`. v1 exposes only a GUI export button;
  any future auto-emission would extend `finalize_post_master_outputs`.
- A click-through bridge from a flagged-row table back into the Colony
  tab's selection. The QC tab's `flagged_keys()` API is the seam this
  bridge would use; the bridge itself is follow-up work.
- QC integration into the `/analysis/` sub-app. v1 puts the configuration
  UX exclusively in the results viewer because that's where live curation
  reactivity lives. If we later want `/analysis/` to share the recipe, the
  sidecar JSON moves up into `pipeline.json` and both apps consume it.

---

## Architectural decisions

### Why `QualityCheck` subclasses `SetAnalyzer` rather than a new ABC family

`SetAnalyzer` already provides the contract we need (`analyze`, `show`,
`dash`, `results`, plus `_filter_by` and `_ensure_float_array` helpers),
already integrates with the `OperationRegistry` and `_param_forms` GUI
machinery via `ColumnRef` / `ColumnRefList` annotations, and already
auto-serializes through `to_json`/`from_json`. A new ABC family would
duplicate every piece of that scaffolding.

`QualityCheck` adds three concepts on top of `SetAnalyzer`:

1. A short class-level `name` composed into output column names
   (`QC_<name>_Flag`, etc.).
2. Severity-driven `Flag` / `Status` semantics computed by the base class
   from the subclass's `_compute()` method.
3. A `flagged_keys()` accessor that drives GUI curation hand-off.

### Why QC lives in the results viewer, not in the `/analysis/` sub-app

The hub mounts each tool as a separate Dash app behind
`DispatcherMiddleware`. The two Dash apps cannot share `dcc.Store`
instances. Putting QC in `/analysis/` would force a mtime-poll-the-disk
loop (~2-second lag between curation click in the viewer and QC card
refresh in `/analysis/`), which contradicts the user's "see metrics
improve in real time" requirement.

Putting QC inside the results-viewer Dash app gives us a true
Dash-callback chain: a single store update propagates synchronously to
the QC card recompute and the heatmap rebuild. The cost is a small
shared-state refactor (move `_schema_cache.py` up one level) so both apps
can mount the existing `OperationRegistry` and `_param_forms` machinery.

### Why a sidecar `qc_recipe.json` rather than extending `pipeline.json`

`pipeline.json` is the CLI's reproducibility-state artifact. QC
configuration is viewer-side curation-adjacent state that the user
mutates as they explore — closer in spirit to `STORE_REMOVED_KEYS` and
the curated `measurements.parquet` mirror than to the immutable
pipeline definition. Adding QC to `pipeline.json` would force the CLI to
know about QC and would couple the analysis sub-app's `RecipeState`
machinery to the viewer's QC tab.

The sidecar `<output>/.viewer_cache/qc_recipe.json` sits alongside the
existing curation state and is wiped whenever the user nukes
`.viewer_cache/` (the existing startup banner already encourages this
for stale tiles). If QC later proves to belong in the reproducibility
surface, the migration is straightforward: lift the sidecar's schema
into `pipeline.json` and have both apps consume it via `RecipeState`.

### Why severity is normalized to a fractional scale

Both v1 checks naturally produce dimensionless severity values
(`|delta| / expected` for the count check; `|SE| / |mean|` for the SE
check). A shared scale lets `QualityCheck` ship default `severity_warn`
and `severity_fail` thresholds that subclasses override per their
domain. Checks whose underlying metric isn't naturally bounded simply
override the class-level threshold attributes.

---

## Library architecture

### Module layout

```
src/phenotypic/analysis/
├── abc_/
│   ├── _quality_check.py           # NEW: QualityCheck(SetAnalyzer) ABC
│   └── _set_analyzer.py            # unchanged
├── _expected_vs_detected.py        # NEW: ExpectedVsDetectedCount
├── _replicate_agreement.py         # NEW: ReplicateAgreement
└── __init__.py                     # MOD: export the new public names

src/phenotypic/tools_/measurement_info/
├── _quality_count.py               # NEW: QUALITY_COUNT MI (per-check)
└── _quality_se.py                  # NEW: QUALITY_SE MI (per-check)
```

### `QualityCheck` ABC contract

```python
class QualityCheck(SetAnalyzer, ABC):
    """Detect quality-control issues in measurement frames."""

    #: Short identifier composed into column names (QC_<name>_Flag, etc.).
    name: ClassVar[str]

    #: Severity at/above which Status="warn". Subclasses override.
    severity_warn: ClassVar[float] = 0.05

    #: Severity at/above which Status="fail" and Flag=True.
    severity_fail: ClassVar[float] = 0.10

    def __init__(
        self,
        on: ColumnRef,
        groupby: ColumnRefList,
        *,
        severity_warn: float | None = None,
        severity_fail: float | None = None,
        agg_func: str | Callable = "mean",
        n_jobs: int = 1,
    ): ...

    @abstractmethod
    def _compute(self, group: pd.DataFrame) -> pd.DataFrame:
        """Add the check's metric columns to one group.

        Must add at minimum the severity column. May add check-specific
        columns. Flag and Status are computed by the base class from
        severity.
        """

    def analyze(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run the check on every group and return the augmented frame.

        Adds three generic columns:
          QC_<name>_Severity (float)
          QC_<name>_Flag     (bool;  severity >= severity_fail)
          QC_<name>_Status   ("pass" | "warn" | "fail")
        plus whatever the subclass added in _compute.

        Rows are never dropped. Stored in self._latest_measurements.
        """

    def summary(self) -> pd.DataFrame:
        """One row per group with counts and severity stats.

        Columns: *self.groupby, num_rows, num_flagged, max_severity, status.
        """

    def flagged_keys(self) -> list[tuple[str, int]]:
        """Return (Metadata_ImageFile, ObjectLabel) tuples for flagged rows.

        Used by the GUI 'Mark all flagged for removal' button. Requires
        the analyzed frame to carry both ``Metadata_ImageFile`` and
        ``ObjectLabel`` columns (the curation key used by
        ``STORE_REMOVED_KEYS``). Returns an empty list when those
        columns are absent or when no rows were flagged.
        """

    @classmethod
    def severity_col(cls) -> str: return f"QC_{cls.name}_Severity"
    @classmethod
    def flag_col(cls)     -> str: return f"QC_{cls.name}_Flag"
    @classmethod
    def status_col(cls)   -> str: return f"QC_{cls.name}_Status"

    # ----- SetAnalyzer abstract-method conformance ----------------------
    # SetAnalyzer declares four abstract methods (analyze, show, results,
    # _apply2group_func). We override analyze() above, give the rest
    # concrete defaults so QualityCheck remains instantiable, and let
    # subclasses override show()/dash() for check-specific plots.

    def results(self) -> pd.DataFrame:
        """Return the augmented frame stored by the most recent analyze()."""
        return self._latest_measurements

    @staticmethod
    def _apply2group_func(group: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Not used by QualityCheck — implement _compute on the subclass.

        QualityCheck.analyze() drives group iteration directly via
        _compute(); the abstract _apply2group_func from SetAnalyzer is
        satisfied here purely to keep the class instantiable. Raises
        NotImplementedError so accidental external calls fail loudly.
        """
        raise NotImplementedError(
            "QualityCheck subclasses implement _compute(group), not "
            "_apply2group_func. analyze() drives the iteration."
        )

    def show(self, *args, **kwargs):
        """QualityCheck plots are Plotly-only — see dash().

        SetAnalyzer's matplotlib show() is not implemented for QC
        because the QC tab is Plotly-driven. Raising rather than
        falling back to a placeholder so notebook users discover
        the right method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement matplotlib "
            f"show(); use dash() for interactive output."
        )
```

**Status tri-state semantics:**

| severity range | Flag | Status |
|---|---|---|
| `< severity_warn` | False | `"pass"` |
| `[severity_warn, severity_fail)` | False | `"warn"` |
| `≥ severity_fail` | True | `"fail"` |

Only `"fail"` triggers `Flag=True`. The "warn" tier is informational and
the GUI renders it in a distinct color but does *not* pre-select those
rows for removal — the user decides what to curate.

### `ExpectedVsDetectedCount`

**File:** `src/phenotypic/analysis/_expected_vs_detected.py`

```python
class ExpectedVsDetectedCount(QualityCheck):
    """Flag groups whose detected colony count diverges from metadata."""

    name = "Count"
    severity_warn = 0.05
    severity_fail = 0.10
    _measurement_infoclass = QUALITY_COUNT

    def __init__(
        self,
        metadata: pd.DataFrame | Path | str,
        groupby: ColumnRefList,           # cols present in BOTH frames
        on: ColumnRef = "ObjectLabel",    # what counts as "detected"
        *,
        severity_warn: float | None = None,
        severity_fail: float | None = None,
        n_jobs: int = 1,
    ):
        # NOTE: ExpectedVsDetectedCount does not expose agg_func — the
        # check is a row-count, not a value aggregation. We pin
        # super().agg_func = "first" internally and override
        # OperationRegistry-driven param-form rendering for this class
        # to hide the field. Passing agg_func via the GUI would
        # silently no-op and confuse the user.
        ...
```

**Behavior:** for each `groupby` combination,
`expected = metadata.groupby(groupby).size()[key]`,
`detected = len(group)`,
`delta = detected - expected`,
`severity = abs(delta) / expected`.

**Unmatched groups** (a group key in the measurement frame that doesn't
appear in the metadata's groupby index) get
`Expected = 0` → `Severity = inf` → `Status = "fail"`, **AND** are
also recorded in the analyzer's `unmatched_groups` attribute (list of
tuples). The QC tab's card-body refresh callback surfaces these in
the per-card summary strip ("3 measurement groups had no metadata
counterpart") so the user can tell a "real fail" apart from a
metadata-mismatch fail. This is a fail-noisily-but-don't-crash
choice — the alternative (raise at analyze() time) would kill the
entire QC card with a stack-trace toast and offer no hint about
which group caused it.

**Output columns** (broadcast across every row in each group):

| Column | Type | Meaning |
|---|---|---|
| `QC_Count_Detected` | int | Rows in `data` for the group |
| `QC_Count_Expected` | int | Rows in `metadata` for the group |
| `QC_Count_Delta` | int | `Detected − Expected` (signed) |
| `QC_Count_Severity` | float | `\|Delta\| / Expected`; `numpy.inf` when `Expected == 0` (always exceeds `severity_fail` → `Status="fail"`, `Flag=True`) |
| `QC_Count_Flag` | bool | Base class |
| `QC_Count_Status` | str | Base class |

**Contract:** `metadata` must contain every column listed in `groupby`.
Validation runs at `__init__` time so the user gets a clear `KeyError`
before they call `analyze`.

**`dash()`:** horizontal lollipop chart of `Delta` per group, baseline
at zero, lollipop heads colored by `Status`. Hover shows
`Detected / Expected / Severity`.

### `ReplicateAgreement`

**File:** `src/phenotypic/analysis/_replicate_agreement.py`

```python
class ReplicateAgreement(QualityCheck):
    """Flag (group, time) bins with poor agreement across replicates."""

    name = "SE"
    severity_warn = 0.10
    severity_fail = 0.20
    _measurement_infoclass = QUALITY_SE

    def __init__(
        self,
        on: ColumnRef,
        groupby: ColumnRefList,
        time_label: ColumnRef = "Metadata_Time",
        *,
        severity_warn: float | None = None,
        severity_fail: float | None = None,
        min_replicates: int = 2,
        eps: float = 1e-9,   # |mean| below this → severity=NaN guard
        agg_func: str = "mean",
        n_jobs: int = 1,
    ): ...
```

**Behavior:** within each group, split by `time_label`, compute
`SE = stddev / sqrt(n)` and `mean` across replicates per timepoint.
`severity = abs(SE) / abs(mean)`. Broadcast back to every replicate in
the `(group, time)` bin.

Three guard paths produce `severity = NaN` and `Status = "pass"` so
under-powered or degenerate bins never gate curation:

1. **`n < min_replicates`** — too few replicates for a meaningful SE.
2. **`|mean| < eps`** (default `eps = 1e-9`) — the relative-SE ratio
   blows up at zero mean (t=0 baselines, blank wells, true-zero
   conditions). Without this guard, `severity = inf` would flag every
   row in those bins.
3. **`stddev == 0` and `mean == 0`** — degenerate bin (all replicates
   exactly zero). The ratio is mathematically undefined; treat as
   pass.

`eps` is exposed as a constructor parameter with the default above so
heavy-tailed datasets can tune it; the default `1e-9` is small enough
to flag near-zero measurements that are genuinely above the
noise floor and large enough to catch sensor-zero readouts.

**Output columns** (broadcast per `(group, time)` bin):

| Column | Type | Meaning |
|---|---|---|
| `QC_SE_Value` | float | Raw `SE = stddev / sqrt(n)` |
| `QC_SE_Mean` | float | Mean across replicates |
| `QC_SE_CV` | float | `stddev / \|mean\|` (different from severity) |
| `QC_SE_NumReplicates` | int | Replicate count contributing to SE |
| `QC_SE_Severity` | float | `\|SE\| / \|mean\|`; NaN when reps < min |
| `QC_SE_Flag` | bool | Base class |
| `QC_SE_Status` | str | Base class |

**`dash()`:** mean ± SE band per group across time (similar shape to
`ModelFitter.dash()` minus the predicted curve). Each group's line is
colored by the worst-status timepoint in that group.

### `MeasurementInfo` classes

There is **no** `QUALITY_CHECK` enum. The generic Flag/Severity/Status
columns are dynamically named per-subclass (`QC_<name>_Flag` etc.),
which a static `MeasurementInfo` enum cannot express — its members
would render as `QC_Flag`/`QC_Severity`/`QC_Status` (the wrong column
names) and `append_rst_to_doc` would produce misleading documentation.
The generic columns are documented in the `QualityCheck` class
docstring as free-text, alongside the column-composition rule. Each
concrete subclass owns its **own** per-check MeasurementInfo for its
non-generic fields (Detected/Expected/Delta, SE/Mean/CV, …):

```python
# tools_/measurement_info/_quality_count.py
class QUALITY_COUNT(MeasurementInfo):
    DETECTED = ("Detected", "Detected colony count in the group.")
    EXPECTED = ("Expected", "Expected colony count from the metadata frame.")
    DELTA    = ("Delta",    "Detected − Expected (signed; negative = missing).")
    @classmethod
    def category(cls) -> str: return "QC_Count"

# tools_/measurement_info/_quality_se.py
class QUALITY_SE(MeasurementInfo):
    VALUE          = ("Value",          "Raw SE = stddev / sqrt(n) across replicates.")
    MEAN           = ("Mean",           "Mean across replicates at this (group, time).")
    CV             = ("CV",             "Coefficient of variation, stddev / |mean|.")
    NUM_REPLICATES = ("NumReplicates",  "Replicate count contributing to the SE.")
    @classmethod
    def category(cls) -> str: return "QC_SE"
```

Each concrete check's docstring composes its `_measurement_infoclass`
documentation via the existing `append_rst_to_doc` pattern used by
`EdgeCorrector`.

---

## GUI architecture

### Module layout

```
src/phenotypic/gui/
├── _param_forms.py                # already shared (no refactor needed)
├── _operation_registry.py         # MOD: register "quality_check" category
├── _schema_cache.py               # NEW: moved up from gui/analysis/
├── _qc_recipe.py                  # NEW: sidecar persistence
├── analysis/_schema_cache.py      # DELETE (replaced by gui/_schema_cache.py)
├── FEATURES.md                    # MOD: rows for QC + Heatmap tabs
└── WORKFLOWS.md                   # MOD: rows for new end-to-end flows

src/phenotypic/gui/results_viewer/
├── _qc_tab/
│   ├── __init__.py
│   ├── _layout.py                 # check-card list, add-check menu
│   ├── _callbacks.py              # STORE_REMOVED_KEYS → recompute cards
│   ├── _check_card.py             # one card per configured QualityCheck
│   └── _ids.py
├── _heatmap_tab/
│   ├── __init__.py
│   ├── _layout.py                 # picker strip + figure container
│   ├── _callbacks.py              # STORE_REMOVED_KEYS + picker changes
│   ├── _figure.py                 # pure Plotly-figure builder (testable)
│   └── _ids.py
├── _layout.py                     # MOD: 4 tabs (Plate / Colony / QC / Heatmap)
├── _callbacks.py                  # MOD: register QC + Heatmap callbacks
├── _ids.py                        # MOD: TAB_QC_ID, TAB_HEATMAP_ID, etc.
└── _app.py                        # MOD: stash QcRecipe on app.server.config
```

### Shared-state refactor: move `_schema_cache.py`

The schema is just "list columns in `measurements.parquet` /
`master_measurements.parquet`" — no analysis-specific logic. Moving it up
one level is a mechanical rename + ~3 import-site updates in the
analysis sub-app. Public API (`MeasurementSchema.columns_for(source)`)
stays identical.

### Live-recompute path

```
User removes colony in Colony tab
    │
    ▼
STORE_REMOVED_KEYS dcc.Store updates
    │
    ├──▶ FilteredMeasurements.save()  (existing path)
    │       (rewrites measurements.parquet + .csv)
    │
    ├──▶ QC tab callback:
    │       For each configured QualityCheck → analyze() → re-render card
    │
    └──▶ Heatmap tab callback:
            Rebuild heatmap figure with the curated cell hidden
```

All three branches fire from the same store change. No polling, no
mtime hacks — a single Dash-app means all three subscribers are on the
same callback graph.

### `QcRecipe` sidecar — `<output>/.viewer_cache/qc_recipe.json`

**Schema:**

```json
{
  "version": 1,
  "checks": [
    {
      "instance_id": "qc-count-1748391827",
      "class": "ExpectedVsDetectedCount",
      "enabled": true,
      "params": {
        "metadata": "/abs/path/to/metadata.csv",
        "groupby": ["Metadata_ImageFile"],
        "on": "ObjectLabel",
        "severity_warn": 0.05,
        "severity_fail": 0.10
      }
    },
    {
      "instance_id": "qc-se-1748391945",
      "class": "ReplicateAgreement",
      "enabled": true,
      "params": {
        "on": "Size_Area",
        "groupby": ["Metadata_Plate", "Metadata_Strain"],
        "time_label": "Metadata_Time",
        "severity_warn": 0.10,
        "severity_fail": 0.20,
        "min_replicates": 3
      }
    }
  ]
}
```

- `instance_id` is generated at "Add check" time (`f"qc-{name}-{int(time.time())}"`
  plus a small random suffix). Stable for the recipe lifetime; drives
  per-card Dash component IDs.
- `enabled` lets the user toggle a check off without losing its config.
- The library accepts `pd.DataFrame | Path | str` for `metadata`; the
  GUI form is restricted to paths (file picker) so JSON serialization
  is always string-as-path.

**API:**

```python
@dataclass
class QcRecipeEntry:
    cls: type[QualityCheck]
    params: dict[str, Any]
    instance_id: str
    enabled: bool = True

@dataclass
class QcRecipeLoadWarning:
    instance_id: str
    class_name: str
    reason: str

@dataclass
class QcRecipe:
    path: Path                                # .viewer_cache/qc_recipe.json
    entries: list[QcRecipeEntry]
    seed_mtime_ns: int | None = None
    load_warnings: list[QcRecipeLoadWarning] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @classmethod
    def load(cls, output_root_path: Path) -> "QcRecipe": ...

    def save(self) -> None: ...
    def add(self, check_cls, params, *, enabled=True) -> str: ...
    def remove(self, instance_id: str) -> bool: ...
    def update(self, instance_id, *, params=None, enabled=None) -> bool: ...
    def instantiate(self) -> list[tuple[str, QualityCheck]]: ...
```

**Class resolution:** failed instantiations land in `load_warnings`
rather than raising, so a single corrupt entry doesn't break the whole
tab. The on-disk file is left untouched until the user takes a UI
action that triggers a save.

**Atomic write:** `save()` writes to `<path>.tmp` and `os.replace()`'s
it over `<path>` to avoid leaving a half-written JSON if the process
is killed mid-write. Mirrors `FilteredMeasurements._save_locked`.
Before writing, `save()` calls
`path.parent.mkdir(parents=True, exist_ok=True)` so the first save
into a freshly-curated output dir doesn't fail on missing
`.viewer_cache/`.

**Corrupt-JSON load:** when `qc_recipe.json` exists but fails
`json.loads()` (partial write from a previous crash, hand-edit typo),
`load()` returns an empty recipe with a single
`QcRecipeLoadWarning(instance_id="__file__", class_name="",
reason="invalid JSON: <details>")`. The viewer boots, the user sees
a banner explaining the problem, and the file is left untouched
until the user takes a UI action — preserving a chance to recover
the file from VCS or by hand.

**`instance_id` collision avoidance:** generated as
`f"qc-{name}-{secrets.token_hex(4)}"` (8 hex chars). The
[1-in-4-billion] collision probability is negligible even under test
harness parallelism. `time.time()`-based suffixes are rejected because
back-to-back adds within the same second would collide and break
pattern-matching ID uniqueness.

**Staleness detection:** `QcRecipe` does **not** include the
`is_stale()` / mtime-refusal pattern from `RecipeState`. No external
process owns `qc_recipe.json`; the only writer is the viewer itself,
under its `_lock`. Two viewer sessions writing to the same output dir
concurrently is an unsupported configuration (the same is true for
`FilteredMeasurements`), and `qc_recipe.json` is small enough that
the standard `.tmp` + `os.replace` is sufficient protection.

**App-config wiring:**

```python
# results_viewer/_app.py — at create_app() boot
schema = MeasurementSchema(output_root=output_root.root)
app.server.config[CFG_MEASUREMENT_SCHEMA] = schema
qc_recipe = QcRecipe.load(output_root.root)
app.server.config[CFG_QC_RECIPE] = qc_recipe
```

(`MeasurementSchema` is a plain dataclass — constructed directly, not
via a `.load()` classmethod. Matches the existing analysis sub-app
usage at `gui/analysis/_app.py`.)

New `_config.py` constants:

- `CFG_QC_RECIPE = "pheno_qc_recipe"` — the `QcRecipe` instance.
- `CFG_QC_INSTANCES_CACHE = "pheno_qc_instances"` — a single dict
  `{revision: list[QualityCheck]}` with one entry; invalidated on
  every recipe-revision change (read-then-discard, not unbounded).
- `CFG_QC_AUGMENTED_FRAME = "pheno_qc_augmented_frame"` — the latest
  merged filtered + QC-columns frame consumed by the Heatmap tab.
  Single value, overwritten on every card-body refresh; size matters
  more than for the instances cache (~MB scale) so the cap-at-one
  policy is enforced rather than just incidental.

### Shared augmented-frame cache

The Heatmap tab's "color by QC severity" feature needs access to the QC
output columns, which are produced inside each `QualityCheck._latest_measurements`
rather than the plain filtered frame. To avoid the Heatmap callback
re-running every check, the QC tab's recompute callback writes a merged
**augmented frame** to `app.server.config[CFG_QC_AUGMENTED_FRAME]`:

- Start from the polars-converted filtered frame.
- Left-join every configured check's `_latest_measurements` on
  `(Metadata_ImageFile, ObjectLabel)` so QC columns ride along without
  losing rows that fell outside any check's groupby.
- Stash under a `(revision, removed_keys_hash)` key on the app config.

**Avoiding the heatmap-read-before-QC-write race.** Both callbacks
subscribe to `STORE_REMOVED_KEYS`; without ordering, the heatmap can
fire first and silently fall back to the plain filtered frame
(serving pre-curation QC severities until the next interaction). To
fix this we introduce a third store:

```python
dcc.Store(id=ids.STORE_QC_AUGMENTED_REVISION, data=0, storage_type="memory")
```

- The QC tab callback bumps `STORE_QC_AUGMENTED_REVISION` **after**
  it has finished writing `CFG_QC_AUGMENTED_FRAME`.
- The Heatmap callback subscribes to `STORE_QC_AUGMENTED_REVISION` as
  an Input (in addition to `STORE_REMOVED_KEYS`), so it only re-fires
  once the QC writer has completed.
- When no QC checks are configured, the QC callback bumps the store
  with the augmented frame set to `None`, so the heatmap still
  refreshes on curation.

This makes the augmented-frame read deterministic at the cost of one
extra store and one extra callback fire per curation tick — a fair
trade for "no silent stale-data UX." The Heatmap callback otherwise
falls back to the plain filtered frame when
`CFG_QC_AUGMENTED_FRAME is None`.

The color-picker option list also reads from the cached augmented
frame: union of `MeasurementSchema.columns_for("measurements")` plus
any `QC_*_Severity` columns present in the augmented frame.

### Live revision signal

```python
dcc.Store(id=ids.STORE_QC_RECIPE_REVISION, data=0, storage_type="memory")
```

Every `QcRecipe.{add, remove, update}` callback writes back
`revision + 1`. The card-render callback uses two Inputs —
`STORE_QC_RECIPE_REVISION` (recipe changed) and `STORE_REMOVED_KEYS`
(data changed) — so the cheaper path is taken when only data changed:
instantiated `QualityCheck` objects are cached on
`app.server.config[CFG_QC_INSTANCES_CACHE]` keyed by revision and
reused when revision is unchanged.

### Export "QC report" button

```
<output_root>/qc.parquet         long format: union of every check's
                                 analyze() output, with a leading
                                 "QC_Check_Name" column as discriminator

<output_root>/qc_summary.json    one entry per instance:
                                 {instance_id, class, params,
                                  num_rows, num_flagged, max_severity,
                                  status_counts: {pass, warn, fail}}
```

Captures the **post-curation** state — uses the filtered frame, not the
master. Button is disabled when no checks are configured. Toast on
success with the absolute paths.

### QC tab UX

```
┌── QC tab body ────────────────────────────────────────────┐
│ Top strip:  [+ Add check]            [Export QC report ▾] │
│                                                           │
│ ⚠ Load-warning banner   (visible only if recipe had       │
│    unresolved entries; lists instance_id + reason)        │
│                                                           │
│ ┌── Check card  (one per enabled, resolved entry) ─────┐ │
│ │ Title row:                                            │ │
│ │   [status badge]  ExpectedVsDetectedCount #1          │ │
│ │   [edit] [enable toggle] [duplicate] [delete]         │ │
│ │                                                       │ │
│ │ Body:                                                 │ │
│ │   Plotly figure (from check.dash())                   │ │
│ │   ┌── Summary strip ────────────────────────────┐   │ │
│ │   │ groups: 4 | flagged: 1 | max severity: 0.12 │   │ │
│ │   └─────────────────────────────────────────────┘   │ │
│ │   [Mark all flagged for removal]                     │ │
│ └───────────────────────────────────────────────────────┘ │
│                                                           │
│ (more cards…)                                             │
└───────────────────────────────────────────────────────────┘
```

Card components use pattern-matching IDs keyed by `instance_id`:

- `{"type": "qc-card-root",      "index": instance_id}`
- `{"type": "qc-card-figure",    "index": instance_id}`
- `{"type": "qc-card-summary",   "index": instance_id}`
- `{"type": "qc-card-edit",      "index": instance_id}`
- `{"type": "qc-card-delete",    "index": instance_id}`
- `{"type": "qc-card-toggle",    "index": instance_id}`
- `{"type": "qc-card-mark-flag", "index": instance_id}`

`+ Add check` and edit/duplicate flows open one `dbc.Modal`:

1. Class dropdown — `OperationRegistry.get_by_category("quality_check")`.
2. Param-form region — `param_form(info, current_values, columns_provider)`
   re-renders when the class changes.

On submit, the callback reads pattern-matched param-input states (same
trick the analysis sub-app uses) into a `dict[str, Any]`, calls
`QcRecipe.add` or `update`, bumps `STORE_QC_RECIPE_REVISION`, closes the
modal.

### QC tab callbacks (sketch)

The callback graph **splits card lifecycle from body refresh** to avoid
Dash's ALL-pattern length-mismatch error when a card is added or
removed:

- **Card-list render callback** — fires on
  `STORE_QC_RECIPE_REVISION` only. Owns the entire
  `CARDS_CONTAINER.children` list. Atomically rebuilds every card
  shell (root div, title bar, body shell with empty figure
  placeholder) so the DOM card count is always in sync with the
  recipe.
- **Card-body refresh callback** — fires on
  `STORE_REMOVED_KEYS` only (NOT recipe revision). Pattern-matches
  on the existing card bodies and updates `figure` + `summary`
  outputs. Because the recipe-revision branch above has already
  re-rendered the card shells, this callback's `State(... ALL)` and
  `Output(... ALL)` lists are guaranteed to have matching lengths.
- **Augmented-frame write** lives in the card-body refresh
  callback's body, alongside writing
  `CFG_QC_AUGMENTED_FRAME` and bumping
  `STORE_QC_AUGMENTED_REVISION` for the heatmap subscriber.

```python
# Card-body refresh. Fires only on data-change (not recipe-change),
# so the cards already exist with the right indices.
@app.callback(
    Output({"type": "qc-card-figure",  "index": ALL}, "figure"),
    Output({"type": "qc-card-summary", "index": ALL}, "children"),
    Output(STORE_QC_AUGMENTED_REVISION, "data"),
    Input(STORE_REMOVED_KEYS,        "data"),
    State({"type": "qc-card-root", "index": ALL}, "id"),
    State(STORE_QC_AUGMENTED_REVISION, "data"),
)
def _refresh_qc_card_bodies(removed_keys, ids, aug_revision):
    recipe      = current_app.config[CFG_QC_RECIPE]
    filtered    = current_app.config[CFG_FILTERED_STATE]
    output_root = current_app.config[CFG_OUTPUT_ROOT]
    # FilteredMeasurements doesn't carry the master frame; it carries the
    # removed-keys set. Compose the post-curation view from the master
    # frame stashed on OutputRoot at boot.
    frame       = filtered.filtered_df(output_root.master_df)

    # FilteredMeasurements is polars-backed; the analysis library is
    # pandas-native. Convert once per recompute and reuse across all
    # cards. Conversion is cheap relative to analyze() and avoids
    # pandas/polars churn inside each subclass's _compute.
    pandas_frame = frame.to_pandas()

    instances = dict(recipe.instantiate())
    figures, summaries = [], []
    augmented = frame  # accumulator for the heatmap-facing frame
    for component_id in ids:
        instance_id = component_id["index"]
        check = instances.get(instance_id)
        if check is None:
            figures.append(_empty_figure("(removed)"))
            summaries.append("")
            continue
        try:
            result = check.analyze(pandas_frame)
        except Exception as exc:
            # Per-card error isolation: don't let one check kill the
            # whole callback. Surface the error in the card body so the
            # user sees what went wrong (typically: missing column, bad
            # metadata path, NaN-only series).
            figures.append(_error_figure(check_name=type(check).__name__,
                                         message=str(exc)))
            summaries.append(f"error: {exc!s}")
            continue
        figures.append(check.dash())
        summaries.append(_render_summary_strip(check.summary()))
        # Left-join this check's QC columns onto the augmented frame on
        # (Metadata_ImageFile, ObjectLabel) so non-grouped rows are
        # preserved with NaN QC values.
        augmented = _left_join_qc_columns(augmented, result,
                                          on=("Metadata_ImageFile",
                                              "ObjectLabel"))

    current_app.config[CFG_QC_AUGMENTED_FRAME] = (
        augmented if instances else None
    )
    return figures, summaries, (aug_revision or 0) + 1

# "Mark all flagged for removal" — pushes to STORE_REMOVED_KEYS.
@app.callback(
    Output(STORE_REMOVED_KEYS, "data", allow_duplicate=True),
    Input({"type": "qc-card-mark-flag", "index": ALL}, "n_clicks"),
    State(STORE_REMOVED_KEYS, "data"),
    prevent_initial_call=True,
)
def _mark_flagged_for_removal(n_clicks_list, current):
    # Identify which card fired via dash.callback_context, look up its
    # QualityCheck instance, take flagged_keys(), merge into current.
    # NOTE: this writes the same store the curation callbacks own, so
    # `allow_duplicate=True` is required. Downstream consumers (this
    # callback's own recompute path) see the union and re-flow.
    ...
```

### Heatmap tab UX

```
┌── Heatmap tab body ───────────────────────────────────────┐
│ Top strip:                                                │
│   Color column:   [Size_Area                          ▾]  │
│   Aggregator:     [mean ▾]                                │
│   Image:          [plate_001.tif                      ▾]  │
│   Time:           ◯═════════════════════════════ (T=4)    │
│                                                           │
├───────────────────────────────────────────────────────────┤
│              ┌──────────────────────────┐                 │
│              │  Plotly heatmap          │                 │
│              │  rows = Grid_RowNum      │                 │
│              │  cols = Grid_ColNum      │                 │
│              │  removed = hatched grey  │                 │
│              └──────────────────────────┘                 │
│   Hover: (row, col) — value — ImageFile — ObjectLabel     │
└───────────────────────────────────────────────────────────┘
```

- **Color column dropdown** = union of `MeasurementSchema.columns_for("measurements")`
  plus any `QC_*_Severity` columns currently emitted (recipe-revision-aware).
- **Aggregator dropdown:** `mean / median / max / min` (polars `GroupBy.agg`).
  Applied **after** the Image picker filter, so it only fires when
  the selected image has multiple rows sharing a
  `(Grid_RowNum, Grid_ColNum, Metadata_Time)` bin — uncommon for a
  single-image view (typically only happens when the pipeline emitted
  multiple measurements per well, or when the user has not yet split
  by ObjectLabel). For the common one-row-per-well case the aggregator
  is a no-op and any choice yields the same heatmap.
- **Image picker:** unique `Metadata_ImageFile` values in the filtered frame.
- **Time slider:** hidden when only one time point exists, when the
  column is absent, or when `Metadata_Time` is non-numeric and
  `pd.to_numeric(..., errors="coerce")` produces all-NaN (e.g.
  values like `"T0"`, `"baseline"`). When coercion succeeds on most
  values but a few NaN slip through, those rows are dropped from the
  time-axis dimension and a small "skipping N non-numeric time
  values" caption appears below the slider. Marks are placed at every
  unique numeric `Metadata_Time` value (not interpolated).
- **Empty state:** when `Grid_RowNum` / `Grid_ColNum` aren't in the frame
  (i.e. the pipeline did not run a `GridMeasureFeatures` step), the tab
  renders an explanation card. No exceptions. These column names come
  from `GRID.ROW_NUM` / `GRID.COL_NUM` in
  `tools_/measurement_info/_grid.py` and are emitted by
  `GridFinder`-aware pipelines.
- **Removed-cell rendering:** dedicated greyed hatch overlay so the user can
  distinguish "excluded" from "low value." Implemented as a second `go.Heatmap`
  trace at zero opacity (for hover) plus a `go.Scatter` of × markers
  sized to scale with the cell — marker size set to
  `min(14, max(6, cell_px * 0.5))` so the × occupies roughly half a
  cell on 384-well plates and remains visible without obscuring
  neighbors on small (e.g. 2×3) grids. Color is `COLOR_MUTED` from
  `gui/_design.py` to avoid clashing with the data colormap.

### Pure figure builder (`_figure.py`)

```python
def build_heatmap_figure(
    frame: pl.DataFrame,
    *,
    color_col: str,
    image_file: str,
    time_value: float | None,
    aggregator: Literal["mean", "median", "max", "min"],
    removed_keys: set[tuple[str, int]],
    grid_row_col: tuple[str, str] = ("Grid_RowNum", "Grid_ColNum"),
) -> go.Figure:
    ...
```

Pure function, no Dash imports — unit-testable with synthetic frames.

### Heatmap tab callbacks (sketch)

```python
@app.callback(
    Output("heatmap-figure", "figure"),
    Input("heatmap-color-picker",          "value"),
    Input("heatmap-image-picker",          "value"),
    Input("heatmap-time-slider",           "value"),
    Input("heatmap-aggregator-picker",     "value"),
    Input(STORE_QC_AUGMENTED_REVISION,     "data"),  # ordering edge
    Input(STORE_REMOVED_KEYS,              "data"),
)
def _render_heatmap(color, image, t, agg, augmented_revision, removed_keys):
    # Prefer the QC-augmented frame so the color picker can target
    # QC_*_Severity columns. Falls back to the plain post-curation view
    # when no checks are configured (CFG_QC_AUGMENTED_FRAME is None).
    augmented = current_app.config.get(CFG_QC_AUGMENTED_FRAME)
    if augmented is not None:
        frame = augmented
    else:
        filtered    = current_app.config[CFG_FILTERED_STATE]
        output_root = current_app.config[CFG_OUTPUT_ROOT]
        frame       = filtered.filtered_df(output_root.master_df)
    return build_heatmap_figure(
        frame=frame,
        color_col=color, image_file=image, time_value=t,
        aggregator=agg,
        removed_keys=_as_key_set(removed_keys),
    )

# Repopulate dropdowns when schema or recipe changes.
@app.callback(
    Output("heatmap-color-picker", "options"),
    Output("heatmap-image-picker", "options"),
    Output("heatmap-time-slider",  "marks"),
    Output("heatmap-time-slider",  "style"),
    Input(STORE_QC_RECIPE_REVISION, "data"),
    Input(STORE_REMOVED_KEYS,       "data"),
)
def _refresh_heatmap_controls(...): ...
```

### Top-level tab integration

```python
tabs = dbc.Tabs(
    [
        dbc.Tab(cards_column,     label="Plate",   tab_id=ids.TAB_PLATE_ID),
        dbc.Tab(colony_tab_body,  label="Colony",  tab_id=ids.TAB_COLONY_ID),
        dbc.Tab(qc_tab_body,      label="QC",      tab_id=ids.TAB_QC_ID),
        dbc.Tab(heatmap_tab_body, label="Heatmap", tab_id=ids.TAB_HEATMAP_ID),
    ],
    id=ids.TABS_ID,
    active_tab=ids.TAB_PLATE_ID,
)
```

Both new tab bodies stay mounted at all times so switching is CSS-only,
matching the existing Plate/Colony behavior.

---

## Testing plan

```
tests/
├── unit/
│   ├── analysis/
│   │   ├── abc_/
│   │   │   └── test_quality_check.py            # severity → flag/status
│   │   │                                          mapping; flagged_keys();
│   │   │                                          summary() shape; tri-state
│   │   │                                          edges.
│   │   ├── test_expected_vs_detected.py         # 96-well plates with N
│   │   │                                          missing wells → exact
│   │   │                                          Delta/Severity. Metadata
│   │   │                                          KeyError fast-fail.
│   │   └── test_replicate_agreement.py          # known mean/stddev arrays
│   │                                              → exact SE/CV/NormSE.
│   │                                              min_replicates NaN path.
│   ├── gui/
│   │   ├── test_qc_recipe.py                    # load(missing) → empty;
│   │   │                                           load(corrupt JSON) →
│   │   │                                           empty + load_warning;
│   │   │                                           add/remove/update + atomic
│   │   │                                           save via .tmp + os.replace;
│   │   │                                           instantiate() with
│   │   │                                           unresolved class →
│   │   │                                           load_warnings; round-trip
│   │   │                                           JSON schema; concurrent
│   │   │                                           save from two threads
│   │   │                                           leaves valid JSON with
│   │   │                                           both writers' entries.
│   │   └── test_heatmap_figure.py               # pure build_heatmap_figure:
│   │                                              grid pivot, aggregator
│   │                                              semantics (incl. NaN-only
│   │                                              pivots from missing
│   │                                              (image, row, col, time)
│   │                                              combos), removed-cell
│   │                                              overlay traces present,
│   │                                              non-numeric time column
│   │                                              → empty-state path.
│   └── tools_/
│       ├── test_quality_count_info.py           # QUALITY_COUNT category +
│       │                                          DETECTED/EXPECTED/DELTA
│       │                                          labels & headers.
│       └── test_quality_se_info.py              # QUALITY_SE category +
│                                                  VALUE/MEAN/CV/NUM_REPLICATES
│                                                  labels & headers.
└── gui/
    ├── test_qc_tab.py                           # Dash smoke test: add a Count
    │                                              check, remove a colony,
    │                                              assert card Severity updates
    │                                              without manual refresh.
    └── test_heatmap_tab.py                      # color-picker, image-picker,
                                                   time slider visibility,
                                                   removed-cell overlay.
```

- **Doctests:** every concrete `QualityCheck` subclass ships two doctest
  examples (basic + advanced). Follows the existing analyzer-doctest
  precedent (`EdgeCorrector`, `LogGrowthModel`): construct a synthetic
  `pd.DataFrame` inline and mark the `analyze()` line with
  `# doctest: +SKIP` when the runtime cost or randomness would make
  literal output matching fragile. Project convention prefers
  `load_synth_yeast_plate()` for image-based examples; analyzer
  doctests operate on frames, so the synthetic-DataFrame route is the
  established analyzer pattern. Run in CI under
  `pytest --doctest-modules`.
- **Library coverage gate:** ≥ 90% on the new files (matches existing
  analysis-module bar).
- **GUI smoke-test gate:** playwright fixtures used by `tests/gui/` carry
  through; new tests follow the same shape as the existing
  `test_colony_view_*` tests.

---

## Ledger updates

### `FEATURES.md`

```
| QC tab — Add check button                  | ✅ shipping | tests/gui/test_qc_tab.py::test_add_check_modal |
| QC tab — Per-check edit modal              | ✅ shipping | tests/gui/test_qc_tab.py::test_edit_check_modal |
| QC tab — Per-check enable toggle           | ✅ shipping | tests/gui/test_qc_tab.py::test_toggle_check_enabled |
| QC tab — Per-check delete button           | ✅ shipping | tests/gui/test_qc_tab.py::test_delete_check |
| QC tab — Per-check duplicate button        | ✅ shipping | tests/gui/test_qc_tab.py::test_duplicate_check |
| QC tab — Status badge (pass/warn/fail)     | ✅ shipping | tests/gui/test_qc_tab.py::test_status_badge_colors |
| QC tab — Plotly figure auto-refresh        | ✅ shipping | tests/gui/test_qc_tab.py::test_card_refresh_on_curation |
| QC tab — Summary strip                     | ✅ shipping | tests/gui/test_qc_tab.py::test_summary_strip_counts |
| QC tab — Mark-flagged-for-removal button   | ✅ shipping | tests/gui/test_qc_tab.py::test_mark_flagged_pushes_to_removed_keys |
| QC tab — Export QC report button           | ✅ shipping | tests/gui/test_qc_tab.py::test_export_emits_qc_parquet_and_summary |
| QC tab — Load-warning banner               | ✅ shipping | tests/gui/test_qc_tab.py::test_load_warning_banner |
| Heatmap tab — Color column dropdown        | ✅ shipping | tests/gui/test_heatmap_tab.py::test_color_picker_lists_measurements_and_qc_severities |
| Heatmap tab — Aggregator dropdown          | ✅ shipping | tests/gui/test_heatmap_tab.py::test_aggregator_semantics |
| Heatmap tab — Image picker                 | ✅ shipping | tests/gui/test_heatmap_tab.py::test_image_picker |
| Heatmap tab — Time slider                  | ✅ shipping | tests/gui/test_heatmap_tab.py::test_time_slider_visibility |
| Heatmap tab — Removed-cell overlay         | ✅ shipping | tests/gui/test_heatmap_tab.py::test_removed_cells_visually_distinct |
| Heatmap tab — Empty-state (no grid)        | ✅ shipping | tests/gui/test_heatmap_tab.py::test_empty_state_when_no_grid |
| STORE_QC_RECIPE_REVISION                   | ✅ shipping | tests/unit/gui/test_qc_recipe.py::test_revision_bumps_on_mutation |
```

### `WORKFLOWS.md`

```
| qc-curation-loop    | Configure Count + SE checks; watch metrics improve as you curate flagged colonies        | scripts/capture_gui_tutorial_screenshots.py::_capture_qc_curation_loop    | docs/source/tutorials/gui/qc_curation_loop.rst    |
| heatmap-exploration | Pick a measurement and walk through time on a plate; spot edge/contamination patterns    | scripts/capture_gui_tutorial_screenshots.py::_capture_heatmap_exploration | docs/source/tutorials/gui/heatmap_exploration.rst |
```

Each requires:

- `_capture_<id>` function in `scripts/capture_gui_tutorial_screenshots.py`
  (CI gate enforces presence via `scripts/check_workflows_md.py`).
- Walkthrough `.rst` page under `docs/source/tutorials/gui/`.
- Refreshed PNGs from
  `uv run python scripts/capture_gui_tutorial_screenshots.py`,
  committed alongside source changes (CI builds them on Ubuntu but
  committed PNGs come from a dev workstation for font consistency).

---

## Build sequence

Bottom-up; each step is independently green-able and unblocks the next.

```
1. tools_/measurement_info/_quality_count.py    + tests
2. tools_/measurement_info/_quality_se.py       + tests
   └── gate: uv run pytest tests/unit/tools_/test_quality_*

3. analysis/abc_/_quality_check.py              + tests
   └── gate: uv run pytest tests/unit/analysis/abc_/test_quality_check.py

4. analysis/_expected_vs_detected.py            + tests + doctest
5. analysis/_replicate_agreement.py             + tests + doctest
   └── gate: uv run pytest tests/unit/analysis/test_expected_vs_detected.py
              tests/unit/analysis/test_replicate_agreement.py
              --doctest-modules

6. Refactor gui/analysis/_schema_cache.py → gui/_schema_cache.py
   (mechanical move + 2-3 import-site updates)
   └── gate: uv run pytest tests/

7. gui/_qc_recipe.py                            + tests
8. gui/_operation_registry.py — add an explicit
   `elif issubclass(obj, QualityCheck): category = "quality_check"`
   branch inside `_discover_analyzers` BEFORE the `ModelFitter` check.
   (Without this branch, QC classes silently inherit the default
   `"Filter"` category and the QC tab's add-check dropdown is empty.)
   └── gate: uv run pytest tests/unit/gui/test_qc_recipe.py
              and `_discover_analyzers` discovers QC subclasses

9. results_viewer/_heatmap_tab/_figure.py      + tests
   (pure function — fastest GUI seam to land)
   └── gate: uv run pytest tests/unit/gui/test_heatmap_figure.py

10. results_viewer/_heatmap_tab/_layout.py + _callbacks.py + _ids.py
11. results_viewer/_layout.py — register Heatmap tab
    └── gate: uv run pytest tests/gui/test_heatmap_tab.py

12. results_viewer/_qc_tab/_check_card.py
13. results_viewer/_qc_tab/_layout.py
14. results_viewer/_qc_tab/_callbacks.py
15. results_viewer/_layout.py — register QC tab
16. results_viewer/_app.py — stash QcRecipe + MeasurementSchema on
                              app.server.config
    └── gate: uv run pytest tests/gui/test_qc_tab.py

17. gui/FEATURES.md rows                        (CI gate)
18. gui/WORKFLOWS.md rows                       (CI gate)
19. scripts/capture_gui_tutorial_screenshots.py — _capture_qc_curation_loop,
                                                  _capture_heatmap_exploration
20. docs/source/tutorials/gui/qc_curation_loop.rst
21. docs/source/tutorials/gui/heatmap_exploration.rst
22. uv run python scripts/capture_gui_tutorial_screenshots.py
    + commit refreshed PNGs
    └── gate: pre-commit hook + gui-docs CI workflow

23. Full integration sanity:
    uv run mypy src/phenotypic
    uv run ruff check --fix
    uv run pytest
```

**Order rationale:**

- Steps 1–5 land a fully usable library API (`from phenotypic.analysis
  import ExpectedVsDetectedCount`) before any GUI work, so the user
  can smoke-test in Jupyter while the GUI is being built.
- Step 9 (heatmap figure builder) is a pure function — the fastest
  CI-green GUI seam, builds confidence in the Plotly approach before
  touching layout.
- Heatmap tab (10–11) lands before QC tab (12–15) because it's
  simpler — one figure, no per-card pattern-matching — and surfaces
  any `_param_forms` / schema integration issues early on a low-risk
  surface.
- Ledger + tutorials (17–22) come last because they reference the
  shipped chrome; building them earlier would invite drift.

---

## Design decisions resolved by plan review

The plan-reviewer pass surfaced five open questions. Resolutions below
so implementers don't re-derive them.

1. **`QUALITY_CHECK` MeasurementInfo disposition.** Dropped. Static
   enum members can't express the per-subclass column-name
   composition (`QC_<name>_Flag`), and a placeholder enum would
   generate misleading RST. Generic columns are documented in the
   `QualityCheck` class docstring as free-text. Each concrete check
   still owns its own per-check MeasurementInfo
   (`QUALITY_COUNT`, `QUALITY_SE`).

2. **Heatmap aggregator semantics — before or after image filter?**
   After. The Image picker filters down to a single image first,
   then the aggregator collapses any remaining multi-row
   `(Grid_RowNum, Grid_ColNum, Metadata_Time)` bins. For the typical
   one-row-per-well case the aggregator is a no-op.

3. **`QcRecipe.add()` write semantics.** Writes the sidecar JSON
   immediately on every `add`/`remove`/`update` (atomic via
   `.tmp` + `os.replace`). No deferred-flush risk: if the viewer
   crashes between an `add()` and the next event, the on-disk file
   already reflects the addition.

4. **Export QC report — what's included?** Only `enabled=True` checks
   contribute rows to `qc.parquet`. The discriminator column is
   `QC_Check_Instance_Id` (the recipe's `instance_id`, not the class
   name), so the user can correlate exported rows back to specific
   recipe entries even when multiple instances of the same class
   exist. `qc_summary.json` includes both `instance_id` and `class`
   per entry for the same reason.

5. **`metadata` path absolute vs relative in the recipe.** Stored
   absolute as written by the file picker. Relative paths would be
   more portable across machines but cause the recipe to silently
   resolve a different file when the user changes `cwd` between
   sessions — a quiet data-mismatch failure mode worse than a loud
   "file not found." Acknowledged limitation: if the output dir is
   rsync'd between machines, the user re-picks the metadata path on
   the new machine. Future-proofing: a sentinel like
   `"{output_root}/metadata.csv"` could be supported in a follow-up.

---

## Open questions / risks

- **Metadata-frame schema assumption:** `ExpectedVsDetectedCount`
  requires `metadata.groupby(groupby).size()` to be meaningful — i.e.,
  metadata must contain every column in `groupby`. Real-world metadata
  CSVs often need an `ExpandMetadata` post-processing step to reach
  that shape. If users repeatedly trip on this, v2 could accept a
  `group_extractor` callable or a join-keys mapping.
- **Pre-instantiated `QualityCheck` cache invalidation:** keeping
  instantiated objects on `app.server.config[CFG_QC_INSTANCES_CACHE]`
  keyed by revision means a long-running viewer session could pile up
  defunct instances if the user thrashes the recipe. Solution if it
  becomes a problem: cap the cache at one entry and discard on every
  revision change.
- **Heatmap performance at scale:** a 384-well, 100-image, 20-timepoint
  dataset is ~770k rows. Per-render aggregation must be vectorized via
  polars (which the filter sidebar already uses); naive pandas
  groupby would lock up. Adding to test plan: a smoke test with a
  synthetic 384-well dataset that asserts < 250ms render time.
- **Sub-app QC integration:** if `/analysis/` later needs to render or
  edit QC checks, the sidecar JSON migrates into `pipeline.json` and
  both apps share a `RecipeState`-style loader. Sidecar shape was
  chosen to make that migration mechanical.
- **Threaded Dash + shared `QualityCheck` instances:** when Dash runs
  with `threaded=True`, multiple requests on the same worker can
  concurrently call `analyze()` on a `QualityCheck` cached in
  `CFG_QC_INSTANCES_CACHE`, racing on
  `self._latest_measurements`. Mitigation: the card-body refresh
  callback is the **only** code path that calls `analyze()` on
  cached instances, and Dash serializes callbacks per-worker, so the
  race window is empty in practice. Documented here so a future
  contributor doesn't move `analyze()` to another callback without
  re-examining. Multi-process deployments (`gunicorn --workers N`)
  give each worker its own copy of `app.server.config` — no
  cross-worker race.
- **`agg_func` exposure on `QualityCheck`:** the base class accepts
  `agg_func` through its `SetAnalyzer` constructor signature, but
  no v1 check actually aggregates values (`ExpectedVsDetectedCount`
  counts rows; `ReplicateAgreement` builds SE/Mean/CV statistics
  inside `_compute`). Subclasses are free to expose it if they
  introduce true value aggregation. `OperationRegistry`'s param-form
  rendering driver should skip `agg_func` for `QualityCheck`
  subclasses unless they explicitly opt in via a class attribute
  flag (`_exposes_agg_func: ClassVar[bool] = False` default).
