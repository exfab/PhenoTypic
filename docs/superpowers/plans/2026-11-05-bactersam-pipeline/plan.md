# BacterSamPipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `phenotypic.prefab.BacterSamPipeline`, a prefab that reproduces the
detection + measurement pipeline of the bacterial growth-curve run it was built
from (the reference run), without colour
correction or denoising.

**Architecture:** One `PrefabPipeline` subclass in
`src/phenotypic/prefab/_bactersam_pipeline.py`, following the existing prefab
convention (explicit `__init__` kwargs → `super().__init__(ops=..., meas=..., ...)`).
Op and measurer **keys are the run's keys** (`Crop`, `Blur`, `RemoveBG`, `Detector`, …)
so the prefab's serialized `pipe_cfgs`/`meas` equal the run's. Equality with the run is
pinned by a golden test against a trimmed copy of the run's `pipeline.json.pht-pipe`.

**Tech Stack:** pydantic v2 operations, `Sam2` GPU detector (`phenotypic.detect.nn`,
torch loaded lazily on first `apply`), pytest.

**Spec:** none separate — the source of truth is the run's
`deliverables/pipeline.json.pht-pipe` at
the reference run's output folder (kept outside the repo)
plus the user's request (no colour correction, no denoising; state that assumption in the
docstring; name `BacterSamPipeline`).

**Worktree / branch:** `.claude/worktrees/bactersam-prefab`, branch
`feat/bactersam-prefab`, off `main@251e364fa`.

## What the run actually contains (measured 2026-11-05)

The run's pipeline **already omits** colour correction and denoising. Its input,
a tree of colour-corrected, denoised OME-Zarr stores, was
produced by a `--mode process` run whose journal records
`ColorCorrector → CalibrateColorRpcc → ColorDenoise`. The run's own `desc` says so:
*"v16 minus DenoiseBlockMatch: input is … already colour-corrected (CalibrateColorRpcc)
and denoised (ColorDenoise)"*. So nothing is removed from the run; the docstring documents
the upstream step instead.

| Key | Class | Non-default params (run value; op default) |
|---|---|---|
| `Crop` | `CropImage` | left 450, right 480, top 430, bottom 450 (default `None`) |
| `Blur` | `BlurGauss` | sigma 5.0 (2.0) |
| `RemoveBG` | `SubtractGaussian` | sigma 800.0 (50.0); `preserve_range=True` is the default |
| `Detector` | `Sam2` | `input_layer="detect_mat"` ("rgb"); all other fields default (`model_size="tiny"`, `points_per_side=32`, `crop_n_layers=1`, `input_scaling="image_max"`, …) |
| `CircularityFilter` | `RemoveLowCircularity` | cutoff 0.5 (0.785) |
| `BorderFilter` | `RemoveBorderObjects` | border_size 20 (1) |
| `SmallObjectFilter` | `SmallObjectRemover` | min_size 600 (64) |
| `GridLargest` | `KeepSectionLargest` | — |
| meas `MeasureNeighborDist`, `MeasureShape`, `MeasureSize`, `MeasureIntensity`, `MeasureColor` | — | all default |
| meas `MeasureTexture`, `_1`, `_2`, `_3` | `MeasureTexture` | scale `[10]`, `[20]`, `[30]`, `[40]` (default `[5]`) |

Pipeline-level: `nrows=8`, `ncols=12`. Run CLI config: `image_type=GridImage`,
`detect_mode=gray`.

## Decisions

- **D1 — Drop the run's plot.** `PlotColonyMetricOverTime` is bound to this experiment's
  metadata (`Metadata_Strain`, `Metadata_Salinity`, `Metadata_pH`,
  `Metadata_Configuration`, `Metadata_CondReplicate`); a prefab cannot assume those
  columns. `plots=[]`.
- **D2 — Keep four texture measurers, one per scale**, generated from a
  `texture_scales: tuple[int, ...] = (10, 20, 30, 40)` kwarg and keyed
  `MeasureTexture`, `MeasureTexture_1`, … exactly as the run. One `MeasureTexture(scale=[10,20,30,40])`
  would emit the same columns but a different `measurements_by_feature/` file layout, so
  it would no longer be "the run's pipeline".
- **D3 — Crop defaults are the run's** (450/480/430/450 px). They are imager-specific;
  the docstring says so and they are kwargs.
- **D4 — `nrows=8, ncols=12` defaults**, exposed as kwargs.
- **D5 — `Sam2.input_layer` is fixed to `"detect_mat"`**, not a kwarg: the blur and
  background subtraction exist only to shape what SAM2 sees, so exposing `"rgb"` would
  silently turn two ops into no-ops for detection. Other SAM2 knobs are kwargs.
- **D6 — No colour/denoise ops anywhere in the tree**, pinned by a test; the docstring
  names the assumed upstream ops and how to run them (`--mode process --layer rgb` with
  a `ColorCorrector`/`CalibrateColorRpcc` + `ColorDenoise` pipeline).

## Global Constraints

- Keyword-only construction; `BacterSamPipeline()` must not load torch or a checkpoint.
- `import phenotypic.prefab` must not add any module to the set pinned by
  `tests/unit/ci/test_startup_imports.py` (`SUBPACKAGE_EXPECTED_DEFERRALS["phenotypic.prefab"] == ("colour",)`).
  Verified 2026-11-05: importing `Sam2`, `RemoveLowCircularity`, `KeepSectionLargest`,
  `MeasureNeighborDist` after `phenotypic.prefab` adds **zero** non-phenotypic modules.
- Google-style docstring with the PrefabPipeline sections from
  `docs/source/contrib_guide/docstring_style.md:342` (Steps, Best For, Consider Also).
- `uv run` only; `ruff check --fix` with explicit paths only.
- No experiment data or experiment metadata names in committed files (the fixture keeps
  only `version`, `pipe_cfgs`, `meas`, `nrows`, `ncols`).

## Review Focus

1. **Run on a plain `Image`** — `KeepSectionLargest` needs a `GridImage`; a user running
   `--image-type Image` should be refused by preflight, not crash mid-run. Pinned in Task 1
   (`test_requires_grid_image`).
2. **Grayscale input** — `MeasureColor` reads RGB; preflight must report `rgb` required.
   Pinned in Task 1 (`test_requires_rgb`).
3. **CLI staging** — the prefab must stage (Sam2 top-level, nothing in a CPU-only slot):
   `split_pipeline_at_gpu` cuts at `Detector`. Pinned in Task 1.
4. **JSON round-trip** — `BacterSamPipeline.from_json(p.to_json())` returns a
   `BacterSamPipeline` with identical config; auto-covered by
   `tests/smoke/test_serialization.py` once exported, plus Task 1's own test.
5. **Kwargs reach their ops** — a changed `min_object_size` etc. must land on the right op,
   not be swallowed. Pinned in Task 1 (`test_kwargs_reach_their_ops`).

---

### Task 1: `BacterSamPipeline` class, export, and tests

**Files:**
- Create: `src/phenotypic/prefab/_bactersam_pipeline.py`
- Modify: `src/phenotypic/prefab/__init__.py` (import + `__all__` + package docstring mentions SAM2/bacteria)
- Create: `tests/unit/prefab/fixtures/bactersam_reference_pipeline.json` (trimmed run JSON: `version`, `pipe_cfgs`, `meas`, `nrows`, `ncols`, `post: {}`, `filters: {}`, `model: null`, `plots: []`)
- Create: `tests/unit/prefab/test_bactersam_pipeline.py`

**Interfaces:**
- Produces:

```python
class BacterSamPipeline(PrefabPipeline):
    def __init__(
        self,
        *,
        crop_left: int = 450, crop_right: int = 480,
        crop_top: int = 430, crop_bottom: int = 450,
        blur_sigma: float = 5.0,
        background_sigma: float = 800.0,
        sam_model_size: Literal["tiny", "small", "base_plus", "large"] = "tiny",  # use Sam2's own Literal type
        sam_points_per_side: int = 32,
        sam_pred_iou_thresh: float = 0.7,
        sam_stability_score_thresh: float = 0.92,
        sam_min_mask_region_area: int = 100,
        sam_crop_n_layers: int = 1,
        sam_device: str = "auto",
        circularity_cutoff: float = 0.5,
        border_size: int = 20,
        min_object_size: int = 600,
        texture_scales: tuple[int, ...] = (10, 20, 30, 40),
        nrows: int = 8, ncols: int = 12,
        benchmark: bool = False, verbose: bool = False,
    ) -> None
```

  `ops` dict keys, in order: `Crop, Blur, RemoveBG, Detector, CircularityFilter,
  BorderFilter, SmallObjectFilter, GridLargest`. `meas` keys: `MeasureNeighborDist,
  MeasureShape, MeasureSize, MeasureIntensity, MeasureColor`, then one `MeasureTexture`
  per scale keyed `MeasureTexture`, `MeasureTexture_1`, …. Use the exact types of the
  underlying fields (`Sam2.model_fields[...]`) for kwarg annotations rather than
  re-declaring Literals.

- [ ] **Step 1: Write the failing tests** in `tests/unit/prefab/test_bactersam_pipeline.py`

```python
FIXTURE = Path(__file__).parent / "fixtures" / "bactersam_reference_pipeline.json"

def _sections(p):  # pipe_cfgs/meas/nrows/ncols of a pipeline's JSON
    d = json.loads(p.to_json()); return {k: d[k] for k in ("pipe_cfgs", "meas", "nrows", "ncols")}

def test_default_config_equals_the_reference_run():
    # normalise the fixture through the current schema, so a field added since 0.20.0 is not a false diff
    assert _sections(BacterSamPipeline()) == _sections(ImagePipeline.from_json(FIXTURE))

def test_no_colour_correction_or_denoising_anywhere_in_the_tree():
    banned = {"ColorCorrector", "CalibrateColorRpcc", "ColorDenoise", "DenoiseBlockMatch",
              "EnhanceBlockMatch", "BayesShrinkCorrector", "VisuShrinkCorrector"}
    names = {type(op).__name__ for _, op in find_operations(BacterSamPipeline(), lambda o: True)}
    assert not names & banned

def test_construction_loads_no_torch():  # subprocess: fresh interpreter, assert "torch" not in sys.modules

def test_sam2_reads_detect_mat():
    assert BacterSamPipeline().get_ops()["Detector"].input_layer == "detect_mat"

def test_texture_measurers_one_per_scale():
    meas = BacterSamPipeline(texture_scales=(7, 9)).meas
    assert [k for k in meas if k.startswith("MeasureTexture")] == ["MeasureTexture", "MeasureTexture_1"]
    assert [meas[k].scale for k in ("MeasureTexture", "MeasureTexture_1")] == [[7], [9]]

def test_kwargs_reach_their_ops():  # parametrize: (kwarg, value, op key, field)
    # crop_left→Crop.left, blur_sigma→Blur.sigma, background_sigma→RemoveBG.sigma,
    # sam_points_per_side→Detector.points_per_side, circularity_cutoff→CircularityFilter.cutoff,
    # border_size→BorderFilter.border_size, min_object_size→SmallObjectFilter.min_size, nrows→pipeline.nrows

def test_stages_at_the_detector():
    plan = split_pipeline_at_gpu(BacterSamPipeline())
    assert plan.gpu_path == ("Detector",)

def test_requires_grid_image():   # preflight_requirements of the pipeline (or its ops) reports grid_image True
def test_requires_rgb():          # … reports rgb True (MeasureColor)

def test_json_round_trip_keeps_class_and_config():
    p = BacterSamPipeline(min_object_size=700)
    q = BacterSamPipeline.from_json(p.to_json())
    assert type(q) is BacterSamPipeline and _sections(q) == _sections(p)
```

  For the two preflight tests, read how `tests/unit/abc_/test_preflight_requirements.py`
  and the CLI preflight aggregate a pipeline's requirements, and call that same entry point.

- [ ] **Step 2: Run them, expect ImportError**

Run: `uv run pytest tests/unit/prefab/test_bactersam_pipeline.py -q -p no:cacheprovider`
Expected: collection error, `cannot import name 'BacterSamPipeline'`.

- [ ] **Step 3: Implement** `BacterSamPipeline` in `src/phenotypic/prefab/_bactersam_pipeline.py`
  and export it. Docstring (PrefabPipeline style) must contain:
  - summary: SAM2-based detection and measurement of bacterial colonies on arrayed agar plates;
  - an **"Input assumptions"** paragraph: images are expected to be **already colour-corrected
    and denoised**; this pipeline performs neither. Name the upstream ops
    (`ColorCorrector` / `CalibrateColorRpcc`, `ColorDenoise`) and the route
    (`python -m phenotypic --mode process --layer rgb` with those ops, then feed the
    resulting OME-Zarr tree as `--input`);
  - **Steps** 1–8 matching the ops table; **Measurements**;
  - **Best For** (bacterial colonies on 96-well grid plates; colonies that classical
    thresholds merge with agar texture), **Consider Also** (`HeavyOtsuPipeline`,
    `HeavyWatershedPipeline`, `RoundPeaksPipeline`);
  - a note that `Sam2` needs a GPU for practical throughput and that the CLI stages it
    automatically; crop defaults are imager-specific.
  - `Args:` for every kwarg. No doctest that runs SAM2 (needs weights); a construction-only
    example is fine.

- [ ] **Step 4: Run the tests, expect PASS**

Run: `uv run pytest tests/unit/prefab/ -q -p no:cacheprovider`
Expected: all pass, including the existing `test_prefab_measures_size.py` (it has both
`MeasureShape` and `MeasureSize`).

- [ ] **Step 5: Lint**

Run: `uv run ruff check --fix src/phenotypic/prefab/_bactersam_pipeline.py src/phenotypic/prefab/__init__.py tests/unit/prefab/test_bactersam_pipeline.py`

- [ ] **Step 6: Commit** (explicit paths)

```bash
git add src/phenotypic/prefab/_bactersam_pipeline.py src/phenotypic/prefab/__init__.py \
        tests/unit/prefab/test_bactersam_pipeline.py tests/unit/prefab/fixtures/bactersam_reference_pipeline.json
git commit -m "feat(prefab): BacterSamPipeline — SAM2 bacterial colony pipeline"
```

### Task 2: GUI builder round-trip + docs

**Files:**
- Modify: `tests/gui/builder/test_state_roundtrip.py:833-840` — add `"BacterSamPipeline"` to the
  `test_scalar_param_prefab_round_trips_byte_identical` parametrization; update the
  "Of the seven prefabs" count in the class docstring (`:770`) to eight.
- Modify: `docs/source/explanation/prefab_pipelines_guide.md` — overview-table row
  (`BacterSamPipeline` | Bacteria | 96-well grid | SAM2 on blurred, background-subtracted
  `detect_mat`) and a `## BacterSamPipeline` section (Best for / Pipeline steps /
  Measurements / When to choose / **Input assumptions**).
- Modify: `src/phenotypic/abc_/_prefab_pipeline.py` — add item 6 to "Available
  PrefabPipeline Subclasses".
- Modify: `tests/unit/ci/test_startup_imports.py:281` comment ("five of the seven
  prefab pipelines") → recount and correct only if the number is now wrong.

- [ ] **Step 1: Add the parametrize entry, run it**

Run: `QT_QPA_PLATFORM=offscreen uv run pytest "tests/gui/builder/test_state_roundtrip.py::TestPrefabRoundTrip" -q -p no:cacheprovider`
Expected: PASS. If it fails on a `Sam2` field (e.g. `checkpoint=None`), stop and report the
diff — that is a builder round-trip bug in `Sam2`, not something to paper over in the prefab.

- [ ] **Step 2: Docs edits** (prose only; no executed notebooks touched).

- [ ] **Step 3: Commit** (explicit paths), `docs(prefab): document BacterSamPipeline`.

### Task 3: Verification

- [ ] **Step 1: Affected surface, once** — one `short`-partition Slurm job
  (`--cpus-per-task=8 --mem=32G --time=0:30:00`), per the `run-phenotypic-test` skill
  (`QT_QPA_PLATFORM=offscreen`, `-n $SLURM_CPUS_PER_TASK`, `-p no:cacheprovider`, no `-x`):
  `tests/unit/prefab tests/smoke/test_serialization.py tests/unit/ci/test_startup_imports.py
  tests/unit/ci/test_deferred_imports.py tests/unit/abc_/test_preflight_requirements.py
  tests/gui/builder/test_state_roundtrip.py tests/unit/cli/test_cli_pipeline_split.py`.
  Expected: all pass.

- [ ] **Step 2: `uv run mypy src/phenotypic/prefab/_bactersam_pipeline.py`** — no new errors.

- [ ] **Step 3: Real-data GPU smoke (recommended).** Write `BacterSamPipeline().to_json()`
  to the scratch dir; run `python -m phenotypic` on **4** stores from
  the run's input tree (one dataset folder) with
  `--image-type GridImage --nrows 8 --ncols 12` on `exfab` or `short_gpu` (pick by
  `--test-only` predicted start; `--constraint=gpu_latest` on `short_gpu`). Output to a
  scratch tree on shared storage, never the run folder. Compare each image's
  `tables/measurements/table.parquet` against the same stem in the run's
  `results/<dataset>/zarr/`: object count and `Size_Area` per label.
  Expected: identical; any difference is reported with numbers, not tolerated silently
  (SAM2 on a different GPU model may differ — say which GPU each ran on).
  **Nothing from this check enters the repo** (user, 2026-11-05): the images, the
  job script and the comparison script all live in the scratch tree, and the CLI
  runs in local mode (no `--slurm`) inside one job.

- [ ] **Step 4: Final review** — one reviewer over the branch diff; then hand back for
  push/PR decision (no push without asking).
