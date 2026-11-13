# P5 · Process-mode measurements and AutoConvertRaw-GC rollout

**Spec:** Part E (E4, E6)

---

**Goal:** `--mode process` runs a pipeline's measurers that need no detected
objects, and stores their tables in each store's own transaction. Then
AutoConvertRaw-GC adds grid embeddings right after calibration and denoising.

### Task 1: Process mode admits `meas`, and refuses object measurers

**Files:**
- Modify: `src/phenotypic/_cli/_cli_preflight.py:263`.
  `MODE_SLOTS["process"] = frozenset({"ops", "meas"})`.
- Modify: `src/phenotypic/abc_/_measure_features.py`. Add
  `requires_objects: ClassVar[bool] = True`, overridden by `MeasureEmbedding`'s
  property (P4 T1).
- Modify: `src/phenotypic/phenotypicCLI.py`, the read-only half. Add
  `_refuse_object_measurers_in_process_mode` beside
  `_refuse_grid_preset_unlike_cli_grid`: above `--overwrite`, `--restart` and the
  `--dry-run` exit, raising `click.UsageError`.
- Modify: `src/phenotypic/_cli/CLAUDE.md`. Document the refusal and the new
  `MODE_SLOTS`.
- Test: `tests/unit/cli/test_process_mode_measurers.py` (new); keep
  `tests/unit/cli/test_cli_preflight_ordering.py` green.

**Interfaces:**
- Produces: `object_measurers_in_process_mode(pipeline) -> list[str]`, the slot
  paths (`meas:<key>`) of entries whose `requires_objects` is true.

- [ ] **Step 1: Write the failing tests**

```python
def test_process_refuses_a_size_measurer(tmp_path, tiny_inputs):
    result = run_cli_process(tiny_inputs, tmp_path, pipeline=with_meas(MeasureSize()))
    assert result.exit_code == 2
    assert "meas:size" in result.output and "process mode stores no objects" in result.output
    assert not (tmp_path / "out").exists()            # refused before anything is written

def test_process_accepts_grid_embedding(tmp_path, tiny_inputs, stub_backbone):
    assert run_cli_process(tiny_inputs, tmp_path, pipeline=with_meas(grid_embedder())).exit_code == 0

def test_preflight_now_walks_meas_in_process_mode():
    assert MODE_SLOTS["process"] == frozenset({"ops", "meas"})
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them, plus the preflight-ordering and reference-metadata
  process tests** (`grep -rl "MODE_SLOTS\|operations_in_scope" tests/`). They
  pass.
- [ ] **Step 5: Commit.**

### Task 2: Process stores carry the run folder

**Files:**
- Modify: `src/phenotypic/_cli/_cli_process_only.py:200-350`.
  `process_single_apply_only_core` calls `pipeline.embed(image)` after `apply()`,
  and passes `embeddings=` and `upstream=` to `_save_store`. The tables land in
  the `.part` before `_consolidate_store_part`.
- Modify: `src/phenotypic/_cli/_cli_failure_tracker.py:214`.
  `PROCESS_LAYER_SEMANTICS_REVISION = 5`.
- Modify: `src/phenotypic/_cli/CLAUDE.md` and root `CLAUDE.md`, where they state
  "currently `4`". Add the 4 → 5 reason: process mode now runs `meas`.
- Test: `tests/unit/cli/test_process_store_tables.py` (new)

**Interfaces:**
- Consumes: P3 `choose_table_run_id`, P4 `write_embedding_tables`.

- [ ] **Step 1: Write the failing tests**

```python
def test_process_store_has_grid_table_and_stays_consolidated(tmp_path, tiny_inputs, stub_backbone):
    out = run_process(tiny_inputs, tmp_path, pipeline=calibrate_denoise_grid())
    store = one_store(out)
    run = latest_table_run(store)
    assert run.embedding_path("dinov2-vitb14", "grid").is_file()
    root = json.loads((store / "zarr.json").read_text())
    nodes = root["consolidated_metadata"]["metadata"]
    assert f"tables/{run.run_id}/embeddings/dinov2-vitb14/grid" in nodes

def test_process_store_stays_byte_reproducible(tmp_path, tiny_inputs, stub_backbone):
    a = run_process(tiny_inputs, tmp_path / "a", pipeline=calibrate_denoise_grid())
    b = run_process(tiny_inputs, tmp_path / "b", pipeline=calibrate_denoise_grid())
    assert tree_digest(one_store(a)) == tree_digest(one_store(b))

def test_revision_bump_invalidates_old_process_continuation():
    assert PROCESS_LAYER_SEMANTICS_REVISION == 5
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.** With the stub backbone, inference is deterministic.
  The real model's bytes are covered by Task 4's pilot.
- [ ] **Step 4: Run them, plus `tests/integration/cli/test_process_objmap_semantics.py`
  and the existing process-mode tests.** They pass.
- [ ] **Step 5: Commit.** Then run the end-of-plan full sharded regression once,
  on Slurm (README conventions).

### Task 3: AutoConvertRaw-GC: wheel, image and store check (repo `software/AutoConvertRaw-GC`)

**Files:**
- Modify: `vendor/`. Build the wheel from `feat/dvc-embeddings-refactor` with
  `uv build --wheel`, replace `phenotypic-0.20.2-py3-none-any.whl`, and update
  `pyproject.toml:36` and `uv.lock`.
- Modify: `Dockerfile:28-35`:
  - install the `foundation` extra with CPU torch
    (`--index-url https://download.pytorch.org/whl/cpu` for `torch`);
  - download `facebook/dinov2-base` at a pinned revision into `/opt/hf` at build
    time;
  - set `HF_HOME=/opt/hf` and `HF_HUB_OFFLINE=1`.
- Modify: `pipelines/exfab_rig1.pht-pipe.json`. Add
  `"meas": {"grid_embedding": MeasureDinoEmbedding(size="base", regions="grid",
  grid=ManualGridFinder(row_edges=[…], col_edges=[…]))}`. **The rig1 edges are
  supplied by the user. Do not invent them**; stop and ask.
- Modify: `src/acr_core/pipeline_info.py`. Add `n_grid_embeddings`, counting
  `MeasureEmbedding` entries with `regions == "grid"`.
- Modify: `src/acr_core/plan.py`. `run.json` gains `expected_embeddings`.
- Modify: `src/acr_core/process.py`. `verify_store(store, expected_bindings,
  expected_embeddings)` checks that the latest run in `tables.runs` holds that
  many `embeddings` entries, each with `stale: false` and every listed file
  present.
- Modify: `src/acr_core/worker.py:59,177,254,275,303`. Thread
  `expected_embeddings`; a record without it reads as 0.
- Test: `tests/test_verify_store_embeddings.py` (new), plus the existing suite

- [ ] **Step 1: Write the failing tests**

```python
def test_missing_grid_entry_is_refused(tmp_path):
    store = fake_store(tmp_path, runs={"1-abc123def456": {"embeddings": {}}})
    assert "expected 1 embedding" in verify_store(store, expected_bindings=0, expected_embeddings=1)

def test_stale_or_missing_file_is_refused(tmp_path):
    entry = {"stale": True, "files": [{"path": "tables/1-abc123def456/embeddings/m/grid/table.parquet"}]}
    store = fake_store(tmp_path, runs={"1-abc123def456": {"embeddings": {"m/grid": entry}}})
    assert "stale" in verify_store(store, expected_bindings=0, expected_embeddings=1)
    entry["stale"] = False
    store = fake_store(tmp_path / "b", runs={"1-abc123def456": {"embeddings": {"m/grid": entry}}})
    assert "missing" in verify_store(store, expected_bindings=0, expected_embeddings=1)

def test_old_run_record_reads_zero(tmp_path):
    assert read_run_record_without_field(tmp_path).expected_embeddings == 0
```

- [ ] **Step 2: Run `uv run pytest`; confirm the new tests fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `uv run pytest` (unit) and `-m integration` as a Slurm job**
  (`sbatch -p intel -c 4 --mem=8G -t 0:30:00 --wrap "uv run pytest -m integration"`).
  They pass.
- [ ] **Step 5: Build the image.** Run `tools/build_image.sh` (Cloud Build) and
  `acr-hpcc pull-image` as a Slurm job (repo CLAUDE.md), then pin the new
  `:<sha>` and `.sif`.
- [ ] **Step 6: Commit** in AutoConvertRaw-GC with explicit paths.

### Task 4: Pilot, then the backfill decision

No code.

- [ ] **Step 1: Run the HPCC smoke pilot.** Use `configs/smoke-hpcc.yaml`, which
  writes only under `data/acr-hpcc-test/`, with arrays on `intel`. Never use
  `data/` and never `preempt`.
- [ ] **Step 2: Record** per-image `process_s`, `process_peak_rss_mb` and the
  grid-table size, from the worker's metrics files. Set `processing.timeout_s`
  and the GC `memory_mib` from the measured p95.
- [ ] **Step 3: Check one pilot store independently.** Run
  `zarr.open_group(store, mode="r")`, read the grid table with `pq.read_table`,
  and confirm its row count equals the number of grid cells.
- [ ] **Step 4: Ask the user** whether to backfill existing done images with
  `--reprocess-changed`. Every image becomes "changed", because the pipeline
  bytes changed, and the backfill reruns calibration and BM3D denoise. Report
  the measured per-image cost when asking.
