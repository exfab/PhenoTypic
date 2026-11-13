# P5 · Process-mode measurements, in place, and the AutoConvertRaw-GC rollout

**Spec:** Part E (E4, E6, E7)

---

**Goal:** `--mode process` runs a pipeline's measurers that need no detected
objects, and stores their tables in each store's own transaction. With
`--output` equal to `--input`, it adds the pipeline's run folder to the input
stores in place, without rewriting pixels. Then AutoConvertRaw-GC adds grid
embeddings right after calibration and denoising.

### Task 1: Process mode admits `meas`, and refuses what it cannot store

**Files:**
- Modify: `src/phenotypic/_cli/_cli_preflight.py:263-267`.
  `MODE_SLOTS["process"] = frozenset({"ops", "meas"})`. This also widens
  `reference_operations_in_scope` (:1588) to process-mode measurers.
- Modify: `src/phenotypic/abc_/_measure_features.py:49`. Add the property
  `requires_objects -> bool`, returning `True`; `MeasureEmbedding` overrides it
  (P4 T1).
- Create: `src/phenotypic/_cli/_cli_process_meas.py` with the shared check.
- Modify: `src/phenotypic/phenotypicCLI.py`, the read-only half. Call it beside
  `_refuse_grid_preset_unlike_cli_grid` (:716, called at :2739): above
  `--overwrite`, `--restart` and the `--dry-run` exit, raising
  `click.UsageError`.
- Modify: `src/phenotypic/_cli/_cli_process_only.py:267`.
  `process_single_apply_only_core` runs the same check before reading the
  image, because AutoConvertRaw-GC calls it directly
  (`acr_core/process_one.py:118`) and never passes the CLI's refusal. It raises
  `ValueError` there.
- Modify: `src/phenotypic/_cli/CLAUDE.md`. Document the refusals and the new
  `MODE_SLOTS`, and update the bullet "A `SubtractBlank` inside a measurer needs
  the table in full mode and not in process mode" (:123): process mode now runs
  `meas`, so it needs the table too.
- Create (helpers), in `tests/_tiny_run.py`: `with_meas(**measurers) -> Path`,
  a pipeline file holding `ops=[BlurGauss(sigma=1.0)]` and those measurers;
  `grid_embedder(**kw) -> MeasureDinoEmbedding`, a `regions="grid"` embedder
  over a 2 × 2 `ManualGridFinder` inside the synth plate
- Test: `tests/unit/cli/test_process_mode_measurers.py` (new); keep
  `tests/unit/cli/test_cli_preflight_ordering.py` green.

**Interfaces:**
- Produces:
  - `object_measurers_in_process_mode(pipeline) -> list[str]`: the slot paths
    (`meas:<key>`) of entries whose `requires_objects` is true
  - `process_meas_refusal(pipeline, *, layer: str, process_format: str) -> str | None`:
    one line naming the problem, or `None`. It refuses:
    - any object measurer: "process mode stores no objects", naming each slot;
    - any `meas` entry when the output is not a store (`--process-format tiff`,
      `--layer objmap`, `--layer detect_mat`), which writes a flat file with
      nowhere to put a table (`_cli_process_only.py:252-264`).

- [ ] **Step 1: Write the failing tests**

```python
def test_process_refuses_a_size_measurer(tmp_path, tiny_inputs):
    result = run_cli(tiny_inputs, tmp_path / "out", mode="process", pipeline=with_meas(size=MeasureSize()),
                     extra=["--layer", "rgb", "--process-format", "zarr"])
    assert result.exit_code == 2
    assert "meas:size" in result.output and "process mode stores no objects" in result.output
    assert not (tmp_path / "out").exists()            # refused before anything is written

@pytest.mark.parametrize("extra", [["--layer", "rgb", "--process-format", "tiff"],
                                   ["--layer", "detect_mat"], ["--layer", "objmap"]])
def test_process_refuses_meas_without_a_store_output(tmp_path, tiny_inputs, stub_backbone, extra):
    result = run_cli(tiny_inputs, tmp_path / "out", mode="process",
                     pipeline=with_meas(emb=grid_embedder()), extra=extra)
    assert result.exit_code == 2 and "store" in result.output

def test_the_core_refuses_what_the_cli_refuses(tmp_path, tiny_inputs):
    with pytest.raises(ValueError, match="process mode stores no objects"):
        process_single_apply_only_core(with_meas(size=MeasureSize()), next(tiny_inputs.rglob("*.tif")),
                                       tiny_inputs, tmp_path / "out", "Image", "rgb", {},
                                       process_format="zarr")

def test_process_accepts_grid_embedding(tmp_path, tiny_inputs, stub_backbone):
    result = run_cli(tiny_inputs, tmp_path / "out", mode="process", pipeline=with_meas(emb=grid_embedder()),
                     extra=["--layer", "rgb", "--process-format", "zarr"])
    assert result.exit_code == 0

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
- Modify: `src/phenotypic/_cli/_cli_process_only.py:267-411`.
  `process_single_apply_only_core` calls `pipeline.measure(image)` and
  `pipeline.embed(image)` after `apply()`, and passes `measurement_table=`,
  `embeddings=` and `upstream=` through `write_process_only_layer` (:160) to
  `_save_store` (:224-249). The run folder lands in the `.part` before
  `_consolidate_store_part`, so the store stays write-once. A store with
  embeddings and no measurements gets its entry from `write_run_entry`
  (P3 T2), which needs no objmap target when `measurements` is `None`.
- Modify: `src/phenotypic/_cli/_cli_failure_tracker.py:214`.
  `PROCESS_LAYER_SEMANTICS_REVISION = 5`.
- Modify: `src/phenotypic/_cli/CLAUDE.md` and root `CLAUDE.md`, where they state
  the revision. Add the 4 → 5 reason: process mode now runs `meas`.
- Create (helpers), in `tests/_tiny_run.py`: `grid_process_pipeline(**kw) -> Path`,
  `with_meas(grid_embedding=grid_embedder(**kw))`
- Test: `tests/unit/cli/test_process_store_tables.py` (new)

**Interfaces:**
- Consumes: P3 `choose_table_run_id` and `write_run_entry`, P4
  `write_embedding_tables`.

- [ ] **Step 1: Write the failing tests**

```python
ZARR_RGB = ["--layer", "rgb", "--process-format", "zarr"]

def test_process_store_has_grid_table_and_stays_consolidated(tmp_path, tiny_inputs, stub_backbone):
    out = tmp_path / "out"
    run_cli(tiny_inputs, out, mode="process", pipeline=grid_process_pipeline(), extra=ZARR_RGB)
    store = next(out.rglob("*.ome.zarr"))
    run = latest_table_run(store)
    assert run.run_id.startswith("1-") and not run.has_measurements
    assert run.embedding_path("dinov2-vitb14", "grid").is_file()
    nodes = json.loads((store / "zarr.json").read_text(encoding="utf-8"))["consolidated_metadata"]["metadata"]
    assert f"tables/{run.run_id}/embeddings/dinov2-vitb14/grid" in nodes
    assert pq.read_table(run.embedding_path("dinov2-vitb14", "grid")).num_rows == 4

def test_process_store_stays_byte_reproducible(tmp_path, tiny_inputs, stub_backbone):
    for name in ("a", "b"):
        run_cli(tiny_inputs, tmp_path / name, mode="process", pipeline=grid_process_pipeline(), extra=ZARR_RGB)
    stores = [next((tmp_path / n).rglob("*.ome.zarr")) for n in ("a", "b")]
    assert tree_digest(stores[0]) == tree_digest(stores[1])

def test_revision_bump_invalidates_old_process_continuation():
    assert PROCESS_LAYER_SEMANTICS_REVISION == 5
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.** With the stub backbone, inference is deterministic.
  The real model's bytes on different CPU types are measured in Task 5's pilot.
- [ ] **Step 4: Run them, plus `tests/integration/cli/test_process_objmap_semantics.py`
  and the existing process-mode tests.** They pass.
- [ ] **Step 5: Commit.**

### Task 3: In place (`--output` equal to `--input`)

An in-place run adds the pipeline's run folder to each input store where it is.
It reuses the measure rewrite's transaction (P3 T3): hard-link the store into a
`.part`, write the run folder (D6), append the process application to the
journal, write the root last, consolidate, and swap. Pixel arrays are hard
links and are never rewritten. It is for pipelines that detect nothing.

**Files:**
- Modify: `src/phenotypic/phenotypicCLI.py:2289-2306`. In process mode, only
  the **equality** case of the overlap refusal is lifted. An output strictly
  inside the input, or an input strictly inside the output, stays refused.
- Create: `src/phenotypic/_cli/_cli_process_in_place.py`, the refusals, the
  per-image core and the work-id digest.
- Modify: `src/phenotypic/phenotypicCLI.py`, the read-only half. When process
  mode's resolved `--output` equals `--input`, call `in_place_refusal` before
  `_refuse_inputs_in_deletion_path` (:636-670) and the `--dry-run` exit, and
  raise its line as `click.UsageError`.
- Modify: `src/phenotypic/sdk_/_measurement_tables.py:638-717`.
  `_rewrite_store_tables` gains `expected_root_sha256: str | None = None` and
  `consolidate: bool = False`:
  - with `expected_root_sha256`, it re-reads the live root's sha256 immediately
    before `promote_store` and raises `StoreRootChangedError` on a mismatch,
    removing the part;
  - with `consolidate`, it calls `_consolidate_store_part`
    (`_image_io_handler.py:201`) on the part after the root.
  Measure mode passes neither, so its behaviour is unchanged.
- Modify: the two per-image callers of `process_single_apply_only_core`,
  `_cli_process_single.py:905` and `_cli_execution_strategies.py:679`. In
  place, each calls
  `process_in_place_core` instead, and the work id takes
  `in_place_input_digest` in place of the store's whole-tree `file_sha256`
  (`_cli_failure_tracker.py:86`). The processing-config digest gains
  `in_place: true`, so an in-place run never reuses a copy run's
  continuation.
- Modify: docs:
  - `docs/source/how_to/pages/zarr_storage.md`: a section on in-place process
    runs: what they do, the refusals, and **where to run them**. An
    AutoConvertRaw-GC reprocess deletes and re-uploads a store
    (`acr_core/worker.py:189-190`), dropping every folder added in place. So
    run in place on a local copy, such as the `/bigdata` mirror, never on the
    bucket or the NAS tree AutoConvertRaw-GC publishes.
  - `docs/source/how_to/pages/dvc.md` already says an in-place run is not a
    DVC stage (P2 T6).
  - `src/phenotypic/_cli/CLAUDE.md` and root `CLAUDE.md`'s `--mode process`
    bullet: in place, its refusals, and that `.phenotypic/` then lives under
    the input root.
- Create (helpers), in `tests/_tiny_run.py`: fixture `process_store_tree`, the
  output of `run_cli(tiny_inputs, out, mode="process", pipeline=grid_process_pipeline(), extra=ZARR_RGB)`
  under `stub_backbone`; each store holds `1-<sha>`, as an AutoConvertRaw-GC
  store does
- Test: `tests/unit/cli/test_process_in_place.py` (new)

**Interfaces:**
- Produces:
  - `in_place_refusal(*, pipeline, input_root: Path, stores: Iterable[Path], layer: str, process_format: str, overwrite: bool, restart: bool) -> str | None`.
    One line for the first refusal that applies, checked in this order, or
    `None`:
    1. `--overwrite` or `--restart`, which would delete inputs;
    2. `--process-format tiff`, or a layer that writes no store;
    3. a pipeline with an `ObjectDetector` anywhere in its tree, any slot, any
       depth (`walk_operations`, `sdk_/_operation_tree.py:119`). A store holds
       one label image, so a second detecting pipeline would leave the first
       one's table pointing at labels it was not computed from;
    4. an `ImageCorrector` anywhere in `ops`, at any depth, which changes the
       pixels the store holds;
    5. an input that is not a PhenoTypic store (no `attributes.phenotypic` in
       its root: a raw image, a third-party OME-Zarr);
    6. an input still in the v1 table layout, pointing at `--mode migrate`;
    7. a `--layer` that is not the store's primary series (`rgb` when present,
       else `gray`).

    Checks 5–7 read one root per store and stop at the first failure.
  - `process_in_place_core(pipeline_path: Path, store: Path, input_root: Path, *, layer: str, read_kwargs: Mapping, commit_guard=None, run_initiation=None, dataset_name=None, reference_pin=None) -> bool`:
    1. read the root's sha256, then `Image.imread(store)`, which keeps the
       store's journal;
    2. open a `process` application, `apply()`, then compare the exported
       layer's level-0 digest with the store's (`level0_digest`); a mismatch
       fails the image without writing;
    3. `measure()` and `embed()`, then build figures as Task 2 does;
    4. write through `_rewrite_store_tables(..., expected_root_sha256=<step 1>, consolidate=True)`,
       with a plan that writes the run folder (`write_run_entry`), the figure
       run, and the image's journal into the root block;
    5. on `StoreRootChangedError`, repeat steps 1–4 once from the new root; a
       second mismatch fails the image.
  - `level0_digest(array: np.ndarray) -> str`: sha256 over dtype, shape and
    C-order bytes, the `objmap_digest` recipe (P3 T2), applied to the store's
    stored level-0 array and to the exported layer cast to the stored dtype.
  - `in_place_input_digest(store: Path) -> str`: the `file_sha256` tree digest
    with the root `zarr.json`, `tables/` and `figures/` left out, so the bytes
    an in-place run writes never change its own work id.
  - Continuation: an image whose store already holds this pipeline's folder
    (`table_run_for_pipeline`) with every entry `stale: false` does no work.

- [ ] **Step 1: Write the failing tests**

```python
ZARR_RGB = ["--layer", "rgb", "--process-format", "zarr"]

def in_place(tree, pipeline, extra=ZARR_RGB):
    return run_cli(tree, tree, mode="process", pipeline=pipeline, extra=extra)

def test_in_place_adds_folder_2_and_leaves_folder_1_byte_identical(process_store_tree, stub_backbone):
    store = next(process_store_tree.rglob("*.ome.zarr"))
    first = latest_table_run(store)
    before = tree_digest(store / "tables" / first.run_id)
    assert in_place(process_store_tree, grid_process_pipeline(pooling="crop_mean")).exit_code == 0
    assert [r.order for r in list_table_runs(store)] == [1, 2]
    assert tree_digest(store / "tables" / first.run_id) == before

def test_pixel_arrays_are_never_rewritten(process_store_tree, stub_backbone):
    store = next(process_store_tree.rglob("*.ome.zarr"))
    chunks = {p: (p.stat().st_ino, file_sha256(p)) for p in (store / "rgb" / "0").rglob("*") if p.is_file()}
    in_place(process_store_tree, grid_process_pipeline(pooling="crop_mean"))
    assert {p: (p.stat().st_ino, file_sha256(p)) for p in chunks} == chunks

def test_an_independent_reader_sees_the_new_folder(process_store_tree, stub_backbone):
    in_place(process_store_tree, grid_process_pipeline(pooling="crop_mean"))
    store = next(process_store_tree.rglob("*.ome.zarr"))
    run = latest_table_run(store)
    group = zarr.open_consolidated(store, mode="r")
    assert f"tables/{run.run_id}/embeddings/dinov2-vitb14/grid" in dict(group.members(max_depth=None))
    assert pq.read_table(run.embedding_path("dinov2-vitb14", "grid")).num_rows == 4

def test_rerunning_does_no_work(process_store_tree, stub_backbone):
    pipeline = grid_process_pipeline(pooling="crop_mean")
    in_place(process_store_tree, pipeline)
    store = next(process_store_tree.rglob("*.ome.zarr"))
    before = tree_digest(store)
    assert in_place(process_store_tree, pipeline).exit_code == 0
    assert tree_digest(store) == before

@pytest.mark.parametrize(("pipeline", "extra", "needle"), [
    (lambda: grid_process_pipeline(), ZARR_RGB + ["--overwrite"], "--overwrite"),
    (lambda: grid_process_pipeline(), ZARR_RGB + ["--restart"], "--restart"),
    (lambda: grid_process_pipeline(), ["--layer", "rgb", "--process-format", "tiff"], "store"),
    (lambda: grid_process_pipeline(), ["--layer", "gray", "--process-format", "zarr"], "primary series"),
    (lambda: with_meas_and_ops(ops=[ImagePipeline(ops=[OtsuDetector()])]), ZARR_RGB, "ObjectDetector"),
    (lambda: with_meas_and_ops(ops=[ColorDenoise()]), ZARR_RGB, "ImageCorrector"),
])
def test_in_place_refusals_write_nothing(process_store_tree, stub_backbone, pipeline, extra, needle):
    before = tree_digest(process_store_tree)
    result = in_place(process_store_tree, pipeline(), extra)
    errors = [line for line in result.output.splitlines() if line.startswith("Error:")]
    assert result.exit_code == 2 and len(errors) == 1 and needle in errors[0]
    assert tree_digest(process_store_tree) == before

def test_a_raw_input_is_refused(tiny_inputs, stub_backbone):
    result = in_place(tiny_inputs, grid_process_pipeline())
    assert result.exit_code == 2 and "PhenoTypic store" in result.output

def test_strict_containment_stays_refused(process_store_tree, stub_backbone):
    inner = next(process_store_tree.iterdir())
    result = run_cli(process_store_tree, inner, mode="process", pipeline=grid_process_pipeline(), extra=ZARR_RGB)
    assert result.exit_code == 2

def test_a_pixel_changing_op_the_class_check_misses_fails_the_image(process_store_tree, stub_backbone):
    sneaky = with_meas_and_ops(ops=[PixelNudgingOp()])          # changes `rgb`, yet is no ImageCorrector
    result = in_place(process_store_tree, sneaky)
    assert result.exit_code == RunExitCode.COMPLETE_WITH_FAILURES
    store = next(process_store_tree.rglob("*.ome.zarr"))
    assert len(list_table_runs(store)) == 1

def test_root_changed_before_swap_rebuilds_once_then_fails(process_store_tree, stub_backbone, monkeypatch):
    # Review Focus 5
    store = next(process_store_tree.rglob("*.ome.zarr"))
    touches = iter([True, False])
    real = _cli_process_in_place.write_run_entry
    def writer(part, *args, **kw):
        if next(touches, True):
            touch_root(store)                       # a reprocess or a second writer, mid-transaction
        return real(part, *args, **kw)
    monkeypatch.setattr(_cli_process_in_place, "write_run_entry", writer)
    assert in_place(process_store_tree, grid_process_pipeline(pooling="crop_mean")).exit_code == 0
    assert [r.order for r in list_table_runs(store)] == [1, 2]

    touches = iter([True, True])
    assert in_place(process_store_tree, grid_process_pipeline(pooling="masked_dense")).exit_code \
        == RunExitCode.COMPLETE_WITH_FAILURES
    assert [r.order for r in list_table_runs(store)] == [1, 2]
```

  Module-local helpers: `with_meas_and_ops(ops)` writes a pipeline file with
  those `ops` and `grid_embedder()` in `meas`; `PixelNudgingOp` is a bare
  `ImageOperation` test double, neither `ImageCorrector` nor `ImageEnhancer`,
  that adds one to `rgb[0, 0]`. The `writer` double changes the live root while
  the part is being populated, which is after the root was read and before the
  pre-swap check. `tiny_inputs` is a raw tree, so the raw-input test needs no
  store.
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.** The single-run guard on `--output` already keeps a
  second run off the same tree; `.phenotypic/` lives under the input root, and
  image discovery skips it and any `*.ome.zarr.part`.
- [ ] **Step 4: Run them, plus Task 1–2's tests and the existing process-mode
  and measure-mode tests.** They pass.
- [ ] **Step 5: Commit.**
- [ ] **Step 6: Run the P5 deep gate, then the end-of-plan full sharded
  regression once, on Slurm** (README conventions). The red GUI set equals the
  README's P0 and P3 rows.

### Task 4: AutoConvertRaw-GC: wheel, image and store check (repo `software/AutoConvertRaw-GC`)

**Files:**
- Modify: `vendor/`. Build the wheel from `feat/dvc-embeddings-refactor` with
  `uv build --wheel`, and replace `phenotypic-0.20.2-py3-none-any.whl`.
- Modify: `pyproject.toml`. The image installs with a frozen `uv sync` from the
  lock (`Dockerfile:30-33`), so CPU torch must come from the lock, not a
  command-line index:
  - the dependency becomes `phenotypic[foundation]`, still sourced from the
    vendored wheel (`pyproject.toml:36`);
  - add `[[tool.uv.index]]` `name = "pytorch-cpu"`,
    `url = "https://download.pytorch.org/whl/cpu"`, `explicit = true`;
  - add `torch = { index = "pytorch-cpu" }` under `[tool.uv.sources]`;
  - re-lock with `uv lock`, and commit `uv.lock`.
- Modify: `Dockerfile:28-35`, after the second `uv sync`:
  - download `facebook/dinov2-base` at a pinned commit into `/opt/hf` with
    `huggingface_hub.snapshot_download(..., revision=<commit>)`;
  - set `HF_HOME=/opt/hf`, `HF_HUB_OFFLINE=1` and
    `PHENOTYPIC_DINO_REVISION=<commit>`, so the offline loader asks for the
    commit it has (P4 T3) rather than `refs/main`;
  - extend the import smoke line (:35) to load the backbone offline and assert
    the resolved revision equals the pin.
- Modify: `pipelines/exfab_rig1.pht-pipe.json`. Add
  `"meas": {"grid_embedding": MeasureDinoEmbedding(size="base", regions="grid",
  grid=ManualGridFinder(row_edges=[…], col_edges=[…]))}`. **The rig1 edges are
  supplied by the user. Do not invent them**; stop and ask.
- Modify: `src/acr_core/pipeline_info.py`. Add `n_grid_embeddings`, counting
  `MeasureEmbedding` entries with `regions == "grid"`.
- Modify: where `run.json` is built, beside `expected_bindings`:
  `src/acr_gc/cli.py:199` and `src/acr_hpcc/launch.py:144`. Each gains
  `"expected_embeddings": info.n_grid_embeddings`.
- Modify: `src/acr_core/process.py:107`. `verify_store(store, expected_bindings,
  expected_embeddings)` checks that the store's latest run in `tables.runs`
  holds that many `embeddings` entries, each with `stale: false` and every
  listed file present.
- Modify: `src/acr_core/worker.py:59,177,239-254,275,303`. Thread
  `expected_embeddings`; a record without it reads as 0.
- Test: `tests/test_verify_store_embeddings.py` (new), plus the existing suite

- [ ] **Step 1: Write the failing tests**

```python
def test_missing_grid_entry_is_refused(tmp_path):
    store = fake_store(tmp_path, runs={"1-abc123def456": {"order": 1, "embeddings": {}}})
    assert "expected 1 embedding" in verify_store(store, expected_bindings=0, expected_embeddings=1)

def test_stale_or_missing_file_is_refused(tmp_path):
    entry = {"stale": True, "files": [{"path": "tables/1-abc123def456/embeddings/m/grid/table.parquet"}]}
    store = fake_store(tmp_path, runs={"1-abc123def456": {"order": 1, "embeddings": {"m/grid": entry}}})
    assert "stale" in verify_store(store, expected_bindings=0, expected_embeddings=1)
    entry["stale"] = False
    store = fake_store(tmp_path / "b", runs={"1-abc123def456": {"order": 1, "embeddings": {"m/grid": entry}}})
    assert "missing" in verify_store(store, expected_bindings=0, expected_embeddings=1)

def test_old_run_record_reads_zero(tmp_path):
    write_run_record(tmp_path, {"images_per_task": 4, "expected_bindings": 1})    # no expected_embeddings
    assert read_run_record(tmp_path).expected_embeddings == 0
```

  `fake_store(dir, runs)` writes a store root whose `attributes.phenotypic.tables`
  is a v2 block with those runs, and no files. `write_run_record` and
  `read_run_record` wrap the worker's `run.json` writer and
  `_read_run_record` (`worker.py:239-254`). All three are module-local.
- [ ] **Step 2: Run `uv run pytest`; confirm the new tests fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run `uv run pytest` (unit) and `-m integration` as a Slurm job**
  (`sbatch -p intel -c 4 --mem=8G -t 0:30:00 --wrap "uv run pytest -m integration"`).
  They pass.
- [ ] **Step 5: Build the image.** Run `tools/build_image.sh` (Cloud Build) and
  `acr-hpcc pull-image` as a Slurm job (repo CLAUDE.md), then pin the new
  `:<sha>` and `.sif`.
- [ ] **Step 6: Commit** in AutoConvertRaw-GC with explicit paths.

### Task 5: Pilot, then the backfill decision

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
- [ ] **Step 4: Measure CPU determinism.** Process one image twice, on two
  different CPU types (an `intel` node and an `epyc` node), and compare the grid
  table's bytes. Process-store bytes feed downstream work ids. If they differ,
  record the limit in `docs/source/how_to/pages/zarr_storage.md`: a reprocess
  on another CPU type re-derives downstream work.
- [ ] **Step 5: Ask the user** whether to backfill existing done images with
  `--reprocess-changed`. Every image becomes "changed", because the pipeline
  bytes changed, and the backfill reruns calibration and BM3D denoise. Report
  the measured per-image cost when asking.
