# P4 · `MeasureDinoEmbedding`

**Spec:** Part E (E1–E3a, E5)

---

**Goal:** A measurer that embeds detected colony crops, or the cells of a
user-set per-rig grid, with DINOv2. It writes each as its own table under the
producing pipeline's `tables/<run>/embeddings/<model>/{crops,grid}/`, with its
own deliverable. It runs on CPU in v1.

### Task 1: `MeasureEmbedding` and pipeline routing

**Files:**
- Create: `src/phenotypic/abc_/_measure_embedding.py`, exported from `phenotypic.abc_`
- Create: `src/phenotypic/sdk_/_embedding_tables.py` (`EmbeddingPayload`)
- Modify: `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py:1122`.
  `measure()` skips `MeasureEmbedding` entries; add `embed()`.
- Test: `tests/unit/abc_/test_measure_embedding.py`, `tests/unit/core/test_pipeline_embed.py` (new)

**Interfaces:**
- Produces:
  - `class MeasureEmbedding(MeasureFeatures, ABC)`:
    - field `regions: Literal["objects", "grid"] = "objects"`
    - property `requires_objects -> bool`: `regions == "objects"`
    - abstract `_embed_boxes(image, boxes: np.ndarray) -> np.ndarray`, where
      `boxes` is `(N, 4)` int `(r0, c0, r1, c1)` and the result is `(N, D)`
      float32
    - abstract `model_token() -> str` (e.g. `"dinov2-vitb14"`)
    - abstract `entry_descriptor() -> dict`, giving `model`, `dim`, `dtype` and
      `pooling`
    - `measure(image) -> pd.DataFrame`. For objects it returns `Object_Label` +
      `embedding`; for grid it returns the `GRID` key columns + `embedding`.
      Each `embedding` cell is a 1-D `np.float16` array.
  - `@dataclass(frozen=True) class EmbeddingPayload`: `key: str` (`"<model>/<crops|grid>"`),
    `frame: pd.DataFrame`, `descriptor: dict`
  - `ImagePipeline.embed(image) -> dict[str, EmbeddingPayload]`

- [ ] **Step 1: Write the failing tests** (with a toy subclass that returns
  `boxes.sum(1)` repeated D=4 times)

```python
def test_measure_skips_and_embed_runs(toy_embedder, plate):
    pipe = ImagePipeline(ops=[OtsuDetector()], meas={"size": MeasureSize(), "emb": toy_embedder})
    img = pipe.apply(plate)
    assert not any("embedding" in c for c in pipe.measure(img).columns)
    payloads = pipe.embed(img)
    assert list(payloads) == ["toy-v1/crops"]
    assert len(payloads["toy-v1/crops"].frame) == img.num_objects

def test_grid_region_needs_no_objects(toy_grid_embedder, plate):
    frame = toy_grid_embedder.measure(plate)          # no detection ran
    assert {"Grid_RowNum", "Grid_ColNum", "embedding"} <= set(frame.columns)
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
  - Crops use `image.objects` bboxes; grid uses `grid.get_row_edges` /
    `get_col_edges` (`abc_/_grid_finder.py:281,292`).
  - The docstrings on `measure()` and `embed()` say which runs which.
- [ ] **Step 4: Run them, plus `tests/unit/core` pipeline tests.** They pass.
- [ ] **Step 5: Commit.**

### Task 2: Embedding tables in the store

**Files:**
- Modify: `src/phenotypic/sdk_/_embedding_tables.py`. Add
  `write_embedding_tables`.
- Modify: `src/phenotypic/_core/_image_parts/_image_io_handler.py`.
  `save2zarr`/`_save_store` take `embeddings: Mapping[str, EmbeddingPayload] | None`
  and write them into the current run folder.
- Modify: `src/phenotypic/sdk_/_measurement_tables.py`. `replace_image_tables`
  takes `embeddings=`.
- Test: `tests/unit/sdk_/test_embedding_tables.py` (new)

**Interfaces:**
- Consumes: P3's `TableRun`, `refresh_embedding_staleness`, and the run descriptor.
- Produces: `write_embedding_tables(part: Path, run_id: str, payloads: Mapping[str, EmbeddingPayload], *, objmap_sha256: str | None, array_sha256: str) -> dict`,
  the `embeddings` block of the run entry. Each entry holds the descriptor plus
  `binds`, `stale: false`, and `files: [{path, sha256, media_type: "application/vnd.apache.parquet"}]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_vectors_are_fixed_size_float16(tmp_path, image_with_crop_payload):
    store = save_with_payload(tmp_path, image_with_crop_payload)
    t = pq.read_table(latest_table_run(store).embedding_path("toy-v1", "crops"))
    assert t.schema.field("embedding").type == pa.list_(pa.float16(), 4)

def test_entry_binds_objmap_and_lists_file_hash(tmp_path, image_with_crop_payload):
    store = save_with_payload(tmp_path, image_with_crop_payload)
    entry = run_entry(store)["embeddings"]["toy-v1/crops"]
    assert entry["binds"]["objmap_sha256"] == sha256_array(image_with_crop_payload.objmap[:])
    assert entry["files"][0]["sha256"] == sha256_file(store / entry["files"][0]["path"])

def test_two_models_coexist(tmp_path, image_with_two_models):
    store = save_with_payload(tmp_path, image_with_two_models)
    assert set(run_entry(store)["embeddings"]) == {"toy-v1/crops", "toy-v2/crops"}

def test_corrupt_embedding_does_not_revoke_measurements(finished_tiny_run_with_embeddings):
    # spec D9: embeddings are additive to measurement authority
    store = one_store(finished_tiny_run_with_embeddings)
    latest_table_run(store).embedding_path("toy-v1", "crops").write_bytes(b"corrupt")
    state = resolve_run_state(finished_tiny_run_with_embeddings, depth="deep")
    assert state.images[only_work_id(state)].verdict == "verified"
    with pytest.raises(EmbeddingTableUnreadableError):
        read_embedding_table(store, "toy-v1/crops")
```

  `read_embedding_table(store, key) -> pa.Table` verifies the entry's file sha256
  and raises `EmbeddingTableUnreadableError` on a mismatch. Both live in
  `_embedding_tables.py`.

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
  - Convert with `pa.FixedSizeListArray.from_arrays(pa.array(flat, pa.float16()), D)`.
  - Reread with `pq.read_table` and check the type before the root is written,
    as `_write_validated_frame` does.
- [ ] **Step 4: Run them; they pass.**
- [ ] **Step 5: Commit.**

### Task 3: `MeasureDinoEmbedding`

**Files:**
- Create: `src/phenotypic/measure/_measure_dino_embedding.py`, exported from
  `phenotypic.measure`
- Create: `src/phenotypic/schema/_measure/_dino_embedding.py`. The
  `DINO_EMBEDDING` enum has one member, `EMBEDDING`, with `label`/`desc` only.
  Export it from `phenotypic.schema`.
- Test: `tests/unit/measure/test_measure_dino_embedding.py` (new). The tune
  annotation gate is `tests/unit/tune/test_annotation_coverage.py`.

**Interfaces:**
- Consumes:
  - `load_dino_backbone` (`detect/nn/_helper/_dino_support.py:193`)
  - `hf_dino_id` (:172)
  - `extract_patch_features` (:286)
  - `backbone_patch_size` (:61)
  - `resize_mask_to_grid`
  - `DinoVersion`, `DinoSize` (`sdk_/typing_.py:130-134`)
  - `ManualGridFinder`
- Produces: `class MeasureDinoEmbedding(MeasureEmbedding)` with fields in this
  exact order:
  1. `size: DinoSize = "base"`
  2. `version: DinoVersion = 2`
  3. `regions: Literal["objects", "grid"] = "objects"`
  4. `grid: ManualGridFinder | None = None` (an `OperationField`)
  5. `pooling: Literal["crop_cls", "crop_mean", "masked_dense"] = "crop_cls"`
  6. `crop_px: int = 224`
  7. `crop_pad_frac: float = 0.25`
  8. `mask_background: bool = False`
  9. `dtype: Literal["float16", "float32"] = "float16"`
  10. `batch_size: int = 64`
  11. `device: Literal["auto", "cpu", "cuda"] = "auto"`

  Plus `_measurement_infoclass: ClassVar = DINO_EMBEDDING`.

  Validation:
  - `grid` is required iff `regions == "grid"`;
  - `crop_pad_frac` and `mask_background` are refused for grid;
  - `crop_px` must be a positive multiple of 14 for v2 and 16 for v3; this is
    checked against the loaded model at first use.

  Tune annotations:
  - `TuneSpec(tunable=False)` on `size`, `version`, `regions`, `grid`, `dtype`,
    `batch_size` and `device`;
  - ranges on `crop_px` (112–448, step 14) and `crop_pad_frac` (0.0–1.0).

- [ ] **Step 1: Write the failing tests.** A stub backbone monkeypatches
  `load_dino_backbone` to return a tiny module with `config.patch_size=14`,
  `config.hidden_size=8` and deterministic tokens. No download, no GPU.

```python
def test_field_order_size_first():
    assert list(MeasureDinoEmbedding.model_fields)[:3] == ["size", "version", "regions"]

def test_model_token(stub_backbone):
    assert MeasureDinoEmbedding(size="base").model_token() == "dinov2-vitb14"

def test_crops_one_row_per_colony(stub_backbone, detected_plate):
    frame = MeasureDinoEmbedding().measure(detected_plate)
    assert len(frame) == detected_plate.num_objects
    assert frame["embedding"].iloc[0].dtype == np.float16

def test_grid_cell_outside_image_fails_the_image(stub_backbone, plate):
    # Review Focus 5
    m = MeasureDinoEmbedding(regions="grid",
                             grid=ManualGridFinder(row_edges=[0, 50, plate.shape[0] + 10], col_edges=[0, 80]))
    with pytest.raises(GridCellOutsideImageError, match=r"cell \(1, 0\).*image is \d+ x \d+"):
        m.measure(plate)

def test_grid_edges_are_dvc_addressable(tmp_path):
    pipe = ImagePipeline(ops=[], meas={"emb": MeasureDinoEmbedding(regions="grid",
                         grid=ManualGridFinder(row_edges=[0, 10], col_edges=[0, 10]))})
    data = json.loads(pipe.to_json())
    assert data["meas"]["emb"]["params"]["grid"]["params"]["row_edges"] == [0, 10]
    # pins the key path the how-to page documents (spec E3a)

def test_backbone_loaded_once_per_process(stub_backbone, detected_plate):
    m = MeasureDinoEmbedding(); m.measure(detected_plate); m.measure(detected_plate)
    assert stub_backbone.load_calls == 1

def test_dinov3_requires_license_acceptance(monkeypatch, stub_backbone, detected_plate):
    monkeypatch.delenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", raising=False)
    with pytest.raises(LicenseNotAcceptedError):
        MeasureDinoEmbedding(version=3).measure(detected_plate)
```

  Use the exception name `require_license_acceptance` already raises.
  `GridCellOutsideImageError(ValueError)` is new and lives in this module.
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
  - **Crop modes:** pad the bbox by `crop_pad_frac`, clip to the image (objects
    only), and resize to `crop_px`. With `mask_background`, zero pixels outside
    the colony. Batch through the backbone under `torch.inference_mode()`.
    - `crop_cls` takes the CLS token.
    - `crop_mean` takes the mean of the patch tokens after `extract_patch_features`
      strips CLS and register tokens.
  - **`masked_dense`:** one tiled pass over the plate, then the patch tokens are
    averaged under each region's mask, resized with `resize_mask_to_grid`.
  - **Backbone cache:** a module-level `functools.lru_cache` keyed by
    `(hf_id, revision, device)`.
  - **`preflight_requirements()`** declares `torch` and `transformers`.
  - Import torch inside functions.
- [ ] **Step 4: Run the tests, the tune annotation gate,
  `tests/unit/ci/test_startup_imports.py` and `tests/unit/ci/test_deferred_imports.py`.**
  They pass.
- [ ] **Step 5: Run a real-model smoke test as a Slurm job.** The test is marked
  `slow` and skipped without the weights: `regions="objects"` on
  `load_synth_yeast_plate()`, `size="small"`, CPU. Record wall time and peak RSS
  in the task's commit message. Use the `slurm-job` skill, `intel`, 4 CPU, 16 GB.
- [ ] **Step 6: Commit.**

### Task 4: CLI wiring and deliverables

**Files:**
- Modify: the per-image writers:
  - `src/phenotypic/_cli/_cli_process_single.py` (full and measure)
  - `src/phenotypic/_cli/_cli_staged_workers.py` (Stage 3)

  Each calls `pipeline.embed(image)` beside `measure()` and passes `embeddings=`
  to the store write.
- Modify: `src/phenotypic/_cli/_cli_finalize_run.py`. After the master, write
  `deliverables/embeddings/<model>/<crops|grid>.parquet` and
  `deliverables/embeddings/manifest.json`, and pass the counts to
  `build_run_metrics(embeddings=...)` (P2 T3). An error raises, and
  finalization fails.
- Modify: `src/phenotypic/sdk_/_io_constants.py`. Add `DIR_EMBEDDINGS`,
  `embeddings_dir(output_dir)` and `embeddings_manifest_path(output_dir)`.
- Modify: `src/phenotypic/_cli/_cli_readme_generator.py`. Document the folder.
- Test: `tests/integration/cli/test_embedding_deliverables.py` (new)

**Interfaces:**
- Consumes: Tasks 1–3, P3 `table_run_for_pipeline`, P2 `build_run_metrics`.
- Produces: `aggregate_embedding_tables(output_dir: Path, stores: Sequence[Path], *, pipeline_sha256: str) -> dict[str, dict[str, int]]`.
  It maps each entry key to `{"rows": n, "stale_skipped": k}`, which is what
  `build_run_metrics(embeddings=...)` takes. Stale entries are skipped with a
  warning naming the store.

- [ ] **Step 1: Write the failing tests**

```python
def test_full_run_publishes_embedding_deliverable(tiny_inputs, tmp_path, stub_backbone):
    run_cli(tiny_inputs, tmp_path, pipeline=with_dino_crops())
    t = pq.read_table(tmp_path / "deliverables/embeddings/dinov2-vitb14/crops.parquet")
    assert {"Metadata_Dataset", "Object_Label", "embedding"} <= set(t.column_names)
    master = pl.read_parquet(tmp_path / "deliverables/master_measurements.parquet")
    assert "embedding" not in master.columns
    assert json.loads((tmp_path / "deliverables/metrics.json").read_text())["embeddings"]["dinov2-vitb14/crops"]["rows"] == t.num_rows

def test_stale_entries_are_skipped_and_counted(tree_with_one_stale_store, stub_backbone):
    out, stores, sha = tree_with_one_stale_store          # 3 stores, one marked stale
    counts = aggregate_embedding_tables(out, stores, pipeline_sha256=sha)
    assert counts["dinov2-vitb14/crops"]["stale_skipped"] == 1
    assert counts["dinov2-vitb14/crops"]["rows"] == rows_in(stores[0]) + rows_in(stores[1])
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the P4 phase gate:** importers of `_cli_process_single`,
  `_cli_staged_workers` and `_cli_finalize_run`, plus `tests/integration/cli`.
  They pass. Run it as a Slurm job if it's over a couple of minutes.
- [ ] **Step 5: Commit.**
