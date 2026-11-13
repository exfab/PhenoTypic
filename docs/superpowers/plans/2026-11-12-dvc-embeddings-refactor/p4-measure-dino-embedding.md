# P4 · `MeasureDinoEmbedding`

**Spec:** Part E (E1–E3a, E5)

---

**Goal:** A measurer that embeds detected colony crops, or the cells of a
user-set per-rig grid, with DINOv2. It writes each as its own table under the
producing pipeline's `tables/<run>/embeddings/<model>/{crops,grid}/`, with its
own deliverable. It runs on CPU in v1.

**Depends on:** P3, and P2 T3 for Task 4 (`build_run_metrics(embeddings=)`).

### Task 1: `MeasureEmbedding` and pipeline routing

**Files:**
- Create: `src/phenotypic/abc_/_measure_embedding.py`, exported from `phenotypic.abc_`
- Create: `src/phenotypic/sdk_/_embedding_tables.py` (`EmbeddingPayload`)
- Modify: `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py:1139`.
  `measure()` skips `MeasureEmbedding` entries; add `embed()`. Pipeline
  validation refuses two embedders that would write one payload key.
- Create (helpers), in `tests/_tiny_run.py`:
  - fixtures `plate` (`load_synth_yeast_plate()`) and `detected_plate` (`plate`
    with `OtsuDetector` applied), replacing the module-local copies in
    `tests/unit/util/test_output_headers.py` and elsewhere for new tests only
  - fixtures `toy_embedder` and `toy_grid_embedder`: a `MeasureEmbedding`
    subclass with `model_token() == "toy-v1"` whose `_embed_boxes` returns
    `boxes.sum(1)` repeated `D=4` times; the grid one has `regions="grid"` and
    a 2 × 2 `ManualGridFinder`
- Test: `tests/unit/abc_/test_measure_embedding.py`, `tests/unit/core/test_pipeline_embed.py` (new)

**Interfaces:**
- Produces:
  - `class MeasureEmbedding(MeasureFeatures, ABC)`:
    - field `regions: Literal["objects", "grid"] = "objects"`
    - property `requires_objects -> bool`: `regions == "objects"`. P5 T1 adds
      the base property on `MeasureFeatures`; both are properties, so mypy sees
      no `ClassVar` override.
    - abstract `_embed_boxes(image, boxes: np.ndarray) -> np.ndarray`, where
      `boxes` is `(N, 4)` int `(r0, c0, r1, c1)` and the result is `(N, D)`
      float32
    - abstract `model_token() -> str` (e.g. `"dinov2-vitb14"`)
    - abstract `entry_descriptor() -> dict`, giving `model`, `dim`, `dtype` and
      `pooling`
    - `payload_key() -> str`: `f"{model_token()}/{'crops' if regions == 'objects' else 'grid'}"`
    - `measure(image) -> pd.DataFrame`, overriding `MeasureFeatures.measure`.
      For objects it returns `Object_Label` + `embedding`; for grid it returns
      the `GRID` key columns + `embedding`. Each `embedding` cell is a 1-D
      `np.float16` array. Errors are wrapped in `OperationFailedError`, as
      `MeasureFeatures.measure` does (`abc_/_measure_features.py:533-539`).
  - `@dataclass(frozen=True) class EmbeddingPayload`: `key: str` (`"<model>/<crops|grid>"`),
    `frame: pd.DataFrame`, `descriptor: dict`
  - `ImagePipeline.embed(image) -> dict[str, EmbeddingPayload]`

- [ ] **Step 1: Write the failing tests**

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

def test_two_embedders_with_one_payload_key_are_refused(toy_embedder):
    twin = toy_embedder.model_copy()                  # same model and region, e.g. another pooling
    with pytest.raises(ValueError, match="emb_a.*emb_b.*toy-v1/crops"):
        ImagePipeline(ops=[OtsuDetector()], meas={"emb_a": toy_embedder, "emb_b": twin})
```

  Spell the `Grid_*` columns through `phenotypic.schema.GRID` in the test, not
  as literals.
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
  `write_embedding_tables` and `read_embedding_table`.
- Modify: `src/phenotypic/sdk_/_measurement_tables.py`. `write_run_entry`
  (P3 T2) writes payloads through `write_embedding_tables`;
  `replace_image_tables` takes `embeddings=`.
- Modify: `src/phenotypic/_core/_image_parts/_image_io_handler.py`.
  `save2zarr`/`_save_store` take `embeddings: Mapping[str, EmbeddingPayload] | None`
  and pass them to `write_run_entry`.
- Create (helpers), in `tests/_tiny_run.py`:
  - fixtures `image_with_crop_payload` (`detected_plate` with CLI provenance
    and one `toy-v1/crops` payload) and `image_with_two_models` (payloads
    `toy-v1/crops` and `toy-v2/crops`)
  - `save_with_payload(dir, image) -> Path`: `save2zarr` with the image's
    tables and payloads
  - `run_entry(store) -> dict`: the latest run's descriptor entry
- Test: `tests/unit/sdk_/test_embedding_tables.py` (new)

**Interfaces:**
- Consumes: P3's `TableRun`, `write_run_entry`, `refresh_embedding_staleness`,
  `objmap_digest`.
- Produces:
  - `write_embedding_tables(part: Path, run_id: str, payloads: Mapping[str, EmbeddingPayload], *, objmap_sha256: str | None, array_sha256: Mapping[str, str]) -> dict`,
    the `embeddings` block of the run entry. `array_sha256` is keyed by
    `input.series`, the same type `refresh_embedding_staleness` takes. Each
    entry holds the descriptor plus `binds`, `stale: false`, and
    `files: [{path, sha256, media_type: "application/vnd.apache.parquet"}]`.
  - `read_embedding_table(store: Path, key: str) -> pa.Table`. It verifies the
    entry's file sha256 and raises `EmbeddingTableUnreadableError` on a
    mismatch or a stale entry.

- [ ] **Step 1: Write the failing tests**

```python
def test_vectors_are_fixed_size_float16(tmp_path, image_with_crop_payload):
    store = save_with_payload(tmp_path, image_with_crop_payload)
    t = pq.read_table(latest_table_run(store).embedding_path("toy-v1", "crops"))
    assert t.schema.field("embedding").type == pa.list_(pa.float16(), 4)

def test_entry_binds_objmap_and_lists_file_hash(tmp_path, image_with_crop_payload):
    store = save_with_payload(tmp_path, image_with_crop_payload)
    entry = run_entry(store)["embeddings"]["toy-v1/crops"]
    assert entry["binds"]["objmap_sha256"] == objmap_digest(image_with_crop_payload.objmap[:])
    assert entry["files"][0]["sha256"] == file_sha256(store / entry["files"][0]["path"])

def test_two_models_coexist(tmp_path, image_with_two_models):
    store = save_with_payload(tmp_path, image_with_two_models)
    assert set(run_entry(store)["embeddings"]) == {"toy-v1/crops", "toy-v2/crops"}

def test_a_corrupt_file_is_unreadable(tmp_path, image_with_crop_payload):
    store = save_with_payload(tmp_path, image_with_crop_payload)
    latest_table_run(store).embedding_path("toy-v1", "crops").write_bytes(b"corrupt")
    with pytest.raises(EmbeddingTableUnreadableError):
        read_embedding_table(store, "toy-v1/crops")
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
  - **Never route embeddings through `_write_validated_frame`**
    (`_measurement_tables.py:152`). Its `pa.Table.from_pandas(frame)` (:165)
    yields `list<halffloat>`, not a fixed-size list.
  - Build the column with `pa.FixedSizeListArray.from_arrays(pa.array(flat, pa.float16()), D)`
    and write it under an explicit schema, so the bytes are deterministic.
  - Reread with `pq.read_table` and check the type before the root is written.
- [ ] **Step 4: Run them; they pass.**
- [ ] **Step 5: Commit.**

### Task 3: `MeasureDinoEmbedding`

**Files:**
- Create: `src/phenotypic/measure/_measure_dino_embedding.py`, exported from
  `phenotypic.measure`
- Modify: `src/phenotypic/detect/nn/_helper/_dino_support.py`:
  - `load_dino_backbone` (:193) gains `revision: str | None = None`, passed to
    both `from_pretrained` calls (:237, :239);
  - add `resolved_revision(model) -> str | None`: `model.config._commit_hash`;
  - add a batched forward helper,
    `embed_crops(model, processor, crops: np.ndarray, *, pooling: Literal["crop_cls", "crop_mean"], device: str, batch_size: int) -> np.ndarray`,
    for `(N, crop_px, crop_px, 3)` uint8 crops. It normalizes with the
    processor's mean and std without resizing, runs batches under
    `torch.inference_mode()`, and takes CLS or the mean of the patch tokens
    (`reshape_patch_tokens`, :248, strips CLS and register tokens).
    `extract_patch_features` (:286) stays the dense single-image path for
    `masked_dense`.
- Create: `src/phenotypic/schema/_measure/_dino_embedding.py`. The
  `DINO_EMBEDDING` enum has one member, `EMBEDDING`, with `label`/`desc` only.
  Export it from `phenotypic.schema`.
- Create (helpers), in `tests/_tiny_run.py`: fixture `stub_backbone`. It
  monkeypatches `load_dino_backbone` in the measurer's module to return a tiny
  module with `config.patch_size=14`, `config.hidden_size=8`,
  `config._commit_hash="0" * 40` and deterministic tokens; counts calls in
  `load_calls`; and calls the backbone cache's `cache_clear()` before and after.
  No download, no GPU.
- Test: `tests/unit/measure/test_measure_dino_embedding.py` (new). The tune
  annotation gate is `tests/unit/tune/test_annotation_coverage.py`.

**Interfaces:**
- Consumes:
  - `load_dino_backbone` (`_dino_support.py:193`), `hf_dino_id` (:172),
    `backbone_patch_size` (:61), `reshape_patch_tokens` (:248),
    `extract_patch_features` (:286), `resize_mask_to_grid` (:424)
  - `require_license_acceptance` (`detect/nn/_helper/_checkpoint_manager.py:791`)
  - `DinoVersion`, `DinoSize` (`sdk_/typing_.py:130-134`)
  - `ManualGridFinder` (`grid/_manual_grid_finder.py:16`)
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

  Plus `_measurement_infoclass: ClassVar = DINO_EMBEDDING` and
  `GridCellOutsideImageError(ValueError)` in this module.

  Validation:
  - `grid` is required iff `regions == "grid"`;
  - `crop_pad_frac` and `mask_background` are refused for grid;
  - `crop_px` must be a positive multiple of the loaded model's patch size
    (14 for v2, 16 for v3), checked at first use.

  Model identity:
  - The model token is `dinov{version}-vit{s|b|l}{patch}`, e.g. `dinov2-vitb14`.
  - The backbone revision comes from `PHENOTYPIC_DINO_REVISION` when set, else
    the hub's default. The entry's `model.revision` is
    `resolved_revision(model)`, the commit actually loaded.
  - The module-level backbone cache is a `functools.lru_cache` keyed by
    `(hf_id, revision, device)`.
  - For `version=3`, the measurer calls `require_license_acceptance` with the
    DINOv3 checkpoint manager's model name and license constants, with
    `interactive=False`, **before** loading. The gate inside
    `load_dino_backbone` (:227-234) stays.

  Tune annotations:
  - `TuneSpec(tunable=False)` on `size`, `version`, `regions`, `grid`, `dtype`,
    `batch_size` and `device`;
  - `crop_px` over the choices `112, 224, 336, 448`, the multiples of both 14
    and 16 in range, so every value is valid for DINOv2 and DINOv3;
  - `crop_pad_frac` over 0.0–1.0.

- [ ] **Step 1: Write the failing tests**

```python
def test_field_order_size_first():
    assert list(MeasureDinoEmbedding.model_fields)[:3] == ["size", "version", "regions"]

def test_model_token(stub_backbone):
    assert MeasureDinoEmbedding(size="base").model_token() == "dinov2-vitb14"

def test_crops_one_row_per_colony(stub_backbone, detected_plate):
    frame = MeasureDinoEmbedding().measure(detected_plate)
    assert len(frame) == detected_plate.num_objects
    assert frame["embedding"].iloc[0].dtype == np.float16

def test_entry_records_the_resolved_revision(stub_backbone, detected_plate):
    m = MeasureDinoEmbedding(); m.measure(detected_plate)
    assert m.entry_descriptor()["model"]["revision"] == "0" * 40

def test_grid_cell_outside_image_fails_the_image(stub_backbone, plate):
    m = MeasureDinoEmbedding(regions="grid",
                             grid=ManualGridFinder(row_edges=[0, 50, plate.shape[0] + 10], col_edges=[0, 80]))
    with pytest.raises(OperationFailedError, match=r"cell \(1, 0\).*image is \d+ x \d+") as excinfo:
        m.measure(plate)
    assert excinfo.value.err_type is GridCellOutsideImageError

def test_grid_edges_are_dvc_addressable():
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
    with pytest.raises(OperationFailedError, match="license") as excinfo:
        MeasureDinoEmbedding(version=3).measure(detected_plate)
    assert excinfo.value.err_type is RuntimeError
    assert stub_backbone.load_calls == 0                     # refused before loading

@pytest.mark.parametrize("crop_px", [112, 224, 336, 448])
def test_every_tunable_crop_px_fits_both_patch_sizes(crop_px):
    assert crop_px % 14 == 0 and crop_px % 16 == 0
```

- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
  - **Crop modes:** pad the bbox by `crop_pad_frac`, clip to the image (objects
    only), and resize to `crop_px`. With `mask_background`, zero pixels outside
    the colony. Then `embed_crops`.
  - **Grid:** a cell is cropped from its rectangle exactly as an object is from
    its bbox. A cell outside the image raises `GridCellOutsideImageError`
    naming the cell and the image size; it is never clipped.
  - **`masked_dense`:** one tiled pass over the plate with
    `extract_patch_features`, then the patch tokens are averaged under each
    region's mask, resized with `resize_mask_to_grid`.
  - **`preflight_requirements()`** declares `torch` and `transformers`.
  - Import torch inside functions.
- [ ] **Step 4: Run the tests, the tune annotation gate,
  `tests/unit/ci/test_startup_imports.py` and `tests/unit/ci/test_deferred_imports.py`.**
  They pass.
- [ ] **Step 5: Run a real-model smoke test as a Slurm job.** The test is marked
  `slow` and skipped without the weights: `regions="objects"` on
  `load_synth_yeast_plate()`, `size="small"`, CPU. Record wall time, peak RSS
  and the resolved revision in the task's commit message. Use the `slurm-job`
  skill, `intel`, 4 CPU, 16 GB.
- [ ] **Step 6: Commit.**

### Task 4: CLI wiring and deliverables

**Files:**
- Modify: the per-image writers:
  - `src/phenotypic/_cli/_cli_process_single.py` (full and measure)
  - `src/phenotypic/_cli/_cli_staged_workers.py` (Stage 3)

  Each calls `pipeline.embed(image)` beside `measure()` and passes
  `embeddings=` to the store write (`save2zarr` or `replace_image_tables`).
- Modify: `src/phenotypic/_cli/_cli_finalize_run.py`. After the master, write
  `deliverables/embeddings/<model>/<crops|grid>.parquet` and
  `deliverables/embeddings/manifest.json`, add the embedding tables to
  `schema.json`, and pass the counts to `build_run_metrics(embeddings=...)`
  (P2 T3). An error raises, and finalization fails.
- Modify: `src/phenotypic/_cli/_cli_output_manager.py:531-545`. The by-feature
  split skips `MeasureEmbedding` entries.
- Modify: `src/phenotypic/_cli/_cli_readme_generator.py:201-210`. The
  measurement section skips `MeasureEmbedding` entries, which publish no mirror
  column; a separate section documents `deliverables/embeddings/`.
- Modify: `src/phenotypic/util/_measurement_outputs.py`. `build_output_schema`
  lists each embedding table's `embedding` column as
  `FixedSizeList<float16>[D]` with its model, member `DINO_EMBEDDING.EMBEDDING`
  and that member's `desc`. The column is named `embedding`, unprefixed, so it
  is resolved through the embedding table, not `_member_for_column` (:435).
- Modify: `src/phenotypic/sdk_/_io_constants.py`. Add `DIR_EMBEDDINGS`,
  `embeddings_dir(output_dir)` and `embeddings_manifest_path(output_dir)`.
- Create (helpers), in `tests/_tiny_run.py`:
  - `with_dino_crops() -> Path`: a pipeline file holding the `tiny_pipeline`
    plus `MeasureDinoEmbedding()` under `meas["emb"]`
  - fixture `finished_tiny_run_with_embeddings`:
    `run_cli(tiny_inputs, out, pipeline=with_dino_crops())` under
    `stub_backbone`
  - fixture `tree_with_one_stale_store`: `finished_tiny_run_with_embeddings`
    with three images, one of whose crop entries is set `stale: true` through a
    resave with a zeroed objmap; it yields `(out, runs)`, the certified
    `TableRun` of each image
  - `rows_in(run) -> int`: the crop table's row count
- Test: `tests/integration/cli/test_embedding_deliverables.py` (new)

**Interfaces:**
- Consumes: Tasks 1–3, P3 `certified_table_run`, P2 `build_run_metrics`.
- Produces: `aggregate_embedding_tables(output_dir: Path, runs: Sequence[TableRun]) -> dict[str, dict[str, int]]`.
  - `runs` is each image's certified run folder (spec D6a), never a run chosen
    by pipeline sha.
  - It streams each key's rows with one `pq.ParquetWriter` per output file,
    one row group per store, so memory stays one store's table at a time
    (about 5 GB of float16 vectors over 33k stores otherwise).
  - It maps each entry key to `{"rows": n, "stale_skipped": k}`, which is what
    `build_run_metrics(embeddings=...)` takes. Stale entries are skipped with a
    warning naming the store.

- [ ] **Step 1: Write the failing tests**

```python
def test_full_run_publishes_embedding_deliverable(finished_tiny_run_with_embeddings):
    out = finished_tiny_run_with_embeddings
    t = pq.read_table(out / "deliverables/embeddings/dinov2-vitb14/crops.parquet")
    assert {"Metadata_Dataset", "Object_Label", "embedding"} <= set(t.column_names)
    master = pl.read_parquet(out / "deliverables/master_measurements.parquet")
    assert "embedding" not in master.columns
    metrics = json.loads((out / "deliverables/metrics.json").read_text(encoding="utf-8"))
    assert metrics["embeddings"]["dinov2-vitb14/crops"]["rows"] == t.num_rows

def test_embeddings_get_no_split_and_no_mirror_section(finished_tiny_run_with_embeddings):
    d = finished_tiny_run_with_embeddings / "deliverables"
    assert not any("Embedding" in p.name for p in (d / "measurements_by_feature").iterdir())
    schema = json.loads((d / "schema.json").read_text(encoding="utf-8"))
    vector = next(c for c in schema["tables"]["embeddings/dinov2-vitb14/crops"] if c["name"] == "embedding")
    assert vector["member"] == str(DINO_EMBEDDING.EMBEDDING)

def test_stale_entries_are_skipped_and_counted(tree_with_one_stale_store):
    out, runs = tree_with_one_stale_store            # 3 images, one stale
    counts = aggregate_embedding_tables(out, runs)
    assert counts["dinov2-vitb14/crops"]["stale_skipped"] == 1
    assert counts["dinov2-vitb14/crops"]["rows"] == sum(rows_in(r) for r in runs if not is_stale(r))

def test_corrupt_embedding_does_not_revoke_measurements(finished_tiny_run_with_embeddings):
    # spec D9: embeddings are additive to measurement authority
    out = finished_tiny_run_with_embeddings
    store = one_store(out)
    latest_measurement_run(store).embedding_path("dinov2-vitb14", "crops").write_bytes(b"corrupt")
    state = resolve_run_state(out, depth="deep")
    assert all(image.verdict == "verified" for image in state.images.values())
    with pytest.raises(EmbeddingTableUnreadableError):
        read_embedding_table(store, "dinov2-vitb14/crops")
```

  `is_stale(run)` is module-local: the run's crop entry has `stale: true`.
- [ ] **Step 2: Run them and confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run the P4 phase gate:** importers of `_cli_process_single`,
  `_cli_staged_workers`, `_cli_finalize_run`, `_cli_output_manager` and
  `_cli_readme_generator`, plus `tests/integration/cli`, then `tests/unit/gui`
  and `tests/gui` once. The red GUI set equals the README's P0 and P3 rows. Run
  it as a Slurm job if it's over a couple of minutes.
- [ ] **Step 5: Commit.**
