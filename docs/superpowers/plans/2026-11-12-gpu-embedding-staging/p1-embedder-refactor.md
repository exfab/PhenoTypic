# P1 · The embedder refactor (prepare / encode / payload)

**Spec:** §4.4 "Embedder refactor", H1, §5 "Byte-identity of the inline path"

---

**Goal:** `MeasureEmbedding` runs as three steps: a CPU `prepare_regions` that
cuts and resizes every crop, a backbone-only `encode`, and `payload_from`. The
inline `payload()` is those three, and its bytes are P0 T2's golden. Stage 4
(P4) calls the same three steps, with a different encoder in the middle.

**Depends on:** P0 T2 (the golden is committed).

### Task 1: The three steps on the ABC

**Files:**
- Modify: `src/phenotypic/abc_/_measure_embedding.py`
  - add `BackboneInputs`, `PreparedRegions`, `RegionEncoder` above
    `MeasureEmbedding` (`:110`);
  - add `prepare_regions`, `encode`, `vectors_from`, `payload_from`,
    `inline_encoder`;
  - `_operate` (`:412`) and `payload` (`:523`) become compositions of them;
  - `_embed_vectors` (`:476`) splits: the shape and finiteness checks move to
    `payload_from`, the dispatch to `vectors_from`.
- Modify: `src/phenotypic/abc_/__init__.py`, exporting `BackboneInputs`,
  `PreparedRegions`, `RegionEncoder`.
- Test: `tests/unit/abc_/test_measure_embedding_steps.py` (new). Shared env;
  uses `toy_embedder` (`tests/_tiny_run.py:1044`) and `toy_grid_embedder`.

**Interfaces:**
- Produces, in `phenotypic.abc_`:
  - `@dataclass(frozen=True) class BackboneInputs`:
    - `kind: Literal["crops", "tiles", "boxes"]`
    - `arrays: tuple[np.ndarray, ...]`. `"crops"`: one `(N, px, px, 3)` uint8
      array. `"tiles"`: one `(3, h, w)` float32 array per tile, the HF
      processor's own `pixel_values` (each side a multiple of the patch), so
      every path normalises a tile exactly as inline does today (plan review
      I1). `"boxes"`: one `(N, 4)` int64 array, for an embedder with no
      backbone (inline only).
    - `pooling: Literal["crop_cls", "crop_mean", "patches", "none"]`
  - `@dataclass(frozen=True) class PreparedRegions` (picklable; it crosses
    DataLoader worker processes in P4):
    - `key: str`, the payload key
    - `keys: pd.DataFrame`, the key columns in table order
    - `key_column: str`, `Object_Label` or `Grid_RowMajorIdx`
    - `inputs: BackboneInputs`
    - `binds: dict[str, Any]`, `input: dict[str, Any]` (today's
      `payload()` values, `:546-569`)
    - `pool_plan: Mapping[str, Any] | None`, what `vectors_from` needs after the
      backbone (masked_dense: tile offsets, plate grid shape, region masks);
      `None` otherwise
  - `class RegionEncoder(Protocol)`: `run(self, inputs: BackboneInputs) -> list[np.ndarray]`.
    For `"crops"`, one `(N, D)` float32 array, already pooled. For `"tiles"`,
    one `(h // p, w // p, D)` float32 array per tile. For `"boxes"`, one
    `(N, D)` array.
  - On `MeasureEmbedding`:
    - `prepare_regions(self, image: Image) -> PreparedRegions`. The ABC default
      gives `kind="boxes"` from `_object_boxes` / `_grid_cells` (`:425`, `:441`);
      it raises `GridCellOutsideImageError` exactly as today.
    - `inline_encoder(self, image: Image) -> RegionEncoder`. The ABC default
      calls `_embed_object_boxes` / `_embed_boxes` on the given image.
    - `encode(self, prepared: PreparedRegions, encoder: RegionEncoder) -> np.ndarray`:
      `self.vectors_from(prepared, encoder.run(prepared.inputs))`. A
      zero-region `prepared` returns a `(0, 0)` float32 array without calling
      `encoder.run` **or `_dim()`**: `_dim()` calls `entry_descriptor()`, which
      would load a second model in Stage 4 (plan review I12).
    - `vectors_from(self, prepared: PreparedRegions, outputs: list[np.ndarray]) -> np.ndarray`:
      `(N, D)` float32, region order.
    - `payload_from(self, prepared: PreparedRegions, vectors: np.ndarray, descriptor: Mapping[str, Any]) -> EmbeddingPayload`:
      reshapes a zero-row `vectors` to `(0, descriptor["dim"])`, checks shape
      `(N, descriptor["dim"])`, casts to float16, refuses
      non-finite values with today's message (`:497-501`), and builds the
      descriptor `{**descriptor, key, input, producer, binds}` exactly as
      `payload()` does (`:565-574`).
    - `payload(self, image) -> EmbeddingPayload` becomes
      `prepared = self.prepare_regions(image)` →
      `self.payload_from(prepared, self.encode(prepared, self.inline_encoder(image)), self.entry_descriptor())`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_payload_is_prepare_encode_payload_from(toy_embedder, detected_plate):
    prepared = toy_embedder.prepare_regions(detected_plate)
    vectors = toy_embedder.encode(prepared, toy_embedder.inline_encoder(detected_plate))
    stepwise = toy_embedder.payload_from(prepared, vectors, toy_embedder.entry_descriptor())
    whole = toy_embedder.payload(detected_plate)
    assert stepwise.key == whole.key and stepwise.descriptor == whole.descriptor
    assert embedding_arrow_table(stepwise).equals(embedding_arrow_table(whole))

def test_prepared_regions_pickle(toy_grid_embedder, plate):
    prepared = toy_grid_embedder.prepare_regions(plate)
    again = pickle.loads(pickle.dumps(prepared))
    assert again.key == prepared.key and again.keys.equals(prepared.keys)

def test_zero_regions_call_neither_the_encoder_nor_dim(toy_embedder, plate, monkeypatch):
    empty = plate.copy(); empty.objmap[:] = 0
    prepared = toy_embedder.prepare_regions(empty)
    class Refuse:
        def run(self, inputs): raise AssertionError("encoder called for zero regions")
    monkeypatch.setattr(type(toy_embedder), "_dim", lambda self: pytest.fail("_dim called"))
    vectors = toy_embedder.encode(prepared, Refuse())
    payload = toy_embedder.payload_from(prepared, vectors, {**toy_embedder.entry_descriptor()})
    assert len(payload.frame) == 0 and payload.dim == 4

def test_payload_from_refuses_non_finite_float16(toy_embedder, detected_plate):
    prepared = toy_embedder.prepare_regions(detected_plate)
    huge = np.full((len(prepared.keys), 4), 1e6, dtype=np.float32)
    with pytest.raises(ValueError, match="not finite in float16"):
        toy_embedder.payload_from(prepared, huge, toy_embedder.entry_descriptor())

def test_grid_outside_the_image_is_raised_by_prepare(plate):
    m = toy_grid_with_edges(row_edges=[0, plate.shape[0] + 10], col_edges=[0, 10])
    with pytest.raises(GridCellOutsideImageError):
        m.prepare_regions(plate)
```

  `toy_grid_with_edges(row_edges, col_edges)` is module-local:
  `toy_grid_embedder.model_copy(update={"grid": ManualGridFinder(...)})`.
  Build the empty plate with whatever `Image` API clears the objmap today
  (`plate.objmap[:] = 0` above stands for it); do not add an API.
- [ ] **Step 2: Run them and confirm they fail.**

  ```bash
  eval $PT tests/unit/abc_/test_measure_embedding_steps.py
  ```

  Expected: `AttributeError: ... has no attribute 'prepare_regions'` (5 failed).
- [ ] **Step 3: Implement** the Interfaces above. `measure()` (`:368`) keeps its
  signature, its OOM pass-through (`is_out_of_memory`, `:60`) and its
  `OperationFailedError` wrapping; `_operate` returns `keys` plus `embedding`
  from `prepare_regions` and `encode`.
- [ ] **Step 4: Run the new tests, the ABC tests and the golden.**

  ```bash
  eval $PT tests/unit/abc_/test_measure_embedding_steps.py tests/unit/abc_/test_measure_embedding.py tests/unit/core/test_pipeline_embed.py
  eval $TPT tests/unit/measure/test_embedding_inline_golden.py
  ```

  Expected: all pass; golden `7 passed`.
- [ ] **Step 5: Commit.**

  ```bash
  git add src/phenotypic/abc_/_measure_embedding.py src/phenotypic/abc_/__init__.py tests/unit/abc_/test_measure_embedding_steps.py
  git commit -m "refactor(embed): prepare_regions / encode / payload_from on MeasureEmbedding (#282 P1)"
  ```

### Task 2: `MeasureDinoEmbedding` on the three steps

**Files:**
- Modify: `src/phenotypic/measure/_measure_dino_embedding.py`
  - `prepare_regions(image, *, processor=None)` override: crops through `_crop`
    (`:477`) and `_square_colony_crop` (`:506`), unchanged, for
    `crop_cls`/`crop_mean`; for `masked_dense`, patch-aligned tiles cut as
    `_masked_dense` does today (`:546-600`), each passed through
    `processor(images=tile, return_tensors="np", **NATIVE_PROCESSOR_KWARGS)`
    (`_dino_support.py:56`, the call `extract_patch_features` makes at
    `:342-344`, with `"np"` in place of `"pt"`; the test compares the two, so a
    processor whose `"np"` and `"pt"` paths differ fails it), with the feature-grid shape and
    per-region masks in `pool_plan`. `processor=None` loads one through
    `load_dino_processor` (below).
  - `_dino_support.py`: add `load_dino_processor(dino_version, dino_size, *, revision=None) -> Any`,
    `functools.lru_cache`d, loading only `AutoImageProcessor` (no weights), so a
    DataLoader worker can normalise tiles without a model.
  - private `_InlineDinoEncoder(model, processor, device, batch_size)`
    implementing `RegionEncoder`: `"crops"` call `embed_crops`
    (`detect/nn/_helper/_dino_support.py:400`) once per `batch_size` slice, as
    `_embed_regions` does (`:442-475`); `"tiles"` run the model on each tile's
    `pixel_values` and `reshape_patch_tokens` (`:280`), the second half of
    `extract_patch_features` (`:320-358`).
  - `vectors_from` for `"patches"`: place tiles into the features grid and
    `_pool_under_mask` (`:615`) per region, as `_masked_dense` does.
  - `entry_descriptor_for(self, model: Any, compute: Mapping[str, Any] | None = None) -> dict`:
    today's `entry_descriptor()` body (`:315-351`) over a given loaded model and
    no `self._backbone()` call. `entry_descriptor()` becomes
    `self.entry_descriptor_for(self._backbone()[0])`. (`compute` is accepted and
    ignored until P2 T1 writes it.)
  - `_embed_boxes`, `_embed_object_boxes`, `_embed_regions` and `_masked_dense`
    are removed once nothing calls them.
- Test: `tests/unit/measure/test_measure_dino_embedding_steps.py` (new) **[torch]**

**Interfaces:**
- Consumes: Task 1.
- Produces:
  - `MeasureDinoEmbedding.prepare_regions(image, *, processor=None) -> PreparedRegions` whose
    `inputs.kind` is `"crops"` (`pooling` `crop_cls`/`crop_mean`) or `"tiles"`
    (`pooling="patches"`). It loads no model; for crop pooling it imports no
    torch. P4's DataLoader workers run it.
  - `load_dino_processor(dino_version, dino_size, *, revision=None) -> Any` (in `_dino_support.py`)
  - `MeasureDinoEmbedding.entry_descriptor_for(model, compute=None) -> dict`
  - `MeasureDinoEmbedding.crop_geometry() -> tuple[int, int]`: `(crop_px, crop_px)`

- [ ] **Step 1: Write the failing tests.**

```python
def test_prepare_loads_no_model(stub_backbone, detected_plate):
    MeasureDinoEmbedding().prepare_regions(detected_plate)
    assert stub_backbone.load_calls == []

def test_prepare_imports_no_torch(detected_plate):
    code = ("import sys; from phenotypic.measure import MeasureDinoEmbedding;"
            "from phenotypic.data import load_synth_yeast_plate;"
            "from phenotypic.detect import OtsuDetector;"
            "p = OtsuDetector().apply(load_synth_yeast_plate());"
            "MeasureDinoEmbedding().prepare_regions(p); print('torch' in sys.modules)")
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          check=True).stdout.strip() == "False"

def test_crops_are_the_inline_crops(stub_backbone, detected_plate):
    m = MeasureDinoEmbedding(crop_pad_frac=0.5, mask_background=True)
    prepared = m.prepare_regions(detected_plate)
    assert prepared.inputs.kind == "crops"
    assert prepared.inputs.arrays[0].shape == (detected_plate.num_objects, 224, 224, 3)
    assert prepared.inputs.arrays[0].dtype == np.uint8

def test_dense_tiles_are_the_processors_pixels(stub_backbone, detected_plate):
    m = MeasureDinoEmbedding(pooling="masked_dense", crop_px=112)
    _, processor = load_dino_backbone(2, "base", "cpu")                      # the stub's
    prepared = m.prepare_regions(detected_plate, processor=processor)
    assert prepared.inputs.kind == "tiles"
    assert all(t.dtype == np.float32 and t.shape[1] % 14 == 0 and t.shape[2] % 14 == 0
               for t in prepared.inputs.arrays)
    first = np.asarray(m._layer_pixels(detected_plate))[:112, :112]
    # the inline path: extract_patch_features asks for "pt" (_dino_support.py:342-344)
    ref = processor(images=first, return_tensors="pt", **NATIVE_PROCESSOR_KWARGS)["pixel_values"][0].numpy()
    assert np.array_equal(prepared.inputs.arrays[0], ref)                    # same bits as inline

def test_zero_regions_trigger_no_second_load(stub_backbone, plate):
    m = MeasureDinoEmbedding()
    empty = plate.copy(); empty.objmap[:] = 0
    m.encode(m.prepare_regions(empty), object())                             # never .run
    assert stub_backbone.load_calls == []

def test_entry_descriptor_for_does_not_load(stub_backbone, detected_plate):
    m = MeasureDinoEmbedding(); m.measure(detected_plate)
    model = _cached_backbone(2, "base", None, "cpu")[0]
    n = len(stub_backbone.load_calls)
    assert m.entry_descriptor_for(model) == m.entry_descriptor()
    assert len(stub_backbone.load_calls) == n
```

  The torch-free test runs in the foundation venv too: it asserts that crop
  `prepare_regions` itself imports no torch, so a fresh subprocess is the probe.
  Whether the `masked_dense` path imports torch (through the HF processor) is
  recorded in the commit message, not asserted.
- [ ] **Step 2: Run them and confirm they fail.**

  ```bash
  eval $TPT tests/unit/measure/test_measure_dino_embedding_steps.py
  ```

  Expected: `AttributeError` or the inherited `"boxes"` kind (6 failed).
- [ ] **Step 3: Implement.** Keep the crop loop's memory bound: `prepare_regions`
  builds all `N` crops at once, which at `crop_px=448` on a 1536-colony plate is
  about 0.9 GB (`:463-464`). That is acceptable in a DataLoader worker. The
  inline path must not regress: `_InlineDinoEncoder` is handed the whole array
  but feeds `embed_crops` one `batch_size` slice at a time, so peak device
  memory is as today. Say so in `prepare_regions`'s docstring.
- [ ] **Step 4: Run the P1 gate.** The golden is the gate: byte-identical.

  ```bash
  eval $TPT tests/unit/measure tests/unit/abc_ tests/unit/core/test_pipeline_embed.py tests/integration/cli/test_embedding_deliverables.py
  eval $PT tests/unit/ci/test_startup_imports.py tests/unit/ci/test_deferred_imports.py tests/unit/tune/test_annotation_coverage.py
  ```

  Expected: all pass; `test_embedding_inline_golden.py` shows `7 passed` with no
  regeneration.
- [ ] **Step 5: Commit.**

  ```bash
  git add src/phenotypic/measure/_measure_dino_embedding.py src/phenotypic/detect/nn/_helper/_dino_support.py tests/unit/measure/test_measure_dino_embedding_steps.py
  git commit -m "refactor(embed): MeasureDinoEmbedding prepares on CPU, encodes through a RegionEncoder (#282 P1)"
  ```
