# P2 · Large models, `compute_dtype`, the `BatchEncoder`, determinism

**Spec:** §4.4 (BatchEncoder, batch size, determinism, batch invariance), §4.5,
H7, U6, §5 "Batch invariance, measured on hardware"

---

**Goal:** DINOv2-giant and DINOv3 H+ and 7B are usable in reduced precision;
every entry records where and how it was computed; a `BatchEncoder` runs the
backbone over crops from many images at a fixed physical shape; and a GPU job on
three GPU models measures whether a crop's vector depends on its neighbours.

**Depends on:** P0 T1 (constants), P1.

### Task 1: Sizes, `compute_dtype`, and the `compute` block

**Files:**
- Modify: `src/phenotypic/sdk_/typing_.py:163`. Add
  `EmbeddingDinoSize = Literal["small", "base", "large", "giant", "huge_plus", "7b"]`
  and `ComputeDType = Literal["float32", "bfloat16", "float16"]`. `DinoSize`
  stays as it is (O1, accepted: the detectors do not gain the new sizes).
- Modify: `src/phenotypic/detect/nn/_helper/_dino_support.py`
  - `_DINOV3_SIZE_TO_REPO` (`:43`) gains `huge_plus` and `7b`; `hf_dino_id`
    (`:173`) maps v2 `giant` to `facebook/dinov2-giant`.
  - `load_dino_backbone` (`:194`) gains `dtype: ComputeDType = "float32"`,
    passed to `AutoModel.from_pretrained` as the torch dtype, then `.to(device)`.
    No `device_map`, so no `accelerate`.
  - `embed_crops` (`:400`) casts the normalised pixels to the model's parameter
    dtype before the forward pass and pools in float32. A float32 model is
    unchanged, so the golden holds.
  - Add `compute_block(*, device: str, stage: Literal["inline", "stage4"], physical_batch: int) -> dict[str, Any]`,
    the **one producer** of every device field (O4's condition). The entry's
    `model.revision` comes only from `resolved_revision(model)` (`:257`); no
    #282 code reads `config._commit_hash` (re-review). It calls
    `backbone_device_name` (`:361`) once, and `entry_descriptor_for` copies its
    `device` and `device_name` into `model.device` / `model.device_name`, so the
    two places cannot disagree.
- Modify: `src/phenotypic/detect/nn/_helper/_checkpoint_manager.py`.
  `Dinov2CheckpointManager._SIZE_TO_REPO` (`:678`) gains `giant`;
  `Dinov3CheckpointManager._SIZE_TO_REPO` (`:721`) gains `huge_plus`, `7b`.
- Modify: `src/phenotypic/measure/_measure_dino_embedding.py`
  - the delegate `load_dino_backbone` (`:44-62`) gains `dtype: ComputeDType = "float32"`
    and passes it through (plan review I14);
  - `size: EmbeddingDinoSize`; `_SIZE_LETTER` (`:41`) becomes `_MODEL_SLUG`,
    giving `vitg14`, `vith16plus`, `vit7b16` for the new sizes (O2, accepted)
    and today's `vit{s,b,l}{patch}` for the rest, pinned by a test;
  - new field `compute_dtype: Annotated[ComputeDType, TuneSpec(tunable=False)] = "float32"`,
    placed after `dtype`;
  - `model_validator` `_size_matches_version`: `giant` needs `version=2`;
    `huge_plus` and `7b` need `version=3`. The message names both values and the
    sizes the version has;
  - `_cached_backbone` (`:65`) keys on `(version, size, revision, device, compute_dtype)`;
  - `entry_descriptor_for` writes top-level `compute_dtype` and the `compute`
    block. The inline `entry_descriptor()` passes
    `compute_block(device=device, stage="inline", physical_batch=self.batch_size)`.
- Modify: `src/phenotypic/abc_/_measure_embedding.py:52`.
  `_DIGEST_NEUTRAL_DEFAULTS` gains `"compute_dtype": "float32"`, so a #281
  pipeline keeps its `params_sha256`.
- Modify: `src/phenotypic/_cli/_cli_embedding_deliverables.py:103,311-323`.
  `_PER_STORE_FIELDS` (`:103`) gains `"compute"`, so `_shared_descriptor`
  (`:311`) drops it as it drops `binds`; `_shared_descriptor` also fills a
  missing `compute_dtype` with `"float32"`.
- Modify (helpers), `tests/_tiny_run.py:1257`: `install_stub_backbone`'s
  `_stub_load` accepts and records `dtype=`; the parametric stub (P0 T2) casts
  its `Linear` to it, so a bf16 load is observable.
- Test: `tests/unit/measure/test_measure_dino_embedding_sizes.py` (new) **[torch]**;
  `tests/unit/cli/test_embedding_grouping_provenance.py` (new, shared env).

**Interfaces:**
- Produces:
  - `MeasureDinoEmbedding.compute_dtype`, and the entry fields `compute_dtype`
    and `compute = {device, device_name, capability, torch, cuda, physical_batch, stage}`.
    `capability` is `"<major>.<minor>"` on CUDA, `None` on CPU; `cuda` is
    `torch.version.cuda` or `None`.
  - `compute_block(...)` (above), the one producer of that block. P4 T3 calls it
    with `stage="stage4"` and the encoder's physical batch.

- [ ] **Step 1: Write the failing tests.**

```python
@pytest.mark.parametrize("size, version", [("giant", 3), ("huge_plus", 2), ("7b", 2)])
def test_impossible_size_version_pairs_are_refused(size, version):
    with pytest.raises(ValidationError, match=f"{size}.*version"):
        MeasureDinoEmbedding(size=size, version=version)

@pytest.mark.parametrize("size, version, token", [
    ("giant", 2, "dinov2-vitg14"), ("huge_plus", 3, "dinov3-vith16plus"), ("7b", 3, "dinov3-vit7b16")])
def test_new_model_tokens(size, version, token):
    assert MeasureDinoEmbedding(size=size, version=version).model_token() == token

def test_compute_dtype_is_descriptor_identity(param_stub_backbone, detected_plate):
    a = MeasureDinoEmbedding().payload(detected_plate).descriptor
    b = MeasureDinoEmbedding(compute_dtype="bfloat16").payload(detected_plate).descriptor
    assert (a["compute_dtype"], b["compute_dtype"]) == ("float32", "bfloat16")
    assert param_stub_backbone.load_calls[-1]["dtype"] == "bfloat16"

@pytest.mark.parametrize("size, version, token", [
    ("small", 2, "dinov2-vits14"), ("base", 2, "dinov2-vitb14"), ("large", 2, "dinov2-vitl14"),
    ("small", 3, "dinov3-vits16"), ("base", 3, "dinov3-vitb16"), ("large", 3, "dinov3-vitl16")])
def test_existing_model_tokens_are_unchanged(size, version, token):
    assert MeasureDinoEmbedding(size=size, version=version).model_token() == token   # O2's condition

def test_model_and_compute_device_fields_agree(param_stub_backbone, detected_plate, monkeypatch):
    monkeypatch.setattr(dino_support, "backbone_device_name", lambda device: "NVIDIA RTX 6000 Ada")
    d = MeasureDinoEmbedding().payload(detected_plate).descriptor
    assert (d["model"]["device"], d["model"]["device_name"]) == (
        d["compute"]["device"], d["compute"]["device_name"])                         # O4's condition

def test_a_bfloat16_load_reaches_the_weights(param_stub_backbone, detected_plate):
    m = MeasureDinoEmbedding(compute_dtype="bfloat16"); m.payload(detected_plate)
    model = m._backbone()[0]
    assert next(model.parameters()).dtype == torch.bfloat16

def test_the_revision_comes_from_resolved_revision(monkeypatch, detected_plate):
    # transformers >= 5.18 sets no config._commit_hash (#281 defect; gridname's
    # snapshot-path fallback in resolved_revision, _dino_support.py:257-272)
    install_stub_backbone(monkeypatch, parametric=True, commit_hash=None,
                          snapshot_revision="d" * 40)
    d = MeasureDinoEmbedding(device="cpu").payload(detected_plate).descriptor
    assert d["model"]["revision"] == "d" * 40

def test_no_282_module_reads_commit_hash_directly():
    offenders = [p for p in SRC.rglob("*.py")
                 if "_commit_hash" in p.read_text(encoding="utf-8") and p.name != "_dino_support.py"]
    assert offenders == []                                       # only resolved_revision reads it

def test_inline_entries_record_their_compute(param_stub_backbone, detected_plate):
    compute = MeasureDinoEmbedding(batch_size=32).payload(detected_plate).descriptor["compute"]
    assert compute["stage"] == "inline" and compute["physical_batch"] == 32
    assert compute["device"] == "cpu" and compute["capability"] is None

def test_default_float32_keeps_the_281_params_digest():
    assert MeasureDinoEmbedding()._params_sha256() == SHA_281_DEFAULT_EMBEDDER
```

  `install_stub_backbone(..., commit_hash=None, snapshot_revision=...)` (P0 T2
  helper options added here; `snapshot_from_revision=True` instead takes the
  revision from the load's `revision=`, for P5 T2) builds a config with no `_commit_hash` and a
  `_name_or_path` laid out as a hub snapshot,
  `<tmp>/models--facebook--dinov2-base/snapshots/<revision>`; spell it as
  gridname's fallback reads it. `SRC` is `src/phenotypic`. The revision test
  needs that #281 fix merged first; until then it fails, which is correct.
  `SHA_281_DEFAULT_EMBEDDER` is a module constant: the digest printed by
  `MeasureDinoEmbedding()._params_sha256()` on the #281 merge commit, pasted in
  by hand with that SHA in a comment.

```python
def test_compute_never_splits_a_deliverable_group():
    base = {"model": {"family": "dinov2", "revision": "a" * 40}, "dim": 768, "dtype": "float16",
            "pooling": "crop_cls", "compute_dtype": "float32"}
    gpu = {**base, "compute": {"device": "cuda", "stage": "stage4", "physical_batch": 64}}
    cpu = {**base, "compute": {"device": "cpu", "stage": "inline", "physical_batch": 64}}
    old = {k: v for k, v in base.items() if k != "compute_dtype"}         # a #281 entry
    assert _shared_descriptor(gpu) == _shared_descriptor(cpu) == _shared_descriptor(old)
    assert _shared_descriptor({**base, "compute_dtype": "bfloat16"}) != _shared_descriptor(base)
```

  `_shared_descriptor` is the grouping function today
  (`_cli_embedding_deliverables.py:311`); do not add a second.
- [ ] **Step 2: Run them and confirm they fail.**

  ```bash
  eval $TPT tests/unit/measure/test_measure_dino_embedding_sizes.py
  eval $PT tests/unit/cli/test_embedding_grouping_provenance.py
  ```

- [ ] **Step 3: Implement** the Files list.
- [ ] **Step 4: Verify the hub facts before anything uses them** (spec §4.5).
  From a node with network, for each new repo id, fetch `config.json` at the
  default branch with `huggingface_hub.hf_hub_download` and record
  `hidden_size`, `num_attention_heads`, `num_hidden_layers`, `patch_size`,
  `num_register_tokens` and the commit. DINOv3 ids need the licence-accepted
  token (stop point 4). Paste the values into the P0 T1 model table if they
  differ, re-run `physical_batch_size.py`, and record the commits in the commit
  message. Expected: `dinov2-giant` 1536-d, patch 14, no registers;
  `dinov3-vith16plus` 1280-d, `dinov3-vit7b16` 4096-d, patch 16, 4 registers.
  Any other value stops the task.
- [ ] **Step 5: Confirm the loader keyword and the load footprint.** In the
  foundation venv, check which of `dtype=` and `torch_dtype=` the installed
  transformers' `from_pretrained` accepts without a deprecation warning, and use
  that one. Record the peak RSS of a `7b` bf16 load in a Slurm job
  (`intel`, 4 CPU, 48 GB, 1 h), if the weights are reachable. Expected: below 48 GB.
- [ ] **Step 6: Run the tests, the golden, the tune gate and the startup guards.**

  ```bash
  eval $TPT tests/unit/measure tests/unit/detect/nn
  eval $PT tests/unit/cli/test_embedding_grouping_provenance.py tests/unit/tune/test_annotation_coverage.py tests/unit/ci/test_startup_imports.py tests/unit/ci/test_deferred_imports.py
  ```

  Expected: all pass; golden `7 passed` (its descriptor comparison drops
  `compute` and `compute_dtype`, P0 T2).
- [ ] **Step 7: Commit.**

  ```bash
  git add src/phenotypic/sdk_/typing_.py src/phenotypic/detect/nn/_helper/_dino_support.py src/phenotypic/detect/nn/_helper/_checkpoint_manager.py src/phenotypic/measure/_measure_dino_embedding.py src/phenotypic/abc_/_measure_embedding.py src/phenotypic/_cli/_cli_embedding_deliverables.py tests/_tiny_run.py tests/unit/measure/test_measure_dino_embedding_sizes.py tests/unit/cli/test_embedding_grouping_provenance.py
  git commit -m "feat(embed): giant, H+ and 7B, compute_dtype, and the compute provenance block (#282 P2)"
  ```

### Task 2: Determinism and the physical batch size

**Files:**
- Create: `src/phenotypic/detect/nn/_helper/_batch_encoder.py` (torch imported
  inside functions only)
- Modify: `src/phenotypic/measure/_measure_dino_embedding.py`: add
  `backbone_key()` (below), importing `BackboneKey` inside the method.
- Test: `tests/unit/detect/nn/test_batch_size.py` (new, shared env; no torch
  needed) and `tests/unit/detect/nn/test_determinism.py` (new) **[torch]**

**Interfaces:**
- Produces, in `_batch_encoder.py`:
  - `@dataclass(frozen=True) class BackboneKey`: `version: int`, `size: str`,
    `revision: str | None`, `compute_dtype: str`; and
    `sort_key(self) -> tuple[int, str, str, str]`, with `revision or ""`, since
    `None` and a string do not compare. It gives H10's backbone-major order.
  - `MeasureDinoEmbedding.backbone_key(self) -> BackboneKey`:
    `BackboneKey(int(self.version), self.size, self.revision, self.compute_dtype)`.
  - Constants `MEMORY_FRACTION`, `RESERVE_BYTES`, `ACTIVATION_FACTOR`, copied
    from P0 T1's `physical_batch_size.py`, whose path the docstring names.
  - `tokens_for(crop_px: int, patch: int, registers: int) -> int`
  - `per_sample_bytes(*, tokens: int, hidden: int, heads: int, dtype_bytes: int) -> int`
  - `physical_batch_size(*, batch_size: int, gpu_total_bytes: int, weight_bytes: int, per_sample: int) -> int`.
    Raises `ValueError` when the budget is not positive (weights do not fit).
  - `check_cublas_workspace() -> None`: the variable check alone, which
    `configure_determinism` calls on CUDA; separate so it is testable without a
    GPU.
  - `configure_determinism() -> dict[str, object]`: applies the five settings in
    the README's Global Constraints and returns them, for the shard log. It
    accepts `CUBLAS_WORKSPACE_CONFIG` of `":4096:8"` or `":16:8"`, the two
    values cuBLAS documents as deterministic, so a user's own `:16:8` is kept
    (re-review item 6). Any other value, or none, raises `RuntimeError` naming
    the variable and both values when the device is CUDA.

- [ ] **Step 1: Write the failing tests.**

```python
def test_pow2_floor_and_the_batch_size_cap():
    per = per_sample_bytes(tokens=tokens_for(224, 14, 0), hidden=768, heads=12, dtype_bytes=4)
    assert physical_batch_size(batch_size=64, gpu_total_bytes=48 << 30,
                               weight_bytes=86_000_000 * 4, per_sample=per) == 64
    m = physical_batch_size(batch_size=4096, gpu_total_bytes=48 << 30,
                            weight_bytes=86_000_000 * 4, per_sample=per)
    assert m & (m - 1) == 0 and m < 4096

def test_the_values_match_the_logic_validation_script():
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("pbs", LOGIC_SCRIPTS / "physical_batch_size.py")
    pbs = module_from_spec(spec); spec.loader.exec_module(pbs)
    assert (MEMORY_FRACTION, RESERVE_BYTES, ACTIVATION_FACTOR) == (
        pbs.MEMORY_FRACTION, pbs.RESERVE_BYTES, pbs.ACTIVATION_FACTOR)

@pytest.mark.parametrize("value, ok", [(":4096:8", True), (":16:8", True), (":0:0", False), (None, False)])
def test_cublas_workspace_values(monkeypatch, value, ok):
    if value is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", value)
    if ok:
        check_cublas_workspace()
    else:
        with pytest.raises(RuntimeError, match=r":4096:8.*:16:8"):
            check_cublas_workspace()

def test_weights_that_do_not_fit_are_refused():
    with pytest.raises(ValueError, match="does not fit"):            # 7B fp32 on a 16 GiB card (P0 T1 check 3)
        physical_batch_size(batch_size=64, gpu_total_bytes=16 << 30,
                            weight_bytes=6_700_000_000 * 4, per_sample=1)
```

  `LOGIC_SCRIPTS` is the repo path of P0 T1's folder. This test reads the
  script, which is allowed: the rule is that the script never imports
  `phenotypic`, not the reverse.

```python
def test_determinism_settings(monkeypatch):
    torch = pytest.importorskip("torch")
    applied = configure_determinism()
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.are_deterministic_algorithms_enabled()
    assert applied["float32_matmul_precision"] == "highest"
```

- [ ] **Step 2: Run them and confirm they fail** (`ModuleNotFoundError`).
- [ ] **Step 3: Implement.** `configure_determinism` is idempotent. The test
  restores torch's global flags in a fixture's teardown, so no later test runs
  deterministic by accident.
- [ ] **Step 4: Run them.** `eval $PT tests/unit/detect/nn/test_batch_size.py`
  and `eval $TPT tests/unit/detect/nn/test_determinism.py`. Expected: pass.
- [ ] **Step 5: Commit.**

  ```bash
  git add src/phenotypic/detect/nn/_helper/_batch_encoder.py src/phenotypic/measure/_measure_dino_embedding.py tests/unit/detect/nn/test_batch_size.py tests/unit/detect/nn/test_determinism.py
  git commit -m "feat(embed): BackboneKey, the physical batch rule and the determinism settings (#282 P2)"
  ```

### Task 3: `BatchEncoder`

**Files:**
- Modify: `src/phenotypic/detect/nn/_helper/_batch_encoder.py`
- Test: `tests/unit/detect/nn/test_batch_encoder.py` (new) **[torch]**, CPU, on
  P0 T2's parametric stub (`_build_param_stub_dino`), and its `leaky=True`
  variant for the mutation check

**Interfaces:**
- Consumes: `BackboneInputs` (P1 T1), `backbone_patch_size`,
  `reshape_patch_tokens` (`_dino_support.py:62,280`), P0 T1's
  `MAX_IN_FLIGHT_OUTPUT_BYTES`.
- Produces:
  - `class BatchEncoder`:
    - `__init__(self, model: Any, processor: Any, *, device: str, compute_dtype: ComputeDType, physical_batch: int, max_in_flight_bytes: int = MAX_IN_FLIGHT_OUTPUT_BYTES)`
    - `submit(self, ticket: Hashable, inputs: BackboneInputs) -> None`.
      Queues `inputs`' rows by `(h, w, pooling)`, one queue per key. A full
      queue runs at once. A zero-row submission completes at once, with no
      forward pass. **When the output bytes held for unfinished tickets exceed
      `max_in_flight_bytes`, every partial queue is flushed** (zero-padded).
      The bound is in bytes, not images (re-review item 4): a crop ticket
      holds `N · D · 4` bytes, so crop pipelines never reach it and keep full
      batches; a `masked_dense` ticket holds its feature grid
      (`ENCODER_OUTPUT_BYTES`), so only those flush early (plan review I2).
    - `held_bytes: int` (read-only): the output bytes held for unfinished tickets.
    - `flush(self) -> None`: runs every partial queue, zero-padded to exactly
      `physical_batch`.
    - `drain(self) -> Iterator[tuple[Hashable, list[np.ndarray]]]`: yields each
      ticket **as soon as** all its rows are done, in completion order, not
      submission order, and drops its buffers when yielded. Outputs are shaped
      as `RegionEncoder.run` (P1 T1) returns them.
    - `physical_batch: int` (read-only), `halve(self) -> bool`: halves it once;
      `False` when already halved or at 1.
    - `as_region_encoder(self) -> RegionEncoder`: a synchronous adaptor
      (`submit`, `flush`, `drain` one ticket), so `MeasureEmbedding.encode` runs on
      it in tests.
  - Every forward pass has exactly `physical_batch` inputs. Crops are
    normalised as `embed_crops` normalises them (`_dino_support.py:449-462`);
    tiles arrive already normalised by the processor (P1 T2). Both are cast to
    `compute_dtype` and run under `torch.inference_mode()`; CLS, mean and patch
    outputs are taken in float32.
  - `class Stage4OutOfMemory(RuntimeError)`: raised by a forward pass that runs
    out of memory when `halve()` returns `False`. A batch that ran out of memory
    is retried whole at the halved size, so no row is lost.

- [ ] **Step 1: Write the failing tests.**

```python
def test_every_forward_pass_has_exactly_m_inputs(stub_model):
    seen = record_batch_sizes(stub_model)
    enc = BatchEncoder(*stub_model, device="cpu", compute_dtype="float32", physical_batch=8)
    enc.submit("a", crops(5)); enc.submit("b", crops(6)); enc.flush()
    assert seen == [8, 8]

def test_vectors_route_back_by_ticket_and_row(stub_model):
    enc = BatchEncoder(*stub_model, device="cpu", compute_dtype="float32", physical_batch=4)
    a, b = crops(3, seed=1), crops(5, seed=2)
    enc.submit("a", a); enc.submit("b", b); enc.flush()
    out = dict(enc.drain())
    assert np.array_equal(out["a"][0], embed_alone(stub_model, a))
    assert np.array_equal(out["b"][0], embed_alone(stub_model, b))

def rows_are_isolated(model_pair) -> bool:
    probe = crops(1, seed=7)
    vectors = []
    for seed in (1, 2, 3):
        enc = BatchEncoder(*model_pair, device="cpu", compute_dtype="float32", physical_batch=4)
        enc.submit("n", crops(seed + 1, seed=seed)); enc.submit("p", probe); enc.flush()
        vectors.append(dict(enc.drain())["p"][0])
    return all(np.array_equal(v, vectors[0]) for v in vectors)

def test_a_row_does_not_depend_on_its_neighbours_on_cpu(stub_model):
    assert rows_are_isolated(stub_model)

def test_the_isolation_check_catches_a_leaky_model(leaky_stub_model):
    assert not rows_are_isolated(leaky_stub_model)              # the check above can fail (plan review I13)

def test_a_finished_image_is_drained_before_a_partial_batch(stub_model):
    enc = BatchEncoder(*stub_model, device="cpu", compute_dtype="float32", physical_batch=4)
    enc.submit("full", crops(4)); enc.submit("partial", crops(1))
    assert [t for t, _ in enc.drain()] == ["full"]              # not held behind "partial"

def test_crop_pipelines_keep_full_batches(stub_model):
    seen = record_batch_sizes(stub_model)
    enc = BatchEncoder(*stub_model, device="cpu", compute_dtype="float32", physical_batch=8)
    for i in range(20):                                          # 20 images of 3 crops: 60 rows
        enc.submit(i, crops(3, seed=i))
    enc.flush()
    assert seen[:-1] == [8] * 7 and len(seen) == 8               # no early flush; only the last is padded

def test_dense_grids_flush_on_the_byte_bound(stub_model):
    seen = record_batch_sizes(stub_model)
    one_grid = tile_output_bytes(stub_model, px=112)             # the stub's grid for one 112-px tile
    enc = BatchEncoder(*stub_model, device="cpu", compute_dtype="float32", physical_batch=8,
                       max_in_flight_bytes=2 * one_grid)
    for t in "abc":
        enc.submit(t, tiles(1, px=112))
    assert {t for t, _ in enc.drain()} >= {"a", "b"}             # the third ticket crossed the bound
    assert enc.held_bytes <= 2 * one_grid

def test_one_queue_per_geometry(stub_model):
    seen = record_batch_shapes(stub_model)
    enc = BatchEncoder(*stub_model, device="cpu", compute_dtype="float32", physical_batch=2)
    enc.submit("a", crops(1, px=112)); enc.submit("b", crops(1, px=224)); enc.flush()
    assert sorted(seen) == [(2, 3, 112, 112), (2, 3, 224, 224)]

def test_an_empty_input_completes_without_a_forward_pass(stub_model):
    seen = record_batch_sizes(stub_model)
    enc = BatchEncoder(*stub_model, device="cpu", compute_dtype="float32", physical_batch=4)
    enc.submit("empty", crops(0))
    assert [t for t, _ in enc.drain()] == ["empty"] and seen == []

def test_oom_once_halves_and_succeeds(stub_model):
    enc = BatchEncoder(*oom_once(stub_model), device="cpu", compute_dtype="float32", physical_batch=8)
    a = crops(8)
    enc.submit("a", a); enc.flush()
    assert enc.physical_batch == 4
    assert np.array_equal(dict(enc.drain())["a"][0], embed_alone(stub_model, a))

def test_oom_twice_raises(stub_model):
    enc = BatchEncoder(*oom_always(stub_model), device="cpu", compute_dtype="float32", physical_batch=8)
    enc.submit("a", crops(8))
    with pytest.raises(Stage4OutOfMemory):
        enc.flush()
    assert enc.physical_batch == 4

@pytest.mark.parametrize("pooling", ["crop_cls", "crop_mean", "masked_dense"])
def test_the_batch_encoder_matches_the_inline_encoder(param_stub_backbone, detected_plate, pooling):
    m = MeasureDinoEmbedding(pooling=pooling, device="cpu", crop_px=112)
    model, processor, _ = m._backbone()
    prepared = m.prepare_regions(detected_plate, processor=processor)
    inline = m.encode(prepared, m.inline_encoder(detected_plate))
    batched = m.encode(prepared, BatchEncoder(model, processor, device="cpu",
                       compute_dtype="float32", physical_batch=64).as_region_encoder())
    assert np.array_equal(inline, batched)
```

  Module-local helpers: `stub_model` and `leaky_stub_model` return
  `_build_param_stub_dino()` and `_build_param_stub_dino(leaky=True)` as
  `(model, processor)`; `crops(n, *, px=224, seed=0)` builds a `"crops"`
  `BackboneInputs` and `tiles(n, *, px)` a `"tiles"` one; `tile_output_bytes`
  is one tile's grid in bytes; `record_batch_sizes` / `record_batch_shapes` add a forward
  pre-hook; `embed_alone` runs the model on just those rows; `oom_once` raises
  `torch.OutOfMemoryError` on the first forward only; `oom_always` on every one.
- [ ] **Step 2: Run them and confirm they fail.**

  ```bash
  eval $TPT tests/unit/detect/nn/test_batch_encoder.py
  ```

- [ ] **Step 3: Implement.** The parametrised last test is the H1 promise at the
  encoder level for all three poolings: on the CPU and the stub, the batched path
  equals the inline path bit for bit. `masked_dense` passes because tiles carry
  the processor's own pixels (P1 T2, plan review I1).
- [ ] **Step 4: Run the P2 unit surface.**

  ```bash
  eval $TPT tests/unit/detect/nn tests/unit/measure tests/unit/abc_
  eval $PT tests/unit/ci/test_startup_imports.py tests/unit/ci/test_deferred_imports.py
  ```

- [ ] **Step 5: Commit.**

  ```bash
  git add src/phenotypic/detect/nn/_helper/_batch_encoder.py tests/unit/detect/nn/test_batch_encoder.py
  git commit -m "feat(embed): BatchEncoder: fixed physical batch, out-of-order drain, bounded in flight (#282 P2)"
  ```

### Task 4: The GPU batch-invariance job (Lane B)

**What it measures** (spec goal 3 as restated, S1): within one GPU model and
one physical batch size, a crop's stored vector does not depend on its batch
neighbours, its position, its shard, or the process. Across GPU models, and
between `m` and `m/2`, the distance is measured and reported, not required to
be zero.

**Files** (beside the plan, because they drive shipped code):
- Create: `docs/superpowers/plans/2026-11-12-gpu-embedding-staging/batch_invariance.py`
- Create: `docs/superpowers/plans/2026-11-12-gpu-embedding-staging/batch_invariance.sbatch`
- Create: `docs/superpowers/plans/2026-11-12-gpu-embedding-staging/submit_batch_invariance.sh`
- Results: `/bigdata/exfab/anguy344/gate-results/c6-batch-invariance/<gpu>.json`
  and the report `docs/superpowers/reports/2026-11-12-gpu-embedding-staging/batch-invariance.md`

**Interfaces:**
- Consumes: `MeasureDinoEmbedding.prepare_regions`, `load_dino_processor`,
  `BatchEncoder`, `configure_determinism`, `physical_batch_size`.
- Produces: per GPU, a JSON report
  `{gpu, capability, torch, cuda, models: {row: {compute_dtype, pooling, m, invariant: bool, cases: {...}, half_m_max_cos, f16_max_abs, f16_overflow, determinism_warnings: [...]}}}`
  and the cross-GPU table.

- [ ] **Step 1: Write `batch_invariance.py <out.json> [--repeat-of <json>]`.**
  Inputs are the three stores spec §2 measured
  (`gpu-baseline/measure_embedding_baseline.py:25`), read only, objects and a
  96-well grid. Rows: DINOv2-base float32 `crop_cls`; DINOv2-base float32
  `masked_dense`; DINOv2-large float32; DINOv2-giant bfloat16; DINOv3-7B
  bfloat16 (when its licence and token are present, stop point 4). For each:
  1. `warnings.simplefilter("error")` around the forward passes, so any
     determinism warning from `use_deterministic_algorithms(warn_only=True)`
     fails the row and is listed (plan review, Minor).
  2. `configure_determinism()`; compute `m` with `physical_batch_size`.
  3. Embed every input in **reference** order at physical batch `m`.
  4. Re-embed under each perturbation, at the same `m`: reversed order; a seeded
     shuffle; each crop moved to position 0 and to position `m − 1`; neighbours
     from another plate; the set split across two `BatchEncoder` instances (two
     "shards"); the last batch padded with 1 vs `m − 1` zeros; and a
     `max_in_flight_bytes=1` run (forced early flushes).
  5. `invariant` is true when every **stored float16** vector is bit-identical
     to the reference in every case. The float32 max abs difference is
     information.
  6. Re-embed at `m/2` and record the max cosine distance to the reference as
     `half_m_max_cos` (information: OOM halving changes `m`, S1).
  7. Record `f16_max_abs` and whether any value exceeds 65504 (the 7B and ViT-g
     rows the spec asks for).
  8. With `--repeat-of`, compare every vector with the earlier process's file:
     a separate-process repeat must be bit-identical too.
- [ ] **Step 2: Write the sbatch script and the submitter.** The submitter adds a
  detached worktree at the SHA under test
  (`/bigdata/exfab/anguy344/gate-worktrees/c6-<sha9>`), submits two jobs per GPU
  type (the second runs `--repeat-of` the first, `afterok`), and an `afterany`
  cleanup, and checks every captured id against `^[0-9]+$`. Per job: 8 CPU,
  64 GB, 1:30:00, `HF_HUB_OFFLINE=1` after the weights are pre-staged,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, the foundation venv,
  `PYTHONPATH=<tree>/src:<tree>`.
  - Ada: `exfab`, `--account=exfab`, `--gres=gpu:1`.
  - A100 and Blackwell: `short_gpu --constraint=gpu_latest`, pinned by GRES type.
    First list the type names with `sinfo -p short_gpu -o '%N %G %f'`; do not
    guess them.
  - Pick each job's partition by measured predicted start where both can serve
    it (README conventions). Never `preempt`.
- [ ] **Step 3: Pre-stage the weights** in a CPU job with network
  (`intel`, 2 CPU, 16 GB, 1 h) into the HF cache the GPU jobs read.
- [ ] **Step 4: Submit, and verify each job started** (`scontrol show job <id> | grep -E 'StartTime|Reason'`).
- [ ] **Step 5: Write the report.** One table per GPU (row, dtype, pooling, m,
  invariant, repeat-identical, f32 max diff, `half_m_max_cos`, f16 max abs,
  determinism warnings) and a cross-GPU table (median and max cosine distance
  per row, Ada vs A100 vs Blackwell).
  - **Every `invariant` and every repeat true, no determinism warning:** P2
    passes its gate.
  - **Any of them false: stop (stop point 3).** Do not start P4. Report the
    case, row and GPU to the user, with the spec's §5 fallback (an
    m = 1-per-crop mode) as the option to decide on.
  - **Any `f16_overflow`:** report it. The run fails that image as non-finite
    (spec §4.8), so it is a docs finding, not a stop.
- [ ] **Step 6: Commit the scripts and the report.**

  ```bash
  git add docs/superpowers/plans/2026-11-12-gpu-embedding-staging/batch_invariance.py docs/superpowers/plans/2026-11-12-gpu-embedding-staging/batch_invariance.sbatch docs/superpowers/plans/2026-11-12-gpu-embedding-staging/submit_batch_invariance.sh docs/superpowers/reports/2026-11-12-gpu-embedding-staging/batch-invariance.md
  git commit -m "docs(measure): GPU batch invariance on Ada, A100 and Blackwell (#282 P2)"
  ```

**P2 phase gate:** the affected surface (importers of `_measure_embedding`,
`_measure_dino_embedding`, `_dino_support`, `_batch_encoder`,
`_cli_embedding_deliverables`) in both lanes, then the **deep P2 review**
(implementation-test-reviewer, Opus), written to
`reports/2026-11-12-gpu-embedding-staging/review-p2.md`.
