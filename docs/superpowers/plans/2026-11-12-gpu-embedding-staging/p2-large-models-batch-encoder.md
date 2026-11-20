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
  stays as it is **(pending O1)**.
- Modify: `src/phenotypic/detect/nn/_helper/_dino_support.py`
  - `_DINOV3_SIZE_TO_REPO` (`:43`) gains `huge_plus` and `7b`; `hf_dino_id`
    (`:173`) maps v2 `giant` to `facebook/dinov2-giant`.
  - `load_dino_backbone` (`:194`) gains `dtype: ComputeDType = "float32"`,
    passed to `AutoModel.from_pretrained` as the torch dtype, then `.to(device)`.
    No `device_map`, so no `accelerate`.
  - `embed_crops` (`:400`) casts the normalised pixels to the model's parameter
    dtype before the forward pass and pools in float32. A float32 model is
    unchanged, so the golden holds.
  - Add `compute_block(*, device: str, stage: Literal["inline", "stage4"], physical_batch: int) -> dict[str, Any]`.
- Modify: `src/phenotypic/detect/nn/_helper/_checkpoint_manager.py`.
  `Dinov2CheckpointManager._SIZE_TO_REPO` (`:678`) gains `giant`;
  `Dinov3CheckpointManager._SIZE_TO_REPO` (`:721`) gains `huge_plus`, `7b`.
- Modify: `src/phenotypic/measure/_measure_dino_embedding.py`
  - `size: EmbeddingDinoSize`; `_SIZE_LETTER` (`:41`) becomes `_MODEL_SLUG`,
    giving `vitg14`, `vith16plus`, `vit7b16` for the new sizes **(pending O2)**
    and today's `vit{s,b,l}{patch}` for the rest;
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
  `_stub_load` accepts and records `dtype=`; the stub honours it (casts its
  weights).
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

def test_compute_dtype_is_descriptor_identity(stub_backbone, detected_plate):
    a = MeasureDinoEmbedding().payload(detected_plate).descriptor
    b = MeasureDinoEmbedding(compute_dtype="bfloat16").payload(detected_plate).descriptor
    assert (a["compute_dtype"], b["compute_dtype"]) == ("float32", "bfloat16")
    assert stub_backbone.load_calls[-1]["dtype"] == "bfloat16"

def test_inline_entries_record_their_compute(stub_backbone, detected_plate):
    compute = MeasureDinoEmbedding(batch_size=32).payload(detected_plate).descriptor["compute"]
    assert compute["stage"] == "inline" and compute["physical_batch"] == 32
    assert compute["device"] == "cpu" and compute["capability"] is None

def test_default_float32_keeps_the_281_params_digest():
    assert MeasureDinoEmbedding()._params_sha256() == SHA_281_DEFAULT_EMBEDDER
```

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
  - `configure_determinism() -> dict[str, object]`: applies the five settings in
    the README's Global Constraints and returns them, for the shard log. It
    raises `RuntimeError` naming the variable when `CUBLAS_WORKSPACE_CONFIG` is
    not `":4096:8"` and the device is CUDA.

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

def test_weights_that_do_not_fit_are_refused():
    with pytest.raises(ValueError, match="does not fit"):
        physical_batch_size(batch_size=64, gpu_total_bytes=40 << 30,
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
  `_build_stub_dino` (`tests/_tiny_run.py`, the stub `install_stub_backbone` loads)

**Interfaces:**
- Consumes: `BackboneInputs` (P1 T1), `backbone_patch_size`,
  `reshape_patch_tokens` (`_dino_support.py:62,280`).
- Produces:
  - `class BatchEncoder`:
    - `__init__(self, model: Any, processor: Any, *, device: str, compute_dtype: ComputeDType, physical_batch: int)`
    - `submit(self, ticket: Hashable, inputs: BackboneInputs) -> None`.
      Queues `inputs`' rows by geometry `(h, w)`; one queue per geometry.
      A zero-row submission completes at once.
    - `flush(self) -> None`: runs every queue's partial batch, zero-padded to
      exactly `physical_batch`.
    - `drain(self) -> Iterator[tuple[Hashable, list[np.ndarray]]]`: yields each
      ticket whose rows are all done, in submission order, with outputs shaped as
      `RegionEncoder.run` (P1 T1) returns them.
    - `physical_batch: int` (read-only), `halve(self) -> bool`: halves it once;
      `False` when already halved or at 1.
    - `as_region_encoder(self) -> RegionEncoder`: a synchronous adaptor
      (`submit`, `flush`, `drain` one ticket), so `MeasureEmbedding.encode` runs on
      it in tests.
  - Every forward pass has exactly `physical_batch` inputs; pixels are
    normalised as `embed_crops` normalises them (`_dino_support.py:449-462`),
    cast to `compute_dtype`, run under `torch.inference_mode()`; CLS, mean and
    patch outputs are taken in float32.
  - `class Stage4OutOfMemory(RuntimeError)`: raised by a forward pass that runs
    out of memory when `halve()` returns `False`.

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

def test_a_row_does_not_depend_on_its_neighbours_on_cpu(stub_model):
    probe = crops(1, seed=7)
    vectors = []
    for seed in (1, 2, 3):
        enc = BatchEncoder(*stub_model, device="cpu", compute_dtype="float32", physical_batch=4)
        enc.submit("n", crops(seed + 1, seed=seed)); enc.submit("p", probe); enc.flush()
        vectors.append(dict(enc.drain())["p"][0])
    assert all(np.array_equal(v, vectors[0]) for v in vectors)

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

def test_oom_halves_once_then_raises(stub_model):
    enc = BatchEncoder(*oom_once_then_always(stub_model), device="cpu",
                       compute_dtype="float32", physical_batch=8)
    enc.submit("a", crops(8))
    with pytest.raises(Stage4OutOfMemory):
        enc.flush()
    assert enc.physical_batch == 4

def test_the_batch_encoder_matches_the_inline_encoder(stub_backbone, detected_plate):
    m = MeasureDinoEmbedding()
    prepared = m.prepare_regions(detected_plate)
    inline = m.encode(prepared, m.inline_encoder(detected_plate))
    model, processor, _ = m._backbone()
    batched = m.encode(prepared, BatchEncoder(model, processor, device="cpu",
                       compute_dtype="float32", physical_batch=64).as_region_encoder())
    assert np.array_equal(inline, batched)
```

  The module-local helpers (`stub_model`, `crops(n, *, px=224, seed=0)`,
  `record_batch_sizes`, `record_batch_shapes`, `embed_alone`,
  `oom_once_then_always`) wrap `_build_stub_dino` with a forward hook or a
  raising stand-in. `oom_once_then_always` raises `torch.OutOfMemoryError` on
  every call, so the first OOM halves 8 → 4 and the second raises.
- [ ] **Step 2: Run them and confirm they fail.**

  ```bash
  eval $TPT tests/unit/detect/nn/test_batch_encoder.py
  ```

- [ ] **Step 3: Implement.** Tiles of one geometry share a queue with crops of
  that geometry only if their pooling matches; key the queue by
  `(h, w, pooling)`. The last test is the H1 promise at the encoder level: on the
  CPU and the stub, the batched path equals the inline path bit for bit.
- [ ] **Step 4: Run the P2 unit surface.**

  ```bash
  eval $TPT tests/unit/detect/nn tests/unit/measure tests/unit/abc_
  eval $PT tests/unit/ci/test_startup_imports.py tests/unit/ci/test_deferred_imports.py
  ```

- [ ] **Step 5: Commit.**

  ```bash
  git add src/phenotypic/detect/nn/_helper/_batch_encoder.py tests/unit/detect/nn/test_batch_encoder.py
  git commit -m "feat(embed): BatchEncoder, fixed physical batch, one queue per geometry (#282 P2)"
  ```

### Task 4: The GPU batch-invariance job (Lane B)

**Files** (beside the plan, because they drive shipped code):
- Create: `docs/superpowers/plans/2026-11-12-gpu-embedding-staging/batch_invariance.py`
- Create: `docs/superpowers/plans/2026-11-12-gpu-embedding-staging/batch_invariance.sbatch`
- Create: `docs/superpowers/plans/2026-11-12-gpu-embedding-staging/submit_batch_invariance.sh`
- Results: `/bigdata/exfab/anguy344/gate-results/c6-batch-invariance/<gpu>.json`
  and the report `docs/superpowers/reports/2026-11-12-gpu-embedding-staging/batch-invariance.md`

**Interfaces:**
- Consumes: `MeasureDinoEmbedding.prepare_regions`, `BatchEncoder`,
  `configure_determinism`, `physical_batch_size`.
- Produces: per GPU, a JSON report
  `{gpu, capability, torch, cuda, models: {model_token: {compute_dtype, m, invariant: bool, cases: {...}, f16_max_abs, f16_overflow}}}`
  and the cross-GPU table.

- [ ] **Step 1: Write `batch_invariance.py <out.json>`.** Inputs are the three
  stores spec §2 measured (`gpu-baseline/measure_embedding_baseline.py:25`),
  read only, objects and a 96-well grid. For each model row:
  DINOv2-base float32, DINOv2-large float32, DINOv2-giant bfloat16, DINOv3-7B
  bfloat16 (when its licence and token are present, stop point 4):
  1. `configure_determinism()`; compute `m` with `physical_batch_size`.
  2. Embed every crop in **reference** order with physical batch `m`.
  3. Re-embed under each perturbation, at the same `m`: reversed order; a seeded
     shuffle; each crop moved to position 0 and to position `m − 1`; neighbours
     from another plate; the set split across two `BatchEncoder` instances (two
     "shards"); the last batch padded with 1 vs `m − 1` zeros.
  4. `invariant` is true when every **stored float16** vector is bit-identical
     to the reference in every case. Also record the float32 max abs difference,
     as information.
  5. Record `f16_max_abs` and whether any value exceeds 65504 (the 7B and
     ViT-g rows the spec asks for).
- [ ] **Step 2: Write the sbatch script and the submitter.** The submitter adds a
  detached worktree at the SHA under test
  (`/bigdata/exfab/anguy344/gate-worktrees/c6-<sha9>`), submits one job per GPU
  type and an `afterany` cleanup, and checks every captured id against
  `^[0-9]+$`. Per job: 8 CPU, 64 GB, 1:30:00, `HF_HUB_OFFLINE=1` after the
  weights are pre-staged, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, the foundation
  venv, `PYTHONPATH=<tree>/src:<tree>`.
  - Ada: `exfab`, `--account=exfab`, `--gres=gpu:1`.
  - A100 and Blackwell: `short_gpu --constraint=gpu_latest`, pinned by GRES type.
    First list the type names with `sinfo -p short_gpu -o '%N %G %f'`; do not
    guess them.
  - Pick each job's partition by measured predicted start where both can serve
    it (README conventions). Never `preempt`.
- [ ] **Step 3: Pre-stage the weights** in a CPU job with network
  (`intel`, 2 CPU, 16 GB, 1 h) into the HF cache the GPU jobs read.
- [ ] **Step 4: Submit, and verify each job started** (`scontrol show job <id> | grep -E 'StartTime|Reason'`).
- [ ] **Step 5: Write the report.** One table per GPU (model, dtype, m,
  invariant, f32 max diff, f16 max abs) and a cross-GPU table (median and max
  cosine distance per model, Ada vs A100 vs Blackwell).
  - **Every `invariant` true:** P2 passes its gate; record it in the report.
  - **Any `invariant` false: stop (stop point 3).** Do not start P4. Report the
    case, model and GPU to the user, with the spec's §5 fallback (an
    m = 1-per-crop mode) as the option to decide on.
  - **Any `f16_overflow`:** report it. The run still fails that image as
    non-finite (spec §4.8), so it is a docs finding, not a stop.
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
