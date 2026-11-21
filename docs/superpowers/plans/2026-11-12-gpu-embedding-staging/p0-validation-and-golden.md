# P0 · Logic validation and the inline golden

**Spec:** §4.4 (batch size), §4.7 (`PF-EMBED-SHARD-AMORTIZE`,
`PF-EMBED-RESOURCES`), §5 (byte identity of the inline path), §7

---

**Goal:** Two things exist before any code changes. The numeric rules Stage 4
rests on (the physical batch, the amortisation threshold, the loader and encoder
memory) are re-derived from scratch in standalone scripts. Today's inline CPU
bytes are pinned by a golden that P1 must keep green.

**Depends on:** #281 merged into `feat/gpu-embedding-staging`. T2 is captured on
that merge commit, before any P1 commit.

### Task 1: Logic-validation scripts (Lane A)

**Files:**
- Create: `docs/superpowers/logic_validation_scripts/2026-11-12-gpu-embedding-staging/physical_batch_size.py`
- Create: `docs/superpowers/logic_validation_scripts/2026-11-12-gpu-embedding-staging/shard_amortization.py`
- Create: `docs/superpowers/logic_validation_scripts/2026-11-12-gpu-embedding-staging/stage4_memory.py`

Stdlib and numpy only. **Never import `phenotypic`.** Each exits non-zero on a
failed check and prints one line per check. (The worklist partition rule is
pinned by a unit test in P5 T1, not by a script: plan review, Cuts.)

**Interfaces:**
- Produces, as module constants that P2 T2 and P5 T3 copy verbatim into `src/`
  (each `src/` constant's docstring names its script):
  - `MEMORY_FRACTION = 0.8`, `RESERVE_BYTES = 1 << 30` (spec §4.4)
  - `ACTIVATION_FACTOR = 12`: bytes per sample are
    `dtype_bytes · (ACTIVATION_FACTOR · T · hidden + heads · T²)`, with
    `T = (crop_px // patch)² + 1 + registers`. One encoder layer's intermediates
    (QKV, attention output, the 4× MLP, residual and norm) plus a materialised
    attention matrix. Conservative: under SDPA the T² term is not allocated.
  - `COLD_LOAD_S = 750.0` (spec §2, the upper end).
  - `WARM_GPU_S_PER_IMAGE = {"base": 1.71, "large": 2.53}`: the **GPU-bound**
    time of spec §2's two embedders, objects plus a 96-cell grid
    (0.72 + 0.99 s base; 1.05 + 1.48 s large). The agar cell's 2.1 s is a CPU
    resize, so it is left out (plan review, Minor).
  - `AMORTIZE_MIN_IMAGES_PER_SHARD(size) = ceil(COLD_LOAD_S / WARM_GPU_S_PER_IMAGE[size])`:
    below it, the cold load is more than half the shard's GPU time. This is an
    efficiency threshold for that pipeline shape, not a correctness one (O6,
    accepted with this wording).
  - `LOADER_BASE_BYTES = 6 << 30` (spec §2, peak RSS) and
    `LOADER_PER_WORKER(H, W, sample_bytes, crop_bytes) = H · W · (3 · sample_bytes + 12) + crop_bytes + (1 << 30)`.
    A worker holds one decoded plate: `rgb` at `sample_bytes` (1 for uint8, 2
    for uint16), and `gray`, `detect_mat` and `objmap` at 4 bytes each; plus
    the crops it cut and interpreter overhead (plan review, Minor).
  - `ENCODER_OUTPUT_BYTES(H, W, patch, hidden) = (H // patch) · (W // patch) · hidden · 4`:
    the largest per-image output the `BatchEncoder` holds (a `masked_dense`
    feature grid), and
    `MAX_IN_FLIGHT_OUTPUT_BYTES = 2 · ENCODER_OUTPUT_BYTES(4012, 6016, 16, 4096)`
    (about 3.1 GB): the bytes of output the encoder may hold for unfinished
    images before it flushes early. The bound is in **bytes, not images**
    (re-review item 4): a crop pipeline's outputs are kilobytes per image and
    never reach it, so it keeps full batches; only `masked_dense` grids do
    (plan review I2).
  - **Under U11 these bounds are conservative.** They are sized for DINOv3-7B
    (4096-d, the 1.54 GB plate grid), and the largest model this PR ships is
    DINOv2-giant at 1536-d, whose plate grid at patch 14 is about 0.75 GB. The
    formulas and the committed constants stay as they are: they remain safe
    upper bounds, and the deferred DINOv3 follow-up needs them.

- [ ] **Step 1: Write `physical_batch_size.py`.** It holds a model table and a
  GPU table and checks the spec's formula over their product.
  - Model table, `(params, hidden, heads, patch, registers)`: DINOv2 small,
    base, large, giant (`1.1e9, 1536, 24, 14, 0`); DINOv3 small, base, large,
    huge_plus (`8.4e8, 1280, 20, 16, 4`), 7b (`6.7e9, 4096, 32, 16, 4`). A
    comment says P2 T1 verifies each row against the hub's `config.json` and
    re-runs this script.
  - **Note (U11, added after the scripts were committed at `ccb017c58`):** the
    DINOv3 `huge_plus` and `7b` rows, and the checks that name them (3 and 3b),
    are informational only. This PR ships no such size, nothing verifies those
    rows against the hub, and the scripts stay as committed: the rows are
    harmless arithmetic, and the deferred follow-up will verify them.
  - GPU table, total bytes: Ada 6000 48 GiB, A100 40 and 80 GiB, H100 80 GiB,
    Blackwell 6000 96 GiB; plus a 16 GiB card (P100 class), which the cluster
    excludes with `gpu_latest` but which gives the script a real "does not fit".
  - Checks:
    1. `m` is a power of two with `1 ≤ m ≤ batch_size` for every model, cluster
       GPU, `crop_px` in `{112, 224, 336, 448}` and `compute_dtype` whose
       budget is positive. The script **computes** the set of pairs whose budget
       is not positive and prints it.
    2. On every cluster GPU (48 GiB and up), every model runs in `bfloat16` at
       every `crop_px`: the unsupported set has no such row.
    3. DINOv3-7B in `float32` on the 16 GiB card is unsupported (weights alone
       are about 25 GiB): the refusal case P2 T2's test uses. (The earlier
       "7B fp32 on a 40 GiB A100" case is wrong: it fits at `m = 64`, plan
       review I6.)
    4. `m` never reads free memory: the function takes total bytes only, and two
       calls with the same inputs agree.
       The upper-bound guard is the budget identity
       `m · per_sample ≤ SPEC_MEMORY_FRACTION · total − weights − RESERVE_BYTES`,
       with `SPEC_MEMORY_FRACTION = 0.8` its own literal, not the formula's
       `MEMORY_FRACTION`. (The earlier `m · per_sample + weights ≤ total` form is
       too weak: pow2floor slack plus the batch cap let a 1.2 plant fit under it.)
    5. Monotone: more GPU memory never gives a smaller `m`; a larger `crop_px`
       never gives a larger `m`.
    6. DINOv2-base at `crop_px=224`, float32, on Ada: `m == 64` (`batch_size`
       binds).
    7. A planted wrong constant (`MEMORY_FRACTION = 1.2`) makes check 4's
       budget-identity guard fail (12 combinations trip it); the script runs
       that self-test last and reports it.
- [ ] **Step 2: Write `shard_amortization.py`.** Checks:
  1. `AMORTIZE_MIN_IMAGES_PER_SHARD("base") == 439` and `("large") == 297`.
  2. At the threshold, `COLD_LOAD_S ≤ n · WARM_GPU_S_PER_IMAGE`: the load is at
     most half the shard's GPU time.
  3. Spec §1's 33,000 images on 4 shards fire no warning.
- [ ] **Step 3: Write `stage4_memory.py`.** Checks:
  1. `LOADER_BASE_BYTES + 7 · LOADER_PER_WORKER(4012, 6016, 1, crops)` for an
     8-CPU shard on spec §2's plates, with `crops` = 66 objects at 224 px, is
     under the 48 GB the gpu-baseline jobs used
     (`gate-results/gpu-baseline/submit.sh`); and the uint16 case
     (`sample_bytes=2`) is printed beside it.
  2. `ENCODER_OUTPUT_BYTES(4012, 6016, 16, 4096)` is 1.54 GB ± 1 %: a 7B
     `masked_dense` plate, the plan review's I2 figure.
  3. `LOADER_BASE_BYTES + MAX_IN_FLIGHT_OUTPUT_BYTES` is under 48 GB, so the
     bound holds the worst case.
  4. A crop pipeline never reaches the bound: 1,536 colonies (a dense 1536-well
     plate) × 4096-d × 4 bytes per image, times the loader's prefetch of
     `2 · (cpus − 1)` images on 8 CPUs, is under `MAX_IN_FLIGHT_OUTPUT_BYTES`.
- [ ] **Step 4: Run all three.**

  ```bash
  D=docs/superpowers/logic_validation_scripts/2026-11-12-gpu-embedding-staging
  for s in physical_batch_size shard_amortization stage4_memory; do uv run --no-project python $D/$s.py || echo "FAIL $s"; done
  ```

  Expected: every check prints `ok`, each script exits 0, no `FAIL` line.
- [ ] **Step 5: Commit.**

  ```bash
  git add docs/superpowers/logic_validation_scripts/2026-11-12-gpu-embedding-staging/
  git commit -m "docs(validate): Stage-4 batch size, amortisation and memory rules (#282 P0)"
  ```

### Task 2: The parametric stub and the inline golden

**Files:**
- Modify (helpers), `tests/_tiny_run.py:1157-1260`:
  - add `_build_param_stub_dino(num_register_tokens=0, *, leaky=False)`: the
    #281 stub (`_build_stub_dino`, `:1167`) plus a fixed-seed
    `torch.nn.Linear(STUB_HIDDEN_SIZE, STUB_HIDDEN_SIZE)` over the tokens and a
    softmax mixing **within** each sample's tokens. It has real parameters, so
    `.to(dtype)` means something (plan review I14), and patch and register
    tokens reach the CLS output. `leaky=True` adds the **batch** mean to every
    row: a model whose rows depend on their neighbours, for P2 T3's mutation
    check (plan review I13).
  - `install_stub_backbone(monkeypatch, *, num_register_tokens=0, parametric=False, leaky=False)`
    (`:1257`) chooses it; the #281 default (`parametric=False`) is unchanged, so
    #281's tests keep their stub.
  - fixture `param_stub_backbone(monkeypatch)`: `stub_backbone` (`:1234`) with
    `parametric=True`, the same `load_calls` namespace. #282 tests use it.
  - `inline_golden_cases() -> dict[str, ImagePipeline]`, the cases below.
- Create: `tests/unit/measure/test_embedding_inline_golden.py` **[torch]**
- Create: `tests/unit/measure/golden/inline_embeddings_281.json`

**Interfaces:**
- Consumes: `detected_plate`, `ImagePipeline.embed`
  (`_core/_pipeline_parts/_image_pipeline_core.py:1377`),
  `embedding_arrow_table` (`sdk_/_embedding_tables.py:160`),
  `_write_embedding_parquet` (`:224`).
- Produces: the golden file,
  `{case: {key: {"vectors_sha256": str, "table_sha256": str, "descriptor": dict}}}`.
  - `vectors_sha256`: sha256 of the float16 vector bytes, C order.
  - `table_sha256`: sha256 of the Parquet bytes `_write_embedding_parquet`
    writes for the payload.
  - `descriptor`: the payload descriptor **minus** `model.device`,
    `model.device_name`, `compute` and `compute_dtype`. The first two are
    provenance; the last two are added by P2 T1. The golden pins what the
    refactor must not move.

Cases, each `ImagePipeline(meas={...})` on `detected_plate` under the
**parametric** stub. Every embedder sets `device="cpu"`, so a GPU node cannot
move the bytes (plan review I13):

| Case | Embedder |
|---|---|
| `objects_cls` | `MeasureDinoEmbedding()` |
| `objects_mean_masked` | `MeasureDinoEmbedding(pooling="crop_mean", mask_background=True, crop_pad_frac=0.5)` |
| `objects_dense` | `MeasureDinoEmbedding(pooling="masked_dense", crop_px=112)` |
| `grid_cls` | `MeasureDinoEmbedding(regions="grid", grid=ManualGridFinder(row_edges=[0, 100, 200], col_edges=[0, 150, 300]))` |
| `gray_objects` | `MeasureDinoEmbedding(layer="gray", name="g")` |
| `registers_mean` | `MeasureDinoEmbedding(pooling="crop_mean")` with `num_register_tokens=4`, so the register drop is exercised (plan review I13) |

- [ ] **Step 1: Write the tests.**

```python
GOLDEN = Path(__file__).parent / "golden" / "inline_embeddings_281.json"

@pytest.mark.parametrize("case", sorted(inline_golden_cases()))
def test_inline_bytes_match_the_281_golden(case, monkeypatch, detected_plate):
    pytest.importorskip("torch")
    install_stub_backbone(monkeypatch, parametric=True,
                          num_register_tokens=4 if case == "registers_mean" else 0)
    payloads = inline_golden_cases()[case].embed(detected_plate)
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))[case]
    assert sorted(payloads) == sorted(expected)
    for key, payload in payloads.items():
        assert golden_record(payload) == expected[key], key

def test_the_registers_case_differs_from_a_registerless_run(monkeypatch, detected_plate):
    pytest.importorskip("torch")
    vectors = []
    for registers in (0, 4):
        install_stub_backbone(monkeypatch, parametric=True, num_register_tokens=registers)
        vectors.append(golden_record(next(iter(
            inline_golden_cases()["registers_mean"].embed(detected_plate).values())))["vectors_sha256"])
    assert vectors[0] != vectors[1]                    # the golden row can fail on a register bug
```

  `golden_record(payload) -> dict` is module-local and builds the three fields.
  Regeneration is `PHENOTYPIC_REGEN_GOLDEN=1`: the test writes the file and then
  fails with "golden regenerated; commit it", so a regeneration is never
  silently green.
- [ ] **Step 2: Capture the golden on the #281 merge commit.**

  ```bash
  PHENOTYPIC_REGEN_GOLDEN=1 eval $TPT tests/unit/measure/test_embedding_inline_golden.py
  ```

  Expected: 6 failures, each "golden regenerated; commit it". The file exists.
- [ ] **Step 3: Run it without the variable.**

  ```bash
  eval $TPT tests/unit/measure/test_embedding_inline_golden.py
  ```

  Expected: `7 passed`. In the shared env (`eval $PT …`) all seven skip.
- [ ] **Step 4: The real-backbone rows** (plan review I1). Add
  `test_real_small_backbone_matches_the_golden`, marked `slow`, skipped unless
  `facebook/dinov2-small` is in the HF cache (`hf_repo_is_cached`,
  `detect/nn/_helper/_checkpoint_manager.py`). It pins three cases with
  `size="small"`: `objects_cls`, `objects_mean_masked` (the `crop_mean` path)
  and `objects_dense` (the processor path P1 T2 reroutes), under the keys
  `real_small_<case>`. Capture them in a Slurm job (`intel`, 4 CPU, 16 GB,
  30 min) and put the resolved revision in the commit message.
- [ ] **Step 5: Commit**, naming the merge SHA the golden was captured on.

  ```bash
  git add tests/unit/measure/test_embedding_inline_golden.py tests/unit/measure/golden/inline_embeddings_281.json tests/_tiny_run.py tests/conftest.py
  git commit -m "test(embed): parametric DINO stub and the inline CPU golden, captured on <merge-sha> (#282 P0)"
  ```
