# P0 · Logic validation and the inline golden

**Spec:** §4.4 (batch size), §4.3 (worklist), §4.7 (`PF-EMBED-SHARD-AMORTIZE`,
`PF-EMBED-RESOURCES`), §5 (byte identity of the inline path), §7

---

**Goal:** Two things exist before any code changes. The numeric rules Stage 4
rests on are re-derived from scratch in standalone scripts. Today's inline CPU
bytes are pinned by a golden that P1 must keep green.

**Depends on:** #281 merged into `feat/gpu-embedding-staging`. T2 must be
captured on that merge commit, before any P1 commit.

### Task 1: Logic-validation scripts (Lane A)

**Files:**
- Create: `docs/superpowers/logic_validation_scripts/2026-11-12-gpu-embedding-staging/physical_batch_size.py`
- Create: `docs/superpowers/logic_validation_scripts/2026-11-12-gpu-embedding-staging/shard_amortization.py`
- Create: `docs/superpowers/logic_validation_scripts/2026-11-12-gpu-embedding-staging/worklist_partition.py`

Stdlib and numpy only. **Never import `phenotypic`.** Each exits non-zero on a
failed check and prints one line per check.

**Interfaces:**
- Produces, as module constants that P2 T2, P5 T1 and P5 T3 copy verbatim into
  `src/` (each `src/` constant's docstring names its script):
  - `MEMORY_FRACTION = 0.8`, `RESERVE_BYTES = 1 << 30` (spec §4.4)
  - `ACTIVATION_FACTOR = 12`: bytes per sample are
    `dtype_bytes · (ACTIVATION_FACTOR · T · hidden + heads · T²)`, with
    `T = (crop_px // patch)² + 1 + registers`. One encoder layer's intermediates
    (QKV, attention output, the 4× MLP, residual and norm) plus a materialised
    attention matrix. Conservative: under SDPA the T² term is not allocated.
  - `COLD_LOAD_S = 750.0`, `WARM_GPU_S_PER_IMAGE = {"base": 3.8, "large": 4.6}`,
    from spec §2 (0.72 + 0.99 + 2.1 s base; 1.05 + 1.48 + 2.1 s large)
  - `AMORTIZE_MIN_IMAGES_PER_SHARD(size) = ceil(COLD_LOAD_S / WARM_GPU_S_PER_IMAGE[size])`
    **(pending O6)**
  - `LOADER_BASE_BYTES = 6 << 30` (spec §2, peak RSS) and
    `LOADER_PER_WORKER(H, W) = 2 · H · W · 3 + (1 << 30)`: a worker holds one
    decoded uint8 plate and its crops, plus interpreter overhead.

- [ ] **Step 1: Write `physical_batch_size.py`.** It holds a model table and a
  GPU table and checks the spec's formula over their product.
  - Model table, one row each, `(params, hidden, heads, patch, registers)`:
    DINOv2 small, base, large, giant (`1.1e9, 1536, 24, 14, 0`); DINOv3 small,
    base, large, huge_plus (`8.4e8, 1280, 20, 16, 4`), 7b
    (`6.7e9, 4096, 32, 16, 4`). A comment says P2 T1 verifies each row against
    the hub's `config.json` and re-runs this script.
  - GPU table, total bytes: Ada 6000 48 GiB, A100 40 and 80 GiB, H100 80 GiB,
    Blackwell 6000 96 GiB.
  - Checks:
    1. `m` is a power of two, `1 ≤ m ≤ batch_size`, for every model, GPU, each
       `crop_px` in `{112, 224, 336, 448}` and each `compute_dtype`, except the
       pairs the script lists as **unsupported** (weights alone exceed the
       budget: 7B in float32 on a 40 GiB A100).
    2. `m` never reads free memory: the function signature takes total bytes
       only, and calling it twice with the same inputs gives the same value.
    3. Monotone: more GPU memory never gives a smaller `m`; a larger `crop_px`
       never gives a larger `m`.
    4. DINOv2-base at `crop_px=224`, float32, on Ada: `m == 64`
       (`batch_size` binds).
    5. DINOv3-7B at `crop_px=448`, bfloat16, on Ada: `m ≥ 16`, so the spec's
       "7B in bf16" recommendation is runnable on `exfab`.
    6. A planted wrong constant (`MEMORY_FRACTION = 1.2`) makes check 1 fail;
       the script runs that self-test last and reports it.
- [ ] **Step 2: Write `shard_amortization.py`.** Checks:
  1. `AMORTIZE_MIN_IMAGES_PER_SHARD("base") == 198` from the constants above.
  2. A shard at the threshold spends at most half its wall time loading.
  3. For spec §1's 33,000 images on 4 shards, no warning fires.
  4. `LOADER_BASE_BYTES + 7 · LOADER_PER_WORKER(4012, 6016)` is the `--mem`
     floor for an 8-CPU shard on spec §2's plates, and it is under the 48 GB the
     gpu-baseline jobs used (`gate-results/gpu-baseline/submit.sh`).
- [ ] **Step 3: Write `worklist_partition.py`.** It re-implements the
  partition rule P5 T1 builds: sort the retryable identities, then split into
  `n_shards` contiguous near-even parts (the rule of `partition_shards`,
  `_cli_staged_slurm.py:64`). Checks, over 1–40 items and 1–8 shards:
  1. No item lost or duplicated.
  2. Part sizes differ by at most 1.
  3. The same set in another order gives the same parts.
  4. After removing a random completed subset (a walltime), the leftover spreads
     over every shard: each of `k` shards gets `floor(n/k)` or `ceil(n/k)`.
  5. More shards than items: the extra parts are empty, and none is `None`.
- [ ] **Step 4: Run all three.**

  ```bash
  D=docs/superpowers/logic_validation_scripts/2026-11-12-gpu-embedding-staging
  for s in physical_batch_size shard_amortization worklist_partition; do uv run --no-project python $D/$s.py || echo "FAIL $s"; done
  ```

  Expected: every check prints `ok`, each script exits 0, no `FAIL` line.
- [ ] **Step 5: Commit.**

  ```bash
  git add docs/superpowers/logic_validation_scripts/2026-11-12-gpu-embedding-staging/
  git commit -m "docs(validate): Stage-4 batch size, amortisation and worklist rules (#282 P0)"
  ```

### Task 2: The inline golden

**Files:**
- Create: `tests/unit/measure/test_embedding_inline_golden.py` **[torch]**
- Create: `tests/unit/measure/golden/inline_embeddings_281.json`
- Create (helpers), in `tests/_tiny_run.py`: `inline_golden_cases() -> dict[str, ImagePipeline]`,
  the cases below, keyed by case name.

**Interfaces:**
- Consumes: `install_stub_backbone(monkeypatch, *, num_register_tokens=...)`
  (`tests/_tiny_run.py:1257`), `detected_plate`, `ImagePipeline.embed`
  (`_core/_pipeline_parts/_image_pipeline_core.py:1377`),
  `embedding_arrow_table` (`sdk_/_embedding_tables.py:160`).
- Produces: the golden file, a JSON object
  `{case: {key: {"vectors_sha256": str, "table_sha256": str, "descriptor": dict}}}`.
  - `vectors_sha256` is sha256 of the float16 vector bytes, C order.
  - `table_sha256` is sha256 of the Parquet bytes `_write_embedding_parquet`
    (`sdk_/_embedding_tables.py:224`) writes for the payload.
  - `descriptor` is the payload descriptor **minus** `model.device`,
    `model.device_name` and, once P2 adds them, `compute` and `compute_dtype`.
    Those are provenance or new fields, and the golden pins the bytes the
    refactor must not move.

Cases (each one `ImagePipeline(meas={...})` on `detected_plate`, stub backbone):

| Case | Embedder |
|---|---|
| `objects_cls` | `MeasureDinoEmbedding()` |
| `objects_mean_masked` | `MeasureDinoEmbedding(pooling="crop_mean", mask_background=True, crop_pad_frac=0.5)` |
| `objects_dense` | `MeasureDinoEmbedding(pooling="masked_dense", crop_px=112)` |
| `grid_cls` | `MeasureDinoEmbedding(regions="grid", grid=ManualGridFinder(row_edges=[0, 100, 200], col_edges=[0, 150, 300]))` |
| `gray_objects` | `MeasureDinoEmbedding(layer="gray", name="g")` |
| `registers` | `objects_cls` under `install_stub_backbone(monkeypatch, num_register_tokens=4)` |

- [ ] **Step 1: Write the test.**

```python
GOLDEN = Path(__file__).parent / "golden" / "inline_embeddings_281.json"

@pytest.mark.parametrize("case", sorted(inline_golden_cases()))
def test_inline_bytes_match_the_281_golden(case, monkeypatch, detected_plate):
    torch = pytest.importorskip("torch")
    install_stub_backbone(monkeypatch, num_register_tokens=4 if case == "registers" else 0)
    payloads = inline_golden_cases()[case].embed(detected_plate)
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))[case]
    assert sorted(payloads) == sorted(expected)
    for key, payload in payloads.items():
        assert golden_record(payload) == expected[key], key

def test_the_golden_was_captured_on_float16_cpu():
    data = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert all(r["descriptor"]["dtype"] == "float16" for c in data.values() for r in c.values())
```

  `golden_record(payload) -> dict` is module-local: it builds the three fields
  above. Regeneration is `PHENOTYPIC_REGEN_GOLDEN=1`, which writes the file and
  then fails the test with "golden regenerated; commit it", so a regeneration is
  never silently green.
- [ ] **Step 2: Capture the golden on the #281 merge commit.**

  ```bash
  PHENOTYPIC_REGEN_GOLDEN=1 eval $TPT tests/unit/measure/test_embedding_inline_golden.py
  ```

  Expected: 6 failures, each "golden regenerated; commit it". The file exists.
- [ ] **Step 3: Run it without the variable.**

  ```bash
  eval $TPT tests/unit/measure/test_embedding_inline_golden.py
  ```

  Expected: `7 passed`. In the shared env (`eval $PT …`) the six skip.
- [ ] **Step 4: Optional real-backbone row.** Add
  `test_real_small_backbone_matches_the_golden`, marked `slow`, skipped unless
  `facebook/dinov2-small` is in the HF cache (`hf_repo_is_cached`,
  `detect/nn/_helper/_checkpoint_manager.py`). It pins `objects_cls` with
  `size="small"` under the key `real_small`. Capture it in a Slurm job (`intel`,
  4 CPU, 16 GB, 30 min), and record the resolved revision in the commit message.
- [ ] **Step 5: Commit**, naming the merge SHA the golden was captured on.

  ```bash
  git add tests/unit/measure/test_embedding_inline_golden.py tests/unit/measure/golden/inline_embeddings_281.json tests/_tiny_run.py tests/conftest.py
  git commit -m "test(embed): inline CPU golden captured on <merge-sha> (#282 P0)"
  ```
