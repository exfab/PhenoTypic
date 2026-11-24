# #282 P1 + P2 deep review (implementation and tests)

**Reviewer:** implementation-test-reviewer (Opus), analysis only.
**Tree:** worktree `gpu-embedding-staging`, branch `feat/gpu-embedding-staging`.
**Range read:** `git diff ebc273386~6..HEAD -- src tests docs/superpowers/plans/2026-11-12-gpu-embedding-staging/*.py`.
I also read `git show 14784930e`.

> **Note on the range.** `ebc273386~6` resolves to `7e7f8b45f`, and
> `14784930e` (the ABC refactor) is *older* than that commit. So the diff
> command in the brief leaves out the whole `abc_/_measure_embedding.py`
> step refactor. I reviewed it separately, from `git show 14784930e`. A
> re-review that uses the same command misses it too: `14784930e^..HEAD`
> is the range that covers P1 and P2.

**Settled, not relitigated:**
- the golden's require-and-skip design and its cosine floor;
- option (a) for C5 (byte equality only for the same pass shape);
- float32 recommended, with bf16 as an opt-in;
- DINOv2 only (U11).

## Summary

- **No critical defect.** The hard-look items mostly hold:
  - **Default digest.** The default digest is stable.
  - **Digest per config.** I computed `_params_sha256()` at HEAD for all six
    golden configs, through the foundation venv. All 6 equal the
    `producer.params_sha256` that the golden JSON recorded on #281.
  - **Device fields.** `compute_block()` is the only call site of
    `backbone_device_name` and `get_device_capability` in `src/` (grep).
  - **Physical batch rule.** `physical_batch_size` matches
    `physical_batch_size.py` on every combination that script checks; the
    test asserts this rather than sampling it.
  - **Determinism and CUBLAS.** The settings and the CUBLAS check do what
    they say. On torch 2.11 in the foundation venv, setting the four flags
    raises no Python warning (I measured it in a scratch run).
- **The T4 job is not vacuous.**
  - Every case uses a fresh encoder with a different batch composition.
  - Each occurrence is compared bit for bit with the reference
    (`f32_bit_mismatch_occurrences`).
  - `half_m` gives nonzero distances on Blackwell float32 (4.9e-9 and
    6.3e-9) and on every bf16 row. That shows the instrument detects a change
    of pass shape, so it is not comparing a run with itself.
  - Every region in the committed run is non-empty: 66 colonies and 96 wells
    per image, 486 vectors per row.
- **6 Important findings.** Each is a real failure or false-provenance
  scenario. Most bite in P4, but each starts in a P2 interface:
  - mixed-`m` provenance after an OOM halving;
  - one `m` per encoder, against a rule that depends on `crop_px`;
  - no single producer for `m`'s inputs;
  - an inline host-memory regression;
  - non-default digests pinned only on Broadwell;
  - a non-OOM forward error orphans other images' rows.
- **17 Minor findings.**

Confidence in correctness for the paths the tests exercise is high. The open
risk is in the contracts P4 will rely on: per-entry `physical_batch`, a
single `m`, and encoder state after an error.

---

## Critical

None.

---

## Important

### I1. An OOM halving makes images of mixed physical batch, and nothing records it

**Where:**
- `_batch_encoder.py:413-423` (`halve`);
- `:554-584` (`_forward` re-queues and halves);
- `:276-281` (`_Ticket` has no record of the pass shape);
- P4 plan, the line that records `compute_block(..., physical_batch=m)`.

**Scenario.**
1. A shard starts at `m=64`.
2. Image A's first 64 crops run at 64.
3. The next pass, which holds A's remaining rows plus B's, runs out of memory.
4. `m` becomes 32, and A's remaining rows run at 32.

A's stored vectors are now a mix. No single-`m` rerun reproduces them: T4
measured up to 4.3e-4 (1 − cos) between `m` and `m/2` for bf16, and up to
6.3e-9 for Blackwell float32. Spec §4.4 says "on OOM, m halves once for the
shard, **recorded**". The only record is the encoder's current `m`, so P4
will write `physical_batch=32` (or 64) for A. S1 promises invariance within
one physical batch, and A's entry then states a pass shape that it does not
have.

**Fix.**
- Give `_Ticket` a `physical_batches: set[int]` and add `self._m` to it in
  `_route` for each row routed.
- Expose it through `drain`, for example as a third tuple element or a
  `ticket_physical_batches()` read before the pop.
- In P4, record `physical_batch` as the one value when the set has one
  member. Otherwise record the list, or add `"halved_mid_image": true` to the
  `compute` block. The block is provenance and is excluded from grouping, so
  this costs no digest.
- Add a test:
  1. use `_raising_oom(None, on_call=2)` with `m=4`;
  2. submit 6 rows of ticket "a";
  3. assert that "a" reports `{4, 2}`.

### I2. One `m` per encoder, but the rule depends on `crop_px` (and `batch_size`)

**Where:**
- `_batch_encoder.py:363-386` (one `physical_batch` for every queue
  geometry);
- `physical_batch_size(*, batch_size, ..., per_sample)` (`:121`);
- P4 `Stage4Outcome.physical_batch: dict[BackboneKey, int]`;
- `BackboneKey` excludes `crop_px` (`test_the_backbone_key_of_an_embedder`
  asserts this).

**Scenario.** Two embedders share a `BackboneKey`, as H10 intends, but differ
in `crop_px`:
- objects at `crop_px=448`;
- a grid at `crop_px=224`.

They get one load and one `BatchEncoder`, so they get one `m`.

- **If `m` comes from the 224 embedder:** the 448 queue's per-sample cost is
  about 4× higher. It can run out of memory, halve once, and run out again,
  which raises `Stage4OutOfMemory`.
- **The next round recomputes the same `m`.** It is a pure function of the
  same inputs, so every round fails the same way.
- **The run then stalls.** U7 ends it unfinished with exit 5, and every plain
  re-run repeats this forever.
- **If `m` comes from the 448 embedder instead:** the 224 queue just
  under-fills, which is safe.
- **The same applies to `batch_size`.** Two embedders with different caps get
  one encoder.

With the default `batch_size=64` cap this does not occur on today's 48 GiB+
cards. I computed giant float32 at 448 px on 48 GiB: the uncapped `m` is 128,
capped to 64. But the rule exists precisely for when the cap does not bind.

**Fix.** Choose one:
- make `m` per queue geometry: `BatchEncoder(physical_batch=Mapping[(h, w), int])`
  or a callable, with halving per geometry;
- or have P4 key the encoder (and `Stage4Outcome.physical_batch`) by
  `(BackboneKey, crop_px)`, still sharing one model load;
- or, minimally, compute `m = min` over the backbone's embedders of
  `physical_batch_size(...)` at each embedder's `crop_px` and `batch_size`.
  State in the `BatchEncoder` docstring that the caller must pass the minimum.

Pin it with a test: two embedders sharing a key with different `crop_px`.

### I3. `m`'s inputs have no single producer; T4's measured `m` and P4's `m` can drift

**Where:**
- `batch_invariance.py:347-369`, which computes `weight_bytes` from the
  loaded parameters, `per_sample` from `model.config`, and the total from
  `torch.cuda.get_device_properties(0).total_memory`;
- the P4 plan, which says only "compute `m`".

The batch-invariance report ends with "P4's worker should compute m the same
way". `physical_batch_size` takes pre-digested numbers. Each caller derives
`weight_bytes`, `tokens`, `heads`, `dtype_bytes` and the total for itself,
and can do it differently:
- from the P0 table's `1.1e9` instead of the exact `1,136,480,768`;
- from `mem_get_info()[1]` instead of `get_device_properties`;
- with `config.patch_size` instead of `backbone_patch_size`.

S1's guarantee, measured by T4, holds only at the `m` that T4 used. A
production `m` one power of two off is measured only by `half_m`, not by
the invariance cases.

**Fix.**
- Add one function to `_batch_encoder.py`:
  `physical_batch_for(model, *, crop_px: int, batch_size: int, device_index: int = 0) -> int`.
  It is the body of `batch_invariance.py:physical_batch_for`.
- Have the script and P4 both call it. This follows the root CLAUDE.md rule:
  "One producer per derived value".
- Test it on the stub against a hand computation, with the device-total
  lookup injected.

### I4. The inline CPU path now holds every crop at once (host-memory regression)

**Where:** `_measure_dino_embedding.py:579-582`
(`crops = np.empty((len(boxes), crop_px, crop_px, 3))`), used by
`MeasureEmbedding.payload` and `_operate` (`abc_/_measure_embedding.py:556`,
`:869`).

The old loop built crops one batch at a time. Its comment ("at crop_px=448 a
1536-colony plate's crops would otherwise hold ~0.9 GB at once") named this
exact guard. P1 Step 3 says "The inline path must not regress", but it
reasons only about *device* memory. On the #281 inline CPU path, which is
every run today, the device memory *is* host memory.

**Scenario.** A local `--njobs N` run with an objects embedder at
`crop_px=448` on dense plates.
- Each worker now holds up to about 0.92 GB more per objects embedder:
  1536 × 448 × 448 × 3 bytes.
- That is on top of the measured peak RSS of about 6 GB (spec §2).
- The extra memory is live while the backbone runs, so the 0.9 GB is
  additive, not transient.
- The `--mem` users sized for #281 can then OOM-kill workers. A host
  `MemoryError` is re-raised unwrapped, so the image retries forever, never
  terminal.

**Fix.** Choose one:
- (a) let `BackboneInputs("crops")` carry a lazy row source (boxes plus a
  `crop(i)` callable). `_InlineDinoEncoder` then materialises one
  `batch_size` slice at a time, and Stage 4's loader calls a
  `materialise()` that stacks them;
- (b) give the inline path its own `prepare_regions(..., materialise=False)`
  variant that keeps the boxes and labels and crops per batch inside
  `_InlineDinoEncoder.run`.

Either way the bytes stay identical: same `_crop` calls, same order. Then
measure peak RSS before and after in one job (the slurm-job skill).

### I5. Digest stability for non-default embedders is pinned only on Broadwell

**Where:**
- `test_measure_dino_embedding_sizes.py:118`, which pins only
  `MeasureDinoEmbedding()`;
- `test_embedding_inline_golden.py:197-206`. The golden compares
  `descriptor`, including `producer.params_sha256`, for all six configs
  (named, gray, grid, `crop_mean`, dense, registers), but only inside
  `test_inline_bytes_match_the_281_golden`, which skips unless the CPU,
  torch build and thread counts match Broadwell.

The descriptor and the parameter digest do not depend on the CPU at all. A
future field added to `MeasureDinoEmbedding` without a
`_DIGEST_NEUTRAL_DEFAULTS` entry would move every non-default `#281` digest.
That is exactly the "#281 table reads stale" failure H7 guards against.
Every non-Broadwell gate (Cascade Lake, Milan, the shared env) would still
pass.

I verified that today the six HEAD digests equal the golden's (6/6). The
check ran in seconds without the capture environment, so it can be
always-on.

**Fix.** Add an always-on test, parametrised over `inline_golden_cases()`. It
asserts that `pipe.meas["emb"]._params_sha256()` equals the `producer.params_sha256`
of every record in `inline_embeddings_281.json[case]`. Optionally it also
compares the golden descriptor's non-provenance fields, minus `model.revision`,
which comes from the stub. It needs no torch. It covers the default too, so
`SHA_281_DEFAULT_EMBEDDER` then becomes redundant.

### I6. A non-OOM error in a forward pass orphans other images' rows

**Where:** `_batch_encoder.py:548` pops the batch, then `:571-572` re-raises
any non-OOM exception without re-queueing.

**Scenario.** A pass holds rows from tickets A, B and C, and the model raises
something that is not an OOM. Examples:
- a `RuntimeError` from a driver Xid;
- a `CUDA error: an illegal memory access`;
- `CUBLAS_STATUS_ALLOC_FAILED` (an allocation failure that torch does *not*
  raise as `OutOfMemoryError`);
- a dtype mismatch.

The rows are gone from the queue. Each ticket's `remaining` never reaches 0,
`held_bytes` keeps their bytes for ever, and `drain` never yields them. A
caller that catches the error and goes on, for example to fail "the image",
cannot tell which images the pass held. A, B and C hang until the shard
ends. Worse, a later `flush` reports nothing about them.

A related gap: `_is_device_oom` (`:284`) misses OOMs that surface as
`RuntimeError("CUDA error: out of memory")` or `CUBLAS_STATUS_ALLOC_FAILED`.
`abc_.is_out_of_memory` has the same blind spot, so the two at least agree.

**Fix.**
- On any exception, re-queue the batch, exactly as the OOM path does, before
  re-raising.
- Document that after a non-OOM error the encoder is poisoned. Set
  `self._failed = exc`, and make `submit`, `flush` and `drain` raise it, so
  P4 must fail the shard rather than half-continue.
- Add a test with a hook that raises `ValueError` on the second pass. Assert
  that `held_bytes` is unchanged, the rows are re-queued, and the next call
  raises.

---

## Minor

M1. **`test_dense_grids_flush_on_the_byte_bound` cannot catch an off-by-one
in the bound** (`test_batch_encoder.py:188`). With `>=` in place of `>` at
`_batch_encoder.py:458`, the flush fires after "b". "c" then stays held, and
every assertion still passes: drained ⊇ {a, b}, `held_bytes` = 1 grid ≤ 2,
`seen == [8]`. Fix:
- assert `list(enc.drain()) == []` and `seen == []` after the second
  submission;
- then assert the exact set {a, b, c} and `held_bytes == 0` after the third.

M2. **`test_model_and_compute_device_fields_agree` cannot detect a second
producer** (`test_measure_dino_embedding_sizes.py:266`). The patched
`backbone_device_name` returns a constant, so code that called it twice, once
for `model.device_name` and once for `compute`, would agree. Patch it with a
counter (`lambda d, it=itertools.count(): f"GPU-{next(it)}"`) and assert that
the two fields are equal and that it was called once per descriptor.

M3. **The `masked_dense` half of `test_the_batch_encoder_matches_the_inline_encoder`
relies on CPU GEMM luck** (`test_batch_encoder.py:240`). Inline runs one tile
per pass, and the encoder runs 64. The module docstring and the next test
state that MKL picks kernels by row count and that rows of a 40-row pass
differ from a 64-row pass. Here it passes on this CPU, but nothing makes it
portable. Run both sides at one pass shape, for example by pinning the
tile-per-pass with `physical_batch=1`, or accept a cosine bound as the next
test does.

M4. **`batch_invariance.py` has two vacuous-pass paths.** The committed run
hits neither.
- `:551` filters the repeat's rows to those the first run finished. If the
  first run errored on every row, `rows == []`, the verdict loop is empty and
  `ok` stays `True`, so it prints `VERDICT PASS` and exits 0.
- `compare_stored` (`:303`) skips empty regions without asserting that
  `total > 0`. A zero-colony detection would make the `objects` half
  vacuous.

Fix: fail when `rows` is empty in repeat mode. Make `invariant` require
`rows > 0` for every region.

M5. **The submitter reuses one worktree per SHA for smoke and full runs**
(`submit_batch_invariance.sh:21,34,104-109`). A smoke's `afterany`
finalizer runs `worktree remove --force`. A full submission at the same SHA
reuses the tree (`[ -d "$TREE" ] ||`), so its later jobs lose
`batch_invariance.py` and `src/` underneath them. The same happens when one
GPU type is resubmitted while another's finalizer is pending. That did not
happen here, since the smoke and full runs were at different SHAs. Fix:
suffix the tree with `$$` or a timestamp, or refuse when it already exists.

M6. **`torch.cuda.empty_cache()` inside the `except` block frees little**
(`_batch_encoder.py:575`). The exception's traceback still references the
model's forward frames, and `pixels` is still bound in `_forward`'s frame.
Move the cache release after the `except` exits, for example by returning a
sentinel and calling `empty_cache()` in `_run_queues` before the retry.

M7. **Duplicated definitions**, each a second producer:
- `_is_device_oom` (`_batch_encoder.py:284`) is `abc_.is_out_of_memory`
  minus `MemoryError`;
- `_COMPUTE_DTYPES` (`:258`) restates `ComputeDType`. Use
  `typing.get_args`;
- "a #281 entry was float32" lives twice:
  `abc_/_measure_embedding.py:55` (`_DIGEST_NEUTRAL_DEFAULTS`) and
  `_cli/_cli_embedding_deliverables.py:110` (`_ABSENT_DESCRIPTOR_DEFAULTS`).
  P4's "publish only" check (`compute_dtype` match) will need a third
  reading. Put one constant in `sdk_` (for example, `COMPUTE_DTYPE_DEFAULT`
  beside `ComputeDType`) and read it from all three.

M8. **The published manifest now carries `compute_dtype` for every
embedder kind** (`_cli_embedding_deliverables.py:330-332`). `setdefault`
applies to non-DINO embedders and to every #281 entry. So `manifest.json`
reports `compute_dtype: "float32"` for tables whose entries never recorded
it, including embedders with no torch backbone. Scope the default to entries
whose `model.family` starts with `dinov`, or document it as the
deliverables' reading.

M9. **`"cuda:N"` is treated as the CPU.**
- `compute_block` (`_dino_support.py:575`, `on_cuda = device == "cuda"`) and
  `backbone_device_name` give `capability=None`, `cuda=None` and
  `device_name=None` for `"cuda:0"`.
- `BatchEncoder` skips `empty_cache` for it.
- `resolve_backbone_device` only returns `"cuda"`, but `BatchEncoder` takes
  any `device: str` from P4.

Use `torch.device(device).type == "cuda"`, and pass the index to
`get_device_capability` and `get_device_name`.

M10. **The inline `physical_batch` records the cap, not the pass shape**
(`_measure_dino_embedding.py:427`). An image with N < `batch_size` regions
runs one pass of N, and the last pass of a larger image is shorter. C5
showed that CPU bytes depend on exactly this. Record
`min(batch_size, n_regions)`, or rename and document the field as the cap for
`stage="inline"`.

M11. **Inline `masked_dense` loads a second processor.**
`MeasureEmbedding.payload` and `_operate` call `prepare_regions(image)` with
no processor. That goes to `load_dino_processor` (`:614`), a separate
`from_pretrained` from `_cached_backbone`'s. With `revision=None` online,
that is one more hub resolution per process, and in principle a different
commit from the weights' own processor. On the inline path the backbone's
processor is already loaded, so pass it in, for example from an override of
`payload` or `_operate`, or let `inline_encoder` hand it to
`prepare_regions`.

M12. **`check_cublas_workspace` cannot tell a variable that was set too
late** (`_batch_encoder.py:181`). PyTorch reads `CUBLAS_WORKSPACE_CONFIG`
once, when it creates its first cuBLAS handle. A local run whose Stage 2
already used the GPU in the CLI process, followed by a `setdefault`, passes
the check without being deterministic. P4's `prepare_stage4_environment` is
planned before Stage 1, which is right. Make it refuse when
`"torch" in sys.modules and torch.cuda.is_initialized()` at the moment it
sets the variable.

M13. **`configure_determinism` changes process-global torch state and never
restores it.** Local Stage 4 runs in the CLI process, so everything after it
in that process runs with deterministic algorithms (`warn_only`) and cuDNN
benchmark off. Either document it, or return a restorer or context manager
(the test fixture at `test_determinism.py` already lists the seven values
to save).

M14. **Missing negative and branch tests for `BatchEncoder`:**
- the refusals in `_admit`: a non-uint8 crop, `(N,H,W,4)`, a `"crops"` input
  with `pooling="patches"`, and a `(4,h,w)` tile;
- the `(1, 3, h, w)` tile squeeze (`:516-517`);
- `_SyncRegionEncoder` refusing while another ticket is held;
- the state after `Stage4OutOfMemory` (`held_bytes`, re-queued rows);
- bf16 batched-versus-inline equality for `masked_dense` at one pass shape.
  This is the case where the cast order matters: tile float32 →
  `compute_dtype` inline, against float32 zeros ← tile → `compute_dtype`
  batched. Today it is equivalent only because the processor returns
  float32, and `_admit` does not enforce that. Add
  `tile.dtype == np.float32` to `_admit`.

M15. **`MAX_IN_FLIGHT_OUTPUT_BYTES` is sized for the deferred 7B and counts
outputs only** (`:57`). Its docstring sizes it from DINOv3-7B grids, which
U11 deferred. Queued input tiles are not counted: about 290 MB of float32
tiles for one 4012 × 6016 plate. Neither are completed-but-undrained
tickets. State both in the docstring, and that the caller must drain after
every `submit`.

M16. **`compute_dtype="float16"` is accepted but never measured.**
- None of T4's rows uses float16.
- ViT-g in fp16 is a well-known overflow risk.
- Overflow gives NaN tokens, then `payload_from`'s non-finite refusal, so
  *every* image gets a terminal per-image failure.

Since H7 is documentation only, say in the field's docstring that float16 is
unmeasured. Or have the T4 script gain a `giant_fp16` row.

M17. **Loose typing.**
- `backbone_key()` returns `Any` (`_measure_dino_embedding.py:371`); use
  `BackboneKey` under `TYPE_CHECKING`.
- `BackboneKey.size` and `compute_dtype` are `str`, where they could be
  `EmbeddingDinoSize` and `ComputeDType`.

---

## Hand-offs to P4 (not P2 defects, but the P2 contract must state them)

- **`_check_geometry` is skipped.** P4 loads through `load_dino_backbone`
  directly ("never `_cached_backbone`"), so `_check_geometry` does not run.
  `prepare_regions` cuts tiles with `_PATCH_SIZE[version]`, and
  `BatchEncoder` routes with `backbone_patch_size(model)`. Call
  `m._check_geometry(model)` once per backbone in Stage 4.
- **`configure_determinism()` needs an explicit device.** With no argument it
  infers CUDA from `torch.cuda.is_available()`, so a
  `PHENOTYPIC_STAGE4_DEVICE=cpu` test on a GPU node needs `device=` passed
  explicitly, or the CUBLAS check fires.
- **I1–I3 and I6 decide the shape of `Stage4Outcome.physical_batch` and the
  shard's error handling.** Settle them before P4 T3.

## What I verified, and how

| Claim | How |
|---|---|
| Six golden configs keep their #281 `params_sha256` at HEAD | ran `_params_sha256()` for each case of `inline_golden_cases()` and compared with `inline_embeddings_281.json`: 6/6 equal |
| `compute_block` is the only producer of device fields | grep of `src/phenotypic` for `backbone_device_name`, `get_device_capability` and `device_name`: one call site each, in `compute_block`; `entry_descriptor_for` copies |
| Rule equals P0 script | read `test_every_combination_the_script_checks_gives_its_m`, which asserts `m` or a refusal for every model × GPU × crop × dtype, and that both branches ran |
| TF32/deterministic setters raise no Python warning on torch 2.11 | scratch run in the foundation venv: no warnings recorded; `fp32_precision` reads `ieee` / `none` |
| T4 cases are not self-comparisons | read `batch_invariance.py` and the committed JSONs in `gate-results/c6-282/`: every case is a fresh encoder; occurrence counts (486 to 163,968) and nonzero `half_m` on Blackwell fp32 and all bf16 rows; regions 66/96 per image |
| Repeat runs are separate processes | job ids differ; Ada and A100 repeats ran on the same node as their first job (gpu12, gpu08), Blackwell on another (gpu16 → gpu17). "Separate process" holds; "separate node" is not claimed |
