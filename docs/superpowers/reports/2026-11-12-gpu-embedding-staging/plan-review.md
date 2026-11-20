# #282 plan review

The reviewer was a fresh `plan-reviewer` (Opus). It read the plan at
`1604d2ff9` and the #281 code at `26d7f86a6`. It returned three messages, and
the orchestrator wrote them here.

**Verdict: needs revision.** Most anchors hold. The store transaction (P3) and
the controller shape (P5) are sound. Four mechanisms are wrong as written.

## Critical

- **C1. The "gated" recovery shortcuts re-run Stages 2 and 3 on resume.**
  - Local: `_terminal_output_exists` also gates Stage 2
    (`_cli_staged_strategy.py:250-259`). An image waiting on Stage 4 has
    consumed its token and raw, so it is re-detected and re-measured, and its
    signals go stale.
  - SLURM: the `run_stage3_step` fall-through hits `emit_missing_prereq` and
    then `SystemExit(1)` (`_cli_staged_slurm_worker.py:539-550`).
  - No-detector SLURM continuation: the planner runs only when a `GpuDetector`
    is present (`phenotypicCLI.py:3792-3813`). `run_single_pass_step` has no
    skip, so it rewrites the root.
  - Fix: under deferral, a Stage-3 marker plus a matching store plus a table
    means Stages 1–3 are done (return early, publish nothing). Add an explicit
    no-detector continuation. Any core re-run deletes that image's Stage-4
    signals.
- **C2. The local completion and exit predicate counts Stage-3 markers, not
  records** (`phenotypicCLI.py:4327-4360`, `_stage3_less_images`). A Stage-4
  failure would exit 1, and a pending image would be marked complete.
  - Fix: base the predicate on records (`_all_accepted_images_succeeded`).
    Locally, every image Stage 4 touches ends published or journalled, or the
    stage raises.
- **C3. Measure-mode failure atomicity.** A deferred measure core commits a
  store whose entry drops its embeddings. A Stage-4 failure then leaves no
  embeddings, an invalid record and exit 5. **User decision U8:** carry the
  old embeddings forward and journal the failure as `MEASURE_FAILURE_STAGE`
  (exit 3).
- **C4. Rounds cannot make progress with more than one backbone under a
  walltime.** Backbone-major passes commit nothing until the last backbone
  finishes, and "complete signals or nothing" re-embeds every key.
  - Fix: reuse signals per key (prepare only the keys without a valid signal at
    the live root), and count written signals as progress, or document the
    walltime needed.
- **C5. Stale signals loop.** No signal root is compared with the live root,
  and the plan's terminal rule on a second occurrence contradicts spec §4.8 and
  the `StoreRootChangedError` docstring (`_measurement_tables.py:1247-1256`).
  - Fix: complete means signal root equals live root; on a mismatch, delete the
    signals and re-embed. Drop the terminal rule.

## Important

- **I1. `masked_dense` tile normalisation differs from inline.** Inline uses the
  HF processor (`_dino_support.py:342-344`). Measured on one tile: 26,190 of
  32,928 values differ, max 4.8e-7.
  - Fix: route tiles through the processor, or record the drift. Add
    `objects_dense` and `crop_mean` to the real-backbone golden, and a
    `masked_dense` row to P2 T4.
- **I2. The BatchEncoder holds back finished images and accumulates memory.**
  It drains in submission order and partial queues block. A plate's 7B float32
  grid is about 1.54 GB.
  - Fix: drain out of order, flush on a bound, and model the memory in P0.
- **I3. DataLoader hazards.**
  - Persistent workers keep stale per-backbone state.
  - Workers fork after CUDA has initialised and after threads exist; use
    forkserver or spawn.
  - Collation turns ndarrays into tensors; use an identity `collate_fn`.
- **I4. `CUBLAS_WORKSPACE_CONFIG` is set only in the SLURM prelude,** so local
  `--embed-stage gpu` always fails. Set it at CLI start, or run local Stage 4 in
  a subprocess.
- **I5. Flaky or non-crashing tests.**
  - The commit-pool call order is compared flat across threads.
  - `crash_once_after` raises inside the executor, so nothing crashes.
  - The fence test omits `commit_guard`.
- **I6. Wrong numbers.** The 40 GiB / 7B fp32 case fits at m = 64. Choose a pair
  that genuinely does not fit.
- **I7. Orphan cleanup is unreachable** (done images never reach Stage 4).
  Sweep the whole inventory instead.
- **I8. Missed event-aggregation callers:** `_cli_state_management.py:428`,
  `_cli_checkpoint_handler.py:544` and `sdk_/generate_report.py:83`. The test
  file is named `test_cli_update_state_stage.py`.
- **I9. The SLURM measure finalizer is not wired** (`measure_failed`, and the
  run proof without the forward inventory).
- **I10. API errors.** `owned_table_run_id`'s `pipeline_sha256` is
  keyword-only. Forward runs should use `journal_table_run(store)`.
- **I11. The stale "recomputation" can never fire,** because its digests come
  from the payload. Compute them from the store. `array_sha256` is a Mapping.
- **I12. The zero-region shortcut in `encode()` triggers a second model load**
  through `_dim()`.
- **I13. Test determinism and tests that cannot fail.**
  - `device="auto"` runs the stub on CUDA on GPU nodes.
  - The float16 golden assertion is a one-value Literal.
  - The register case uses `crop_cls`, which never reads registers.
  - The stub has no cross-row interaction, so the batch-isolation test cannot
    fail.
- **I14. The stub has no parameters,** so it can't honour `dtype`. The
  measurer's `load_dino_backbone` delegate is missing from P2 T1.
- **I15. `SLURM_CPUS_PER_TASK` raises `KeyError` when unset.** Use the
  `_cli_utils` helper.
- **I16. Local `--embed-stage gpu` with no CUDA fails only after Stages 1–3.**
  Refuse or warn at startup.

## Minor

- **P2 T4 additions:** m against m/2, a separate-process repeat, failing on any
  determinism warning, and a `masked_dense` row.
- **OOM tests:** add an OOM-once-then-succeed case.
- **The amortisation constant:** GPU-only time gives about 439 images, not 198,
  and the true value depends on the pipeline.
- **Loader memory:** `LOADER_PER_WORKER` ignores the extra layers and uint16
  plates.
- **Republish byte-identity:** the "byte-identical to #281" promise cannot hold,
  because entries gain `compute`.
- **R3's remedy** names `--gpu-slurm` for no-detector runs.
- **tracked_state.md** rows are needed for the signals, the worklist and the
  counters.
- **A walltime kill between promote's two renames** can lose a store. This is
  pre-existing, but Stage 4 crosses that window more often.
- **The GUI run console** resolves no-detector runs inline.

## Open questions

| # | Verdict |
|---|---|
| O1 Separate `EmbeddingDinoSize` | agree |
| O2 Tokens `vitg14` / `vith16plus` / `vit7b16` | agree; pin the existing tokens |
| O3 Refuse a non-DINO embedder | agree |
| O4 `model.device` plus `compute.device` | acceptable, if one producer writes both and a test asserts they are equal |
| O5 `gpu` with no embedder resolves inline | agree |
| O6 Amortisation warning | agree; fix the constant, and word it as efficiency |
| O7 SLURM measure via the controller | **user: approved (U10)**; needs I9 and C3 |
| O8 Journal the leftovers, exit 3 | disagree. **User: exit 5, unrecorded (U7)** |
| O9 Resolve the revision at submission only for Stage-4 runs | agree |

## Spec changes the review prompted (2026-11-19, user decisions)

These are in the spec as U5 (amended), U7, U8, U9 (superseding H9), U10, and
goal 3 restated per GPU model and physical batch (S1):

- **U7:** a stall ends unfinished, exit 5.
- **U8:** a measure failure keeps the old embeddings.
- **U9:** an explicit CPU pin wins and the run resolves inline.
- **U10:** SLURM measure runs on the controller.
- **U5:** a continuation reuses the recorded revision.

The "one writer per artifact" wording in `_cli/CLAUDE.md` is to be restated
in P7 as "nothing is rewritten after the record that certifies it".

## Cuts

- `worklist_partition.py`: a unit test is enough.
- `_journal_stage4_no_progress`: gone under U7.
- Republish: one carrier, the manifest flag. The controller config and
  `--republish` read it.
- A 1-thread commit pool.
