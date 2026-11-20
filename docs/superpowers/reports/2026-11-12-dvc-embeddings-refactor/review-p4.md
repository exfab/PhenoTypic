# P4 deep review: DINOv2 embeddings

Reviewer: fresh `implementation-test-reviewer` (Opus) over
`git diff 206ba673b..9c3848ae0 -- src tests`. The review came back as three
messages; the orchestrator wrote it here. The repro scripts and mutant plugins
are in `/bigdata/exfab/anguy344/gate-results/rev-p4/`.

Baseline, measured:
- torch venv, the six P4 test files: `102 passed, 0 skipped`;
- shared env, the import guards: `249 passed`;
- no torch or transformers import at module scope.

No Critical findings.

## Important

### I1. A corrupt embedding fails finalization after the master and mirror are republished

- **Where:** `_cli_finalize_run.py`. `aggregate_embedding_tables` runs after
  `finalize_post_master_outputs` and before `publish_run_summaries`. The raise
  is in `_cli_embedding_deliverables._publish_key` → `_unreadable`.
- **Repro** (`repro_corrupt_finalize.py`): a full run exits 0; corrupt
  plate_2's crops parquet; `--mode recompile` exits 1. After that:
  - rewritten: master and mirror;
  - not rewritten: `metrics.json`, `schema.json`, the embeddings manifest and
    `crops.parquet`, all still from the previous finalization.
- **Two keys:** key A's parquet is already replaced when key B fails, so the
  old manifest carries a stale sha256 for A.
- **Re-run:** a plain forward re-run then exits 0 ("All images already
  processed!") without re-finalizing.
- **Fix:** validate every non-stale entry (present, sha256, schema) before
  anything is written, or stage every key and promote all of them, manifest
  last. Either way a failure changes no deliverable.

### I2. Vectors from different models or poolings are concatenated into one deliverable

- **Where:** `_publish_key` collects distinct descriptors silently. Rows carry
  no descriptor attribution.
- **Repro** (`repro_mixed_descriptors.py`): two stores of one key, with
  different pooling and revision, go into one file of 1104 rows, and the
  manifest lists 2 descriptors.
- **Reachable:**
  - `revision=None`: each process resolves the hub's latest commit;
  - `device="auto"`: CUDA on some nodes, CPU on others;
  - a partly re-measured tree.

  The pipeline sha does not change in any of these.
- **Policy:** user question Q1.

### I3. A symlinked `results/` makes the embedding step raise

- **Where:** `_cli_embedding_deliverables._store_identity` uses
  `Path(store).resolve().relative_to(output_dir.resolve())`.
- **Repro** (`repro_symlinked_results.py`): `ValueError: … is not in the subpath`.
- **Same pattern elsewhere:** `read_embedding_table`'s
  `path.resolve().is_relative_to(store.resolve())` refuses a parquet that is a
  symlink out of the store. That is DVC `cache.type=symlink`, where
  `PF-DVC-LINKED-CACHE` is only a warning, and the refusal gives a misleading
  "missing" reason.
- **Fix:** compute paths from the unresolved forms, or use `os.path.samefile`.

## Minor

- **M1.** Already queued by the orchestrator at the C19 gate:
  - (a) declare the embedding deliverables in the aggregate proof's
    `required_outputs`;
  - (b) on the `shard_paths` branch, `aggregated_sources` is the full
    selection, so a store the projection excluded still contributes
    embeddings.
- **M2.** `_publish_key` treats any `stale` other than `True` as readable,
  while `read_embedding_table` refuses anything other than `False`. A missing or
  `null` `stale` therefore fails with the wrong remedy. Use one predicate,
  `stale is not False`, in both places.
- **M3.** No CLI path can bring a certified folder with `stale: true` to
  finalization: any write that changes the objmap rewrites the root and voids
  the record (`repro_stale_reachability.py`). Keep the branch as a defence. The
  README sentence promising stale entries "listed in manifest.json" describes
  something users will not see.
- **M4.** Byte stability holds only per device and BLAS path. The `batch_size`
  docstring ("never the vectors") overstates it. Document `device="cpu"` for
  reproducible runs.
- **M5.** Spec E3 says `torch.inference_mode()`. `masked_dense` goes through
  `extract_patch_features`, which uses `no_grad()`. Same outputs.
- **M6.** Spec E1 says "`objects` requires a non-empty objmap"; the
  implementation returns an empty table. See user question Q3.
- **M7.** `embed()` and `MeasureEmbedding.measure` wrap `MemoryError` and CUDA
  OOM into `RuntimeError` / `OperationFailedError`, which become terminal
  scientific failures. `measure()` does the same, so this is consistent, but an
  embedder is the step most likely to run out of memory. Re-raise those
  unwrapped.
- **M8.** `_remove_unpublished` removes `deliverables/embeddings/` whole,
  including any user file in it.
- **M9.** Each local worker loads its own backbone: about 16 × 350 MB at
  `--njobs 16` with ViT-B. Not documented.

## Test quality

- **T1.** Three mutants survive all 51 tests in
  test_measure_dino_embedding.py:
  - pooling always `crop_cls`;
  - `masked_dense` pooling the whole box instead of the mask;
  - grid cells padded and squared.
- **T2.** `test_cls_and_mean_pooling_read_the_same_tokens` is a tautology: the
  stub's CLS is the patch mean. Give the stub a CLS that is not the mean, and
  assert at the measurer that the poolings differ.
- **T3.** The stub has `num_register_tokens=0`, so no measurer-level test sees
  register tokens. Add a 4-register stub variant for `crop_mean` and
  `masked_dense`.
- **T4.** Every plate in `tiny_inputs` and `three_stores` is identical, so a
  swapped attribution is invisible. Make one plate differ, and assert that each
  store's deliverable rows equal `read_embedding_table(store)`.
- **T5.** `test_stale_entries_are_skipped_and_counted` drives the aggregator
  directly, on runs whose records no longer verify. It is a unit test of the
  function, not end-to-end D7 coverage.
- **T6.** `test_a_corrupt_embedding_fails_finalization_naming_the_store`
  asserts only the exit code and message. Assert that no deliverable changed.
- **T7.** The byte-stability test runs in one process with one cached stub.
- **T8.** Paths with no embedding test:
  - SLURM `shard_paths` (the fan-out and recompile shards);
  - a grid embedder through the CLI;
  - `--restart`;
  - continuation then re-finalization;
  - a forward run after a failed finalization.

## Checked and correct

- The writer: fixed-size float16 lists, explicit schema, reread before landing,
  file sha256 in the root.
- Binds on both sides.
- No stale recompute in measure (D7).
- The certified run, not the latest (D6a).
- Streaming one row group per store, in sorted order, with no wall-clock field.
- Aggregation only in the finalizer task, with no sidecar.
- Every per-image writer embeds; measure keeps embeddings (OQ5).
- The crop geometry.
- Revision provenance; safetensors only; the DINOv3 refusal before loading.
- The D9 split.

## Open questions for the user

1. **Q1 (I2).** A key whose certified folders carry more than one descriptor:
   (a) fail finalization, naming the stores per descriptor; (b) warn, and add a
   per-row `descriptor_index` into the manifest's `descriptors`; (c) warn only.
   The reviewer recommends (b). A related question: should `revision=None` draw
   a preflight warning?
2. **Q2 (I1).** The remedy for a corrupt embedding: (a) a whole-tree
   `--mode measure`, re-embedding every image, plus clearer waiter text; or (b)
   a scoped remedy that re-embeds only the named stores.
3. **Q3 (M6).** Should spec E1 be amended to say an empty objmap gives an empty
   table?

## Orchestrator triage

Fixed in the review-fix cluster (C19r) within the spec:

- **I1:** validate everything before publishing anything. Atomic all-or-nothing,
  manifest last.
- **I3:** unresolved paths in both places.
- **M1a, M1b.**
- **M2:** one predicate.
- **M3:** README wording only; the branch stays.
- **M4:** docstrings, plus a `device="cpu"` note.
- **M5:** `extract_patch_features` under `inference_mode`, if the detectors'
  outputs are unchanged; otherwise record it.
- **M7:** re-raise `MemoryError` and `torch.OutOfMemoryError` unwrapped from
  `embed()` and the embedder's `measure`.
- **M8:** delete only what finalization wrote, then prune empty directories.
- **M9:** document the per-worker backbone memory.
- **T1–T7:** applied as written. T1's three mutants must die.
- **T8:** tests for a grid embedder through the CLI, `--restart`, continuation
  then re-finalization, and the I1 re-run. SLURM `shard_paths` is covered at
  unit level through the M1b fix.

## User decisions (2026-11-18)

- **Q1: warn, and tag rows.** A key whose certified folders carry more than one
  descriptor publishes with a warning naming the descriptors and their stores.
  Each row carries a `descriptor_index` into the manifest's `descriptors` list.
  A run step's preflight also warns when a `MeasureDinoEmbedding` leaves
  `revision` unpinned.
- **Q2: a whole-tree re-measure is the remedy.** No scoped option. The
  finalization error and the `phenotypic-wait` exit-5 text point at the
  finalizer log and name the store.
- **Q3: amend the spec.** E1 now reads "An empty objmap gives an empty table,
  not an error".
- **P3 review OQ1 (ngio):** the user approved running the check.
