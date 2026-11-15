# C17 gate: P4 Tasks 1 and 2, `MeasureEmbedding` and embedding tables

Reported by the C17 implementer; written here by the orchestrator. Logs:
`/bigdata/exfab/anguy344/gate-results/c17/`.

## Commits

- **`62b81c535` (T1)** adds `abc_/_measure_embedding.py` (`MeasureEmbedding`,
  exported from `phenotypic.abc_`) and `sdk_/_embedding_tables.py`
  (`EmbeddingPayload`).
  - `ImagePipeline.measure()` skips embedders and `embed()` runs them.
  - A `meas` validator refuses two embedders that share one payload key, on
    construction and on `set_meas`.
  - `Image._store_series_array(name)` is now the one definition of a series'
    stored form, and the store writer uses it.
- **`86bee685d` (T1)** adds `GridCellOutsideImageError` in the ABC module,
  worded to match T3's planned test.
- **`5da466d77` (T2)** adds `write_embedding_tables`, `read_embedding_table`
  and `EmbeddingTableUnreadableError`.
  - `write_run_entry` writes payloads, and `save2zarr`, `_save_store`,
    `_write_store_part`, `_write_table_runs` and `replace_image_tables` take
    `embeddings=`.
  - The names are exported lazily from `phenotypic.sdk_`, so pyarrow stays
    unimported.

## Decisions (implementer's calls, accepted)

- **`embeddings=None` means "this write has no embeddings"** (review OQ5).
  Under D6 a writer replaces its own folder and no row of the previous write
  survives, so a same-pipeline rewrite without embeddings drops that run's
  embeddings. Measure mode must pass `pipeline.embed(image)` (C19).
  - The P3 carry test is renamed
    `test_a_same_pipeline_replace_image_tables_without_embeddings_drops_that_runs_embeddings`,
    and no "until P4" wording remains.
  - A new test checks that another pipeline's replace with embeddings leaves
    the first folder byte-identical.
  - Per D7, `replace_image_tables` does not recompute `stale`: new entries are
    written `stale: false`, bound to the image the caller embedded.
- **Binds come from the payload.** The plan's `write_embedding_tables`
  signature has no objmap path, so `EmbeddingPayload.descriptor` carries the D3
  producer fields and `binds` from `MeasureEmbedding.payload(image)`. The
  writer adds `stale: false` and `files`, then recomputes `stale` against what
  it writes, so a payload bound to another objmap is written stale (tested).
  `grid_sha256` is the sha256 of the edges JSON.
- **The ABC declares `grid`** next to `regions`; it is required iff
  `regions == "grid"`.
- **Two refusals and an edge case:**
  - `measure(include_meta=True)` on an embedder raises `ValueError`, because an
    embedding table carries no metadata (E5).
  - An image with zero objects gives an empty crop table, not an error. E1's
    "requires a non-empty objmap" is read as the `requires_objects` need, so a
    colony-less plate does not fail.
- **Reading back.** `read_embedding_table(store, key, *, run_id=None)` reads the
  highest-numbered run declaring the key, or the named run. An undeclared key
  raises `KeyError`. A stale, missing or hash-mismatched entry raises
  `EmbeddingTableUnreadableError`.
- **A save with embeddings and no measurement table owns its folder.** It
  writes a run entry without `measurements`, the grid-only process case
  (tested).

## Verification

- **Seen red first:** each new file failed at collection with an ImportError
  for the missing name; the plan's four T2 tests and three T1 tests were among
  them.
- **Focused runs:**
  - The T1 files: `18 passed`.
  - The T2 files plus the carry rules: `24 passed`.
  - The import guards: `249 passed`.
- **Slurm runs:**
  - `32097004` (tests/unit/core and tests/unit/abc_): `1316 passed`.
  - `32097186` (the same plus the import guards, at `5da466d77`): `1565 passed`.
  - `32097187` (all of tests/unit/sdk_): `1901 passed, 7 skipped, 5 xfailed`.
    The xfails were already there.
- **The orchestrator's independent rerun** of the four C17 test files at
  `5da466d77`: `42 passed`.
- ruff: clean. mypy: no new errors. Remaining errors are the pre-existing ones
  in `_image_io_handler.py` and `_measurement_tables.py`, and the 3 that
  `_store_series_array` introduced were fixed.
- **Not run here:** tests/unit/cli, tests/integration and the GUI tests. The CLI
  surface is the P4 phase gate at C19.

## Handed to C18

- Import `GridCellOutsideImageError` from the ABC module.
- Pydantic keeps a redeclared parent field in the parent's position, so T3's
  field order needs the repo's reorder pattern.
