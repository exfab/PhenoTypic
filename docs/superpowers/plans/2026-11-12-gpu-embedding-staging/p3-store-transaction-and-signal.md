# P3 · `replace_run_embeddings` and the Stage-4 signal

**Spec:** §4.2, §4.8 "Root changed before commit", §5 "`replace_run_embeddings`"

---

**Goal:** Stage 4's two durable writes exist and are tested on their own: the
per-key signal under `.phenotypic/progress/`, and the store transaction that
rewrites one run's `embeddings` block, root last.

**Depends on:** P1 (payloads). Independent of P2, so it may run while P2 T4's
GPU jobs queue.

### Task 1: `replace_run_embeddings`

**Files:**
- Modify: `src/phenotypic/sdk_/_measurement_tables.py`. Add
  `replace_run_embeddings` after `replace_image_tables` (`:1460-1645`), built on
  `_rewrite_store_tables` (`:1301`).
- Modify: `src/phenotypic/sdk_/__init__.py`, exporting it beside
  `replace_image_tables`.
- Create (helpers), in `tests/_tiny_run.py`:
  - fixture `staged_store_with_tables(tmp_path, measured_image) -> tuple[Path, str]`:
    a promoted v2 store with a measurement table and a metadata table in run
    `run_id`, a second run folder from `pipeline_file_b`, an `upstream` entry,
    and no embeddings; yields `(store, run_id)`. Built with today's writers
    (`replace_image_tables`, `resave_with_pipeline`, `:672`), not by hand.
  - `objects_payload_for(store) -> dict[str, EmbeddingPayload]`: a `toy-v1/rgb/objects`
    payload bound to the store's own objmap (`objmap_digest`, `:245`).
  - `root_sha256(store) -> str`: sha256 of `zarr.json`.
- Test: `tests/unit/sdk_/test_replace_run_embeddings.py` (new, shared env)

**Interfaces:**
- Consumes: `_rewrite_store_tables(..., clear_table_run=..., expected_root_sha256=..., consolidate=...)`
  (`:1301-1457`); `write_embedding_tables` (`sdk_/_embedding_tables.py:246`);
  `refresh_embedding_staleness` (`:255`); `TableRun` (`sdk_/_table_runs.py:281`);
  `StoreRootChangedError` (`:1247`).
- Produces:
  `replace_run_embeddings(store_path: Path, run_id: str, payloads: Mapping[str, EmbeddingPayload], *, expected_root_sha256: str, objmap_sha256: str | None, array_sha256: Mapping[str, str], durable: bool | None = None, commit_guard: CommitGuard | None = None) -> TableRun`
  - `clear_table_run=f"{run_id}/embeddings"`; every other file is carried by
    `_rewrite_store_tables`'s hard-link copy.
  - The plan refuses, before any part exists, a store whose tables schema is
    not 2 or that has no run `run_id` (`ValueError` naming both).
  - The run's entry gets `embeddings = write_embedding_tables(part, run_id, payloads, objmap_sha256=..., array_sha256=...)`:
    a total description of this write, as in `replace_image_tables`.
    `stale` is recomputed for **that run only**, against the digests the caller
    passes. **The caller computes them by reading the live store**, never from
    the payloads' own `binds` (plan review I11): P4 T3's commit reads the
    store's level-0 objmap and each bound series' level-0 array
    (`objmap_digest`, `array_digest`, `:223-252`) after checking the root, so a
    payload computed from another image does read `stale: true`.
    `array_sha256` is a `Mapping[str, str]` keyed by series name, as
    `refresh_embedding_staleness` takes it (`:255`).
  - Every other run entry, `upstream`, `metadata_table`, figures and the
    journal are byte-equal in the new root.
  - The root is written last and re-consolidated when it was consolidated
    (automatic in `_rewrite_store_tables`, `:1393-1395`).
  - Raises `StoreRootChangedError` when the live root is not
    `expected_root_sha256`, at either check; the part is removed.

- [ ] **Step 1: Write the failing tests.**

```python
def test_only_the_runs_embeddings_change(staged_store_with_tables):
    store, run_id = staged_store_with_tables
    before = root_tables(store)
    replace_run_embeddings(store, run_id, objects_payload_for(store), **digests(store))
    after = root_tables(store)
    assert {k: v for k, v in after["runs"].items() if k != run_id} == \
           {k: v for k, v in before["runs"].items() if k != run_id}
    assert after["upstream"] == before["upstream"]
    assert set(after["runs"][run_id]["embeddings"]) == {"toy-v1/rgb/objects"}
    assert {k: v for k, v in after["runs"][run_id].items() if k != "embeddings"} == \
           {k: v for k, v in before["runs"][run_id].items() if k != "embeddings"}

def test_measurement_and_metadata_inodes_are_carried(staged_store_with_tables):
    store, run_id = staged_store_with_tables
    run = owned_run(store, run_id)
    inodes = {p: p.stat().st_ino for p in (run.measurements_path(), run.metadata_path())}
    replace_run_embeddings(store, run_id, objects_payload_for(store), **digests(store))
    assert {p: p.stat().st_ino for p in inodes} == inodes

def test_other_runs_files_are_byte_equal(staged_store_with_tables):
    store, run_id = staged_store_with_tables
    other = other_run_dir(store, run_id)
    digest = tree_digest(other)
    replace_run_embeddings(store, run_id, objects_payload_for(store), **digests(store))
    assert tree_digest(other) == digest

def test_stale_is_recomputed_against_the_passed_digests(staged_store_with_tables):
    store, run_id = staged_store_with_tables
    replace_run_embeddings(store, run_id, objects_payload_for(store),
                           expected_root_sha256=root_sha256(store),
                           objmap_sha256="0" * 64, array_sha256={})
    assert root_tables(store)["runs"][run_id]["embeddings"]["toy-v1/rgb/objects"]["stale"] is True

def test_a_changed_root_is_refused_and_leaves_no_part(staged_store_with_tables):
    store, run_id = staged_store_with_tables
    planned = root_sha256(store)
    touch_root(store)
    with pytest.raises(StoreRootChangedError):
        replace_run_embeddings(store, run_id, objects_payload_for(store),
                               expected_root_sha256=planned, **digests(store, root=False))
    assert not list(store.parent.glob(store.name + "*.part"))
    replace_run_embeddings(store, run_id, objects_payload_for(store), **digests(store))  # the retry

def test_a_consolidated_root_stays_consolidated(staged_store_with_tables):
    store, run_id = staged_store_with_tables
    consolidate_store(store)
    replace_run_embeddings(store, run_id, objects_payload_for(store), **digests(store))
    meta = json.loads((store / "zarr.json").read_text(encoding="utf-8"))["consolidated_metadata"]
    assert any(k.endswith(f"{run_id}/embeddings/toy-v1/rgb/objects") for k in meta["metadata"])

def test_an_independent_reader_sees_the_table(staged_store_with_tables):
    store, run_id = staged_store_with_tables
    replace_run_embeddings(store, run_id, objects_payload_for(store), **digests(store))
    group = zarr.open_group(str(store), mode="r")
    assert f"tables/{run_id}/embeddings/toy-v1/rgb/objects" in group
    t = pq.read_table(store / f"tables/{run_id}/embeddings/toy-v1/rgb/objects/table.parquet")
    assert t.schema.field("embedding").type == pa.list_(pa.float16(), 4)

def test_an_unknown_run_is_refused_before_any_part(staged_store_with_tables):
    store, _ = staged_store_with_tables
    with pytest.raises(ValueError, match="no run '9-unknown'"):
        replace_run_embeddings(store, "9-unknown", objects_payload_for(store), **digests(store))
    assert not list(store.parent.glob(store.name + "*.part"))
```

  Module-local: `digests(store, *, root=True)` gives
  `expected_root_sha256` (when `root`), `objmap_sha256` and `array_sha256`
  computed from the store; `owned_run(store, run_id) -> TableRun`;
  `other_run_dir(store, run_id) -> Path`; `consolidate_store(store)` calls the
  `_consolidate_store_part` writer (`_core/_image_parts/_image_io_handler.py`)
  the way a consolidated store is published today. `root_tables`,
  `touch_root` and `tree_digest` exist (`tests/_tiny_run.py:888,896,529`).
- [ ] **Step 2: Run them and confirm they fail** (`ImportError`).

  ```bash
  eval $PT tests/unit/sdk_/test_replace_run_embeddings.py
  ```

- [ ] **Step 3: Implement** the Interfaces above.
- [ ] **Step 4: Run them, plus the existing store-transaction tests.**

  ```bash
  eval $PT tests/unit/sdk_/test_replace_run_embeddings.py tests/unit/sdk_/test_embedding_tables.py tests/unit/cli/test_embedded_measurement_tables.py tests/unit/cli/test_embedded_measurement_replacement.py
  ```

  Expected: all pass.
- [ ] **Step 5: Commit.**

  ```bash
  git add src/phenotypic/sdk_/_measurement_tables.py src/phenotypic/sdk_/__init__.py tests/_tiny_run.py tests/conftest.py tests/unit/sdk_/test_replace_run_embeddings.py
  git commit -m "feat(store): replace_run_embeddings, one run's embeddings, root last (#282 P3)"
  ```

### Task 2: The Stage-4 signal

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py:852`. Add
  `DIR_STAGE4_PAYLOAD: Final[str] = "stage4_payload"` beside `DIR_STAGE2_DONE`.
- Create: `src/phenotypic/_cli/_cli_stage4_signal.py`
- Test: `tests/unit/cli/test_cli_stage4_signal.py` (new, shared env)

**Interfaces:**
- Consumes: `embedding_arrow_table` (`sdk_/_embedding_tables.py:160`),
  `split_embedding_key` (`sdk_/_table_runs.py:205`), `progress_dir`, the
  temp-then-`os.replace` pattern of `write_stage2_token`
  (`_cli_stage2_token.py:185`).
- Produces:
  - `stage4_signal_path(output_dir: Path, dataset: str, key: str, stem: str) -> Path`:
    `progress/stage4_payload/<ds>/<model>/<layer>/<region>/<stem>.parquet`
  - `write_stage4_signal(output_dir: Path, dataset: str, stem: str, payload: EmbeddingPayload, *, work_id: str, run_id: str, root_sha256: str, commit_guard: CommitGuard | None = None) -> Path`.
    The table is `embedding_arrow_table(payload)` with schema metadata key
    `b"phenotypic.stage4"`, a JSON object
    `{work_id, run_id, root_sha256, key, descriptor}`. Atomic.
  - `@dataclass(frozen=True) class Stage4Signal`: `payload: EmbeddingPayload`,
    `root_sha256: str`
  - `read_stage4_signal(path: Path, *, work_id: str, run_id: str) -> Stage4Signal | None`.
    `None` when the file is absent, unreadable, or fenced to another
    `work_id`/`run_id`; an unreadable file logs one warning naming it.
  - `classify_stage4_signals(output_dir: Path, dataset: str, stem: str, keys: Iterable[str], *, work_id: str, run_id: str, live_root_sha256: str) -> Stage4SignalState`
    with `@dataclass(frozen=True) class Stage4SignalState`: `valid: frozenset[str]`
    (readable, fenced to this work id and run, **and** `root_sha256 == live_root_sha256`)
    and `stale: frozenset[str]` (present but unreadable, fenced elsewhere, or
    computed against another root). Per key, so a round reuses the keys it
    already has (plan review C4), and against the **live** root, so a signal
    from before a reprocess is never committed (C5).
  - `stage4_signals_complete(state: Stage4SignalState, keys: Iterable[str]) -> bool`:
    `state.valid == set(keys)`.
  - `delete_stage4_signal_keys(output_dir: Path, dataset: str, stem: str, keys: Iterable[str], *, commit_guard: CommitGuard | None = None) -> int`:
    removes those keys' files only (the stale ones, before re-embedding them).
  - `delete_stage4_signals(output_dir: Path, dataset: str, stem: str, *, commit_guard: CommitGuard | None = None) -> int`:
    removes the stem's file under every key directory of the dataset; returns
    the count.

- [ ] **Step 1: Write the failing tests.**

```python
def test_a_signal_round_trips_to_the_same_store_bytes(tmp_path, image_with_objects_payload):
    payload = payloads_of(image_with_objects_payload)["toy-v1/rgb/objects"]
    path = write_stage4_signal(tmp_path, "ds", "img", payload, work_id="w", run_id="1-abc", root_sha256="r")
    back = read_stage4_signal(path, work_id="w", run_id="1-abc")
    assert back.root_sha256 == "r" and back.payload.descriptor == payload.descriptor
    assert embedding_arrow_table(back.payload).equals(embedding_arrow_table(payload))

def test_a_signal_of_another_work_id_or_run_is_ignored(tmp_path, image_with_objects_payload):
    payload = payloads_of(image_with_objects_payload)["toy-v1/rgb/objects"]
    path = write_stage4_signal(tmp_path, "ds", "img", payload, work_id="w", run_id="1-abc", root_sha256="r")
    assert read_stage4_signal(path, work_id="other", run_id="1-abc") is None
    assert read_stage4_signal(path, work_id="w", run_id="2-abc") is None

def test_the_path_is_keyed_by_model_layer_region(tmp_path):
    p = stage4_signal_path(tmp_path, "ds", "dinov2-vitb14/gray/grid-wells", "img")
    assert p.relative_to(tmp_path).parts[-6:] == (
        "stage4_payload", "ds", "dinov2-vitb14", "gray", "grid-wells", "img.parquet")

def test_signals_are_classified_per_key_against_the_live_root(tmp_path, image_with_two_models):
    a, b = payloads_of(image_with_two_models).values()
    write_stage4_signal(tmp_path, "ds", "img", a, work_id="w", run_id="1-x", root_sha256="live")
    state = classify_stage4_signals(tmp_path, "ds", "img", [a.key, b.key], work_id="w",
                                    run_id="1-x", live_root_sha256="live")
    assert state.valid == {a.key} and state.stale == frozenset()          # one key reusable
    assert not stage4_signals_complete(state, [a.key, b.key])
    write_stage4_signal(tmp_path, "ds", "img", b, work_id="w", run_id="1-x", root_sha256="old")
    state = classify_stage4_signals(tmp_path, "ds", "img", [a.key, b.key], work_id="w",
                                    run_id="1-x", live_root_sha256="live")
    assert state.valid == {a.key} and state.stale == {b.key}              # computed before a reprocess

def test_only_the_named_keys_are_deleted(tmp_path, image_with_two_models):
    a, b = payloads_of(image_with_two_models).values()
    for p in (a, b):
        write_stage4_signal(tmp_path, "ds", "img", p, work_id="w", run_id="1-x", root_sha256="r")
    assert delete_stage4_signal_keys(tmp_path, "ds", "img", [b.key]) == 1
    assert stage4_signal_path(tmp_path, "ds", a.key, "img").is_file()

def test_a_corrupt_signal_reads_as_absent(tmp_path, image_with_objects_payload, caplog):
    payload = payloads_of(image_with_objects_payload)["toy-v1/rgb/objects"]
    path = write_stage4_signal(tmp_path, "ds", "img", payload, work_id="w", run_id="1-x", root_sha256="r")
    path.write_bytes(b"corrupt")
    assert read_stage4_signal(path, work_id="w", run_id="1-x") is None
    assert str(path) in caplog.text

def test_delete_removes_every_key_of_the_stem_only(tmp_path, image_with_two_models):
    for p in payloads_of(image_with_two_models).values():
        for stem in ("img", "other"):
            write_stage4_signal(tmp_path, "ds", stem, p, work_id="w", run_id="1-x", root_sha256="r")
    assert delete_stage4_signals(tmp_path, "ds", "img") == 2
    assert len(list((tmp_path / ".phenotypic").rglob("other.parquet"))) == 2
```

- [ ] **Step 2: Run them and confirm they fail** (`ModuleNotFoundError`).

  ```bash
  eval $PT tests/unit/cli/test_cli_stage4_signal.py
  ```

- [ ] **Step 3: Implement.** The writer creates parents, writes a sibling temp
  file, `fsync`s it when `durable_writes_enabled()`, then `os.replace`s it under
  `commit_guard`.
- [ ] **Step 4: Run the P3 gate.** Both task files, plus importers of
  `_measurement_tables` and `_io_constants` (`grep -rl` over `tests/`), once:

  ```bash
  eval $PT tests/unit/sdk_ tests/unit/cli/test_cli_stage4_signal.py tests/unit/cli/test_cli_stage2_token.py
  ```

  Expected: all pass. Then the independent-reader check on one store the tests
  wrote (`zarr.open_group` + `pyarrow.parquet.read_table`), outside pytest.
- [ ] **Step 5: Commit.**

  ```bash
  git add src/phenotypic/sdk_/_io_constants.py src/phenotypic/_cli/_cli_stage4_signal.py tests/unit/cli/test_cli_stage4_signal.py
  git commit -m "feat(cli): the Stage-4 signal, fenced by work id, run and root (#282 P3)"
  ```
