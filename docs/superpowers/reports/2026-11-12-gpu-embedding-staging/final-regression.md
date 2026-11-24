# Full regression and docs build (#282, P7 Task 2, Steps 4–6)

**Gated SHA: `c355c0c9e`** (`feat/gpu-embedding-staging`), at the end of the
simplify pass. Logs, JUnit XML and the harness are in
`/bigdata/exfab/anguy344/gate-results/c19-final/`. `make_gate.sh` holds the gate
and `iso/make_iso.sh` the isolation runs. The simplify pass (Step 1) and mypy
and ruff (Step 3) are in [simplify.md](simplify.md). Step 2's focused re-run is
subsumed by this full run.

**Result: no #282 regression.** The 13 expected-red GUI tests are red. One
load flake and two environment errors are the only other reds.

## Environment check

| Venv | Contents |
|---|---|
| Shared (`.venv` of the `gpu-embedding-staging` worktree) | Python 3.12.10, pytest 9.0.1, xdist 3.8.0, optuna 4.9.0, astropy 8.0.1, napari 0.6.6; no torch |
| Foundation (`gate-results/c18/venv-foundation`) | Python 3.12.10, pytest 9.0.1, xdist 3.8.0, torch 2.11.0+cu130, optuna, transformers; **no astropy, no napari** |

The shared venv has the `tune` and `topology` extras, so the 24 false failures
from a missing optuna or astropy do not apply. Raw output is in
`c19-final/env_check.txt`.

## Jobs

| Job | Id | Partition | Shape | Wall |
|---|---|---|---|---|
| Shards | `32431259` (`_0`–`_23`) | `short` | 8 CPU, 24 GB, `-n $SLURM_CPUS_PER_TASK` | 3:46 to 22:36 per shard |
| Foundation lane | `32431260` | `short` | 16 CPU, 32 GB, `-n 16`, `OMP_NUM_THREADS=1` | 22:36 |
| Golden | `32431261` | `short`, `--constraint=broadwell` (node `i01`) | 4 CPU, `OMP_NUM_THREADS=4`, one process, no `-m` filter | 6:04 |
| Docs build | `32431262` | `batch` | 8 CPU, `sphinx-build -j 8 -D nbsphinx_execute=never` | 43:38 |
| Cleanup | `32431263` | `short` | `afterany` on all four; removes both worktrees | see Cleanup |
| Isolation: `test_slurm` | `32431918` | `short` | 8 CPU, shared venv, both SHAs | 7:17 |
| Isolation: golden | `32431919` | `short`, broadwell (node `i01`) | 4 CPU, `OMP_NUM_THREADS=4`, foundation venv, both SHAs | 5:40 |
| Isolation cleanup | `32431920` | `short` | `afterany` | — |
| Supplementary torch run | `32431923` | `short` | 8 CPU, foundation venv | 3:46 |
| Supplementary cleanup | `32431924` | `short` | `afterany` | 0:29 |

- **Trees:** each run is in its own detached worktree under
  `/bigdata/exfab/anguy344/gate-worktrees/`:
  - `c19-c355c0c9e` for the gate;
  - `c19-docs-c355c0c9e` for the docs build, so files Sphinx generates cannot
    reach a running shard;
  - `c19iso-c355c0c9e` and `c19iso-99bbc39cd` for the isolation runs;
  - `c19supp-c355c0c9e` for the supplementary run.
- **Every log** printed `dirty=0` at start and end and imported `phenotypic`
  from its own tree's `src/`. The docs log printed `dirty=0` and
  `dirty_end=0`, with `SPHINX_EXIT=0`.
- **Pytest flags:** the shards and the foundation lane ran
  `-q -p no:randomly -p no:cacheprovider -o addopts= -m "not slow"`, the same
  as #281's full run. None ran with `-n auto` or `-x`.

## Scope

- **Shards:** every `test_*.py` under the four testpaths, collected with
  `find`. That is **950 files**: 800 in `tests/unit`, 3 in `tests/smoke`, 88 in
  `tests/integration` and 59 in `tests/gui`. They were assigned round-robin by
  index over the 24 shards.
- **Foundation lane: 63 files.** The union of:
  - every file matching `importorskip("torch")`, `import torch` / `from torch`,
    `stub_backbone` or `param_stub_backbone`;
  - everything in `tests/unit/measure` and `tests/unit/detect/nn`;
  - every `test_*.py` that `git diff --diff-filter=A 99bbc39cd..c355c0c9e -- tests`
    lists. That diff adds 32 paths, which includes the `_fakes` helpers and the
    golden data.
- **Golden job:** `tests/unit/measure/test_embedding_inline_golden.py` alone,
  in its capture environment. The CPU was `Intel(R) Xeon(R) CPU E5-2683 v4 @
  2.10GHz`, with torch 2.11.0+cu130, `OMP_NUM_THREADS=4` and
  `torch.get_num_threads()` 4. The job had no `-m` filter, so the slow
  real-DINOv2-small rows were collected too.

## Counts

| Lane | Tests | Passed | Failed | Errors | Skipped | xfailed |
|---|---|---|---|---|---|---|
| Shards | 17,562 | 17,321 | 14 | 0 | 204 | 23 |
| Foundation | 1,315 | 1,295 | 0 | 0 | 20 | 0 |
| Golden | 25 | 23 | 0 | 2 | 0 | 0 |
| Supplementary (`test_process_in_place.py`, torch) | 89 | 89 | 0 | 0 | 0 | 0 |

The counts come from the JUnit XML. Every shard's summary line agrees with
them.

## The reds

| # | Test | Lane | Attribution | Evidence |
|---|---|---|---|---|
| 1–13 | The 13 rows of the README's expected-red table (`test_validation` 1, `test_setup_authoring` 2, `integration/gui/tune/test_setup_view` 3, `test_curation_does_not_unfence_the_run` 1, `test_measurement_routes` 1, `test_measurement_source` 5) | shards 6, 8, 9, 20, 22 | **expected** (#283, inherited from #281) | Exactly the README set, name for name. No other GUI test is red. |
| 14 | `tests/unit/gui/run_console/test_slurm.py::test_streamed_submitter_timeout_stops_pipe_inheriting_descendant` | shard 16 (node `c10`) | **load flake, pre-existing** | See below. |
| 15–16 | `tests/unit/measure/test_embedding_inline_golden.py::test_real_small_backbone_matches_the_golden` and `::test_real_small_vectors_are_near_the_golden` (errors in the `real_small_payloads` fixture) | golden | **environment artifact** (an invalid stored Hugging Face token) | See below. |

### 14: `test_streamed_submitter_timeout_stops_pipe_inheriting_descendant`

- **What failed:** `descendant_pid_path.read_text()` raised, because no pid
  file existed. The test kills its submitter after `timeout=0.2` s. The pid
  file is written by a grandchild Python process, which must start up first.
  On a loaded node, 0.2 s is not long enough for that.
- **Node load:** shard 16 shared `c10` with five other gate shards (12–17),
  each running `-n 8`.
- **In isolation** (job `32431918`, node `r41`), at both `c355c0c9e` and
  `99bbc39cd`:
  - the test alone passed **5 of 5** times;
  - the whole file under `-n 8` passed `30 passed` both times.
- **Not #282 code:** `git diff 99bbc39cd..c355c0c9e -- src/phenotypic/_gui tests/unit/gui/run_console`
  is empty. #282 changes no file under `_gui/` (README invariant).
- It belongs with the run-console timing flakes #281's report lists.

### 15–16: the two real-DINOv2-small golden rows

- **What failed:** loading `facebook/dinov2-small` raised
  `OSError: facebook/dinov2-small is not a local folder and is not a valid model identifier`.
  The skip guard passed because the weights are cached.
- **The stored token is invalid.**
  - `huggingface_hub.whoami()` returns
    `HfHubHTTPError: Invalid user token. The token stored is invalid.`
  - `model_info("facebook/dinov2-small")` with that token gives a **401
    `RepositoryNotFoundError`**.
  - The same call with `token=False` returns sha `ed25f3a31f01…`.
  - So the stored token makes a public repo look missing. This is the
    misleading message #281's report names as a follow-up.
- **The fixture expects an anonymous run.** Its docstring says "Run
  anonymously (`HF_HUB_DISABLE_IMPLICIT_TOKEN=1`) for this public repo." The
  gate job did not set that variable. That omission was the gate harness's,
  not the code's.
- **Isolation, job `32431919`,** on `i01` in the capture environment:

  | Tree | Stored token | `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` |
  |---|---|---|
  | `c355c0c9e`, the two tests | `2 errors` (same `OSError`) | **`2 passed`**, including the real-row **byte** match |
  | `99bbc39cd`, `iso/probe_small.py` (the same `MeasureDinoEmbedding(size="small", device="cpu")` load through `ImagePipeline.embed`; the golden file does not exist at #281) | `PROBE FAILED` (same `OSError`) | `PROBE OK dinov2-vits14/rgb/objects (552, 2)` |

  The failure is the same at #281, and it goes away when no token is sent.
  So it is environment, not #282.
- **The golden bytes reproduce.** In the gate's golden job, all 6
  `test_inline_bytes_match_the_281_golden[...]` rows ran and passed in the
  capture environment, and none skipped. The 6 cosine rows and the 9
  params-digest rows also passed. In the isolation job, the real-small rows
  matched the golden bytes once the token was dropped.

## Skips

- **Shard skips: 174 of the 204 need torch, transformers or model weights.**
  - 8 are whole-module `collection skipped` entries, one per torch-only file
    that has a module-level `importorskip("torch")`: `test_stage4_worker`,
    `test_stage4_slurm_worker`, `test_stage4_deferral`, `test_local_stage4`,
    `test_measure_stage4`, `test_embed_stage_refusals`, `test_determinism` and
    `test_batch_encoder`. All 8 files are in the foundation lane, where their
    197 tests all passed.
  - Of the other 166, 119 passed in the foundation lane.
  - 30 passed in the supplementary run (see the next point).
  - 6 are the golden byte rows. They skipped in the foundation lane too,
    because its `OMP_NUM_THREADS=1` is not the capture environment. All 6
    passed in the golden job.
  - 11 skipped in both lanes. These are SAM2 (3), micro_sam (3), SAM3 (1) and
    the DINOv2 detectors' functional tests (4), skipped for "a loadable DINOv2
    backbone (ungated)". The DINOv2 detector skips are probably the same
    invalid token: their probe loads from the hub. None of the 11 is a #282
    file.
- **A selection gap in the foundation lane.** `tests/unit/cli/test_process_in_place.py`
  skips its 30 `[dino]` cases with `importlib.util.find_spec("torch") is None`
  and the reason "Requires torch (stub backbone)". The lane's grep did not
  match that form, so those cases ran in neither lane.
  - It is the only `find_spec("torch")` or "Requires torch" file outside the
    63 (`iso/supp_files.txt`).
  - It matters for #282 because process mode must stay inline on CPU, byte
    for byte (H5).
  - The supplementary run `32431923` ran the whole file in the foundation venv
    at `c355c0c9e`: **89 passed**, 30 of them `[dino]`.
  - A later gate's foundation grep should add `find_spec\("torch"\)`.
- **The other 30 shard skips are pre-existing:**
  - the migrate test bed and its fixtures (9);
  - Chrome/kaleido (5);
  - exiftool (3);
  - Windows-only behaviour (3);
  - golden-capture switches (2), Postgres (2) and HDF tripwires (2);
  - 3 single tripwire or bridge skips (`_output_consistency`,
    `_processing_state_conflict`, the zarr sync bridge), plus 1 empty
    parameter set.
- **The foundation lane's 20 skips:**
  - the same 11 model skips;
  - the 6 golden byte rows;
  - 2 golden-capture switches (`PHENOTYPIC_CAPTURE_GOLDEN=1`);
  - 1 Chrome/kaleido skip.

## mypy and ruff (from [simplify.md](simplify.md); not re-run here)

- **mypy:** `python -m mypy src/phenotypic`, run in a detached worktree at
  each end of the simplify pass.
  - `1d5b28476` (before): **494 errors in 162 files**.
  - `711e6b66b` (after): **492 errors in 161 files**. That snapshot's `src/`
    is `HEAD`'s, byte for byte, and `c355c0c9e` changes only a report.
  - The only per-file change is in `_cli/_cli_staged_resume.py`, which goes
    from 2 errors to 0 (S6). No file gained an error.
  - The "before" end is the pre-simplify commit, not the #281 merge commit
    `99bbc39cd` that the plan names. No count at `99bbc39cd` is recorded.
- **ruff:** `uv run ruff check` over every `src/` file changed since
  `1d5b28476` printed `All checks passed!`.

## Docs build

Job `32431262` ran at `c355c0c9e`. It was compared with C18's build at
`fac392a2d` (`gate-results/c18-282/sphinx-fac392a2d….err`, job `32427445`),
which used the same command.

- **Comparison method:** each log's `WARNING` and `ERROR` lines, with the
  worktree prefix stripped, compared as multisets.

| Build | `build succeeded, N warnings` | WARNING/ERROR lines |
|---|---|---|
| C18, `fac392a2d` | 737 | 737 |
| C19, `c355c0c9e` | 720 | 720 |

- **Gone: exactly 17,** the ones simplify fixed.
  - 15 duplicate object descriptions (D1): `PreparedRegions` 7, `TableRun` 5,
    `EmbeddingPayload` 3.
  - 2 "Inline literal start-string without end-string" warnings on
    `entry_descriptor_for:26` (D2): one through `measure/__init__`, one
    through the module.
- **New: none.** The C19 multiset minus C18's is empty.
- **Still there, as simplify.md expected (R1, R2):**
  - the 2 DINO_EMBEDDING.rst:84 `str.format` warnings;
  - the 2 `model_json_schema:8` warnings attributed to
    `_measure_embedding.py` and `_measure_dino_embedding.py`.
- **Rendered pages, checked in the HTML:**
  - `phenotypic.sdk_.TableRun.html`, `phenotypic.abc_.PreparedRegions.html`
    and `phenotypic.sdk_.EmbeddingPayload.html` each carry one `id=` per field.
  - Each field's `#:` text renders. For example, `TableRun.order` reads
    "The store-local call-order number (spec D2)."
  - `MeasureDinoEmbedding.entry_descriptor_for`'s **Returns** section renders
    as a whole: "The descriptor {model, dim, dtype, compute_dtype, pooling,
    compute}, whose model is {family, hf_id, …}".

## Cleanup

Every worktree was removed by its `afterany` cleanup, each with status 0:

- `32431263` removed the gate tree and the docs tree. It took 17:07, most of
  it deleting the docs tree's `uv sync` venv.
- `32431920` removed both isolation trees.
- `32431924` removed the supplementary tree.

Afterwards, `git worktree list` shows no `c19*` entry, and
`gate-worktrees/` holds none.

## Follow-ups

- **The stored Hugging Face token is invalid.** Refresh it with
  `hf auth login --force`. Until then:
  - any gate that runs the golden's real rows must set
    `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`;
  - the DINOv2 detectors' functional tests stay skipped.
- **The 401 surfaces as "not a valid model identifier"** from the shared DINO
  loader. A clearer message is still a follow-up, as in #281's report.
- **Add `find_spec\("torch"\)` to the foundation-lane grep** in later gates'
  `make_gate.sh`.
- **`test_streamed_submitter_timeout_stops_pipe_inheriting_descendant` has a
  `timeout=0.2` s budget,** which a loaded node misses. It is another
  run-console timing flake.
