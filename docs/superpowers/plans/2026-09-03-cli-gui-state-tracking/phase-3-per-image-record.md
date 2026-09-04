# Phase 3 — One per-image record

**Depends on:** P1, P2. **Blocks:** P4–P7.

**Spec:** §6.1 (one record), §6.2 (store immutability) — as amended by
[D-A](OPEN-QUESTIONS.md#d-a-per-store-metadata-is-written-at-promote-time-not-backfilled)
and [O-2](OPEN-QUESTIONS.md#o-2-stages-is-an-open-map-with-no-name-validation).

**Goal:** `image_complete/`, `stage2_done/` and `stage3_complete/` — three parallel
`<ds>/<stem>.*` trees answering three sub-questions about the same image, spelled in three
different places — become one record at
`.phenotypic/progress/images/<dataset>/<stem>.json` with an open `stages` map. "Is this
image done?" becomes one JSON read instead of one read plus up to three `is_file()` probes
across three directory trees.

`stage2_raw/<ds>/<stem>.npy` **stays a separate file**. It is bulk replay data, not a
record, and the staged engine's Stage-3 replay reads it as an array.

### What D-A cuts from this phase

Spec §6.3's hardlink re-promote and §6.4's certified-rewrite protocol are **not built**.
Per-store metadata is written at promote time (P4 Task 5), so there is no post-proof store
mutation to certify. The pre-existing
`refresh_success_markers_after_metadata_migration` (`_cli_completion.py:305`) is
**untouched here** and stays scoped to `--mode migrate` in P7 — it serves one historical
case and keeps `RuntimeError` for an artifact that moved without a covering receipt, which
is INV-IMMUTABLE's exception and its only one.

`stages` therefore carries no `backfilled` key. The map stays open, so adding one later is
additive.

---

## File Structure

| File | Responsibility |
|---|---|
| **Create** `src/phenotypic/_cli/_cli_image_record.py` | `publish_image_record`, `read_image_record`, `record_stage`, `KNOWN_STAGES`. The single writer. ~220 lines. |
| **Modify** `src/phenotypic/_cli/_cli_completion.py` | `publish_image_success` / `valid_image_success` delegate to the record. |
| **Modify** `src/phenotypic/_cli/_cli_stage2_token.py` | `write_stage2_token` / `stage2_token_exists` / `delete_stage2_token` become `stages.stage2` operations. `stage2_raw` helpers unchanged. |
| **Modify** `src/phenotypic/_cli/_cli_staged_resume.py` | `stage3_completion_exists` / `write_stage3_completion_marker` / `remove_stage3_completion_marker` become `stages.stage3` operations. `classify_staged_image` reads one record. |
| **Modify** `src/phenotypic/sdk_/_run_state.py` | The deep path reads the record instead of the legacy marker. |
| **Delete** | `DIR_STAGE2_DONE` / `DIR_STAGE3_COMPLETE` path helpers, the inline `"stage3_complete"` literal at `_cli_staged_resume.py:141`, and `_STAGE2_DIR` at `_cli_stage2_token.py:42`. |
| **Test** `tests/unit/cli/test_image_record.py` *(new)* | Record schema, stage independence, O-2 advisory. |
| **Test** `tests/unit/cli/test_staged_resume_equivalence.py` *(new)* | The gate: resume decisions are unchanged. |

**The staged engine's resume logic is the risk in this phase**, not the record format.
`classify_staged_image` (`_cli_staged_resume.py:197`) decides, per image, whether to run
stage 1, 2 or 3, from four independent filesystem probes. Collapsing those into one read
must not change a single one of its decisions.

---

## Interfaces

**Produces:**

```python
# phenotypic._cli._cli_image_record

#: Stage names this build understands. `stages` stays an OPEN map (§6.1) -- an
#: unknown key is surfaced as a RunState advisory (O-2), never rejected.
KNOWN_STAGES: frozenset[str] = frozenset({"stage1", "stage2", "stage3", "measured"})

RECORD_VERSION: int = 1

def publish_image_record(
    output_dir: Path,
    *,
    work_id: str,
    dataset: str,
    image_stem: str,
    relative_image_path: str,
    mode: str,
    stages: Mapping[str, Mapping[str, object]],
    artifacts: Mapping[str, Path],
    attempt_id: str,
    scheduler_epoch: str,
    commit_guard: "CommitGuard | None" = None,
) -> Path: ...

def read_image_record(
    output_dir: Path, dataset: str, image_stem: str
) -> dict[str, object] | None: ...

def record_stage(
    output_dir: Path, dataset: str, image_stem: str, stage: str,
    payload: Mapping[str, object], *, commit_guard=None,
) -> Path: ...

def consume_stage(
    output_dir: Path, dataset: str, image_stem: str, stage: str
) -> bool: ...
```

**Consumes:** `phenotypic.sdk_.image_record_path` (P1),
`phenotypic._cli._cli_identity.scheduler_epoch` plumbing (P2).

---

## Task 1: The record writer and reader

**Files:**
- Create: `src/phenotypic/_cli/_cli_image_record.py`
- Test: `tests/unit/cli/test_image_record.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_the_record_is_one_file_carrying_every_stage(tmp_path):
    """Spec §6.1: 'Is this image done?' becomes ONE JSON read instead of one read
    plus up to three is_file() probes across three directory trees."""
    from phenotypic._cli._cli_image_record import publish_image_record, read_image_record
    from phenotypic.sdk_ import image_record_path

    store = tmp_path / "results" / "plate" / "zarr" / "a.ome.zarr"
    store.mkdir(parents=True)
    (store / "zarr.json").write_text("{}", encoding="utf-8")

    publish_image_record(
        tmp_path,
        work_id="w", dataset="plate", image_stem="a",
        relative_image_path="a.tif", mode="full",
        stages={
            "stage1": {"at": "2026-09-03T00:00:00Z"},
            "stage2": {"at": "2026-09-03T00:00:01Z", "objmap_shape": [1024, 1024]},
            "stage3": {"at": "2026-09-03T00:00:02Z"},
            "measured": {"at": "2026-09-03T00:00:03Z"},
        },
        artifacts={"store": store},
        attempt_id="attempt", scheduler_epoch="epoch",
    )

    assert image_record_path(tmp_path, "plate", "a").is_file()
    record = read_image_record(tmp_path, "plate", "a")
    assert set(record["stages"]) == {"stage1", "stage2", "stage3", "measured"}
    assert record["artifacts"]["store"]["kind"] == "store"


def test_stages_is_an_open_map(tmp_path):
    """§6.1: `stages` and `artifacts` are open maps -- that is what makes a future
    stage additive rather than a schema break."""
    from phenotypic._cli._cli_image_record import publish_image_record, read_image_record

    publish_image_record(
        tmp_path, work_id="w", dataset="plate", image_stem="a",
        relative_image_path="a.tif", mode="full",
        stages={"stage1": {"at": "t"}, "some_future_stage": {"at": "t"}},
        artifacts={}, attempt_id="x", scheduler_epoch="e",
    )
    assert "some_future_stage" in read_image_record(tmp_path, "plate", "a")["stages"]


def test_an_unknown_stage_becomes_an_advisory_not_a_failure(tmp_path):
    """O-2: an open map with no name validation means a typo like `stage_2` reads
    as 'stage 2 not done' and never errors. Surface it without closing the map."""
    from phenotypic._cli._cli_image_record import publish_image_record
    from phenotypic.sdk_ import resolve_run_state

    publish_image_record(
        tmp_path, work_id="w", dataset="plate", image_stem="a",
        relative_image_path="a.tif", mode="full",
        stages={"stage_2": {"at": "t"}},      # note the typo
        artifacts={}, attempt_id="x", scheduler_epoch="e",
    )
    state = resolve_run_state(tmp_path, depth="deep")
    assert any("stage_2" in advisory for advisory in state.advisories)


def test_recording_one_stage_leaves_the_others_untouched(tmp_path):
    """The three collapsed trees were independently writable and must stay so --
    Stage 2 and Stage 3 run in different jobs, on different nodes, minutes apart."""
    from phenotypic._cli._cli_image_record import read_image_record, record_stage

    record_stage(tmp_path, "plate", "a", "stage1", {"at": "t1"})
    record_stage(tmp_path, "plate", "a", "stage2", {"at": "t2"})
    record = read_image_record(tmp_path, "plate", "a")
    assert set(record["stages"]) == {"stage1", "stage2"}


def test_reading_a_corrupt_record_is_none_not_an_error(tmp_path):
    """INV-DEGRADE."""
    from phenotypic._cli._cli_image_record import read_image_record
    from phenotypic.sdk_ import image_record_path

    path = image_record_path(tmp_path, "plate", "a")
    path.parent.mkdir(parents=True)
    path.write_text("{truncated", encoding="utf-8")
    assert read_image_record(tmp_path, "plate", "a") is None
```

- [ ] **Step 2: Run to verify failure.** Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

The record, per §6.1 minus `backfilled`:

```json
{
  "version": 1,
  "work_id": "…", "dataset": "…", "image_stem": "…",
  "relative_image_path": "…", "mode": "full|process|measure",
  "stages": {
    "stage1":   {"at": "…"},
    "stage2":   {"at": "…", "objmap_shape": [1024, 1024], "detector_seconds": 1.23},
    "stage3":   {"at": "…"},
    "measured": {"at": "…"}
  },
  "artifacts": {
    "store":        {"kind": "store", "path": "…", "sha256": "<root zarr.json digest>"},
    "measurements": {"kind": "file",  "path": "…", "size": 12345, "sha256": "…"},
    "metadata":     {"kind": "file",  "path": "…", "size": 234,   "sha256": "…"},
    "overlay":      {"kind": "file",  "path": "…", "size": 67890, "sha256": "…"}
  },
  "attempt_id": "…", "scheduler_epoch": "…", "completed_at": "…"
}
```

`record_stage` is **read-modify-write under the existing `atomic_write_json` +
`pre_replace` revalidation**, exactly as `publish_image_success` does today
(`_cli_completion.py:163`). Two stages written concurrently for the same image is not a
real case — the staged engine runs them in different jobs, serialized by the stage-2 token
— but the read-modify-write must still not lose a key on a retry.

`consume_stage` replaces `delete_stage2_token`'s unlink: it removes one key from `stages`
and rewrites. **Consumption must be idempotent** — Stage 3 already tolerates a token that
another attempt consumed.

- [ ] **Step 4: Run the tests.** Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_image_record.py tests/unit/cli/test_image_record.py
git commit -m "feat(cli): one per-image record with an open stages map

Spec §6.1. No `backfilled` stage (D-A). An unrecognised stage name becomes a
RunState advisory rather than silently reading as not-done (O-2)."
```

---

## Task 2: Migrate `publish_image_success` and `valid_image_success` onto the record

**Files:**
- Modify: `src/phenotypic/_cli/_cli_completion.py:163`, `:255`
- Modify: `src/phenotypic/sdk_/_run_state.py`
- Test: `tests/unit/cli/test_image_record.py`, `tests/unit/sdk_/test_run_state.py`

- [ ] **Step 1: Write the failing test**

```python
def test_publish_image_success_writes_the_record_not_the_legacy_marker(tmp_path):
    from phenotypic.sdk_ import image_completion_marker_path, image_record_path

    _publish_a_successful_image(tmp_path, dataset="plate", stem="a")
    assert image_record_path(tmp_path, "plate", "a").is_file()
    assert not image_completion_marker_path(tmp_path, "plate", "a").exists(), (
        "the legacy image_complete/ marker is still being written; D1 is a clean "
        "break, not a dual write"
    )


def test_valid_image_success_still_rejects_a_tampered_artifact(tmp_path):
    """The artifact-digest contract is unchanged by the collapse. This is the
    property `_walk_current_success` has today and the one P6 will lean on."""
    from phenotypic._cli._cli_completion import valid_image_success

    store = _publish_a_successful_image(tmp_path, dataset="plate", stem="a")
    assert valid_image_success(tmp_path, dataset="plate", image_stem="a", work_id="w")
    (store / "zarr.json").write_text('{"tampered": true}', encoding="utf-8")
    assert not valid_image_success(
        tmp_path, dataset="plate", image_stem="a", work_id="w"
    )
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

`publish_image_success` keeps its signature and its artifact-validation body — the marker
it writes is now the record, with `stages={"measured": {...}}` plus whatever the caller
passes. **D1 is a clean break: no dual write.** A tree with `image_complete/` and no
`images/` is a legacy tree, and P7's migrate converts it; every other mode refuses it.

`valid_image_success` reads the record. Its `SUCCESS_MARKER_VERSION` check becomes
`RECORD_VERSION`.

Update `sdk_/_run_state.py`'s deep path to read the record and populate `ImageState.stages`
from it — **and delete the P1 comment saying the single-key `stages` is temporary.**

- [ ] **Step 4: Run the tests plus the completion regression**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli tests/unit/sdk_ -q
```

- [ ] **Step 5: Commit**

```bash
git add -A src/phenotypic tests/unit
git commit -m "refactor(cli): publish_image_success writes the collapsed record

Spec §6.1, D1 -- clean break, no dual write. Artifact-digest validation is
unchanged; only where the descriptors live moved."
```

---

## Task 3: Stage 2 and Stage 3 become `stages` entries

**Files:**
- Modify: `src/phenotypic/_cli/_cli_stage2_token.py`
- Modify: `src/phenotypic/_cli/_cli_staged_resume.py`
- Test: `tests/unit/cli/test_staged_resume_equivalence.py`

**This is the risky task.** `classify_staged_image` (`_cli_staged_resume.py:197`) decides
per image whether to run stage 1, 2 or 3. Collapsing four filesystem probes into one read
must change none of its decisions.

- [ ] **Step 1: Write the equivalence gate BEFORE touching the staged engine**

`tests/unit/cli/test_staged_resume_equivalence.py`:

```python
"""The staged resume decisions must be identical after the marker collapse.

classify_staged_image reads four independent filesystem probes today: a valid stage-1
store, a stage-2 token, a retained stage-2 raw array, and a stage-3 completion marker.
Spec §6.1 collapses three of those into one record read. The record format is not the
risk -- the resume DECISIONS are, because a wrong one either reprocesses 6,000 images
or silently skips one.

This is a table test over every reachable combination, written against the CURRENT
behaviour and run before and after the change.
"""

import itertools

import pytest

_COMBOS = list(itertools.product([False, True], repeat=4))  # store, s2tok, s2raw, s3

#: Captured from the PRE-CHANGE behaviour in Step 2, as a literal table. Do not
#: derive these by reasoning about what the classifier should do -- the point is to
#: freeze what it DOES, so the collapse is provably behaviour-preserving.
_EXPECTED: dict[tuple[bool, bool, bool, bool], str] = {}   # filled in Step 2


@pytest.mark.parametrize("store,s2_token,s2_raw,s3_done", _COMBOS)
def test_classification_is_unchanged_by_the_collapse(
    tmp_path, store, s2_token, s2_raw, s3_done
):
    from phenotypic._cli._cli_staged_resume import classify_staged_image

    item = _plant(tmp_path, store=store, s2_token=s2_token, s2_raw=s2_raw, s3_done=s3_done)
    actual = classify_staged_image(
        tmp_path,
        dataset=item.dataset,
        image_stem=item.image_stem,
        work_id=item.work_id,
        image_path=item.image_path,
    )
    assert actual == _EXPECTED[(store, s2_token, s2_raw, s3_done)]
```

**Confirm `classify_staged_image`'s real signature before writing `_plant`:**

```bash
sed -n '197,230p' src/phenotypic/_cli/_cli_staged_resume.py
```

The keyword names above are the shape the function is expected to have; use whatever it
actually takes. `_plant` creates or omits each of the four artifacts and returns a small
record carrying the arguments the call needs.

- [ ] **Step 2: Populate `_EXPECTED` from the CURRENT code, before changing it**

Run the parametrized test against unmodified `main` with `_EXPECTED` empty, capture each
actual classification, and write those sixteen values into `_EXPECTED` **as a literal
table**. Then re-run: all sixteen pass. That table is now the contract.

Do **not** derive `_EXPECTED` by reasoning about what the classifier should do. The point is
to freeze what it *does*, so the collapse is provably behaviour-preserving. If one of the
sixteen looks wrong, record it in a comment and leave it — fixing a resume bug inside a
refactor makes both unreviewable.

- [ ] **Step 3: Collapse the two trees**

- `write_stage2_token` → `record_stage(..., "stage2", {...})`
- `stage2_token_exists` → `"stage2" in (read_image_record(...) or {}).get("stages", {})`
- `delete_stage2_token` → `consume_stage(..., "stage2")`
- `write_stage3_completion_marker` → `record_stage(..., "stage3", {...})`
- `stage3_completion_exists` → the same membership test on `"stage3"`
- `remove_stage3_completion_marker` → `consume_stage(..., "stage3")`

Keep the function names — the SLURM observer imports `stage3_completion_exists`
(`_slurm_observer.py`), and renaming it is P6's job, not this task's.

Delete `_STAGE2_DIR` (`_cli_stage2_token.py:42`), the inline `"stage3_complete"` literal
(`_cli_staged_resume.py:141`), and their path helpers. `stage2_raw_path`,
`write_stage2_raw`, `load_stage2_raw` and `delete_stage2_raw` are **unchanged**.

- [ ] **Step 4: Re-run the equivalence gate**

Run: `uv run pytest tests/unit/cli/test_staged_resume_equivalence.py -v`
Expected: all sixteen PASS, against the table captured in Step 2.

**If any combination changes, stop.** The collapse has altered a resume decision, and that
is the failure this task exists to prevent.

- [ ] **Step 5: Run the staged-engine regression**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli -k staged -q
```

- [ ] **Step 6: Phase gate**

```bash
uv run mypy src/phenotypic
uv run ruff check --fix src/phenotypic/_cli/_cli_image_record.py \
  src/phenotypic/_cli/_cli_stage2_token.py src/phenotypic/_cli/_cli_staged_resume.py \
  src/phenotypic/_cli/_cli_completion.py src/phenotypic/sdk_/_run_state.py \
  tests/unit/cli/
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli tests/unit/sdk_ -q
```

- [ ] **Step 7: Commit**

```bash
git add -A src/phenotypic/_cli tests/unit/cli
git commit -m "refactor(cli): stage2_done/ and stage3_complete/ become stages entries

Spec §6.1. Three parallel <ds>/<stem> trees, spelled in three places, become one
record. The sixteen-combination classify_staged_image table was captured from the
pre-change behaviour and is unchanged after -- the resume decisions are the risk
here, not the format."
```
