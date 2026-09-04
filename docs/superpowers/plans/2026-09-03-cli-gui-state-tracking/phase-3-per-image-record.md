# Phase 3 — One per-image record

**Depends on:** P1, P2. **Blocks:** P4–P7.

**Spec:** §6.1 (one record), §6.2 (store immutability) — as amended by
[D-A](OPEN-QUESTIONS.md#d-a-per-store-metadata-is-written-at-promote-time-not-backfilled)
and [O-2](OPEN-QUESTIONS.md#o-2-stages-is-an-open-map-with-no-name-validation).

**Goal:** `image_complete/` and `stage3_complete/` — two parallel `<ds>/<stem>.json` trees
answering two sub-questions about the same image — become one record at
`.phenotypic/progress/images/<dataset>/<stem>.json` with an open `stages` map. "Is this
image done?" becomes one JSON read instead of one read plus `is_file()` probes across
separate directory trees.

### Two things stay separate files, for different reasons

`stage2_raw/<ds>/<stem>.npy` — bulk replay data, not a record. Spec §6.1 already says so.

**`stage2_done/<ds>/<stem>.json` — narrowed out of the collapse (user ruling, round 3).**
Spec §6.1 folds it into `stages.stage2`; it must not be.

The token is not a *record*, it is a **consumable claim**: Stage 3 takes it by
`unlink()`ing it. That operation is atomic on every filesystem and needs **no lock at
all**. Folding it into the shared record replaces it with a **locked read-modify-write of a
JSON file**, and three things compound in the environment this actually runs in:

1. **The lock is `flock`**, not POSIX record locking (`sdk_/_file_locking.py:101`,
   `fcntl.flock(handle.fileno(), LOCK_EX | LOCK_NB)`). On network filesystems `flock` is the
   weaker of the two options — `F_SETLK` record locks are the ones with defined cross-node
   semantics. GPFS does support `flock` cluster-wide, so this works; it is nonetheless the
   fragile choice where an atomic `unlink` was the robust one.
2. **The cited precedent does not transfer.** `.aggregate_publication.lock`
   (`_cli_output_manager.py:1556`) is a **run-level singleton** — one writer, negligible
   contention, and a missed lock costs a duplicate publish. The collapse would apply the same
   primitive **per image across a 6,000-task SLURM array** (flow-r3 C4).
3. **It trades down.** An operation needing no lock becomes one needing a *distributed*
   lock, to achieve exactly what the unlink already achieved.

So `write_stage2_token`, `stage2_token_exists` and `delete_stage2_token` keep their file and
their unlink. `_STAGE2_DIR` moves into `_io_constants` beside its siblings (audit S9's real
ask) rather than being deleted.

**What this costs:** "is this image done?" is one record read **plus one `is_file()` probe**
for the Stage-2 claim, not a single read. That is the honest price of keeping an atomic
claim atomic, and it is still two probes fewer than today's four.

### What D-A cuts from this phase

Spec §6.3's hardlink re-promote and §6.4's certified-rewrite protocol are **not built**.
Per-store metadata is written at promote time (P4 Task 5), so there is no post-proof store
mutation to certify. The pre-existing
`refresh_success_markers_after_metadata_migration` (`_cli_completion.py:305`) is
**untouched here** and stays scoped to `--mode migrate` in P7 — it serves one historical
case and keeps `RuntimeError` for an artifact that moved without a covering receipt, which
is INV-PROVEN's certified-transition exception.

`stages` therefore carries no `backfilled` key. The map stays open, so adding one later is
additive.

---

## File Structure

| File | Responsibility |
|---|---|
| **Create** `src/phenotypic/sdk_/_image_record.py` | **Readers and shared vocabulary:** `read_image_record`, `RECORD_VERSION`, the four `STAGE_*` constants. ~70 lines. |
| **Create** `src/phenotypic/_cli/_cli_image_record.py` | **Writers only:** `publish_image_record`, `record_stage`, `consume_stage`. Imports the vocabulary from `sdk_`. ~180 lines. |

> **The split is forced, and discovering it in P3 is much cheaper than in P6 (N-3).** P6
> Task 0 moves `valid_image_success` into `sdk_/_run_state.py` (CAN-8), and after P3 that
> function reads the record. INV-LAYER forbids `sdk_` importing `phenotypic._cli` at module
> scope *or inside a function*, so `sdk_` would have to parse the record and know
> `RECORD_VERSION` and `STAGE_MEASURED` without importing them. The only two outs are
> duplicating the constants — which **CAN-27's own resolution rejects by name**, having just
> replaced `KNOWN_STAGES` for exactly that reason — or re-implementing record parsing in
> `sdk_`, which creates the second reader of the record schema that CAN-8 exists to prevent.
>
> Putting readers plus vocabulary in `sdk_` and writers in `_cli` satisfies **CAN-27** (one
> spelling), **CAN-8** (one predicate) and **INV-LAYER** at once, and it is the same
> read/write asymmetry spec §5.2 already declares for run state. One line in this table now,
> versus an unresolvable conflict four phases later.
| **Modify** `src/phenotypic/_cli/_cli_completion.py` | `publish_image_success` / `valid_image_success` delegate to the record — **and `authorized_measurement_sources` (`:768`), which nobody listed.** See below. |
| **Modify** `src/phenotypic/_cli/_cli_migrate_image.py:567` | **The migrator is a second producer of this schema**, not a stage that runs before one (CAN-7). It calls `publish_image_success` directly. |
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

## Every consumer of the marker surface, and what each maps to

**Measured, not estimated: 136 reads across 20 modules.** This table is the phase's
completeness check — six separate round-1/2/3 findings were all "a reader in a file the plan
never named", so the file list is the artifact most likely to be wrong.

| Module | Reads | Maps to |
|---|---|---|
| `_cli_completion.py` | 11 | record reader/writer (Tasks 1–2) |
| `_cli_stage2_token.py` | 12 | **unchanged** — the token stays a file |
| `_cli_staged_resume.py` | 30 | `stages.stage3`; `classify_staged_image` reads one record + one token probe |
| `_cli_staged_strategy.py` | 10 | `record_stage` / token unlink |
| `_cli_staged_workers.py` | 9 | same |
| `_cli_staged_slurm_worker.py` | 8 | same |
| `_cli_recompile_recovery.py` | 8 | `RECORD_VERSION`, `image_record_path` (P4) |
| `_cli_recompile_tables.py` | 4 | read-back deleted; merge preserves identity (P4) |
| `_cli_recompile_slurm_scripts.py` | 4 | `SUCCESS_MARKER_VERSION` → `RECORD_VERSION` |
| `phenotypicCLI.py` | 11 | gate + the promoter's move to migrate (P7) |
| `_hdf_to_zarr.py` | 6 | record reader — **`sdk_` → `sdk_` after N-3's split** |
| `_cli_migrate.py`, `_cli_migrate_image.py` | 4 | migrate's own publisher (P7) |
| `_slurm_observer.py` | 2 | `stages.stage3` (P6 Task 6) |
| `_io_constants.py`, `sdk_/__init__.py` | 7 | constants + path helpers |
| **`_cli_migrate_manifest.py`** | **3** | **was named in no plan doc** — P7 |
| **`_cli_staged_controller.py`** | **3** | **was named in no plan doc** — `stage3_completion_exists` at `:86` |
| **`_cli_staged_orchestration.py`** | **2** | **was named in no plan doc** — `stage3_completion_exists` at `:277` |

The last three are cross-**process** readers in both environments and cross-**node** readers
under SLURM: the controller and orchestrator ask "did Stage 3 finish for this image?" about
work running in another job entirely. That is precisely why the marker exists and cannot be
replaced by in-memory state — and precisely why all three needed naming.

**Regenerate this table before implementing**, and treat a module appearing in the grep but
not the table as a blocking finding:

```bash
grep -rln 'image_completion_marker_path\|DIR_IMAGE_COMPLETE\|SUCCESS_MARKER_VERSION\|stage2_token\|stage3_completion\|stage2_done\|stage3_complete' src/
```

---

## Interfaces

**Produces:**

```python
# phenotypic.sdk_._image_record   -- READERS and shared vocabulary (N-3)
#
# Here, not in _cli, because P6 Task 0 moves valid_image_success into
# sdk_/_run_state.py and INV-LAYER forbids it importing _cli. Same read/write
# asymmetry spec §5.2 declares for run state.

#: The stage names, as shared constants imported by every writer and reader
#: (CAN-27). `stages` stays an OPEN map (§6.1) -- a future stage is additive --
#: but the names THIS build writes cannot be misspelled, because there is exactly
#: one place they are spelled. This replaces O-2's KNOWN_STAGES + advisory, which
#: could not be built without either breaking INV-LAYER (the advisory is emitted
#: from sdk_, which may not import _cli) or duplicating the set.
STAGE_STAGE1: Final[str] = "stage1"
STAGE_STAGE2: Final[str] = "stage2"
STAGE_STAGE3: Final[str] = "stage3"
STAGE_MEASURED: Final[str] = "measured"

RECORD_VERSION: int = 1

def read_image_record(
    output_dir: Path, dataset: str, image_stem: str
) -> dict[str, object] | None: ...
```

```python
# phenotypic._cli._cli_image_record   -- WRITERS only
#
# from phenotypic.sdk_._image_record import RECORD_VERSION, STAGE_MEASURED, ...

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


def test_the_stage_names_come_from_one_shared_constant(tmp_path):
    """CAN-27, replacing O-2's advisory.

    O-2 proposed a KNOWN_STAGES frozenset in _cli_image_record.py feeding an
    advisory emitted by resolve_run_state in sdk_ -- which INV-LAYER forbids from
    importing _cli. Resolving that meant either duplicating the frozenset (a second
    home for a fact, which the plan's own constraints reject) or breaking the
    invariant P1 Task 1 spends a whole task pinning.

    So CLOSE the typo class rather than reporting it: one module constant, used by
    both the writer and the reader, so a misspelled stage cannot be constructed.
    That is derivation over tracking, and it is strictly less code than the
    advisory it replaces. The map stays OPEN -- a future stage is still additive.
    """
    from phenotypic._cli import _cli_stage2_token, _cli_staged_resume
    from phenotypic._cli._cli_image_record import STAGE_MEASURED, STAGE_STAGE2, STAGE_STAGE3

    assert _cli_stage2_token.STAGE_STAGE2 is STAGE_STAGE2
    assert _cli_staged_resume.STAGE_STAGE3 is STAGE_STAGE3


def test_recording_one_stage_leaves_the_others_untouched(tmp_path):
    """The three collapsed trees were independently writable and must stay so --
    Stage 2 and Stage 3 run in different jobs, on different nodes, minutes apart."""
    from phenotypic._cli._cli_image_record import read_image_record, record_stage

    record_stage(tmp_path, "plate", "a", "stage1", {"at": "t1"})
    record_stage(tmp_path, "plate", "a", "stage2", {"at": "t2"})
    record = read_image_record(tmp_path, "plate", "a")
    assert set(record["stages"]) == {"stage1", "stage2"}


def test_reading_a_corrupt_record_is_none_not_an_error(tmp_path):
    """INV-VERDICT, degrade half."""
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

### The read-modify-write needs real lost-update protection (CAN-6)

**The precedent the first draft cited does not exist.** It claimed `record_stage` is
"read-modify-write under the existing `atomic_write_json` + `pre_replace` revalidation,
exactly as `publish_image_success` does today". Neither half holds:

- `publish_image_success` passes `pre_replace` **only** when
  `expected_artifact_descriptors is not None` (`_cli_completion.py:243-249`), and that
  callback re-validates **artifact descriptors** (`:204-224`). It never re-reads the marker.
- `atomic_write_json` (`sdk_/_atomic_io.py:209-240`) is a temp-write plus `os.replace` with
  an optional pre-rename hook. **No CAS, no re-read, no version check.**

And the collapse creates the hazard that absence of CAS makes real. Today `stage2_done/`,
`stage3_complete/` and `image_complete/` are three files, so three writers cannot lose each
other's writes. After the collapse they are one file with three writers.

`publish_image_success` writes a **complete** dict today (`_cli_completion.py:225-241`). If
`publish_image_record` keeps doing that, it clobbers `stages.stage1`/`stage2` that
`record_stage` wrote. In the SLURM Stage-3 worker the order happens to be
`publish_image_success` → `write_stage3_completion_marker` → `delete_stage2_token`
(`_cli_staged_slurm_worker.py:487-514`), so `stage3` survives **by ordering luck**; the
local paths (`_cli_staged_workers.py:551,560`; `_cli_staged_strategy.py:307,316`) need the
same audit.

**Four rules, each a test:**

1. **`publish_image_record` merges `stages`, never replaces the map.** Its `stages`
   parameter is a *contribution*, unioned with what is on disk.
2. **Every writer takes the lock — including `publish_image_record` (flow-r2 C4).** Rule 1
   makes publishing a read-modify-write, so exempting it reproduces the exact failure rule 1
   exists to prevent, one function over:

   ```
   publish reads {stage2}
                          record_stage("stage3") writes {stage2, stage3}   (holds the lock)
   publish writes {stage2, measured}                                        (does not)
   → stage3 lost
   ```

   With all three writers serialized, `consume_stage` and a concurrent `record_stage` on a
   *different* key are then trivially safe — one file, serialized read-modify-write,
   independent keys.
3. **The lock is a SIBLING file, never the record itself.** `exclusive_path_lock` opens its
   path `"a+b"` (`sdk_/_file_locking.py:41`), so anchoring on
   `images/<ds>/<stem>.json` would **create a zero-byte record** for every image the lock
   ever touches — which then reads as a *corrupt* record rather than an absent one. Every
   such image would degrade toward `incomplete`: INV-VERDICT doing the right thing for
   entirely the wrong reason, on the whole tree. Follow the precedent's shape exactly —
   `aggregate_measurements` anchors on a sibling,
   `phenotypic_cache_dir(output_dir) / ".aggregate_publication.lock"`
   (`_cli_output_manager.py:1556-1558`).
4. **Merging is identity-fenced, and `consume_stage` is idempotent.** Nothing may merge
   forward a stage entry from a superseded `scheduler_epoch`: a stale `stage2` merged into a
   fresh record makes `classify_staged_image` skip Stage 2 while the matching raw `.npy` is
   gone. **Merge only entries whose recorded epoch matches the current one; otherwise
   replace.** This is one rule covering two places — here, and CAN-13's merge-not-overwrite
   conversion in P7, which merges an *old-build* legacy marker into a current record and had
   no fence at all.

```python
def test_publishing_a_record_does_not_clobber_an_earlier_stage(tmp_path):
    """CAN-6. Three writers, one file. Today they are three files and this
    cannot happen; after the collapse it is the default unless publish merges."""
    from phenotypic._cli._cli_image_record import (
        publish_image_record, read_image_record, record_stage,
    )

    record_stage(tmp_path, "plate", "a", "stage2", {"at": "t2"})
    publish_image_record(
        tmp_path, work_id="w", dataset="plate", image_stem="a",
        relative_image_path="a.tif", mode="full",
        stages={"measured": {"at": "t4"}},
        artifacts={}, attempt_id="x", scheduler_epoch="e",
    )
    stages = read_image_record(tmp_path, "plate", "a")["stages"]
    assert set(stages) == {"stage2", "measured"}, (
        "publish replaced the stages map instead of merging into it; "
        "stage2 was lost"
    )


def test_concurrent_stage_writes_do_not_lose_each_other(tmp_path):
    """The lost-update case the collapse creates. Two threads, two stages, one
    file. Without the lock one write wins and the other vanishes silently."""
    import concurrent.futures as cf

    from phenotypic._cli._cli_image_record import read_image_record, record_stage

    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(
            lambda s: record_stage(tmp_path, "plate", "a", s, {"at": s}),
            ["stage1", "stage2"],
        ))
    assert set(read_image_record(tmp_path, "plate", "a")["stages"]) == {"stage1", "stage2"}


def test_consuming_an_absent_stage_is_a_no_op(tmp_path):
    from phenotypic._cli._cli_image_record import consume_stage

    assert consume_stage(tmp_path, "plate", "a", "stage2") is False   # never raises
```

- [ ] **Step 3b: Audit every stage-3 / stage-2-token mutation — there are ≥9, not 4**

The first draft named four sites. The real set (flow-r2 C4):

| Site | What it does |
|---|---|
| `_cli_staged_slurm_worker.py:503` | stage-3 marker |
| `_cli_staged_slurm_worker.py:514` | token consumption |
| `_cli_staged_strategy.py:307,316` | stage-3 + token |
| `_cli_staged_strategy.py:482` | **a fourth token consumption** |
| `_cli_staged_workers.py:551,560` | stage-3 + token — **but see below** |
| `_cli_staged_resume.py:363` | stage-3 marker, with a `legacy_migration=True` kwarg **`record_stage`'s signature has no place for** |
| `_cli_staged_resume.py:392` | inside `clear_downstream_artifacts_for_stage1` |
| `_cli_staged_resume.py:450` | — |

**`_cli_staged_workers.py:551` is guarded by `if work_id is None:`.** So on the normal local
staged path — `work_id` set — neither the stage-3 marker nor the token consumption happens
there at all. Auditing its *ordering* answers a question about a branch that does not run.
**Find where the token is consumed when `work_id` is not None**, and audit that instead.

**`_cli_staged_resume.py:363`'s `legacy_migration=True` needs a decision, not a translation.**
`record_stage(output_dir, dataset, stem, stage, payload)` has nowhere to put it. Either it
belongs in the stage payload (a stage recorded by migration is a fact about that stage), or
the call site is migrate-only and moves to P7 with the rest of the legacy paths. Decide and
say which; do not silently drop the kwarg.

Record at each site whether it publishes before recording, or relies on the merge rule.
**"It works by ordering luck" is a finding, not a design.**

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

**And it must reject a stage-2-only record (CAN-23).** After the collapse, a Stage-2 GPU
worker creates `images/<ds>/<stem>.json` carrying `stages.stage2` and **no artifacts**.
Today the two facts live in two trees, so mistaking a stage-2 token for a success proof is
impossible; after the collapse it is one missing check away.

```python
def test_a_stage2_only_record_is_not_a_success_proof(tmp_path):
    """CAN-23. The collapse merges a Stage-2 token and a success proof into one
    file. A record with no artifacts certifies nothing."""
    from phenotypic._cli._cli_completion import valid_image_success
    from phenotypic._cli._cli_image_record import record_stage

    record_stage(tmp_path, "plate", "a", "stage2", {"at": "t"})
    assert not valid_image_success(
        tmp_path, dataset="plate", image_stem="a", work_id="w"
    )
```

The existing "artifacts must be a non-empty dict" guard (`_cli_completion.py:274-276`)
already does this — **confirm it survives the rewrite** rather than assuming it does, and
keep the test as the thing that says so.

- [ ] **Step 3b: Move `authorized_measurement_sources` with the markers (CAN-22)**

`_cli_completion.py:786` globs `progress_dir/DIR_IMAGE_COMPLETE/*/*.json` and `:838` reads
`image_completion_marker_path`. P3's first draft listed only the two publishers under
`_cli_completion.py`, so this function would keep reading a tree P3 deletes.

**The failure mode is silent and severe.** It returns `{}` — which is a *valid* schema-3
result meaning "no successful measurements yet", not an error — and P4's `finalize_run`
step 1 then writes an **empty master with no exception raised**. A successful-looking run
that discarded every measurement.

```python
def test_authorized_sources_reads_records_not_the_deleted_tree(tmp_path):
    """CAN-22 / GEN-G02. An empty mapping is a VALID result here, so a missed
    migration produces an empty master rather than a traceback."""
    from phenotypic._cli._cli_completion import authorized_measurement_sources

    _publish_two_successful_images(tmp_path)
    sources = authorized_measurement_sources(tmp_path)
    assert sources, "authorized_measurement_sources is still reading image_complete/"
    assert len(sources) == 2
```

- [ ] **Step 3c: Revise the migrator's publisher (CAN-7)**

`_cli_migrate_image.py:567` calls `publish_image_success` directly, so the HDF→Zarr migrator
is an **alternative producer of the same record schema** — not a stage that runs before one.
After P3 it emits records too. Verify that, rather than assuming it: build a fixture through
the **real** migrator and assert the record it writes validates under `valid_image_success`.
A hand-planted fixture cannot catch this class of drift, which is exactly how it survived
the first draft.

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

# CAN-16: the first draft used product([False,True], repeat=4) -- store, s2_token,
# s2_raw, s3_done -- which covers <=16 of >=192 reachable cells and collapses four
# distinct store predicates into one boolean. classify_staged_image's real
# signature is keyword-only (_cli_staged_resume.py:197-206):
#
#     (output_dir, dataset, image, input_root, process_only_layer,
#      markers_required, expected_work_id)
#
# and it branches on:
#   - expected_work_id is None  -> selects between staged_store_matches_work_id /
#     _staged_store_has_work_id and valid_stage1_store / valid_staged_store (:227-232)
#   - markers_required                                              (:258-265)
#   - process_only_layer in {None, "objmap", <other>}     (:222-226, :242-256)
#   - the in-store measurement table                       (:250-256, :258)
#
# For the task this plan itself calls "the risky task", a gate that misses 90% of
# the space does not gate. Parametrize the axes that actually branch.
_STORE_STATES = ["absent", "stage1_only", "matching_work_id", "mismatched_work_id"]
_LAYERS = [None, "objmap", "rgb"]

_COMBOS = [
    (store, s2_token, s2_raw, s3_done, layer, markers_required, expect_work_id)
    for store in _STORE_STATES
    for s2_token in (False, True)
    for s2_raw in (False, True)
    for s3_done in (False, True)
    for layer in _LAYERS
    for markers_required in (False, True)
    for expect_work_id in (False, True)
]

#: Captured from the PRE-CHANGE behaviour in Step 2, as a literal table. Do not
#: derive these by reasoning about what the classifier should do -- the point is to
#: freeze what it DOES, so the collapse is provably behaviour-preserving.
_EXPECTED: dict[tuple[bool, bool, bool, bool], str] = {}   # filled in Step 2


@pytest.mark.parametrize(
    "store,s2_token,s2_raw,s3_done,layer,markers_required,expect_work_id", _COMBOS
)
def test_classification_is_unchanged_by_the_collapse(
    tmp_path, store, s2_token, s2_raw, s3_done, layer, markers_required, expect_work_id
):
    from phenotypic._cli._cli_staged_resume import classify_staged_image

    item = _plant(
        tmp_path, store=store, s2_token=s2_token, s2_raw=s2_raw, s3_done=s3_done,
        layer=layer,
    )
    actual = classify_staged_image(
        tmp_path,
        dataset=item.dataset,
        image=item.image,
        input_root=item.input_root,
        process_only_layer=layer,
        markers_required=markers_required,
        expected_work_id=item.work_id if expect_work_id else None,
    )
    key = (store, s2_token, s2_raw, s3_done, layer, markers_required, expect_work_id)
    assert actual == _EXPECTED[key]
```

**Confirm the real signature before writing `_plant`:**

```bash
sed -n '197,270p' src/phenotypic/_cli/_cli_staged_resume.py
```

Use whatever the function actually takes. `_plant` builds each store state — `absent`,
`stage1_only`, `matching_work_id`, `mismatched_work_id` — and returns a record carrying the
call's arguments.

**If the full product is unreasonably large in wall-clock**, drop axes by *evidence*, not by
convenience: read the branch conditions, find the axes that provably cannot interact, and
write the reduction down as a comment naming the lines that justify it. **Do not silently
sample.**

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
