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

The token is not a *record*. It is a **consumable signal**: Stage 3 clears it by
`unlink()`ing it — atomic on every filesystem, needing **no lock at all**. Folding it into
the shared record replaces that with a **read-modify-write of a JSON file**, and three
things compound in the environment this actually runs in:

*(Precise about the mechanism, because an earlier draft of this section called the token a
"claim" that Stage 3 "takes". It does not: `_cli_staged_slurm_worker.py:487-516` publishes,
writes the stage-3 marker, and only **then** deletes the token. It signals "work available"
and is cleared on completion — it never serialized anything. What actually keeps writers
apart is disjoint work partitioning; see INV-ONEWRITER.)*

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
| **Create** `src/phenotypic/sdk_/_image_record.py` | **Readers and shared vocabulary:** `read_image_record`, `RECORD_VERSION`, the four `STAGE_*` constants, the two `ARTIFACT_KIND_*` constants, and the two `PROVENANCE_*` constants (U-10). Exported from `sdk_/__init__.py` so test snippets can use the package path. ~75 lines. |
| **Create** `src/phenotypic/_cli/_cli_image_record.py` | **Writers only:** `publish_image_record`, `record_stage`, `consume_stage`. Imports the vocabulary from `sdk_`. ~180 lines. |
| **Modify** `src/phenotypic/_cli/_cli_completion.py` | `publish_image_success` / `valid_image_success` delegate to the record — **and `authorized_measurement_sources` (`:768`), which nobody listed.** See below. |
| **Modify** `src/phenotypic/_cli/_cli_migrate_image.py:567` | **The migrator is a second producer of this schema**, not a stage that runs before one (CAN-7). It calls `publish_image_success` directly. |
| **Modify** `src/phenotypic/_cli/_cli_stage2_token.py` | **Only `_STAGE2_DIR` moves to `_io_constants`** (audit S9). `write_stage2_token` / `stage2_token_exists` / `delete_stage2_token` and the `stage2_raw` helpers are **unchanged** — U-9 keeps the token a file with an atomic unlink. |
| **Modify** `src/phenotypic/_cli/_cli_staged_resume.py` | `stage3_completion_exists` / `write_stage3_completion_marker` / `remove_stage3_completion_marker` become `stages.stage3` operations. `classify_staged_image` reads one record **plus one token probe**, and **FLOW-40's raw-presence branch (`:279-283`) survives verbatim**. |
| **Modify** `src/phenotypic/sdk_/_run_state.py` | The deep path reads the record instead of the legacy marker. |
| **Delete** | `DIR_STAGE3_COMPLETE`'s path helper and the inline `"stage3_complete"` literal at `_cli_staged_resume.py:141`. **`DIR_STAGE2_DONE` is NOT deleted** — it moves. |
| **Test** `tests/unit/cli/test_image_record.py` *(new)* | Record schema, stage independence, shared `STAGE_*` constants. |
| **Test** `tests/unit/cli/test_staged_resume_equivalence.py` *(new)* | The gate: resume decisions are unchanged. |

> **The reader/writer split is forced, and discovering it in P3 is much cheaper than in P6
> (N-3).** P6 Task 0 moves `valid_image_success` into `sdk_/_run_state.py` (CAN-8), and after
> P3 that function reads the record. INV-LAYER forbids `sdk_` importing `phenotypic._cli` at
> module scope *or inside a function*, so `sdk_` would have to parse the record and know
> `RECORD_VERSION` and `STAGE_MEASURED` without importing them. The only two outs are
> duplicating the constants — which **CAN-27's own resolution rejects by name**, having just
> replaced `KNOWN_STAGES` for exactly that reason — or re-implementing record parsing in
> `sdk_`, which creates the second reader of the record schema that CAN-8 exists to prevent.
>
> Putting readers plus vocabulary in `sdk_` and writers in `_cli` satisfies **CAN-27** (one
> spelling), **CAN-8** (one predicate) and **INV-LAYER** at once, and it is the same
> read/write asymmetry spec §5.2 already declares for run state.
>
> **Every code snippet in this plan must import readers from `phenotypic.sdk_` and writers
> from `phenotypic._cli` (gen-r3 C4).** An earlier revision changed this table and left
> twelve snippets importing readers from `_cli` — an implementer resolving the resulting
> `ImportError` by majority would put them back in `_cli`, re-creating the exact deadlock
> this split prevents, and discovering it four phases later.

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

# NOT DEFINED HERE -- corrected at cluster 3.1. This block used to declare
# ARTIFACT_KIND_STORE/FILE in this module, reasoning that the kinds "would
# otherwise be spelled at each comparison". The rule is right; the premise is
# false about this tree. They have lived in `sdk_/_io_constants.py` since the
# marker schema was written, and EIGHT modules import them from there --
# `_run_state`, `_cli_completion`, `_cli_migrate_image`, `_hdf_to_zarr`, the
# two recompile modules and `sdk_/__init__`. Declaring them here would create
# the second home the argument exists to prevent.
#
# Import them: `from ._io_constants import ARTIFACT_KIND_FILE, ...`

#: The record's `provenance` values (U-10). Compared in sdk_ by
#: valid_image_success and written in _cli by migrate, so the spelling lives here
#: rather than in either one. ABSENT MEANS FORWARD -- the strict reading is the
#: default, so a writer that forgets the field produces a fenced record, never an
#: accepted-on-sight one.
PROVENANCE_FORWARD: Final[str] = "forward"
PROVENANCE_MIGRATED: Final[str] = "migrated"

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
    lifecycle_epoch: str,
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
`phenotypic._cli._cli_identity`'s minted identity (P2).

> ### ⚠ CORRECTED after the P2 gate — there is no `scheduler_epoch` plumbing
>
> This task used to name `scheduler_epoch` as a parameter of the record writer,
> in three examples, and as a record key — and to say it consumed
> `_cli_identity.scheduler_epoch` plumbing "(P2)". **§5.1's five-token collapse
> was WITHDRAWN, not deferred** (`design.md:323-345`, user-ruled): the CLI
> writers still take **`lifecycle_epoch`**, and `publish_image_success` writes
> that name into every image marker. Written as it stood, this task instructed
> an implementer to pass a keyword that does not exist.
>
> **`scheduler_epoch` is still a real name — on the reader side only.**
> `RunIdentity.scheduler_epoch` (`sdk_/_state_types.py:79`) and
> `_run_state._scheduler_epoch()` are live, and P4/P5 legitimately namespace
> `measurement_shards/<scheduler_epoch>/` by that value. The withdrawal killed
> the *rename of the writers*, not the reader's field. Do not "fix" those.
>
> This record's field is `lifecycle_epoch` because the value it carries already
> has exactly one on-disk name, and giving a new artifact a second one is the
> collapse the withdrawal rejected, arriving from the other direction.

---

## Task 1: The record writer and reader

**Files:**
- Create: `src/phenotypic/sdk_/_image_record.py` — **readers and vocabulary**
- Create: `src/phenotypic/_cli/_cli_image_record.py` — **writers only**
- Modify: `src/phenotypic/sdk_/_run_state.py` — import `STAGE_MEASURED` and
  `PROVENANCE_MIGRATED` instead of re-spelling them
- Modify: `src/phenotypic/sdk_/__init__.py` — export the new vocabulary
- Test: `tests/unit/cli/test_image_record.py`

> ### ⚠ CORRECTED at cluster 3.1 — this block used to name only the `_cli` module
>
> **The File Structure table wins, and it is not close.** That table (top of
> this file) has always said *create both*, with the `sdk_` module first and a
> separate line budget. This block said one. Four independent reasons the table
> is the authority:
>
> 1. **Task 1's own tests import `phenotypic.sdk_._image_record` four times** —
>    more specific than either prose block.
> 2. **INV-LAYER makes the alternative impossible.** `_run_state.py` is `sdk_`
>    and needs `STAGE_MEASURED`; readers in `_cli` leave `sdk_` unable to read a
>    record at all.
> 3. **P6 Task 0 moves `valid_image_success` into `sdk_`**, where it must parse
>    a record whose reader would then be in `_cli`.
> 4. **This plan warns against exactly this**, 180 lines above the block that
>    caused it — *"an implementer resolving the resulting `ImportError` by
>    majority would put them back in `_cli`, re-creating the exact deadlock this
>    split prevents, and discovering it four phases later."*
>
> That warning was written about twelve snippets left importing readers from
> `_cli`; **this block was the same revision's other half**. A document that
> names a trap and then steps into it one edit later is worth flagging on its
> own: treat that revision's output with suspicion rather than rediscovering
> this. Its other tell is mechanical — three of the five snippets below still
> carry a column-0 `import` inside a `def`, which is a syntax error if copied
> and the signature of a machine edit applied without parsing the result.
>
> **`ARTIFACT_KIND_*` is NOT part of the new module**, against the Interfaces
> block's instruction. That block argues the kinds "would otherwise be spelled
> at each comparison" — a sound rule, and false about this tree: they have
> lived in `_io_constants.py` since the marker schema was written and **eight
> modules import them from there**. Following it would create the second home
> the argument exists to prevent. `RECORD_VERSION` is new and does belong in
> the new module.
>
> **`STAGE_MEASURED` resolves the opposite way, from a rule that reads the
> same.** `_run_state.py` had a private `_STAGE_MEASURED` whose own comment said
> it stood *"until P3 replaces the reader"* — so here the new home is the real
> one and the existing copy is a placeholder that says so. `_PROVENANCE_MIGRATED`
> was the same duplication one noun over and is closed with it. The question is
> never *"should this move?"* but *"which home is the real one?"*

- [ ] **Step 1: Write the failing tests**

```python
def test_the_record_is_one_file_carrying_every_stage(tmp_path):
    """Spec §6.1: 'Is this image done?' becomes ONE JSON read instead of one read
    plus up to three is_file() probes across three directory trees."""
    from phenotypic.sdk_._image_record import read_image_record
from phenotypic._cli._cli_image_record import publish_image_record
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
        attempt_id="attempt", lifecycle_epoch="epoch",
    )

    assert image_record_path(tmp_path, "plate", "a").is_file()
    record = read_image_record(tmp_path, "plate", "a")
    assert set(record["stages"]) == {"stage1", "stage2", "stage3", "measured"}
    assert record["artifacts"]["store"]["kind"] == "store"


def test_stages_is_an_open_map(tmp_path):
    """§6.1: `stages` and `artifacts` are open maps -- that is what makes a future
    stage additive rather than a schema break."""
    from phenotypic.sdk_._image_record import read_image_record
from phenotypic._cli._cli_image_record import publish_image_record
    publish_image_record(
        tmp_path, work_id="w", dataset="plate", image_stem="a",
        relative_image_path="a.tif", mode="full",
        stages={"stage1": {"at": "t"}, "some_future_stage": {"at": "t"}},
        artifacts={}, attempt_id="x", lifecycle_epoch="e",
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
    from phenotypic.sdk_._image_record import STAGE_MEASURED, STAGE_STAGE2, STAGE_STAGE3
    assert _cli_stage2_token.STAGE_STAGE2 is STAGE_STAGE2
    assert _cli_staged_resume.STAGE_STAGE3 is STAGE_STAGE3


def test_recording_one_stage_leaves_the_others_untouched(tmp_path):
    """The three collapsed trees were independently writable and must stay so --
    Stage 2 and Stage 3 run in different jobs, on different nodes, minutes apart."""
    from phenotypic.sdk_._image_record import read_image_record
from phenotypic._cli._cli_image_record import record_stage
    record_stage(tmp_path, "plate", "a", "stage1", {"at": "t1"})
    record_stage(tmp_path, "plate", "a", "stage2", {"at": "t2"})
    record = read_image_record(tmp_path, "plate", "a")
    assert set(record["stages"]) == {"stage1", "stage2"}


def test_reading_a_corrupt_record_is_none_not_an_error(tmp_path):
    """INV-VERDICT, degrade half."""
    from phenotypic.sdk_._image_record import read_image_record
    from phenotypic.sdk_ import image_record_path

    path = image_record_path(tmp_path, "plate", "a")
    path.parent.mkdir(parents=True)
    path.write_text("{truncated", encoding="utf-8")
    assert read_image_record(tmp_path, "plate", "a") is None
```

- [ ] **Step 2: Run to verify failure.** Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

The record, per §6.1 minus `backfilled`, plus `provenance` (U-10):

```json
{
  "version": 1,
  "work_id": "…", "dataset": "…", "image_stem": "…",
  "relative_image_path": "…", "mode": "full|process|measure",
  "provenance": "forward",
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
  "attempt_id": "…", "lifecycle_epoch": "…", "completed_at": "…"
}
```

> **`provenance` — two values, and it changes how the record is VERIFIED (U-10).** This is
> not a descriptive label; it is read by `valid_image_success`, so it belongs in the schema
> with the identity fields rather than in a metadata bag.
>
> | Value | Written by | `valid_image_success` behaviour |
> |---|---|---|
> | `"forward"` | every normal run — full, process, measure, recompile | artifacts verify **and** `work_id` matches the recomputed identity |
> | `"migrated"` | **only** `--mode migrate`, on a pre-markers tree | artifacts verify. **No `work_id` comparison** — see U-10 in P7 |
>
> Three rules, each a test in Task 1:
>
> 1. **Absent means `"forward"`.** A record written before this field existed, or by any
>    writer that forgets it, must be fenced — the safe default is the strict one. Read it as
>    `record.get("provenance", "forward")`, never a bare subscript.
> 2. **`"migrated"` is write-once and non-propagating.** Only migrate may write it, and any
>    forward run that rewrites the record replaces it with `"forward"`. That is what makes
>    U-10 self-limiting rather than a permanent hole; P7's
>    `test_the_marking_does_not_survive_reprocessing` is the guard.
> 3. **The relaxation is per-record, never global.** A `valid_image_success` that stops
>    comparing `work_id` for unmarked records strips the fence from every modern tree. P7's
>    `test_an_unmarked_record_is_still_fenced_on_work_id` exists for exactly that mistake.
>
> `ARTIFACT_KIND_*` and the `STAGE_*` constants live in `sdk_/_image_record.py`; put
> `PROVENANCE_FORWARD` / `PROVENANCE_MIGRATED` there too, for the same reason — the value is
> compared in `sdk_` and written in `_cli`, so a spelling that lives in only one of them is
> the duplication N-3 exists to close.

### The read-modify-write needs real lost-update protection (CAN-6)

**The precedent the first draft cited does not exist.** It claimed `record_stage` is
"read-modify-write under the existing `atomic_write_json` + `pre_replace` revalidation,
exactly as `publish_image_success` does today". Neither half holds:

- `publish_image_success` passes `pre_replace` **only** when
  `expected_artifact_descriptors is not None` (`_cli_completion.py:243-249`), and that
  callback re-validates **artifact descriptors** (`:204-224`). It never re-reads the marker.
- `atomic_write_json` (`sdk_/_atomic_io.py:209-240`) is a temp-write plus `os.replace` with
  an optional pre-rename hook. **No CAS, no re-read, no version check.**

And the collapse creates the hazard that absence of CAS makes real. Today
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
2. **NO LOCK — establish the single-writer invariant instead (user ruling, round 3).**

   An earlier draft put an `exclusive_path_lock` on every record writer, on the strength of
   a reviewer-supplied lost-update sequence. **The sequence was never shown to be
   reachable.** Before adding a lock, prove the race exists:

   **INV-ONEWRITER — at most one process writes a given image's record at a time.** Two
   independent mechanisms enforce it, and neither is the token:
   - **Disjoint work partitioning.** Each SLURM array task owns a disjoint image list;
     locally the process pool partitions the same way. One image → one task → one writer.
   - **Stage sequencing.** For a given image, stage 2 cannot start before stage 1's store
     exists, and stage 3 replays stage 2's `.npy`. Concurrency is **across** images, and
     each image has its own file.

     > **This one is weaker than it reads, and must not be cited as the proof (flow-r4).**
     > "The stages never overlap for one image" is a claim about *logical* ordering, not
     > about process concurrency — two processes can both believe they own stage 3 of image
     > X. What actually stops them is `active_job_id` + `scheduler_job_is_active`
     > (`_cli_staged_controller.py:314-322`) and `active_ledger_job_ids`
     > (`phenotypicCLI.py:2016`), both listed below. Cite those and the invariant is
     > checkable; cite sequencing alone and it is unfalsifiable.
     >
     > Also not what separates them: **the recovery controller runs under the *same* epoch
     > as its predecessor** (`_cli_staged_controller.py:279,289` return unless
     > `state["epoch"] == epoch`), so `assert_active_epoch` does not fence a recovery
     > controller from its predecessor's still-running workers. `active_job_id` does all the
     > work, and it appears in no other plan document.

   So `publish_image_record` and `record_stage` do read-merge-`atomic_write_json`, with **no
   lock**. `atomic_write_json` is temp-write + `os.replace`, which is atomic — so a crash
   mid-write leaves the old record intact, which is the property that actually matters here.

   **Correcting the token's role, because the first draft got it wrong.** It is *not* a
   mutex and does not serialize anything. `_cli_staged_slurm_worker.py:487-516` publishes,
   writes the stage-3 marker, and only **then** deletes the token — it is a *"work
   available"* signal cleared on completion. It is still kept as a file (U-9), but for the
   plain reason that `unlink` is atomic and a locked JSON edit is not.

   **A third mechanism covers the two writers that are not per-image at all (gen-r4).**
   `_migrate_legacy_success_evidence` and `refresh_success_markers_after_metadata_migration`
   sweep *every* image in one pass from the driver process. Partitioning says nothing about
   them — there is nothing to partition. What protects them is **mode exclusivity**: both
   run only under `--mode migrate` (U-7), which refuses to operate on a tree with a live
   array and is itself the thing every other writing mode is fenced behind. State that as
   the third mechanism rather than leaving it implicit, because it is the one whose
   precondition can be *removed by a later edit* — the day a sweep writer is called from a
   normal run, INV-ONEWRITER is false and nothing in the partitioning argument notices.

   **What would overturn this**, and what to check before implementing — three paths, not
   two:
   - **A requeued or preempted task racing its own replacement.** SLURM
     `PreemptMode=REQUEUE` with `GraceTime=0` restarts the *same job id* from line one, so
     it is not concurrent with itself — **and, independently, the controller refuses to
     submit while the previous stage job is alive**: `_cli_staged_controller.py:314-322`
     reads `active_job_id` and returns early unless `scheduler_job_is_active(...)` is
     `False`. Note `is not False` — a *failed* scheduler query returns `None` and also
     blocks, so it fails safe. **That fail-safe is now load-bearing:** an optimization that
     treats an unknown scheduler state as inactive breaks INV-ONEWRITER silently.
   - **A `--restart` against a live array.** Refused outright in the CLI **before**
     `clear_machine_state`: `phenotypicCLI.py:2016-2023` exits non-zero when
     `active_ledger_job_ids` is non-empty, and the `restart` branch is at `:2123`.
     `initialize_slurm_lifecycle` (`_cli_slurm_lifecycle.py:104-111`) raises on a conflicting
     generation as well. **Residual, pre-existing and not opened by this change:** the guard
     short-circuits to `active_jobs = []` when `lifecycle["active"] is False`, so a
     deactivated-but-still-draining array is uncovered.
   - **A driver-side sweep writer running while an array is live.** This is the coexistence
     window of P7 Task 5 Step 1c, and the reason its rule — drain or `scancel` before
     migrating — belongs in the refusal message and not only in the docs.

   **Verify all three, and if any admits an overlap, add the lock on that path only** —
   not on all writers.

   ```python
   def test_one_image_has_one_writer(tmp_path):
       """INV-ONEWRITER. The partitioning, not a lock, is what makes the merge safe.
       If this ever fails, the fix is the partitioning -- a lock would only hide it."""
       from phenotypic._cli._cli_staged_slurm import build_shard_assignments

       shards = build_shard_assignments(_six_images(tmp_path), k=3)
       owners = [img for shard in shards for img in shard]
       assert len(owners) == len(set(owners)), "an image appears in two shards"
   ```

   > **The cost argument that used to sit here was false, and is struck (flow-r4).** It
   > claimed a per-image lock would mean "6,000 lock files, `flock` contention on GPFS, and
   > `flock`'s weaker cross-node semantics — pure cost". Every one of those record writes
   > **already** passes through a run-level singleton `flock` on GPFS:
   > `_cli_staged_slurm_worker.py:115-126` builds the `commit_guard` from
   > `generation_publication_guard`, which is
   > `exclusive_path_lock(lifecycle_lock_path(output_dir), timeout=300.0)`
   > (`_cli_slurm_lifecycle.py:204`). That function's own comment records the measured
   > contention — *"up to the account's concurrency cap, observed 60-90+ ... serializes
   > through this single lock"*, with the timeout raised from 60s to 300s because of it.
   >
   > A per-image sibling lock would be **strictly less contended than the lock already in
   > the path**. So the lock was never expensive, and the decision to drop it rests on
   > reachability alone — which is where it belongs, and where it holds. Leaving the false
   > price in the plan would mean the next person to find a reachable race weighs it against
   > a cost that was never real.

   **What it would actually cost, if a race is ever demonstrated: zero new files.**
   `commit_guard` currently wraps **only** the `os.replace` — `sdk_/_atomic_io.py:87` enters
   `publication_commit(commit_guard)` immediately before `pre_replace` and the rename, never
   around the read. So the fix is not a new per-image lock; it is **moving the read inside
   the existing `commit_guard`** in `publish_image_record` / `record_stage`. That guard is
   already threaded through every staged call site, and `record_stage`'s specified signature
   already accepts it. Locally (`commit_guard is None`) the partitioning argument stands
   unaided. Name this before implementing, so a demonstrated race produces a three-line
   change rather than a new mechanism.

   **INV-ONEWRITER is a requirement this change CREATES, not a property it inherits.** Say
   so plainly, because it is what makes the verification worth doing. Today the markers are
   separate files: a duplicate writer writes its own content to its own file, and the worst
   case is a redundant identical write. After the collapse it is one file under a
   read-modify-write over disjoint keys, and that same duplicate writer **drops a key**. The
   change converts a benign duplicate into a silently lost stage.
4. **Merging is fenced on `work_id`, and `consume_stage` is idempotent.**

   > **An earlier draft fenced on the record's epoch field (then spelled
   > `scheduler_epoch`, now `lifecycle_epoch` — see the correction above).
   > That was unimplementable, and it
   > invented a guard that already exists twice (flow-r3 C3).**
   >
   > Unimplementable because `lifecycle_epoch` is a **record-level** field, beside
   > `attempt_id` and `completed_at` and *outside* `stages` — so there is no per-entry epoch
   > to compare, and the rule degenerates to "replace the whole map", which is a different
   > and blunter behaviour. `record_stage`'s signature has no epoch parameter and is
   > `sdk_`-free by design, so it cannot read one. And neither collapsed artifact carries an
   > epoch today: the stage-3 marker's payload is `{version, dataset, image_name, stem,
   > legacy_migration, completed_at}` (`_cli_staged_resume.py:168-178`).
   >
   > Unnecessary because the hazard it named — a stale `stage2` surviving into a fresh
   > record while its raw `.npy` is gone — **is already guarded**:
   > - `_cli_staged_resume.py:279-283` (**FLOW-40**) is an explicit raw-presence branch whose
   >   comment at `:268-278` says it must *not* be folded into `stage2_done`, because doing
   >   so flips a token-present/raw-missing image that has a parquet all the way to
   >   `complete`;
   > - a stale *raw* is caught by the store's `work_id` (`:229-239`) — a mismatch makes
   >   `stage2_store_valid` False, returns `"stage1"`, and clears both token and raw.
   >
   > So the raw `.npy` is not a second condition to add. **It is the condition, it already
   > exists, and it is what the fence was reaching for.**

   Fence the merge on **`work_id`**: entries from a record whose `work_id` differs are not
   merged forward. Keep `consume_stage` idempotent — Stage 3 already tolerates a token
   another attempt consumed — and keep the CAN-13 cross-reference, because P7's
   merge-not-overwrite conversion genuinely does need a `work_id` fence when folding an
   old-build marker into a current record.

   **Name FLOW-40 as load-bearing in Task 3 Step 3.** The collapse rewrites the function
   containing it and deletes the module whose docstring explains the consumable-token
   semantics. CAN-16's `s2_raw` axis covers the branch *behaviourally*, which is good; what
   would be lost is the comment telling a future reader **why** it cannot be simplified.

```python
def test_publishing_a_record_does_not_clobber_an_earlier_stage(tmp_path):
    """CAN-6. Three writers, one file. Today they are three files and this
    cannot happen; after the collapse it is the default unless publish merges."""
    from phenotypic.sdk_._image_record import read_image_record
from phenotypic._cli._cli_image_record import publish_image_record, record_stage

    record_stage(tmp_path, "plate", "a", "stage2", {"at": "t2"})
    publish_image_record(
        tmp_path, work_id="w", dataset="plate", image_stem="a",
        relative_image_path="a.tif", mode="full",
        stages={"measured": {"at": "t4"}},
        artifacts={}, attempt_id="x", lifecycle_epoch="e",
    )
    stages = read_image_record(tmp_path, "plate", "a")["stages"]
    assert set(stages) == {"stage2", "measured"}, (
        "publish replaced the stages map instead of merging into it; "
        "stage2 was lost"
    )


def test_a_crash_mid_write_leaves_the_previous_record_intact(tmp_path):
    """What atomic_write_json buys WITHOUT a lock, and the property that actually
    matters here: temp-write + os.replace, so an interrupted write leaves the old
    record whole rather than a truncated one.

    Note this is NOT the lost-update test an earlier draft had. That test spawned
    two threads writing one record -- a scenario INV-ONEWRITER says cannot occur,
    since work is partitioned per image. Testing an unreachable race would have
    justified a lock the design does not need. If you believe the race IS reachable,
    fix the partitioning; do not add a lock."""
    from phenotypic.sdk_._image_record import read_image_record
from phenotypic._cli._cli_image_record import record_stage
    record_stage(tmp_path, "plate", "a", "stage1", {"at": "t1"})
    before = read_image_record(tmp_path, "plate", "a")
    with _fail_during_write():
        with pytest.raises(OSError):
            record_stage(tmp_path, "plate", "a", "stage2", {"at": "t2"})
    assert read_image_record(tmp_path, "plate", "a") == before


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

> ### This task must ARM the schema gate, in the same commit
>
> **`_schema_shape.SCHEMA_GATE_ARMED` is `False` when it ships from P1, deliberately.**
>
> *(The flag lives in `sdk_/_schema_shape.py`, not `_cli_schema_gate`. P1 moved the detection
> there so `resolve_run_state` could emit §4.3's reader advisory without importing `_cli`,
> and deliberately left **no re-export** — a re-exported copy would read correctly while
> being inert under monkeypatch. **Arm it here, and arm it there.**)*
> At P1 the "legacy" shape and the *current* shape are the same shape — the forward path
> still writes `image_complete/` and `datasets.<ds>.completed`, and does not yet write
> `restart_epoch` — so three of the five signals fire on a tree the build just wrote.
> Verified empirically at P1, not assumed:
>
> ```
> verdict on a tree THIS BUILD just wrote: ConversionVerdict.CONVERT
> ```
>
> An armed gate at P1 would refuse every resume of every mode, and P1 would not leave a
> working tree — failing the rule that each phase ends at a commit passing its own gate.
>
> **This task is what makes the legacy shape actually legacy**, so this task arms it. Set
> `SCHEMA_GATE_ARMED = True` in the same commit that moves the publisher off
> `image_complete/`.
>
> **The ordering is not left to memory.**
> `test_the_gate_is_armed_exactly_when_the_forward_path_stops_writing_markers`
> (`tests/unit/cli/test_schema_gate.py`) calls the **real** `publish_image_success` and
> asserts `marker_written is not SCHEMA_GATE_ARMED`. So it fails if the publisher moves
> without arming, **and** fails if anyone arms it early. Both windows are closed by one
> assertion tied to observable behaviour rather than to a note.
>
> Between P1 and this task the protection is: gate present, unarmed, and CAN-11's hazard —
> a legacy tree silently producing an **empty master with no exception**, because `{}` from
> `authorized_measurement_sources` is a *valid* result — does not arise, since the publisher
> has not moved yet. The moment it moves, the gate must arm. That is the whole window.

> ### And this task must stop `save_processing_state` WRITING the demoted sets
>
> **Without this, signal 3 is permanently un-dischargeable and the change ships an infinite
> refusal loop.** Found during P1 execution; four review rounds missed it, and it is
> invisible today only because the gate is unarmed.
>
> The chain, every link verified in source:
>
> 1. Spec §4.2: `processing_state.datasets.{completed,failed,started}` — **"Deleted from the
>    file."**
> 2. The gate's **signal 3** (`sdk_/_schema_shape.py:246`, moved there in P1) returns `CONVERT` when any
>    dataset entry carries `"completed"`.
> 3. `save_processing_state` (`_cli_state_management.py:79-85`) writes `COMPLETED`, `FAILED`,
>    `ERRORS` and `INITIAL_IMAGES` for every dataset, **unconditionally, on every save.**
> 4. **No task in this plan changes that writer.** A grep for
>    `save_processing_state|DatasetState|ProcessingStateKey` across all eight phase docs
>    returns exactly one hit — an unrelated prose citation in P7 Task 1 about reading
>    `VERSION`.
>
> So P7 Task 3 deletes the fields from an existing file, the next forward run's
> `save_processing_state` puts them straight back, and signal 3 fires again. The steady state
> after P7 is **migrate → run → refused → migrate → run → refused**, escapable only by
> `--overwrite`, which deletes the outputs. That is INV-DISCHARGEABLE violated permanently
> rather than for one shape.
>
> **Do not fix it by dropping signal 3.** That leaves the file carrying the demoted evidence
> §4.2 exists to remove, which is the whole point of the section.
>
> **The fix is writer-side, and it is safe — checked, not assumed.** Stop writing the four
> keys in `save_processing_state`. Nothing needs them on disk:
>
> - `load_processing_state` **re-aggregates from the event log on every load**
>   (`_cli_state_management.py:122-135` → `aggregate_state_from_events`), which is why §4.2
>   calls the stored copy *"a cache of a cache"*.
> - The reader **already tolerates their absence**: `:159` is
>   `ds_dict.get(ProcessingStateKey.COMPLETED, [])` — a `.get` with a default, not a
>   subscript. So an older build reading a newer file degrades to the event log rather than
>   raising.
>
> It belongs in **this task** rather than a later one, because this is the commit that arms
> the gate. Publisher moves, writer stops emitting demoted evidence, gate arms — one
> transition, one commit. Splitting them opens exactly the window the arming test exists to
> close.
>
> ```python
> def test_a_forward_run_does_not_reintroduce_the_demoted_dataset_sets(tmp_path):
>     """The infinite-refusal defect. P7 Task 3 deletes these from the file; if
>     save_processing_state re-adds them, signal 3 fires on the very next run and
>     the tree is refused by every writing mode, forever."""
>     import json
>     from phenotypic._cli._cli_schema_gate import requires_conversion
>
>     root = _run_one_forward_pass(tmp_path)
>     state = json.loads(_state_path(root).read_text())
>     for entry in state["datasets"].values():
>         assert "completed" not in entry, "the demoted sets came back"
>     assert requires_conversion(root) is None, (
>         "a tree this build just wrote classifies CONVERT -- migrate cannot "
>         "discharge it, so every writing mode refuses it permanently"
>     )
> ```

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

> **Two assertions arrive here from Task 1 — not new scope.**
> `test_the_stage_names_come_from_one_shared_constant` was written in Task 1
> and ends:
>
> ```python
> assert _cli_stage2_token.STAGE_STAGE2 is STAGE_STAGE2
> assert _cli_staged_resume.STAGE_STAGE3 is STAGE_STAGE3
> ```
>
> Those two modules do not import those constants until **this task**, so in
> cluster 3.1 they fail for a reason that is not a defect. Task 1 landed its
> own half — the constants have one home, and the writer and reader both use
> it (`test_the_stage_names_have_exactly_one_home`) — and these two move here,
> with the imports they check.
>
> **Keep `is`, do not weaken to `==`.** A shared-*object* check is the whole
> content of CAN-27: `==` passes for two modules that happen to spell
> `"stage2"` identically, which is precisely the state the constant exists to
> make unrepresentable.

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
#: The key is the full seven-axis tuple, matching _COMBOS. An earlier draft typed it
#: as four bools, left over from the product(repeat=4) CAN-16 replaced.
_EXPECTED: dict[
    tuple[str, bool, bool, bool, str | None, bool, bool], str
] = {}   # filled in Step 2


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
actual classification, and write those values into `_EXPECTED` **as a literal table**. Then
re-run: all of them pass. That table is now the contract.

**The count is 384, not sixteen (flow-r4).** `_COMBOS` is a seven-axis product —
4 stores × 2 × 2 × 2 × 3 layers × 2 × 2 — since CAN-16 replaced the original four-boolean
`product(repeat=4)`. Four places in this task still said "sixteen", including Step 4's gate
criterion, which is the one that matters: a hard gate whose success condition names the wrong
number cannot be checked. If you take the evidence-based axis reduction the note below
licenses, **write the reduced number here once it is decided** — not before.

Do **not** derive `_EXPECTED` by reasoning about what the classifier should do. The point is
to freeze what it *does*, so the collapse is provably behaviour-preserving. If one of the
one of them looks wrong, record it in a comment and leave it — fixing a resume bug inside
a refactor makes both unreviewable.

- [ ] **Step 3: Collapse the two trees**

> **FLOW-40 is load-bearing here, and this is the step that rewrites the function it lives
> in (rule 4 at `:457`, carried out).** `_cli_staged_resume.py:279-283` is an explicit
> **raw-presence** branch: it consults the retained Stage-2 `.npy` directly, not the record
> and not the token. The collapse rewrites `classify_staged_image` around it, so it is the
> branch most likely to be "simplified" into a record lookup by someone who has just been
> told the record is the single authority. **It survives verbatim.** The raw array and the
> record answer different questions — the record says a stage was *reported*, the raw array
> says the data to replay is *still there* — and a Stage-3 replay needs the second.

- `write_stage2_token`, `stage2_token_exists`, `delete_stage2_token` — **unchanged**
  (U-9). The token keeps its file and its atomic `unlink`; only `_STAGE2_DIR` moves.
- `write_stage3_completion_marker` → `record_stage(..., "stage3", {...})`
- `stage3_completion_exists` → the same membership test on `"stage3"`
- `remove_stage3_completion_marker` → `consume_stage(..., "stage3")`

Keep the function names — the SLURM observer imports `stage3_completion_exists`
(`_slurm_observer.py`), and renaming it is P6's job, not this task's.

**Move** `_STAGE2_DIR` (`_cli_stage2_token.py:42`) into `_io_constants` — do **not**
delete it (U-9). Delete the inline `"stage3_complete"` literal
(`_cli_staged_resume.py:141`), and their path helpers. `stage2_raw_path`,
`write_stage2_raw`, `load_stage2_raw` and `delete_stage2_raw` are **unchanged**.

- [ ] **Step 4: Re-run the equivalence gate**

Run: `uv run pytest tests/unit/cli/test_staged_resume_equivalence.py -v`
Expected: all 384 PASS, against the table captured in Step 2 — or the reduced count, if
Step 2's evidence-based reduction was taken and recorded.

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
git commit -m "refactor(cli): stage3_complete/ becomes a stages entry

Spec §6.1. Two of the three parallel <ds>/<stem> trees become one record --
stage2_done/ keeps its file and its atomic unlink (U-9), and only _STAGE2_DIR
moves. An earlier draft of this message said three; collapsing the token was the
thing the user overruled.

Two trees, spelled in three places, become one
record. The 384-combination classify_staged_image table was captured from the
pre-change behaviour and is unchanged after -- the resume decisions are the risk
here, not the format."
```
