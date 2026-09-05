# Phase 1 — `sdk_/_run_state.py`: the one reader

**Depends on:** nothing. **Blocks:** P2–P7. *(P0 no longer gates this phase — S-5 was demoted to a step in this phase's own gate, CAN-26.)*

**Spec:** §4 (authority model), §5.2 (function surface), §9 (`RunState`), §9.1 (the
verification cache), §13 (error handling) — as amended by
[D-A, D-B and D-C](OPEN-QUESTIONS.md).

**What this phase does NOT do:** it moves **no consumers**. `_output_consistency.py`,
`RunRegistry`, the SLURM observer and `_snapshot_status.py` all keep working exactly as
they do today. Nothing is deleted. This phase adds a module and its tests, and nothing
else calls it yet. That is deliberate — it is the only phase whose correctness can be
established in isolation, and P6 depends on it being right.

**Read [`OPEN-QUESTIONS.md`](OPEN-QUESTIONS.md) before starting.** D-B decides the cache's
shape, and Q2/Q3/Q4 define the verdict precedence, the `ImageState` type, and the layering
rule this phase implements.

---

## File Structure

| File | Responsibility |
|---|---|
| **Create** `src/phenotypic/sdk_/_state_types.py` | **The four frozen dataclasses only** — `RunIdentity`, `ImageState`, `RunDiagnostics`, `RunState`. No logic, no I/O, no imports from either module below. ~90 lines. |
| **Create** `src/phenotypic/sdk_/_run_state.py` | The four public readers. Imports the types from `_state_types`, the cache from `_verification_cache`, and nothing from `phenotypic._cli`. ~340 lines. |
| **Create** `src/phenotypic/sdk_/_verification_cache.py` | The in-process, identity-fenced cache and its currency rule. Imports `ImageState` from `_state_types`. ~110 lines. |
| **Modify** `src/phenotypic/sdk_/_io_constants.py` | Add `DIR_IMAGE_RECORDS` and `image_record_path()`. |
| **Modify** `src/phenotypic/sdk_/__init__.py` | Export `RunIdentity`, `ImageState`, `RunState`, `RunDiagnostics`, `resolve_run_state`, `run_identity`, `assert_identity_current`, `clear_verification_cache`. **Not** `mint_run_identity` — that is a writer and lives CLI-side (P2). |
| **Create** `tests/unit/sdk_/test_run_state.py` | Verdict matrix, depth behaviour, advisories, the degrade half of INV-VERDICT. |
| **Create** `tests/unit/sdk_/test_verification_cache.py` | INV-VERDICT mutation suite. **The highest-value test in the change** (spec §14). |
| **Create** `tests/unit/sdk_/test_run_state_layering.py` | INV-LAYER. |

> **Three modules, not two — CAN-14's fix created an import cycle (gen-r3 C5).**
> Caching whole `ImageState` objects means `_verification_cache` needs `ImageState`, while
> `_run_state` needs `cached_states`/`remember_states`. Importing each from the other at
> module scope is a cycle, and the obvious escapes are both bad: a deferred import inside a
> function hides the dependency exactly where INV-LAYER's AST test is looking, and moving
> the cache into `_run_state` loses the small auditable surface INV-VERDICT's mutation suite
> targets.
>
> Hoisting the dataclasses into a leaf module resolves it and costs nothing — they are
> frozen data with no behaviour. Dependency order is strictly
> `_state_types` ← `_verification_cache` ← `_run_state`, with no edge back.
>
> **INV-LAYER binds all three**, so the AST test's `_MODULES` tuple covers `_state_types.py`
> too.

**Why two modules and not one:** the cache is the only part of this phase that can produce
a *wrong* answer rather than a slow one. Keeping it in its own file with its own test
module means a reviewer can read all of it at once, and means INV-VERDICT's mutation tests
target a surface small enough to be exhaustive.

**No `verification_cache.json`, and no `VERIFICATION_CACHE_JSON` constant** — unless S-5
returned `ON-DISK TIER NEEDED`. See Task 3 Step 8.

---

## Interfaces

**Produces** (P2–P7 consume these exact signatures):

```python
# phenotypic.sdk_._run_state

@dataclass(frozen=True)
class RunIdentity:
    processing_generation: str      # content-derived from P2 onward
    restart_epoch: int
    scheduler_epoch: str | None
    owner_generation: str | None
    inventory_digest: str
    scientific_config_digest: str
    finalization_input_digest: str
    def digest(self) -> str: ...

@dataclass(frozen=True)
class ImageState:
    work_id: str
    dataset: str
    image_stem: str
    stages: Mapping[str, Mapping[str, object]]   # open map; §6.1 minus `backfilled` (D-A)
    #: Spec §9 annotates `images` as "work_id -> stages + VERDICT". A bool plus an
    #: unread `reason` string was not that. The verdict makes `RunDiagnostics`'s
    #: accepted/verified/failed one-line derivations over `images` instead of
    #: cached counts of a collection the caller already holds (SIMP-R1-09).
    verdict: Literal["verified", "unverified", "failed"]
    reason: str | None

@dataclass(frozen=True)
class RunDiagnostics:
    #: Derived from `images` -- one-line projections, not cached counts. The
    #: demoted trio (manifest_completed, manifest_total, event_log_present) is
    #: DROPPED per U-5: verified zero consumers survive P6, and carrying demoted
    #: evidence into RunState is what keeps it alive as a quasi-evidence surface.
    accepted: int
    verified: int
    failed: int

@dataclass(frozen=True)
class RunState:
    completion: Literal["complete", "incomplete", "failed", "active"]
    identity: RunIdentity
    images: Mapping[str, ImageState]     # work_id -> ImageState
    advisories: tuple[str, ...]
    diagnostics: RunDiagnostics
    depth: Literal["shallow", "deep"]
    verified_at: datetime | None

def run_identity(output_dir: Path) -> RunIdentity | None: ...
def assert_identity_current(output_dir: Path, identity: RunIdentity) -> None: ...
def resolve_run_state(
    output_dir: Path, *, depth: Literal["shallow", "deep"] = "deep"
) -> RunState: ...
def finalization_input_object(output_dir: Path) -> dict[str, object]: ...
```

```python
# phenotypic.sdk_._verification_cache

@dataclass(frozen=True)
class CachedVerification:
    #: The WHOLE ImageState, not just a verdict (CAN-14). RunState.images needs
    #: `stages` per image, so an entry carrying only (work_id, verdict, stats)
    #: forces shallow to re-read every record JSON -- the ~10^4 marker-reads half
    #: of audit §4's cost, left in place. Caching the state is what makes §9.2's
    #: claim true rather than half-true.
    state: ImageState
    stat_tuples: Mapping[str, tuple[int, int]]   # relative path -> (size, mtime_ns)

#: Per-output, identity-fenced, replaced WHOLESALE on identity change (CAN-28).
#: No LRU, no _MAX_ENTRIES, no eviction policy: entries are already fenced, so the
#: only usable ones are those under the current identity for the current output.
#: Wholesale replacement is inherently bounded by "images in the runs currently
#: being asked about" -- a TIGHTER bound than a 200k-entry LRU, and it follows
#: from the fence instead of being a policy layered on top of it.
def cached_states(
    output_dir: Path, identity_digest: str
) -> Mapping[str, CachedVerification] | None: ...
def remember_states(
    output_dir: Path, identity_digest: str,
    entries: Mapping[str, CachedVerification],
) -> None: ...
def entry_is_still_current(output_dir: Path, entry: CachedVerification) -> bool: ...
def clear_verification_cache(output_dir: Path | None = None) -> None: ...
def tracked_output_count() -> int: ...   # test-only introspection
```

**Consumes:** nothing from this plan. From the existing tree:
`phenotypic.sdk_.resolve_processing_state_path`, `phenotypic_cache_dir`, `progress_dir`,
`image_completion_marker_path`, `run_completion_marker_path`,
`aggregate_publication_marker_path`, `STORE_ROOT_JSON`, `source_image_stem`.

---

## Task 1: Constants, package wiring, and INV-LAYER

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py`
- Create: `src/phenotypic/sdk_/_run_state.py` (stub)
- Create: `tests/unit/sdk_/test_run_state_layering.py`

- [ ] **Step 1: Write the failing layering test**

`tests/unit/sdk_/test_run_state_layering.py`:

```python
"""INV-LAYER: sdk_/_run_state.py never reaches into phenotypic._cli.

Spec §5.2 calls the read/write asymmetry "structural, not conventional": _run_state.py
exports only readers, so the GUI cannot reach a publish_* function. Structure that
nothing tests is convention with extra steps -- the GUI's 25 private phenotypic._cli
imports across 9 modules are what that looks like at scale (audit §7).

A LAZY import inside a function body is also a violation, not a loophole: it would
drag back load_processing_state's event-log replay, which spec §4.2 deletes. See
OPEN-QUESTIONS Q4. The AST walk catches both forms.
"""

from __future__ import annotations

import ast
from pathlib import Path

import phenotypic.sdk_._run_state as run_state
import phenotypic.sdk_._verification_cache as verification_cache

import phenotypic.sdk_._state_types as state_types

# All THREE modules, not two (gen-r3 C5): the dataclasses live in their own leaf
# module to break the _run_state <-> _verification_cache cycle, and INV-LAYER binds
# the leaf exactly as it binds the other two.
_MODULES = (
    Path(state_types.__file__),
    Path(verification_cache.__file__),
    Path(run_state.__file__),
)


def test_neither_module_ever_names_the_cli_package():
    offenders: list[str] = []
    for source in _MODULES:
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(("phenotypic._cli", "._cli")):
                    offenders.append(f"{source.name}:{node.lineno} from {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("phenotypic._cli"):
                        offenders.append(
                            f"{source.name}:{node.lineno} import {alias.name}"
                        )
    assert not offenders, (
        "INV-LAYER: the run-state readers must not import phenotypic._cli. "
        f"Found: {offenders}"
    )


def test_run_state_exports_no_writer():
    forbidden = ("publish", "write", "mint", "append", "save", "delete")
    exported = getattr(run_state, "__all__", None)
    assert exported is not None, "_run_state.py must declare __all__"
    bad = [
        name
        for name in exported
        if any(name.lower().startswith(prefix) for prefix in forbidden)
    ]
    assert not bad, f"_run_state.py exports writers: {bad}"
```

- [ ] **Step 2: Run it to see it fail**

Run: `uv run pytest tests/unit/sdk_/test_run_state_layering.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'phenotypic.sdk_._run_state'`.

- [ ] **Step 3: Add the constants**

In `src/phenotypic/sdk_/_io_constants.py`, beside `DIR_IMAGE_COMPLETE` (line 663):

```python
#: One record per image, replacing ``image_complete/`` and
#: ``stage3_complete/`` (spec §6.1). ``stage2_raw/`` stays a separate tree: it is
#: bulk replay data, not a record.
DIR_IMAGE_RECORDS: Final[str] = "images"
```

and beside `image_completion_marker_path` (line 1952):

```python
def image_record_path(output_dir: Path, dataset: str, image_stem: str) -> Path:
    """Return ``<output>/.phenotypic/progress/images/<dataset>/<stem>.json``."""
    return progress_dir(output_dir) / DIR_IMAGE_RECORDS / dataset / f"{image_stem}.json"
```

**Do not add `VERIFICATION_CACHE_JSON` or `verification_cache_path()`.** Under D-B the
cache is in process and has no path. Add them only under Task 3 Step 8, and only if S-5
said so.

- [ ] **Step 4: Create all THREE module stubs — `__all__` carries only what exists**

> **`_run_state.py`'s `__all__` starts with the four dataclass names, not eight.** The four
> function names are added by the tasks that define them (Task 3 Step 4, Task 4 Step 3, Task
> 5 Step 3). Listing a name `_run_state.py` does not yet define is `F822` under ruff's
> default `F` set — four real errors, measured at the end of Task 2, not a hypothetical.

`sdk_/_state_types.py` first — it is the leaf the other two import from, and it
carries only frozen dataclasses (gen-r3 C5). Then `_verification_cache.py`, then
`_run_state.py`. Dependency order is `_state_types ← _verification_cache ←
_run_state`, with no edge back; creating them in that order makes a cycle
impossible to introduce by accident.

`src/phenotypic/sdk_/_run_state.py`:

```python
"""Read-only resolution of a run's completion state.

**Readers only.** Spec §5.2 makes the read/write asymmetry structural: every function
that *publishes* state stays in :mod:`phenotypic._cli`, so a GUI import of this module
cannot reach one. INV-LAYER (``tests/unit/sdk_/test_run_state_layering.py``) enforces
both halves -- no ``phenotypic._cli`` import, and no writer in ``__all__``.

This module reads ``processing_state.json`` as plain JSON and never replays the event
log. That is possible because spec §4.2 demotes the event log out of the evidence set
and deletes ``processing_state.datasets.{completed,failed,started}`` from the file:
what remains that a verdict depends on is ``config.work_ids`` and the digests, all
literal JSON fields. See OPEN-QUESTIONS Q4.
"""

from __future__ import annotations

__all__ = [
    "ImageState",
    "RunDiagnostics",
    "RunIdentity",
    "RunState",
    "assert_identity_current",
    "finalization_input_object",
    "resolve_run_state",
    "run_identity",
]
```

`src/phenotypic/sdk_/_verification_cache.py` — header only for now; the body lands in
Task 3.

- [ ] **Step 5: Run the layering test — it must pass**

Run: `uv run pytest tests/unit/sdk_/test_run_state_layering.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Prove the test can fail**

Temporarily add `from phenotypic._cli._cli_completion import valid_image_success` inside a
function body in `_run_state.py`, re-run, confirm
`test_neither_module_ever_names_the_cli_package` FAILS, then remove it. Repeat with a
module-scope import. **A test that has never been seen to fail is not evidence**, and the
lazy-import form is the one a future contributor will actually reach for.

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/sdk_/_io_constants.py src/phenotypic/sdk_/_run_state.py \
        src/phenotypic/sdk_/_verification_cache.py \
        tests/unit/sdk_/test_run_state_layering.py
git commit -m "feat(sdk): add the run-state module boundary and pin INV-LAYER

Spec §5.2. The modules are stubs; the test is the point -- 'structural, not
conventional' needs something that fails. Both the module-scope and the lazy
in-function import forms were confirmed to trip it."
```

---

## Task 2: The state types

**Files:**
- Modify: `src/phenotypic/sdk_/_state_types.py` — **the dataclasses live here**, not in
  `_run_state.py` (gen-r3 C5)
- Test: `tests/unit/sdk_/test_run_state.py`

Resolves [Q3](OPEN-QUESTIONS.md#q3-imagestate-is-used-but-never-defined).

- [ ] **Step 1: Write the failing test**

```python
import dataclasses


def test_the_demoted_sources_live_only_under_diagnostics():
    """Spec §9: a predicate reaching into state.diagnostics is visibly wrong.

    This does not stop someone writing `if state.diagnostics.verified ==
    state.diagnostics.accepted`, but it does pin WHERE the demoted evidence lives.
    manifest counts and the event log were evidence; §4.2 demoted them. If they
    reappear as top-level RunState fields, the demotion has been undone.
    """
    from phenotypic.sdk_ import RunDiagnostics, RunState

    top = {f.name for f in dataclasses.fields(RunState)}
    assert top == {
        "completion",
        "identity",
        "images",
        "advisories",
        "diagnostics",
        "depth",
        "verified_at",
    }
    diag = {f.name for f in dataclasses.fields(RunDiagnostics)}
    assert diag == {"accepted", "verified", "failed"}, (
        "U-5 dropped manifest_completed/manifest_total/event_log_present after "
        "verifying zero consumers survive P6. Carrying demoted evidence into "
        "RunState is what keeps it alive as a quasi-evidence surface."
    )


def test_image_state_stages_carry_no_backfilled_key():
    """D-A: per-store metadata is written at promote time, so there is no
    backfill stage. `stages` stays an open map, so re-adding one later is
    additive -- but nothing in this phase may write or read that key."""
    from phenotypic.sdk_ import ImageState

    state = ImageState(
        work_id="w", dataset="d", image_stem="s",
        stages={"measured": {"at": "2026-09-03T00:00:00Z"}},
        verdict="verified", reason=None,
    )
    assert "backfilled" not in state.stages
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/unit/sdk_/test_run_state.py -v`
Expected: FAIL — `ImportError: cannot import name 'RunState'`.

- [ ] **Step 3: Add the dataclasses to `_state_types.py`**

Not to `_run_state.py` (gen-r3 C5). Imports at module top: `hashlib`, `json`, `dataclass`,
`datetime`, `Path`, `Literal`, `Mapping`):

```python
Completion = Literal["complete", "incomplete", "failed", "active"]
Depth = Literal["shallow", "deep"]


@dataclass(frozen=True)
class RunIdentity:
    """The identity of one run configuration (see the note below on counts) (spec §5.1).

    Two tokens are content-derived (`work_id`, `processing_generation`) -- anything still saying THREE predates U-4, so resume and fencing are emergent rather
    than bookkeeping: two invocations with the same inputs mint the same identity
    without either having read the other's state.
    """

    processing_generation: str
    restart_epoch: int
    scheduler_epoch: str | None
    owner_generation: str | None
    inventory_digest: str
    scientific_config_digest: str
    finalization_input_digest: str

    def digest(self) -> str:
        """Return a stable digest of the fencing-relevant tokens.

        ``scheduler_epoch`` and ``owner_generation`` are excluded: they are
        liveness facts, not configuration, and folding them in would discard the
        verification cache every time a job is submitted against unchanged work.
        """
        payload = {
            "processing_generation": self.processing_generation,
            "restart_epoch": self.restart_epoch,
            "inventory_digest": self.inventory_digest,
            "scientific_config_digest": self.scientific_config_digest,
            "finalization_input_digest": self.finalization_input_digest,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class ImageState:
    """One image's stages and whether its declared artifacts still match disk.

    ``stages`` is the open map from spec §6.1 -- ``stage1``/``stage2``/``stage3``/
    ``measured`` today, more later. Nothing here enumerates its keys; a caller
    asking "did stage 3 run?" reads ``"stage3" in state.stages``, which is what
    makes a future stage additive rather than a schema break.

    Under D-A there is no ``backfilled`` stage: per-store metadata is written in
    the store's original promote, so there is nothing to record having happened
    afterwards.
    """

    work_id: str
    dataset: str
    image_stem: str
    stages: Mapping[str, Mapping[str, object]]
    #: Spec §9 annotates `images` as "work_id -> stages + VERDICT". A bool plus an
    #: unread `reason` was not that (SIMP-R1-09).
    verdict: Literal["verified", "unverified", "failed"]
    reason: str | None = None


@dataclass(frozen=True)
class RunDiagnostics:
    """Counts derived from ``images``. **Nothing branches on these** (§4.2, §9).

    One-line projections over ``ImageState.verdict``, not cached counts of a
    collection the caller already holds.

    ``manifest.json``'s counts and the event log's presence were in an earlier
    draft of this dataclass and are **dropped** (U-5): verified zero consumers
    survive P6, and carrying demoted evidence into ``RunState`` is what keeps it
    alive as a quasi-evidence surface. The files remain on disk for a human
    debugging a run.
    """

    accepted: int
    verified: int
    failed: int


@dataclass(frozen=True)
class RunState:
    """The single answer to "is this run done?" (spec §4.3, §9)."""

    completion: Completion
    identity: RunIdentity
    images: Mapping[str, ImageState]
    advisories: tuple[str, ...]
    diagnostics: RunDiagnostics
    depth: Depth
    verified_at: datetime | None = None
```

- [ ] **Step 4: Export the FOUR DATACLASSES from `sdk_/__init__.py` — not the functions**

Add to the import block and to `__all__`, in alphabetical position: `ImageState`,
`RunDiagnostics`, `RunIdentity`, `RunState`. **Stop there.**

> **The nine-name list this step used to carry breaks `import phenotypic` for the whole
> repo (found during execution, P1-C1).** Five of the nine do not exist until later tasks —
> `clear_verification_cache` is Task 3 Step 4; `run_identity`, `assert_identity_current` and
> `finalization_input_object` are Task 4 Step 3; `resolve_run_state` is Task 5 Step 3. A
> `from ._run_state import resolve_run_state` against a module that does not yet define it
> is an `ImportError` at **package import time**, so following the old list literally left
> the tree unimportable at the end of Task 2 — every test in the repo, not only this
> phase's.
>
> **Each function is exported by the task that implements it.** That is what keeps the tree
> green at every commit, which is the phase-gate contract and the property that makes a
> bisect land on a phase boundary rather than mid-rewrite.
>
> **`__all__` grows incrementally too, and for a harder reason than the exports.** An earlier
> version of this note said keeping all eight names was harmless because nothing validates
> `__all__` at import. **Import is not the only validator: `F822` (*undefined name in
> `__all__`*) is in ruff's default `F` set, which is the group this repo actually enables.**
> Measured, not predicted — the four function names produced four `F822` errors against
> `_run_state.py` at the end of Task 2.
>
> So each task adds its name to `__all__` and to `sdk_/__init__.py` **together**, in the task
> that defines it.
>
> The one argument for keeping `__all__` whole was that `test_run_state_exports_no_writer`
> reads it. That argument does not survive inspection: **P6 Task 7 Step 3 deletes that test**
> (CAN-31 — it asserts on `__all__` name *prefixes*, so a writer named `record_stage` passes
> it). Contorting the module to satisfy a test the plan already schedules for deletion is the
> wrong way round.

**Where `__init__.py` imports the dataclasses from: `._run_state`, not `._state_types`.**
`_run_state.py` re-exports the four from the leaf module. One extra hop, and it buys the
thing gen-r3 C5 did *not* intend to change: the spec's Interfaces block declares the surface
as `# phenotypic.sdk_._run_state`, and moving the definitions into a leaf was a
cycle-breaking mechanism, not an interface change. The re-export keeps that declaration true
and leaves `_state_types` a private implementation detail that a later refactor can move
again without touching a single consumer.

- [ ] **Step 5: Run the test.** Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/sdk_/_run_state.py src/phenotypic/sdk_/__init__.py \
        tests/unit/sdk_/test_run_state.py
git commit -m "feat(sdk): define RunIdentity, ImageState, RunDiagnostics, RunState

Spec §9. ImageState is defined here because the spec uses it and never declares it
(OPEN-QUESTIONS Q3). stages carries no `backfilled` key (D-A)."
```

---

## Task 3: The in-process verification cache and INV-VERDICT

**This is the highest-value task in the phase.** Spec §14 names its mutation tests as the
highest-value test in the whole change.

**Files:**
- Modify: `src/phenotypic/sdk_/_verification_cache.py`
- Test: `tests/unit/sdk_/test_verification_cache.py`

**Shape, per D-B and CAN-28:** a module-level map of
`output_dir → (identity_digest, dict[work_id, CachedVerification])`, **replaced wholesale
when the identity changes**. No LRU, no `_MAX_ENTRIES`, no eviction policy.

Entries are already identity-fenced, so the only entries that can ever be *used* are those
under the current identity for the current output; everything else is dead weight an LRU
would exist only to sweep. Wholesale replacement is bounded by "images in the runs
currently being asked about" — a **tighter** bound than a 200k-entry LRU — and it *follows
from* the fence rather than being a policy layered on top of it.

Bounded is still not optional: audit §5, S22 and S23 are all findings about unbounded
module globals in this codebase (`LocalRunner._instances`, `_terminal_job_cache`,
`_LAST_DUMPED`), and shipping a fourth while deleting the machinery that made the first
three necessary would be indefensible. The fence is what makes it bounded.

- [ ] **Step 1: Write the INV-VERDICT mutation suite first**

`tests/unit/sdk_/test_verification_cache.py`:

```python
"""INV-VERDICT: nothing may improve a verdict except a successful deep verification.

Spec §9.1 states the invariant and §14 calls these the highest-value tests in the
change. The current design's whole point is that it never trusts a cache, so the
correctness argument for introducing one has to be executable.

Each test corrupts the cache a different way and asserts the verdict never IMPROVES.
A cache that degrades to today's behaviour is correct; a cache that turns an
incomplete run into a complete one is the bug this file exists to prevent shipping.

D-B moved the cache in-process, so the "forge the file" cases here forge the dict.
The invariant is about what a cache may CAUSE, not where it lives, so it binds
identically. If S-5 added an on-disk tier, Step 8 adds the JSON-corruption cases.
"""

from __future__ import annotations

import pytest

from phenotypic.sdk_ import clear_verification_cache, resolve_run_state


@pytest.fixture(autouse=True)
def _isolate_cache():
    """A module-level cache is shared state; a leaked entry makes the next test lie."""
    clear_verification_cache()
    yield
    clear_verification_cache()


@pytest.fixture
def complete_run(tmp_path):
    from tests._output_layout import build_complete_run

    return build_complete_run(tmp_path)


@pytest.fixture
def incomplete_run(tmp_path):
    from tests._output_layout import build_incomplete_run

    return build_incomplete_run(tmp_path)


def test_a_forged_entry_cannot_manufacture_complete(incomplete_run):
    """The adversarial case: every cached state claims verdict="verified"."""
    import dataclasses

    from phenotypic.sdk_ import run_identity
    from phenotypic.sdk_._verification_cache import (
        CachedVerification,
        remember_states,
    )

    baseline = resolve_run_state(incomplete_run, depth="deep").completion
    assert baseline != "complete"

    identity = run_identity(incomplete_run)
    forged = {
        work_id: CachedVerification(
            state=dataclasses.replace(image, verdict="verified"),
            stat_tuples={},
        )
        for work_id, image in resolve_run_state(incomplete_run, depth="deep").images.items()
    }
    remember_states(incomplete_run, identity.digest(), forged)

    after = resolve_run_state(incomplete_run, depth="shallow").completion
    assert after == baseline, (
        "a forged cache changed the verdict; a positive verdict must never come "
        "from a cache entry alone -- INV-VERDICT"
    )


def test_a_stale_identity_never_matches(complete_run):
    from phenotypic.sdk_._verification_cache import cached_states

    state = resolve_run_state(complete_run, depth="deep")
    work_id = next(iter(state.images))
    assert cached_states(complete_run, state.identity.digest())[work_id]
    assert cached_states(complete_run, "0" * 64) is None, (
        "an entry minted under a different identity was reused"
    )


def test_a_tampered_artifact_falls_through_even_with_a_warm_cache(complete_run):
    """The stat tuple is the currency check; content still decides."""
    resolve_run_state(complete_run, depth="deep")
    overlay = next(complete_run.rglob("overlays/**/*.png"), None)
    if overlay is None:
        pytest.skip("fixture has no overlay artifact")
    overlay.write_bytes(overlay.read_bytes() + b"tamper")

    assert resolve_run_state(complete_run, depth="shallow").completion != "complete"


def test_ctime_is_not_part_of_the_currency_check(complete_run):
    """Audit S3 / spec §9.1: ctime_ns moves on chmod, chown, hardlink and rsync -a,
    all routine on GPFS. size + mtime_ns already covers every write the publication
    contract makes, so a chmod must invalidate nothing."""
    warm = resolve_run_state(complete_run, depth="deep")
    for path in complete_run.rglob("*.png"):
        path.chmod(0o644)
    state = resolve_run_state(complete_run, depth="shallow")
    assert state.completion == warm.completion
    assert state.depth == "shallow", (
        "a chmod invalidated the cache -- ctime_ns has leaked into the currency "
        "check that audit S3 removed"
    )


def test_an_identity_change_replaces_the_output_entry_wholesale(complete_run):
    """CAN-28. The bound comes from the FENCE, not from an eviction policy: only
    entries under the current identity for the current output are usable, so a new
    identity replaces the whole per-output map rather than accumulating alongside it.

    Audit §5, S22 and S23 are three findings about unbounded module globals in this
    codebase. This is bounded by 'images in the runs currently being asked about' --
    tighter than the 200k-entry LRU an earlier draft proposed, and with no policy to
    get wrong.
    """
    from phenotypic.sdk_._verification_cache import cached_states, tracked_output_count

    resolve_run_state(complete_run, depth="deep")
    first = _identity_digest(complete_run)
    assert cached_states(complete_run, first) is not None

    _edit_pipeline_json(complete_run)          # new identity
    resolve_run_state(complete_run, depth="deep")

    assert cached_states(complete_run, first) is None, "stale identity entries survived"
    assert tracked_output_count() == 1, "the output accumulated a second entry"


def test_clear_scoped_to_one_output_does_not_clear_another(tmp_path):
    from tests._output_layout import build_complete_run

    a = build_complete_run(tmp_path / "a")
    b = build_complete_run(tmp_path / "b")
    resolve_run_state(a, depth="deep")
    resolve_run_state(b, depth="deep")
    clear_verification_cache(a)
    assert resolve_run_state(a, depth="shallow").depth == "deep"
    assert resolve_run_state(b, depth="shallow").depth == "shallow"
```

- [ ] **Step 2: Add the two fixture builders**

`tests/_output_layout.py` already holds `write_master` / `write_measurements_mirror` (used
by `tests/e2e/gui/test_heatmap_tab.py`). Add beside them:

```python
def build_complete_run(tmp_path: Path) -> Path:
    """Return an output tree whose deep verdict is `complete`.

    Deliberately minimal: two images in one dataset, each with a promoted store, an
    embedded measurement table and an overlay; a success marker for each; an
    aggregate proof; a run proof. Anything more makes a failing test hard to read.

    Built by calling the REAL publishers, never by hand-writing JSON: a fixture that
    hand-writes the format under test keeps passing after the format changes, which
    is the failure mode this whole plan is about. P3 swaps `publish_image_success`
    for the record writer and this function does not change.
    """
    from phenotypic._cli._cli_completion import (
        publish_aggregate_snapshot,
        publish_image_success,
    )

    output = tmp_path / "run"
    for stem in ("a", "b"):
        store = _promote_minimal_store(output, dataset="plate", stem=stem)
        overlay = _write_overlay(output, dataset="plate", stem=stem)
        publish_image_success(
            output,
            work_id=f"work-{stem}",
            dataset="plate",
            relative_image_path=f"{stem}.tif",
            image_stem=stem,
            mode="full",
            attempt_id=f"attempt-{stem}",
            lifecycle_epoch="local",       # `scheduler_epoch` from P2 Task 4 onward
            artifacts={"store": store, "overlay": overlay},
        )
    _write_processing_state(output, work_ids={"plate": {"a.tif": "work-a", "b.tif": "work-b"}})
    _write_master_and_mirror(output)       # tests._output_layout helpers, already present
    publish_aggregate_snapshot(output)
    _publish_run_completion(output)
    return output


def build_incomplete_run(tmp_path: Path) -> Path:
    """The same tree with the second image's success marker removed.

    Removing the MARKER rather than the artifacts is deliberate: it is the state a
    run killed between promoting a store and publishing its proof actually leaves,
    and the one the verdict ladder has to call `incomplete` rather than `complete`.
    """
    output = build_complete_run(tmp_path)
    image_completion_marker_path(output, "plate", "b").unlink()
    return output
```

`_promote_minimal_store`, `_write_overlay`, `_write_processing_state` and
`_publish_run_completion` are small local helpers in the same module; `_write_master_and_mirror`
wraps the existing `write_master` / `write_measurements_mirror` already in
`tests/_output_layout.py`.

- [ ] **Step 3: Run the suite to verify it fails**

Run: `uv run pytest tests/unit/sdk_/test_verification_cache.py -v`
Expected: FAIL — `ImportError: cannot import name 'clear_verification_cache'`.

- [ ] **Step 4: Implement the cache**

`src/phenotypic/sdk_/_verification_cache.py`:

```python
"""The in-process verification cache.

Audit **S1** -- the finding spec §9.1 responds to -- proposed a *process-level* cache
keyed on the marker file's stat tuple. §9.1 escalated that to a file on disk. Decision
D-B (OPEN-QUESTIONS) took it back: every cadence the audit measured is a repeated call
inside ONE long-lived process (the observer's 2 s tick, the viewer's 5-10 s poll,
``OutputRoot.discover``'s double read, ``OutputMutationGuard``'s double read), and an
in-memory cache serves all of them without adding a tracked artifact to a design whose
purpose is removing them.

INVARIANT (INV-VERDICT) -- **nothing may improve a verdict except a successful deep
verification.** No function here returns a verdict to a caller that has not deep-verified.
``entry_is_still_current`` answers one question: *may a previously deep-verified result
stand?* The caller supplies the verdict from its own deep pass, and a ``True`` here merely
licenses skipping that pass next time. A stale, replaced or forged entry therefore degrades
to today's behaviour and never past it.

``ctime_ns`` is deliberately absent from the stat tuple (audit S3): it moves on ``chmod``,
ownership change, hardlink and ``rsync -a``, all routine on a shared filesystem, and
``size`` + ``mtime_ns`` already covers every write the publication contract makes.

**Bounded by the fence, not by a policy** (CAN-28). ``LocalRunner._instances``,
``_terminal_job_cache`` and ``_LAST_DUMPED`` are three unbounded module globals this
codebase already carries as audit findings (§5, S22, S23), so a fourth is not acceptable.
But an LRU would be a *second* mechanism on top of one that already bounds this: entries
are identity-fenced, so only those under the current identity for the current output can
ever be used. Replacing each output's map wholesale on an identity change is therefore both
tighter and simpler than evicting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ._state_types import ImageState   # NOT ._run_state -- see below

#: output_dir (resolved, as str) -> (identity_digest, {work_id: CachedVerification})
_CACHE: dict[str, tuple[str, dict[str, "CachedVerification"]]] = {}


@dataclass(frozen=True)
class CachedVerification:
    #: The WHOLE ImageState (CAN-14). RunState.images needs `stages` per image, so
    #: an entry carrying only a verdict forces shallow to re-read every record --
    #: the ~10^4 marker-reads half of audit §4's cost, left in place.
    state: ImageState
    stat_tuples: Mapping[str, tuple[int, int]]
```

Four rules the implementation must obey, each enforced by one of the tests above:

1. `cached_states` returns `None` unless the stored `identity_digest` matches **exactly**.
   There is no partial trust: an identity change discards that output's whole map.
2. `entry_is_still_current` returns `False` for an empty `stat_tuples` map, a missing file,
   an `OSError`, or any changed `(size, mtime_ns)`. It never raises.
3. `remember_states` **replaces** the output's entry wholesale under the new identity
   digest. There is no eviction path to get wrong.
4. `clear_verification_cache(output_dir=None)` clears **that output's** entry, or all of
   them when `output_dir` is `None`. P2 wires it to `clear_machine_state`.

- [ ] **Step 5: Run the suite.** Expected: PASS (6 passed).

- [ ] **Step 6: Prove each test can fail (spec §14; project test-integrity rule)**

Reintroduce one at a time and confirm the named test fails:

| Bug to reintroduce | Test that must fail |
|---|---|
| `cached_states` ignores `identity_digest` | `test_a_stale_identity_never_matches` |
| `resolve_run_state` returns the cached verdict without re-stat | `test_a_forged_entry_cannot_manufacture_complete` |
| add `st_ctime_ns` to the stat tuple | `test_ctime_is_not_part_of_the_currency_check` |
| `remember_states` merges into the existing map instead of replacing it | `test_an_identity_change_replaces_the_output_entry_wholesale` |
| `clear_verification_cache` ignores `output_dir` and clears everything | `test_clear_scoped_to_one_output_does_not_clear_another` |

Record the five confirmations in the commit body. **A mutation not demonstrated is a
mutation not tested.**

- [ ] **Step 7: `mypy` and `ruff` on the two new modules**

```bash
uv run mypy src/phenotypic/sdk_/_verification_cache.py
uv run ruff check --fix src/phenotypic/sdk_/_verification_cache.py \
  tests/unit/sdk_/test_verification_cache.py
```

- [ ] **Step 8: Measure cold start, then decide on the on-disk tier (CAN-26)**

> ### ⚠ The threshold is stated in the wrong units, so it does not name a decidable condition
>
> **"Under 30 s projected at N=6000" cannot be evaluated**, because cold-deep cost is not a
> function of image count. It is a function of **marker-bound bytes**, and on a real tree it
> is dominated by one artifact class the image count says nothing about.
>
> Measured on the 6,657-image Maresca tree — each image binds three artifacts:
>
> | Artifact | Size | Share of the pass |
> |---|---|---|
> | overlay PNG | ~11 MB | **~73 GB — 99%** |
> | embedded measurements table | ~66 KB | ~440 MB |
> | store root `zarr.json` | ~2 KB | ~13 MB |
>
> Two runs with **identical `N`** differ by orders of magnitude depending on overlay
> resolution — and `--mode process` writes no overlays at all, so its cold pass is ~1% of a
> full run's at the same image count. `N` is the wrong independent variable.
>
> **This is the defect CAN-26 was reaching for and did not finish.** It correctly identified
> that S-5 measured an *approximation* of the shipped predicate, and moved the measurement
> here to fix that — but left the threshold expressed in the approximation's own units. The
> measurement was corrected; the condition it feeds was not.
>
> **Restate the question in bytes before asking it:** *"a cold deep pass hashes ~74 GB on a
> 6,657-image tree, and the in-process cache reduces steady state to a stat sweep — should a
> fresh process pay that once?"* That is decidable. *"Is it under 30 s at N=6000"* is not.
>
> **P1 ships no `VERIFICATION_CACHE_JSON`** — not because the number came in under
> threshold (it does not; ~74 GB is structurally above 30 s at any plausible bandwidth) but
> because the decision as posed cannot be made correctly from this measurement, and P1's gate
> is not where a new tracked artifact should enter a design whose thesis is removing them.
> D-B removed it deliberately; re-adding it on a threshold stated in the wrong units is
> weaker ground than D-B stood on.

> ### If the tier ever does ship, it changes an invariant — say so out loud
>
> The plan specifies the six corruption cases. It does not name the thing D-B's reasoning
> actually turned on:
>
> **An in-process entry can only have been written by a deep pass in this process. An
> on-disk entry cannot.**
>
> INV-VERDICT's argument is that *an entry only licenses skipping a pass the caller already
> performed*. On disk, "the caller" becomes **some** process — possibly an older build,
> possibly mid-write, possibly another user. That is a real weakening of the invariant, it is
> defensible, and it must be stated in the module docstring rather than inherited silently
> from wording written for the in-process case.

> ### Flagged for P4, not P1: the dominant cost is a *derived display artifact*
>
> ~99% of the cold pass is re-hashing overlay PNGs — a rendering, not a measurement. The
> overlay is marker-bound today, so **a completion verdict is gated on 73 GB of PNG**.
>
> Whether a display artifact belongs in a content proof at all is a real question, and
> answering it would cut the cold pass by ~99% **with no new tracked file** — a better
> outcome than the on-disk tier, reached by removing something rather than adding something.
>
> **It is not P1's to decide**: it changes what the proofs certify, which is P4's subject.
> Recorded here so P4 meets it with the measurement already in hand.

S-5 moved here from P0, because it measured a hand-rolled *approximation* of the
marker-hashing loop; now that `resolve_run_state` exists, measure **the real
predicate**:

```bash
# cold process, real tree on GPFS, N as large as available
uv run python -c "
import time; from pathlib import Path
from phenotypic.sdk_ import resolve_run_state
t=time.perf_counter(); s=resolve_run_state(Path('<tree>'), depth='deep')
print(f'cold_deep={time.perf_counter()-t:.1f}s images={len(s.images)}')"
```

**Under 30 s projected at N=6000 → in-process only. No new file ships.** That is the
expected outcome and the one D-B prefers; say so in the commit body, so a later reader
can tell the tier was *measured away* rather than forgotten.

Only if it exceeds 30 s:

1. Add `VERIFICATION_CACHE_JSON` and `verification_cache_path()` to `_io_constants.py`,
   with a docstring naming the measured cold-start number that justifies them.
2. Add `load_verification_cache` / `store_verification_cache` to
   `_verification_cache.py`, backing the in-process LRU. The file carries a top-level
   `identity_digest`; a mismatch discards **the whole file**, not the mismatched entries.
3. `store_verification_cache` wraps `atomic_write_json` in `try/except OSError` and
   swallows — spec §9.1's "best-effort … must never turn an unwritable output into an
   error".
4. Add these cases to the mutation suite, each asserting the verdict never improves:
   `truncated` (`"{"`), `null`, `wrong-type` (`"[]"`), `binary-garbage`, `deleted`, and
   an `unwritable cache directory` case that asserts `resolve_run_state` returns
   `depth="deep"` rather than raising.
5. Prove each of those can fail too.

- [ ] **Step 9: Commit**

```bash
git add src/phenotypic/sdk_/_verification_cache.py \
        tests/unit/sdk_/test_verification_cache.py tests/_output_layout.py
git commit -m "feat(sdk): add the in-process verification cache and pin INV-VERDICT

Spec §9.1, §14, as amended by D-B: audit S1 asked for a process-level cache and
that is what this is. S-5 measured cold start at <N>s, so no on-disk tier ships.

Each of the five mutations below was reintroduced and the named test confirmed to
fail:
  identity ignored          -> test_a_stale_identity_never_matches
  verdict straight from cache -> test_a_forged_entry_cannot_manufacture_complete
  ctime in the stat tuple   -> test_ctime_is_not_part_of_the_currency_check
  eviction removed          -> test_an_identity_change_replaces_the_output_entry_wholesale
  clear() unscoped          -> test_clear_scoped_to_one_output_does_not_clear_another"
```

---

## Task 3b: `requires_conversion` — the schema gate (CAN-11)

**Files:**
- Create: `src/phenotypic/sdk_/_schema_shape.py` — **detection**: `requires_conversion`,
  `ConversionVerdict`, `describe_required_conversion`, `describe_conversion_advisory`,
  `SCHEMA_GATE_ARMED`
- Create: `src/phenotypic/_cli/_cli_schema_gate.py` — **refusal**: `refuse_unconverted_schema`
- Modify: `src/phenotypic/phenotypicCLI.py`
- Test: `tests/unit/cli/test_schema_gate.py` *(new)*

> **Two modules, not one — INV-LAYER forces the split, and it was found in execution.**
> This task originally created only the `_cli` module. But §4.3's reader half puts a
> `needs_conversion` advisory on `resolve_run_state`, which lives in `sdk_` — and
> **INV-LAYER forbids `sdk_` importing `phenotypic._cli`**, with an AST test that fails the
> build. A detector reachable from only one side of that line cannot serve both consumers.
>
> So detection moved to `sdk_/_schema_shape.py` and `_cli_schema_gate.py` became the thin
> refusing wrapper over it. The CLI raises `SystemExit`; the SDK returns an advisory string;
> neither re-implements the signal list, which stays specified once in
> [phase-7-migrate-mode.md](phase-7-migrate-mode.md) Task 1.
>
> **`SCHEMA_GATE_ARMED` is why the advisory does not fire before the gate exists.** The flag
> lives beside the detector and both consumers ask it, so *"is the gate live?"* has one
> answer rather than one per caller.

**Moved here from P7 Task 1, and the move is the point.** P3 Task 2 is an explicit clean
break — `publish_image_success` writes the new record and `valid_image_success` reads it —
while the gate that refuses a legacy tree originally did not land until **four phases
later**. In between, on a legacy tree, `valid_image_success` returns `False` for every
image, `authorized_measurement_sources` returns `{}` — a *valid* schema-3 result meaning
"nothing succeeded yet" — and P4's `finalize_run` writes an **empty master with no
exception raised**. A successful-looking run that silently discarded every measurement.
P4 Step 5, P5 and P6 all specify runs against real trees, so this is not a bisect concern.

It has no dependency on the rest of P7: it only detects the old shape and errors.

**This is a CLI-side writer-adjacent module, so it does not live in `sdk_`** — but it is
built in P1 because P1 is what precedes P3. That is the only reason it is here.

The full specification — the detection signals, the two outcomes, the shapes that classify
without obvious behaviour, and every test — is written once in
[phase-7-migrate-mode.md](phase-7-migrate-mode.md) Task 1. **Build it from there; do not
restate it here.** A second copy of the signal list is exactly the duplication this change
exists to remove, and CAN-4 is what happens when a fact gets a second home.

- [ ] **Step 1: Build P7 Task 1 now, in full, including Steps 3b and 3c.**
- [ ] **Step 2: Confirm the ordering it protects.**

```python
def test_a_legacy_tree_is_refused_before_the_clean_break_can_empty_it(tmp_path):
    """CAN-11. Without the gate, P3's clean break turns a legacy tree into an
    empty master rather than an error -- because `{}` from
    authorized_measurement_sources is a VALID result, not a failure."""
    import pytest

    _build_legacy_tree(tmp_path)
    with pytest.raises(SystemExit, match="--mode migrate"):
        _invoke_cli(mode="full", output=tmp_path)
```

> ### `_invoke_cli` — defined here, because it is used 27 times and defined nowhere
>
> A grep for `def _invoke_cli` across the whole plan and spec returns **nothing**, while the
> name appears **27 times** (1 here, 26 in P7). The plan-writing rule against *"references
> to types, functions, or methods not defined in any task"* was missed for this one. It is
> defined by this task, in `tests/unit/cli/test_schema_gate.py`, and P7 imports it.
>
> **It wraps `CliRunner().invoke` and returns click's `Result`:**
>
> ```python
> def _invoke_cli(mode, output, cli_inputs, *, dry_run=False) -> Result:
>     pipeline, images = cli_inputs
>     args = ["--mode", mode, "--output", str(output)]
>     if mode in ("full", "measure", "process"):
>         args += ["--pipeline", str(pipeline)]
>     if mode in ("full", "process"):
>         args += ["--input", str(images)]
>     if mode == "process":
>         args += ["--layer", "rgb"]
>     if dry_run:
>         args.append("--dry-run")
>     return CliRunner().invoke(phenotypic_cli, args)
> ```
>
> **Assert on `result.exit_code` and `result.output`. Never
> `pytest.raises(SystemExit, match=...)`.** Two independent reasons, both measured on click
> 8.3.1 rather than reasoned about:
>
> 1. **Through `CliRunner`, `pytest.raises` cannot fire.** A `SystemExit` *is* raised — click
>    converts the `UsageError` into one — but `invoke` **catches** it and stores it on
>    `result.exception`. (An earlier report said no `SystemExit` is raised at all; that was
>    wrong about the mechanism and right about the outcome. If a call site ever invokes the
>    command function *directly*, `SystemExit` does propagate and `pytest.raises` works
>    there.)
> 2. **`match=` fails regardless of which form is used.** `str(SystemExit(2))` is `"2"`, not
>    the message. So a test written as `pytest.raises(SystemExit, match="--mode migrate")`
>    cannot fail for its own reason even where the exception propagates — it is matching
>    against an exit code.
>
> Reason 2 is the one to carry: an implementer who reads only *"it is fine when called
> directly"* will still write a test that cannot fail for the reason it exists.
>
> **This affects P7 Task 1 Step 1 (three tests), P7 Tasks 3 and 5, and Step 2 below.**

- [ ] **Step 3: Commit**

The code and its commit message are P7 Task 1's; this task only moves *when* it is built.
Commit under that message rather than inventing a second one:

```bash
git add -A src/phenotypic/sdk_ tests/unit/sdk_
git commit -m "feat(sdk): requires_conversion, built in P1 rather than P7 (CAN-11)

Spec §11.1. Same code and tests as P7 Task 1 -- only the build order moves. P3's
clean break turns an unconverted tree into an EMPTY master rather than an error,
because {} from authorized_measurement_sources is a valid result and not a
failure. The gate has to exist before the break, so it is built here."
```

**Do not restate U-6 here.** There is no `BELOW_FLOOR` outcome and no version floor —
`requires_conversion` keys on the pre-markers *shape* (schema `2.0.0` with no `work_ids`).
P7 Task 1 is the single home for that reasoning.

---

## Task 4: `run_identity` and `assert_identity_current`

**Files:**
- Modify: `src/phenotypic/sdk_/_run_state.py`
- Test: `tests/unit/sdk_/test_run_state.py`

In this phase `run_identity` reads the tokens that **already exist** in
`processing_state.json`: `processing_generation` (still a `uuid4().hex` until P2),
`pipeline_sha256`, `metadata_sha256`, `include_dataset_column`, `no_qc`, `work_ids`.
`restart_epoch` defaults to `0` when absent — P2 introduces the writer. This is what keeps
P1 independently landable: the module works on today's trees.

- [ ] **Step 1: Write the failing tests**

```python
def test_run_identity_is_none_for_a_tree_with_no_processing_state(tmp_path):
    from phenotypic.sdk_ import run_identity

    assert run_identity(tmp_path) is None


def test_run_identity_reads_todays_state_file(complete_run):
    """P1 lands before P2, so it must work on a uuid4 processing_generation and a
    state file with no restart_epoch field at all."""
    from phenotypic.sdk_ import run_identity

    identity = run_identity(complete_run)
    assert identity is not None
    assert identity.restart_epoch == 0
    assert len(identity.inventory_digest) == 64
    assert identity.finalization_input_digest


def test_assert_identity_current_names_the_field_that_changed(complete_run):
    """D6: a config change still hard-errors with the SPECIFIC mismatch. A generic
    'identity changed' would make the content-derived generation a worse diagnostic
    than the uuid it replaces."""
    import dataclasses

    import pytest

    from phenotypic.sdk_ import assert_identity_current, run_identity

    identity = run_identity(complete_run)
    stale = dataclasses.replace(identity, inventory_digest="0" * 64)
    with pytest.raises(RuntimeError, match="inventory_digest"):
        assert_identity_current(complete_run, stale)


def test_finalization_input_digest_is_a_versioned_object(complete_run):
    """Spec §5.5: adding a field is a schema_version bump handled by the reader, not
    a second tree migration."""
    from phenotypic.sdk_ import finalization_input_object

    obj = finalization_input_object(complete_run)
    assert obj["schema_version"] == 1
    assert set(obj) == {
        "schema_version",
        "metadata_sha256",
        "include_dataset_column",
        "no_qc",
    }


def test_scheduler_epoch_and_owner_generation_are_not_in_the_digest(complete_run):
    """They are liveness facts, not configuration. Folding them in would discard the
    cache every time a job is submitted against unchanged work."""
    import dataclasses

    from phenotypic.sdk_ import run_identity

    identity = run_identity(complete_run)
    moved = dataclasses.replace(
        identity, scheduler_epoch="other", owner_generation="other"
    )
    assert moved.digest() == identity.digest()
```

- [ ] **Step 2: Run to verify failure.** Expected: `ImportError` / `AttributeError`.

- [ ] **Step 3: Implement**

```python
def _read_state_config(output_dir: Path) -> dict[str, object] | None:
    """Return ``processing_state.json``'s ``config`` block, or ``None``.

    Plain JSON, no event-log replay -- see the module docstring and OPEN-QUESTIONS
    Q4. Every failure returns ``None`` rather than raising (INV-VERDICT's degrade half).
    """
    from ._io_constants import resolve_processing_state_path

    try:
        raw = json.loads(
            resolve_processing_state_path(output_dir).read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError):
        return None
    config = raw.get("config") if isinstance(raw, dict) else None
    return config if isinstance(config, dict) else None


def finalization_input_object(output_dir: Path) -> dict[str, object]:
    """Return the versioned finalization-input object (spec §5.5)."""
    config = _read_state_config(output_dir) or {}
    return {
        "schema_version": 1,
        "metadata_sha256": config.get("metadata_sha256"),
        "include_dataset_column": config.get("include_dataset_column"),
        "no_qc": config.get("no_qc", False),
    }
```

`run_identity` composes those plus `_canonical_digest(config.get("work_ids", {}))`.
`assert_identity_current` compares field by field and raises
`RuntimeError(f"{field} changed: expected {a!r}, found {b!r}")` on the **first** mismatch.

### Hoist `_canonical_digest` — do not add a third copy (CAN-29)

It currently lives in two places, `_cli_completion.py:861` and `_cli_failure_tracker.py`,
and INV-LAYER forbids `_run_state.py` importing either.

An earlier draft added a **third** private copy here, pinned the three against each other
with a keeper test, and had P6 collapse them and delete the test again. **Hoist it once
instead:** move it to `sdk_/_digests.py` and have both CLI sites import it. That is less
total work than add-three-then-collapse, it deletes the keeper test and P6's step, and it
removes two copies rather than adding one.

It does not breach P1's "moves no consumers" rule. That rule is about *state* consumers —
the reason P1's correctness can be established in isolation. A pure function with no I/O
cannot change a verdict.

```python
def test_the_hoisted_digest_matches_what_both_cli_sites_produced(tmp_path):
    """One-shot proof of the hoist. Delete this test in the same commit -- it
    exists only while there are copies that could disagree, and after the hoist
    there are none.

    ensure_ascii=False matters (DF-19): the existing two use it, and a digest that
    disagrees with itself on a non-ASCII dataset name would invalidate every proof
    written by the other half of the code.
    """
    from phenotypic.sdk_._digests import canonical_digest

    probe = {"b": [1, 2, {"c": None}], "a": "é"}
    assert canonical_digest(probe) == (
        "the value both CLI copies produced for this probe, recorded here at hoist time"
    )
```

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/_run_state.py tests/unit/sdk_/test_run_state.py
git commit -m "feat(sdk): add run_identity and assert_identity_current

Spec §5.2, §5.5. Reads processing_state.json as plain JSON with no event-log replay
-- the property INV-LAYER protects. Third _canonical_digest copy is pinned against
the two CLI ones until P6 collapses them."
```

---

## Task 5: `resolve_run_state` — the deep path

> ### Task 5 inherits two tests from Task 3 — this is a deliverable, not a courtesy
>
> **Found during execution: Task 3's headline mutation cannot be proved inside Task 3.**
> Its suite is written against `resolve_run_state` and `run_identity`, which do not exist
> until Tasks 5 and 4. Four of Task 3's five mutations are provable at the cache surface
> alone; the fifth — *a forged cache entry cannot manufacture `complete`* — is irreducibly
> an integration test, because the invariant is about what the cache may **cause in the
> resolver**, and the resolver is what would contain the bug.
>
> So these two move here, and **Task 5 is not done until they are green and mutation-proved**:
>
> | Test | The mutation it must catch |
> |---|---|
> | `test_a_forged_entry_cannot_manufacture_complete` | `resolve_run_state` returns a verdict straight from the cache without re-verifying |
> | `test_a_tampered_artifact_falls_through_even_with_a_warm_cache` | a warm entry suppresses the deep re-check |
> | **`test_a_warm_cache_is_actually_used`** | **the cache is written and never read** |
>
> **That third one is not symmetry — it closes a hole the other two cannot see.** Every
> INV-VERDICT test proves the cache cannot make an answer *better* than the truth. **None of
> them would notice a cache that nothing ever reads.** Such a cache degrades perfectly,
> passes the entire mutation suite, and ships audit S1's finding — the double walk, the
> re-hash on every poll — completely unfixed. A suite that can only detect the cache being
> too generous is a suite that cannot detect the cache being absent.
>
> Assert the observable consequence, not an internal counter: a second `resolve_run_state`
> call at `depth="shallow"` over an unchanged tree must not re-hash. Count `valid_image_success`
> invocations, or stat calls, or wall-clock against a deliberately slow artifact — whichever
> the implementation makes honest, but it must fail if the read path is removed.

> ### Three obligations `resolve_run_state` inherits from the cache — read before writing it
>
> Found while building Task 3, and cheaper to know now than to discover as red tests:
>
> 1. **`depth="deep"` must populate the cache** through `remember_states`, keyed on
>    `run_identity(output_dir).digest()`.
> 2. **`depth="shallow"` must fall back to a full deep pass** when the cache is cold or
>    stale, and must report the depth it *actually performed* in `RunState.depth` — not the
>    depth it was asked for.
> 3. **A store is fenced by its root `zarr.json`, never by the store directory.**
>    `entry_is_still_current` fails closed on anything that is not a regular file, because a
>    directory's `mtime_ns` tracks only its own entries — rewriting
>    `tables/measurements/table.parquet` inside a store leaves the store root's mtime
>    untouched. That is spec §0's *"a valid root does not imply unchanged contents"* reached
>    through the cache instead of through the root. **A resolver that records the store
>    directory will silently never hit the cache** — a failure whose only symptom is lost
>    performance, which is why it would survive indefinitely.
>
> Task 3's agent writes both and hands them over as a diff, so this task pastes rather than
> reinvents. **Do not treat a handed-over test as pre-verified** — run its mutation here.
>
> **Why this is written down rather than left to a message between agents:** "assigned to a
> later task" is precisely how this plan lost things four times in review. If the tests are
> absent when the phase gate runs, the spec-adherence reviewer's category B (*planned, not
> done*) is what catches it — but the plan has to say they were planned for that to work.

> ### Fixtures come from Task 3 — do not write your own
>
> `build_complete_run` and `build_incomplete_run` live in `tests/_output_layout.py` and are
> **built by Task 3 Step 2**. Tasks 4, 5 and 6 consume them as the `complete_run` /
> `incomplete_run` fixtures. A second pair under different names in the same file is a merge
> conflict and two subtly different notions of "complete".

**Files:**
- Modify: `src/phenotypic/sdk_/_run_state.py`
- Test: `tests/unit/sdk_/test_run_state.py`

Until P3 lands, the deep path reads **today's** `image_complete/<ds>/<stem>.json` markers
and maps them into `ImageState` with a single-key `stages = {"measured": {...}}`. P3
replaces the reader, not the caller. **Say so in a comment**, or the next reader will think
the single-key `stages` is the design.

- [ ] **Step 1: Write the failing verdict-matrix test**

```python
@pytest.mark.parametrize(
    "mutate,expected",
    [
        pytest.param(lambda d: None, "complete", id="untouched"),
        pytest.param(_remove_one_image_marker, "incomplete", id="missing-marker"),
        pytest.param(_remove_run_proof, "incomplete", id="no-run-proof"),
        pytest.param(_add_terminal_failure, "failed", id="terminal-failure"),
        pytest.param(_mark_slurm_lifecycle_active, "active", id="live-worker"),
        pytest.param(_corrupt_run_proof, "incomplete", id="unreadable-proof"),
        pytest.param(_corrupt_processing_state, "incomplete", id="unreadable-state"),
    ],
)
def test_the_verdict_matrix(complete_run, mutate, expected):
    mutate(complete_run)
    assert resolve_run_state(complete_run, depth="deep").completion == expected


def test_a_live_worker_does_not_mask_a_valid_run_proof(complete_run):
    """Q2: `complete` outranks `active`.

    A run proof covers the CURRENT inventory, so a live worker at that point is
    either fenced by restart_epoch or is a new invocation that has already changed
    the inventory -- in which case rule 1 does not fire and this is not the case
    being decided.
    """
    _mark_slurm_lifecycle_active(complete_run)
    assert resolve_run_state(complete_run, depth="deep").completion == "complete"


def test_an_active_run_outranks_a_stale_terminal_failure(incomplete_run):
    """Q2 rule 2 over rule 3: a failure from a previous attempt must not mask an
    attempt currently retrying it."""
    _add_terminal_failure(incomplete_run)
    _mark_slurm_lifecycle_active(incomplete_run)
    assert resolve_run_state(incomplete_run, depth="deep").completion == "active"


def test_an_unconverted_h5_is_an_advisory_and_never_a_gate(complete_run):
    """Spec §4.3: half-migrated trees contribute an advisory -- informational, not a
    gate. Today they reach `contradictory` and flag the whole output read-only for a
    reason the user cannot act on."""
    hdf = complete_run / "results" / "plate" / "hdf"
    hdf.mkdir(parents=True, exist_ok=True)
    (hdf / "legacy.h5").write_bytes(b"\x89HDF\r\n\x1a\n")
    state = resolve_run_state(complete_run, depth="deep")
    assert state.completion == "complete"
    assert any("migrate" in advisory for advisory in state.advisories)


def test_a_store_built_against_older_metadata_is_an_advisory(complete_run):
    """D-A: stores keep the metadata snapshot they were built against, and each
    store's phenotypic.metadata.snapshot_sha256 records which one. When that differs
    from the run's current metadata_sha256, say so -- derived from what the store
    already carries, never tracked, and never a gate."""
    _rewrite_metadata_csv(complete_run, b"Metadata_Well,Metadata_Strain\nA1,new\n")
    state = resolve_run_state(complete_run, depth="deep")
    assert state.completion == "complete"
    assert any("metadata" in advisory for advisory in state.advisories)


def test_an_empty_directory_is_incomplete_and_never_raises(tmp_path):
    """INV-VERDICT, degrade half. An unmanaged directory is not an error -- the GUI points at
    arbitrary paths and must get an answer, not a traceback."""
    state = resolve_run_state(tmp_path, depth="deep")
    assert state.completion == "incomplete"
    assert state.images == {}
```

- [ ] **Step 2: Run to verify failure.** Expected: `AttributeError: resolve_run_state`.

- [ ] **Step 3: Implement the deep path**

```python
def resolve_run_state(output_dir: Path, *, depth: Depth = "deep") -> RunState:
    """Resolve one run's completion state (spec §4.3, §9).

    Verdict precedence is total and ordered (OPEN-QUESTIONS Q2):
    ``complete`` > ``active`` > ``failed`` > ``incomplete``. First match wins.

    ``depth="shallow"`` re-stats the in-process cache's recorded tuples and falls
    through to ``deep`` for any image that is absent from the cache, moved, minted
    under a different identity, or unreadable. It **never** yields a positive verdict
    from a cache entry alone -- INV-VERDICT.

    Args:
        output_dir: Run output root. May be any directory, including one this
            package has never written to.
        depth: ``"deep"`` re-verifies every declared artifact's content.
            ``"shallow"`` re-stats. See spec §9's caller/depth table.

    Returns:
        A :class:`RunState`. **Never raises** for an unreadable or absent tree --
        every parse failure degrades toward ``incomplete`` (INV-VERDICT's degrade half).
    """
```

Body, in order — each step is one of the four verdicts and nothing else:

1. `identity = run_identity(output_dir)`; on `None`, return an `incomplete` `RunState`
   with advisory `"no processing state"` and empty `images`.
2. Build `images` by walking `config.get("work_ids", {})` — **the accepted-inventory
   authority**, never a directory listing. A `work_id` with no marker is an unverified
   `ImageState`, not an absent one; that is what makes "which images are missing?"
   answerable.

   > **Every `config` read in this function uses `.get`, never a subscript (flow-r4 N-4).**
   > The docstring says *"Never raises"*, and `config["work_ids"]` breaks that promise on
   > the one tree it most matters for: **U-6's detection signal is the absence of the
   > `work_ids` key**, so a pre-markers tree raises `KeyError` from inside the function
   > whose job is to classify it. `resolve_run_state` must return `incomplete` with an
   > advisory there, so that `requires_conversion` can do its work — a traceback is not a
   > verdict. The same applies to `pipeline_sha256`: a missing key means the comparison is
   > **unsatisfiable**, hence `False`, never an exception.
3. `completion` by the Q2 ladder. **Rule 1 asks both of §4.3's clauses and the full
   five-way comparison** — see below.
4. `advisories`, each derived and each non-gating:
   - `datasets_needing_migration()` — the existing shared predicate — for unconverted `.h5`
   - any store whose root `attributes.phenotypic.metadata_table.snapshot_sha256` ≠ the run's
     current `metadata_sha256` (D-A). **Two corrections from round 2, both load-bearing:**
     - **The key is `metadata_table`, not `metadata`.** `phenotypic.metadata` is already
       taken by `{protected, public, imported}` image-metadata sections
       (`sdk_/ngff_.py:569-580`). P4 Task 2 adds the new key.
     - **It must be read from the root, not the Parquet.** Today the digest lives only as
       Arrow schema metadata on `table.parquet` (`ngff_.py:95`). Reading it there would mean
       opening a **Parquet footer per store** from `sdk_` on the deep path — not "one
       attribute read from a value the store already carries", and a cost §9.2's numbers do
       not include. P4 Task 2 mirrors it into the root so this stays a plain `zarr.json` read.
     - Until P4 lands, `--mode measure`'s in-place branch can leave the root stale
       (`_measurement_tables.py:284-290`); the docstring says so.
   - **any image whose record carries `provenance: "migrated"` (U-10)** — the configuration
     fence is unavailable for it. One advisory naming the affected images, not one per
     image.

     > **This advisory is the visible half of U-10 and is not optional.** The ruling accepts
     > a real weakening — a migrated image is verified on artifact validity alone, with no
     > `work_id` comparison — on the grounds that it is **visible** rather than silent.
     > Delete the advisory and the trade the user agreed to is no longer the trade being
     > made: what remains is an invisible hole. Word it so a reader learns three things: the
     > fence is unavailable, which images, and that reprocessing clears it.
     >
     > It is an **advisory, not a gate** — same as every other entry here. A migrated tree is
     > usable; the point is that the user knows what it is.
5. `diagnostics` — counts derived from `images` only. **`manifest_completed`,
   `manifest_total` and `event_log_present` are dropped** (U-5): verified zero consumers
   survive P6, and carrying demoted evidence into `RunState` is what keeps it alive as a
   quasi-evidence surface.

### Rule 1 in full — CAN-4 and U-2

The one-line version ("is there a valid run proof, and does its `inventory_digest` match")
was **wrong twice over**, and both errors were silent:

- It dropped §4.3's **first clause** — "every accepted image has a valid proof". U-2 keeps
  it. This is what makes completion O(N) in per-image proofs, and therefore what makes the
  verification cache load-bearing rather than marginal.
- It dropped **four of the five comparisons** `current_aggregate_is_current` makes today
  (`_cli_completion.py:738-745`).

Rule 1 is therefore:

**`--mode process` takes a different rule 1, and the code already says so.** A process run
publishes **no aggregate proof at all**, so its run proof carries no `source_set_digest` and
no `source_image_count`, and its `finalization_input_digest` is a digest of
`{"process_only_layer": …}` rather than of `{metadata_sha256, include_dataset_column,
no_qc}`. Three of the five comparisons below are therefore inapplicable, not merely
different. `_cli_completion.py` carries **five** carve-outs for exactly this — `:722`,
`:763`, `:1008`, `:1020`, `:1092` — and a flat conjunction that ignores them makes every
process tree read `incomplete` forever (N-4).

```python
config = _read_state_config(output_dir) or {}
if config.get("process_only_layer"):
    # Clause 1 unchanged: every accepted image still needs a valid proof.
    # Clause 2 compares only what a process run actually publishes.
    return (
        all(image.verdict == "verified" for image in images.values())
        and proof is not None
        and proof["inventory_digest"]          == _canonical_digest(
            config.get("work_ids", {})
        )
        and proof["scientific_config_digest"]  == config.get("pipeline_sha256")
        and proof["finalization_input_digest"] == _canonical_digest(
            {"process_only_layer": config["process_only_layer"]}
        )
    )
```

The full-run form:

```python
# Clause 1 -- every accepted image has a valid proof.
if not all(image.verdict == "verified" for image in images.values()):
    ...falls through to rule 2

# Clause 2 -- a valid run proof covers the CURRENT inventory. Five comparisons,
# not one. Every value is a literal `config` field, so this costs nothing under
# INV-LAYER's plain-JSON constraint.
proof = valid_run_proof(output_dir)                     # local reader, sdk_-side
proof is not None
and proof["inventory_digest"]           == _canonical_digest(config.get("work_ids", {}))
and proof["finalization_input_digest"]  == _canonical_digest(finalization_input_object(...))
and proof["scientific_config_digest"]   == config.get("pipeline_sha256")
and proof["source_set_digest"]          == _canonical_digest(sorted(verified_work_ids))
and proof["source_image_count"]         == len(verified_work_ids)
```

> ### ⚠ Rule 1 is written against a proof shape that does not exist yet (found by executing)
>
> **Taken literally, rule 1 can never fire, so `resolve_run_state` can never return
> `complete` on any tree the real publishers built.** That contradicts this phase's own
> claim to work on today's trees — the property that makes P1 independently landable.
>
> `publish_run_completion_evidence` (`_cli_completion.py:979-996`) writes exactly seven
> fields:
>
> ```
> schema_version · generation · execution_epoch · mode · status ·
> finalizer_succeeded · completed_at
> ```
>
> **None of the five rule 1 compares.** Three separate mismatches:
>
> | Rule 1 reads | Reality |
> |---|---|
> | `proof["source_set_digest"]` | not in the run proof — it is in the **aggregate** proof, bound by `publication_id`. U-4 moves it in P4. |
> | `proof["source_image_count"]` | same |
> | `proof["finalization_input_digest"]` | the publisher digests the three values **without** `schema_version`; §5.5's object carries it, so the two cannot match |
>
> **The reader accepts both shapes and says where each half is dropped.** That is not a
> workaround — §5.5's own rule is that version skew is *"handled by the reader"*:
>
> - `_source_set_binding` reads the run proof when it carries the fields (post-U-4) and
>   otherwise follows `publication_id` to the aggregate proof.
> - `_accepted_finalization_digests` accepts the versioned and unversioned spellings.
>
> **Both must carry a comment naming P4 as the point where the legacy half goes**, or the
> compatibility outlives the reason for it — which is how a temporary shim becomes permanent
> and then becomes load-bearing.

> ### ⚠ The verdict matrix contradicts a test twelve lines below it
>
> `pytest.param(_mark_slurm_lifecycle_active, "active", id="live-worker")` is applied to
> `complete_run`, and `test_a_live_worker_does_not_mask_a_valid_run_proof(complete_run)`
> asserts `complete` for the **identical mutation on the identical fixture**. Both cannot
> hold. The matrix row is the wrong one: split it into
> `test_a_live_worker_over_an_unfinished_run_reads_active(incomplete_run)`.
>
> Three more matrix rows in the same block cannot fire as written, and each fails silently
> rather than loudly — they pass, proving nothing:
>
> - **`terminal-failure` on a complete run.** Rule 3 requires records *"with no superseding
>   success proof"*, so a row beside a valid marker is superseded and the verdict stays
>   `complete`. Worse, `append_terminal_failure` **refuses to write at all** while
>   `valid_image_success` is true, so the mutation writes nothing.
> - **`_edit_metadata_csv` changes no digest** — the same class as `_edit_pipeline_json`,
>   which cluster 1.2 already found. The finalization inputs read `config.metadata_sha256`
>   from `processing_state.json`, a literal field, not a re-derivation from the CSV. Use
>   `bump_metadata_snapshot_digest`.
> - **`test_a_dead_gui_owner_does_not_pin_the_verdict_at_active` cannot fail.** On
>   `complete_run`, rule 1 answers before rule 2 is consulted, so the assertion holds whether
>   or not CAN-24's liveness check exists. Needs the `incomplete_run` form **and** a positive
>   control, or "always answer not-alive" passes too.

> ### ⚠ The five comparison cases do not isolate their comparisons
>
> Step 6 asks for a proof that each of rule 1's five comparisons is load-bearing, but every
> parametrized case moves several at once: `_add_an_image` shifts `inventory_digest`,
> `source_set_digest` **and** `source_image_count` together *and* breaks clause 1, so
> deleting any one comparison leaves three others catching it and the case stays green.
>
> That is not a flaw in the chosen cases — **every realistic tree change has this shape**,
> because inventory is upstream of everything else. So the proof needs two families:
> **proof-field falsification** (each case caught by exactly one comparison — that is the
> can-fail proof) and **realistic tree changes** (the behaviour a user sees). Clause 1 needs
> its own case, since it is unobservable unless the aggregate proof is re-pointed at the
> reduced success set — which is what a *partial* publication looks like.

> ### `_run_process_mode` is defined nowhere
>
> Task 5 Step 5 calls `_run_process_mode(tmp_path, layer="objmap")`; a grep across the plan
> and spec finds no definition. **Same class as the `_invoke_cli` gap** — 27 uses, zero
> definitions. Implement it as `build_complete_run(tmp_path, process_only_layer=...)` driving
> the real publishers rather than shelling out to the CLI.

**Why each one matters** — a reviewer who deletes any of these is reintroducing a
documented defect:

| Comparison | What breaks without it |
|---|---|
| `inventory_digest` | a new image under a rolling input never invalidates completion |
| `finalization_input_digest` | **§7.4's late-metadata guarantee stops working.** It is real today *only* because of this comparison: a metadata edit leaves `work_ids` untouched, so nothing else notices |
| `scientific_config_digest` | a pipeline edit leaves the run reading `complete` |
| `source_set_digest` | the only check that the published master covers the succeeded set — **this is what makes CAN-5's partial shard set undetectable** |
| `source_image_count` | a cheap arity cross-check on the same |

`source_set_digest` now also lives in the **run** proof, not just the aggregate — that is
U-4's replacement for the cut `publication_id`, and it is what lets the aggregate↔run
binding be stated directly instead of through an opaque hash.

- [ ] **Step 4: Run the tests.** Expected: PASS (13 passed).

- [ ] **Step 5: Add the five comparison rows and the stale-owner row**

Each is a one-line mutation of the `complete_run` fixture, and each must produce
`incomplete`:

```python
@pytest.mark.parametrize(
    "mutate,reason",
    [
        (_edit_metadata_csv,      "finalization_input_digest"),
        (_edit_pipeline_json,     "scientific_config_digest"),
        (_add_an_image,           "inventory_digest"),
        (_succeed_one_more_image, "source_set_digest"),
        (_drop_a_source_count,    "source_image_count"),
    ],
)
def test_each_dropped_comparison_is_load_bearing(complete_run, mutate, reason):
    """CAN-4. The one-line rule 1 kept only inventory_digest. Each of these
    would have read `complete` under it."""
    mutate(complete_run)
    assert resolve_run_state(complete_run, depth="deep").completion == "incomplete", (
        f"{reason} is not being compared; §7.4 and CAN-5 both depend on it"
    )


def test_a_process_run_reads_complete(tmp_path):
    """N-4. A process run publishes no aggregate proof, so three of rule 1's five
    comparisons are inapplicable -- not merely different. The flat conjunction
    CAN-4's fix introduced made every process tree read `incomplete` forever.

    _cli_completion.py carries five carve-outs for this (:722, :763, :1008, :1020,
    :1092). Rule 1 needs the same shape, and process mode is in scope elsewhere in
    this change -- CAN-20 parametrizes identity over it, CAN-32 classifies it for
    requires_conversion -- so it cannot be waved off as out of scope.
    """
    output = _run_process_mode(tmp_path, layer="objmap")
    assert resolve_run_state(output, depth="deep").completion == "complete"


def test_a_process_run_still_detects_a_pipeline_edit(tmp_path):
    """The carve-out narrows the comparison set; it does not disable it."""
    output = _run_process_mode(tmp_path, layer="objmap")
    _edit_pipeline_json(output)
    assert resolve_run_state(output, depth="deep").completion == "incomplete"


def test_a_dead_gui_owner_does_not_pin_the_verdict_at_active(complete_run):
    """CAN-24. Nothing in the codebase repairs gui_launch_owner.json (audit S7,
    verified), so a SIGKILLed GUI leaves status: "running" forever. Without a
    liveness check on the authority itself, Q2 rule 2 is unsound and this output
    reads `active` until someone edits JSON by hand.

    The repair implementation lands in P6 Task 5, where
    _assert_output_claimable_locked is rewritten. The LADDER's obligation is here,
    because the ladder is built here.
    """
    _write_owner_record(complete_run, status="running", pid=_a_dead_pid())
    assert resolve_run_state(complete_run, depth="deep").completion == "complete"
```

- [ ] **Step 6: Prove the precedence tests can fail**

Swap ladder rules 1 and 2; confirm `test_a_live_worker_does_not_mask_a_valid_run_proof`
fails. Swap 2 and 3; confirm `test_an_active_run_outranks_a_stale_terminal_failure` fails.
Make the metadata advisory a gate (return `incomplete`); confirm
`test_a_store_built_against_older_metadata_is_an_advisory` fails. Delete the
`provenance: "migrated"` advisory entirely; confirm P7's
`test_the_unavailable_fence_is_surfaced_as_an_advisory` fails — that one is the visible half
of U-10, so a suite that stays green without it has stopped enforcing the ruling. **Delete
each of the five
comparisons in turn and confirm the matching parametrized case fails** — that is the check
that stops rule 1 collapsing back to one line. Restore all of them.

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/sdk_/_run_state.py tests/unit/sdk_/test_run_state.py
git commit -m "feat(sdk): resolve_run_state, deep path, with the Q2 verdict ladder

Spec §4.3, §9. Replaces ~23 classification rules with four ordered questions and
deletes `contradictory` as a reachable state. The D-A metadata-divergence advisory
is derived from each store's own snapshot_sha256 and is never a gate; all three
precedence tests were confirmed to fail when the ladder is reordered."
```

---

## Task 6: `resolve_run_state` — the shallow path

**Files:**
- Modify: `src/phenotypic/sdk_/_run_state.py`
- Test: `tests/unit/sdk_/test_run_state.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_shallow_after_deep_does_not_re_hash_artifacts(complete_run, monkeypatch):
    """Spec §9.2: adding 10 images to 6,000 should cost 6,000 stats and 10 deep
    verifications, not 6,000 re-hashes. On a 10,000-image run on GPFS, one badge
    refresh is currently ~10^4 marker reads and 2-3 x 10^4 file hashes. Per tab.
    Every five seconds."""
    import hashlib

    resolve_run_state(complete_run, depth="deep")   # warm

    calls = {"n": 0}
    real = hashlib.sha256

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(hashlib, "sha256", counting)
    state = resolve_run_state(complete_run, depth="shallow")
    assert state.completion == "complete"
    assert state.depth == "shallow"
    # Identity digests still hash a few small payloads; artifact CONTENT must not.
    assert calls["n"] <= 8, (
        f"shallow re-hashed artifacts ({calls['n']} sha256 calls); the whole point "
        "of §9.1 is that it re-stats instead"
    )


def test_a_new_image_escalates_the_whole_resolution(complete_run):
    resolve_run_state(complete_run, depth="deep")
    _add_third_image(complete_run)
    state = resolve_run_state(complete_run, depth="shallow")
    assert state.depth == "deep", "a cache miss must escalate"
    assert state.completion == "complete"


def test_shallow_with_a_cold_cache_equals_deep(complete_run):
    from phenotypic.sdk_ import clear_verification_cache

    clear_verification_cache()
    cold = resolve_run_state(complete_run, depth="shallow")
    deep = resolve_run_state(complete_run, depth="deep")
    assert cold.completion == deep.completion
    assert set(cold.images) == set(deep.images)
    assert cold.depth == "deep", "a cold shallow call is a deep call, and says so"
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

For each accepted `work_id`: if `cached_states(output_dir, identity.digest())` yields an entry for the
work_id)` returns an entry **and** `entry_is_still_current(output_dir, entry)`, reuse that
image's previous deep result. Otherwise mark the resolution escalated. If any image
escalated, re-run the deep verification **for the escalated images only**, remember the
results, and set `RunState.depth = "deep"`.

Setting `depth = "deep"` whenever *anything* escalated is deliberate: `depth` is what a
caller reads to know whether the answer is authoritative, and "mostly shallow" is not a
useful third value.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Run the Phase-1 test selection**

Via the **`run-phenotypic-test`** skill:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/sdk_/test_run_state.py \
  tests/unit/sdk_/test_verification_cache.py \
  tests/unit/sdk_/test_run_state_layering.py -p no:randomly -q
```

Record the count.

- [ ] **Step 6: Phase gate**

```bash
uv run mypy src/phenotypic
uv run ruff check --fix src/phenotypic/sdk_/_run_state.py \
  src/phenotypic/sdk_/_verification_cache.py src/phenotypic/sdk_/_io_constants.py \
  src/phenotypic/sdk_/__init__.py tests/unit/sdk_/ tests/_output_layout.py
```

Then the CLI + GUI regression selection, which must be **unchanged** — this phase moved no
consumers. Any new failure here means something was wired up that should not have been:

```bash
QT_QPA_PLATFORM=offscreen uv run pytest tests/unit/cli tests/unit/gui -q
```

- [ ] **Step 7: Commit**

```bash
git add -A src/phenotypic/sdk_ tests/unit/sdk_
git commit -m "feat(sdk): shallow resolution via the in-process verification cache

Spec §9.1, §9.2. A steady-state badge refresh drops from ~10^4 artifact hashes to
~10^4 stats on the first tick and ~0 after, within one process. No consumer moved
in this phase."
```
