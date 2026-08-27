# Phase 3 — Scratch store lifecycle

**Spec:** §3, §6. **Depends on:** phase 2. **Blocks:** phase 4.

**Deliverable:** a **measured** retention cap on scope revisions per session, enforced
oldest-first and never evicting the focused scope, plus a session-exit sweep at builder
startup.

> **Why this is a phase and not a cleanup.** Viv-rebuild decision E named "a scratch dir to
> garbage-collect" as an accepted cost but set no policy. `wipe_cache` (`:56`) and
> `wipe_scope` (`:90`) exist, and `wipe_scope` runs when a scope's fingerprint changes — but
> nothing bounds accumulation across *revisions* within a session, and nothing reclaims a
> dead session's tree. With one store per node per scope revision, a long authoring session
> grows without limit.

---

### Task 3.1: Measure before capping

**Files:**
- Create: `docs/superpowers/logic_validation_scripts/2026-08-26-builder-preview-viv/preview_scratch_budget.py`

**Interfaces:**
- Produces: `PREVIEW_SCOPE_RETENTION` — the number task 3.2 enforces.

- [ ] **Step 1: Measure one real scope's on-disk cost**

```bash
uv run du -sh <preview_cache_root>/<session>/<scope_hash>
uv run find <preview_cache_root>/<session>/<scope_hash> -type f | wc -l
```
Record bytes and inode count per scope revision at a realistic node count. Inodes matter as
much as bytes here — the backend spec's §1.4 records ~40 files per pyramided store, and a
preview store is single-level (16 files), so a 10-node pipeline is ~160 files **per
revision**.

- [ ] **Step 2: Write the budget script**

Per root `CLAUDE.md`: stdlib + numpy/scipy only, never imports `phenotypic`, exits non-zero
on failure.

```python
"""Re-derive the builder preview scratch budget.

Claim under test (builder-preview-viv spec section 3): one store per node per
scope revision accumulates without bound in a long authoring session, so the
retention cap must be chosen against a measured per-revision cost rather than
picked.

Fill MEASURED_* from task 3.1 step 1, then this script derives the cap.
Exits non-zero while the measurement is missing.
"""

import sys

#: Bytes for one scope revision, measured on a real session (task 3.1 step 1).
MEASURED_BYTES_PER_REVISION: int | None = None
#: Files for one scope revision, measured the same way.
MEASURED_FILES_PER_REVISION: int | None = None

#: Budgets the cap must fit inside.
BYTE_BUDGET = 2 * 1024**3
FILE_BUDGET = 20_000


def cap_from(bytes_per: int, files_per: int) -> int:
    """Largest revision count fitting both budgets."""
    return max(1, min(BYTE_BUDGET // bytes_per, FILE_BUDGET // files_per))


def main() -> int:
    if MEASURED_BYTES_PER_REVISION is None or MEASURED_FILES_PER_REVISION is None:
        print("NO MEASUREMENT: fill MEASURED_* from task 3.1 step 1")
        return 1
    cap = cap_from(MEASURED_BYTES_PER_REVISION, MEASURED_FILES_PER_REVISION)
    print(
        f"per revision: {MEASURED_BYTES_PER_REVISION / 1e6:.1f} MB, "
        f"{MEASURED_FILES_PER_REVISION} files"
    )
    print(f"PREVIEW_SCOPE_RETENTION = {cap}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Run it — failing while unmeasured is the point**

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-08-26-builder-preview-viv/preview_scratch_budget.py
```
Expected: `NO MEASUREMENT`, exit **1**. Fill the constants from step 1, re-run, and record
the derived cap.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/logic_validation_scripts/2026-08-26-builder-preview-viv/
git commit -m "docs(gui): measure the builder preview scratch budget"
```

---

### Task 3.2: Enforce the cap oldest-first

**Files:**
- Modify: `src/phenotypic/gui/builder/_preview_cache.py`
- Test: `tests/unit/gui/builder/test_preview_retention.py` (create)

**Interfaces:**
- Consumes: `PREVIEW_SCOPE_RETENTION` from task 3.1.
- Produces: `enforce_scope_retention(session_id, *, focused_scope_hash) -> list[str]`
  returning the scope hashes evicted.

- [ ] **Step 1: Write the failing tests**

```python
"""Scope revisions are capped oldest-first, sparing the focused scope.

The second test is the one with teeth: evicting the scope the user is looking
at would be a cache policy that produces a blank pane, which reads as a bug in
the renderer rather than in the cache.
"""

from phenotypic.gui.builder._preview_cache import (
    PREVIEW_SCOPE_RETENTION,
    enforce_scope_retention,
)


def test_evicts_oldest_first_down_to_the_cap(session_with_many_scopes):
    session_id, hashes_oldest_first = session_with_many_scopes
    evicted = enforce_scope_retention(
        session_id, focused_scope_hash=hashes_oldest_first[-1]
    )
    survivors = [h for h in hashes_oldest_first if h not in evicted]
    assert len(survivors) <= PREVIEW_SCOPE_RETENTION
    assert evicted == hashes_oldest_first[: len(evicted)]


def test_never_evicts_the_focused_scope(session_with_many_scopes):
    session_id, hashes_oldest_first = session_with_many_scopes
    focused = hashes_oldest_first[0]
    evicted = enforce_scope_retention(session_id, focused_scope_hash=focused)
    assert focused not in evicted
```

Note the deliberate tension: `test_never_evicts_the_focused_scope` focuses the **oldest**
scope, which is exactly the case a naive oldest-first sweep gets wrong.

- [ ] **Step 2: Run, implement, run**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/builder/test_preview_retention.py -v
```
Expected: FAIL (no `enforce_scope_retention`), then PASS after implementing.

Order revisions by the manifest's own recorded time rather than by directory mtime — a
store directory's mtime does not move when a nested chunk is rewritten, which is the same
trap the freshness checks already avoid.

- [ ] **Step 3: Call it where a scope is promoted**

`_preview_cache.py:158` `_promote_scope_state` is the natural hook. Confirm:

```bash
uv run sed -n '151,175p' src/phenotypic/gui/builder/_preview_cache.py
```

- [ ] **Step 4: Commit**

```bash
uv run ruff check --fix src/phenotypic/gui/builder/_preview_cache.py \
                        tests/unit/gui/builder/test_preview_retention.py
git add src/phenotypic/gui/builder/_preview_cache.py \
        tests/unit/gui/builder/test_preview_retention.py
git commit -m "feat(gui): cap retained preview scope revisions oldest-first"
```

---

### Task 3.3: Sweep dead sessions at startup

**Files:**
- Modify: `src/phenotypic/gui/builder/_preview_cache.py` (`init_cache`, `:61`)
- Test: `tests/unit/gui/builder/test_preview_retention.py` (extend)

- [ ] **Step 1: Write the test**

```python
def test_startup_sweeps_sessions_with_no_live_dash_session(stale_session_tree):
    """A session id with no live Dash session is reclaimable at startup.

    Startup is the only safe moment: mid-run there is no way to distinguish a
    dead session from one whose browser tab is merely backgrounded.
    """
    from phenotypic.gui.builder._preview_cache import init_cache, preview_cache_root

    stale_id, live_id = stale_session_tree
    init_cache(live_session_ids={live_id})

    assert not (preview_cache_root() / stale_id).exists()
    assert (preview_cache_root() / live_id).exists()
```

- [ ] **Step 2: Implement**

Give `init_cache` an optional `live_session_ids` parameter defaulting to `None`, meaning
"sweep nothing" — so an existing caller that does not pass it keeps today's behaviour
exactly. The builder app passes the live set at startup.

**Startup is the only safe moment** to sweep: mid-run there is no way to distinguish a dead
session from a backgrounded browser tab, and a wrong guess deletes the stores under a
working pane.

- [ ] **Step 3: Run, lint, commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/unit/gui/builder/test_preview_retention.py -v
uv run ruff check --fix src/phenotypic/gui/builder/_preview_cache.py \
                        tests/unit/gui/builder/test_preview_retention.py
git add src/phenotypic/gui/builder tests/unit/gui/builder
git commit -m "feat(gui): sweep dead-session preview sandboxes at startup"
```
