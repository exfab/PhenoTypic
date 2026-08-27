# Concern ledger — GUI simplification, Viv rebuild, builder preview

Append-only entries; statuses updated in place. A `resolved` entry naming what changed IS
the provenance lock for that change.

**Prefixes:** `GEN` general-reviewer · `FLOW` data-flow-reviewer · `SIMP`
simplicity-reviewer · `SEC` security specialist · `ALGO` algorithm-fidelity ·
`USER` orchestrator-raised user ruling.

**Statuses:** `open` · `resolved (round N: …)` · `settled-by-user (round N: …)` ·
`conflict (vs <ID>)` · `advisory`.

---

## Round 0 — user rulings

### USER-1 [Critical] [settled-by-user (round 0: NFR lines added to all three specs)]
- Raised: round 0, orchestrator (spec-anchor check)
- Concern: None of the three specs carried a performance/NFR line, which the refinery
  requires as an anchor for precedence tier 8. Without one, every simplicity-vs-performance
  dispute in the Viv rebuild would resolve against performance by default — and that
  rebuild is *entirely motivated* by performance.
- Resolution: User ruled "Interactive over ssh would be nice but correctness is most
  important", and "no performance requirements" for the removals spec. Added as
  `gui-simplification-removals` §9.1, `viewer-viv-rebuild` §9.1, and
  `builder-preview-viv` "Non-functional requirements". **Binding: correctness. Target,
  non-binding: interactive over an SSH tunnel.** The target ranks above tier 8 and below
  correctness, data integrity, and reference faithfulness.
- **Permanent.** No reviewer may re-raise absent new evidence.

### USER-2 [n/a] [settled-by-user (round 0: Summary accepted as Objective)]
- Raised: round 0, orchestrator (spec-anchor check)
- Concern: Specs 1 and 2 have no heading literally named "Objective & Non-goals".
- Resolution: Orchestrator ruling — each spec's `## Summary` states what the change
  achieves and both carry an explicit `## 9. Non-goals`. Accepted as satisfying the anchor;
  no spec edit made. Spec 3 has both headings literally.

---

## Round 1

### ORCH-1 [n/a] [resolved (round 1: verified empirically, plans updated)]
- Raised: round 1, orchestrator (pre-empting SEC-a)
- Concern: Both byte-route plans assert that `is_safe_path_component`
  (`src/phenotypic/gui/_shared/tiles.py:755`) rejects `..`, and both merely instruct the
  executor to *check* it. Nobody had run it. If it did not, both routes would be traversable
  — the highest-severity unknown in the set. Brief §11 item 6 recorded it as unverified.
- Evidence gathered (run in this worktree, 2026-08-26):

  ```text
  '..' -> False   '.' -> False    '...' -> False   '.hidden' -> False
  'a/b' -> False  'a\b' -> False  '%2e%2e' -> False  '' -> False
  'rgb' -> True   'gray' -> True  'detect_mat' -> True  'OME' -> True
  'zarr.json' -> True  'c.0.0.0' -> True  '0.0' -> True
  'labels' -> True     'objmap' -> True   'tables' -> True
  ```

  Implementation: rejects empty, any leading dot, `/`, `\`, literal `..`, then requires
  `^[A-Za-z0-9._-]+$` (`_NAME_RE`, `:752`). `%2e%2e` fails the regex directly; after
  Werkzeug decoding it becomes `..` and fails the explicit check. Zarr's `"."`-separated
  chunk keys (`c.0.0.0`, `0.0`) pass, as does `zarr.json`.
- Resolution: **The guard is sound for both routes.** Two consequences, both applied:
  1. The label group is NOT blocked by `_READABLE_ROOTS`. Only `segments[0]` is gated, and
     the label path is `<primary>/labels/objmap` whose head is `rgb` or `gray` — both
     allow-listed. The allow-list still blocks `tables/measurements/table.parquet`, which
     the guard alone would pass (`'tables' -> True`). So the allow-list is load-bearing and
     is not a functional bug.
  2. Both plans' "go check this" steps are replaced with the verified result, so an
     executor does not re-derive it.

### ORCH-2 [Major] [open] `spec-change` `needs-user-input`
- Raised: round 1, orchestrator (pre-empting SEC-d)
- Concern: **`builder-preview-viv` spec §2 overstates what `_validate` does.** The spec says
  session isolation is a correctness requirement and that "`_validate` is what keeps one
  browser session out of another's sandbox". Read at
  `src/phenotypic/gui/builder/_preview_tiles.py:107`, `_validate` is a **pure shape check**:

  ```python
  is_safe_path_component(session_id)
  and bool(_HASH_RE.match(scope_hash))     # ^[0-9a-f]{40}$
  and is_safe_path_component(block_id)
  and channel in _VALID_CHANNELS
  ```

  It authenticates nothing. It stops path traversal through those components; it does not
  establish that the caller owns the session. **Isolation rests entirely on session-id
  unguessability.**
- Mitigating evidence: session ids are `uuid.uuid4().hex`
  (`builder/_callbacks.py:3662, :4083`; `builder/_state.py:77`) — 122 bits, not guessable.
  So the property holds in practice; the *spec's account of why* is wrong.
- Residual exposure: session ids travel **in the URL path**, so they reach browser history,
  `Referer` headers, and any proxy or server log. The deployment is Open OnDemand-style
  proxying on a shared-filesystem HPC cluster (root `CLAUDE.md`, "GUI hub"), so those logs
  are not necessarily private to the user.
- Suggested direction: amend spec §2 to say isolation is a **bearer-token property of the
  session id**, not an authentication check, and state the logging exposure as an accepted
  risk (or require the id move out of the path). Do **not** silently reword the plan around
  a spec that misdescribes its own security model.
- Resolution: — *(gated to the user with the other spec-change items)*

### ORCH-3 [Minor] [resolved (round 1: plan risk closed, helpers are trivial)]
- Raised: round 1, orchestrator (pre-empting FLOW-E)
- Concern: `plans/2026-08-26-builder-preview-viv/phase-1` posits `read_manifest_by_hash` and
  `scope_dir_by_hash` and flags them as possibly nonexistent. Confirmed: they do **not**
  exist — `_preview_cache.py:83, :95` key by `scope_path` (a list), not by hash, and
  `scope_hash` (`:71`) is a one-way sha1.
- Resolution: **the risk closes favourably.** `_scope_path` (`:78`) builds
  `preview_cache_root() / session_id / scope_hash(scope_path)` — the on-disk directory *is*
  named by the hash, so a hash-keyed lookup needs no reverse index, just a variant that
  takes the hash directly. The plan's mitigation (thin wrappers beside the existing
  helpers) is correct and is ~4 lines. Plan updated to say so rather than leaving the
  executor to discover it.

### ORCH-4 [Minor] [open]
- Raised: round 1, orchestrator
- Concern: `plans/2026-08-26-builder-preview-viv/phase-1` proposes calling
  `_validate(session_id, scope_hash, block_id, "gray")` — passing a **hardcoded fake
  channel** purely to satisfy a 4th parameter the byte route has no use for.
  `_VALID_CHANNELS = ("rgb","gray","detect_mat","objmap","overlay")`
  (`_preview_tiles.py:37`), so "gray" is merely the arbitrary member that passes.
  A guard invoked with a fabricated argument is one refactor away from being invoked
  wrongly.
- Suggested direction: extract the session/scope/block half of `_validate` into a
  channel-free helper both routes call; leave channel validation in the DZI route. Likely
  overlaps a `SIMP-` concern on route duplication — merge if so.
- Resolution: —

### ORCH-5 [Major] [resolved (round 1: `⏸ unmounted` specified in both phases)]
- Raised: round 1, orchestrator (pre-empting GEN-d)
- Concern: removals spec §6 requires unmounted surfaces to be "**marked as unmounted with a
  pointer to this spec**, not deleted outright — the ledger's job is to describe what a user
  can reach, and 'exists but unreachable' is a state it should carry." The plan's phases 4
  and 5 deferred *how* to an executor check, with an instruction to escalate if the
  vocabulary could not express it. That left a spec-stated requirement resting on an
  unverified assumption, across two phases.
- Evidence (`scripts/check_features_md.py`): the vocabulary is three constants —
  `✅ shipping` (`:33`), `🚧 in progress` (`:34`), `🔭 planned` (`:35`). Crucially, the
  loop at `:148-176` **never validates status against an allowed set**. It branches:
  `== STATUS_IN_PROGRESS` → collect (failed by `--strict` at `:178-184`);
  `!= STATUS_SHIPPING` → `continue`, silently skipped; else resolve refs.
- Resolution: **use `⏸ unmounted`, and no script change is required.** It falls into the
  `!= STATUS_SHIPPING` skip branch, so it stops ref resolution, and it is not
  `🚧 in progress`, so `--strict` passes it. Neither existing status is right:
  `✅ shipping` would keep resolving refs *and* falsely claim reachability; `🚧 in progress`
  fails the merge gate; `🔭 planned` passes but describes a fully built, parked surface as
  unbuilt. Both phase docs now specify the value and cite this evidence, and the
  escalate-to-user step is removed.

### ORCH-6 [Minor] [advisory]
- Raised: round 1, orchestrator
- Concern: the silent-skip behaviour above is a latent trap **pre-existing this work**: a
  typo'd status (`✅ shiping`) is silently skipped rather than rejected, so its row is never
  ref-checked and the gate reports OK. `⏸ unmounted` works *because* of that permissiveness.
- Suggested direction: a closed-set validation in `check_features_md.py` rejecting unknown
  statuses would make the vocabulary real. **Out of scope for these three plans** — it is a
  pre-existing gate weakness, not something this change introduces. Recorded so the
  dependence is explicit rather than accidental.
- Resolution: advisory; no plan change.

### Round 1 reports

*(entries appended as reports arrive)*
