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

Received: `sec-r1` (VERDICT: REVISE), `flow-r1` (VERDICT: BLOCKING), `simp-r1` (VERDICT:
BLOCKING). Outstanding at time of writing: `gen-r1`, `algo-r1` — **not** treated as APPROVE.

Full reports: `reports/round-1-{security,data-flow,simplicity}.md`.

**Every load-bearing claim below was independently verified by the orchestrator before
being acted on.** Five of them break code this plan set had already written.

---

## Round 1 — user rulings

### USER-3 [settled-by-user (round 1: apply all three)]
- Concern: three spec statements are factually wrong about the landed code.
- Ruling: apply all three. (1) Viv §1's "specification only — there is no zarr code in
  `src/`"; (2) Viv §6.2 + backend §3.4's claim that Stage 2 writes the objmap in-store;
  (3) Viv §4's unrestricted byte-route tail. The backend spec `2026-08-18` is on the parent
  branch — record the §3.4 correction as a note there rather than editing that branch.

### USER-4 [settled-by-user (round 1: accepted bearer-token risk)] — resolves SEC-3, FLOW-3, ORCH-2
- Concern: `builder-preview-viv` §2 credits `_validate` with session isolation. Verified
  false — `_preview_tiles.py:107-116` is shape validation only, with nothing binding the
  requester to a session. Real isolation is `uuid.uuid4().hex` (122 bits) carried **in the
  URL path**, so it reaches access logs, the OOD reverse proxy's logs, browser history and
  `Referer`.
- Ruling: **record it honestly as an accepted bearer-token / capability-URL risk.** Do not
  add server-side binding, and do not move the id out of the path — that would diverge from
  the existing `/preview-tiles/` route for no behaviour change. Amend the spec to describe
  the capability model and say the id must be treated as a secret.
- **Permanent.** SEC-4's `CONFLICT with SEC-3` resolves this way: the test asserts the
  property that actually holds, not a binding that does not exist.

### USER-5 [settled-by-user (round 1: fold)] — resolves SIMP-10
- Ruling: **pair 3 folds into pair 2.** `viewer-viv-rebuild` gains a phase 6 (preview byte
  route + shared asset mount + render swap); its phase 5 absorbs the verification.
  `specs/2026-08-26-builder-preview-viv/` retires as a document, its content replacing
  viv-rebuild §7's "cycle 3, out of scope" paragraph.
- Consequences: phase 1 writes the shared resolver with **both** callers in view; phase 3
  writes `build_source_spec` at its final signature instead of refactoring its own work two
  phases later; viv-rebuild §7 OQ1 (one `_assets/viv/` for both sub-apps) is answered while
  that artifact is being built; one ledger-and-gates pass instead of two.

### USER-6 [settled-by-user (round 1: drop entirely)] — resolves SIMP-8, FLOW-11
- Concern: attacked from both sides. SIMP-8: an accepted *cost* ("a scratch dir to
  garbage-collect") became a designed subsystem. FLOW-11 (verified): `init_cache()`
  (`_preview_cache.py:61-68`) already calls `wipe_cache()` — recursive delete of the whole
  cache root — plus an `atexit` wipe, so the startup sweep is redundant *and* the plan's
  "`live_session_ids=None` keeps today's behaviour exactly" was false, *and* its own test
  asserting a live session's tree survives was impossible.
- Ruling: **drop the phase entirely.** Startup and atexit already wipe everything;
  `wipe_scope` reclaims on fingerprint change. The residual gap — one session authoring many
  distinct scopes without a restart — is unevidenced. Record that as the stated policy and
  amend the spec §3 text accordingly.
- Note: the two reviewers pointed opposite ways (SIMP-8 said keep the sweep, drop the cap;
  FLOW-11 said the sweep's premise is false). Resolved on **evidence over argument**, per
  the conflict ladder rung 2.

### USER-7 [settled-by-user (round 1: colony script only, no 4096)] — resolves SIMP-3, SIMP-4, SIMP-5
- Ruling: keep **only** `colony_view_budget.py`, and **drop its unsourced 4096 draw-call
  ceiling** so it fails closed on "no measurement" and nothing else. Delete
  `tile_fetch_budget.py` (re-derives nothing; exits 0 regardless, so it gates nothing) and
  `preview_scratch_budget.py` (moot under USER-6; divided a real measurement by two invented
  budgets). The 3 MB per-tile figure becomes one line of arithmetic in `spike/README.md`.
- Rationale carried forward: CLAUDE.md mandates a script for a numeric invariant "a reader
  would otherwise take on faith". `1024×1024×3` is not such an invariant; the surviving
  script is kept because its number lands in shipped code as a behavioural cap.

---

## Round 1 — orchestrator-verified reviewer findings

Each verified by reading the cited file before being accepted.

### SEC-2 [Major] [open] — **overturns ORCH-1's conclusion**
`_READABLE_ROOTS` is tested on the **unresolved** `segments[0]` while `send_file` sends
`resolved`. A symlink inside a readable root (`<store>/rgb/x -> ../tables/measurements/table.parquet`)
passes the head check *and* `is_relative_to` containment, and the parquet is served.
ORCH-1 concluded the allow-list "is the only thing keeping `tables/` off the wire", which
implied it did keep it off. It does not, as written. **Fix:** enforce after resolution —
`rel = resolved.relative_to(store_resolved)`, test `rel.parts[0]`. One check, not two.

### SEC-4 [Major] [open] — resolved by USER-4
`test_one_session_cannot_reach_anothers_sandbox` sends session **A's** id with **B's** hash,
so it 404s on a manifest miss, not on isolation. It would pass with every isolation
mechanism deleted. The permutation that matters — presenting **B's** id — succeeds today.
Rewrite to assert the property USER-4 recorded.

### FLOW-1 [Critical] [open]
Both byte routes carry no generation token, so a client can combine metadata from promote N
with chunks from N+1 (`promote_store`, `ngff_.py:1235-1300`, renames the whole directory).
Benign for a run store re-promote (extent unchanged); **not** benign for the builder
preview, where re-running a node legitimately changes extent → decode error or plausible
wrong pixels. The old path was coherent *because* of `_store_content_token`
(`_tile_routes.py:505-527`), which the rebuild deletes without replacement.
**Fix:** put the token in the URL so a new promote yields a new base URL.

### FLOW-2 [Critical] [open] — verified
`_store_for_block` reads `manifest.get("blocks", [])` / `entry["block_id"]`. The real shape
is `{"version","fingerprint","fingerprint_inputs","scope_key","nodes": {block_id: {"store",…}},"error"}`
(`_preview_cache.py:257-263`) — `nodes` is a **dict keyed by block_id**; no `"blocks"` list
exists. As written every preview-zarr request 404s. The landed DZI route already does it
correctly (`_preview_tiles.py:127-140`).

### FLOW-5 [High] [open] — verified
`build_phenotypic_attributes` **omits** the `labels` key when `has_labels=False`
(`ngff_.py:576-581`), with an inline note that an earlier draft emitted it unconditionally
and `assert_store_conforms` then `FileNotFoundError`'d (ledger C3).
`save_intermediate_zarr` sets `write_objmap = "objmap" in layers`, so most preview stores
have no `labels` key. `block["labels"]["objmap"]` `KeyError`s. **Fix:** `labelPath` optional
through the spec dict, the façade and the Layers panel; add a label-less fixture.

### FLOW-6 [High] [open] — verified
`_write_store_part` appends `"original"` to `series_names` when the image carries one
(`_image_io_handler.py:1012-1014`), and that list lands in `attributes.phenotypic.series`.
So the Layers panel lists `original` while the route 404s it — **the hard-coding the
label-path rule forbids, reappearing one layer down**. Task 3.1's own test
(`set(spec["series"]) <= {"rgb","gray","detect_mat"}`) fails against such a store.
**Fix:** derive the readable set per store from `series` + `labels`, or invert to a
deny-list on `tables/`.

### FLOW-9 [High] [resolved (round 1: three proof tests named and run in both plans)] — verified
The curation gate names the wrong file. All 15 tests in `test_colony_callbacks_helpers.py`
drive pure `_triage_callbacks` helpers against hand-built `ctx.triggered` dicts; nothing
asserts a radial exists on a tile or that anything reaches disk — so it would pass unmodified
while the deck.gl rewrite removed the radial entirely. The real proofs exist and are
collected but **no pytest command in either plan runs them**:
`tests/gui/results_viewer/colony_view/test_grid.py:454`,
`tests/integration/gui/test_triage_callbacks.py:227`, `tests/unit/cli/test_cli_error_outputs.py`.
Keep the "unmodified" rule; stop calling it the proof.

### FLOW-14 [Medium] [open] — DRIFT.md correction
D-1's "all of them already key on the root `zarr.json`" is true for `_tile_routes` and
`_preview_tiles` and **vacuous for the crop path**: `crop_colony` (`tiles.py:715`) still
passes `os.stat(store).st_mtime_ns`, which `crop_store_rgb` then `del`s (`:585-587`) because
crop reads are windowed and nothing caches on it. No bug — but a reader auditing the fourth
fix finds a live directory-mtime call and reasonably concludes the trap is open.

### FLOW-17 [Medium] [open] — cross-sub-app hazard
`builder/_preview_tiles.py:31` imports `_TILE_NAME_RE` and `_json_error` **from
`results_viewer/_tile_routes`**, and `_validate` returns `_json_error(...)`. Reading viv
phase 3 step 5's "remove them with their tests" as deleting `_tile_routes.py` breaks the
builder preview *and* the new preview route at import, in a different sub-app from the one
being edited. Same shape as the `_dzi_tiler` misreading DRIFT D-5 guards against.

### Accepted without further verification (consistent with verified facts)
FLOW-4, FLOW-7, FLOW-8, FLOW-10, FLOW-13, FLOW-15, FLOW-16, FLOW-18, FLOW-19;
SEC-5, SEC-7, SEC-8, SEC-9; SIMP-1, SIMP-2, SIMP-7, SIMP-9.

### SIMP-6 [Major] [open] — ORCH-4 merged in as an alias
Both routes need one `resolve_within_root(root, tail, *, allowed_roots=None) -> Path`.
Extract to `gui/_shared/tiles.py` beside `is_safe_path_component`. A path-escape guard is a
security primitive and USER-1 makes correctness binding; two copies drift **silently**,
because each plan tests only its own copy — as SEC-2 already demonstrates, the allow-list
exists in one copy and not the other in the very first version. Same commit: split
`_validate`'s session/scope/block half into a channel-free `_validate_scope` (ORCH-4).
Under USER-5 both callers are now in one plan, so this is written once, not extracted later.

---

## Round 1 — general and algorithm-fidelity reports

Received late; both `VERDICT: REVISE` / `BLOCKING`. Full reports in `reports/`.
**Round 1 therefore had full panel coverage after all** — the earlier coverage-gap note is
withdrawn.

### GEN-6 [Critical] [resolved (round 1: all phases anchor on headings)]
**Overturns brief §9.** Every `FEATURES.md` range in the removals spec is wrong by ≈ +7 past
line ~400. `:372-377` are **Colony curation rows**; `### Results Timeline tab` is at `:379`.
Phase 1 as written deleted four curation rows — a spec §5 violation that phase 5's diff
guard missed because it watched code, not the ledger. Offsets tabulated in the plan README;
guard extended to `FEATURES.md`. `WORKFLOWS.md` rows were correct.

### GEN-8 [Critical] [resolved (round 1: paths corrected)]
`tests/unit/gui/browse/` does not exist — browse and builder GUI tests live at
`tests/gui/`. Every `pytest tests/unit/gui/browse` command was an exit-4 usage error, and
`tests/unit/gui` gates silently exclude `tests/gui`, where the curation proof lives.

### GEN-7 [Critical] [resolved (round 1: task 2.4b added)]
Five test files break in phase 2, listed nowhere: `test_thumb_routes.py` (collection
`ImportError`), 33/22/7 timeline refs in `test_ids.py` / `test_layout.py` /
`test_compare_layout.py`, and one integration test. Two `✅ shipping` FEATURES.md rows
(`:64`, `:87`) point into that set from sections phase 2 never edits.

### GEN-9 [Major] [resolved (round 1: `--skip-cli`)]
`--smoke` is not a flag — the script takes `--force` / `--headed` / `--skip-cli`
(`:2913-2926`); argparse exits 2. It appeared in eight places.

### ALGO-1 [Major] [resolved (round 1: recorded as a declared deviation)] `spec-change`
**Overturns an orchestrator edit made this round.** I reasoned from the absence of camera
code in `_smart_grid/` that napari shares a camera *value*. Verified in napari source:
`VispyCamera(view, self.viewer.camera, …)` **per grid view** appended to `grid_cameras`
(`_vispy/canvas.py:1121-1123`), each event-connected to the shared model
(`_vispy/camera.py:50-56`), reconciled every draw under `# sync all cameras`
(`canvas.py:646-648`). It **is** a sync protocol. The deck.gl single-`viewState` design is a
**declared deviation**, not fidelity — spec §6.2's "sharing one camera" needs the same
correction.

### ALGO-2 [Critical] [resolved (round 1: per-view target specified, test strengthened)]
deck.gl `target` lives in **viewState**, not `View`. As specified, every cell inherited one
target and the grid rendered **the same colony N times** — and the e2e test asserted only
`len(set(zooms)) == 1`, which that bug satisfies perfectly. The one test guarding the task
passed hardest when the task was most wrong.

### ALGO-8 [Major] [open]
`test_recorded_ladder_matches_the_ceil_formula` computes the formula **in the test body**
and compares it to its own parametrization; it imports `select_pyramid_level` and never
calls it, so it would pass with `phenotypic` uninstalled and cannot catch the `floor`
regression its docstring names. Its companion `assert level >= 0` is unconditional.
`rgb_store` is undefined anywhere. **Positive:** all five parametrized values recomputed
against `ngff_.py:167-169` and are correct.

### ALGO-3/4/5, ALGO-6/7, GEN-13 [open]
Port description omits the visible-layers predicate's overlay coupling, the detach +
draw-order mechanism, and the two user toggles (partly applied); `cleanup_clones`' narrow
claim holds but the leak *class* survives as an unbounded Viv `TileLayer` texture cache;
`colony_view_budget.py` models 64² D3 crops rather than D1's 1024² tiles, and nothing
re-runs it as a gate.

### Still open at end of round 1
GEN-1..5, GEN-10..17 (minus resolved), FLOW-1 (token specified, invalidation cost unsettled),
ALGO-3..8. **Not converged.** Round 2 is diff-scoped verification per the cadence rules.

---

## Round 2 — diff-scoped verification

Full panel re-dispatched against the round-0 → round-1 diffs, plus a fresh resolution
verifier. **No Critical remains anywhere.**

| Reviewer | R1 | R2 |
|---|---|---|
| security | REVISE | **APPROVE** |
| simplicity | BLOCKING | **NON-BLOCKING** |
| general | BLOCKING | REVISE (was BLOCKING on GEN-6 residue, now fixed) |
| data-flow | BLOCKING | REVISE |
| algorithm-fidelity | REVISE | REVISE |

### GEN-6 [Critical] [resolved (round 2: Files headers anchored, task 2.6 corrected)]
**The round-1 fix was itself incomplete — flagged independently by general and simplicity.**
Prose and the README table were corrected; the dangerous ranges survived in two **Files**
headers, and **phase 2 task 2.6 — the task that deletes the rows — had no in-task correction
at all**, plus a stale conditional contradicting phase 1's new instruction. Both headers now
anchor on content, and the conditional is gone. General independently re-derived all eight
offsets in the README table and confirmed every cell.

### ALGO-12 [Major] [resolved (round 2: phase 0 step 1 split, step 1b added)]
**Verified empirically by the orchestrator:** `python -m http.server` has no `Range` support
— `curl -r 0-15` against a 100,000-byte file returned `code=200 size=100000`. Phase 0's gate
expected an unreachable `206`, and since Q1/Q2/Q4 drove the same server the **entire spike
would have measured the no-Range regime** — the one where shard amplification bites — with
no byte counts recorded to notice. Step 1 now uses `http.server` only for metadata questions
and a Range-capable Werkzeug handler for the byte question; step 1b records the ~16×
amplification number (≈3 MB per tile with Range vs ≈50 MB shard without), which existed in
no document.

### ALGO-11 [Major] [resolved (round 2: switched to the `View.viewState` merge form)]
**Overturns the round-1 fix.** The keyed-`viewState` map is correct deck.gl
(`docs/developer-guide/views.md`) but **is a sync protocol**: `onViewStateChange` fires with
`{viewId, viewState}` for the one view a gesture touched, so keeping `zoom` common requires
fanning it back across every entry — the per-view reconciliation task 4.2 disclaims. Now
uses `View.viewState` merge (`docs/api-reference/core/view.md`): one dynamic viewState
carrying `zoom`, each `View` overriding only `target`. One zoom, so it cannot drift.

### FLOW-20 [High] [resolved (round 2: 409 client contract specified)]
The token closed the **server** half of FLOW-1 and left the client half unwritten. Zarr fills
an unreadable chunk with `fill_value` and store implementations commonly map a failed fetch
to "absent" — so a swallowed 409 renders **black tiles** after every promote: the
plausible-wrong-pixels failure the token exists to prevent, relocated to the client and made
*harder* to see, because a black tile reads as empty data. Façade now carries
`onChunkResponse` with a three-outcome contract (404 retry / 409 re-fetch spec + `setSource`
/ 422 surface) and a `refetchSource` hook, without which a 409 is unrecoverable until reload.

### FLOW-24 [Medium] [resolved (round 2)] — **orchestrator error, introduced by the FLOW-9 fix**
`tests/unit/gui/results_viewer/colony_view` does not exist (that package is at
`tests/gui/results_viewer/colony_view/`). pytest exits 4 on an unknown path and runs **none**
of the others in the command — including the curation test the task exists to run. Same class
as GEN-8, reintroduced inside the fix for it.

### SIMP-14 / GEN-14 / FLOW-23 [Major] [resolved (round 2: test restored as step 4b)]
Round 1 replaced task 1.2 wholesale and **deleted spec §8's nested-chunk staleness test**
while four documents still promised it. The token does not cover it — the token keys on the
root `zarr.json`, which an in-place chunk rewrite does not touch, so the URL stays valid,
which is correct and is the property worth pinning. Restored, now also asserting the token
did *not* move.

### FLOW-21 [Medium-High] [resolved (round 2)] · SEC-11 [Minor] [resolved]
Same defect from two angles. `root.resolve(strict=True)` sat **outside** the try; the token
and readable-roots calls were unguarded. `promote_store` renames the whole store directory,
so that window is the **routine** path. Both resolves are now inside the try; both routes
guard. Two round-2 residues also fixed: `KeyError` named (`require_readable_store` raises
it and it is not an `OSError`), and `StoreUnreadable` → **422** to match what Colony already
does, rather than 404 — which would say "no such image" when the truth is a run-wide
decode failure.

### SEC-8 [Minor] [resolved (round 2: `mode=0o700`)]
Re-raised on the reviewer's argument that it should not have travelled with the dropped
scratch phase — a directory mode is not a retention cap. It matters *more* now: spec §7 makes
session-id secrecy the recorded mitigation while the directories are **named by that secret**
at default umask.

### SEC-13 [Advisory] [resolved] · SEC-12 [Advisory] [advisory, no action]
`allowed_roots` is now required with no permissive value. SEC-12's 409-vs-404 existence
oracle accepted: 409's diagnostic value exceeds the leak, and the landed `/preview-tiles/`
route already signals existence the same way.

### Also resolved in round 2
GEN-5 (façade Interfaces contradicted its own implementation), GEN-13 (cap script now
re-run in phase 5, with a skip-if-phase-4-cut clause), GEN-15 (codec test now asserts a read
**throws**), GEN-17 (asset mount inverted — Dash takes one `assets_folder` per app and the
builder's holds eight files), GEN-18 (README dependency contradicted phase 5's own header),
GEN-19, GEN-20, ALGO-3/4/5 (all four deviations now ride one gate; the `TileLayer` bound is
closed-form ≈36 MB/level, not linear in cell count — a partial walk-back of ALGO-5's alarm),
ALGO-6/7 (budget script now models the tile **union**, since cells are viewports onto one
store), ALGO-13, SIMP-11 (phase 3 cut-order), SIMP-15 (phase 5 runs last), SIMP-17 (orphaned
stack decided: retire), SIMP-13(2) + GEN-8 (both DoD gates widened to `tests/gui`), FLOW-25,
FLOW-26.

### GEN-4 [withdrawn as moot]
The consumer-set test was dropped under SIMP-9, so its dependency on removals phase 2 no
longer exists. Confirmed by the reviewer who raised it.

### Open at end of round 2
Minors only: GEN-10, GEN-11, GEN-12 (both nuances now applied — reviewer read a stale state),
SIMP-12 (class-B "an earlier draft" notes), SIMP-16 (relocate the port drift register),
FLOW-10 (spec §3 clause), FLOW-22 (memoization — reviewer confirmed already applied).

### Gated to the user, unapplied
1. **Viv spec §6.2** still says the napari grid shares "one camera" — the strawman ALGO-1
   refuted. The plan is corrected; the spec is not.
2. **Removals spec §6** still carries `FEATURES.md:372-394` and the seven other stale
   ranges. General recommends batching this with USER-3 rather than deferring: §6 currently
   instructs a violation of §5 in the same document, and the fix is eight heading names for
   eight line ranges.
