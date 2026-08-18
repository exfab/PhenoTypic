# C2 — orchestrator decisions (authoritative)

Written 2026-08-18. **Messages from team-lead → C2 are not arriving** (approval
sent twice, both accepted into the inbox, neither seen). This file is the
authoritative channel. C2: read this, act on it, do not wait for a message.

---

## DECISION 1 — Option 1 APPROVED. Proceed with T7 part 2.

Promote the three small ones; allowlist only `_metadata_context`.

| Move | To | Why |
|---|---|---|
| `grid_feasibility` (11 lines) | `_services/tune_spec.py` | beside its only production consumer |
| `sandbox_fingerprint` (3 lines) | `_services/sandbox.py` | it hashes a `SandboxRoot` and takes nothing else — arguably its correct home |
| `tune_presets_dir` + 3 `SANDBOX_*` constants | `sdk_/_io_constants.py` | it is a path helper, and the Global Constraints already say every artifact path resolves there. This is Task 2's `IMAGE_EXTS` move repeated — follow that shape exactly |
| `resolve_metadata_csv` | **ALLOWLIST** | 596 lines, transitively `_source_context` → `_classifier`. Not promotable at this scope |

`GUI_IMPORT_ALLOWLIST` goes from 1 entry to 2. That is the approved outcome.

### Why not the alternatives

- **Option 2 (4 allowlist entries)** is the failure mode named in the brief. An
  allowlist that absorbs every inconvenient import is not a boundary, it is a
  comment.
- **Option 3 (drop `_setup_authoring`)** looks cheaper and is not: `tune_spec.py`
  would ship incomplete and Phase 2B's `tune_put_spec` needs exactly those
  authoring functions, so the problem returns in a colder context.

### Conditions

1. **Every promotion keeps a re-export at the old location.** Derive each shim
   surface from the code (AST), never from `__all__` or a list — you have already
   proven why twice, most recently with `_params_from_best_params_payload` and
   `render_tokens`.
2. **The allowlist entry carries a comment** naming what it wraps and that it is
   temporary, pending a later phase promoting or inverting `_metadata_context`.
   An entry with no stated expiry is how a boundary rots. The expiry is tracked
   in this plan.
3. **Mutation-test the new allowlist entry**: add an un-allowlisted
   `phenotypic.gui` import to `tune_spec.py`, confirm the tier gate fires.
4. **Do NOT promote `resolve_metadata_csv` or `_metadata_context`.** Inverting
   that dependency is a real design decision and belongs to a phase with room
   for it.

## DECISION 2 — YES, fix the tier-gate hole. Separate commit.

`test_import_purity.py::test_service_module_does_not_import_gui` misses
`from phenotypic import gui`; your M3 proved it. It is C1's file and outside your
nominal scope, and you were right to ask — but it is the orchestrator's own
defect, written weaker than the `argv` version you were told to copy. Leaving a
known hole in the boundary *while adding an allowlist entry to it* is
indefensible: the allowlist only means anything if the gate around it is sound.

Take your two-line fix, re-run M3 to prove `from phenotypic import gui` now
fails, and commit it **separately** from T7 so the history shows the boundary
being strengthened rather than the fix hiding inside a feature commit.

## DECISION 3 — D1, D2, D4 all approved

- **D1 (lazy PEP 562 shim)** is not just approved, it is a **plan defect you
  caught**. The plan's eager `from ._space_view import ...` sketch keeps `dash`
  on `gui.tune._space`'s import path, so under the literal plan the split would
  have been *cosmetic* and the strict xfail would never have flipped. That the
  plan's own success criterion would not have fired is the tell.
- **D2 (structural Protocol)** is the right inversion. `ast.walk` descends into
  `if TYPE_CHECKING:` bodies, so the annotation import is rejected too.
- **D4 (verbatim merge, module-level imports preserved)** is correct and is the
  judgment I want. Rewriting them lazy would be a behaviour change riding inside
  a move — precisely what the review protocol tells reviewers to hunt for.
  Keeping the originals and dropping the four duplicates is right, and verifying
  `tune._spec` / `._evaluation` stay optuna-free was the necessary check.

## Standing lesson promoted from M7

**No assertion in this codebase may be a substring search over source text.**
Your M7 reproduced that failure in a test written from the orchestrator's own
idiom: it passed on a *docstring mention* after the real import was deleted.
Third instance in this project. Use parsed imports (AST) or a runtime probe.
This is being carried into the plan's Global Constraints.
