# What this review slice is, and what it deliberately leaves out

**This branch is a REVIEW SLICE, not a feature branch.** It is `main` plus one
subset of the change on `cli-gui-state-tracking`, cut so the diff fits a review
budget. It is disposable and is never merged.

## You are seeing tests without the source they exercise

The implementation lives in the sibling slice **`review/state-and-migrate`**
(27 files, 7,931 lines), which carries all source for the run-state model and
`--mode migrate`. The two together exceed the budget, so they are reviewed
separately.

**Read these as instruments, not as the change.** The useful question here is
*"could this test fail?"* — not *"is the implementation correct?"*, which the
sibling slice is for.

| File | tests | Exercises |
|---|---|---|
| `tests/unit/cli/test_migrate_state.py` | 59 | migrate planners/appliers, retention, `--revert` |
| `tests/unit/sdk_/test_run_state.py` | 56 | verdict precedence, liveness fence, schema advisories |
| `tests/unit/cli/test_run_identity.py` | 37 | the five identity tokens; what does and does not mint a new one |
| `tests/unit/cli/test_image_record.py` | 24 | per-image records, `record_rejection`, provenance fencing |
| `tests/unit/sdk_/test_verification_cache_disk.py` | 29 | the on-disk cache tier, and that it never yields a verdict alone |
| `tests/unit/sdk_/test_verification_cache.py` | 20 | the in-process cache tier |

## Conventions that are deliberate, not accidental

- **Docstrings carry a "fires when" clause.** They state what deleting the
  behaviour under test would break. That is the test's claim; if the body
  cannot fail for the stated reason, that is a finding worth reporting.
- **Paired opposite-outcome tests are intentional** (`..._still_counts_on_an_
  unrestarted_run` / `..._is_fenced_on_a_restarted_run`). Both halves are
  needed or the fence is unproven; they are not duplicates.
- **`xfail(strict=True)` is used as an instrument**, so a mark that stops being
  true fails loudly rather than passing silently.

## Full context

- Branch under review: `cli-gui-state-tracking` (241 commits)
- Regression at `92f55986`: **zero failing names, 12,535 tests, 24 shards**
- State model reference: `docs/source/contrib_guide/tracked_state.md`
