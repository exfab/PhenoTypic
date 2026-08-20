# C7b cluster gate — journal storage failure semantics (B1–B4)

**Scope.** `e5c5f5d0` (B1, B2, B4) and `1b55076c` (B3), branch `feat/mcp-server`
(both confirmed ancestors of `HEAD` = `7dc9e916`). Analysis only — no files under
`src/` or `tests/` were modified; the working tree was restored after every
mutation (`git status` clean).

---

## Verdict

**C7b is NOT sound as committed.** Two blockers, both in load-bearing behaviour
the commits explicitly reason about and get wrong:

1. **B1's retry converts a survivable worker death into permanent destruction of
   the shared study.** Reproduced deterministically: a torn append + the bounded
   retry + one further record makes every subsequent open of `journal.log` raise
   `json.JSONDecodeError` forever — and the new predicate classifies that
   permanent corruption as *transient*. The docstring's stated basis ("optuna's
   reader tolerates a torn *trailing* line") is only true while the torn line is
   still trailing, which on a live fleet it stops being within one trial.
2. **B3's 3.0 s bound is smaller than the read it now bounds.** Measured against
   a real journal study: the snapshot costs **1.9–2.4 s at 200 trials** and
   **5.3–6.0 s at 400 trials**, ~96% of it fANOVA. At the commit's own "real
   case" (200 trials) the Monitor sits at 65–80% of its budget on an idle node;
   past ~250 trials it exceeds it on *every* tick, so the live view is
   permanently replaced by a fallback that, mid-run, does not exist.

Plus **three confirmed false greens** in B3's suite: `_read_importances` always
returning `None` **and** `_best=None` both leave the entire tune + tune-GUI
suite green (1092 passed, 2 skipped).

B2 and B4 are sound in substance; each carries one non-blocking defect.

---

## Blockers

### BLOCK-1 — the retry can permanently destroy the shared study, and calls it transient

`src/phenotypic/tune/strategy/_optuna_support.py:404` (the `JSONDecodeError`
arm) + `:436` (the survivability claim in `retry_on_transient_db_error`).

The docstring reasons:

> a retry after a partial append can leave a torn record in the log. Both are
> survivable — a non-terminal trial is excluded from winner selection and from
> the budget gate, and optuna's reader tolerates a torn *trailing* line

Half of that is right, half is not. Verified against optuna 4.9.0
(`optuna/storages/journal/_file.py:read_logs`): a bad line sets
`last_decode_error` and `continue`s; the error is raised on the **next**
iteration (`if last_decode_error is not None: raise last_decode_error`). So a
torn record is skipped only while it is the file's last line. As soon as any
worker appends after it, every reader raises — including the writer's own
`_sync_with_backend`, and including a `create=True` open, so the whole fleet
dies at startup.

Reproduced (deterministic, `optuna 4.9.0`):

```
torn write + bounded retry re-appends + one more record
  -> after torn+retry+one more record: JSONDecodeError | predicate says transient=True
  -> retry #2 (same log):               JSONDecodeError | predicate says transient=True
```

and, on a `create=True` open (a fresh worker joining the fleet), the failure is
raised out of `optuna.create_study` → `JournalStorage._sync_with_backend` →
`apply_logs` → `read_logs`.

Two consequences:

* **The retry manufactures the corruption it cannot recover from.** The failure
  B1 exists to absorb — an `EIO` on the GPFS `write`/`flush` inside
  `JournalFileBackend.append_logs` — is exactly the failure that can leave a
  short write on disk. Before B1 that killed one worker. After B1 the worker
  survives, re-appends, and the next record any worker writes destroys the
  study for everyone. The blast radius went from *one Slurm task* to *the whole
  campaign*.
* **The predicate is lying about the class.** `is_transient_storage_error`
  returns `True` for that `JSONDecodeError` (`:404`), so
  `retry_on_transient_db_error` retries a permanently-dead study three times.
  Bounded, so not a hang — but "transient" is documented as "clears on its own",
  and this never does.

The *other* half of the docstring's claim checks out and should be kept: a
**duplicated complete** record (the fsync-failed-after-the-bytes-landed case)
replays cleanly — verified, `trials=3, ['COMPLETE','COMPLETE','COMPLETE']`.

**Fix.** Repair the torn tail rather than reasoning that it is harmless. Under
the journal's own lock, on open (and before any retried append), if the log's
last byte is not `\n`, truncate to the last `\n`. This is safe *specifically*
for the trailing record and no other: `read_logs` deletes the offset entry for a
bad line (`del self._log_number_offset[log_number + 1]`), so no live reader holds
a byte offset into it, and an incomplete trailing record was never acknowledged
to anybody. Verified: truncate-to-last-newline restores a study to its full
pre-corruption trial set. Note this contradicts the blanket "never edit the log"
in the new docs paragraph (see NB-5) — the general no-compaction argument is
correct, but it should carve out the trailing-record repair, which is the one
edit that is both safe and required.

Failing test to add: build a journal study, inject a *short write* into
`JournalFileBackend.append_logs` (write `payload[:k]` then raise `OSError(EIO)`)
rather than raising before the write as
`test_a_transient_journal_append_failure_no_longer_kills_the_worker`
(`tests/unit/tune/test_journal_failure_semantics.py:200-234`) does today, let the
retry land, then append one more trial and assert a fresh
`OptunaStudyStore(create=False)` still reads every trial. That test is RED today.

### BLOCK-2 — B3's bound is smaller than the read, so the live Monitor degrades permanently past ~250 trials

`src/phenotypic/gui/tune/_callbacks.py:124` (`_LIVE_READ_TIMEOUT_S = 3.0`),
`:912` (`_snapshot_live_study`), `:945` (`_importances=_read_importances(store)`),
`src/phenotypic/gui/tune/_layout.py:534,666` (`dcc.Interval(interval=3000)`).

Measured against a real `journal://` study built through the real
`ask → set_trial_user_attrs → tell` path (12 knobs, 6 terms, a quarter pruned —
the same workload B4's own growth test uses), timing exactly what
`_snapshot_live_study` does:

| trials | open | trials | best | pareto | **param_importances** | **total** |
|---|---|---|---|---|---|---|
| 200 | 0.066–0.077 s | 0.003 s | 0.003 s | 0.000 s | **1.87–2.36 s** | **1.94–2.44 s** |
| 400 | 0.128–0.145 s | 0.004 s | 0.004 s | 0.000 s | **5.12–5.82 s** | **5.26–5.97 s** |

(idle compute node `r29`, journal on node-local `/scratch`; a 200-trial log on
GPFS `/bigdata` opened in 0.041 s, so the filesystem is not the variable — fANOVA
is 96% of the cost.)

The commit's justification is:

> The cost is that an expensive fANOVA now spends the same budget — but what it
> replaced was freezing the Dash worker for exactly as long, so a degraded tick
> is strictly better.

It is not the same budget, and a degraded tick is not strictly better:

* **Before:** the open (0.07 s) fitted the bound; the reads then ran unbounded on
  the Dash thread and **completed**. The user saw a correct live view, slowly.
* **After:** the whole read shares one 3.0 s budget it cannot make at ordinary
  campaign size. Every tick times out, so the live view is *never* shown again
  for that run — not degraded once, degraded permanently.
* **The fallback is empty mid-run.** `_load_journal` (`:804`) reads
  `trials.parquet`, and `trials.parquet` is written only at export/finalize
  (`_tune_cli/_run.py:1150-1185`, `_export_trials_parquet`). During the live run
  it does not exist, so `_load_journal` returns `None` and the Monitor renders
  **"No trials yet."** for the entire campaign.
* **The note is wrong.** `_NOTE_LIVE_UNREACHABLE` (`:147`) says *"couldn't reach
  the live study -- check network / ~/.pgpass"* — for a local-file backend that
  was reached and read successfully, and merely had a slow fANOVA. This is the
  dishonest-degradation case: the UI blames the network for a compute cost.
* **Unbounded backlog.** `_LIVE_OPEN_POOL` is `max_workers=1` (`:141`).
  Enqueue rate is one job / 3 s; drain rate one / ~5.5 s at 400 trials. The
  queue grows without bound for the life of the page, each abandoned job running
  a full fANOVA, pegging one core on the GUI host indefinitely. The comment at
  `:138-140` reasons about this correctly for the *open* it was written for ("a
  poll that arrives while a previous open is still in flight simply queues
  behind it and will itself time out") — that argument depended on the queued
  work being ~70 ms, and no longer holds.

**Fix.** Take the importances out of the shared budget; they are the only
expensive read and the only optional one. Either (a) give
`_read_importances` its own future/timeout so `trials`/`best` still land inside
3.0 s and the figure degrades alone (which is what `_read_importances`' own
docstring claims it does — true for a *failing* fANOVA, false for a slow one), or
(b) memoize importances keyed on trial count and recompute at most every Nth
tick. (b) is the smaller change and matches how the figure is actually read.
Separately, `_monitor_degrade_note` should distinguish "timed out" from "could
not connect" — a `journal://` URL should never tell the user to check
`~/.pgpass`.

---

## False greens

All three were confirmed by mutation: the mutant was applied to `src/`, the
suite run, and the tree restored. Baseline for the four C7b files is 48 passed.

| # | Mutation | Result |
|---|---|---|
| FG-1 | `_read_importances` (`_callbacks.py:949`) → `return None` unconditionally | **48 passed** |
| FG-2 | `_best=store.best()` (`:943`) → `_best=None` | **48 passed** |
| FG-3 | `_pareto_front=list(store.pareto_front())` (`:944`) → `_pareto_front=[]` | **48 passed** |
| control | `_snapshot_live_study` returns the store unmaterialized (reverts B3) | 3 failed, 45 passed ✅ |

FG-1 and FG-2 were re-confirmed **repo-wide**, applied together, across
`tests/unit/tune tests/unit/gui/tune tests/integration/gui/test_tune_live_timeout.py
tests/integration/gui/test_tune_monitor.py`: **1092 passed, 2 skipped**.

Why they survive: no test in the repository names `_snapshot_live_study`,
`_StudySnapshot`, or `_read_importances` (`grep -rl` over `tests/` → only
`tests/unit/gui/tune/test_study_read.py`, which tests `gap_badge` as a pure
function over a hand-built store). The three B3 tests all monkeypatch
`_open_live_study` and reach the snapshot only indirectly:

* `test_the_returned_snapshot_never_touches_storage_again`
  (`tests/integration/gui/test_tune_live_timeout.py:264-289`) asserts
  `store.reads == 1`, but `_CountingStore` increments `reads` **only in
  `trials`** (`:213`) — `best`, `pareto_front`, and `param_importances` are not
  counted, and their returned values are never asserted. The loop at `:283-287`
  calls all four and checks none of them.
* `test_a_degenerate_importance_model_costs_only_the_importance_figure`
  (`:292-309`) asserts `param_importances() is None` — the mutant's return value.

**User-visible consequence** of what these hide: the Monitor's fANOVA importance
bar chart silently empty forever (FG-1), and the gap badge — the run-quality
signal it exists for — stuck at `--`/stable forever (FG-2), with a fully green
suite either way.

**Fix.** One test asserting `_snapshot_live_study` against a **real**
`OptunaStudyStore` over a real journal study (3–4 trials is enough) — that the
snapshot's `trials`, `best()`, `pareto_front()`, and `param_importances()` equal
what the live store returns. Today no test constructs a real store for this path
at all, which is also why `pareto_front()`'s single-objective `[]` contract
(`_optuna_store.py:537-546`) is unverified through the snapshot.

---

## Non-blocking

**NB-1 — `backing_file_for_url` mis-resolves a *relative* SQLite URL, and the
doctest codifies the error.** `src/phenotypic/tune/_study/_optuna_store.py:120`
takes `urlsplit(url).path`, which for the three-slash form
`sqlite:///out/.pht-tune-cache/study.db` yields `/out/.pht-tune-cache/study.db` —
an absolute filesystem-root path. SQLAlchemy reads that same URL as the
**relative** path `out/.pht-tune-cache/study.db`. Verified live:

```
relative file created: True 8192
backing_file_for_url -> /out/.pht-tune-cache/study.db
guard: RAISED FileNotFoundError /out/.pht-tune-cache/study.db
```

`_default_study_db_url` (`_tune_cli/_run.py:101`) does **not** absolutize (unlike
`_default_journal_url`, which does so deliberately), so `-o out` produces exactly
this URL. Blast radius is bounded today because `create=False` has exactly one
call site (`gui/tune/_callbacks.py:882`) — the Monitor, which then degrades
permanently for any local sqlite run started with a relative `-o`. This is
**pre-existing** (the old inline check in `_open_live_study` computed the same
wrong path), but B2 promotes it into shared library API with a doctest at `:107`
asserting the wrong mapping, and the commit message says "the CLI's read-only
opens get it too" — which is aspirational today and would newly break CLI resume
the moment it becomes true. Doctests are not collected (`pyproject.toml:203`
carries no `--doctest-modules`), so nothing catches it. Fix: extract the database
as `urlsplit(url).path[1:]` — SQLAlchemy's actual rule — which also handles
`sqlite:////abs/x.db` → `/abs/x.db` and `sqlite:///C:/x.db` → `C:/x.db`.

**NB-2 — B1 taught the predicate about journal failures but did not extend the
retry to the journal's read path.** `retry_on_transient_db_error` still wraps
only `ask` / `set_trial_user_attrs` / `tell` (`strategy/_optuna.py:301,412-446`).
Unwrapped, and each a `JournalStorage` sync that hits the failing filesystem:
`is_exhausted()` (`:453-467`, `get_trials` — once per engine loop iteration),
`OptunaPruningChannel.report` (`:144-146` — a storage *append*, once per ASHA
rung per image) and `.should_prune()` (`:148-150` — a storage read, same
frequency). On a pruned run these are an order of magnitude more storage touches
than ask/tell, and a GPFS `EIO` in any of them still kills the worker on first
occurrence — the exact failure the commit message says is now absorbed. Strictly
this is pre-existing coverage (the RDB era wrapped the same three sites), so it
is a scope gap rather than a regression, but the commit message overstates what
was fixed. `_fail_stale_trials` is fine — it swallows everything (`:117-126`).

**NB-3 — B4's test pins the rate at one point, not the linearity the conclusion
rests on.** `test_journal_growth_per_trial_stays_near_the_measured_rate`
measures at `_GROWTH_TRIALS = 20` only. The extrapolation to 11,000 trials
(`_JOURNAL_SIZE_WARN_BYTES`) needs *linear* growth, which the single point
cannot fail on. I confirmed linearity independently today — 200 trials → 1.17 MB
(5,850 B/trial), 400 trials → 2.34 MB (5,850 B/trial), matching the commit's
quoted 1,168,306 B at 200 exactly — so the claim is true for optuna 4.9.0, just
unguarded. Cheap fix: measure at two sizes and assert the ratio.

**NB-4 — B4 is advisory only, and that is the right call at the measured
scale.** Nothing enforces the 64 MiB bound; `warn_if_journal_oversized`
(`_optuna_store.py:157-184`) logs and returns, swallowing stat failures. The
measurement supports the conclusion: 200 trials = 1.17 MB, replayed in 0.066 s
locally and 0.041 s on GPFS, i.e. the "0.07 s" claim reproduces and GPFS is not
worse. The remedy it names (fresh `-o`, or Postgres) is the only one available
given byte-offset addressing. No change needed — with the NB-5 caveat.

**NB-5 — the docs paragraph is accurate about disk and wrong about the
Monitor.** `docs/source/how_to/pages/tune_distributed_hpcc.md` now says "every
worker start and every Monitor poll re-reads the whole log — 0.07 s at 200
trials … Nothing needs doing at campaign scale." The replay figure is right; but
the Monitor poll's actual per-tick cost is 1.9–2.4 s at 200 trials, because it is
dominated by fANOVA, not replay (BLOCK-2). Separately, "never editing the log"
is too broad — see BLOCK-1's carve-out for the trailing-record repair.

**NB-6 — `_LIVE_CONNECT_TIMEOUT_S` and `_LIVE_READ_TIMEOUT_S` are both 3.0 and
nest.** A Postgres connect is allowed the full 3 s at the URL level
(`_ensure_connect_timeout`) *inside* the 3 s total read bound, so the connect
alone can consume the entire budget and leave nothing for the reads. Pre-existing
in effect, but worth making the inner bound a fraction of the outer.

---

## What was verified sound

* **B2's core claim.** A `create=False` open of an absent journal raises
  `FileNotFoundError` and creates neither the log nor its parent tree
  (independently re-confirmed); an existing study still opens; server-backed
  (`postgresql+psycopg://…`) and in-memory (`sqlite:///:memory:`) URLs return
  `None` from `backing_file_for_url` and are not probed — confirmed by both code
  and test. The legitimate "Monitor opened moments before the first worker
  writes" case degrades correctly and self-heals on the next 3 s tick once the
  log appears (the guard re-runs every poll). Its note is slightly
  misleading — "check network / ~/.pgpass" for "the run has not started" — the
  same `_monitor_degrade_note` defect as BLOCK-2.
* **B1's end-to-end retry test is real, not a predicate assertion.**
  `test_a_transient_journal_append_failure_no_longer_kills_the_worker` injects a
  real `OSError(EIO)` into the real `JournalFileBackend.append_logs` during a
  real `study.ask()` against a real journal study, and asserts the injection
  fired (`failures == [1]`). It fails if the predicate reverts.
  `test_a_real_torn_append_raises_something_the_predicate_accepts` likewise
  drives the real reader over a real torn log. Neither is a false green — the gap
  is that neither injects a *short write* (BLOCK-1).
* **The predicate is not too broad in the latency sense.** Every retry path is
  bounded at 3 attempts / 0.3 s total; `exc.errno is None` falls through to
  `False`; the exact-message `ValueError` arm is pinned against the installed
  optuna's source by
  `test_journal_torn_line_message_is_still_optunas`. The excluded classes
  (`ENOSPC`/`EDQUOT`/`EACCES`/`EROFS`/`ENOENT`, the stolen-lock `RuntimeError`)
  are excluded for the stated reasons and tested. Its breadth problem is
  semantic, not temporal (BLOCK-1).
* **B3's core mechanism.** The reads genuinely run inside the bound: the hanging
  read test gates a real block inside `_snapshot_live_study` and proves the poll
  returns without joining the worker, and the control mutation (returning the
  store unmaterialized) turns the suite red.
* **Test-suite hygiene.** 48 passed on the four C7b files; 1092 passed / 2
  skipped repo-wide across `tests/unit/tune`, `tests/unit/gui/tune`, and the two
  GUI integration files, run with `-rs` and `QT_QPA_PLATFORM=offscreen`. Both
  skips are the Postgres-gated pair. optuna 4.9.0 present — nothing skipped
  silently.

---

## Reproduction

Measurements were taken with throwaway scripts under the session scratchpad
(`bench_snapshot.py` and inline heredocs); none touch the repository. Key
commands:

* snapshot cost: build N trials through `store.study.ask()` /
  `set_trial_user_attrs` / `tell`, then time
  `OptunaStudyStore(create=False)` + `.trials` + `.best()` + `.pareto_front()` +
  `.param_importances()`.
* corruption: append `record[:30]` (no newline) to a healthy `journal.log`, then
  the full record, then one more, then reopen.
* mutations: patch `src/`, run the four C7b test files with
  `-q -p no:randomly`, `git checkout -- src tests`.
