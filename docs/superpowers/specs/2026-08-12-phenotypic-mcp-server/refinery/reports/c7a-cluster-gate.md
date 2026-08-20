# C7a cluster gate — `journal://` storage dispatch

**Commit:** `933ce52e` "feat(tune): dispatch storage by scheme so a fleet can share a journal (P1/C7a)"
**Branch:** `feat/mcp-server` @ `/bigdata/exfab/anguy344/PhenoTypic`
**Scope:** correctness of the change as committed, and whether its tests prove it.
**Method:** read + reproduction. Every claim below is backed by a run recorded in this file, not by inspection.

---

## Verdict

**One blocker, one required doc correction. Otherwise sound.**

The design is right and the dispatch is correct. But:

1. **B1 (blocker) — the documented default invocation crashes.** `-o ./out --slurm`, the
   exact command this commit adds to `tune_distributed_hpcc.md`, dies with
   `PermissionError: '/out'` after leaving a half-written run directory. Any relative `-o`
   — including the CLI's own default, `./<name>_tune` — is silently reinterpreted as an
   absolute path rooted at `/`. Sibling spellings (`-o tmp/run1`) land on a *writable but
   node-local* `/tmp/run1`, where the fleet does not share a journal at all and nothing
   errors until finalize. This is C7a's own new code and nothing else in flight touches it.
2. **B2 (doc correction) — C7a's shipped description of its own limitation is wrong.** The
   how-to and the module docstring call the missing heartbeat cosmetic ("zombie rows and a
   slightly optimistic in-flight count"). Measured, it is not: under the journal default a
   zombie `RUNNING` trial *wins* the study and publishes the untuned base pipeline. See
   below — this is the answer to the gate's central question, and it is a real widening.

Everything else checks out. The dispatch is threaded through every construction site,
the H1 SQLite refusal is not bypassable by URL spelling, unknown and malformed schemes
fail loudly, and the fleet-sharing property holds under real concurrency (I verified it;
the suite does not).

---

## B1 (blocker) — a relative `-o` under `--slurm` writes to the filesystem root

`src/phenotypic/tune/_study/_storage.py:109` (`journal_url_for_path`) prepends `/` to any
path that does not already start with one, on the assumption — stated in its own docstring
at `:96` ("the absolute append-only log path") — that callers pass an absolute path. The
caller does not:

- `src/phenotypic/tune/_tune_cli/_run.py:115` — `_default_journal_url` passes
  `io.tune_cache_journal_path(output_dir)` verbatim.
- `src/phenotypic/tune/_tune_cli/_run.py:625` — `run_tuning` does `output_dir = Path(output_dir)`, no `resolve()`.
- `src/phenotypic/tune/__main__.py:253` — `output_dir = Path(args.output) if args.output else _default_output(args.input)`.
- `src/phenotypic/tune/__main__.py:37` — `_default_output` returns `Path(f"./{...}_tune")`, i.e. **relative by default**.

The SQLite default does not have this problem: `sqlite:///out/x.db` is a *relative* SQLite
path to SQLAlchemy, so the two run-local defaults disagree about what a relative path
means. Only the journal branch absolutizes.

Reproduction of the documented command
(`docs/source/how_to/pages/tune_distributed_hpcc.md`, the `-o ./out ... --slurm` example
this commit adds), with `SlurmExecutor` faked so nothing is submitted:

```
cwd: /scratch/anguy344/27568810/tmpcvjt0ye4
RAISED: PermissionError [Errno 13] Permission denied: '/out'
marker written? True
tuning_spec written? True
marker storage_url: journal:///out/.pht-tune-cache/journal.log
```

Two things to note. First, it raises **after** `run.json` and `tuning_spec.json` have
landed — precisely the half-written-directory condition
`test_the_sqlite_refusal_lands_before_any_artifact` exists to prevent for the sibling H1
guard. Second, the marker records a URL pointing at `/out`, so the GUI Monitor will chase
that path too.

The silent variant is worse. `-o tmp/run1 --slurm` resolves to `/tmp/run1/...`:

```
relative slurm url: journal:///outdir/.pht-tune-cache/journal.log -> path: /outdir/.pht-tune-cache/journal.log  absolute? True
local sqlite for same: sqlite:///outdir/.pht-tune-cache/study.db
relative tmp/run1 -> /tmp/run1/.pht-tune-cache/journal.log
```

`/tmp` is writable and **node-local**. The submitter pre-creates a study in the head
node's `/tmp`; each compute node opens its own `/tmp` copy, finds no study, and
(`create=True`) starts a private one. N workers run N independent studies, the shared
budget never drains as intended, and finalize reads the head node's empty copy. Nothing
errors until the very end.

**Fix.** Absolutize at the resolver, in `_default_journal_url` (`_run.py:115`):

```python
return journal_url_for_path(io.tune_cache_journal_path(Path(output_dir).absolute()))
```

Prefer `.absolute()` over `.resolve()` on this cluster: `resolve()` also collapses
symlinks, and `/rhome` ↔ `/bigdata` symlink layouts can make the submitter's resolved path
differ from what an operator typed. Additionally make `journal_url_for_path` enforce its
own contract — raise `ValueError` on a relative path rather than silently rooting it — so
the next caller cannot reintroduce this.

**Related, same function.** `urlsplit` puts the first segment of a two-slash URL in
`netloc`, which `journal_path_from_url` (`_storage.py:141`) reads past and discards:

```
'journal://relative/x.log' -> PermissionError: [Errno 13] Permission denied: '/x.log'
```

An operator typing `--storage-url journal://mydir/journal.log` (a natural mistake, since
`journal:///` needs three slashes) loses `mydir` with no diagnostic. Reject a non-empty
`netloc` in `journal_path_from_url`.

**Test gap.** No test in the suite passes a relative path to any of these helpers — every
one uses `tmp_path`, which is absolute. Confirmed by mutation (see *Mutation results*).

---

## B2 — does the journal backend make the known `best()` bug more reachable? **Yes, decisively.**

This is the question the gate was asked. The bug itself
(`_optuna_store.py:304-309` + `:276-279`) is pre-existing and out of scope; I am not
re-reporting it. The finding is about what C7a changes.

**It converts a self-healing failure into a permanent one, on exactly the path that
produces it.**

`build_optuna_storage` (`_storage.py:203-211`) forwards `heartbeat_interval=60` /
`grace_period=180` only on the RDB branch — correctly, since `JournalStorage` implements
neither heartbeat method. The consequence chain is what matters. Measured on optuna 4.9.0,
against a clean checkout of `933ce52e`.

**Journal backend** — one `RUNNING` trial (what a walltime-killed worker leaves behind)
plus one real `COMPLETE` trial at cost 0.42:

```
optuna 4.9.0
type <class 'optuna.storages.journal._storage.JournalStorage'>
has get_heartbeat_interval: False
has record_heartbeat: False
fail_stale_trials: OK (no raise)
states: [(0, 'RUNNING', None), (1, 'COMPLETE', 0.42)]
best(): (0, 0.0, False)
completed_count(): 2
reopened states: [(0, 'RUNNING'), (1, 'COMPLETE')]
reopened best(): (0, 0.0)
after reopen fail_stale ok; states: [(0, 'RUNNING'), (1, 'COMPLETE')]
```

**RDB backend**, same scenario, `heartbeat_interval=1` / `grace_period=1`, one
`fail_stale_trials` after the grace window:

```
rdb has record_heartbeat: True interval: 1
before: [(0, 'RUNNING'), (1, 'COMPLETE')]
after fail_stale: [(0, 'FAIL'), (1, 'COMPLETE')]
best after reclaim: (1, 0.42, False)
```

So:

- Under **RDB**, `OptunaStrategy.suggest` (`strategy/_optuna.py:297`) calls
  `_fail_stale_trials` before every ask; the zombie flips to `FAIL` within
  `grace_period`, `best()` filters it, and the run publishes the real winner. The bug
  needs a rare "no worker ever asks again" window to bite.
- Under **journal**, `_heartbeat_interval` (`strategy/_optuna.py:57-61`) returns `None`
  (no `get_heartbeat_interval`), so no heartbeat thread ever starts, and
  `optuna.storages.fail_stale_trials` **silently no-ops** — it returns cleanly and changes
  nothing, so there is not even a warning. The zombie is permanent for the life of the
  study.

And the zombie wins. `params` are stamped onto the trial only at *tell* time
(`strategy/_optuna.py:413`, `set_trial_user_attrs` inside `register_result`), so a trial
killed mid-evaluation carries no `_ATTR_PARAMS`. `_to_trial` gives it `score = 0.0`
(`_optuna_store.py:279`) and `params = {}` (`:284`), `best()` returns it, and
`finalize_distributed_study` → `_headline_winner` (`_finalize.py:153`) →
`_pipeline_for_trial(spec, headline)` → `build_pipeline(spec.pipeline, {})` publishes the
**untuned base pipeline** as `best_pipeline.json` with a perfect score.

The companion off-by-one is reachable in the same breath: `_require_terminal_study`
(`_finalize.py:150`) is handed `n_seen=len(trials)`, which counts the zombie, so the
finalize gate opens one real trial early. `completed_count()` (`:315`) counts it too
(`2`, above, for one completed trial).

At ~30 min per evaluation on a 200-trial campaign, walltime-killed workers are the normal
end state, not an edge case — every killed worker deposits exactly one zero-cost zombie.

**What this means for C7a.** The code fix belongs to the queued `best()` work, not here —
and that work appears to be landing right now: while I was running these probes, uncommitted
edits appeared in the working tree replacing the `0.0` substitution with `math.inf`, adding
a `PHENO_SCORE` sidecar, adding `terminal_trials()`, and re-basing `_require_terminal_study`
on `n_done`. All of my measurements above were re-run against a **clean checkout of
`933ce52e`** in a separate worktree, so they describe the committed state, not that
in-flight work.

What remains C7a's own responsibility is that it ships a description of this limitation
that is materially wrong:

- `_storage.py:163-168` — "the loss is stale-trial reclamation, documented in §7 L2 as
  bounded rather than fatal". Not bounded: it changes which pipeline gets published.
- `docs/source/how_to/pages/tune_distributed_hpcc.md` — "This does not stall the budget …
  but it leaves zombie rows and a slightly optimistic in-flight count. Postgres reclaims
  those." A user reads that as cosmetic.
- The commit message's "bounded — `is_exhausted` counts only COMPLETE and PRUNED, so the
  budget still drains" is accurate about the budget (`strategy/_optuna.py:453-467` does
  filter on those two states) and silent about selection, which is the consequence that
  matters.

Once the queued fix lands, the honest wording is that the journal backend leaves zombie
`RUNNING` rows that nothing reclaims, that they are excluded from selection and from the
budget, and that they inflate the raw trial count the Monitor displays. Until it lands,
the doc should say plainly that a killed worker can win the study.

---

## What is genuinely correct

**Scheme dispatch is complete.** I grepped every optuna-storage construction site under
`src/phenotypic` (`RDBStorage`, `get_storage`, `load_study`, `create_study`, `storage=`).
The only live ones are `_optuna_store.py:92` (create), `:117` (load),
`strategy/_optuna.py:256` (the no-`.study` fallback), and
`strategy/_optuna_support.py:200` (`is_legacy_study_present`, which is handed a storage
*object*, not a URL, so it inherits the dispatch). All are covered. No missed site.

**Failure modes are loud, not silent** (except the relative-path case in B1):

```
'garbage://x/y'          -> NoSuchModuleError: Can't load plugin: sqlalchemy.dialects:garbage
'journal'                -> ArgumentError: Could not parse SQLAlchemy URL from given URL string
''                       -> ArgumentError: Could not parse SQLAlchemy URL from given URL string
'journal:///'            -> IsADirectoryError: [Errno 21] Is a directory: '/'
'journalx:///a'          -> NoSuchModuleError: Can't load plugin: sqlalchemy.dialects:journalx
```

An unwritable journal path raises `PermissionError` from the `mkdir` at `_storage.py:199`.
Nothing falls back silently to RDB.

**The H1 SQLite refusal is not bypassable by spelling.** `is_sqlite_url` classifies every
variant I tried, including the ones that dodge a naive `startswith`:

```
'sqlite:///a.db' True   'SQLITE:///a.db' True   'sqlite+pysqlite:///a.db' True
' sqlite:///a.db' True  'sqlite:///a.db?x=1' True  'Sqlite:///a.db' True
'sqlite://' True        'sqlite:a.db' True
```

The guard sits at `_run.py:793`, inside `_validate_slurm_request`, which `run_tuning`
calls at `:642` — before `deliverables_dir(...).mkdir` at `:654`. Pre-artifact, as
claimed, and pinned by `test_the_sqlite_refusal_lands_before_any_artifact`.

**The fleet-sharing property holds.** The suite does not test it (see false greens), so I
ran it: 8 spawned processes, 25 `append`s each, one journal, symlink lock.

```
exitcodes: [0, 0, 0, 0, 0, 0, 0, 0]
expected 200 got 200
unique (w,i) pairs: 200
missing: [] count 0
journal size bytes: 84689
```

No loss, no duplication, no corruption. This is the property P1 exists for and it works.

**`_optuna_store.py` was not widened by its own 34-line diff.** The only behavioural change
is `storage_url.startswith("sqlite")` → `is_sqlite_url(storage_url)`, which is strictly
tighter (`sqliteX://` no longer misclassifies). The heartbeat kwargs survived the
refactor and are pinned by a *pre-existing* test,
`tests/unit/tune/test_stale_trial_reconciliation.py:218` — dropping them from the RDB
branch fails `assert captured["heartbeat_interval"] == 60`. That is the one mutation that
would have widened the bug to *all* backends, and it is genuinely caught.

**Test file health.** `tests/unit/tune/test_journal_storage_dispatch.py` — 25 passed, 0
skipped, 1.8 s, optuna 4.9.0 present. The 578 lines are not padding: the negative control
(`test_optuna_string_resolver_rejects_journal_url`) does the important work of proving the
dispatch is not a pass-through wrapper, and the discriminating assertions are the lock
*type*, the persist-and-reopen round trip, the parent-`mkdir`, the pre-artifact ordering,
and the Monitor's `note == ""`.

---

## False greens

**FG1 — "a fleet can share a journal" is proved by one sequential process.**
`test_two_worker_stores_share_one_journal_study` (`test_journal_storage_dispatch.py:296`) opens two stores in the *same*
process and appends alternately. Nothing about that requires a lock, an atomic append, or
even a filesystem — an implementation that kept an in-process registry would pass. The
one property the feature exists for is untested. It does hold (my 8-process run above),
but the suite is not what establishes that. A `multiprocessing` spawn test of ~4 workers
is cheap and would close it.

**FG2 — the relative-path hole is invisible to the whole tune suite.** Mutation: absolutize
in `journal_url_for_path` (`.absolute()`), the fix B1 asks for. See *Mutation results*.

**FG3 — non-discriminating existence assertion.** `test_journal_storage_dispatch.py:534`, in `test_slurm_default_is_the_journal_backend`,
asserts `io.tune_cache_journal_path(out).exists()` as evidence that "the pre-created study
is there so no worker races". File existence proves nothing — the `create=False` path
creates an empty `journal.log` as a side effect (below). The test is rescued by the
`optuna.load_study(...)` two lines later, which *is* discriminating; the `exists()` line
should be dropped or replaced with a trial-count assertion so it does not read as
evidence.

**FG4 — no cross-backend equivalence test.** `test_journal_study_persists_and_resumes_trials`
covers journal; nothing runs the same assertions against sqlite and compares. The
differences are real and now known (heartbeat/reclamation, retry coverage below), so a
parametrized equivalence test would be the honest place to record which behaviours are
*not* equivalent.

---

## Non-blocking

**N1 — the "read-only" Monitor open writes to the run directory.**
`gui/tune/_callbacks.py:864` skips the phantom-file guard for non-SQLite schemes
(`database = url.path if scheme == "sqlite" else ""`), so a journal URL goes straight into
`OptunaStudyStore(create=False)` → `build_optuna_storage` → `mkdir(parents=True)` + an
empty `journal.log`:

```
before: parent exists? False  file exists? False
create=False raised: KeyError 'Record does not exist.'
after : parent exists? True   file exists? True  size 0
```

The failure is caught and degraded as designed, so this is cosmetic today — but a
read-only viewer materializing `.pht-tune-cache/` in someone's output tree is a surprise,
and it is why FG3's `exists()` assertion is worthless. This is the C7b Monitor
phantom-file item; the concrete manifestation is worth recording against it.

**N2 — the transient-retry wrapper is inert on the new default path.**
`retry_on_transient_db_error` (`strategy/_optuna_support.py:313-322`) matches
`sqlalchemy.exc.OperationalError` only. A journal lock timeout or I/O error is not that
class, so it propagates on the first occurrence and kills the worker. Loud, and the other
workers continue, so it is not a correctness hazard — but the resilience a fleet used to
have around `ask`/`tell` is now switched off by default. This is exactly C7b's B1 (the
backend-aware retry predicate), correctly scoped out; noting it so the sequencing is
deliberate rather than accidental.

**N3 — `journal.log` growth is unbounded and unmonitored.** Documented in the commit and
the how-to. My 200-trial concurrency run produced 84,689 bytes (~423 B/trial by
`append`; ask/tell trials carry more). A 200-trial campaign is trivially small; the
concern is only a long-lived reused output directory. C7b sizing work.

**N4 — H2 returns whenever `$PHENOTYPIC_TUNE_STORAGE_URL` is set.** The commit's H2 claim
("each run gets its own study file, so two concurrent studies cannot pool trials under the
shared `_STUDY_NAME`") holds for the *default* only. An operator who exports a journal URL
into the environment — the pattern the how-to teaches for Postgres — reattaches every
subsequent run to one study file under one hardcoded study name, exactly as before. The
how-to warns about this in its Postgres paragraph; the warning is scheme-independent and
would read better in the storage table.

---

## Mutation results

Applied to `src/phenotypic/tune/_study/_storage.py:109` — the B1 fix, as a mutation:

```python
raw = Path(journal_path).absolute().as_posix()
```

Then `uv run pytest tests/unit/tune/ -q -rs -p no:randomly`.

Both runs used the project venv with `PYTHONPATH` pointed at the worktree, so neither saw
the live tree's in-flight edits.

```
BASELINE (worktree at 933ce52e, unmutated)
=========== 896 passed, 2 skipped, 186 warnings in 353.80s (0:05:53) ===========

MUTATED  (same worktree, journal_url_for_path absolutized)
=========== 896 passed, 2 skipped, 186 warnings in 334.38s (0:05:34) ===========
```

**Identical.** The tune suite cannot distinguish "relative paths are silently rooted at
`/`" from "relative paths are resolved against the cwd" — it never exercises either. Both
skips are the two Postgres integration tests
(`test_optuna_pg_integration.py`, `test_pg_marker_skips.py:15`); optuna itself is present,
so nothing else skipped silently.

The gap is structural, not accidental: every journal helper call in `tests/` derives its
path from `tmp_path` or a `tmp_path`-rooted `out`, and `pytest`'s `tmp_path` is always
absolute —

```
tests/unit/tune/test_journal_storage_dispatch.py:149  journal_url_for_path(path)          # tmp_path/...
tests/unit/tune/test_journal_storage_dispatch.py:165  journal_url_for_path(target)        # tmp_path/...
tests/unit/tune/test_run_marker.py:161                journal_url_for_path(out/...)       # tmp_path/out
tests/unit/tune/test_run_tuning_slurm.py:115,184,270  journal_url_for_path(tmp_path/...)
tests/unit/tune/test_tune_cli.py:448                  journal_url_for_path(out/...)
```

One test — `_resolve_storage_url(None, Path("out"), slurm=True)` asserted to produce a URL
whose path is `Path("out/.pht-tune-cache/journal.log").absolute()` — closes it.

The main working tree was never left mutated: my one-line change to `_storage.py` there was
reverted before these runs (`git checkout --`, `git diff --stat` clean for that file).

---

## Environment

- `optuna 4.9.0` present; `tests/unit/tune/test_journal_storage_dispatch.py` reports
  **25 passed, 0 skipped** — the `pytestmark` skipif did not fire, so the tests that
  matter really ran.
- **Working-tree caveat.** Partway through this gate, uncommitted edits from another agent
  appeared in `/bigdata/exfab/anguy344/PhenoTypic` (`_optuna_store.py`, `_finalize.py`,
  `_run.py`, `_io_constants.py`, `_study_store.py`, `_protocol.py`, `_screening.py`,
  `_optuna_support.py`) — the queued `best()` fix. Every measurement and every line
  number in this report was therefore taken from, or re-verified against, a detached
  worktree checked out at `933ce52e`
  (`<scratchpad>/wt-clean`), not the live tree. I made no edit to the repository outside
  this report; the one mutation I applied to `_storage.py` in the main tree was reverted
  (`git checkout --`, verified clean) and re-applied inside a second isolated worktree
  (`<scratchpad>/wt-c7a`).
- All probes run via the project venv on `gpu12`.
- Probe scripts: `<scratchpad>/probe1.py` (journal zombie/heartbeat), `probe2.py` (RDB
  reclamation contrast), `probe3.py` (8-process concurrency), `probe4.py` (relative paths
  + bad schemes), `probe5.py` (documented `-o ./out --slurm` reproduction), `probe6.py`
  (read-only side effects), `probe7.py` (H1 guard bypass attempts).
