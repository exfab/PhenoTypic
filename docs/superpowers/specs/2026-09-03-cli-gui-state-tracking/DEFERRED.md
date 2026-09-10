# Deferred findings

Audit findings from [`audit.md`](audit.md) that the
[design](design.md) does **not** address, with why, and what each would take.

Kept as a document rather than dropped so a future reader can tell the
difference between "considered and deferred" and "not noticed".

## Addressed by the design (not deferred)

For reference, so this list is unambiguous: **S1** (memoized completion) →
design §9; **S2** (two overlapping fingerprints) → §9/§11; **S3** (`ctime_ns`)
→ §9.1; **S4** (nine sources) → §4; **S5** (duplicate event-log replay) → §11;
**S6** (per-image marker collapse) → §6.1; **S8**, **S9**, **S11** (partial),
**S13**, **S15**, **S18**, **S19**, **S21** → §11.1–11.2.

---

## Deferred by explicit scope decision

### D-1. SLURM observer decision tree

**Audit:** the ~15-branch tree in `_observe_record`, its 30-second
reconciliation grace window, and `squeue`/`sacct` state ranking.

**Why deferred:** only ~185 of its lines are filesystem-derived; the rest is
scheduler domain and cannot be consolidated. It is the least testable code in
the GUI (needs a real or fake scheduler), holds the trickiest concurrency, and
its failure mode — "run stuck in `reconciling`" — is directly user-visible.

**Already taken:** the design moves its two `_cli_completion` call sites and its
Stage-3 probe onto the consolidated contract, so the 2-second `O(N)` SHA-256
walk is fixed without touching the tree.

**What it would take:** fake-scheduler test coverage first, then folding
`_staged_terminal_observation` and `_run_marker_observation` behind
`resolve_run_state`.

### D-2. Stale GUI owner record has no repair path — **recommended for P6**

**Audit S7.** Nothing in the codebase deletes or repairs
`gui_launch_owner.json`. A SIGKILLed GUI leaves `status: "running"`;
`rehydrate_from_sandbox` downgrades it in memory only (`persist=False`), and
`_assert_output_claimable_locked` then refuses the output permanently, with no
UI affordance to clear it.

**Why here and not in the design:** it is a behaviour addition (liveness check
or an explicit "release this output" action), not a consolidation.

**Note:** the design rewrites `_assert_output_claimable_locked` — the exact
predicate that causes the dead-end. Folding this into **P6** would be cheap and
is recommended. Left out only because it was not in the agreed scope.

---

## Deferred: unrelated churn

These are real findings inside files the design does not otherwise touch.
Fixing them here would bury the change in unrelated diff.

| # | Finding | Sites |
|---|---|---|
| S10 | GUI hand-joins `.phenotypic/logs/{gui,slurm}`; never imports `logs_dir()` (8 CLI callers, 0 GUI). The `gui`/`slurm` role names are unconstantized on both sides | `run_console/_slurm.py:438`, `_slurm_observer.py:909-910`, + 10 CLI sites |
| S12 | Three GUI re-implementations of atomic write that skip the `fsync` the SDK helper performs | `_curation_labels.py:453,953,961`; `_filtered_state.py:578` |
| S14 | `VERIFIED_PARQUET` re-spelled; `ERROR_ANALYSIS_{PARQUET,CSV,HTML}` re-derived from a format string | `_error_tab/_publication.py:596,849-851` |
| S16 | 13 `output_dir / "results"` joins bypassing `results_dir()`; migrate/recompile are the holdouts | `_cli_migrate.py` ×7, `_cli_recompile_tables.py` ×2, others |
| S17 | Sidebar classifier LRU keys on run-root `mtime_ns`, but all four markers it reads sit at depth ≥ 2 where a POSIX root-dir mtime never moves — so a run finishing under a live GUI leaves stale badges. Contradicts the cache's own docstring | `shell/_classifier.py:201-207` |
| S20 | Vestigial viewer DZI cache-dir helper — path computed, documented as never created, dead since the Viv rebuild | `_output_root.py:751` |
| S22 | `_LAST_DUMPED`: unbounded module dict keyed on `id(image)`, a CPython address reusable after GC | `builder/_point_picker.py:549` |
| S23 | Unbounded caches: `_terminal_job_cache` (sacct states, never evicted), `LocalRunner._instances` (self-documented `TODO(perf)`), `builder_tiles/<session>/` never garbage-collected — its sibling `_preview_cache` wipes at launch *and* `atexit` | `_manifest_builder.py:73`, `_runner.py:129`, `_point_picker.py:106` |
| S24 | Three independent implementations of "authorize a Zarr member for serving". They agree today; nothing enforces that they keep agreeing | `_zarr_routes.py:166`, `browse/_tile_routes.py:258`, `_shared/tiles.py:970` |
| S25 | `RC_INTERVAL_LOG` at 1000 ms drives six server-side callbacks; the cheap ones could move to a slower interval | `run_console/_callbacks.py` |
| S26 | `.gui_log/stdout.log` is append-only with no rotation and **the GUI never reads it back** — the UI reads only the in-memory `deque(maxlen=5000)`. Either read it (which would fix S27) or stop writing it | `_runner.py:430-432` |
| S27 | Page reload loses `RC_STORE_ACTIVE_RUN_RECEIPT` (memory storage); no path re-attaches Cancel or the live log tail to a running job. `session` storage would fix most of it | `run_console/_layout.py:235` |
| S28 | `browse/_source_lister.list_datasets` is an uncached unbounded-depth `os.walk` **inside a Dash callback**, with no progress reporting and no cancellation — unlike `OutputRoot.discover`, which has both | `browse/_source_lister.py:38` |

---

## Deferred: performance, not correctness

### D-3. Incremental aggregation

Under a rolling input dataset, adding 10 images to 6,000 rebuilds
`master_measurements.parquet` from all 6,010 embedded tables. That is inherent
to design §7.5 — master is a pure function of the authorized embedded tables,
which is the property that makes it derivable rather than tracked.

Incremental aggregation is possible (append-only shards keyed by `work_id`,
merged lazily) but it reintroduces exactly the stale-cache hazard §7.5 exists to
forbid. It should not be attempted until the fan-out (§8) is measured under
S-2/S-3 and shown to be the bottleneck.

### D-4. Third-party Zarr probing walks the chunk tree twice

`_store_revision_snapshot` fully descends a Zarr chunk tree, twice, for any
store lacking a `publication_protocol` declaration — i.e. every third-party
store Browse probes. PhenoTypic-written stores short-circuit via
`store_publication_token`. Browse probes the selection plus all neighbours and
the filmstrip.

Deferred because it is entirely inside the Browse pixel path, which this design
does not touch.

### D-5. `BrowseCache.usage()` rglobs every cache entry

Called on every `/api/browse/dataset/status` poll and after every `prune()`. At
the 10 GiB high-water mark that is a full-tree stat sweep per poll tick. Same
rationale as D-4.
