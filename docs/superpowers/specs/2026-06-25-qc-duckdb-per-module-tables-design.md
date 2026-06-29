# QC DuckDB Per-Module Tables — Design

- **Date:** 2026-06-25
- **Status:** Draft (revised after independent review)
- **Branch:** `feature-gui/deliverable-standalone`
- **Related:** `2026-06-24-deliverables-standalone-gui-design.md`,
  `2026-06-23-analysis-subpackage-reorg-design.md`

## 1. Summary

Replace the flat, two-file QC artifact
(`deliverables/qc/qc_summary.parquet` + `qc_members.parquet`, plus the
`qc_config.json` provenance snapshot) with a single embedded **DuckDB**
database, `deliverables/qc/qc.duckdb`, in which **each configured QC
module gets its own self-describing table** plus a **catalog** table that
records every module's column roles. The compute seam
(`phenotypic.sdk_._qc_recipe._runner.run_qc`) becomes the DuckDB writer;
the GUI results-viewer QC **Review** tab and the **Error** tab's QC-reading
helper are redesigned around a new catalog-driven data API; and the
in-session live recompute is extended so editing a QC module's settings
durably recomputes (closing today's gap where only curation recompute is
durable).

This is a **full cutover**: the parquet QC artifact and the GUI "Export QC
report" files are removed; nothing reads them afterward.

The **static CLI QC dashboard** (`qc.html`) identified during scoping is
**out of scope** for this change (see §12). Curation labels and review
state remain **separate files** (see §9).

## 2. Motivation

Two problems with today's flat artifact:

1. **The uniform schema is lossy.** `run_qc` forces every check into a
   fixed summary schema (`instance_id, class, <groupby…>, metric, status,
   flag, n_members, n_flagged, rank`) and a fixed members schema
   (`instance_id, <groupby…>, Metadata_ImageFile, Object_Label,
   member_value`). But the seven `QualityCheck` subclasses diverge sharply:
   - `GridOccupancy` emits `Filled/Expected/Vacant` and has rows for
     *vacant expected positions* (no detected object at all);
   - `TukeyOutlierFraction` emits `LowerFence/UpperFence/NumOutliers`;
   - `ICC` emits `NumSubjects/NumRaters`;
   - `ReplicateAgreement` (`SE`) emits `Value/Mean/CV/NumReplicates`;
   - `SE/ZMax/MAD/Tukey` additionally carry a `time_label` dimension;
   - `groupby` is configured per-instance (`["Plate"]`,
     `["Metadata_ImageFile"]`, …).

   The members parquet discards every check-specific column (keeps only
   `member_value`), and the union summary forces NaN-padded foreign-module
   columns into every row.

2. **The GUI data layer pays for the flat schema.** Because structure is
   erased on write, the Review tab's data layer reverse-engineers it on
   read — most notably `groupby_cols_for`, which guesses a module's group
   keys via "a column that is neither a fixed lead/tail name nor all-null
   for this instance." This heuristic, the `_SUMMARY_LEAD/_TAIL` slicing
   constants, and the `dataset_by_image_map` / `time_by_key_map`
   master-reconstruction helpers all exist *only* to compensate for the
   flat schema.

A per-module, self-describing table + catalog removes the lossiness and
deletes the reverse-engineering.

## 3. Background — current state

### 3.1 The compute seam

`phenotypic.sdk_._qc_recipe._runner.run_qc(measurements_df, pipeline,
output_dir, qc_output_dir=None)` is the single GUI-free write seam. It is
called from exactly two places with identical semantics:

- CLI `finalize_post_master_outputs` (`_cli/_cli_output_manager.py:662`),
  once per finalize — reached by both the forward `aggregate_measurements`
  path and the `--recompile` worker (`_run_post_master_steps`);
- GUI in-session recompute `_recompute_after_curation`
  (`_qc_tab/review/_callbacks.py:712`), synchronous, same-session.

It is pure with respect to review progress (never writes
`review_state.json`) and never writes `measurements.parquet`.

The QC config itself lives in `deliverables/pipeline.json` under the `qc`
key, as a list of `QcRecipeEntry` (`{instance_id, class, enabled,
params}`) reachable via `ImagePipeline.get_qc()`/`set_qc()`. `QcRecipe`
(`sdk_/_qc_recipe/_recipe.py`) performs scoped atomic read-modify-write of
just that array.

### 3.2 The `QualityCheck` contract

`QualityCheck.analyze(frame)` returns the **augmented frame** = input +
`QC_<name>_Metric/Flag/Status` + any check-specific columns from
`_compute`. `summary()` and `group_members()` are projections of it.
`metric_col()/flag_col()/status_col()` derive column names from `name`.
The today's runner maps these into the flat parquet schema via
`_build_summary_rows` (with `_rank_worst_first`) and `_build_member_rows`.

### 3.3 Readers (the cutover surface)

All reads funnel through two loaders in
`_qc_tab/review/_data.py`:

- `load_qc_summary(output_root)` → `pl.read_parquet(layout.qc_summary_parquet)`
- `load_qc_members(output_root)` → `pl.read_parquet(layout.qc_members_parquet)`

Consumers:

- **Review tab** (`_qc_tab/review/_callbacks.py`): module picker, worklist,
  group detail/gallery, post-recompute reload (8 call sites).
- **Error tab** (`_error_tab/_data.py:126-127`): `verified_good_keys`
  reads both to derive the verified-good baseline (unlabeled objects in
  ≥1 reviewed QC group).

No reader exists for `qc_config.json` (write-only provenance). No in-app
reader exists for the GUI Export `qc.parquet`/`qc_summary.json`
(user-facing download only; referenced by tests).

### 3.4 The two side stores

- **`curation_labels.parquet`** (`_curation_labels.py`): object-keyed
  `(Metadata_ImageFile, Object_Label, Curation_Category, Bbox_CenterRR,
  Bbox_CenterCC)`, written live per mark/unmark under an `RLock`, re-keyed
  by centroid fingerprint on load, **never wiped by the CLI**. It is an
  *input* to QC (removed keys are anti-joined before `run_qc` via
  `build_recompute_frame`) and the source for `errors/<category>.parquet`.
- **`review_state.json`** (`_review_state.py`): group-keyed
  `{instance_id: {reviewed: [encoded_group_key…], last}}`, GUI-written per
  click, **reset by the CLI on every finalize/recompile**
  (`_reset_qc_review_state`). It is an *annotation* on QC outputs.

## 4. Goals / non-goals

### Goals

- One DuckDB file, `deliverables/qc/qc.duckdb`, with one self-describing
  table per QC module + a catalog table.
- A self-describing `QualityCheck` contract so heterogeneous checks each
  define their own table without a fixed schema.
- `run_qc` writes DuckDB; every recompute is an atomic full rebuild (CLI
  finalize/recompile and GUI live recompute alike).
- GUI Review tab + Error-tab QC helper redesigned onto a catalog-driven
  data API; delete the flat-schema reverse-engineering.
- Live durable recompute on **both** triggers: edit a module's settings,
  and mark a bad segmentation.
- Full cutover: remove the parquet QC artifact and the Export files.

### Non-goals

- The static CLI `qc.html` dashboard (deferred — §12).
- Moving `curation_labels.parquet` or `review_state.json` into the DB.
- Changing the `qc` recipe storage in `pipeline.json` (unchanged).
- Changing the curation/error-triage model beyond rewiring
  `verified_good_keys`.
- GPU/staged-engine changes (QC runs only at finalize, unchanged).

## 5. Decisions (resolved during brainstorming)

| # | Decision |
|---|----------|
| D1 | **Engine:** DuckDB (new dependency, `uv add duckdb`). One file `deliverables/qc/qc.duckdb`. |
| D2 | **Per-module self-describing tables + a `qc_modules` catalog.** The catalog absorbs `qc_config.json`'s provenance role. |
| D3 | **Self-describing `QualityCheck` contract:** `to_table()` + `table_spec()`. |
| D4 | **`run_qc` is the DuckDB writer.** Every recompute is an atomic full rebuild — CLI finalize/recompile and GUI live recompute alike (both settings-edit and curation trigger a full rebuild; no single-module patch path). |
| D5 | **Full parquet cutover.** Delete `qc_summary.parquet`, `qc_members.parquet`, `qc_config.json`, and the Export `qc.parquet`/`qc_summary.json`. |
| D6 | **GUI data API redesigned now** (catalog-driven). Delete `groupby_cols_for`, lead/tail slicing constants, master-reconstruction maps. Error tab keeps its design; only `verified_good_keys` rewires. |
| D7 | **`qc.html` static dashboard: out of scope** (deferred follow-up). |
| D8 | **Curation labels + review state stay separate files.** `qc.duckdb` holds only computed QC analysis + catalog. |

## 6. Architecture

### 6.1 Storage — `qc.duckdb`

Location: `deliverables/qc/qc.duckdb`, resolved via `BundleLayout`
(new accessor `layout.qc_duckdb`), so a standalone deliverables bundle
reads/writes inside the bundle. The legacy `resolve_qc_dir` read-fallback
to `<output>/qc/` is retained for path resolution.

The database contains:

- **`qc_modules`** — the catalog, one row per configured (enabled) module.
- **`qc_<instance_id>`** — one **data table per module**, holding that
  module's own augmented/projected frame (self-describing columns).
- **`qc_<instance_id>__summary`** — the per-module worklist summary
  (group-level, worst-first `rank`), written by `run_qc` reusing the
  existing `_rank_worst_first` ordering helper so the worst-first logic is
  not reimplemented in SQL.

Table names are derived from `instance_id` via a deterministic, SQL-safe
sanitizer (instance ids look like `qc-SE-1a2b3c4d`; sanitize to a valid
identifier, e.g. `qc_se_1a2b3c4d`). The catalog stores both the raw
`instance_id` and the resolved table name so the mapping is explicit and
never re-derived by consumers.

> **Sub-decision (deferred to implementation):** `qc_<id>__summary` MAY be
> a DuckDB `VIEW` over `qc_<id>` (rank via window function) instead of a
> materialized table, IF the rank/status window-function port proves clean.
> The spec mandates the *interface* (a per-module summary readable by the
> data API), not the physical form. Default to a materialized companion
> table to reuse `_rank_worst_first`.

### 6.2 Catalog schema — `qc_modules`

One row per enabled module:

| Column | Type | Meaning |
|--------|------|---------|
| `instance_id` | TEXT | Recipe entry id (PK). |
| `class` | TEXT | `QualityCheck` subclass name (e.g. `ICC`). |
| `name` | TEXT | The check's short `name` (e.g. `ICC`). |
| `table_name` | TEXT | Sanitized data-table name. |
| `summary_table` | TEXT | Sanitized summary table/view name. |
| `ordinal` | INTEGER | Recipe order (drives module-picker order). |
| `groupby_cols` | TEXT (JSON) | Ordered group-key column names. |
| `metric_col` | TEXT | `QC_<name>_Metric`. |
| `status_col` | TEXT | `QC_<name>_Status`. |
| `flag_col` | TEXT | `QC_<name>_Flag`. |
| `on_col` | TEXT | The check's `on` measurement column. |
| `member_key_cols` | TEXT (JSON) | Per-object curation-key columns, or `[]`. |
| `supports_object_curation` | BOOLEAN | Whether rows map to curatable detected objects (False for diagnostic-only modules like `GridOccupancy`). |
| `time_col` | TEXT NULL | Time-course facet column (`time_label`) or NULL. |
| `higher_is_bad` | BOOLEAN | The check's `_HIGHER_IS_BAD`. |
| `extra_cols` | TEXT (JSON) | Check-specific columns (for richer detail rendering). |
| `params` | TEXT (JSON) | The enabled entry's params snapshot (provenance, replaces `qc_config.json`). |
| `warn_threshold` | DOUBLE | For status legends/labels. |
| `fail_threshold` | DOUBLE | For status legends/labels. |

Rationale for the three "new" columns the flat schema faked:

- **`member_key_cols`** — member identity is not universally
  `(Metadata_ImageFile, Object_Label)`; name it per module.
- **`supports_object_curation`** — some checks are **diagnostic-only**.
  `GridOccupancy` reports per-*plate* occupancy (filled vs expected counts,
  broadcast as scalars onto the group's rows — there are **no** per-vacant-
  position rows), so per-object curation and the tile gallery are not
  meaningful for it. A diagnostic-only module's table is **group-level**
  (one row per group); the Review tab hides "mark flagged for removal" +
  the gallery and `verified_good_keys` skips it. Object-curation modules
  (`supports_object_curation=True`) store **member-level** rows. The flag
  makes the granularity explicit instead of inferring it.
- **`time_col`** — lets the per-module table carry its own `time_label`
  and dataset context so the GUI stops reconstructing `Metadata_Time`/
  `Metadata_Dataset` from the master frame.

### 6.3 Self-describing `QualityCheck` contract

Add to `analysis/abc_/_quality_check.py`:

- **`to_table(self) -> pd.DataFrame`** — return the module's natural,
  self-describing frame to persist. **Precondition:** `analyze()` has run
  (it reads `self.results()` / `_latest_measurements`); `run_qc` always
  calls `analyze()` first — document this so out-of-`run_qc` callers don't
  hit an `AttributeError`. The default (member-level) implementation
  projects the augmented frame to: `groupby` cols + member-key cols (when
  present) + `on` + all `QC_<name>_*` columns (metric/flag/status +
  check-specific extras) + context columns — `Metadata_Dataset` and the
  column **named by `self.time_label`** (e.g. `"Metadata_Time"`, not a
  literal `"time_label"` column) — when those columns are present in the
  frame. Diagnostic-only checks override to return a **group-level** frame:
  `GridOccupancy.to_table()` returns one row per group
  (`groupby + Filled + Expected + Vacant + Metric + Status + Flag`) rather
  than per-colony rows.
- **`table_spec(self) -> QcTableSpec`** — return the catalog descriptor
  (a small frozen dataclass / pydantic model) populated from class
  attributes (`name`, `metric_col()`, `_HIGHER_IS_BAD`, …) and instance
  config (`groupby`, `on`, thresholds, `time_label` when defined). Carries
  `member_key_cols`, `supports_object_curation`, `extra_cols`, `time_col`.

`supports_object_curation` defaults to `True` and is overridden to
`False` on diagnostic-only checks (`GridOccupancy`). `member_key_cols`
defaults to `("Metadata_ImageFile", "Object_Label")` and is `()` when the
check has no per-object key.

The per-module **summary** continues to come from the existing
`check.summary()` + `_rank_worst_first` so the ordering semantics
(`fail > warn > pass`, then bad-direction metric extremity, NaN last) are
preserved exactly.

### 6.4 `run_qc` rewrite — the DuckDB writer

New signature (backward-compatible call shape; no new positional args):

```
run_qc(measurements_df, pipeline, output_dir, *, qc_output_dir=None)
```

`run_qc` always performs an **atomic full rebuild** — there is no
single-module patch path (per D4 / the settings-edit decision):

1. Build the whole DB in a temp file (`qc.duckdb.tmp`): create
   `qc_modules` + each enabled module's data and summary tables, for every
   entry that analyzes successfully. Tolerant: a check that fails to build
   or analyze is skipped with a WARNING, exactly as today.
2. **Atomically replace** `qc.duckdb` via `os.replace` (POSIX-atomic; see
   §11 for the Windows open-handle caveat + retry). Readers never observe
   a half-written DB.

An empty / 0-enabled-check pipeline writes nothing, matching today's no-op
(see §11). The tolerant per-check helper (today `_run_one_check`) is
retained; `_build_summary_rows`/`_build_member_rows` are replaced by
`check.to_table()` (group-level for diagnostic-only checks, member-level
otherwise — §6.2/§6.3) + the per-module summary (always group-level
worst-first, via `check.summary()` + `_rank_worst_first`). The
`_atomic_write` parquet writers are removed.

### 6.5 CLI wiring

- `finalize_post_master_outputs` continues to call `run_qc(post_df,
  pipeline, output_dir)` (full rebuild) — unchanged call site, new writer.
  Because both `aggregate_measurements` and the `--recompile` worker reach
  finalize, **recompile rebuilds the DB automatically** (explicit
  requirement).
- `_reset_qc_review_state` is unchanged (still resets `review_state.json`
  on finalize).
- `migrate_legacy_qc` still relocates a legacy root `<output>/qc/` into
  `deliverables/qc/`; the legacy *parquet* artifact inside it is simply
  not read (the DB is rebuilt from the pipeline + frame).

### 6.6 GUI data API redesign

Replace the two loaders + the reverse-engineering helpers in
`_qc_tab/review/_data.py` with a catalog-driven API backed by a small
DuckDB read helper (`_qc_tab/review/_db.py` or similar):

- `open_qc_db(output_root) -> connection | None` — open a **short-lived,
  `read_only=True`** connection; `None` when `qc.duckdb` is absent (no QC
  configured / never finalized). Connections are **never held across Dash
  callbacks** — each data-API function opens, queries, closes, and returns
  a polars frame / plain list. This keeps the single-writer invariant
  trivial and avoids holding an OS file lock that would block the CLI's
  `os.replace` on Windows (§11).
- `list_modules(output_root) -> list[QcModule]` — read `qc_modules`
  ordered by `ordinal`. Replaces `module_options` + `groupby_cols_for`.
  `QcModule` carries every catalog field, so consumers never re-derive
  group keys, member keys, or capability.
- `module_summary(output_root, instance_id) -> pl.DataFrame` — the
  worklist, worst-first. Replaces `module_worklist`.
- `module_members(output_root, instance_id, group_key) -> pl.DataFrame` —
  the group's member rows *with* check-specific columns. Replaces
  `group_member_keys`/`group_record` and removes the
  `dataset_by_image_map`/`time_by_key_map` master reconstruction (the
  table carries dataset/time context directly).
- `summary_stats(...)` — retained (counts fail/warn/pass/insufficient +
  robust median), computed from `module_summary` (per-module frame is
  small).
- `build_recompute_frame(...)` — **unchanged**; still reads the
  post-applied mirror minus removals and feeds `run_qc`.

Deleted: `load_qc_summary`, `load_qc_members`, `module_options`,
`groupby_cols_for`, `module_worklist`, `group_member_keys`,
`group_record`, `dataset_by_image_map`, `time_by_key_map`,
`facet_keys_by_timepoint` (time facets now driven by the table's
`time_col`), and `_SUMMARY_LEAD/_TAIL`, `_MEMBERS_LEAD/_TAIL`.

Review-tab callbacks update to the new API. Diagnostic-only modules
(`supports_object_curation == False`) render their figure + summary but
hide the curation radial and tile gallery.

### 6.7 Error tab

The Error tab keeps its design and figures. Only `verified_good_keys` /
`_module_reviewed_member_keys` (`_error_tab/_data.py:104,146`) rewire:
for each reviewed `(instance_id, group key)`, resolve members via
`module_members(...)` filtered to that module's `member_key_cols`,
skipping modules with `supports_object_curation == False`. This also
**decouples the Error tab from Review-tab internals** (it stops importing
`groupby_cols_for`/`_eq_or_null`/`load_qc_*`).

### 6.8 Live recompute model

Both triggers run the **same** same-session, synchronous, **full-rebuild**
recompute (extending the existing `_recompute_after_curation` pattern;
recompute is far cheaper than image processing, as the user noted): read
the curated post-applied frame (`build_recompute_frame`, removals
anti-joined) → `run_qc` full rebuild → reload.

- **Mark a bad segmentation** (Review subview, curation): a removed object
  can shift any module's groups → full rebuild. Unchanged from today
  except the writer target.
- **Edit a module's settings** (Configure subview): **new durable
  behavior** — today `_refresh_qc_card_bodies` only re-analyzes in memory
  for the card preview and does not rewrite the artifact, so the
  worklist/Review tab go stale. The new recompute fires on the
  `STORE_QC_RECIPE_REVISION` bump emitted by `_on_modal_submit` and runs
  `run_qc` against a pipeline whose `get_qc()` reflects the just-saved
  recipe (the edit is persisted via `QcRecipe` before the recompute).
  Because it is a full rebuild it **auto-handles add / disable / delete**
  (only enabled entries are written) — no per-module DROP/patch
  bookkeeping and no new "which module changed" Dash store is required.

Subprocess was considered and rejected: the recompute is in-process today,
cheap, and a subprocess would add IPC + a second DuckDB writer with no
benefit. DuckDB single-writer is satisfied trivially by the single GUI
process under the existing recompute lock.

### 6.9 `review_state` reconciliation (robustness gain)

Today's stale-review guard ("a reviewed `instance_id` absent from the
summary → drop its keys") is a brittle proxy. With the catalog, after each
recompute the GUI can ask precisely whether a reviewed encoded group key
still exists in `qc_<id>__summary` and prune only genuinely-vanished
groups. Settings-edits change `groupby`/thresholds and thus group keys, so
this reconciliation replaces a heuristic with an exact existence check.
`review_state.json`'s on-disk format is unchanged.

## 7. Dependency

Add `duckdb` to `pyproject.toml` core dependencies (`uv add duckdb`).
DuckDB ships cross-platform CPython wheels (macOS/Windows/Linux), so it
does not need the Windows try/except treatment that `rawpy`/`pympler`
require. Polars ↔ DuckDB interop is via Arrow (`duckdb` can return a
polars frame / Arrow table directly). Add an import-smoke assertion to the
QC test module.

## 8. Cutover plan

### Deleted (no readers remain)

- `qc_summary.parquet`, `qc_members.parquet`, `qc_config.json` writes in
  `run_qc`.
- `QC_SUMMARY_PARQUET`, `QC_MEMBERS_PARQUET`, `QC_CONFIG_JSON` constants
  and `qc_summary_parquet_path`/`qc_members_parquet_path`/
  `qc_config_json_path` helpers + `BundleLayout.qc_summary_parquet`/
  `qc_members_parquet`/`qc_config_json` accessors (replaced by a new
  `QC_DUCKDB` constant / `qc_duckdb_path` helper + `BundleLayout.qc_duckdb`).
  **These nine symbols are public `phenotypic.sdk_` exports** (present in
  `__all__`); per the cutover decision they are **removed outright** — an
  intentional breaking change with no known external consumers on this
  branch. Imports fail loudly with `ImportError`; **no deprecation shim**.
  The break is called out in the commit message and the root `CLAUDE.md`
  update.
- The GUI "Export QC report" button (`_on_export_click`,
  `_export_qc_report`) and its `qc.parquet`/`qc_summary.json` outputs —
  superseded by the durable DB (re-add an export-the-DB button only if
  desired later; out of scope here). Its `FEATURES.md` row is removed/updated.

### Migrated

- `_qc_tab/review/_data.py` → new catalog-driven API (§6.6).
- `_qc_tab/review/_callbacks.py` → new API call sites.
- `_error_tab/_data.py` → `verified_good_keys` onto the new API.
- `analysis/abc_/_quality_check.py` → `to_table()` + `table_spec()`;
  `GridOccupancy` (+ any other diagnostic-only check) overrides.
- `sdk_/_qc_recipe/_runner.py` → DuckDB writer.
- `_io_constants.py` / `BundleLayout` → `QC_DUCKDB` constant + `qc_duckdb`
  path helper/accessor.

### Docs / ledgers

- `CLAUDE.md` (root): update the `deliverables/qc/` description and the
  `run_qc`/finalize gotchas to describe `qc.duckdb`.
- `gui/CLAUDE.md`: update the Error-analysis tab section (verified-good now
  reads `qc.duckdb`), and the QC artifact references.
- `gui/FEATURES.md`: update QC tab rows (Export removed; Review reads DB);
  `gui-checks` `features-md-gate` requires this since `gui/` changes.

### Tests (9 files touched)

- `tests/unit/qc/test_run_qc.py` — rewrite for the DuckDB writer + catalog.
- `tests/unit/gui/results_viewer/test_qc_review_data.py`,
  `test_qc_review_layout.py` — new data API.
- `tests/integration/gui/test_qc_review_recompute.py` — full-rebuild +
  single-module-patch recompute.
- `tests/integration/cli/test_finalize_qc.py` — DB written at finalize;
  recompile rebuilds.
- `tests/e2e/gui/test_qc_tab.py` — delete **both** export tests
  (`test_export_emits_qc_parquet_and_summary` and
  `test_export_button_disabled_when_no_checks`, since the Export button is
  removed); assert the DB-backed worklist instead.
- `tests/gui/results_viewer/error_tab/test_error_data.py`,
  `test_error_tab_integration.py` — verified-good from DB.
- `tests/unit/sdk_/test_bundle_layout.py` — `qc_duckdb` accessor; remove
  the deleted parquet accessors.

New tests:

- Catalog round-trip (each of the 7 checks → its own table + correct
  catalog row, incl. `supports_object_curation` False for `GridOccupancy`).
- Single-module patch leaves other modules' tables byte-identical.
- Atomic rebuild: a reader during a rebuild never sees a partial DB.
- `to_table()` carries check-specific extras (e.g. Tukey fences).

## 9. Curation labels + review state — stay separate (rationale)

`qc.duckdb` holds **only** computed QC analysis tables + catalog.
`curation_labels.parquet` and `review_state.json` remain separate files
because:

1. **Atomic full-rebuild stays a clean swap.** `curation_labels` must
   survive every rebuild (CLI never wipes it) and is mutated live between
   rebuilds. Co-locating it would turn the rebuild into a
   read-preserve-merge-and-re-key operation.
2. **Write-frequency / single-writer mismatch.** `curation_labels` writes
   on every mark/unmark under a lock with an mtime staleness guard; a
   click-frequency mutable table inside a file the CLI atomically replaces
   invites contention.
3. **Neither is a QC analysis table.** One is object-keyed with
   fingerprint re-keying; the other is group-keyed review progress.
4. **A tested invariant stays intact:** "`run_qc` never touches
   `review_state`; the CLI never wipes `curation_labels`."

Only their **QC-join code paths** migrate to the catalog-driven API (§6.7,
§6.9); their formats and ownership are unchanged.

## 10. Data flow (after)

```
curation_labels.parquet ──anti-join──▶ post-applied frame ──run_qc──▶ qc.duckdb
   (object-keyed, never wiped,                                   ├─ qc_modules (catalog)
    re-keyed on load, hot writes)                                ├─ qc_<id>          (data, self-describing)
                                                                 └─ qc_<id>__summary (worklist, ranked)
                                                                          │
                                       review_state.json ◀──annotates──── (group-keyed, reset each finalize)
                                                                          │
                              GUI Review tab / Error tab ◀── catalog-driven data API
```

## 11. Error handling & edge cases

- **No QC configured / no enabled entries:** match today's no-op — `run_qc`
  writes nothing; the data API's `open_qc_db` returns `None`; the Review
  tab shows its empty state. (Alternative: always write a schema-only
  catalog. Default to no-op to preserve current "absent artifact" UX.)
- **A check fails to build/analyze:** skipped with a WARNING (tolerant),
  exactly as today; it simply gets no table/catalog row.
- **Diagnostic-only modules (`GridOccupancy`):** render figure + summary;
  curation radial + tile gallery hidden; excluded from `verified_good_keys`.
- **Corrupt/locked DB:** `open_qc_db` returns `None` on failure (logged),
  mirroring `_read_optional_parquet`'s defensive behavior — a corrupt
  artifact is non-fatal.
- **Standalone bundle portability:** `qc.duckdb` requires DuckDB to open
  externally (unlike parquet). Accepted per D1; if a portable export is
  later wanted, it belongs with the deferred `qc.html` work.
- **Concurrency:** single GUI process = single DuckDB writer; recompute
  runs under the existing lock. CLI finalize writes via temp + atomic
  `os.replace`, so a viewer reading mid-finalize either sees the old DB or
  the new one, never a partial.
- **Windows open-handle caveat:** DuckDB takes an OS file lock on an open
  connection, and Windows `os.replace` fails if another process holds a
  handle on the target. Mitigation: the data API uses short-lived
  `read_only` connections (§6.6) that are never held across callbacks, so
  the GUI does not keep `qc.duckdb` open between reads; and the CLI rebuild
  wraps `os.replace` in a brief bounded retry on `PermissionError` for the
  rare race where a read overlaps the swap. POSIX is unaffected — an open
  handle keeps the old inode alive across the replace.
- **Legacy runs:** old `qc_*.parquet` artifacts are ignored (not read);
  the DB is rebuilt on the next finalize/recompile. `migrate_legacy_qc`
  still relocates a legacy `<output>/qc/` directory.

## 12. Out of scope / future

- **Static CLI `qc.html` dashboard.** Identified as a distinct second QC
  surface (the headless analog of `dashboard.html`, rendering each
  module's `dash()` figure + summary table). The per-module DuckDB tables
  + each check's existing `dash()` make it cheap to add later in the same
  finalize pass that already analyzes each check, but it is **deferred** to
  a follow-up spec. No `qc.html` is produced by this change.
- **Re-introducing a "download QC" affordance** (export the DB, or render
  parquet on demand) — deferred with `qc.html`.
- **Folding `review_state` into the DB** as a reset-on-rebuild table — a
  defensible middle path, rejected here to keep the rebuild a clean swap.

## 13. Open questions

- Exact `instance_id` → table-name sanitization rule (must be stable,
  collision-free, valid DuckDB identifier). Resolve in implementation.
- Whether `qc_<id>__summary` is a materialized table or a SQL view (§6.1
  sub-decision) — default materialized, reusing `_rank_worst_first`.
- Whether `run_qc` should always write a schema-only catalog vs. no-op when
  there are no enabled checks (§11) — default no-op.
