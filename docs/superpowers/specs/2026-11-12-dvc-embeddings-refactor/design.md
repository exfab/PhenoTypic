# Design: DVC-compatible outputs, CSV → TSV, and DINOv2 embeddings

**Status:** approved · 2026-11-12 (no open questions)
**Repos:** PhenoTypic (`src/` identical to `main` @ `251e364fa`), AutoConvertRaw-GC @ `33046cb`
**Companion artifacts (private):**
`docs/superpowers/artifacts/2026-11-12-dino-embeddings/measurement-data-flow.html`
(<https://claude.ai/artifact/Gr5JRyKm4kiCL9QYkDcyXd>) and
`docs/superpowers/artifacts/2026-11-12-dino-embeddings/dino-embedding-feasibility.html`
(<https://claude.ai/artifact/Uug4G7E8dzByEA6Ghfi7LT>). Where this document and the
feasibility page disagree, this document wins (§9 lists the corrections).

---

## 1. Intent

Two requests arrived together because both change the CLI and the per-image store:

1. **DINOv2 embeddings, and one store holding several pipelines' results.**
   - PhenoTypic gains a `MeasureDinoEmbedding` with two cases: embeddings of
     detected colony **crops**, and embeddings of **manually defined grid** cells.
   - AutoConvertRaw-GC's preprocessing (calibrate → denoise) runs grid embeddings
     in process mode and stores them in each OME-Zarr it publishes. A downstream
     analysis pipeline stores its own tables.
   - Each pipeline's tables live under `tables/<call order>-<pipeline hash>/`, so
     results from several pipelines coexist.
   - Nothing is silently overwritten or dropped by a later rewrite.
2. **DVC compatibility.** PhenoTypic runs should work as DVC pipeline stages, and
   every generated CSV should become TSV.

**What DVC is for** (user, 2026-11-12): compatibility with automation and
data-engineering pipeline infrastructure. An orchestrator (CI, Airflow, Prefect,
Dagster, a cloud scheduler) drives PhenoTypic unattended, decides from its outputs
whether to continue, and feeds its tables to other systems.

**The shape:** a PhenoTypic CLI run is one DVC stage.
- Its pipeline file and tune spec are DVC *params*.
- Its `deliverables/`, `results/` and `.phenotypic/` are DVC *outs*.
- It emits a machine-readable run summary as DVC *metrics*, plus TSV tables for
  `dvc plots`.
- It runs in a **persistent** workspace: the same `--output` on the cluster
  every time (user).

DVC versioning of the raw image corpus is out of scope (§2.2).

## 2. Goals, non-goals, anti-goals

### 2.1 Goals

- **G1** `dvc exp run -S <pipeline file>:<key>=<value>` edits a PhenoTypic pipeline
  file and the result still loads in PhenoTypic.
- **G2** A `dvc.yaml` stage that runs `python -m phenotypic` (local, or SLURM with
  `--wait`) reproduces without DVC deleting resumable state, figures or embeddings.
- **G3** Every generated tabular text file is TSV. Parquet companions are unchanged.
- **G4** A byte-stable `deliverables/metrics.json` run summary that
  `dvc metrics show/diff` and an orchestrator can both read (C4).
- **G8** An orchestrator can act on a run without parsing logs:
  - distinct exit codes (C10);
  - byte-stable deliverables, so unchanged inputs never trigger downstream stages
    (C11);
  - a machine-readable column schema (C12).
- **G5** One folder per measuring pipeline under the store's `tables/`, named
  `<call order>-<pipeline hash>`. Several pipelines' results coexist in one store,
  and every store rewrite carries the other pipelines' folders across.
- **G6** `MeasureDinoEmbedding` produces two kinds of table, each under its
  pipeline's folder and its model's name (CPU, v1):
  - **crop** embeddings, one vector per detected colony;
  - **grid** embeddings, one vector per manually defined grid cell.

  Both are aggregated to their own deliverable, never into master, mirror, splits
  or TSV.
- **G7** Process mode runs the pipeline's measurements and stores them in each
  store, inside the store's own transaction. AutoConvertRaw-GC stores thereby
  carry grid embeddings computed right after color correction and denoising.

### 2.2 Non-goals

- DVC tracking of the raw corpus or AutoConvertRaw-GC's bucket and NAS trees. Those
  keep their own marker system.
- Converting pipelines to YAML.
- A `dvc.yaml` generator. A documented template is enough.
- GPU execution of embeddings (Stage-2 dense features or a Stage 4). This is a
  later change, chosen after v1's CPU timings are measured.
- Dense patch-feature grids in stores (0.3–9.5 TB over the corpus).
- Copying an input store's run folders into result stores. A verifiable
  reference is recorded instead (D8).
- Aggregating an upstream (process-mode) run's tables into a downstream run's
  deliverables.
- Showing embeddings in the GUI.
- Byte-reproducible per-image stores from full runs. Their journals carry wall-clock
  times. Deliverables *are* in scope (C11).

### 2.3 Anti-goals (failure even if everything "works")

- A store rewrite that drops another pipeline's run folder or a carried
  embedding entry.
- A re-measure that renumbers or duplicates a pipeline's folder.
- A per-colony vector readable against an objmap it was not computed from.
- Embedding vectors appearing in `master_measurements.parquet`,
  `measurements.*`, the splits, or any TSV.
- `dvc repro` deleting a run's resumable state, figures, or embeddings.
- Any rewrite of the user's metadata snapshot bytes.
- The suffix rename changing AutoConvertRaw-GC's `pipeline_sha256`.

## 3. Constraints (verified)

### 3.1 DVC

Verified 2026-11-12 against the DVC docs (context7 `/treeverse/dvc.org`) and the
DVC source (`iterative/dvc@main`).

| Fact | Source |
|---|---|
| Params files may be YAML 1.2, JSON, TOML 1.0 or Python. | docs: `user-guide/project-structure/dvcyaml-files.md`, `command-reference/params` |
| The parser is chosen by the **last** suffix, lower-cased. `.json`, `.toml` and `.py` have handlers; **any other suffix gets the YAML loader, dumper and modifier.** | source: `dvc/utils/serialize/__init__.py` (`LOADERS`/`DUMPERS`/`MODIFIERS` are `defaultdict(lambda: …yaml)`) |
| `modify_json` writes through `json.dump` with no indent or sort arguments. The intermediate helpers were not inspected, so the exact output formatting is **unverified**; expect a one-line file. | source: `dvc/utils/serialize/_json.py` |
| Stage outputs are deleted before the stage runs unless `persist: true`. | docs: `command-reference/repro.md` |
| `cache: false` on any output disables the run cache for that stage. | docs: `dvcyaml-files.md` output subfields |
| `cache.type` default is `reflink,copy`. Reflinks need Btrfs, XFS or OCFS2 on Linux, so on GPFS every checkout is a **copy**. With `hardlink`/`symlink`, DVC makes linked files read-only, and modifying them in place corrupts the cache. | docs: `project-structure/configuration.md`, `data-management/large-dataset-optimization.md` |
| `.dvcignore` patterns from a higher directory can exclude paths inside an output directory, but `dvc repro`/`dvc exp run` **may remove ignored files**. A `.dvcignore` inside an output directory is an error. | docs: `project-structure/dvcignore-files.md` |
| Metrics: JSON, TOML 1.0, YAML 1.2. Plots: JSON, YAML 1.2, CSV, **TSV**, or images. Parquet is neither. | docs: `command-reference/metrics`, `command-reference/plots` |
| `dvc.yaml` may declare `metrics:`/`plots:` at top level, outside any stage, pointing at files inside a stage's directory out. Two stages may not declare the same or overlapping outs. | docs: `start/data-pipelines/metrics-parameters-plots.md`, `command-reference/stage/add.md` |
| A directory output is one `.dir` cache entry. For "tens of thousands of small files or object-store-scale datasets" DVC's docs point to lakeFS. | docs: `command-reference/add.md` |

### 3.2 PhenoTypic

| Fact | Where |
|---|---|
| Typed config suffixes are `.json.pht-pipe`, `.json.pht-op`, `.json.pht-cc`, `.json.pht-tune`. Plain `.json` is already accepted as legacy for pipelines and tune specs. Matching is centralised in `has_config_suffix` / `ensure_typed_json_suffix`. | `sdk_/_io_constants.py:103-154` |
| Suffix footprint: `pht-pipe` in 20 src / 54 test / 10 doc files; `pht-op` 3/2/0; `pht-cc` 3/2/0; `pht-tune` 11/21/0. No such file is git-tracked. | grep, 2026-11-12 |
| The output root holds three siblings: `deliverables/`, `results/`, `.phenotypic/`. | `_cli_readme_generator.py:98-125`, `phenotypic_cache_dir` (`_io_constants.py:1069`) |
| Generated CSV writers: the mirror and splits (`_cli_output_manager.py`), named analysis (`:752`), error-analysis publication (`_gui/results_viewer/_error_tab/_publication.py:588`), curation labels (`_curation_labels.py:965`), filtered-state export (`_filtered_state.py:586`), migrate's `metadata.canonical.csv` (`sdk_/_metadata_migration.py:2935`). The analysis manifest records `csv` and `csv_sha256` fields (`plotting/_pipeline/_analysis_artifacts.py:37-41`). | grep |
| Input readers: `join_metadata` already picks tab vs comma by suffix (`post/_join_metadata.py:186`). `_metadata_migration.py:469` and `_cli_reference.py:400` assume comma. `--metadata` is CSV-only in the CLI (`metadata_csv_payload`). | grep; `_cli/CLAUDE.md` |
| A full store save starts from an empty `.part` and carries only figure runs. The root is rebuilt. | `_image_io_handler.py:1468`, `:1478` |
| The measure/recompile rewrite hard-link-copies the store, then `rmtree(part/tables)`. Top-level groups and root attributes survive. | `sdk_/_measurement_tables.py:638-718` |
| Process stores are consolidated inside the `.part` and are "written once and never mutated". | `_cli_process_only.py:224-247`, `_image_io_handler.py:1511` |
| `MeasureFeatures` frames merge into the one measurement table. A column no `MeasurementInfo` owns is copied into every split. | `_image_pipeline_core.py:1122`, `util/_measurement_outputs.py:96,200` |
| A `GpuDetector` in any `meas` slot is refused. Non-`GpuDetector` measurers are not. | `_cli_validation.py:324-367` |
| DINO loading and patch extraction already exist: `load_dino_backbone` (:193), `hf_dino_id` (:172), `extract_patch_features` (:286), plus the license gate. | `detect/nn/_helper/_dino_support.py`, `_checkpoint_manager.py` |
| Process mode executes only `ops` (`MODE_SLOTS`). | `_cli/CLAUDE.md`, run preflight |
| SLURM runs without `--wait` return at submission; the finalizer publishes later. | `_cli/CLAUDE.md` "SLURM chaining" |

### 3.3 AutoConvertRaw-GC

| Fact | Where |
|---|---|
| Runs `process_single_apply_only_core(layer="rgb", process_format="zarr")` in a child process. | `src/acr_core/process_one.py:118` |
| Publishing deletes the remote prefix, then uploads the tree. | `src/acr_core/worker.py:189-190` |
| The run-record pipeline name is `pipeline.json.pht-pipe`. The configs point at `pipelines/exfab_rig1.json.pht-pipe`. `pipeline_sha256` hashes file bytes. | `src/acr_core/plan.py:27`, `configs/*.yaml:32-33`, `worker.py:286` |
| GC VMs are Spot `c3d-highmem-4` with no external IP; the pilot ran 612 s/image with a 16.3 GB peak. The PhenoTypic wheel is vendored at 0.20.2. | `configs/exfab.yaml:40-49`, `vendor/` |

### 3.4 Prior art: tables and embeddings in OME-Zarr

Checked 2026-11-12.

| Fact | Source |
|---|---|
| OME-NGFF has **no table or embedding spec**. The AnnData-based table proposal, whose stated scope included low-dimensional embeddings, was closed unmerged on 2023-09-25. | <https://github.com/ome/ngff/pull/64> |
| RFC-8 (Collections and Extensibility) is "R1, under review". It defines no table format; it links derived data to images, and allows prefixed extension node types and attribute keys. | <https://ngff.openmicroscopy.org/rfc/8/index.html> |
| ngio/Fractal keeps tables in `<image>/tables/<name>/`, with Parquet as one allowed backend. A feature table carries `type: "feature_table"`, `table_version`, `region` (label path), `backend`, `index_key`, `index_type` and `instance_key`. | <https://biovisioncenter.github.io/ngio/stable/table_specs/overview/>, `…/table_types/feature_table/` |
| SpatialData stores tables as AnnData. Per-object embeddings conventionally live in `obsm/<key>` as N × D matrices, and the table links to labels through `uns/spatialdata_attrs` (`region`, `region_key`, `instance_key`). | <https://spatialdata.scverse.org/en/latest/generated/spatialdata.models.TableModel.html> |
| Label `image-label.properties` holds a per-`label-value` JSON object. That is fit for class names, not vectors. | <https://ngff-spec.readthedocs.io/en/latest/schemas/label> |

## 4. Approaches considered

### 4.1 Pipeline file format

| Option | Trade-off | Verdict |
|---|---|---|
| **Keep JSON, flip the suffix to `.pht-pipe.json`** | DVC uses its JSON handler; content and `pipeline_sha256` are unchanged; a small, central change | **picked** |
| Convert to YAML | Works with DVC, but changes every pipeline's hash, which forces an AutoConvertRaw-GC mass reprocess; adds a second format to support | rejected |
| Keep the suffix and ship a `params.yaml` mirror | Two sources of truth that drift | rejected |

### 4.2 Embedding storage

| Option | Trade-off | Verdict |
|---|---|---|
| **Per-pipeline folders under `tables/`, with embeddings inside their pipeline's folder** | `tables/` sub-schema v2 plus a migration; several pipelines coexist; the carry step is per run folder | **picked** (user) |
| Top-level `embeddings/` namespace with its own carry step | Separates embeddings from the pipeline that produced them, and still leaves one flat measurement table per store | superseded |
| `tables/embeddings/` beside today's flat tables | Deleted by every `--mode measure` (`rmtree(part/tables)`) | rejected |
| Wide columns in the measurement table | Spreads to master, mirror, every split and every TSV | rejected |
| Sidecar file outside the store | Robust, but not in the OME-Zarr store, which the user asked for | fallback only |

### 4.3 AutoConvertRaw-GC integration

A **native image-scope embedder run by process mode** (picked). The alternatives
were a post-step in the child that mutates and re-consolidates a promoted store,
and a sidecar file. Both break the immutability or in-store requirements.

### 4.4 Where per-colony embedding runs

**CPU inside `meas`** (picked for v1; no staged-engine change). The deferred
alternatives are dense features from Stage 2 pooled in Stage 3, and a new GPU
Stage 4.

### 4.5 DVC output layout

| Option | Trade-off | Verdict |
|---|---|---|
| **Three outs, `deliverables/`, `results/` and `.phenotypic/`, all `persist: true`** | A `dvc checkout` of any revision restores the run state that matches its stores. Costs: churn in `dvc.lock` every run, a growing cache, and no hardlink/symlink cache (C6) | **picked** <span>(user)</span> |
| Two outs, `.phenotypic/` untracked | Less churn, but rolling back outputs leaves state from another revision behind | superseded |
| Move `.phenotypic/` outside the output root | Large change to continuation, SLURM scripts, locks and Stage-2 signals | rejected |
| The whole output root as one out | Works, but downstream consumers can't name "just the deliverables", and `results/` can't get its own cache setting (C8) | rejected |

### 4.6 Linking a result store to its input store

| Option | Trade-off | Verdict |
|---|---|---|
| A · copy the input's run folders in | Self-contained, but duplicates another pipeline's results into a store it never wrote, and the copy goes stale silently | rejected |
| **C · record a reference** (input path + root digest) | No copy; a changed input is detectable; reading upstream tables needs the input store to be reachable | **picked** (call) |
| D · record nothing | Simplest, but you can't tell whether the input changed | rejected |

### 4.7 Tabular text format

**Generated files become TSV only** (picked). Writing CSV and TSV side by side
doubles the files and leaves readers unsure which one is current.

## 5. Design

The five parts below are independent enough to plan and ship in the order of §6.

### Part A: typed config suffix flip

- **A1** New canonical suffixes: `.pht-pipe.json`, `.pht-op.json`, `.pht-cc.json`,
  `.pht-tune.json`. The old `.json.pht-*` forms become **legacy read** suffixes.
  Plain `.json` stays legacy-readable where it already is (pipelines, tune specs).
- **A2** `ensure_typed_json_suffix(path, suffix)`:
  - A bare stem gets the new suffix.
  - `x.json` becomes `x.<typed>.json` (insert before `.json`). Today it becomes
    `x.json.<typed>`.
  - A path that already ends in a recognised suffix, old or new, is returned as
    given. *(Builder's call: honouring an explicit old suffix avoids writing a
    file somewhere other than where the user asked.)*
- **A3** Canonical run artifacts use the new name:
  `deliverables/pipeline.pht-pipe.json`, `deliverables/best_pipeline.pht-pipe.json`,
  and `.phenotypic/pipeline.pht-pipe.json`.
  - Every reader (`resolve_*`, `BundleLayout`, the GUI pickers, migrate's
    `*.pht-pipe` glob at `_cli_migrate.py:679`) also accepts the old names.
  - `migration_lock_path` maps legacy `pipeline.json`, the old name and the new
    name to the **same** lock identity, extending the existing rule at
    `_io_constants.py:248-259`.
- **A4** One current pipeline file per folder:
  - `_seed_pipeline_config` seeds only when no config exists under any of the
    three names.
  - When finalize rewrites the pipeline file, it publishes the new name first and
    then removes an old-name sibling. *(Builder's call.)*
- **A5** AutoConvertRaw-GC:
  - rename `pipelines/exfab_rig1.json.pht-pipe` to `pipelines/exfab_rig1.pht-pipe.json`;
  - set `PIPELINE_NAME = "pipeline.pht-pipe.json"`;
  - update the three configs and `pipeline_info.py`'s docstring.

  The bytes are unchanged, so `pipeline_sha256` is unchanged. A test asserts that.
- **A6** DVC note for the how-to page (§5 Part C): DVC addresses params by dotted
  path (`pipe_cfgs.<OpKey>.params.<field>`). An operation key containing `.` cannot
  be addressed. Document it; no code check in v1.

### Part B: CSV → TSV

- **B1 Format.** UTF-8, one header row, `\t` separator, `\n` line ends, with fields
  containing tab, newline or quote quoted with `"`. This is polars `write_csv(
  separator="\t")` / pandas `to_csv(sep="\t")` defaults. Extension `.tsv`.
- **B2 Generated files that switch:**
  - `measurements.tsv`
  - `measurements_by_feature/<Measurer>.tsv`
  - `measurements_by_category/<Category>.tsv`
  - `<AnalysisClass>.tsv`
  - `error_analysis.tsv`
  - the GUI curation-label, filtered-state and error-tab exports
  - migrate's `metadata.canonical.tsv`

  Rename the constants (`MEASUREMENTS_CSV` → `MEASUREMENTS_TSV`, …) and keep
  `_LEGACY_*_CSV` names for reads.
- **B3 Analysis manifest.** Fields `csv`/`csv_sha256` become `tsv`/`tsv_sha256`,
  with a manifest `schema_version` bump. Readers accept the old shape.
- **B4 User-supplied inputs accept both** `.csv` and `.tsv`, delimiter chosen by
  suffix, as `join_metadata` already does. That covers `--metadata`, reference
  metadata and tune ground truth. Extend this to `_metadata_migration.py:469`,
  `_cli_reference.py:400` and `metadata_csv_payload`. Any other suffix is still
  refused.
- **B5 Snapshots keep the user's bytes and extension:**
  `deliverables/metadata.<ext>` and `.phenotypic/reference_metadata.<ext>`, where
  `<ext>` is the input's extension. Readers locate the snapshot under either
  extension. Existing trees keep `metadata.csv`.
- **B6 Old run folders.** Readers (results viewer, analysis app, `BundleLayout`,
  the README generator) fall back to the `.csv` name when no `.tsv` exists. A
  finalize into an old folder publishes the TSV, then removes the stale
  **generated** CSV sibling. This applies only to files in the B2 list, never to
  the metadata snapshot. *(Builder's call.)*
- **B7 Public API is not renamed.** A Python method or argument named `csv` keeps
  its name and behaviour. Docstring examples that write `.csv` (e.g.
  `_objects_accessor.py:690`) stay user-chosen. B applies to files the CLI and GUI
  generate.
- **B8** Update the README generator, the how-to and tutorial pages,
  `FEATURES.md`/`WORKFLOWS.md` where they name a file (`gui-tutorial-capture`
  skill), and the root and `_cli` CLAUDE.md output inventories.

### Part C: DVC-compatible run layout

- **C1 Outs.** The supported DVC shape is three outs, all `persist: true`
  (user decision):
  - `<out>/deliverables`
  - `<out>/results`
  - `<out>/.phenotypic`

  Add a regression test that no PhenoTypic path writes anything at the output
  root outside these three. Metrics and plots are declared in `dvc.yaml`'s
  **top-level** `metrics:`/`plots:` sections. Those may point inside an out
  (`deliverables/metrics.json`), so they create no overlapping outputs (DVC docs,
  `start/data-pipelines/metrics-parameters-plots.md`).
- **C2 `persist: true` is required, not optional.** Continuation, figure runs and
  embeddings all depend on outputs surviving a re-run. `--overwrite` remains the
  only wipe.
- **C3 SLURM.** A DVC stage must pass `--wait`. Without it the command returns
  before the finalizer publishes, and DVC hashes an incomplete tree. Document this.
  *(Optional for the builder: if DVC sets a documented environment variable during
  `repro`, warn when `--slurm` is used without `--wait`. Don't add the check
  without that documentation.)*
- **C4 Metrics.** New `deliverables/metrics.json`, written by `finalize_run` (the
  one FINAL writer):
  - **Content:** `schema_version`, `phenotypic_version`, `pipeline_sha256`;
    `images.{accepted, succeeded, failed}`; `objects.measured`;
    `metadata.phantoms`; and, when embeddings ran, per entry id `{rows, stale_skipped}`.
  - **Derivation:** values come from the run state (`resolve_run_state`) and the
    published frames, never from a new tracked value. This is a deliverable that
    PhenoTypic never reads back, so it is not a fifth tracked state.
  - **Stability:** no timestamps, sorted keys, so identical runs give identical
    bytes and `dvc metrics diff` is meaningful.
- **C5 Params.** The `--pipeline` file and any tune spec are listed as DVC params
  files. With A1 they end in `.json`, so DVC reads and writes them as JSON.
- **C6 Cache type: `reflink,copy` only.** With `.phenotypic/` tracked,
  `hardlink` and `symlink` are **unsupported**.
  - DVC makes linked files read-only. `.phenotypic/` holds files that are appended
    in place by concurrent writers: the terminal-failure journal, the event log
    and the SLURM lifecycle ledger.
  - These cannot become write-then-rename: concurrent SLURM workers rely on
    appends.
  - In-place writers under `deliverables/` (DuckDB `qc.duckdb`, GUI curation)
    have the same problem.
  - On GPFS the default `reflink,copy` resolves to copy, so the default is already
    correct. It costs one extra copy of each output in the cache.

  The run preflight gains a warning when the enclosing DVC project's
  `.dvc/config` selects `hardlink` or `symlink` *(call)*. The check reads one
  config file and never imports DVC.
- **C7 Post-run curation.** GUI curation changes files inside the `deliverables`
  out, so `dvc status` reports the out as modified until the user runs
  `dvc commit`. This is intended (user decision, §8 Q2): no code change, and the
  how-to page documents the commit step.
- **C8 Results size.** Whether `results/` should be `cache: true` or `false` is
  decided by measurement, not reasoning. Time `dvc status` and `dvc commit` on the
  migrated maresca copy (`2026-08-11-ome-zarr`) as a Slurm job, one config per
  fresh node, and record the numbers in the how-to page.
- **C9 How-to page** `docs/source/how_to/pages/dvc.md`. Contents:
  - a local `dvc.yaml`, a SLURM one (`--wait`), and a process-mode one, since
    process mode creates no `deliverables/` and its outs list must say so;
  - the three outs with `persist`, and the cache-type rule (C6);
  - `dvc gc` for the cache growth that tracking `.phenotypic/` causes;
  - params addressing with an example `dvc exp run -S`;
  - top-level `metrics:`/`plots:`;
  - the one-line-JSON caveat (§3.1) and the `dvc commit` step after curation.
- **C10 Exit codes are a contract** *(call)*. Four distinct codes:
  - **0**: complete, no terminal failures;
  - a distinct non-zero: **complete with terminal image failures**;
  - **1**: refused, or a usage/config error (today's behaviour);
  - a distinct code: **submitted to SLURM without `--wait`**, so nothing is
    finished.

  The builder confirms what a run with terminal failures returns today; I found
  no distinct code for it. The values are documented and pinned by tests.
- **C11 Byte-stable deliverables** *(call)*. For identical inputs, pipeline and
  PhenoTypic version, these files are byte-identical across runs:
  - every table under `deliverables/` (Parquet and TSV);
  - `deliverables/embeddings/`;
  - `metrics.json`;
  - `schema.json`.

  That means no wall-clock or host values in them, stable row and column order
  (the contiguous shard split already gives order independence, `87f933cb`), and
  pinned Parquet writer metadata. Reports, dashboards and `.phenotypic/` are
  exempt. A test runs the same tiny run twice and compares digests. This matters
  because DVC re-runs every downstream stage when an upstream output's bytes
  change.
- **C12 `deliverables/schema.json`** *(call)*. Every column of each published
  table, with:
  - name, dtype and unit;
  - the producing measurer;
  - the `MeasurementInfo` member and its `desc`.

  It is generated from the schema enums the README generator already reads (one
  producer). Embedding tables list `embedding` as `FixedSizeList<float16>[D]`
  with the model.
- **C13 Restored state fails safe.** A `dvc checkout` of another revision restores
  `.phenotypic/` from that revision.
  - Per-image records bind store digests, so a record restored beside a store it
    doesn't describe is rejected, and the image re-runs. That outcome is behind,
    never wrong.
  - Liveness and ownership files (tracked state #3) restored from another revision
    must not block or mislead. The live-SLURM guard already asks the scheduler
    rather than trusting the fence. The builder verifies the same for
    `gui_launch_owner` and the migration lease.
- **C14 Outs must exist.** DVC expects declared outs to exist after the command.
  The builder verifies how DVC reports a missing out, then makes every mode create
  the outs its template declares, even on a run where every image fails.

### Part D: per-pipeline tables in the store

This part replaces the earlier top-level `embeddings/` group. An embedding belongs
to the pipeline that produced it, exactly as that pipeline's scalar measurements
do, so both live in the pipeline's own folder under `tables/` (user decision).

- **D1 Layout.** `tables/` stays the top-level group, since it is already part of
  the store schema. Below it there is one folder per pipeline application that
  wrote measurements into this store:

  ```
  tables/
  ├── <seq>-<sha12>/                       one folder per measuring pipeline
  │   ├── measurements/table.parquet       scalar measurements (was tables/measurements/)
  │   ├── metadata/pht-metadata.parquet    metadata projection (was tables/metadata/)
  │   └── embeddings/<model>/              e.g. dinov2-vitb14
  │       ├── crops/table.parquet          Object_Label | embedding
  │       └── grid/table.parquet           Grid_RowMajorIdx | Grid_RowNum | Grid_ColNum |
  │                                        Grid_{Row,Col}Interval{Start,End} | embedding
  └── …
  ```

  - `embedding` is `FixedSizeList<float16>[D]`.
  - Every level is a Zarr v3 group document.
  - Every subtable is optional. AutoConvertRaw-GC's folder holds only
    `embeddings/<model>/grid/`.
  - `<model>` names the model, so a second model is a sibling folder, never extra
    columns.
  - Nothing goes in `ome.series` or `attributes.ome`.
- **D2 Folder name.**
  - `<seq>` is the provenance journal's application `sequence` for the
    application that **first** wrote the folder.
  - `<sha12>` is the first 12 hex of the pipeline sha256 the journal records,
    which is the pipeline file's bytes. It is the same digest the figure run
    folders and AutoConvertRaw-GC use.
  - The builder verifies how a staged run (Stages 1 and 3) and `--mode measure`
    map onto journal applications before relying on `<seq>`.
- **D3 Descriptor.** `attributes.phenotypic.tables` moves to `schema_version: 2`:

  ```json
  { "schema_version": 2,
    "runs": {
      "2-a71e09c2b4f5": {
        "sequence": 2, "pipeline_sha256": "<64 hex>",
        "measurements": { "…today's v1 measurement descriptor: path, measurement_columns, target…" },
        "metadata":     { "…today's metadata descriptor…" },
        "metadata_table": { "snapshot_sha256": "…", "join_keys": ["…"], "join_kind": "…" },
        "embeddings": {
          "dinov2-vitb14/crops": { "…entry…" },
          "dinov2-vitb14/grid":  { "…entry…" } } } },
    "upstream": {
      "1-3f9c2a1b7d04": { "source": "<input path relative to --input>",
                          "root_sha256": "<64 hex>", "pipeline_sha256": "<64 hex>" } } }
  ```

  The root-level `metadata_table` block moves into its run. Each embedding entry
  has these fields:

  | Field | Contents |
  |---|---|
  | `model` | `{family, hf_id, revision, patch_size}`, where `revision` is the resolved hub commit |
  | `dim`, `dtype` | vector length and stored type |
  | `pooling` | the pooling mode used |
  | `key` | `Object_Label` for crops, `Grid_RowMajorIdx` for grid |
  | `input` | `{series, level}` |
  | `producer` | `{class, params_sha256}` |
  | `binds` | for crops, `{objmap_path, objmap_sha256}`; for grid, `{array_sha256, grid_sha256}` |
  | `stale` | `false` when written |
  | `files` | `[{path, sha256, media_type}]` |
- **D4 Interop attributes** go on each table group (§3.4):
  - Scalar `measurements/` groups carry ngio's feature-table attributes:
    `{type: "feature_table", table_version: "1", region: {path: <relative path to
    the objmap label>}, backend: "parquet", index_key: "Object_Label", index_type:
    "int", instance_key: "Object_Label"}`.
  - Embedding table groups carry `type: "phenotypic:embedding_table"`, a prefixed
    extension type in RFC-8's style, plus `backend` and `index_key`.
  - The builder installs ngio in a test and checks whether it discovers tables
    nested one level down. If it doesn't, keep the attributes, which are harmless,
    and record the limit. Do not flatten the layout to suit ngio.
- **D5 Carry rule.** Every writer carries every run folder other than its own,
  and the `upstream` block, unchanged. The writers are:
  - save2zarr (full, Stage 1, Stage 3);
  - process `_save_store`;
  - the measure/recompile rewrite.

  Further rules:
  - The measure rewrite's `rmtree(part/tables)` (`_measurement_tables.py:694`)
    narrows to `rmtree(part/tables/<own run>)`. That folder's files are rewritten
    as new inodes, which is the figures rule.
  - Only `--overwrite` deletes run folders.
  - Forward modes refuse a store still in the v1 flat layout and point at
    `--mode migrate`, as they already refuse unconverted trees.
- **D6 Which folder a writer owns.** A writer looks for an existing run folder
  with its pipeline's sha256.
  - If one exists, it replaces that folder in place, keeping its original
    `<seq>`. This covers `--mode measure` and re-runs.
  - Otherwise it creates `<this application's sequence>-<sha12>`.
  - The journal still logs every application.
  - This matches how measure mode already reuses a figure run folder for the same
    pipeline. *(Call.)*
- **D7 Stale binding** is recomputed on every write, never latched:
  - **Crops:** compare the entry's `objmap_sha256` with the objmap being written.
  - **Grid:** compare `array_sha256` with the embedded level-0 array.
  - Set `stale = (they differ)`, and keep the files either way.
  - Readers and aggregation refuse a stale entry.
  - A latched flag would be wrong. Staged Stage 1 writes an all-zeros objmap,
    which would mark every carried crop entry stale even after Stage 3 restores
    the identical objmap.
  - Digests are computed from in-memory arrays at write time.
  - The measure rewrite writes no objmap and leaves `stale` unchanged.
- **D8 Upstream reference (option C).**
  - When an input image is a PhenoTypic store with a v2 `tables` descriptor, the
    result store's `upstream` records each of the input's run folders, along with
    any entries in the input's own `upstream`, so the chain is complete.
  - Nothing is copied.
  - A reader that follows a reference compares `root_sha256` and reports
    "upstream changed" on a mismatch, for example after an AutoConvertRaw-GC
    reprocess. *(Call.)*
- **D9 Integrity.**
  - File `sha256`s live in the root, so the completion marker covers them.
  - Missing or corrupted embedding bytes make that entry unreadable. They do not
    revoke the run's measurement authority: embeddings are additive. *(Call.)*
- **D10 Migration.** `--mode migrate` converts a v1 store's `tables/measurements`
  and `tables/metadata` into `tables/<seq>-<sha12>/`. `<seq>` and the hash come
  from the latest journal application that measured. When the journal has no
  pipeline digest for it, which is possible for a converted `legacy` application,
  the folder is `<seq>-unknown` with `pipeline_sha256: null`. Never fabricate a
  digest. *(Call.)*

### Part E: producers, deliverables, AutoConvertRaw-GC

- **E1 `MeasureEmbedding` ABC** (`abc_/`), a `MeasureFeatures` subclass with
  `regions: Literal["objects", "grid"]`:
  - `objects` requires a non-empty objmap and writes `crops/`.
  - `grid` requires a grid definition, not an objmap, and writes `grid/`.

  A whole-image embedding is a 1 × 1 grid, so there is no separate image scope.
  *(Call.)* `measure(image)` returns the table's frame directly, for notebook use.
- **E2 Pipeline routing.**
  - `ImagePipeline.measure()` keeps its return contract and does not run
    `MeasureEmbedding` entries.
  - A new `ImagePipeline.embed(image) -> dict["<model>/<crops|grid>", EmbeddingPayload]`
    runs them.
  - The CLI per-image writers call both, then write one run folder (D1). These are
    the single-pass worker, Stage 3, measure mode and process mode.
  - Both docstrings say so. *(Call.)*
- **E3 `MeasureDinoEmbedding`** (`measure/_measure_dino_embedding.py`). Fields:

  | Field | Type, default | Notes |
  |---|---|---|
  | `dino_version` | `DinoVersion = 2` | |
  | `dino_size` | `DinoSize = "base"` | |
  | `regions` | `"objects"` | |
  | `grid` | `ManualGridFinder` (an `OperationField`) | required when `regions="grid"`, refused otherwise; the rig's cell edges in pixels (E3a) |
  | `pooling` | `"crop_cls"` | `crop_cls`, `crop_mean` or `masked_dense`; a grid cell is cropped from its rectangle exactly as an object is from its bbox |
  | `crop_px` | `224` | validated as a multiple of the loaded patch size |
  | `crop_pad_frac` | `0.25` | objects only |
  | `mask_background` | `False` | objects only |
  | `dtype` | `"float16"` | |
  | `batch_size` | `64` | |
  | `device` | `"auto"` | |

- **E3a Grid coordinates are per-rig parameters** (user decision). A rig puts
  plates at the same pixels in every image, so a grid is fixed pixel edges, the
  way the checker-card ROIs are fixed pixel boxes. The edges are **parameters,
  never constants**, so a user with a different rig sets their own:
  - They live in the pipeline file as `ManualGridFinder(row_edges=[…],
    col_edges=[…])`. Changing rigs means changing those two lists, or using a
    pipeline file per rig.
  - They are addressable as DVC params, e.g.
    `dvc exp run -S 'rig2.pht-pipe.json:meas.<key>.params.grid.params.row_edges=[…]'`.
    The builder confirms the exact key path the serializer produces.
  - Coordinates are in the pixel frame of the image the measurer sees, after
    `ops`. Color correction and denoising don't change geometry.
  - A cell that falls outside an image fails that image, with a message naming the
    cell and the image size, rather than being clipped *(call)*.
  - For `phenotypic-tune`, the edges are `TuneSpec(tunable=False)` structural
    plate geometry, as `ManualGridFinder.nrows/ncols` already are. They are
    user-set, not searched *(call)*.
  - No coordinates are shipped as defaults. The rig1 edges are supplied by the
    user when the AutoConvertRaw-GC pipeline is updated (E6).

  Implementation notes:
  - The model token is `dinov{version}-vit{s|b|l}{patch}`, e.g. `dinov2-vitb14`.
  - Reuse the `_dino_support.py` helpers.
  - Keep a module-level backbone cache keyed by `(hf_id, revision, device)`.
  - Run under `torch.inference_mode()` with `model.eval()`.
  - Import heavy modules lazily (startup-import guards).
  - DINOv3 goes through `PHENOTYPIC_ACCEPT_MODEL_LICENSE`.
  - Declare `preflight_requirements()` for `torch`/`transformers`.
  - Tune annotations go on numeric fields, per the `adding-an-operation` skill.
  - Add a `DINO_EMBEDDING` schema enum, authoring `label`/`desc` only (no
    `bio_desc`), exported from `phenotypic.schema`.
- **E4 Process mode runs the pipeline's measurements** (user decision).
  - `MODE_SLOTS["process"]` becomes `ops` + `meas`, and the preflight's mode walk
    follows. `post`/`filters`/`model` still don't run, and no deliverables are
    built.
  - `process_single_apply_only_core` runs `measure()` and `embed()` after
    `apply()`, and writes the run folder into the `.part` before consolidation.
    The store stays write-once.
  - A process store holds no objmap (`write_objmap=False`). So a measurer that
    needs objects is refused in the read-only half, with one usage-error line
    naming it, rather than writing a table keyed on labels the store can't
    resolve. *(Call.)*
  - **Bump `PROCESS_LAYER_SEMANTICS_REVISION` 4 → 5.** Process mode used to ignore
    `meas`, so a process tree whose pipeline already had measurers would otherwise
    resume onto stores that lack the tables it now writes. This corrects an
    earlier statement that no bump was needed.
- **E5 Deliverables.** The finalizer aggregates **its own** run folder:
  - the master comes from `measurements/`, exactly as today;
  - each embedding table goes to
    `deliverables/embeddings/<model>/<crops|grid>.parquet`, with the master's
    intrinsic identity columns plus the table's key and `embedding`;
  - `deliverables/embeddings/manifest.json` holds each entry's descriptor, row
    count, sha256 and the stores skipped as stale.

  Rules:
  - These files are Parquet only. They are never joined to metadata, and never in
    master, mirror, splits or any TSV.
  - A failure fails finalization, like the master. *(Call.)*
  - Upstream tables (D8) are not aggregated into a downstream run's deliverables
    in v1.
- **E6 AutoConvertRaw-GC rollout** (after the new wheel is built):
  - add `MeasureDinoEmbedding(regions="grid", grid=ManualGridFinder(...),
    dino_size="base")` to the pipeline's `meas`, with rig1's edges supplied by the
    user;
  - vendor the new wheel;
  - install the `foundation` extra in the `Dockerfile` with CPU torch, bake in the
    pinned weights, and set `HF_HUB_OFFLINE=1`;
  - make `verify_store` check that the run folder's grid entry exists (an
    `expected_embeddings` field in `run.json`, written by `plan.py`), the way
    figures are checked;
  - re-pilot memory and timeout on the smoke configs, never on `data/`.

  Backfill of done images is an explicit `--reprocess-changed` decision at
  rollout. It reruns calibration and BM3D denoise.

## 6. Order

| Phase | Content | Depends on |
|---|---|---|
| P0 | Part A suffix flip, PhenoTypic and AutoConvertRaw-GC | none |
| P1 | Part B TSV | none (parallel with P0) |
| P2 | Part C DVC layout, `metrics.json`, how-to page, C8 measurement | P0, P1 (the template names their files) |
| P3 | Part D: `tables/` v2 per-pipeline layout, carry rule, upstream reference, ngio attributes, `--mode migrate` conversion, reader updates (finalizer, GUI, measure, marker) | none (parallel with P0–P2) |
| P4 | Part E1–E3a, E5: `MeasureEmbedding`, `MeasureDinoEmbedding` (crops + grid), `embed()`, deliverables | P3 |
| P5 | Part E4 + E6: process-mode measurements + AutoConvertRaw-GC rollout | P3, P4, P0 |
| Later | GPU execution (Stage-2 dense or Stage 4) | P4 timings |

P3 is now the largest phase. Every reader of `tables/measurements/table.parquet`
changes: the finalizer, the GUI routes, measure mode, migrate and the completion
marker. Derive its affected test surface from importers, not from directory names
(root CLAUDE.md).

Testing follows root CLAUDE.md: step guards, then per-task files, the affected
surface once per phase, and the full sharded regression once at the end, on Slurm.
Store changes also need an **independent reader** check (zarr-python + pyarrow,
never only PhenoTypic's own reopen), per the `working-with-ome-zarr` skill.

## 7. Who decided what

**Decided by the user:**
- flip the suffix (all four typed suffixes);
- keep JSON (no YAML);
- all generated CSV becomes TSV, as a hard cutover;
- curation leaves `deliverables` modified until `dvc commit`;
- DVC is for automation and data-engineering pipeline infrastructure; a run is
  one DVC stage;
- automation runs in a persistent workspace;
- `.phenotypic/` is tracked as a DVC output, beside `deliverables/` and
  `results/`;
- the feasibility recommendations:
  - CPU-first `MeasureDinoEmbedding` in `meas`;
  - AutoConvertRaw-GC Option A;
  - DINOv2 base, fp16, `crop_cls`;
  - `FixedSizeList` vectors;
  - embeddings out of the scalar table and its own deliverable;
- the store layout: one folder per pipeline under `tables/`, named
  `<call order>-<pipeline hash>`;
- embeddings under their pipeline's folder, with the model naming the folder;
- crop and grid embeddings as two separate tables;
- grid embeddings from manually defined grid coordinates;
- process mode runs the pipeline's measurements and stores them in each store;
- the grid is fixed per rig, with user-set coordinates;
- `tables/` stays the top-level group.

**Calls made in this document (check these):**
- A2 honours explicit old suffixes;
- A4 removes the old-name pipeline file after publishing the new one;
- B5 keeps the snapshot's extension;
- B6 removes stale generated CSVs;
- B7 leaves public API names alone;
- C4's metrics content;
- D2 keys folders by pipeline-file bytes, so a formatting-only change gives a new
  folder;
- D4 adds ngio attributes to scalar tables and a prefixed type to embedding tables;
- D6 has a re-measure with the same pipeline replace its folder and keep its
  `<seq>`;
- D7 marks stale entries rather than deleting them;
- D8 records the upstream link as a reference (option C), not a copy;
- D9 keeps embeddings out of measurement authority;
- D10 names a migrated folder `<seq>-unknown` when no digest exists;
- E1 drops the separate image-level scope (a whole image is a 1 × 1 grid);
- E2's `measure()`/`embed()` split;
- E3a fails an image whose grid cell falls outside it, and makes grid edges user-set rather than tune-searched;
- E4 refuses object measurers in process mode;
- E5 fails finalization on embedding aggregation errors;
- figures stay in `figures/<date>-<sha12>/` in this change;
- C6 adds a preflight warning on a hardlink/symlink DVC cache;
- C10 defines four exit codes;
- C11 makes deliverables byte-stable;
- C12 adds `schema.json`.

**Left to the builder:**
- the exact TSV quoting configuration;
- the writer audit for C6;
- the C8 measurement;
- how staged and measure runs map to journal applications (D2);
- whether ngio discovers nested tables (D4);
- the DVC env-var check (C3, optional);
- today's exit code for a run with terminal failures (C10);
- liveness files restored by a checkout (C13);
- DVC's missing-out behaviour (C14);
- test placement;
- naming of private helpers.

## 8. Questions

### Resolved (user, 2026-11-12)

- **Q1 (P1) Downstream readers get a hard cutover.** Generated files are TSV-only
  from the release that ships P1. There is no dual CSV+TSV transition release.
  PhenoTypic's own readers still fall back to `.csv` in old run folders (B6); that
  is about reading old trees, not about external scripts. The release notes and
  the deliverables `README.md` state the rename.
- **Q2 (P2) Curation leaves the `deliverables` out modified.** The GUI keeps writing
  `deliverables/qc/` and the curated `measurements.parquet` in place after the
  stage. `dvc status` reporting `deliverables` as modified until the user runs
  `dvc commit` is the intended behaviour, not a defect. Curation state stays in
  `deliverables/`, so the bundle remains self-contained. The how-to page (C9)
  documents the `dvc commit` step after curating.

- **Q3 (P4) The grid is at the same pixels in every image of a rig**, and its
  coordinates are user-set parameters so other rigs can use their own (E3a).

## 9. Corrections to earlier artifacts and drafts

- **The feasibility page's top-level `embeddings/` group, and this document's
  first Part D, are superseded** by per-pipeline folders under `tables/` (Part D).
- **The process-mode revision bump *is* needed** (4 → 5, E4). An earlier draft said
  it wasn't, because adding an embedder changes the pipeline. That missed
  pipelines that already had measurers, which process mode used to ignore.
- **The image-level CLS + mean vectors are dropped** in favour of grid embeddings
  (E1).
- **The design keeps `meas` and adds an explicit `ImagePipeline.embed()`** (E2),
  where the feasibility page offered a `MeasureEmbedding` or a new `embed` slot.
- **`.phenotypic/` is now tracked**, a third out (user decision, §4.5, C1). The
  earlier two-out layout, which left it untracked, is superseded, and so is the
  earlier statement that `.phenotypic/` had to move out of the output folder.
- **Byte-stable deliverables moved from non-goal to goal** (C11), once the DVC
  purpose became automation.
