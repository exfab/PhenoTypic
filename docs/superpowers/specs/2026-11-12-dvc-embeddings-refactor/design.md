# Design: DVC-compatible outputs, CSV → TSV, and DINOv2 embeddings

**Status:** approved · 2026-11-12 · revised 2026-11-12 (in place, call-order counter, plan-review fixes)
**Repos:** PhenoTypic `main` @ `3248a9173`, AutoConvertRaw-GC @ `33046cb`
**Background:** how measurements flow through the CLI today,
`docs/superpowers/artifacts/2026-11-12-dino-embeddings/measurement-data-flow.html`.

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
   - Nothing is silently overwritten or dropped by a later rewrite, and a re-run
     of the same pipeline replaces its own folder rather than duplicating rows.
   - A pipeline that detects nothing can add its folder to existing stores **in
     place**, so a run over AutoConvertRaw-GC's corrected stores skips color
     correction and denoising without copying pixels (E7).
2. **DVC compatibility.** PhenoTypic runs should work as DVC pipeline stages, and
   every generated CSV should become TSV.

**What DVC is for:** compatibility with automation and
data-engineering pipeline infrastructure. An orchestrator (CI, Airflow, Prefect,
Dagster, a cloud scheduler) drives PhenoTypic unattended, decides from its outputs
whether to continue, and feeds its tables to other systems.

**The shape:** a PhenoTypic CLI run is one DVC stage.
- Its pipeline file and tune spec are DVC *params*.
- Its `deliverables/`, `results/` and `.phenotypic/` are DVC *outs*.
- It emits a machine-readable run summary as DVC *metrics*, plus TSV tables for
  `dvc plots`.
- It runs in a **persistent** workspace: the same `--output` on the cluster
  every time.

DVC versioning of the raw image corpus is out of scope (§2.2).

## 2. Goals, non-goals, anti-goals

### 2.1 Goals

- **G1** `dvc exp run -S <pipeline file>:<key>=<value>` edits a PhenoTypic pipeline
  file and the result still loads in PhenoTypic.
- **G2** A `dvc.yaml` stage that runs `python -m phenotypic`, locally or on SLURM
  through the run waiter (C3), reproduces without DVC deleting resumable state,
  figures or embeddings.
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
- **G9** `--mode process` with `--output` equal to `--input` adds the pipeline's
  run folder to the input stores in place, without rewriting pixels, for
  pipelines that detect no objects (E7). Every other `--output` writes new
  stores, as today.

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
- **Any change under `src/phenotypic/_gui/`.** GUI work is deferred to issue #283.
  Every name the GUI imports from `phenotypic.sdk_` stays defined, and every
  `sdk_` function the GUI calls keeps its name and signature. Beyond that, GUI
  breakage is accepted until #283, and the GUI tests it turns red **stay red**;
  the plan lists them by name. The known breaks:
  - SLURM submission exits `4`, which the run console reads as a failed
    submission (`_gui/run_console/_slurm.py:826`);
  - the results viewer's Colony picker stats the v1 table path, so it finds no
    table in a v2 store (`_gui/results_viewer/_measurement_source.py:107`);
  - the tune Deploy validator refuses a legacy `.json.pht-tune` spec
    (`_gui/tune/_validation.py:61`);
  - the builder's directory browser and the shell classifier list a new
    `x.pht-tune.json` as a pipeline, because plain `.json` is still a legacy
    pipeline suffix (`_gui/builder/_directory_browser.py:32`,
    `_gui/shell/_classifier.py:222`);
  - curation keeps rewriting `measurements.csv`, so curated edits reach the
    mirror's `.parquet` and `.csv` but not its `.tsv` (B9);
  - the error-analysis deliverable stays `error_analysis.csv`, because GUI code
    writes it (B9).

  **#281 does not merge to `main` before #283 fixes these.**
- In-place runs of pipelines that detect objects (E7). A store holds one label
  image; storing one per run is a follow-up.
- An in-place run as a DVC stage. Its input tree would also be its out.
- Byte-reproducible per-image stores from full runs. Their journals carry wall-clock
  times. Deliverables *are* in scope (C11).

### 2.3 Anti-goals (failure even if everything "works")

- A store rewrite that drops another pipeline's run folder or a carried
  embedding entry.
- A re-measure that renumbers or duplicates a pipeline's folder, or leaves rows
  from its previous write beside the new ones.
- Two run folders in one store with the same number.
- An in-place run that changes a store's pixels, or deletes or overwrites an
  input.
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
| Every SLURM run returns at submission, and its dependent finalizer is the sole publisher. 0.21.0 removed `--wait` (2026-11-05, commit `281430b2e`). | `_cli/CLAUDE.md` "SLURM chaining", `phenotypicCLI.py` main flow |
| A journal application's `sequence` is `len(applications) + 1`, local to one image's journal. A forward run builds the journal from its input, so over a raw input it is always `1`. `--mode measure` appends no application. | `_core/_provenance.py:374`, `_cli_process_single.py:300-321`, `phenotypicCLI.py:3526-3528` |
| Process mode refuses input and output roots that overlap, **equality included**. `--overwrite`/`--restart` refuse a run input they would delete. | `phenotypicCLI.py:2289-2306`, `:636-670` |
| Operations that change `rgb`/`gray` are `ImageCorrector` subclasses (`CalibrateColorRpcc`, `ColorDenoise`, `DenoiseBlockMatch`, `CropImage`, the shrink correctors, `GridCorrector`s). Enhancers change only `detect_mat`. | `abc_/_image_corrector.py`, `correction/` |
| GUI code writes the CLI's error-analysis deliverable (`publish_error_analysis`, `combined.write_csv`) and rewrites `BundleLayout.mirror_csv` on every curation save. | `_cli_error_outputs.py:81-96`, `_gui/results_viewer/_error_tab/_publication.py:588`, `_curation_labels.py:839-850` |
| A per-image record fingerprints its store by the root `zarr.json`, so any root rewrite stales it unless the record is republished (measure mode's `_republish_table_marker`). | `sdk_/_run_state.py:517-539`, `_cli_recompile_tables.py:51`, called at `_cli_process_single.py:567-581` |

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

## 4. Design

The five parts below are independent enough to plan and ship in the order of §5.

### Part A: typed config suffix flip

- **A1** New canonical suffixes: `.pht-pipe.json`, `.pht-op.json`, `.pht-cc.json`,
  `.pht-tune.json`. The old `.json.pht-*` forms become **legacy read** suffixes.
  Plain `.json` stays legacy-readable where it already is (pipelines, tune specs).
- **A2** `ensure_typed_json_suffix(path, suffix)`:
  - A bare stem gets the new suffix.
  - `x.json` becomes `x.<typed>.json` (insert before `.json`). Today it becomes
    `x.json.<typed>`.
  - A path that already ends in a recognised suffix, old or new, is returned as
    given, so a file is never written somewhere other than where the caller
    asked.
- **A3** Canonical run artifacts use the new name:
  `deliverables/pipeline.pht-pipe.json`, `deliverables/best_pipeline.pht-pipe.json`,
  and `.phenotypic/pipeline.pht-pipe.json`.
  - Every non-GUI reader (`resolve_*`, `BundleLayout`, migrate's
    `*.pht-pipe` glob at `_cli_migrate.py:679`) also accepts the old names.
  - `migration_lock_path` maps legacy `pipeline.json`, the old name and the new
    name to the **same** lock identity, extending the existing rule at
    `_io_constants.py:248-259`.
- **A4** One current pipeline file per folder:
  - `_seed_pipeline_config` seeds only when no config exists under any of the
    three names.
  - When finalize rewrites the pipeline file, it publishes the new name first and
    then removes an old-name sibling.
  - A plain `pipeline.json` is never removed. It may be the user's own file or
    a V1 run's, and the resolvers prefer the typed names over it.
- **A5** AutoConvertRaw-GC:
  - rename `pipelines/exfab_rig1.json.pht-pipe` to `pipelines/exfab_rig1.pht-pipe.json`;
  - set `PIPELINE_NAME = "pipeline.pht-pipe.json"`;
  - update the three configs and `pipeline_info.py`'s docstring.

  The bytes are unchanged, so `pipeline_sha256` is unchanged. A test asserts that.
  Stores written after the rename record the new basename in their provenance,
  so they are not byte-identical to stores written before it. The P5 rollout
  changes those stores anyway.
- **A6** DVC note for the how-to page (C9): DVC addresses params by dotted
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
  - migrate's `metadata.canonical.tsv`

  Add the new constants and path functions (`MEASUREMENTS_TSV`,
  `measurements_tsv_path`, …) **beside** the old ones. The old names stay
  defined as legacy read names (B8).
- **B3 Analysis manifest.** Fields `csv`/`csv_sha256` become `tsv`/`tsv_sha256`,
  with a manifest `schema_version` bump. Readers accept the old shape.
- **B4 User-supplied inputs accept both** `.csv` and `.tsv`, delimiter chosen by
  suffix, as `join_metadata` already does. That covers `--metadata`, reference
  metadata and tune ground truth. Extend this to `_metadata_migration.py:469`,
  `_cli_reference.py:400` and `metadata_csv_payload`. Any other suffix is
  refused, in every mode, as a usage error (exit 2). For process mode, where
  `--metadata` used to be ignored with a warning, this refusal is new.
  - A user `.tsv` is read with the same `"` quoting the generated TSV uses
    (B1), which is what Excel, R and pandas write.
- **B5 Snapshots keep the user's bytes and extension:**
  `deliverables/metadata.<ext>` and `.phenotypic/reference_metadata.<ext>`, where
  `<ext>` is the input's extension, lower-cased (`plates.TSV` snapshots to
  `metadata.tsv`). Readers locate the snapshot under either extension. Existing
  trees keep `metadata.csv`.
  - Exactly one snapshot exists. If both suffixes exist, any run that would
    read the snapshot refuses with a usage error before writing anything. A
    full or recompile run given `--metadata` replaces the snapshot instead,
    which is how the pair is resolved; measure and migrate always refuse.
  - A `--metadata` whose suffix differs from the snapshot's **replaces** it, as
    a same-suffix edit always has. The new snapshot is published first and the
    other suffix's file removed after, so a crash leaves two files (a loud
    refusal), never none (a silent run without the metadata join).
- **B6 Old run folders.** Readers (results viewer, analysis app, `BundleLayout`,
  the README generator) fall back to the `.csv` name when no `.tsv` exists. A
  finalize into an old folder publishes the TSV, then removes the stale
  **generated** CSV sibling. This applies only to files in the B2 list, never to
  the metadata snapshot, `error_analysis.csv`, or the `measurements.csv` mirror
  that B9 keeps. A generated CSV whose producer has left the pipeline is not
  republished, so it stays.
  - No CLI reader prefers `measurements.tsv` over `measurements.csv` until #283:
    curation updates only the `.csv` and `.parquet` (B9), so a TSV-first reader
    would read uncurated data.
- **B7 Public API is not renamed.** A Python method or argument named `csv` keeps
  its name and behaviour. Docstring examples that write `.csv` (e.g.
  `_objects_accessor.py:690`) stay user-chosen. B applies to files the CLI
  generates; GUI exports follow in #283.
- **B8** Update the README generator, the how-to pages, and the root and `_cli`
  CLAUDE.md output inventories. Every name the GUI imports from `sdk_` stays
  defined with its signature: `MEASUREMENTS_CSV` (re-exported by
  `_gui/_config.py`), `MEASUREMENT_TABLE_RELATIVE_PATH`,
  `measurements_csv_path`, and `metadata_csv_deliverable_path(output_dir)` with
  one argument. A test walks `src/phenotypic/_gui/**` for names imported from
  `phenotypic.sdk_` and asserts each still resolves, with an unchanged signature
  for each function the GUI calls.
- **B9 What stays CSV until #283.** Both are written by GUI code (§3.2):
  - `error_analysis.csv`;
  - the curated mirror. `BundleLayout.mirror_csv` keeps pointing at
    `measurements.csv`, the GUI's write target, and a new `mirror_tsv` names the
    TSV. Finalize writes both the `.tsv` and the `.csv` mirror, so the GUI keeps
    a current file to curate. `measurements_tsv` joins
    `_GUI_WRITTEN_PROOF_DESCRIPTORS` with `measurements_csv`, so a curation does
    not invalidate the run proof.

### Part C: DVC-compatible run layout

- **C1 Outs.** The supported DVC shape is three outs, all `persist: true`

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
- **C3 SLURM: a run waiter that never publishes.** A SLURM run returns at
  submission, but DVC hashes a stage's outs the moment its command exits, and
  DVC aborts a stage whose command exits non-zero, which every submission does
  (exit 4, C10; verified on DVC 3.67.1 in P2 T6). So a SLURM-backed stage is one
  command: the waiter launches the submission itself, then waits for it.

  ```yaml
  cmd: phenotypic-wait --output <out> -- python -m phenotypic … -o <out> --slurm …
  ```

  `phenotypic-wait` is a new console script (`phenotypic._cli._cli_wait`). It
  blocks until the run is finished, then exits with the C10 code that the run
  would have returned locally.

  **Launch (`--` form).** Everything after `--` is the submit command, run as a
  child process (an argv list, no shell) with the waiter's working directory,
  environment and terminal:
  - child exit `4` (submitted): the waiter polls, as below;
  - child exit `0` or `3` (a local run that already finished): returned as-is,
    with no poll;
  - any other child exit (`1`, `2`, …) is returned as-is and nothing is polled;
    a child killed by signal N returns `128 + N`, the shell's convention;
  - refused before launching, as a usage error (`2`): an empty command after
    `--`; a child that names an output (`-o`/`--output`) resolving to a
    different path from `--output`; a child running `--mode migrate`.
    A child that names no output (a wrapper script) is not checked. Only the
    tokens after the program `phenotypic` (or `-m phenotypic`) are read, so a
    wrapper's own `-o` (`srun -o log …`) is not taken as the run's output.
  - `--timeout-seconds` bounds the wait, never the submit.
  - **Attach to a live run (user decision 2026-11-13, P2 review F2).** Before
    launching, the waiter checks `live_slurm_job_ids(output_dir)`. If a job of
    that output is live, it skips the submit, prints that it is attaching to
    the live generation, and waits on it exactly as after a `4`. That is the
    recovery after the job running `dvc repro` hits its walltime: re-run it. The
    waiter cannot verify that the live run used the same pipeline; only one
    live run per output can exist, and the message names the generation.
  - **`--accept-failures` (user decision 2026-11-13, F1).** Maps a final `3`,
    from the wait or from a local child, to `0` after printing the
    terminal-failure count, so a DVC stage over a dataset with one empty plate
    completes; `metrics.json`'s `images.failed` still records the count. Off by
    default: the exit-code contract is unchanged.

  Without `--`, the waiter only waits, for a run submitted some other way. That
  form is unchanged and is what a user reaches for after a submit they typed
  by hand.

  **Waiting:**
  - It polls `resolve_run_state(output_dir, depth="shallow")` together with
    `live_slurm_job_ids(output_dir)` (`_cli/_cli_staged_orchestration.py`) on an
    interval. The default is 60 s, set with `--poll-seconds`.
  - Both checks are needed. `resolve_run_state` takes an `active` lifecycle fence
    at face value and never asks the scheduler, so a finalizer killed for running
    out of memory would read as `active` forever. The scheduler query is what
    tells a running run from a dead one. That query lives in `_cli`, which
    `sdk_` may not import, so the waiter lives in `_cli`.
  - **Any live job keeps it waiting, whatever the run state reads.** A measure or
    same-metadata recompile resubmission leaves the previous run's proof
    current, so `complete` alone does not mean this submission finished. A
    scheduler that can't be queried (`unverified`/`unresolved` jobs) also keeps
    it waiting, with the reason printed each poll. It never guesses.
  - **With no job live, it accepts only a publication by this submission.** At
    its first poll it reads the lifecycle fence's generation. Published means one
    of: that generation's fence closed **by its finalizer**; a run proof whose
    `execution_epoch` is that generation; or an aggregate proof stamped with that
    generation. Then it exits `0`, or the complete-with-failures code when
    images failed terminally **and every accepted image is resolved** (a
    success or a terminal failure record). An image never attempted (an
    array task killed before it wrote a record) makes the run unfinished, not
    complete-with-failures. Anything else, including that, is a run that ended
    without finishing, and gets `5` (F5: the meaning of `5` is "ended without
    finishing; re-run the same command", which covers both "published nothing"
    and "published a partial aggregate").
  - A `--mode measure` resubmission's failures are keyed by measure work ids,
    outside the forward run's inventory, so `resolve_run_state` cannot see
    them. Measure workers record a measure failure as a terminal-journal row
    (mode `measure`, `lifecycle_epoch` = the generation), which they did not do
    before #281; the waiter counts the distinct work ids journalled under its
    bound generation, so a SLURM measure with a failure exits `3` as the local
    one does (P2 review I5). `metrics.json` counts the same rows on SLURM, and
    the invocation's own failure count locally, so it agrees with the exit
    code. These rows are keyed by measure work ids, so they never affect a
    forward run's continuation.
  - Two fields make that binding exact, with no clock comparison:
    - the finalizer stamps the generation it ran under into the aggregate proof
      (`execution_epoch`, as the run proof already has). A proof without the
      field, written before this change, never counts for a new submission;
    - the finalizer's close of the fence records `closed_by: "finalizer"`. A
      cancel (`cancel_generation`, `deactivate_orchestration`) leaves it unset,
      so a cancelled resubmission ends unpublished.

    Matching state alone is not enough: a `--mode measure` resubmission keeps no
    run state, so an earlier run's aggregate proof would still match after the
    new finalizer died.
  - Process mode publishes no aggregate proof, so a process run whose images
    are all resolved, with some terminal failures, and no live job exits
    complete-with-failures. A cancelled process run (fence closed without
    `closed_by`) exits `5`.
  - It does not serve `--mode migrate`, which keeps its lifecycle under an
    external control root.
  - It never aggregates, publishes or cancels, so the double publication the
    old `--wait` had cannot recur. The submission it may launch is the ordinary
    CLI, whose SLURM path publishes nothing (the finalizer is the sole
    publisher), so launching it changes nothing about who writes. The waiter's
    own only write is the verification cache `resolve_run_state` already
    refreshes.
  - In the `--` form its first poll comes after the submit returns, so the
    generation it binds to is the one that submission opened.
  - `--timeout-seconds` (default: none) gives up with its own exit code, without
    cancelling anything.
  - The DVC stage itself runs inside a small, long-walltime Slurm job (1 CPU),
    because head nodes kill long-running processes. The how-to page (C9) gives
    the `sbatch` line.
- **C4 Metrics.** New `deliverables/metrics.json`, written by `finalize_run` (the
  one FINAL writer):
  - **Content:** `schema_version`, `phenotypic_version`, `pipeline_sha256`;
    `images.{accepted, succeeded, failed}`; `objects.measured`;
    `metadata.phantoms`; and, when embeddings ran, per entry id `{rows, stale_skipped}`.
  - **Derivation:** values come from the run state (`resolve_run_state`) and the
    published frames, never from a new tracked value. This is a deliverable that
    PhenoTypic never reads back, so it is not a fifth tracked state.
  - `pipeline_sha256` is the sha every aggregated run folder shares, and `null`
    when they differ. It is not the run config's, which still names the full
    run's pipeline after `--mode measure` (D6a).
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
  `.dvc/config` selects `hardlink` or `symlink`. The check reads one
  config file and never imports DVC.
- **C7 Post-run curation.** GUI curation changes files inside the `deliverables`
  out, so `dvc status` reports the out as modified until the user runs
  `dvc commit`. This is intended: no code change, and the
  how-to page documents the commit step.
- **C8 Results size.** Whether `results/` should be `cache: true` or `false` is
  decided by measurement, not reasoning. Time `dvc status` and `dvc commit` on a
  bounded subset of the migrated maresca copy (`2026-08-11-ome-zarr`, 1,354 GB in
  full): one dataset, `hdf/` excluded, in a scratch DVC repo on `/bigdata`. Run
  it as a Slurm job with an explicit `--time`, one config per fresh node,
  extrapolate to the full tree, and record the numbers in the how-to page.
- **C9 How-to page** `docs/source/how_to/pages/dvc.md`. Contents:
  - a local `dvc.yaml`, a SLURM one (one `phenotypic-wait … -- <submit>`
    command, C3), and a process-mode one, since
    process mode creates no `deliverables/` and its outs list must say so;
  - the three outs with `persist`, and the cache-type rule (C6);
  - `dvc gc` for the cache growth that tracking `.phenotypic/` causes;
  - params addressing with an example `dvc exp run -S`;
  - top-level `metrics:`/`plots:`;
  - the one-line-JSON caveat (§3.1) and the `dvc commit` step after curation;
  - that an in-place run (E7) is not a DVC stage: a DVC stage writes new stores.
- **C10 Exit codes are a contract**. Four distinct codes:
  - **0**: complete, no terminal failures;
  - a distinct non-zero: **complete with terminal image failures**: every
    accepted image is resolved and at least one failed terminally. This holds
    for every mode, local and SLURM, recompile and staged GPU included (user
    decision 2026-11-13, P2 review F3/F4): a recompile over a run that carries
    failures exits `3`, and a staged-GPU run whose only unfinished images are
    terminal failures exits `3`, not `1`;
  - **1**: a runtime or config error (today's behaviour). A refusal raised as a
    Click usage error keeps Click's `2`;
  - a distinct code: **submitted to SLURM**, so nothing is finished yet. Every
    SLURM run returns this at submission. `phenotypic-wait` (C3) returns one of the
    codes above once the run ends, plus its own codes for a run that ended
    without finishing (`5`) and for `--timeout-seconds` (`6`); in the `--` form, a
    submit that did not exit `4` returns the submit's own code.

  The builder confirms what a run with terminal failures returns today; I found
  no distinct code for it. The values are documented and pinned by tests.
- **C11 Byte-stable deliverables**. For identical inputs, pipeline and
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
- **C12 `deliverables/schema.json`**. Every column of each published
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

An embedding belongs to the pipeline that produced it, exactly as that pipeline's
scalar measurements do, so both live in the pipeline's own folder under `tables/`.

- **D1 Layout.** `tables/` stays the top-level group, since it is already part of
  the store schema. Below it there is one folder per pipeline application that
  wrote measurements into this store:

  ```
  tables/
  ├── <n>-<sha12>/                         one folder per measuring pipeline
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
- **D2 Folder name `<n>-<sha12>`.**
  - `<n>` is a **store-local call-order counter**. A new folder gets 1 + the
    highest number already used in this store, counting both its run folders and
    its `upstream` entries (D8). So a result store whose input carries
    AutoConvertRaw-GC's `1-<sha>` names its own folder `2-<sha>`.
  - Numbers are never reused or renumbered, so two folders in one store never
    share a number, and "the latest run" never ties.
  - The journal's `sequence` is not used: it is local to one journal, and a
    forward run over a raw input is always `1` (§3.2). The journal still logs
    every application.
  - `<sha12>` is the first 12 hex of the pipeline sha256 the journal records,
    which is the pipeline file's bytes. It is the same digest the figure run
    folders and AutoConvertRaw-GC use.
  - When no pipeline digest is known (a programmatic `save2zarr`, an empty
    journal, or a converted `legacy` application), the folder is `<n>-unknown`
    with `pipeline_sha256: null`. Never fabricate a digest.
- **D3 Descriptor.** `attributes.phenotypic.tables` moves to `schema_version: 2`:

  ```json
  { "schema_version": 2,
    "runs": {
      "2-a71e09c2b4f5": {
        "order": 2, "pipeline_sha256": "<64 hex>",
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
  - A write that produces no tables of its own (the initial-checkpoint saves,
    the Stage 1 save) carries **every** folder, its own pipeline's included. It
    excludes its own folder only when it writes that folder's tables, the rule
    figures already follow (`_image_io_handler.py:1468-1470`).
  - Only `--overwrite` deletes run folders.
  - Forward modes refuse a store still in the v1 flat layout and point at
    `--mode migrate`, as they already refuse unconverted trees. The check is
    CLI-only, in the read-only half, and stops at the first v1 store. It is not
    added to `datasets_needing_migration`, which every `resolve_run_state` call
    (GUI and `phenotypic-wait` polls included) runs.
- **D6 Which folder a writer owns, and how re-runs avoid duplicates.**
  - A writer looks for an existing run folder with its pipeline's sha256. If one
    exists, it **replaces** that folder, keeping its number. The folder's files
    are rewritten as new inodes, and no row from the previous write survives. So
    re-running a pipeline, in any mode, never duplicates measurements.
  - Otherwise it creates a new folder with the next number (D2). Every other
    folder is carried untouched (D5).
  - A write with an unknown sha replaces the highest-numbered `-unknown` folder,
    so notebook re-saves don't pile up folders.
  - So `--mode measure` over a full run's stores replaces the full run's folder
    when the pipeline is the same, and adds a folder beside it when the pipeline
    differs. This matches how measure mode already reuses a figure run folder for
    the same pipeline.
- **D6a Which folder a reader uses.**
  - Finalize, recompile and measure aggregate, per image, the folder that
    image's record certifies: the completion marker already names the table it
    hashes (`marker_claims_measurement_authority`). They never select by "the
    run's pipeline sha", which goes wrong after `--mode measure`: measure mode
    keeps no run state, so the run config still names the full run's pipeline.
  - A reader without a record (the `sdk_` readers, notebooks) defaults to the
    highest-numbered folder that holds `measurements`. A grid-only process
    folder has none.
- **D7 Stale binding** is recomputed on every write, never latched:
  - **Crops:** compare the entry's `objmap_sha256` with the objmap being written.
  - **Grid:** compare `array_sha256` with the embedded level-0 array.
  - Set `stale = (they differ)`, and keep the files either way.
  - Readers and aggregation refuse a stale entry.
  - Recomputing is what lets a staged run work: Stage 1 writes an all-zeros
    objmap, and Stage 3's identical objmap clears the flag again.
  - Digests are computed from in-memory arrays at write time.
  - The measure rewrite writes no objmap and leaves `stale` unchanged.
- **D8 Upstream reference.**
  - When an input image is a PhenoTypic store with a v2 `tables` descriptor, the
    result store's `upstream` records each of the input's run folders, along with
    any entries in the input's own `upstream`, so the chain is complete.
  - Nothing is copied.
  - A reader that follows a reference compares `root_sha256` and reports
    "upstream changed" on a mismatch, for example after an AutoConvertRaw-GC
    reprocess.
- **D9 Integrity.**
  - File `sha256`s live in the root, so the completion marker covers them.
  - Missing or corrupted embedding bytes make that entry unreadable. They do not
    revoke the run's measurement authority: embeddings are additive.
- **D10 Migration.** `--mode migrate` converts a v1 store's `tables/measurements`
  and `tables/metadata` into `tables/1-<sha12>/`.
  - The sha is the most recent figure run's sha when present, since a v1 table
    rewritten by `--mode measure` was never journaled. Otherwise it is the
    latest journal application that measured. With neither, the folder is
    `1-unknown` (D2).
  - `replace_embedded_measurement_table`, which migrate uses to import legacy
    per-image Parquets, writes v2 directly, so no tree is left in a mixed layout.
    Every glob of the flat path (`_cli_migrate.py:1057-1061`) goes through the
    resolver.
  - The conversion rewrites each root, so it republishes each converted image's
    record and then the run proof, as measure mode does. A tree that read
    `complete` before migrate reads `complete` after it.

### Part E: producers, deliverables, AutoConvertRaw-GC

- **E1 `MeasureEmbedding` ABC** (`abc_/`), a `MeasureFeatures` subclass with
  `regions: Literal["objects", "grid"]`:
  - `objects` requires a non-empty objmap and writes `crops/`.
  - `grid` requires a grid definition, not an objmap, and writes `grid/`.

  A whole-image embedding is a 1 × 1 grid, so there is no separate image scope.
  `measure(image)` returns the table's frame directly, for notebook use.
- **E2 Pipeline routing.**
  - `ImagePipeline.measure()` keeps its return contract and does not run
    `MeasureEmbedding` entries.
  - A new `ImagePipeline.embed(image) -> dict["<model>/<crops|grid>", EmbeddingPayload]`
    runs them.
  - The CLI per-image writers call both, then write one run folder (D1). These are
    the single-pass worker, Stage 3, measure mode and process mode.
  - Both docstrings say so.
- **E3 `MeasureDinoEmbedding`** (`measure/_measure_dino_embedding.py`). Fields:

  | Field | Type, default | Notes |
  |---|---|---|
  | `size` | `DinoSize = "base"` | first field; `small`, `base` or `large` |
  | `version` | `DinoVersion = 2` | the DINO version; the class name already says DINO |
  | `revision` | `None` (a `str` when pinned) | the weights' Hugging Face commit (any ref the hub accepts). `None` loads the latest. The pin is in the pipeline file, so changing it changes the pipeline hash. Either way the entry records the commit actually loaded (D3) |
  | `regions` | `"objects"` | |
  | `grid` | `ManualGridFinder` (an `OperationField`) | required when `regions="grid"`, refused otherwise; the rig's cell edges in pixels (E3a) |
  | `pooling` | `"crop_cls"` | `crop_cls`, `crop_mean` or `masked_dense`; a grid cell is cropped from its rectangle exactly as an object is from its bbox |
  | `crop_px` | `224` | validated as a multiple of the loaded patch size |
  | `crop_pad_frac` | `0.25` | objects only |
  | `mask_background` | `False` | objects only |
  | `dtype` | `"float16"` | |
  | `batch_size` | `64` | |
  | `device` | `"auto"` | |

- **E3a Grid coordinates are per-rig parameters.** A rig puts
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
    cell and the image size, rather than being clipped.
  - For `phenotypic-tune`, the edges are `TuneSpec(tunable=False)` structural
    plate geometry, as `ManualGridFinder.nrows/ncols` already are. They are
    user-set, not searched.
  - No coordinates are shipped as defaults. The rig1 edges are supplied by the
    user when the AutoConvertRaw-GC pipeline is updated (E6).

  Implementation notes:
  - The model token is `dinov{version}-vit{s|b|l}{patch}`, built from `version`
    and `size`, e.g. `dinov2-vitb14`. The fields drop the `dino_` prefix because
    the class name already says DINO.
  - Reuse the `_dino_support.py` helpers.
  - Keep a module-level backbone cache keyed by `(hf_id, revision, device)`.
  - Run under `torch.inference_mode()` with `model.eval()`.
  - Import heavy modules lazily (startup-import guards).
  - DINOv3 goes through `PHENOTYPIC_ACCEPT_MODEL_LICENSE`.
  - Declare `preflight_requirements()` for `torch`/`transformers`.
  - Tune annotations go on numeric fields, per the `adding-an-operation` skill.
  - Add a `DINO_EMBEDDING` schema enum, authoring `label`/`desc` only (no
    `bio_desc`), exported from `phenotypic.schema`.
- **E4 Process mode runs the pipeline's measurements.**
  - `MODE_SLOTS["process"]` becomes `ops` + `meas`, and the preflight's mode walk
    follows. `post`/`filters`/`model` still don't run, and no deliverables are
    built.
  - `process_single_apply_only_core` runs `measure()` and `embed()` after
    `apply()`, and writes the run folder into the `.part` before consolidation.
    The store stays write-once.
  - A process store holds no objmap (`write_objmap=False`). So a measurer that
    needs objects is refused in the read-only half, with one usage-error line
    naming it, rather than writing a table keyed on labels the store can't
    resolve.
  - `meas` is refused unless the output is a store (`--layer rgb`/`gray` with
    zarr format). A flat TIFF or PNG export has nowhere to put a table.
  - The same two checks also run in `process_single_apply_only_core`, because
    AutoConvertRaw-GC calls it directly and never passes the CLI's refusal.
  - **Bump `PROCESS_LAYER_SEMANTICS_REVISION` 4 → 5.** Process mode used to ignore
    `meas`, so a process tree whose pipeline already had measurers would otherwise
    resume onto stores that lack the tables it now writes.
- **E5 Deliverables.** The finalizer aggregates **its own** run folder, chosen
  per image by D6a:
  - the master comes from `measurements/`, exactly as today;
  - each embedding table goes to
    `deliverables/embeddings/<model>/<crops|grid>.parquet`, with the master's
    intrinsic identity columns plus the table's key and `embedding`;
  - `deliverables/embeddings/manifest.json` holds each entry's descriptor, row
    count, sha256 and the stores skipped as stale.

  Rules:
  - These files are Parquet only. They are never joined to metadata, and never in
    master, mirror, splits or any TSV.
  - A failure fails finalization, like the master.
  - Upstream tables (D8) are not aggregated into a downstream run's deliverables
    in v1.
  - A `MeasureEmbedding` gets no by-feature split and no column section in the
    deliverables README, which document mirror columns. The README documents
    the embedding deliverables separately. The vector column is named
    `embedding`, and `schema.json` lists it under each embedding table (C12).
- **E6 AutoConvertRaw-GC rollout** (after the new wheel is built):
  - add `MeasureDinoEmbedding(size="base", revision=<pinned commit>, regions="grid",
    grid=ManualGridFinder(...))` to the pipeline's `meas`, with rig1's edges
    supplied by the user;
  - vendor the new wheel;
  - install the `foundation` extra in the `Dockerfile` with CPU torch, bake in the
    pinned weights, and set `HF_HUB_OFFLINE=1`;
  - make `verify_store` check that the run folder's grid entry exists (an
    `expected_embeddings` field in `run.json`, written by `plan.py`), the way
    figures are checked;
  - re-pilot memory and timeout on the smoke configs, never on `data/`.

  Backfill of done images is an explicit `--reprocess-changed` decision at
  rollout. It reruns calibration and BM3D denoise.
- **E7 In place.** `--mode process` with `--output` resolving to the same
  directory as `--input` adds the pipeline's run folder to each input store
  where it is. Any other `--output` writes new stores, as today. An output
  strictly inside an input, or an input strictly inside the output, stays
  refused (§3.2).
  - **A store update that never rewrites pixels.** Hard-link the store into a
    `.part`, write the run folder (D6) and the run's figure folder, append the
    process application to the journal, write the root last, consolidate, and
    swap, reusing the measure rewrite's transaction. Pixel arrays are hard links
    and are never rewritten.
  - Before the swap, the writer checks that the live root still has the digest
    it started from. If it changed, the writer rebuilds once from the new root,
    and otherwise fails the image. The output's existing single-run guard
    already keeps a second run off the same tree.
  - `.phenotypic/` lives under the input root, as under any `--output`.
  - **Refused in the read-only half**, one usage-error line each, before
    anything is written:
    - `--overwrite` and `--restart`, which would delete inputs;
    - `--process-format tiff`, and any layer that writes no store;
    - an input that is not a PhenoTypic store (a raw image, a third-party
      OME-Zarr), which has nothing to write into;
    - a `--layer` that is not the store's primary series;
    - an `ObjectDetector` anywhere in the pipeline. A store holds one label
      image, so a second detecting pipeline would leave the first one's table
      pointing at labels it was not computed from;
    - an `ImageCorrector` anywhere in `ops`, which changes the pixels the store
      holds.
  - **Checked per image.** After `apply()`, the exported layer's level-0 digest
    must equal the store's. Otherwise that image fails without writing. This
    catches any pixel-changing op the class check misses.
  - **Continuation.** Re-running the same command does no work for an image
    whose store already holds this pipeline's folder with every entry
    `stale: false`. The run's own write changes the store's root, so the work id
    must not include the bytes the run writes.
  - **Where to run it.** An AutoConvertRaw-GC reprocess deletes and re-uploads a
    store, dropping every folder added in place. So in place runs on a local
    copy, such as the `/bigdata` mirror, never on the bucket or the NAS tree
    AutoConvertRaw-GC publishes. The how-to page says so.

## 5. Order

| Phase | Content | Depends on |
|---|---|---|
| P0 | Part A suffix flip, PhenoTypic and AutoConvertRaw-GC | none |
| P1 | Part B TSV | none (parallel with P0) |
| P2 | Part C DVC layout, `metrics.json`, how-to page, C8 measurement | P0, P1 (the template names their files) |
| P3 | Part D: `tables/` v2 per-pipeline layout, carry rule, upstream reference, ngio attributes, `--mode migrate` conversion, reader updates (finalizer, measure, recompile, marker, and the `sdk_` readers the GUI calls) | none (parallel with P0–P2) |
| P4 | Part E1–E3a, E5: `MeasureEmbedding`, `MeasureDinoEmbedding` (crops + grid), `embed()`, deliverables | P3 |
| P5 | Part E4, E6, E7: process-mode measurements, in place, AutoConvertRaw-GC rollout | P3, P4, P0 |
| Later | GPU execution (Stage-2 dense or Stage 4) | P4 timings |

P3 is now the largest phase. Every reader of `tables/measurements/table.parquet`
changes: the finalizer, measure mode, recompile, migrate, the `sdk_` readers and the completion
marker. Derive its affected test surface from importers, not from directory names
(root CLAUDE.md).

Testing follows root CLAUDE.md: step guards, then per-task files, the affected
surface once per phase, and the full sharded regression once at the end, on Slurm.
Store changes also need an **independent reader** check (zarr-python + pyarrow,
never only PhenoTypic's own reopen), per the `working-with-ome-zarr` skill.

## 6. Left to the implementer

- the exact TSV quoting configuration;
- the writer audit for C6;
- the C8 measurement;
- whether ngio discovers nested tables (D4);
- how the in-place work id leaves out the bytes the run writes (E7);
- the exact numeric values of the C10 codes, chosen to avoid the codes Click and Python already use (1, 2);
- today's exit code for a run with terminal failures (C10);
- liveness files restored by a checkout (C13);
- DVC's missing-out behaviour (C14);
- test placement;
- naming of private helpers.
