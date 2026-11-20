# Hand-off to #283 (GUI follow-up of #281)

Items found during #281's execution that live under `src/phenotypic/_gui/` or GUI
tests, which #281 may not touch. Appended at each gate. The README's expected-red
table is the authoritative list of red tests; this file collects everything else.

## From P0 (suffix flip)

- `src/phenotypic/_gui/CLAUDE.md:257` documents `PIPELINE_JSON # "pipeline.json.pht-pipe"`;
  the constant is now `pipeline.pht-pipe.json`.
- `tests/e2e/gui/` still writes `.json.pht-pipe` / `.json.pht-tune` names (outside
  `testpaths`; needs `PLAYWRIGHT=1`).
- `_gui/tune/_validation.py:61` accepts only `CONFIG_SUFFIX_TUNING`; it should accept
  `LEGACY_TYPED_CONFIG_SUFFIXES` too (fixes the six P0 expected-red tests).
- The GUI's `RecipeState` writer (`_gui/analysis/_recipe_state.py:909`) writes the new
  name beside an old one; `QcRecipe` (sdk_) now retires the old sibling, the GUI should
  match (spec A4).
- Builder directory browser and shell classifier list `x.pht-tune.json` as a pipeline
  (`_gui/builder/_directory_browser.py:32`, `_gui/shell/_classifier.py:222`).
- The GUI's tune surfaces look for the tuned winner under the canonical name only.
  `_gui/tune/_run_root.py:120,137,150` set `best_pipeline_path` from
  `best_pipeline_path(path)`, and `_gui/tune/_export.py:36,86` uses the same name, so
  a pre-flip tune folder that holds only `best_pipeline.json.pht-pipe` shows no
  winner. The sdk_ reader is `resolve_best_pipeline_path`; nothing calls it yet, on
  purpose (review M8).
- The GUI's tune writers do not retire the pre-flip twin.
  `_gui/tune/_export.py:36,51` (winner, per-axis Pareto), `_gui/tune/_winner.py:100`
  (curated winner) and `_gui/tune/_callbacks.py:2424` (tuning spec) write the new
  name beside any `.json.pht-*` file. The CLI's tune writers now publish, then unlink
  the twin (`old_typed_config_sibling`, `tune/_tune_cli/_run.py`
  `_publish_typed_config`; review M8). The GUI should do the same (spec A4).

## Outside `testpaths`, not run by gates

- `tests/migration/test_metadata_schema_migration.py` writes legacy-named inputs at
  ~10 sites without `# legacy name, read path` markers.

## From P1 (TSV)

- `tests/unit/gui/test_curation_does_not_unfence_the_run.py::test_the_proof_still_records_every_published_artifact`
  pins the proof's `required_outputs` to the pre-TSV set; it must add `measurements_tsv`.
- Curation rewrites `measurements.csv`/`.parquet` but not `measurements.tsv` (spec B9).
  `measurements_tsv` is already in `_GUI_WRITTEN_PROOF_DESCRIPTORS`, so until #283
  nothing checks the TSV's bytes; #283 should make curation write the TSV. Only
  then may a reader prefer the TSV: `measurements_table_path` /
  `BundleLayout.measurements_table_path()` exist but no reader calls them (spec B6).
- The run console's metadata check now refuses some tables it used to read.
  `_gui/run_console/_request_safety.py:403` calls `read_metadata_csv`, which raises
  `ValueError` for any suffix other than `.csv`/`.tsv` (spec B4), so a
  comma-separated `plates.txt` shows the metadata compatibility as blocked where it
  used to parse. A `.tsv` table is now accepted there. The message the user sees
  should name the allowed suffixes; the CLI's is
  "must be a .csv or .tsv table".
- `_gui/analysis/_callbacks.py:95` defaults `MergeMetadata`'s `metadata_path` to
  `"metadata.csv"`. A run given a `.tsv` table snapshots to `metadata.tsv` (spec B5),
  so the default names a file that run does not have. Resolve the default with
  `resolve_metadata_snapshot_path`.
- A tree holding both `metadata.csv` and `metadata.tsv` is refused by every CLI mode
  (exit 2, review M4) and by `resolve_metadata_snapshot_path` (`ValueError`). No GUI
  code calls that resolver today; one added by #283 (the `MergeMetadata` default
  above, say) should show the message rather than a traceback.
- Fixed outside the GUI, for reference: `post/_join_metadata.py` (`JoinMetadata`)
  matched its suffix case-sensitively and refused `plates.TSV`; it now goes through
  `delimiter_for` (review M9).

## From P2 (DVC run contract)

- The run console reads exit `4` (`RunExitCode.SUBMITTED`) as a failed SLURM
  submission (`_gui/run_console/_slurm.py:826`). Merge-gate item.
- A `.phenotypic/` restored by `dvc checkout` with a stale **active** lifecycle fence
  reads as `active` in `resolve_run_state` (sdk_ never asks the scheduler), so the GUI
  launch check treats the tree as running until the next CLI run, whose
  `retire_dead_lifecycle_fence` closes it (C10, `f62e895c6`). The GUI should either ask
  the scheduler (as `_refuse_live_slurm_run` does) or call the same retirement.

## From P3 (per-pipeline tables)

- A store's tables now live one folder per measuring pipeline,
  `tables/<n>-<sha12>/{measurements,metadata}/` (spec D1, P3 T2 `1a634df93`).
  `_gui/results_viewer/_measurement_source.py:107-108` stats
  `store / MEASUREMENT_TABLE_RELATIVE_PATH`, the v1 flat path, which no new
  store has. The colour-by-measurement picker and the measurement routes find
  no table. These are the README's P3 rows: `test_measurement_routes` (6 red at
  the C12 gate) and `test_measurement_source` (5 red). Replace the stat with
  `phenotypic.sdk_.latest_measurement_run(store)` and its
  `.measurements_path()` (spec D6a's default for a reader without a per-image
  record). `MEASUREMENT_TABLE_RELATIVE_PATH` stays defined as the v1 path, for
  migrate and for this code until it moves.
- `read_embedded_measurement_descriptor`, `embedded_measurement_columns` and
  `read_embedded_measurement_column` keep their names and signatures, and T4
  points them at `latest_measurement_run`. A GUI caller of those three needs
  no change once T4 lands. Between P3 T2 and T4 they raise `KeyError` on a v2
  store.
- The root-level `attributes.phenotypic.metadata_table` block moved into each
  run entry, `tables.runs.<run>.metadata_table` (spec D3). Any GUI code that
  reads the recorded metadata snapshot from the root must read it from the
  run instead. T4 moves `_run_state`'s reader.
- A store may hold several measuring runs. A GUI that shows "the" measurements
  of a store needs a run choice, or at least has to say which run it shows
  (the highest-numbered one with `measurements`, `<n>-unknown` when the
  pipeline digest is unknown).
- `tests/unit/gui/results_viewer/test_measurement_join_migration_run.py` (also a
  README P3 row) **skipped** at the C12 gate: the live migrated run it reads
  was absent or mid-republish. Its red/green state under v2 is unverified.
