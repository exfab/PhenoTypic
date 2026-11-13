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

## Outside `testpaths`, not run by gates

- `tests/migration/test_metadata_schema_migration.py` writes legacy-named inputs at
  ~10 sites without `# legacy name, read path` markers.

## From P1 (TSV)

- `tests/unit/gui/test_curation_does_not_unfence_the_run.py::test_the_proof_still_records_every_published_artifact`
  pins the proof's `required_outputs` to the pre-TSV set; it must add `measurements_tsv`.
- Curation rewrites `measurements.csv`/`.parquet` but not `measurements.tsv` (spec B9).
  `measurements_tsv` is already in `_GUI_WRITTEN_PROOF_DESCRIPTORS`, so until #283
  nothing checks the TSV's bytes; #283 should make curation write the TSV.
