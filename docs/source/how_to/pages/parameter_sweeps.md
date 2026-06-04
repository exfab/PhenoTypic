# Parameter Tuning

Systematically search pipeline parameters to find good settings using the
`phenotypic.tune` engine. (`phenotypic.tune` replaced the legacy
`phenotypic.sweep` module in a hard cutover.)

## Basic Usage

```bash
python -m phenotypic.tune tuning_spec.json -i /path/to/plates/ -o /path/to/output/
```

The engine loads the images under `-i`, runs the strategy in the
`tuning_spec.json` (grid or random), and writes the best pipeline and a
parameter-importance report under `<output>/deliverables/`
(`best_pipeline.json`, `tuning_spec.json`, `param_importance.json`) plus the
`trials.parquet` journal at the output-dir root. Re-running against an output
dir that already has a `trials.parquet` resumes rather than restarts.

A full tuning how-to (authoring a `tuning_spec.json`, scorers, search spaces,
and importance screening) is documented separately.
