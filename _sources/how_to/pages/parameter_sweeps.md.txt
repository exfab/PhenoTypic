# Parameter Sweeps

Systematically vary pipeline parameters to find optimal settings using
the `phenotypic.sweep` module.

## Basic Usage

```bash
python -m phenotypic.sweep sweep_config.yaml /path/to/plates/ /path/to/output/
```

## Sweep Configuration

Define parameter ranges in a YAML file:

```yaml
pipeline: pipeline.json
sweep:
  GaussianBlur.sigma: [1.0, 2.0, 3.0, 5.0]
  CLAHE.clip_limit: [0.005, 0.01, 0.02, 0.05]
```

Each combination is tested, and results are saved with the parameter values
in the output path.

## Analyzing Results

Sweep outputs include measurement CSVs for each parameter combination.
Compare detection counts, colony sizes, or other metrics to identify
the best settings for your imaging conditions.

For full sweep configuration options, see the
[sweep module documentation](../../api_reference/index.rst).
