# Edge Effects in Plate-Based Assays

Colonies on the outer rows and columns of an agar plate consistently
grow differently from those in the interior. This systematic bias — the
"edge effect" — can confound quantitative comparisons between strains
or conditions.

## Causes

- **Temperature gradients** — the plate edge loses heat faster than the
  center, creating a radial temperature gradient that affects growth rate
- **Humidity gradients** — agar near the plate edge dries faster,
  concentrating nutrients and changing water activity
- **Gas exchange** — outer wells have greater access to atmospheric oxygen
- **Nutrient diffusion** — edge colonies have fewer neighbors competing
  for nutrients

The magnitude of the edge effect depends on incubator design, plate
type, and organism growth rate. It is typically strongest in 384-well
high-density plates.

## Detection

Edge-affected colonies are usually **larger** or **more intensely
pigmented** than interior colonies of the same strain. This is visible
in the raw measurements as a ring of elevated values around the plate
perimeter.

PhenoTypic's `EdgeCorrector` identifies edge-affected wells using a
permutation test against interior wells at matching grid positions.

## Correction

`EdgeCorrector` operates on measurement DataFrames (not images). It:

1. Identifies interior colonies (not on the outer rows/columns)
2. Selects the top-*n* interior values as a reference baseline
3. Tests each edge colony against the interior distribution
4. Adjusts values that are statistically elevated

```python
from phenotypic.analysis import EdgeCorrector

corrector = EdgeCorrector(
    on="Area",
    groupby=["Strain"],
    nrows=8, ncols=12,
    top_n=3, pvalue=0.05,
)
corrected_df = corrector.correct(measurements_df)
```

## Experimental Mitigation

Statistical correction is a post-hoc remedy. For critical experiments,
consider:

- **Randomized plate layout** — distribute strains randomly across the
  plate so edge effects average out across conditions
- **Border wells as controls** — inoculate border wells with a reference
  strain and exclude them from analysis
- **Humidified incubation** — reduces evaporation gradients
- **Multiple replicates** — average over several plates with different
  strain positions
