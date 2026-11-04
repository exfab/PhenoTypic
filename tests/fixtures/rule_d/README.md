# `tests/fixtures/rule_d/`

Golden data for the Rule D port
(`src/phenotypic/analysis/qc/_fitting_window_rule.py`), consumed by
`tests/unit/analysis/test_fitting_window_rule.py`.

**Synthetic input only.** Every row in `synthetic_series.parquet` was drawn from
a seeded random generator. No measurement, image, strain or condition in it is
real: conditions are named `Condition A` to `Condition K`, replicate groups are
numbered, and every radius comes from a formula.

**The answers come from the reference, not from the port.** The windows,
outcomes, sentences and event lists in `synthetic_windows.json` were recorded by
running the reference implementation itself — the polars scripts
`prototype_rule_d.py` (`abrupt_bar`, `rule_d`, `_derivatives`) and
`c10_cv_by_ph.py` (`_cv_series`, `_cv_by_time`, `_detected`) — on that table. A
fixture recorded from the port would pin the port to its own output and be blind
to every transcription error in it. Only the reference scripts, and the helper
module they import their column names from, stay outside this repository;
`synthetic_windows.json` records the scripts' sha256 under `reference_sha256`.
The generators that built the synthetic table and ran the reference on it are
kept in the project's planning files.

## `synthetic_series.parquet`

5,496 rows, one per colony per frame, 30 frames six hours apart:

| Column | Meaning |
|---|---|
| `Metadata_Condition` | `Condition A` to `Condition K` |
| `Metadata_BioReplicate` | the replicate group, 1 to 3 (1 to 10 for E); null on an unpinned detection |
| `Metadata_FrameIndex` | 1 to 30, 1-based like the frame index the CLI stamps |
| `Metadata_Time` | elapsed hours, 0 to 174 |
| `Object_Label` | the colony's label; null on a phantom (undetected) row |
| `Size_RobustMeanRadius` | the radius, px; null on a phantom, and on some detected rows |
| `QC_MetadataOnly` | true on a phantom row |

Colony `k` of replicate group `g` at frame `f` has radius
`r(t) * exp(spread(f) * z[g, k] + jitter * e[g, k, f])`, with `z` drawn once
per colony and `e` drawn anew every frame. `spread(f)` decides where a
condition's replicates agree, and so where its window lies; the growth curve
`r(t)` does not enter the rule at all.

| Condition | Built to show |
|---|---|
| A, B, C, D | the Fitting windows page's series. A and B share a growth curve but settle at different rates; C's replicates split apart late; D's agreement jumps from frame to frame |
| E | ten groups of six colonies, so the counting step's reduction order is visible in the last bits |
| F | four frames with no detection, so the counted frame axis has gaps |
| G | sparse detection, missing metric values and six unpinned detections: a group below the minimum replicate count, a group whose mean is undefined, and rows the detection filters must drop |
| H | detected for 8 frames only |
| I | detected for 12 frames only |
| J | a gradual drift the window rides through, then an abrupt rise in rate that ends it |
| K | replicates still converging when the run ends |

A to D are row for row the page's series at the same seed (7); E to K draw from
a separate generator stream, so adding them cannot move A to D.

## `synthetic_windows.json`

Three runs, each with its own pooled bar, because the bar is a property of the
whole run:

| Run | Conditions | Outcomes |
|---|---|---|
| `page` | A to D | A 9–26, B 19–26, C 9–17, D `too_noisy` |
| `coverage` | A to K | 7 `found`, 1 each of `too_noisy`, `too_few_frames`, `too_few_evaluable_frames`, `abrupt_event_at_start` |
| `alone` | D only | D `found` 5–26 |

The `alone` run is the reason a condition alone is never `too_noisy`: with no
quieter conditions in the pool, the bar is D's own quiet level plus ten of its
own noise units. `no_smoothed_value`, the sixth exit, cannot be reached from
data (see the behavioural control in the test).

Each condition record carries:

* `frames` / `values` / `n_counted` — `c10_cv_by_ph._cv_series(...,
  statistic="mean")`'s output: the mean replicate CV at each **counted** frame,
  and how many replicate groups stand behind it. Frames where no group counted
  are absent, not zero.
* `n_groups` — the median of `n_counted`.
* `window`, `events`, `reason` — `rule_d`'s three reference outputs, verbatim.
* `outcome` — the port's `WindowOutcome` name for that sentence.

**`outcome` is a recorded interpretation, not independent evidence.** The
reference returns only a sentence, so the generator maps it to an outcome name
with a fixed table that raises on anything unmapped. The oracular assertion is
`result.detail == condition["reason"]`, sentence against sentence; the
behavioural controls in the test are what show the port classifies its exits
correctly.

`series_sha256` is the parquet the runs were recorded from; the test checks it,
so the two files cannot drift apart.

## Tolerances

Windows, outcomes, sentences and event lists are compared **exactly**. Floats
are compared at `rtol=1e-12`. The reference is nondeterministic in its last
digits — polars' `group_by` is unordered, so each frame's mean is summed in
whatever order the groups come out in — so an exact float comparison would be a
test that fails for no reason. `1e-12` is far below anything that could move a
window: the bar would have to change by percent, not by parts in 10¹⁵.

Regenerating rewrites the JSON's last digits without anything being wrong, so no
checksum of the JSON is kept.
