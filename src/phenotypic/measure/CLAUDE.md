# phenotypic.measure

## Implementation Conventions

### A measurement that runs an ImageOperation must run it on a provenance-detached copy

**Any `MeasureFeatures` subclass that applies an `ImageOperation`, `ObjectDetector`
or `ImagePipeline` internally must hand it a copy whose trailing provenance
application is closed — never the original `image`, and never a bare
`image.copy()`.** A measurement is a *reader*; a nested operation it runs to
derive something (a centre, a reference mask, a support region) is a private
probe whose steps do not belong in the plate's provenance.

Passing `inplace=False` is **not** sufficient, and assuming it is has already
cost one production run. `ImagePipeline.apply` does
`img = image if inplace else image.copy()` and then opens a provenance
application on `img` — but `Image.copy()` carries the journal *contents* across,
including the application the enclosing measure still has open. What happens
next depends on the `_application_owner_depth` contextvar, so the bug is
invisible in the obvious test:

| `_application_owner_depth` | context | behaviour |
|---|---|---|
| `> 0` | programmatic (`pipeline.apply`, notebooks, most tests) | `provenance_application` **joins** the open application and mutates it. Works by accident. |
| `== 0` | CLI stage 3 (`stage3_merge_measure_core`) | it **appends**, and `_append_application` refuses while the last application is unfinished. |

The depth-0 path raises:

    ValueError: cannot start a new provenance application before the last ends

which surfaced as *every* `MeasureOrientationZones` call failing across a whole
run while reproducing cleanly outside the CLI.

The correct construction — see `_canonical_zone_measure.py::_detected_centers`
(line ~229) for the reference implementation:

```python
probe_source = image.copy()
probe_journal = deepcopy(image._metadata.provenance_journal)
for application in probe_journal.get("applications", []):
    if application.get("status") not in {"complete", "failed"}:
        application["status"] = "complete"
probe_journal["status"] = "complete"
probe_source._metadata.provenance_journal = probe_journal

result = self.some_operation.apply(probe_source, inplace=False)
```

**Close the trailing application; do not clear the journal.** An empty
`applications` list satisfies the append path but breaks the join path —
`_current_application` raises `provenance journal has no application to mutate`
because it checks only for emptiness, not status. Closing satisfies both: an
append is legal once the last entry is terminal, and a join still finds a
non-empty list. The probe's journal is discarded with the probe.

When you add such a measurement, test it at **both** ownership depths. A test
that only exercises the default programmatic path passes on broken code:

```python
with provenance_application(image, kind="programmatic"):
    token = _application_owner_depth.set(0)   # emulate CLI stage 3
    try:
        op.measure(image)
    finally:
        _application_owner_depth.reset(token)
```

### Zone measurements share one canonical resolver

`MeasureSymZones` and `MeasureOrientationZones` both subclass
`CanonicalZoneMeasure` (`_canonical_zone_measure.py`), the private base that
declares the single "Method B" zone-resolution surface: `center_detector`,
`legacy_mode`, `outer_zone_percentile`, and the `zone_*` parameters. Add a zone
parameter there, not on one subclass.

- `center_detector` defaults to a scene-derived `ImagePipeline`
  (`SetDetectMode(mode="gray")` + `InoculumDetector(...)`) rather than the mask
  centroid. It is the nested operation the rule above exists for.
- Non-legacy mode **requires `method="distance"`** — anything else raises
  `canonical Method B requires method='distance'`. The guard is a pydantic
  `model_validator`, `_canonical_center_is_distance_based`, so it fires at
  construction rather than at measure time.
- Orientation serialization is a **hard cutover** (`e817205d`): a pre-cutover
  parameter block does not round-trip. Rebuild such configs from defaults rather
  than porting the old fields across.

### Measurement frames carry no experimental annotation

A measurement frame holds what the pipeline measured — `Metadata_ImageName`,
`Grid_RowNum`/`Grid_ColNum`, `Object_Label`, `Bbox_*` and feature columns. It has
no strain, medium, pH or timepoint: the CLI ships those separately as
`deliverables/metadata.csv`. Anything that needs to *group* by an experimental
factor (a `tune` scorer's `replicate_groupby`, a QC check, a plot) must join it
in first with `phenotypic.post.JoinMetadata`, which runs inside
`measure(apply_post=True)` and therefore lands before any scorer sees the frame.

This matters because the consumers fail **silently**:
`ReferenceFreeScorer._size_cv` falls back to a single whole-frame group when its
`replicate_groupby` columns are absent, with no warning — so a misconfigured
grouping looks correct and quietly computes the statistic over the wrong
population.
