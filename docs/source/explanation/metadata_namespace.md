# Flat metadata namespace and semantic owners

PhenoTypic writes every metadata field as `Metadata_<Label>`. The prefix is
short and predictable for people entering CSV headers, while the public enum
that owns a label keeps its semantic category available to code:

| Public enum | Example member | Emitted header | REMBI module |
|---|---|---|---|
| `IMAGE` | `IMAGE.IMAGE_NAME` | `Metadata_ImageName` | ImageData |
| `STUDY` | `STUDY.TITLE` | `Metadata_Title` | Study |
| `EXPERIMENT` | `EXPERIMENT.DATASET` | `Metadata_Dataset` | Study |
| `GENETIC` | `GENETIC.STRAIN` | `Metadata_Strain` | Biosample |
| `SAMPLE` | `SAMPLE.BIO_REPLICATE` | `Metadata_BioReplicate` | Biosample |
| `CONDITION` | `CONDITION.MEDIA` | `Metadata_Media` | SpecimenPreparation |
| `CULTURE` | `CULTURE.TIME` | `Metadata_Time` | Biosample |
| `PLATE` | `PLATE.PLATE_ID` | `Metadata_PlateID` | SpecimenPreparation |
| `ACQUISITION` | `ACQUISITION.INSTRUMENT` | `Metadata_Instrument` | ImageAcquisition |

All nine enums inherit {class}`phenotypic.schema.MetadataInfo`, whose
`category()` is `"Metadata"`. Category strings therefore cannot distinguish
owners. Use enum identity and the ownership APIs instead:

```python
from phenotypic.schema import GENETIC, MetadataInfo
from phenotypic.sdk_ import (
    metadata_member_for_header,
    metadata_owner_for_header,
)

member = metadata_member_for_header("Metadata_Strain")
assert member is GENETIC.STRAIN
assert metadata_owner_for_header("Metadata_Strain") is GENETIC
assert "Metadata_Strain" in GENETIC.header_set()
assert issubclass(metadata_owner_for_header("Strain"), MetadataInfo)
```

The lookup functions also accept a bare known label such as `Strain` and an
exact historical spelling such as `MetadataGenetic_Strain`. Corresponding
`metadata_member_for_label()` and `metadata_owner_for_label()` helpers are
available when the input is conceptually a label. Unknown metadata remains
valid but has no owner and routes to REMBI `Uncategorized`.

Use {func}`phenotypic.sdk_.is_metadata_header` to identify the metadata family.
It recognizes canonical `Metadata_*` headers and the finite exact set of
historical headers. It intentionally rejects arbitrary lookalikes such as
`MetadataFoo_Bar`. Use {func}`phenotypic.sdk_.normalize_metadata_columns` at a
DataFrame ingress boundary. It returns a copy, canonicalizes bare and historical
headers, and coalesces duplicate aliases only when their dtypes are compatible
and their overlapping non-null values agree.

```{seealso}
To attach metadata to a run and emit `deliverables/rembi.yaml`, see
{doc}`/how_to/pages/rembi_metadata`.
```

## Column order

Measurement sheets order the metadata front block by a bench-scientist
narrative, not by spelling or REMBI module:

1. **Identity:** fields owned by `SAMPLE`, then `PLATE`
2. **Strain:** fields owned by `GENETIC`
3. **Condition:** fields owned by `CONDITION`, then `CULTURE`
4. **Design and provenance:** fields owned by `EXPERIMENT`, `STUDY`, then
   `ACQUISITION`

Unknown `Metadata_*` fields trail the known owner blocks. Within an owner,
columns follow enum declaration order. The `IMAGE` block is per-image
provenance and appears after measurements, before the per-object info block.
REMBI classification is a separate provenance axis and does not drive this
presentation order.

## Compatibility and migration

Stored-header compatibility is permanent. Ordinary reads normalize exact old
headers such as `MetadataImage_ImageName` and `MetadataCulture_Time` in memory
without rewriting their source. The previous Python enum class names remain as
one-transition-release aliases: importing one emits `DeprecationWarning`, and
the alias is absent from `phenotypic.schema.__all__` and schema discovery. New
code should use `IMAGE`, `CULTURE`, and the other canonical owner names.

`--mode recompile` is the explicit mutation boundary for a bundle. Local and
SLURM recompiles preflight and migrate bundle-owned authoritative legacy data
before aggregation. A conflict or failed target stops recompile before new
aggregate outputs are published. Canonical bundles are an idempotent no-op
apart from preflight.

Migration uses source and plan fingerprints plus a prepared/applied receipt
journal. CSV, parquet, typed pipeline JSON, and HDF metadata attributes are
replaced atomically. HDF migration writes and validates a sibling copy, changes
only metadata attributes and the metadata-schema marker, and leaves the HDF
layout `schema_version`, arrays, grid state, and unrelated attributes alone.
Receipts make an interrupted bundle migration resumable and can be passed to
`rollback_metadata_migration()`.

An external file supplied with `--metadata` is always read-only during
recompile: PhenoTypic normalizes its contents in memory without writing to the
source, and the regenerated bundle-owned `deliverables/metadata.csv` uses
canonical headers. To mutate a standalone external file, call
`migrate_metadata_file()` explicitly.

```python
from pathlib import Path
from phenotypic.sdk_ import (
    migrate_metadata_bundle,
    migrate_metadata_file,
    preflight_metadata_schema,
    rollback_metadata_migration,
)

source = Path("metadata.csv")
report = preflight_metadata_schema(source)
result = migrate_metadata_file(
    source,
    expected_source_fingerprint=report.source_fingerprint,
)

bundle_report = preflight_metadata_schema(Path("out"))
bundle_result = migrate_metadata_bundle(
    Path("out"),
    expected_plan_fingerprint=bundle_report.plan_fingerprint,
)

if bundle_result.receipt_path is not None:
    rollback_metadata_migration(bundle_result.receipt_path)
```

Always inspect `status`, `conflicts`, and the proposed header maps in a
preflight report before invoking standalone migration. A blocked report never
mutates its target.
