"""Label-uniqueness gate for the metadata-namespace enums (Task B8).

``metadata_category_for_label`` maps a bare label (e.g. ``"Strain"``) to the
single category that owns it (``"MetadataGenetic"``). That routing is only
well-defined if no bare label is declared by two different metadata enums. This
gate fails CI the moment a future enum reuses a label already claimed elsewhere
in the ``Metadata`` column family, which would otherwise silently mis-route the
column (schema-aware prefixing + the REMBI reverse index both key off the label).
"""
from collections import defaultdict

import phenotypic.schema as schema
from phenotypic.schema import MeasurementInfo


def _metadata_enums():
    for name in schema.__all__:
        obj = getattr(schema, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, MeasurementInfo)
            and obj is not MeasurementInfo
            and list(obj)
            and obj.category().startswith("Metadata")
        ):
            yield obj


def test_metadata_labels_are_globally_unique():
    owners: dict[str, list[str]] = defaultdict(list)
    for enum in _metadata_enums():
        for member in enum:
            owners[member.label].append(f"{enum.__name__}.{member.name}")
    collisions = {label: sites for label, sites in owners.items() if len(sites) > 1}
    assert not collisions, (
        "metadata labels must be unique across the Metadata-namespace enums so "
        "metadata_category_for_label cannot mis-route:\n"
        + "\n".join(f"  {label!r}: {sites}" for label, sites in collisions.items())
    )
