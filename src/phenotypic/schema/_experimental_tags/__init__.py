"""Standardized biological / experimental metadata-tag vocabulary.

Seven ``MeasurementInfo`` subclasses grouping recommended ``Metadata_*`` tags for
arrayed colony phenotyping. They all share the ``Metadata_`` namespace
(``category() == "Metadata"``); the grouping is organizational, giving users
canonical names + descriptions + auto-generated documentation tables that drop
straight into the ``--metadata`` CSV join and the ``post/`` metadata operations.

This is a *recommended vocabulary, not a validator* — arbitrary metadata columns are
still accepted everywhere. Re-exported from :mod:`phenotypic.schema`:

    from phenotypic.schema import SAMPLE_METADATA, CONDITION_METADATA
"""

from ._acquisition import ACQUISITION_METADATA
from ._condition import CONDITION_METADATA
from ._culture import CULTURE_METADATA
from ._experiment import EXPERIMENT_METADATA
from ._genetic import GENETIC_METADATA
from ._plate import PLATE_METADATA
from ._sample import SAMPLE_METADATA

__all__ = [
    "ACQUISITION_METADATA",
    "CONDITION_METADATA",
    "CULTURE_METADATA",
    "EXPERIMENT_METADATA",
    "GENETIC_METADATA",
    "PLATE_METADATA",
    "SAMPLE_METADATA",
]
