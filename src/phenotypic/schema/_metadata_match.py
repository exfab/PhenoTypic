"""Column names describing how a row matched the external ``--metadata`` CSV."""

from __future__ import annotations

from ._measurement_info import Entry
from ._tiers import QualityInfo


class METADATA_MATCH(QualityInfo):
    """Metadata-join provenance flags attached to the post-applied mirror.

    Emitted only by the CLI's ``--metadata`` left join (into
    ``deliverables/measurements.{csv,parquet}`` and everything derived from it).
    The clean ``master_measurements.*`` archive never carries these columns.
    """

    @classmethod
    def category(cls) -> str:
        return "QC"

    METADATA_ONLY = Entry(
        "MetadataOnly",
        "True when the row came from a --metadata CSV key that matched no "
        "measured object; every measurement/info column on the row is null.",
    )
