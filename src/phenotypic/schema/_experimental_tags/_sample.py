"""Sample identity and provenance metadata tags for the PhenoTypic module."""

from .._measurement_info import MeasurementInfo


class SAMPLE_METADATA(MeasurementInfo):
    """Recommended ``Metadata_*`` tags identifying a sample and its provenance.

    These distinguish individual biological samples and track where each colony came
    from (replicate, clone, source plate/well, library). Members render as
    ``Metadata_<Label>`` (e.g. ``Metadata_Replicate``) and share the ``Metadata_``
    namespace with the other experimental-tag enums. Recommended vocabulary, not a
    validator: arbitrary metadata columns are still accepted.
    """

    @classmethod
    def category(cls) -> str:
        return "Metadata"

    SAMPLE_ID = "SampleID", "Unique identifier for the biological sample."
    REPLICATE = "Replicate", "Biological replicate identifier."
    TECHNICAL_REPLICATE = "TechnicalReplicate", "Technical replicate identifier."
    CLONE = "Clone", "Clone or isolate identifier."
    LIBRARY_ID = (
        "LibraryID",
        "Source library or collection identifier (e.g. a deletion collection).",
    )
    SOURCE_PLATE = (
        "SourcePlate",
        "Identifier of the source plate the sample was pinned from.",
    )
    SOURCE_WELL = "SourceWell", "Well position on the source plate (e.g. A1)."
    BARCODE = "Barcode", "Molecular or sample barcode."
    CONTROL = "Control", "Control designation (e.g. positive, negative, blank)."
