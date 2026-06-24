"""Sample identity and provenance metadata tags for the PhenoTypic module."""

from .._measurement_info import Entry
from .._tiers import IdentityInfo


class SAMPLE_METADATA(IdentityInfo):
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

    SAMPLE_ID = Entry("SampleID", "Unique identifier for the biological sample.")
    BIO_REPLICATE = Entry("BioReplicate",
                          "Biological replicate identifier. Biological replicates are"
                          " parallel measurements of biologically distinct samples that"
                          " capture random biological variation.", )
    EXP_REPLICATE = Entry("ExpReplicate",
                          "Experimental replicate identifier. An example of an"
                          " experimental replicate would be a pinned colony of the"
                          " same strain on another plate being subjected to the same"
                          " conditions.")
    TECHNICAL_REPLICATE = Entry("TechReplicate", "Technical replicate identifier.")
    CLONE = Entry("Clone", "Clone or isolate identifier.")
    LIBRARY_ID = Entry(
            "LibraryID",
            "Source library or collection identifier (e.g. a deletion collection).",
    )
    SOURCE_PLATE = Entry(
            "SourcePlate",
            "Identifier of the source plate the sample was pinned from.",
    )
    SOURCE_WELL = Entry("SourceWell", "Well position on the source plate (e.g. A1).")
    BARCODE = Entry("Barcode", "Molecular or sample barcode.")
    CONTROL = Entry("Control", "Control designation (e.g. positive, negative, blank).")
