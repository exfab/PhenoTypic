"""Sample identity and provenance metadata tags for the PhenoTypic module."""

from .._measurement_info import Entry
from .._rembi import REMBI_MODULE
from .._tiers import MetadataInfo


class SAMPLE(MetadataInfo):
    """Recommended metadata tags identifying a sample and its provenance.

    These distinguish individual biological samples and track where each colony came
    from (replicate, clone, source plate/well, library). Members render in the
    shared ``Metadata_<Label>`` namespace with the other experimental-tag enums.
    Recommended vocabulary, not a validator: arbitrary metadata columns are still
    accepted.
    """

    @classmethod
    def rembi_module(cls) -> REMBI_MODULE:
        return REMBI_MODULE.BIOSAMPLE

    SAMPLE_ID = Entry("SampleID", "Unique identifier for the biological sample.")
    BIO_REPLICATE = Entry("BioReplicate",
                          "Biological replicate identifier. Biological replicates are"
                          " parallel measurements of biologically distinct samples that"
                          " capture random biological variation.", )
    COND_REPLICATE = Entry("CondReplicate",
                           "Experimental Conditional replicate identifier. An example "
                           " of an experimental replicate would be a pinned colony of "
                           " the same strain on another plate being subjected to the "
                           "same conditions.")
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
    SOURCE_WELL = Entry("SourceWell",
                        "Well position on the source plate (e.g. A1). This can "
                        "potentially be distinct from `Grid` values when using "
                        "irregular or sparse grid parameters.")
    BARCODE = Entry("Barcode", "Molecular or sample barcode.")
    CONTROL = Entry("Control", "Control designation (e.g. positive, negative, blank).")
