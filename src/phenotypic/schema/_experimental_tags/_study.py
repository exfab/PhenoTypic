"""Study-level (REMBI Study component) metadata tags."""

from .._measurement_info import Entry
from .._rembi import REMBI_MODULE
from .._tiers import MetadataInfo


class STUDY(MetadataInfo):
    """Recommended metadata tags for the REMBI Study component.

    One set per run (title, authors, license, …). Mirrors REMBI's Study field
    names. Members render in the shared ``Metadata_<Label>`` namespace.
    Structured REMBI lists (authors/publications/links) are flattened to scalar
    tags whose value may be a delimited string.
    Recommended vocabulary, not a validator.
    """

    @classmethod
    def rembi_module(cls) -> REMBI_MODULE:
        return REMBI_MODULE.STUDY

    TITLE = Entry("Title", "Study title.")
    DESCRIPTION = Entry("Description", "Free-text study description.")
    PRIVATE_UNTIL_DATE = Entry(
        "PrivateUntilDate", "Embargo date until which the study stays private.")
    KEYWORDS = Entry("Keywords", "Keywords describing the study.")
    AUTHOR = Entry("Author", "Study author(s); delimited string when multiple.")
    LICENSE = Entry("License", "Data license (e.g. CC0, CC-BY-4.0).")
    FUNDING = Entry("Funding", "Funding statement or grant reference(s).")
    PUBLICATIONS = Entry("Publications", "Associated publication(s) or DOI(s).")
    LINKS = Entry("Links", "Related links or external resource URLs.")
    ACKNOWLEDGEMENTS = Entry("Acknowledgements", "Acknowledgements text.")
