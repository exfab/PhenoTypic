"""Experiment-level and bookkeeping metadata tags for the PhenoTypic module."""

from .._measurement_info import Entry
from .._rembi import REMBI_MODULE
from .._tiers import IdentityInfo


class EXPERIMENT_METADATA(IdentityInfo):
    """Recommended ``Metadata_*`` tags for experiment-level bookkeeping.

    These group results by experiment/project and carry free-form provenance
    (experiment id, project, dataset, protocol, notes). ``Dataset`` matches the
    CLI-emitted ``Metadata_Dataset`` column. Members render as ``Metadata_<Label>``
    (e.g. ``Metadata_ExperimentID``) and share the ``Metadata_`` namespace with the
    other experimental-tag enums. Recommended vocabulary, not a validator.
    """

    @classmethod
    def category(cls) -> str:
        return "Metadata"

    @classmethod
    def rembi_module(cls) -> REMBI_MODULE:
        return REMBI_MODULE.STUDY

    EXPERIMENT_ID = Entry("ExperimentID", "Unique identifier for the experiment.")
    PROJECT = Entry("Project", "Project name or identifier.")
    DATASET = Entry(
        "Dataset",
        "Dataset name (matches the CLI-emitted Metadata_Dataset column).",
    )
    PROTOCOL = Entry("Protocol", "Protocol name or version followed.")
    NOTES = Entry("Notes", "Free-text notes or comments.")
