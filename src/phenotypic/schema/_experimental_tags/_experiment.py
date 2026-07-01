"""Experiment-level and bookkeeping metadata tags for the PhenoTypic module."""

from .._measurement_info import Entry
from .._rembi import REMBI_MODULE
from .._tiers import IdentityInfo


class EXPERIMENT_METADATA(IdentityInfo):
    """Recommended ``MetadataExperiment_*`` tags for experiment-level bookkeeping.

    These group results by experiment/project and carry free-form provenance
    (experiment id, project, dataset, protocol, notes). ``Dataset`` matches the
    CLI-emitted ``MetadataExperiment_Dataset`` column. Members render as
    ``MetadataExperiment_<Label>`` (e.g. ``MetadataExperiment_ExperimentID``) in the
    ``Metadata`` column family shared with the other experimental-tag enums.
    Recommended vocabulary, not a validator.
    """

    @classmethod
    def category(cls) -> str:
        return "MetadataExperiment"

    @classmethod
    def rembi_module(cls) -> REMBI_MODULE:
        return REMBI_MODULE.STUDY

    EXPERIMENT_ID = Entry("ExperimentID", "Unique identifier for the experiment.")
    PROJECT = Entry("Project", "Project name or identifier.")
    DATASET = Entry(
        "Dataset",
        "Dataset name (matches the CLI-emitted MetadataExperiment_Dataset column).",
    )
    PROTOCOL = Entry("Protocol", "Protocol name or version followed.")
    NOTES = Entry("Notes", "Free-text notes or comments.")
