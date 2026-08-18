"""Experiment-level and bookkeeping metadata tags for the PhenoTypic module."""

from .._measurement_info import Entry
from .._rembi import REMBI_MODULE
from .._tiers import MetadataInfo


class EXPERIMENT(MetadataInfo):
    """Recommended metadata tags for experiment-level bookkeeping.

    These group results by experiment/project and carry free-form provenance
    (experiment id, project, dataset, protocol, notes). Members render in the
    shared ``Metadata_<Label>`` namespace with the other experimental-tag enums.
    Recommended vocabulary, not a validator.
    """

    @classmethod
    def rembi_module(cls) -> REMBI_MODULE:
        return REMBI_MODULE.STUDY

    EXPERIMENT_ID = Entry("ExperimentID", "Unique identifier for the experiment.")
    PROJECT = Entry("Project", "Project name or identifier.")
    DATASET = Entry(
        "Dataset",
        "Dataset name used to group CLI-emitted measurements.",
    )
    PROTOCOL = Entry("Protocol", "Protocol name or version followed.")
    NOTES = Entry("Notes", "Free-text notes or comments.")
