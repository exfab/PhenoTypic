"""Organism and genetics metadata tags for the PhenoTypic module."""

from ..._base._measurement_info import Entry
from ..._base._rembi import REMBI_MODULE
from ..._base._tiers import MetadataInfo


class GENETIC(MetadataInfo):
    """Recommended metadata tags describing the organism and its genetics.

    These name the genetic identity of the colonies on a plate (species, strain,
    genotype, markers). Like all experimental-tag enums they belong to the
    shared ``Metadata_<Label>`` namespace and slot directly into the ``--metadata``
    CSV join and the ``post/`` metadata operations. This is a recommended
    vocabulary, not a validator: arbitrary metadata columns are still accepted.
    """

    @classmethod
    def rembi_module(cls) -> REMBI_MODULE:
        return REMBI_MODULE.BIOSAMPLE

    ORGANISM = Entry("Organism", "Species or organism name (for example Saccharomyces cerevisiae).")
    STRAIN = Entry("Strain", "Strain name or identifier (for example BY4741).")
    GENOTYPE = Entry("Genotype", "Genotype description of the strain.")
    BACKGROUND = Entry("Background", "Genetic background or parent strain.")
    ALLELE = Entry("Allele", "Specific allele or mutation under study.")
    PLASMID = Entry("Plasmid", "Plasmid carried by the sample.")
    SELECTION_MARKER = Entry(
        "SelectionMarker",
        "Selectable marker gene (for example URA3, KanMX).",
    )
    MATING_TYPE = Entry("MatingType", "Mating type for yeast (for example MATa, MATalpha).")
    PLOIDY = Entry("Ploidy", "Ploidy of the organism (for example haploid, diploid).")
