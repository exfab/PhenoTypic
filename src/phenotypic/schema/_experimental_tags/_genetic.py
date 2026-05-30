"""Organism and genetics metadata tags for the PhenoTypic module."""

from .._measurement_info import MeasurementInfo


class GENETIC_METADATA(MeasurementInfo):
    """Recommended ``Metadata_*`` tags describing the organism and its genetics.

    These name the genetic identity of the colonies on a plate (species, strain,
    genotype, markers). Like all experimental-tag enums they share the
    ``Metadata_`` namespace, so members render as ``Metadata_<Label>`` (e.g.
    ``Metadata_Strain``) and slot directly into the ``--metadata`` CSV join and the
    ``post/`` metadata operations. This is a recommended vocabulary, not a validator:
    arbitrary metadata columns are still accepted.
    """

    @classmethod
    def category(cls) -> str:
        return "Metadata"

    ORGANISM = "Organism", "Species or organism name (e.g. Saccharomyces cerevisiae)."
    STRAIN = "Strain", "Strain name or identifier (e.g. BY4741)."
    GENOTYPE = "Genotype", "Genotype description of the strain."
    BACKGROUND = "Background", "Genetic background or parent strain."
    ALLELE = "Allele", "Specific allele or mutation under study."
    PLASMID = "Plasmid", "Plasmid carried by the sample."
    SELECTION_MARKER = (
        "SelectionMarker",
        "Selectable marker gene (e.g. URA3, KanMX).",
    )
    MATING_TYPE = "MatingType", "Mating type for yeast (e.g. MATa, MATalpha)."
    PLOIDY = "Ploidy", "Ploidy of the organism (e.g. haploid, diploid)."
