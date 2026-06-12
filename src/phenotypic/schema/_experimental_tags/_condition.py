"""Media and growth-condition metadata tags for the PhenoTypic module."""

from .._measurement_info import Entry, MeasurementInfo


class CONDITION_METADATA(MeasurementInfo):
    """Recommended ``Metadata_*`` tags describing media and experimental conditions.

    These name the chemical environment and perturbations applied to the colonies
    (medium, carbon/nitrogen source, supplements, treatments, compounds, stress).
    Members render as ``Metadata_<Label>`` (e.g. ``Metadata_Treatment``) and share the
    ``Metadata_`` namespace with the other experimental-tag enums. Recommended
    vocabulary, not a validator.
    """

    @classmethod
    def category(cls) -> str:
        return "Metadata"

    MEDIA = Entry("Media", "Growth medium name (e.g. YPD, SC, LB).")
    CARBON_SOURCE = Entry("CarbonSource", "Primary carbon source (e.g. glucose, galactose).")
    NITROGEN_SOURCE = Entry("NitrogenSource", "Primary nitrogen source.")
    PH = Entry("pH", "pH of the growth medium.")
    SUPPLEMENT = Entry("Supplement", "Medium supplement or additive.")
    ANTIBIOTIC = Entry("Antibiotic", "Antibiotic added to the medium.")
    INDUCER = Entry("Inducer", "Inducing agent (e.g. galactose, IPTG, doxycycline).")
    TREATMENT = Entry("Treatment", "Experimental treatment or perturbation.")
    COMPOUND = Entry("Compound", "Chemical compound or drug applied.")
    CONCENTRATION = Entry("Concentration", "Concentration of the compound or treatment.")
    DOSE = Entry("Dose", "Dose applied to the sample.")
    STRESS = Entry("Stress", "Applied stress condition (e.g. heat, osmotic, oxidative).")
