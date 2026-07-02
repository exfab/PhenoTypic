"""Scheme-B per-enum metadata category prefixes (Task B2)."""
from phenotypic import schema

EXPECTED = {
    "METADATA": "MetadataImage",
    "STUDY_METADATA": "MetadataStudy",
    "EXPERIMENT_METADATA": "MetadataExperiment",
    "GENETIC_METADATA": "MetadataGenetic",
    "SAMPLE_METADATA": "MetadataSample",
    "CONDITION_METADATA": "MetadataCondition",
    "CULTURE_METADATA": "MetadataCulture",
    "PLATE_METADATA": "MetadataPlate",
    "ACQUISITION_METADATA": "MetadataAcquisition",
}


def test_scheme_b_category_names():
    for enum_name, cat in EXPECTED.items():
        assert getattr(schema, enum_name).category() == cat


def test_headers_self_describing():
    assert schema.GENETIC_METADATA.STRAIN.value == "MetadataGenetic_Strain"
    assert schema.METADATA.IMAGE_NAME.value == "MetadataImage_ImageName"
