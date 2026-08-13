"""Representative rendered metadata headers."""
from phenotypic import schema

def test_headers_self_describing():
    assert schema.GENETIC_METADATA.STRAIN.value == "MetadataGenetic_Strain"
    assert schema.METADATA.IMAGE_NAME.value == "MetadataImage_ImageName"
