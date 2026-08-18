"""Representative rendered metadata headers."""
from phenotypic import schema

def test_headers_self_describing():
    assert schema.GENETIC.STRAIN.value == "Metadata_Strain"
    assert schema.IMAGE.IMAGE_NAME.value == "Metadata_ImageName"
