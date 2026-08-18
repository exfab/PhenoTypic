from phenotypic.schema import header_to_module, REMBI_MODULE


def test_metadata_headers_mapped():
    idx = header_to_module()
    assert idx["Metadata_Strain"] is REMBI_MODULE.BIOSAMPLE
    assert idx["Metadata_Dataset"] is REMBI_MODULE.STUDY
    assert idx["Metadata_Time"] is REMBI_MODULE.BIOSAMPLE
    assert idx["Metadata_ImageName"] is REMBI_MODULE.IMAGE_DATA


def test_measurement_headers_are_analyzed():
    idx = header_to_module()
    assert idx["Shape_Area"] is REMBI_MODULE.ANALYZED_DATA
