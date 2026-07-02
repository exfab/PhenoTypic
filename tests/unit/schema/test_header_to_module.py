from phenotypic.schema import header_to_module, REMBI_MODULE


def test_metadata_headers_mapped():
    idx = header_to_module()
    assert idx["MetadataGenetic_Strain"] is REMBI_MODULE.BIOSAMPLE
    assert idx["MetadataExperiment_Dataset"] is REMBI_MODULE.STUDY
    assert idx["MetadataCulture_Time"] is REMBI_MODULE.BIOSAMPLE
    assert idx["MetadataImage_ImageName"] is REMBI_MODULE.IMAGE_DATA


def test_measurement_headers_are_analyzed():
    idx = header_to_module()
    assert idx["Shape_Area"] is REMBI_MODULE.ANALYZED_DATA
