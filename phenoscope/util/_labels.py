# ----- PhenoScope Variable Label Names -----

METADATA_LABELS = [
    (STATUS_VALIDITY_LABEL := "status_valid_analysis"),
    (PLATE_NAME_LABEL := "plate_name")
]

NUMERIC_METADATA_LABELS = [
    (SAMPLING_DAY_LABEL := "SamplingDay"),
]

PHENOSCOPE_MEASUREMENTS = [
    "Intensity_IntegratedColorIntensityRed",
    "Intensity_IntegratedColorIntensityGreen",
    "Intensity_IntegratedColorIntensityBlue",
    "ImgHeight",
    "ImgWidth"
]

BASIC_CP_API_MEASUREMENT_LABELS = [
    'AreaShape',
    'Intensity',
    'Texture'
]

class MetadataLabels:
    STATUS_VALIDITY_LABEL = 'Status_ValidSegmentation'
    PLATE_NAME_LABEL = 'Metadata_PlateName'

class NumericMetadataLabels:
    SAMPLING_DAY_LABEL = "Metadata_SamplingDay"

class PhenoscopeMeasurementLabels:
    INTEGRATED_INTENSITY_RED = 'Intensity_IntegratedColorIntensityRed'
    INTEGRATED_INTENSITY_BLUE = 'Intensity_IntegratedColorIntensityBlue'
    INTEGRATED_INTENSITY_GREEN = 'Intensity_IntegratedColorIntensityGreen'
    IMAGE_HEIGHT = 'Attribute_ImageHeight'
    IMAGE_WIDTH = 'Attribute_ImageWidth'

class FinderLabels:
    ROW_COORD = 'rr'
    COL_COORD = 'cc'
    GRIDROW = 'gridrow'
    GRIDROW_MSE = 'gridrow_mse'

    GRIDCOL = 'gridcol'
    GRIDCOL_MSE = 'gridcol_mse'

    GRID_SECTION = 'grid_section'
    GRID_SECTION_MSE = 'grid_section_mse'

class BlobFinderLabels(FinderLabels):
    SEARCH_METHOD_LOG = 'log'
    SEARCH_METHOD_DOG = 'dog'
    SEARCH_METHOD_DOH = 'doh'
    SIGMA = 'sigma'



