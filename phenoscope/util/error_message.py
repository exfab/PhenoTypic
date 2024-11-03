INTERFACE_ERROR_MSG = "An interface method was called when it was not supposed to be. Make sure any inherited classes properly overload this method."
NO_OUTPUT_ERROR_MSG = 'No output was returned in this operation'
OUTPUT_NOT_IMAGE_MSG = "This method's output is not a phenoscope. Image object even though it should be."
OUTPUT_NOT_TABLE_MSG = "This method's output is not a pandas DataFrame even though it should be."

INVALID_MASK_SHAPE_MSG = 'Object Mask shape should be the same as the image shape.'
INVALID_MAP_SHAPE_MSG = 'Object map_modification should be the same as the image shape.'

ARRAY_CHANGE_ERROR_MSG = 'The image array of the input was changed. This operation should not change the image array of the input.'
ENHANCED_ARRAY_CHANGE_ERROR_MSG = 'The enhanced image array of the input was changed. This operation should not change the enhanced image array of the input'
MASK_CHANGE_ERROR_MSG = 'The object mask of the input was changed. This operation should not change the object mask of the input'
MAP_CHANGE_ERROR_MSG = ' The object map_modification of the input was changed. This operation should not change the object map_modification of the input'

MISSING_MASK_ERROR_MSG = 'This image is missing an object mask. This operation requires the image to have an object mask. Run a detector on the image first.'
MISSING_MAP_ERROR_MSG = 'This image is missing an object mask. This operation requires the image to have an object mask. Run a detector on the image first.'

GRID_SERIES_INPUT_IMAGE_ERROR_MSG = 'For GridOperation classes with the exception of GridExtractor objects, the input must be an instance of the GriddedImage object type.'