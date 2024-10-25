from . import FeatureExtractor
from ..util.error_message import INTERFACE_ERROR_MSG

class GridFeatureExtractor(FeatureExtractor):
    def get_section_measurements(self,key):
        raise NotImplementedError(INTERFACE_ERROR_MSG)
