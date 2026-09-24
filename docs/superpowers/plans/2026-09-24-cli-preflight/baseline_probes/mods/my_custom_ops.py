"""Custom op defined outside phenotypic; NOT self-registering."""
import numpy as np
from phenotypic.abc_ import ObjectDetector

class MyThreshDetector(ObjectDetector):
    thresh: float = 0.5
    def _operate(self, image):
        m = image.detect_mat[:] > self.thresh
        image.objmask[:] = m
        return image
