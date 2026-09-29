"""A custom operation defined outside ``phenotypic`` that does NOT register itself.

Importing this module alone is not enough for ``ImagePipeline.from_json`` to
resolve ``CustomThresholdDetector``: pipeline JSON records bare class names and
resolution searches only the ``phenotypic`` namespace. See
``register_custom_threshold_detector`` for the self-registering companion.
"""

from phenotypic.abc_ import ObjectDetector


class CustomThresholdDetector(ObjectDetector):
    """Foreground is every pixel of the detection matrix above ``thresh``."""

    thresh: float = 0.5

    def _operate(self, image):
        image.objmask[:] = image.detect_mat[:] > self.thresh
        return image
