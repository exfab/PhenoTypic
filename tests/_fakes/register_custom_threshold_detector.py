"""Import side effect: register ``CustomThresholdDetector`` into ``phenotypic``.

Used through ``PHENOTYPIC_PRELOAD_MODULES`` by the tests that prove a custom
operation deserializes in every process that loads a pipeline, including
joblib/loky workers that never run the CLI's ``main``.
"""

import phenotypic

from tests._fakes.custom_threshold_detector import CustomThresholdDetector

phenotypic.CustomThresholdDetector = CustomThresholdDetector
