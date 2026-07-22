"""CPU-only GpuDetector for tests (no torch). Shared by unit + integration."""

import numpy as np
import time
from pydantic import PrivateAttr
from skimage.measure import label as _sk_label

from phenotypic.abc_ import GpuDetector


class FakeGpuDetector(GpuDetector):
    """Thresholds the (stacked) input; labels it (instance) or returns the
    binary mask (semantic). ``output_kind``/``input_layer``/``supports_batching``
    are overrideable per test."""

    threshold: float = 0.5
    delay_seconds: float = 0.0
    # The synthetic threshold mask can legitimately touch the frame and does
    # not encode a model-produced background instance. Production detectors
    # retain the GpuDetector default of dropping that background label.
    drop_frame_background: bool = False
    _loaded: bool = PrivateAttr(default=False)

    def _ensure_model_loaded(self) -> None:
        self._loaded = True

    def _infer_one(self, sample):
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        gray = sample.mean(axis=-1) if sample.ndim == 3 else sample
        peak = gray.max()
        mask = gray > (self.threshold * peak) if peak > 0 else gray > 0
        if self.output_kind == "instance":
            return _sk_label(mask).astype(np.uint16)
        return mask
