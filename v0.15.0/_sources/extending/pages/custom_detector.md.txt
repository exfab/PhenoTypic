# Custom Detector

Subclass `ObjectDetector` to create a detector that writes to `objmask`
and `objmap`.

```python
from phenotypic.abc_ import ObjectDetector
import numpy as np
from skimage.measure import label

class MyDetector(ObjectDetector):
    """Detect colonies using a fixed intensity threshold."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def _operate(self, image):
        mask = image.detect_mat[:] > self.threshold
        image.objmask[:] = mask
        image.objmap[:] = label(mask).astype(np.uint16)
        return image
```

## Contract

- Read from `detect_mat`
- Write to `objmask` (bool) and `objmap` (uint16 labeled array)
- Do not modify `rgb`, `gray`, or `detect_mat`
- Return the modified image
