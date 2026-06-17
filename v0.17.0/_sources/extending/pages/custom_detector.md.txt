# Custom Detector

Subclass `ObjectDetector` to create a detector that writes to `objmask`
and `objmap`. Operations are pydantic models: declare each parameter as
an annotated class-level field — there is no hand-written `__init__`.

```python
from phenotypic.abc_ import ObjectDetector
import numpy as np
from skimage.measure import label

class MyDetector(ObjectDetector):
    """Detect colonies using a fixed intensity threshold.

    Args:
        threshold: Intensity cutoff; pixels above it are marked as colony.
    """

    threshold: float = 0.5

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
