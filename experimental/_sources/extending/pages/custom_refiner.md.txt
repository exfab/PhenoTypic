# Custom Refiner

Subclass `ObjectRefiner` to clean up detection masks.

```python
from phenotypic.abc_ import ObjectRefiner
from skimage.measure import label
import numpy as np

class RemoveLargeObjects(ObjectRefiner):
    """Remove objects larger than a maximum area."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size

    def _operate(self, image):
        objmap = image.objmap[:]
        for label_id in range(1, objmap.max() + 1):
            if (objmap == label_id).sum() > self.max_size:
                objmap[objmap == label_id] = 0
        # Re-label to ensure consecutive IDs
        image.objmap[:] = label(objmap > 0).astype(np.uint16)
        image.objmask[:] = objmap > 0
        return image
```

## Contract

- Read from and write to `objmask` and `objmap`
- Do not modify `rgb`, `gray`, or `detect_mat`
- Return the modified image
