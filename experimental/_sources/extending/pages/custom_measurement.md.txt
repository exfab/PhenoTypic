# Custom Measurement

Subclass `MeasureFeatures` to extract custom features from detected
colonies.

```python
from phenotypic.abc_ import MeasureFeatures
import pandas as pd
import numpy as np

class MeasureAspectRatio(MeasureFeatures):
    """Measure the aspect ratio of each colony's bounding box."""

    def _operate(self, image):
        from skimage.measure import regionprops

        objmap = image.objmap[:]
        rows = []
        for prop in regionprops(objmap):
            height = prop.bbox[2] - prop.bbox[0]
            width = prop.bbox[3] - prop.bbox[1]
            rows.append({
                "ObjectLabel": prop.label,
                "AspectRatio": height / max(width, 1),
            })
        return pd.DataFrame(rows)
```

## Contract

- Read from `objmap` and image data (`gray`, `rgb`)
- Return a `pd.DataFrame` with one row per object
- Include an `ObjectLabel` column matching `objmap` labels
- Do not modify the image
