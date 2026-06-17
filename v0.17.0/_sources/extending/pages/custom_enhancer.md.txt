# Custom Enhancer

Subclass `ImageEnhancer` to create an operation that modifies `detect_mat`.
Operations are pydantic models: declare each parameter as an annotated
class-level field — there is no hand-written `__init__`.

```python
from phenotypic.abc_ import ImageEnhancer
import numpy as np

class LogTransform(ImageEnhancer):
    """Apply a log transform to compress the dynamic range of detect_mat.

    Args:
        c: Scaling coefficient applied after the log transform.
    """

    c: float = 1.0

    def _operate(self, image):
        mat = image.detect_mat[:]
        image.detect_mat[:] = self.c * np.log1p(mat)
        return image
```

## Contract

- Read from and write to `detect_mat`
- Do not modify `rgb`, `gray`, `objmask`, or `objmap`
- Return the modified image
