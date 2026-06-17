# Using Mixins

PhenoTypic provides mixin classes that add common capabilities to custom
operations without duplicating code.

## FootprintMixin

Adds structuring element (footprint) handling to operations that need
morphological kernels.

```python
from phenotypic.abc_ import ObjectRefiner
from phenotypic.tools_.mixin import FootprintMixin

class MyRefiner(FootprintMixin, ObjectRefiner):
    # Parameters are annotated class-level fields (operations are
    # pydantic models); the mixin contributes helper methods.
    shape: str = "disk"
    width: int = 5

    def _operate(self, image):
        footprint = self._make_footprint(self.shape, self.width)
        # Use footprint for morphological operations
        ...
        return image
```

## GridInferenceMixin

Provides grid-related utilities for operations that need to work with
grid section boundaries.

## ClipControlMixin

Adds configurable output clipping to enhancers that might produce
out-of-range values.

## When to Use Mixins

Use a mixin when your custom operation needs functionality that is
already implemented elsewhere. Mixins follow Python's cooperative
multiple inheritance pattern — place them before the ABC in the class
definition.
