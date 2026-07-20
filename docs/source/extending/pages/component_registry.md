# Component Registry System

PhenoTypic uses Python's import system as its component registry.
Operations and serializable pipeline components are discoverable by class name,
enabling JSON serialization and dynamic pipeline construction.

## How It Works

When `ImagePipeline.from_json()` encounters an operation like
`"GaussianBlur"`, it resolves the class by:

1. Searching known PhenoTypic modules (`phenotypic.enhance`,
   `phenotypic.detect`, `phenotypic.refine`, etc.)
2. Importing the class by name
3. Instantiating it with the saved parameters

## Registering Custom Operations

Custom operations are automatically discoverable when:

1. The class is importable from the current Python environment
2. The class name matches the name stored in the JSON

For operations defined in your own package:

```python
# my_package/my_enhancer.py
from phenotypic.abc_ import ImageEnhancer

class MyCustomEnhancer(ImageEnhancer):
    strength: float = 1.0

    def _operate(self, image):
        ...
        return image
```

When loading a pipeline that contains `MyCustomEnhancer`, ensure
`my_package` is installed and importable.

## Pipeline plot bindings

There is no image-level plot registry. Plot-capable objects opt into one of the
lifecycles in `phenotypic.abc_.plotting` and are listed explicitly in
`ImagePipeline(plots=[...])`. Serialization records either an identity-preserving
reference to another pipeline slot or an inline plot model.

## Naming Conventions

- Operation class names should be descriptive and end with their type:
  `GaussianBlur` (enhancer), `OtsuDetector` (detector),
  `SmallObjectRemover` (refiner)
- Avoid generic names like `MyOperation` — the class name appears in
  pipeline JSON files and should be self-documenting
