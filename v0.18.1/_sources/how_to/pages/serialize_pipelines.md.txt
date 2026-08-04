# Serialize and Share Pipelines

Save pipeline configurations to JSON for reproducibility and sharing.

## Save to JSON

```python
import phenotypic as pht
from phenotypic.enhance import BlurGauss, EnhanceLocalContrast
from phenotypic.detect import OtsuDetector

pipeline = pht.ImagePipeline(
    ops=[BlurGauss(sigma=2.0), EnhanceLocalContrast(clip_limit=0.01), OtsuDetector()],
    name="yeast_detection_v1",
)

# Save to a typed pipeline config file
pipeline.to_json("yeast_detection_v1.json.pht-pipe")

# Get as string (for logging, databases, etc.)
json_str = pipeline.to_json()
```

## Load from JSON

```python
loaded = pht.ImagePipeline.from_json("yeast_detection_v1.json.pht-pipe")
```

The PhenoTypic version is recorded in the JSON. You will get a warning if
the saved and current versions differ.

## What Is Captured

- All operation classes and their parameters
- Measurement configuration
- Pipeline name and description
- PhenoTypic version

## What Is Not Captured

- Internal state (attributes starting with `_`)
- DataFrame results
- Image data
