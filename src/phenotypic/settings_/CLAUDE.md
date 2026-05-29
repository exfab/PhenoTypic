# Settings Module

Global configuration via `phenotypic.settings_`. **Configure before importing other modules.**

```python
import phenotypic.settings_ as settings
settings.VALIDATE_OPS = False   # Default: False. Enable for debugging/development
settings.MPL.FIGSIZE = (12, 8)  # Default: (8, 6)

from phenotypic import ImagePipeline  # Import AFTER configuring
```

## Options

- **`VALIDATE_OPS`** — operation input/output validation. Default: False.
  Enable during development/debugging to validate operation contracts.
- **`MPL.FIGSIZE`** — default matplotlib figure size `(width, height)` in inches.
