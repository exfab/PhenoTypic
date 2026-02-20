"""Parameter sweep manifest generation for pipeline configurations.

This module provides a clean API for generating cartesian-product parameter
sweeps from :class:`~phenotypic.ImagePipeline` operations using a
``Sweep`` + ``generate_sweep_manifest`` interface.

**Public API:**

- :class:`Sweep` — declare an operation and its swept/fixed parameters.
- :class:`Presence` — like ``Sweep``, but also tests omitting the operation.
- :class:`Fixed` — explicitly mark a value as fixed (escape hatch for tuples).
- :func:`generate_sweep_manifest` — build manifest of all pipeline combinations.
- :func:`load_sweep_manifest` — reload pipelines from a saved manifest JSON.

Example:
    >>> from phenotypic.sweep import Sweep, generate_sweep_manifest
    >>> from phenotypic.enhance import GaussianBlur
    >>> from phenotypic.detect import OtsuDetector
    >>> config = [
    ...     Sweep(GaussianBlur, sigma=(1.0, 2.0), truncate=4.0),
    ...     Sweep(OtsuDetector, ignore_zeros=(True, False)),
    ... ]
    >>> manifest = generate_sweep_manifest(config)
    >>> manifest['total_pipelines']
    4
"""

from ._sweep_types import Fixed, Presence, Sweep
from ._generate_sweep import generate_sweep_manifest, load_sweep_manifest

__all__ = [
    "Fixed",
    "Presence",
    "Sweep",
    "generate_sweep_manifest",
    "load_sweep_manifest",
]
