"""What an operation needs from its run, stated by the operation itself.

Spec: ``docs/superpowers/specs/2026-09-24-cli-preflight/design.md`` §3.

The CLI's run preflight reads these declarations to refuse an incompatible
run before any image is processed: a grid operation under ``--image-type
Image``, an RGB-reading operation on grayscale inputs, a detector whose
optional package or model weights are missing. Declaring them on the
operation, beside the code that creates the requirement, keeps them from
drifting the way a checker-side table would as operations are added.

Standard library only: this module is imported by ``phenotypic.abc_`` and must
not add to its import cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class WeightRequirement:
    """Model weights an operation loads at run time.

    Attributes:
        model: A readable identifier, e.g. ``"sam2:tiny"`` or ``"dinov3:base"``.
        license_key: The ``PHENOTYPIC_ACCEPT_MODEL_LICENSE`` token that must be
            accepted before the weights may be downloaded or used, or ``None``
            when the weights are not gated.
        is_cached: Reports whether the weights are already on local disk,
            without importing ``torch`` and without network access. Returns
            ``None`` when that cannot be told.
    """

    model: str
    license_key: Optional[str]
    is_cached: Callable[[], Optional[bool]]


@dataclass(frozen=True)
class OperationRequirements:
    """Everything an operation needs from the run that executes it.

    Attributes:
        grid_image: The operation raises ``GridImageInputError`` on a plain
            ``Image``.
        rgb_input: The operation reads RGB pixels and fails on a grayscale
            image.
        modules: Importable module names the operation imports lazily; checked
            with ``importlib.util.find_spec``, never imported.
        extra: The ``pyproject`` extra that provides ``modules``, named in the
            remedy; ``None`` when no extra does (e.g. a conda-only package).
        weights: Model weights the operation loads.
    """

    grid_image: bool = False
    rgb_input: bool = False
    modules: tuple[str, ...] = ()
    extra: Optional[str] = None
    weights: tuple[WeightRequirement, ...] = ()
