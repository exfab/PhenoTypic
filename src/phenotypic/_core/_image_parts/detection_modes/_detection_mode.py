"""Abstract base class and registry for detection modes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from phenotypic._core._image import Image

_DETECTION_MODE_REGISTRY: dict[str, DetectionMode] = {}


class DetectionMode(ABC):
    """Base class for detection matrix source modes.

    Subclasses define how the detection matrix is computed from
    raw image data (grayscale, individual RGB channels, etc.).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in ``detect_mode`` strings."""

    @property
    @abstractmethod
    def requires_rgb(self) -> bool:
        """Whether this mode needs RGB data to compute."""

    @abstractmethod
    def compute(self, image: Image) -> np.ndarray:
        """Return a fresh detection matrix from *image*.

        Args:
            image: The ``Image`` instance to compute from.

        Returns:
            A 2-D float32 array normalised to [0, 1].
        """

    @abstractmethod
    def compute_from_rgb(self, rgb: np.ndarray, *, image: Image) -> np.ndarray:
        """Project a substitute RGB array to a 2-D detection matrix.

        Unlike :meth:`compute`, the pixel data comes from *rgb* rather than from
        *image*. ``image`` supplies colour configuration only (``gamma``,
        ``illuminant``, ``_observer``), which the CIE L*a*b* modes require and the
        others ignore.

        Args:
            rgb: Float RGB array normalized to [0, 1], shape ``(rows, cols, 3)``.
            image: Source image, consulted for colour configuration only.

        Returns:
            A 2-D float32 array normalized to [0, 1].
        """


def register_detection_mode(cls: type[DetectionMode]) -> type[DetectionMode]:
    """Class decorator that instantiates *cls* and registers it by name.

    Raises:
        ValueError: If a mode with the same name is already registered.
    """
    instance = cls()
    name = instance.name
    if name in _DETECTION_MODE_REGISTRY:
        raise ValueError(
            f"Detection mode {name!r} is already registered "
            f"by {type(_DETECTION_MODE_REGISTRY[name]).__name__}"
        )
    _DETECTION_MODE_REGISTRY[name] = instance
    return cls


def get_detection_mode(name: str) -> DetectionMode:
    """Look up a registered detection mode by *name*.

    Raises:
        ValueError: If *name* is not registered.
    """
    try:
        return _DETECTION_MODE_REGISTRY[name]
    except KeyError:
        valid = ", ".join(sorted(_DETECTION_MODE_REGISTRY))
        raise ValueError(
            f"Unknown detect_mode {name!r}. "
            f"Available modes: {valid}"
        ) from None


def available_modes() -> tuple[str, ...]:
    """Return the names of all registered detection modes."""
    return tuple(sorted(_DETECTION_MODE_REGISTRY))
