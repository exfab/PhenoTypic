from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from phenotypic._core._image import Image
    from phenotypic._core._grid_image import GridImage

from ._object_detector import ObjectDetector


# <<Interface>>
class GpuDetector(ObjectDetector, ABC):
    """Marker ABC for object detectors that require GPU acceleration.

    Subclass GpuDetector when your detection algorithm depends on a GPU
    (e.g., deep-learning foundation models like SAM2 or micro-sam).

    When a pipeline contains a GpuDetector, the CLI enforces:

    - **Local execution:** Sequential processing (n_jobs=1) to avoid
      multiple workers competing for the same GPU.
    - **SLURM execution:** Automatically requests GPU resources
      (``--gpus-per-node=1``) if the user hasn't specified GPU args.
      Raises an error if the target partition has no GPUs.
    - **No GPU available:** Raises RuntimeError at pipeline validation
      time with a clear message.

    **When to subclass GpuDetector vs ObjectDetector**

    Subclass GpuDetector if your detector relies on GPU-accelerated inference:

    - Deep-learning models (SAM2, micro-sam, or custom neural networks).
    - Any algorithm that requires ``torch``, ``tensorflow``, or similar
      GPU-backed frameworks at inference time.
    - Detectors where CPU fallback is technically possible but
      impractically slow for production use.

    Subclass ObjectDetector directly if your algorithm is CPU-based:

    - Classical computer vision (thresholding, edge detection, watershed).
    - Algorithms implemented with NumPy, SciPy, or scikit-image.
    - Detectors that run in milliseconds on CPU.

    **Lazy model loading**

    GpuDetector subclasses should defer model construction to the first
    ``apply()`` call rather than ``__init__()``. This enables:

    - Fast construction and serialization round-trips without GPU/torch.
    - Pipeline ``to_json()``/``from_json()`` without importing heavy
      dependencies.
    - Parameter inspection and validation before committing GPU memory.

    Use a ``_ensure_model_loaded()`` pattern::

        from pydantic import PrivateAttr

        class MyGpuDetector(GpuDetector):
            model_size: str = "small"  # Annotated class-level fields
            device: str = "auto"
            # underscore-prefixed private attr → skipped by serialization
            _model: object = PrivateAttr(default=None)

            def _ensure_model_loaded(self):
                if self._model is not None:
                    return
                import torch  # lazy import
                # ... build model ...

            def _operate(self, image):
                self._ensure_model_loaded()
                # ... run inference ...
                return image

    Notes:
        This is a marker ABC with no additional methods beyond those
        inherited from ObjectDetector. It exists to categorize
        GPU-requiring detectors in the class hierarchy and enable the
        CLI to make informed resource-allocation decisions.
    """

    @overload
    @abstractmethod
    def _operate(self, image: Image) -> Image: ...

    @overload
    @abstractmethod
    def _operate(self, image: GridImage) -> GridImage: ...

    @abstractmethod
    def _operate(self, image: Image) -> Image:
        return image
