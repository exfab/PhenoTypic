from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List

import numpy as np

from phenotypic.tools_.typing_ import GpuInputLayer, GpuOutputKind

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from ._object_detector import ObjectDetector


# <<Interface>>
class GpuDetector(ObjectDetector, ABC):
    """Interface ABC for GPU-accelerated object detectors (batched/streaming).

    Subclass GpuDetector when your detection algorithm depends on a GPU
    (e.g., deep-learning foundation models like SAM2 or micro-sam).

    GpuDetector provides a concrete ``_operate`` built from a small set of
    overridable hooks — ``preprocess`` (raw ``input_layer`` array → model-ready
    sample), ``collate`` (samples → batch), ``infer_batch`` (batch → per-sample
    results), and ``_write_object_output`` (result → ``objmap``/``objmask``).
    The single-image notebook path and the batched CLI engine drive the *same*
    hooks, so a detector implemented once runs in both. Capability is declared
    via three fields — ``input_layer`` (``rgb``/``gray``/``detect_mat``),
    ``supports_batching``, and ``output_kind`` (``instance``/``semantic``).

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

            def _infer_one(self, sample):
                # ``sample`` is a preprocessed (H, W, 3) array. Return a uint16
                # labeled objmap (output_kind="instance") or a bool mask
                # (output_kind="semantic"). The base _operate/infer_batch wire
                # this into the image; do NOT override _operate.
                # ... run inference ...
                return objmap

    Notes:
        ``_operate`` is concrete here and should not be overridden. Non-batchable
        subclasses (SAM2, micro-sam) implement just ``_ensure_model_loaded`` +
        ``_infer_one``; the default ``infer_batch`` loops ``_infer_one`` and is
        the sole caller of ``_ensure_model_loaded``. Batchable subclasses
        (Spec 2 foundation models) instead override ``infer_batch`` with a true
        ``(N, C, H, W)`` forward — no engine changes needed. The class also lets
        the CLI make informed GPU resource-allocation decisions.
    """

    # Capability / routing markers — pydantic FIELDS (not ClassVar) so they
    # serialize and round-trip (Spec 1 §4, review S4). Subclasses override the
    # defaults; "instance" keeps existing SAM behavior unchanged.
    input_layer: GpuInputLayer = "rgb"
    supports_batching: bool = False
    output_kind: GpuOutputKind = "instance"

    @abstractmethod
    def _ensure_model_loaded(self) -> None:
        """Build/load the GPU model on first use (idempotent)."""

    def preprocess(self, array: np.ndarray) -> Any:
        """Turn a raw ``input_layer`` array into a model-ready sample (CPU).

        Default: a single-channel 2D layer (``gray``/``detect_mat``) is stacked
        into an ``(H, W, 3)`` block so 3-channel models (SAM/DINO ViT) consume
        it unchanged; ``rgb`` passes through untouched. Subclasses may override
        for model-specific normalization (e.g. uint8 coercion).
        """
        if array.ndim == 2:
            return np.stack([array, array, array], axis=-1)
        return array

    def collate(self, samples: List[Any]) -> Any:
        """Merge per-sample ``preprocess`` outputs into a batch.

        Default returns the list unchanged (consumed by the looped
        ``infer_batch``). Batchable subclasses override to stack into a tensor.
        """
        return samples

    def infer_batch(self, batch: Any) -> List[np.ndarray]:
        """Run inference over a collated batch; return one result per sample.

        Each result is a uint16 labeled map (``output_kind="instance"``) or a
        boolean mask (``output_kind="semantic"``). The default loops
        ``_infer_one`` (correct for ``supports_batching=False``); batchable
        subclasses override with a true ``(N, C, H, W)`` forward.
        """
        self._ensure_model_loaded()
        return [self._infer_one(sample) for sample in batch]

    def _infer_one(self, sample: Any) -> np.ndarray:
        """Run the model on ONE preprocessed sample. Subclasses must implement.

        Returns a uint16 labeled objmap (instance) or a boolean mask (semantic).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _infer_one()"
        )

    def _write_object_output(self, image: "Image", result: np.ndarray) -> None:
        """Write one ``infer_batch`` result onto the image per ``output_kind``.

        - ``instance`` -> ``image.objmap[:]`` (detector-controlled labels).
        - ``semantic`` -> ``image.objmask[:]`` (auto-labels into the shared
          ``objmap`` backend, exactly like a threshold detector; see Spec 1 §8).
        """
        if self.output_kind == "instance":
            image.objmap[:] = result.astype(np.uint16)
        else:  # semantic
            image.objmask[:] = result.astype(bool)

    def _operate(self, image: "Image") -> "Image":
        """Run GPU detection on one image (notebook / single-image path).

        Reads the declared ``input_layer``, preprocesses, runs a one-element
        batch through ``collate`` + ``infer_batch``, and writes the result via
        ``output_kind``. The batched CLI engine drives the same
        ``preprocess``/``collate``/``infer_batch`` methods over many images.
        """
        array = getattr(image, self.input_layer)[:]
        sample = self.preprocess(array)
        batch = self.collate([sample])
        results = self.infer_batch(batch)
        self._write_object_output(image, results[0])
        return image
