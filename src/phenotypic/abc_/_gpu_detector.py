from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated, Any, List, Literal

import numpy as np

from phenotypic.sdk_.typing_ import GpuInputLayer, GpuOutputKind, TuneSpec

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from ._object_detector import ObjectDetector


_UINT8_CONVERSION_CHUNK_BYTES = 16 * 1024 * 1024


def _conversion_rows(array: np.ndarray) -> int:
    """Return a row count that bounds float64 conversion temporaries."""
    values_per_row = int(np.prod(array.shape[1:], dtype=np.int64))
    bytes_per_row = max(1, values_per_row * np.dtype(np.float64).itemsize)
    return max(1, _UINT8_CONVERSION_CHUNK_BYTES // bytes_per_row)


def _scale_to_uint8(
    array: np.ndarray,
    scaling: Literal["dtype_range", "image_max"],
) -> np.ndarray:
    """Scale a non-uint8 array into uint8 using bounded row chunks."""
    output = np.empty(array.shape, dtype=np.uint8)
    if scaling == "image_max":
        scale_max = array.max()
    elif np.issubdtype(array.dtype, np.integer):
        scale_max = np.iinfo(array.dtype).max
    else:
        scale_max = 1.0

    if scale_max <= 0:
        output.fill(0)
        return output

    rows_per_chunk = _conversion_rows(array)
    for row_start in range(0, array.shape[0], rows_per_chunk):
        row_stop = min(row_start + rows_per_chunk, array.shape[0])
        chunk = array[row_start:row_stop]
        if scaling == "image_max":
            # Keep this expression identical to the legacy whole-array path.
            output[row_start:row_stop] = (
                chunk / scale_max * 255
            ).astype(np.uint8)
        else:
            output[row_start:row_stop] = (
                np.clip(chunk, 0, scale_max) / scale_max * 255
            ).astype(np.uint8)
    return output


# <<Interface>>
class GpuDetector(ObjectDetector, ABC):
    """Interface ABC for GPU-accelerated object detectors (batched/streaming).

    Subclass GpuDetector when your detection algorithm depends on a GPU
    (e.g., deep-learning foundation models like SAM2 or micro-sam).

    GpuDetector provides a concrete ``_operate`` built from a small set of
    protected, overridable hooks — ``_preprocess`` (raw ``input_layer`` array →
    model-ready sample), ``_collate`` (samples → batch), ``_infer_batch``
    (batch → per-sample results), and ``_write_object_output`` (result →
    ``objmap``/``objmask``). The single-image notebook path and the batched CLI
    engine drive the *same* hooks, so a detector implemented once runs in both.
    Capability is declared via three fields — ``input_layer``
    (``rgb``/``gray``/``detect_mat``; defaults to the layer the model was
    trained on), ``supports_batching``, and ``output_kind``
    (``instance``/``semantic``).

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
                # ``sample`` is a preprocessed (H, W, 3) uint8 array. Return a
                # uint16 labeled objmap (output_kind="instance") or a bool mask
                # (output_kind="semantic"). The base _operate/_infer_batch wire
                # this into the image; do NOT override _operate.
                # ... run inference ...
                return objmap

    Notes:
        ``_operate`` is concrete here and should not be overridden. Non-batchable
        subclasses (SAM2, micro-sam) implement just ``_ensure_model_loaded`` +
        ``_infer_one``; the default ``_infer_batch`` loops ``_infer_one`` and is
        the sole caller of ``_ensure_model_loaded``. Batchable subclasses
        (Spec 2 foundation models) instead override ``_infer_batch`` with a true
        ``(N, C, H, W)`` forward — no engine changes needed. The class also lets
        the CLI make informed GPU resource-allocation decisions.
    """

    # Capability / routing markers — pydantic FIELDS (not ClassVar) so they
    # serialize and round-trip (Spec 1 §4, review S4). Subclasses override the
    # defaults; "instance" keeps existing SAM behavior unchanged.
    input_layer: GpuInputLayer = "rgb"
    input_scaling: Annotated[
        Literal["dtype_range", "image_max"], TuneSpec(tunable=False)
    ] = "image_max"
    supports_batching: bool = False
    output_kind: GpuOutputKind = "instance"

    # Post-inference cleanup: zero the background instance BEFORE relabeling. A
    # class-agnostic segmenter (SAM2 etc.) can emit the plate background as a
    # positive-labelled mask framing the image; left in place it survives into
    # measurement and bridges every colony it touches when ``relabel`` binarizes
    # ``objmap > 0``, collapsing all instances into one blob. Zeroing the
    # border-plurality label first removes the bridge (see
    # ``ObjectMap.drop_frame_background``). ``instance`` output only.
    drop_frame_background: bool = True
    # Post-inference cleanup: split a single instance label that spans spatially
    # disconnected blobs into separate instances by connected components. A SAM
    # mask (or a tile-merged objmap) can paint one label across distant regions;
    # relabeling by connectivity gives each connected region its own id. Binary
    # connected-components, so two *touching* distinct labels merge into one —
    # which is why the background is dropped first, above.
    # ``instance`` output only — ``semantic`` already auto-labels by connectivity.
    split_disconnected_labels: bool = True
    # Connectivity for the relabel (1 = 4-neighbour, 2 = 8-neighbour). Structural,
    # never tuned (TuneSpec(tunable=False) satisfies the annotation-coverage gate).
    connectivity: Annotated[int, TuneSpec(tunable=False)] = 2

    @abstractmethod
    def _ensure_model_loaded(self) -> None:
        """Build/load the GPU model on first use (idempotent)."""

    def _preprocess(self, array: np.ndarray) -> Any:
        """Turn a raw ``input_layer`` array into a model-ready ``uint8`` sample.

        Non-uint8 inputs are converted in bounded row chunks. ``dtype_range``
        maps integer dtype limits (for example, uint16 0..65535) or normalized
        float 0..1 to 0..255; ``image_max`` retains the legacy per-image maximum
        normalization. ``dtype_range`` values outside its range are clipped. A
        2D layer is converted before it is stacked into an ``(H, W, 3)`` block,
        avoiding three-channel conversion temporaries. An already-uint8 3D array
        passes through without a copy. Subclasses rarely need to override this
        method.
        """
        if array.dtype != np.uint8:
            array = _scale_to_uint8(array, self.input_scaling)
        if array.ndim == 2:
            array = np.stack([array, array, array], axis=-1)
        return array

    def _collate(self, samples: List[Any]) -> Any:
        """Merge per-sample ``_preprocess`` outputs into a batch.

        Default returns the list unchanged (consumed by the looped
        ``_infer_batch``). Batchable subclasses override to stack into a tensor.
        """
        return samples

    def _infer_batch(self, batch: Any) -> List[np.ndarray]:
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
          When ``drop_frame_background`` is set, the border-plurality background
          label is zeroed first. When ``split_disconnected_labels`` is set, the
          objmap is then relabeled by connected components (``connectivity``) so
          one label spanning spatially disconnected blobs becomes separate
          instances -- order matters: dropping the background before the relabel
          stops it from bridging every colony into one blob.
        - ``semantic`` -> ``image.objmask[:]`` (auto-labels into the shared
          ``objmap`` backend, exactly like a threshold detector; see Spec 1 §8).
          Already connectivity-labeled, so ``split_disconnected_labels`` is a
          no-op here.
        """
        if self.output_kind == "instance":
            image.objmap[:] = result.astype(np.uint16)
            if self.drop_frame_background:
                image.objmap.drop_frame_background()
            if self.split_disconnected_labels:
                image.objmap.relabel(connectivity=self.connectivity)
        else:  # semantic
            image.objmask[:] = result.astype(bool)

    def _operate(self, image: "Image") -> "Image":
        """Run GPU detection on one image (notebook / single-image path).

        Reads the declared ``input_layer``, preprocesses, runs a one-element
        batch through ``_collate`` + ``_infer_batch``, and writes the result via
        ``output_kind``. The batched CLI engine drives the same
        ``_preprocess``/``_collate``/``_infer_batch`` methods over many images.
        """
        array = getattr(image, self.input_layer)[:]
        sample = self._preprocess(array)
        batch = self._collate([sample])
        results = self._infer_batch(batch)
        self._write_object_output(image, results[0])
        return image
