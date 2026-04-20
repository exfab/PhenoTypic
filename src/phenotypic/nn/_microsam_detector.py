"""MicroSamDetector -- microscopy-finetuned SAM-based colony detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import GpuDetector
from phenotypic.nn._checkpoint_manager import (
    Device,
    MicroSamCheckpointManager,
    MicroSamModelType,
)


class MicroSamDetector(GpuDetector):
    """Detect colonies using micro-sam's microscopy-finetuned SAM models.

    Run automatic instance segmentation with micro-sam, a toolkit that
    provides Segment Anything Model (SAM) checkpoints finetuned on
    large-scale microscopy datasets. Unlike general-purpose SAM2, micro-sam
    models have been specifically trained on light-microscopy and
    electron-microscopy images, giving them a strong prior for cell and
    colony boundaries at microscopic scales.

    micro-sam wraps the original SAM architecture (SAM1) and exposes an
    ``automatic_instance_segmentation`` function that lays a prompt grid
    over the image, generates candidate masks, merges overlapping
    predictions, and returns a fully labeled integer array -- one label
    per detected object. This means MicroSamDetector produces an
    ``objmap`` directly without the mask-assembly step required by
    :class:`Sam2Detector`.

    Best For:
        * Agar plate images captured under standard brightfield or
          darkfield microscopy illumination.
        * Plates with colonies of varying size and morphology, where
          domain-specific finetuning outperforms general-purpose models.
        * Research workflows that benefit from a pretrained microscopy
          model with no additional training (zero-shot on new plate types).
        * Light-microscopy screening assays (yeast, bacteria, fungi) where
          colony contrast is moderate and classical thresholds struggle.
        * Electron-microscopy organelle segmentation when using the
          ``"vit_b_em_organelles"`` or ``"vit_l_em_organelles"`` models.

    Consider Also:
        * :class:`Sam2Detector` when processing general-purpose or
          non-microscopy images, or when fine-grained control over mask
          generation parameters (``points_per_side``, ``pred_iou_thresh``,
          ``stability_score_thresh``) is needed.
        * :class:`OtsuDetector` for a fast, parameter-free baseline when
          the plate histogram is cleanly bimodal (well-separated colonies).
        * :class:`WatershedDetector` when touching colonies must be split
          using classical morphological methods and GPU hardware is
          unavailable.
        * :class:`HysteresisDetector` when colony intensity varies across
          the plate and a dual-threshold approach captures both bright and
          faint colonies without GPU overhead.

    Args:
        model_type: micro-sam model identifier. Available models:

            * ``"vit_t_lm"`` -- ViT-Tiny, light microscopy (fastest).
            * ``"vit_b_lm"`` -- ViT-Base, light microscopy (default;
              best speed/accuracy trade-off).
            * ``"vit_l_lm"`` -- ViT-Large, light microscopy (highest
              accuracy, most VRAM).
            * ``"vit_b_em_organelles"`` -- ViT-Base, electron microscopy
              organelles.
            * ``"vit_l_em_organelles"`` -- ViT-Large, electron microscopy
              organelles.
            * ``"vit_t"``, ``"vit_b"``, ``"vit_l"``, ``"vit_h"`` -- base
              SAM checkpoints without microscopy finetuning.

        device: PyTorch device for inference. ``"auto"`` (default) probes
            accelerators in priority order: CUDA, Apple MPS, Intel XPU,
            Habana HPU, then raises if none found. Pass ``"cpu"`` to force
            CPU inference (very slow). Any valid PyTorch device string is
            accepted.

    Returns:
        Image: Input image with ``objmask`` set to a binary colony mask
        and ``objmap`` set to the labeled instance segmentation produced
        by micro-sam. Each unique positive integer in ``objmap``
        corresponds to a single detected colony.

    Raises:
        ImportError: If ``micro_sam`` is not installed. ``micro_sam`` is
            only available on conda-forge (not PyPI), so it is not
            included in any ``phenotypic`` extra. See the "Enabling
            micro_sam" section of
            ``docs/source/how_to/pages/gpu_detection_setup.md`` for a
            pixi-based recipe that installs ``phenotypic`` and
            ``micro_sam`` together in a single environment.
        RuntimeError: If ``device="auto"`` and no GPU/accelerator is
            available.

    Notes:
        **Lazy model loading.** The SAM model is not loaded until the
        first call to :meth:`apply`. This enables fast construction,
        serialization round-trips (``to_json`` / ``from_json``), and
        parameter inspection without allocating GPU memory. After
        deserialization the internal ``_predictor`` is rebuilt
        transparently on the next :meth:`apply` call.

        **RGB input.** MicroSamDetector reads ``image.rgb[:]`` directly
        (not ``detect_mat``). SAM models were trained on colour images;
        the classical enhancement pipeline targets thresholding, not
        foundation models. If the image has higher-than-8-bit dynamic
        range it is rescaled to uint8 before inference.

        **Checkpoint caching.** micro-sam manages its own cache via
        ``platformdirs`` (respects the ``MICROSAM_CACHEDIR`` environment
        variable). Use ``python -m phenotypic.nn download --model-type
        microsam`` to pre-download checkpoints on login nodes before
        submitting SLURM jobs on compute nodes without internet access.

    References:
        [1] A. Archit *et al.*, "Segment anything for microscopy,"
        *Nature Methods*, 2024. doi:10.1038/s41592-024-02580-4

    See Also:
        :class:`Sam2Detector`
            General-purpose SAM2 detector with configurable mask
            generation parameters.
        :doc:`/how_to/pages/gpu_detection_setup`
            Installation, checkpoint management, and SLURM deployment.
        :doc:`/explanation/detection_strategies_compared`
            Comparison of all available detection strategies.

    Examples:
        Detect colonies on a synthetic yeast plate:

        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.nn import MicroSamDetector
        >>> image = load_synth_yeast_plate()
        >>> detector = MicroSamDetector(model_type="vit_b_lm")
        >>> detected = detector.apply(image)  # doctest: +SKIP
        >>> detected.num_objects > 0  # doctest: +SKIP
        True

        Use MicroSamDetector in a pipeline with post-processing:

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.data import load_synth_yeast_plate
        >>> from phenotypic.nn import MicroSamDetector
        >>> pipeline = ImagePipeline(
        ...     ops=[MicroSamDetector(model_type="vit_b_lm", device="cpu")]
        ... )
        >>> json_str = pipeline.to_json()  # serializes without torch
        >>> restored = ImagePipeline.from_json(json_str)
        >>> image = load_synth_yeast_plate()
        >>> result = restored.operate([image])[0]  # doctest: +SKIP
        >>> result.num_objects > 0  # doctest: +SKIP
        True
    """

    def __init__(
        self,
        model_type: MicroSamModelType = "vit_b_lm",
        device: Device = "auto",
    ):
        super().__init__()
        if model_type not in MicroSamCheckpointManager.MODELS:
            raise ValueError(
                f"Unknown model_type={model_type!r}. "
                f"Choose from: {list(MicroSamCheckpointManager.MODELS)}"
            )
        self.model_type = model_type
        self.device = device
        self._predictor = None  # _ prefix -> skipped by serialization

    def _ensure_model_loaded(self) -> None:
        """Load the micro-sam predictor on first use."""
        if getattr(self, "_predictor", None) is not None:
            return
        try:
            from micro_sam.util import get_sam_model
        except ImportError:
            raise ImportError(
                "MicroSamDetector requires the `micro_sam` package, "
                "which is conda-only (not available on PyPI). See the "
                "'Enabling micro_sam' section of "
                "docs/source/how_to/pages/gpu_detection_setup.md for a "
                "pixi-based recipe that installs phenotypic and "
                "micro_sam together."
            ) from None

        from phenotypic.nn._checkpoint_manager import resolve_device

        self._predictor = get_sam_model(
            model_type=self.model_type,
            device=resolve_device(self.device),
        )

    def _operate(self, image: Image) -> Image:
        """Segment colonies via micro-sam automatic instance segmentation.

        Reads the RGB image, ensures uint8 range, runs micro-sam's
        ``automatic_instance_segmentation``, and writes the resulting
        labeled array to ``image.objmap`` and the derived binary mask
        to ``image.objmask``.

        Args:
            image: Input plate image with RGB data available.

        Returns:
            Image with ``objmask`` and ``objmap`` populated.
        """
        import numpy as np

        self._ensure_model_loaded()

        from micro_sam.automatic_segmentation import (
            automatic_instance_segmentation,
        )

        rgb = image.rgb[:]
        if rgb.dtype != np.uint8:
            max_val = rgb.max()
            if max_val > 0:
                rgb = (rgb / max_val * 255).astype(np.uint8)
            else:
                rgb = np.zeros(rgb.shape, dtype=np.uint8)

        labeled = automatic_instance_segmentation(self._predictor, rgb)
        objmap = labeled.astype(np.uint16)

        image.objmask = objmap > 0
        image.objmap[:] = objmap
        return image


# Propagate the _operate docstring to the public apply method
MicroSamDetector.apply.__doc__ = MicroSamDetector._operate.__doc__
