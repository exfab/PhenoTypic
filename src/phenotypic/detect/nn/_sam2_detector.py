"""SAM2 automatic mask generator wrapped as a PhenoTypic object detector."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import PrivateAttr

if TYPE_CHECKING:
    from phenotypic._core._image import Image

from phenotypic.abc_ import GpuDetector
from phenotypic.detect.nn._checkpoint_manager import (
    Device,
    Sam2ModelSize,
)


class Sam2Detector(GpuDetector):
    """Detect colonies using Meta's SAM2 foundation model.

    Run SAM2's automatic mask generator over the full-colour plate image
    to segment every visually distinct region, then assemble the predicted
    masks into a labelled object map.  SAM2 lays a dense grid of prompt
    points (``points_per_side x points_per_side``) across the image,
    predicts candidate masks at each point, filters by quality scores, and
    applies non-maximum suppression to produce a final set of
    non-overlapping instance masks.

    Because SAM2 operates on **RGB input** (not ``detect_mat``), classical
    enhancement operations placed before this detector in a pipeline are
    ignored.  The model sees the original colour image regardless of any
    preceding enhancers.

    The model is loaded lazily on the first call to
    :meth:`~phenotypic.abc_.ImageOperation.apply` (not during construction),
    so a ``Sam2Detector`` can be serialised, round-tripped through JSON,
    and inspected without a GPU or PyTorch installed.

    Args:
        model_size: SAM2 model variant.  ``"tiny"`` (~39 MB) is fastest
            and sufficient for most colony plates; ``"large"`` (~900 MB)
            offers the highest mask quality at the cost of VRAM and
            latency.  Typical choice: ``"tiny"`` or ``"small"`` for
            routine screening, ``"large"`` for publication figures.
        points_per_side: Number of point prompts along each image axis.
            The model evaluates ``points_per_side ** 2`` candidate
            locations.  Increase for dense plates with many small
            colonies; decrease to speed up inference on sparse plates.
            Typical range: 16--64.  Default 32 (1024 points).
        pred_iou_thresh: Minimum predicted IoU score for a mask to be
            kept.  SAM2 self-estimates mask quality; masks below this
            threshold are discarded.  Raise to keep only high-confidence
            detections; lower to catch faint or ambiguous colonies.
            Typical range: 0.5--0.95.  Default 0.7.
        stability_score_thresh: Minimum mask stability score.  Measures
            how much the mask changes under small perturbations of the
            logit threshold.  Higher values keep only masks with crisp
            boundaries; lower values retain masks with soft edges.
            Typical range: 0.8--0.98.  Default 0.92.
        min_mask_region_area: Minimum mask area in pixels.  Masks smaller
            than this are removed after generation.  Increase to suppress
            agar texture, dust, and other small artefacts that SAM2
            segments as objects.  Typical range: 50--500 depending on
            image resolution.  Default 100.
        device: PyTorch device for inference.  ``"auto"`` probes
            accelerators in priority order (CUDA, MPS, XPU, HPU) and
            raises ``RuntimeError`` if none is found.  Pass ``"cpu"``
            to force CPU inference (very slow).
        checkpoint: Path to a custom SAM2 checkpoint file.  When *None*
            (default), the standard checkpoint for *model_size* is
            downloaded automatically to the ``torch.hub`` cache.  Use
            this to load a finetuned model or an offline-cached file.
        config: SAM2 config YAML identifier for a custom checkpoint.
            Required when *checkpoint* points to a non-standard model.
            When *None* and *checkpoint* is set, the config for
            *model_size* is used as a fallback.

    Returns:
        Image: Input image with ``objmask`` set to a binary colony mask
        and ``objmap`` set to a labelled instance map where each colony
        receives a unique integer label (1, 2, ..., *N*).  Masks are
        painted largest-first so smaller colonies overwrite at overlaps,
        preserving small-colony identity.

    Raises:
        ImportError: If ``sam2`` or ``torch`` is not installed.  Install
            with ``pip install phenotypic[torch]``.
        RuntimeError: If ``device="auto"`` and no GPU/accelerator is
            available.

    Best For:
        * Plates where colony appearance varies widely (mixed species,
          pigmented mutants, translucent microcolonies) and no single
          intensity threshold captures all objects.
        * Complex backgrounds (textured agar, condensation, scratches)
          that confuse classical thresholding methods.
        * Exploratory analysis on new plate types before investing time
          in tuning classical detector parameters.
        * Dense plates with heterogeneous colony morphologies where
          watershed over-segments or under-segments.

    Consider Also:
        * :class:`~phenotypic.detect.OtsuDetector` for well-lit plates
          with a clean bimodal histogram -- faster and requires no GPU.
        * :class:`~phenotypic.detect.WatershedDetector` when touching
          colonies of similar appearance must be split and a GPU is
          unavailable.
        * :class:`~phenotypic.detect.nn.MicroSamDetector` for microscopy images
          where a domain-finetuned SAM model may outperform the general-
          purpose SAM2 checkpoint.
        * :class:`~phenotypic.detect.HysteresisDetector` when colony
          intensity varies but a dual-threshold approach suffices.

    References:
        [1] N. Ravi et al., "SAM 2: Segment Anything in Images and
        Videos," *arXiv:2408.00714*, 2024.

    See Also:
        :class:`~phenotypic.detect.nn.MicroSamDetector`
            Microscopy-finetuned SAM for domain-specific segmentation.
        :doc:`/how_to/pages/gpu_detection_setup`
            Installation, checkpoint management, and SLURM deployment.
        :doc:`/explanation/detection_strategies_compared`
            Comparison of all detection strategies.

    Examples:
        Construct a detector and inspect its default parameters (no GPU
        or ``sam2`` package required):

        >>> from phenotypic.detect.nn import Sam2Detector
        >>> det = Sam2Detector(model_size="tiny", points_per_side=32)
        >>> det.model_size
        'tiny'

        Build a pipeline that uses SAM2 for detection and serialise it
        to JSON (round-trips without GPU dependencies):

        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.detect.nn import Sam2Detector
        >>> pipe = ImagePipeline(ops=[Sam2Detector(model_size="small")])
        >>> json_str = pipe.to_json()
        >>> pipe2 = ImagePipeline.from_json(json_str)
        >>> type(pipe2.get_ops()["Sam2Detector"])
        <class 'phenotypic.detect.nn._sam2_detector.Sam2Detector'>
    """

    model_size: Sam2ModelSize = "tiny"
    points_per_side: int = 32
    pred_iou_thresh: float = 0.7
    stability_score_thresh: float = 0.92
    min_mask_region_area: int = 100
    device: Device = "auto"
    checkpoint: str | Path | None = None
    config: str | None = None

    # Lazy SAM2 mask generator — PrivateAttr -> skipped by serialization.
    _generator: object = PrivateAttr(default=None)

    def _ensure_model_loaded(self) -> None:
        """Build the SAM2 mask generator on first use.

        Guards against redundant loads after deserialization where the
        ``_generator`` attribute may not exist.
        """
        if getattr(self, "_generator", None) is not None:
            return

        try:
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError:
            raise ImportError(
                "Sam2Detector requires the sam2 package. "
                "Install with: pip install phenotypic[torch]"
            ) from None

        from phenotypic.detect.nn._checkpoint_manager import (
            Sam2CheckpointManager,
            resolve_device,
        )

        device = resolve_device(self.device)
        mgr = Sam2CheckpointManager()

        if self.checkpoint is not None:
            ckpt = str(self.checkpoint)
            cfg = self.config or mgr.get_config(self.model_size)
        else:
            ckpt = str(mgr.get_checkpoint(self.model_size))
            cfg = mgr.get_config(self.model_size)

        # sam2 1.1.0 registers `pkg://sam2` as the Hydra search root, but the
        # config YAMLs live at `sam2/configs/sam2.1/…`. Prepend `configs/` at
        # the call site so the serialized identifier (used in to_json/from_json)
        # stays portable across upstream layout changes.
        if not cfg.startswith("configs/") and not cfg.startswith("/"):
            cfg = f"configs/{cfg}"

        model = build_sam2(cfg, ckpt, device=device, apply_postprocessing=False)
        self._generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=self.points_per_side,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_region_area,
        )

    def _operate(self, image: Image) -> Image:
        """Segment colonies via SAM2 automatic mask generation.

        Reads the full-colour RGB image, runs SAM2's automatic mask
        generator, and paints the resulting instance masks onto a
        uint16 object map.  Masks are sorted largest-first so that
        smaller colonies overwrite at overlaps, preserving their
        identity in the final label map.

        Args:
            image: Input plate image.  Must have RGB data available
                via ``image.rgb[:]``.

        Returns:
            Image with ``objmask`` and ``objmap`` populated.
        """
        import numpy as np

        self._ensure_model_loaded()

        rgb = image.rgb[:]
        if rgb.dtype != np.uint8:
            max_val = rgb.max()
            if max_val > 0:
                rgb = (rgb / max_val * 255).astype(np.uint8)
            else:
                rgb = np.zeros(rgb.shape, dtype=np.uint8)

        masks = self._generator.generate(rgb)  # type: ignore[attr-defined]

        h, w = image.shape[:2]
        objmap = np.zeros((h, w), dtype=np.uint16)

        if masks:
            max_labels = int(np.iinfo(np.uint16).max)
            if len(masks) > max_labels:
                import warnings

                warnings.warn(
                    f"SAM2 generated {len(masks)} masks, exceeding uint16 "
                    f"range. Only the first {max_labels} will be labeled.",
                    UserWarning,
                    stacklevel=2,
                )
                masks = masks[:max_labels]

            # Sort largest-first; paint in order so smaller colonies
            # overwrite at overlaps, preserving small-colony identity.
            masks = sorted(masks, key=lambda m: m["area"], reverse=True)
            for idx, m in enumerate(masks, start=1):
                objmap[m["segmentation"]] = idx

        image.objmask = objmap > 0
        image.objmap[:] = objmap
        return image


# Expose the class docstring on .apply() for Sphinx autodoc
Sam2Detector.apply.__doc__ = Sam2Detector._operate.__doc__
