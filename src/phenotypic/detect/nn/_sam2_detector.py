"""SAM2 automatic mask generator wrapped as a PhenoTypic object detector."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, PrivateAttr

from phenotypic.abc_ import GpuDetector
from phenotypic.detect.nn._checkpoint_manager import (
    Device,
    Sam2ModelSize,
)
from phenotypic.sdk_.typing_ import GpuInputLayer, TuneSpec


def build_sam2_generator(
    model_size: Sam2ModelSize,
    *,
    device: str,
    points_per_side: int = 32,
    points_per_batch: int = 8,
    pred_iou_thresh: float = 0.7,
    stability_score_thresh: float = 0.92,
    min_mask_region_area: int = 100,
    crop_n_layers: int = 0,
    crop_nms_thresh: float = 0.7,
    crop_overlap_ratio: float = 512 / 1500,
    crop_n_points_downscale_factor: int = 1,
    box_nms_thresh: float = 0.7,
    checkpoint: str | Path | None = None,
    config: str | None = None,
) -> object:
    """Build a ``SAM2AutomaticMaskGenerator`` (shared by SAM2 + DinoSam2).

    Centralises the ``build_sam2`` + generator construction so detectors that
    need SAM2 class-agnostic proposals (``Sam2Detector``,
    ``DinoSam2Detector``) do not duplicate the checkpoint-resolution and
    Hydra-config-prefix logic. There is no public accessor for an existing
    generator, so each detector calls this to rebuild its own.

    The ``crop_*`` arguments expose SAM2's native crop pyramid; they default
    to the upstream ``SAM2AutomaticMaskGenerator`` values, so an un-set call
    reproduces SAM2's stock single-pass behaviour.

    Args:
        model_size: SAM2 variant (``"tiny"`` … ``"large"``).
        device: Resolved torch device string (from ``resolve_device``).
        points_per_side: Point-prompt grid density (``points_per_side ** 2``).
        points_per_batch: Number of point prompts decoded together. Lowering
            this bounds peak full-resolution mask memory without changing the
            prompt grid or segmentation resolution.
        pred_iou_thresh: Minimum predicted-IoU score to keep a mask.
        stability_score_thresh: Minimum mask-stability score.
        min_mask_region_area: Minimum mask area in pixels.
        crop_n_layers: Number of additional crop-pyramid layers
            (SAM2 default ``0`` = single full-image pass).
        crop_nms_thresh: Box-IoU cutoff for NMS between masks from different
            crops (SAM2 default ``0.7``).
        crop_overlap_ratio: Fractional overlap of first-layer crops
            (SAM2 default ``512 / 1500``).
        crop_n_points_downscale_factor: Per-layer point-grid downscale factor
            (SAM2 default ``1``).
        box_nms_thresh: Box-IoU cutoff for NMS between the dense point grid's
            redundant proposals *within* one crop (SAM2 default ``0.7``).
        checkpoint: Optional path to a custom checkpoint.
        config: Optional SAM2 config YAML identifier for a custom checkpoint.

    Returns:
        A configured ``SAM2AutomaticMaskGenerator`` instance.

    Raises:
        ImportError: If the ``sam2`` package is not installed.
    """
    try:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2
    except ImportError:
        raise ImportError(
            "SAM2 proposals require the sam2 package. "
            "Install with: pip install phenotypic[torch]"
        ) from None

    from phenotypic.detect.nn._checkpoint_manager import Sam2CheckpointManager

    mgr = Sam2CheckpointManager()
    if checkpoint is not None:
        ckpt = str(checkpoint)
        cfg = config or mgr.get_config(model_size)
    else:
        ckpt = str(mgr.get_checkpoint(model_size))
        cfg = mgr.get_config(model_size)

    # sam2 1.1.0 registers `pkg://sam2` as the Hydra search root, but the
    # config YAMLs live at `sam2/configs/sam2.1/…`. Prepend `configs/` at the
    # call site so the serialized identifier stays portable.
    if not cfg.startswith("configs/") and not cfg.startswith("/"):
        cfg = f"configs/{cfg}"

    model = build_sam2(cfg, ckpt, device=device, apply_postprocessing=False)
    return SAM2AutomaticMaskGenerator(
        model,
        points_per_side=points_per_side,
        points_per_batch=points_per_batch,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        min_mask_region_area=min_mask_region_area,
        crop_n_layers=crop_n_layers,
        crop_nms_thresh=crop_nms_thresh,
        crop_overlap_ratio=crop_overlap_ratio,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        box_nms_thresh=box_nms_thresh,
        output_mode="uncompressed_rle",
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

    SAM2 was trained on **RGB input**, so ``input_layer`` defaults to
    ``"rgb"`` and preceding classical enhancers (which target ``detect_mat``)
    are ignored by default.  Set ``input_layer="detect_mat"`` (or ``"gray"``)
    to instead feed that layer, which ``_preprocess`` stacks to three channels.

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
        points_per_batch: Number of point prompts decoded together. Lower this
            first if SAM2 runs out of memory; it preserves prompt positions,
            crop resolution, and quality thresholds at the cost of throughput.
            Default 8.
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
        crop_n_layers: Number of additional **crop-pyramid layers** SAM2 runs
            for higher accuracy on large or dense plates.  SAM2's encoder
            resizes the whole image to a fixed **1024x1024 square** -- a
            non-aspect-preserving squash, so a 4:3 plate enters the model as
            ellipses -- and small colonies on a multi-megapixel plate can be
            lost to downsampling.  ``0`` keeps a single full-image pass; the
            ``n``-th added layer (``n`` = 1, 2, ...) re-tiles the *entire*
            image into ``(2 ** n) ** 2`` overlapping crops -- 4 at layer 1,
            16 at layer 2 -- each encoded nearer native resolution, and merges
            them by NMS that prefers masks from smaller crops.  The full-image
            pass is always included, so ``crop_n_layers=1`` costs 5 encoder
            passes and ``2`` costs 21.  Default 1.
        crop_nms_thresh: Box-IoU cutoff for non-maximum suppression between
            masks from different crops -- deduplicates a colony seen in two
            overlapping crops.  Typical range 0.5--0.9.  Default 0.7 (the SAM2
            default).
        crop_overlap_ratio: Fraction of image length by which first-layer
            crops overlap (later layers scale this down).  Set this so the
            overlap exceeds the largest colony diameter, guaranteeing every
            colony appears whole in at least one crop.  Default ``512 / 1500``
            (the SAM2 default).
        crop_n_points_downscale_factor: Point-grid density divisor per crop
            layer -- ``points_per_side`` in layer ``n`` is scaled by
            ``crop_n_points_downscale_factor ** n``.  Default 1 (the SAM2
            default).
        box_nms_thresh: Box-IoU cutoff for non-maximum suppression between
            the dense point grid's redundant proposals *within* one crop
            (distinct from ``crop_nms_thresh``, which deduplicates *across*
            crops).  Typical range 0.5--0.9.  Default 0.7 (the SAM2 default).
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
        input_layer: Image layer fed to the model -- ``"rgb"`` (default; the
            layer SAM2 was trained on), ``"gray"``, or ``"detect_mat"``.
            Single-channel layers are stacked to 3 channels and coerced to
            uint8 by ``_preprocess`` before inference.

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
    points_per_batch: Annotated[int, TuneSpec(tunable=False)] = Field(
        default=8, ge=1
    )
    pred_iou_thresh: float = 0.7
    stability_score_thresh: float = 0.92
    min_mask_region_area: int = 100
    # Native SAM2 crop-pyramid knobs — defaults mirror upstream
    # ``SAM2AutomaticMaskGenerator`` except ``crop_n_layers``, which we
    # engage by default (see docstring for why).
    crop_nms_thresh: Annotated[float, TuneSpec(0.0, 1.0)] = 0.7
    crop_overlap_ratio: Annotated[float, TuneSpec(0.0, 0.5)] = 512 / 1500
    crop_n_points_downscale_factor: Annotated[int, TuneSpec(1, 2)] = 1
    box_nms_thresh: Annotated[float, TuneSpec(0.0, 1.0)] = 0.7
    # 1 crop layer = 5 encoder passes: the always-present full-image pass plus
    # (2 ** 1) ** 2 = 4 crops. Engages SAM2's edge rejection, crop overlap,
    # full-image fallback, and resolution-preferring NMS. ~3.91 -> ~1.9 native
    # px per encoder px on a 4000x3000 plate, at ~5x the inference cost.
    crop_n_layers: Annotated[int, TuneSpec(0, 2)] = 1
    device: Device = "auto"
    checkpoint: str | Path | None = None
    config: str | None = None
    input_layer: GpuInputLayer = "rgb"

    # Lazy SAM2 mask generator — PrivateAttr -> skipped by serialization.
    _generator: object = PrivateAttr(default=None)

    def _ensure_model_loaded(self) -> None:
        """Build the SAM2 mask generator on first use.

        Guards against redundant loads after deserialization where the
        ``_generator`` attribute may not exist.
        """
        if getattr(self, "_generator", None) is not None:
            return

        from phenotypic.detect.nn._checkpoint_manager import resolve_device

        device = resolve_device(self.device)
        self._generator = build_sam2_generator(
            self.model_size,
            device=device,
            points_per_side=self.points_per_side,
            points_per_batch=self.points_per_batch,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            min_mask_region_area=self.min_mask_region_area,
            crop_n_layers=self.crop_n_layers,
            crop_nms_thresh=self.crop_nms_thresh,
            crop_overlap_ratio=self.crop_overlap_ratio,
            crop_n_points_downscale_factor=self.crop_n_points_downscale_factor,
            box_nms_thresh=self.box_nms_thresh,
            checkpoint=self.checkpoint,
            config=self.config,
        )

    def _infer_one(self, sample):
        """Segment colonies in one preprocessed sample via SAM2 AMG.

        Returns a uint16 labeled objmap (largest-first painting preserves
        small-colony identity at overlaps).
        """
        rgb = sample
        masks = self._generator.generate(rgb)  # type: ignore[attr-defined]

        h, w = rgb.shape[:2]
        from phenotypic.detect.nn._sam2_rle import (
            normalize_rle_records,
            paint_rle_records,
        )

        normalize_rle_records(masks, expected_shape=(h, w))
        return paint_rle_records(
            masks,
            (h, w),
            detector_name="SAM2",
            truncate_before_sort=True,
        )


# Expose the class docstring on .apply() for Sphinx autodoc
Sam2Detector.apply.__doc__ = Sam2Detector.__doc__
