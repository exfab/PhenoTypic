"""SAM3 text-prompted, true-batch instance detector (Spec 2a, Tasks 4-5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, List

from pydantic import PrivateAttr

from phenotypic.abc_ import GpuDetector
from phenotypic.detect.nn._checkpoint_manager import Device
from phenotypic.tools_.typing_ import GpuInputLayer, GpuOutputKind, TuneSpec

if TYPE_CHECKING:
    import numpy as np


# ---------------------------------------------------------------------------
# Fixed geometric tiling (D-tiling: no grid awareness — a GpuDetector runs
# before GridFinder and only ever sees a raw input_layer array).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Tile:
    """One axis-aligned crop rectangle in full-image coordinates.

    Attributes:
        y0: Top row (inclusive).
        x0: Left column (inclusive).
        y1: Bottom row (exclusive).
        x1: Right column (exclusive).
    """

    y0: int
    x0: int
    y1: int
    x1: int

    @property
    def h(self) -> int:
        """Tile height in pixels (``y1 - y0``)."""
        return self.y1 - self.y0

    @property
    def w(self) -> int:
        """Tile width in pixels (``x1 - x0``)."""
        return self.x1 - self.x0


def _tile_starts(extent: int, tile_px: int, stride: int) -> list[int]:
    """Return tile start offsets along one axis, covering ``[0, extent)``.

    The final start is clamped so the last tile ends exactly at ``extent``
    (overlapping the previous tile rather than spilling past the edge).

    Args:
        extent: Axis length in pixels.
        tile_px: Nominal tile size along this axis.
        stride: Step between consecutive tile starts (``tile_px`` minus overlap).

    Returns:
        Sorted, de-duplicated list of start offsets.
    """
    if extent <= tile_px:
        return [0]
    starts: list[int] = []
    pos = 0
    last_start = extent - tile_px
    while pos < last_start:
        starts.append(pos)
        pos += stride
    starts.append(last_start)
    # De-dup while preserving order (the clamp can coincide with a step).
    seen: set[int] = set()
    return [s for s in starts if not (s in seen or seen.add(s))]


def _plan_tiles(
    shape: tuple[int, int], tile_px: int, overlap: float
) -> list[_Tile]:
    """Plan fixed ~``tile_px`` tiles with fractional ``overlap`` over an image.

    The union of the returned tiles always covers the full image; tiles never
    exceed ``tile_px`` on either axis and never spill past the image bounds.
    An image that already fits one tile yields a single full-image tile.

    Args:
        shape: ``(H, W)`` of the full image.
        tile_px: Nominal tile size in pixels (SAM3 runs at 1008 px internally).
        overlap: Fractional overlap between neighbouring tiles, in ``[0, 1)``.

    Returns:
        List of :class:`_Tile` rectangles in full-image coordinates.
    """
    h, w = int(shape[0]), int(shape[1])
    stride = max(1, int(round(tile_px * (1.0 - overlap))))
    tiles: list[_Tile] = []
    for y0 in _tile_starts(h, tile_px, stride):
        for x0 in _tile_starts(w, tile_px, stride):
            y1 = min(y0 + tile_px, h)
            x1 = min(x0 + tile_px, w)
            tiles.append(_Tile(y0, x0, y1, x1))
    return tiles


def _iou(mask_a: "np.ndarray", mask_b: "np.ndarray") -> float:
    """Intersection-over-union of two boolean masks (0.0 if both empty)."""
    inter = int((mask_a & mask_b).sum())
    if inter == 0:
        return 0.0
    union = int((mask_a | mask_b).sum())
    return inter / union if union else 0.0


def _merge_tiles_iou_nms(
    objmaps: List["np.ndarray"], iou_thresh: float
) -> "np.ndarray":
    """Greedy IoU-NMS merge of per-tile objmaps into one contiguous objmap.

    Each input objmap is already offset into full-image coordinates (same
    shape). Instances are collected across all tiles, sorted largest-first,
    and greedily kept unless they overlap an already-kept instance by more than
    ``iou_thresh`` (a cross-tile duplicate). Survivors are relabelled ``1..N``
    largest-first so smaller colonies overwrite at overlaps, preserving
    small-colony identity (mirrors ``Sam2Detector``'s painting order).

    Args:
        objmaps: Per-tile uint16 objmaps, each in full-image coordinates.
        iou_thresh: IoU above which two instances are treated as duplicates.

    Returns:
        A single uint16 objmap with contiguous labels ``1..N``.
    """
    import numpy as np

    if not objmaps:
        raise ValueError("_merge_tiles_iou_nms requires at least one objmap")
    shape = objmaps[0].shape

    masks: list[np.ndarray] = []
    for objmap in objmaps:
        for label in np.unique(objmap):
            if label == 0:
                continue
            masks.append(objmap == label)
    if not masks:
        return np.zeros(shape, dtype=np.uint16)

    masks.sort(key=lambda m: int(m.sum()), reverse=True)
    kept: list[np.ndarray] = []
    for cand in masks:
        if any(_iou(cand, k) > iou_thresh for k in kept):
            continue
        kept.append(cand)

    max_labels = int(np.iinfo(np.uint16).max)
    if len(kept) > max_labels:
        import warnings

        warnings.warn(
            f"SAM3 merged {len(kept)} instances, exceeding uint16 range. "
            f"Only the first {max_labels} (largest) will be labeled.",
            UserWarning,
            stacklevel=2,
        )
        kept = kept[:max_labels]

    objmap = np.zeros(shape, dtype=np.uint16)
    for idx, mask in enumerate(kept, start=1):
        objmap[mask] = idx
    return objmap


class Sam3Detector(GpuDetector):
    """Detect colonies with Meta's SAM3 text-prompted foundation model.

    SAM3 segments every region matching a short **text prompt** (default
    ``"colony"``) in one true ``(N, C, H, W)`` batched forward pass, then
    assembles the predicted instance masks into a labelled object map.  Unlike
    SAM2's dense point grid, SAM3 has a single checkpoint and is prompted by
    free text, so the only "knob" describing *what* to find is the
    :attr:`prompt`.

    Because SAM3 operates on **RGB input** (not ``detect_mat``), classical
    enhancement operations placed before this detector in a pipeline are
    ignored — the model sees the original colour image regardless.

    The model and processor are loaded lazily on the first call to
    :meth:`~phenotypic.abc_.ImageOperation.apply` (not during construction), so
    a ``Sam3Detector`` can be serialised, round-tripped through JSON, and
    inspected without a GPU, PyTorch, or ``transformers`` installed.

    **Gated weights.** SAM3 weights (~3.45 GB) are gated on Hugging Face under
    the SAM License.  Accept the gate at ``https://huggingface.co/facebook/sam3``
    and authenticate once (``uv run hf auth login`` or ``HF_TOKEN``); see
    :doc:`/how_to/pages/gpu_detection_setup`.

    **Dense plates.** SAM3 caps at 200 instances per forward and runs at
    1008 px internally, so dense plates are tiled (fixed ~:attr:`tile_px`
    tiles with :attr:`tile_overlap`), each tile inferred and offset back to
    full coordinates, then merged across tiles by IoU-NMS.

    Args:
        prompt: Free-text description of what to segment.  Override per run
            (e.g. ``"bacterial colony"``, ``"yeast colony"``).  SAM3 has no
            prompt-free "segment everything" mode — a prompt is required.
        score_thresh: Minimum instance confidence to keep.  Raise to keep only
            high-confidence detections; lower to catch faint colonies.
            Typical range 0.3--0.7.  Default 0.5.
        mask_threshold: Probability cutoff binarising each soft mask.  Default
            0.5.
        min_mask_region_area: Minimum mask area in pixels; smaller masks are
            dropped after generation (suppresses agar texture / dust).
            Default 100 (matches ``Sam2Detector``).
        tile_px: Nominal tile size in pixels for dense-plate tiling; images
            that fit one tile run un-tiled.  Default 1008 (SAM3's internal
            resolution).
        tile_overlap: Fractional overlap between neighbouring tiles.  Default
            0.15.
        max_instances_per_tile: SAM3's hard 200-instance-per-forward cap;
            structural, not tuned.
        device: PyTorch device for inference.  ``"auto"`` probes accelerators
            and raises ``RuntimeError`` if none is found.

    Returns:
        Image: Input image with ``objmap`` set to a labelled instance map
        (each colony a unique integer label) and ``objmask`` to the derived
        binary mask.  Masks are painted largest-first so smaller colonies
        keep their identity at overlaps.

    Raises:
        ImportError: If ``transformers`` / ``torch`` are not installed.  Install
            with ``pip install phenotypic[foundation]``.
        RuntimeError: If ``device="auto"`` and no accelerator is available, or
            the gated weights cannot be downloaded (access not granted / no
            token).

    Best For:
        * Plates of mixed species / morphologies where one text prompt
          captures the target better than a tuned intensity threshold.
        * Heterogeneous backgrounds that confuse classical thresholding.

    Consider Also:
        * :class:`~phenotypic.detect.nn.Sam2Detector` for an ungated,
          prompt-free automatic mask generator.
        * :class:`~phenotypic.detect.nn.DinoSam2Detector` for an ungated
          training-free SAM2-proposals + DINOv2-scoring instance detector.

    References:
        [1] Meta AI, "SAM 3: Segment Anything with Concepts," 2025.

    See Also:
        :doc:`/how_to/pages/gpu_detection_setup`
            Installation, gated-weight acceptance, and SLURM deployment.

    Examples:
        Construct a detector and inspect its prompt (no GPU or weights
        required):

        >>> from phenotypic.detect.nn import Sam3Detector
        >>> det = Sam3Detector(prompt="yeast colony")
        >>> det.prompt
        'yeast colony'

        Build a pipeline and serialise it to JSON (round-trips without GPU
        dependencies):

        >>> from phenotypic import ImagePipeline
        >>> pipe = ImagePipeline(ops=[Sam3Detector(prompt="colony")])
        >>> restored = ImagePipeline.from_json(pipe.to_json())
        >>> type(restored.get_ops()["Sam3Detector"])
        <class 'phenotypic.detect.nn._sam3_detector.Sam3Detector'>
    """

    # Capabilities — SAM3 is text-prompted, instance-native, true-batch.
    input_layer: GpuInputLayer = "rgb"
    output_kind: GpuOutputKind = "instance"
    supports_batching: bool = True

    # ``prompt`` is parameterised free text — a plain str (no TuneSpec/Enum).
    prompt: str = "colony"
    score_thresh: Annotated[float, TuneSpec(0.0, 1.0)] = 0.5
    mask_threshold: Annotated[float, TuneSpec(0.0, 1.0)] = 0.5
    min_mask_region_area: Annotated[int, TuneSpec(0, 500)] = 100

    # Tiling (Task 5) — fixed geometric tiles, no grid awareness.
    tile_px: Annotated[int, TuneSpec(512, 2048)] = 1008
    tile_overlap: Annotated[float, TuneSpec(0.0, 0.4)] = 0.15
    # SAM3's hard num_queries cap per forward — structural, never tuned.
    max_instances_per_tile: Annotated[int, TuneSpec(tunable=False)] = 200

    device: Device = "auto"

    # Lazy runtime state — PrivateAttr → skipped by serialization.
    _model: Any = PrivateAttr(default=None)
    _processor: Any = PrivateAttr(default=None)
    _device: Any = PrivateAttr(default=None)

    def _ensure_model_loaded(self) -> None:
        """Build the SAM3 model + processor on first use (idempotent).

        Lazy-imports ``transformers``/``torch`` so the detector constructs and
        serialises without them. Routes the gated weight pull through
        ``Sam3CheckpointManager`` (honouring ``PHENOTYPIC_ACCEPT_MODEL_LICENSE``
        and ``HF_HUB_OFFLINE``).
        """
        if getattr(self, "_model", None) is not None:
            return

        try:
            from transformers import Sam3Model, Sam3Processor
        except ImportError:
            raise ImportError(
                "Sam3Detector requires transformers (>=5.2.0). "
                "Install with: pip install phenotypic[foundation]"
            ) from None

        from phenotypic.detect.nn._checkpoint_manager import (
            Sam3CheckpointManager,
            resolve_device,
        )

        self._device = resolve_device(self.device)
        repo_id = Sam3CheckpointManager.repo_id
        self._model = Sam3Model.from_pretrained(repo_id).to(self._device)
        self._processor = Sam3Processor.from_pretrained(repo_id)

    # ------------------------------------------------------------------
    # uint8 coercion (mirror Sam2Detector._infer_one normalisation)
    # ------------------------------------------------------------------

    @staticmethod
    def _to_uint8(sample: "np.ndarray") -> "np.ndarray":
        """Coerce an ``(H, W, 3)`` sample to uint8 for the SAM3 processor."""
        import numpy as np

        rgb = sample
        if rgb.dtype != np.uint8:
            max_val = rgb.max()
            if max_val > 0:
                rgb = (rgb / max_val * 255).astype(np.uint8)
            else:
                rgb = np.zeros(rgb.shape, dtype=np.uint8)
        return rgb

    # ------------------------------------------------------------------
    # Per-tile forward + objmap painting
    # ------------------------------------------------------------------

    def _forward_tiles(self, images: List["np.ndarray"]) -> List["np.ndarray"]:
        """Run one true-batch SAM3 forward over a list of uint8 crops.

        Each crop's ``target_sizes`` is its OWN ``(H, W)`` (C4), so masks map
        back to the crop, not the full image. Returns one uint16 objmap per
        crop (in crop-local coordinates).
        """
        import torch

        self._ensure_model_loaded()
        inputs = self._processor(
            images=images,
            text=[self.prompt] * len(images),
            return_tensors="pt",
        ).to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        target_sizes = [(img.shape[0], img.shape[1]) for img in images]
        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=self.score_thresh,
            mask_threshold=self.mask_threshold,
            target_sizes=target_sizes,
        )
        return [
            self._paint_objmap(r, (img.shape[0], img.shape[1]))
            for r, img in zip(results, images)
        ]

    def _paint_objmap(
        self, result: dict, shape: tuple[int, int]
    ) -> "np.ndarray":
        """Paint one post-processed SAM3 result into a uint16 objmap.

        Masks are sorted largest-first and painted so smaller colonies
        overwrite at overlaps (preserving small-colony identity, like
        ``Sam2Detector``). ``min_mask_region_area`` drops tiny masks.
        """
        import numpy as np

        objmap = np.zeros(shape, dtype=np.uint16)
        masks = result.get("masks")
        if masks is None or len(masks) == 0:
            return objmap

        bool_masks: list[np.ndarray] = []
        for m in masks:
            arr = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
            arr = arr.astype(bool)
            if int(arr.sum()) >= self.min_mask_region_area:
                bool_masks.append(arr)
        if not bool_masks:
            return objmap

        bool_masks.sort(key=lambda a: int(a.sum()), reverse=True)
        max_labels = int(np.iinfo(np.uint16).max)
        if len(bool_masks) > max_labels:
            import warnings

            warnings.warn(
                f"SAM3 produced {len(bool_masks)} masks, exceeding uint16 "
                f"range. Only the first {max_labels} will be labeled.",
                UserWarning,
                stacklevel=2,
            )
            bool_masks = bool_masks[:max_labels]
        for idx, mask in enumerate(bool_masks, start=1):
            objmap[mask] = idx
        return objmap

    # ------------------------------------------------------------------
    # True-batch inference with fixed geometric tiling (C4)
    # ------------------------------------------------------------------

    def infer_batch(self, batch: Any) -> List["np.ndarray"]:
        """Run SAM3 over a collated batch; one uint16 objmap per sample.

        Each sample is tiled into fixed ~:attr:`tile_px` crops; all crops from
        all samples are regrouped and forwarded per tile-batch (C4), each
        crop's objmap is offset back to full-image coordinates, and per-sample
        crops are merged by IoU-NMS. Samples that fit one tile run un-tiled.
        """
        import numpy as np

        self._ensure_model_loaded()

        # Plan tiles per sample; collect every crop into one flat batch.
        plans: list[list[_Tile]] = []
        flat_crops: list[np.ndarray] = []
        owner: list[int] = []  # which sample each flat crop belongs to
        for s_idx, sample in enumerate(batch):
            arr = self._to_uint8(sample)
            tiles = _plan_tiles(
                (arr.shape[0], arr.shape[1]), self.tile_px, self.tile_overlap
            )
            plans.append(tiles)
            for t in tiles:
                flat_crops.append(arr[t.y0:t.y1, t.x0:t.x1])
                owner.append(s_idx)

        # One true-batch forward over every crop.
        crop_objmaps = self._forward_tiles(flat_crops) if flat_crops else []

        # Offset each crop objmap into full-image coords, group by sample.
        per_sample_full: list[list[np.ndarray]] = [[] for _ in batch]
        cursor = 0
        for s_idx, sample in enumerate(batch):
            arr = self._to_uint8(sample)
            full_shape = (arr.shape[0], arr.shape[1])
            for t in plans[s_idx]:
                crop_obj = crop_objmaps[cursor]
                cursor += 1
                full = np.zeros(full_shape, dtype=np.uint16)
                full[t.y0:t.y1, t.x0:t.x1] = crop_obj
                per_sample_full[s_idx].append(full)

        results: list[np.ndarray] = []
        for s_idx, sample in enumerate(batch):
            arr = self._to_uint8(sample)
            full_shape = (arr.shape[0], arr.shape[1])
            tile_objmaps = per_sample_full[s_idx]
            if len(tile_objmaps) == 1:
                # Single-tile path — relabel contiguously, no merge needed.
                results.append(
                    _merge_tiles_iou_nms(tile_objmaps, iou_thresh=1.0)
                )
            elif not tile_objmaps:
                results.append(np.zeros(full_shape, dtype=np.uint16))
            else:
                results.append(
                    _merge_tiles_iou_nms(tile_objmaps, iou_thresh=0.5)
                )
        return results


# Expose the class docstring on .apply() for Sphinx autodoc
Sam3Detector.apply.__doc__ = Sam3Detector.__doc__
