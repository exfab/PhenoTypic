"""SAM3 text-prompted, true-batch instance detector (Spec 2a, Tasks 4-5)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, List

from pydantic import PrivateAttr

from phenotypic.abc_ import GpuDetector
from phenotypic.detect.nn._helper._checkpoint_manager import Device

# Shared fixed-geometric tiling and the cross-tile instance merges, both owned
# by _tiling.py so the semantic detectors reuse the tiling. Re-exported here for
# back-compat with callers/tests that import these names from _sam3.
from phenotypic.detect.nn._helper._tiling import (
    _iou,
    _merge_tiles_iou_nms,
    _plan_tiles,
    _Tile,
    _tile_starts,
    assign_by_centroid_core,
)
from phenotypic.sdk_.typing_ import GpuInputLayer, GpuOutputKind, TuneSpec

if TYPE_CHECKING:
    import numpy as np

__all__ = ["Sam3"]

# Silence "imported but unused" — these are intentional back-compat re-exports.
_ = (_Tile, _tile_starts, _iou, _merge_tiles_iou_nms)


class Sam3(GpuDetector):
    """Detect colonies with Meta's SAM3 text-prompted foundation model.

    SAM3 segments every region matching a short **text prompt** (default
    ``"colony"``) in one true ``(N, C, H, W)`` batched forward pass, then
    assembles the predicted instance masks into a labeled object map.  Unlike
    SAM2's dense point grid, SAM3 has a single checkpoint and is prompted by
    free text, so the only "knob" describing *what* to find is the
    :attr:`prompt`.

    Because SAM3 operates on **RGB input** (not ``detect_mat``), classical
    enhancement operations placed before this detector in a pipeline are
    ignored — the model sees the original colour image regardless.

    The model and processor are loaded lazily on the first call to
    :meth:`~phenotypic.abc_.ImageOperation.apply` (not during construction), so
    a ``Sam3`` can be serialised, round-tripped through JSON, and
    inspected without a GPU, PyTorch, or ``transformers`` installed.

    **Gated weights.** SAM3 weights (~3.45 GB) are gated on Hugging Face under
    the SAM License.  Accept the gate at ``https://huggingface.co/facebook/sam3``
    and authenticate once (``uv run hf auth login`` or ``HF_TOKEN``); see
    :doc:`/how_to/pages/gpu_detection_setup`.

    **Dense plates.** SAM3 caps at 200 instances per forward and runs at
    1008 px internally, so dense plates are tiled (fixed ~:attr:`tile_px`
    tiles with :attr:`tile_overlap`) and each tile is inferred separately.
    The tiles are merged by *centroid-in-core* assignment
    (:func:`~phenotypic.detect.nn._helper._tiling.assign_by_centroid_core`): each
    instance is kept by the one tile whose core contains its centroid, so a
    colony straddling a seam cannot be duplicated, nor can the fragment a
    neighbouring tile saw survive as its own colony.

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
            Default 100 (matches ``Sam2``).
        tile_px: Nominal tile size in pixels for dense-plate tiling; images
            that fit one tile run un-tiled.  Default 1008 (SAM3's internal
            resolution).
        tile_overlap: Fractional overlap between neighbouring tiles.  Default
            0.15.
        tile_merge_iou: **Deprecated and ignored.** The cross-tile merge is now
            centroid-in-core
            (:func:`~phenotypic.detect.nn._helper._tiling.assign_by_centroid_core`),
            which assigns each instance to exactly one tile and therefore needs
            no IoU threshold. The field is retained only so pipelines
            serialised before the change keep deserialising; setting it has no
            effect. Default 0.5.
        max_instances_per_tile: SAM3's hard 200-instance-per-forward cap;
            structural, not tuned.
        device: PyTorch device for inference.  ``"auto"`` probes accelerators
            and raises ``RuntimeError`` if none is found.
        input_layer: Image layer fed to the model -- ``"rgb"`` (default; the
            layer SAM3 was trained on), ``"gray"``, or ``"detect_mat"``.
            Single-channel layers are stacked to 3 channels and coerced to
            uint8 by ``_preprocess``.

    Returns:
        Image: Input image with ``objmap`` set to a labeled instance map
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
        * :class:`~phenotypic.detect.nn.Sam2` for an ungated,
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

        >>> from phenotypic.detect.nn import Sam3
        >>> det = Sam3(prompt="yeast colony")
        >>> det.prompt
        'yeast colony'

        Build a pipeline and serialise it to JSON (round-trips without GPU
        dependencies):

        >>> from phenotypic import ImagePipeline
        >>> pipe = ImagePipeline(ops=[Sam3(prompt="colony")])
        >>> restored = ImagePipeline.from_json(pipe.to_json())
        >>> type(restored.get_ops()["Sam3"])
        <class 'phenotypic.detect.nn._sam3.Sam3'>
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
    # Deprecated: the tiled instance merge is centroid-in-core
    # (_tiling.assign_by_centroid_core), which needs no IoU threshold. Retained
    # so existing serialized pipelines keep deserializing.
    tile_merge_iou: Annotated[float, TuneSpec(0.0, 1.0)] = 0.5
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
                    "Sam3 requires transformers (>=5.2.0). "
                    "Install with: pip install phenotypic[foundation]"
            ) from None

        from phenotypic.detect.nn._helper._checkpoint_manager import (
            Sam3CheckpointManager,
            resolve_device,
        )

        self._device = resolve_device(self.device)
        repo_id = Sam3CheckpointManager.repo_id
        self._model = Sam3Model.from_pretrained(repo_id).to(self._device)
        self._processor = Sam3Processor.from_pretrained(repo_id)

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
        ``Sam2``). ``min_mask_region_area`` drops tiny masks.
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

    def _infer_batch(self, batch: Any) -> List["np.ndarray"]:
        """Run SAM3 over a collated batch; one uint16 objmap per sample.

        Each sample is tiled into fixed ~:attr:`tile_px` crops; all crops from
        all samples are regrouped and forwarded in one tile-batch (C4). The
        per-crop objmaps stay **tile-local** all the way to
        :func:`~phenotypic.detect.nn._helper._tiling.assign_by_centroid_core`, which
        owns the offset into full-image coordinates: it needs each instance's
        tile-local centroid to decide which tile's core claims it. Offsetting
        the crops here would make the merge add ``tile.y0``/``tile.x0`` a
        second time. Samples that fit one tile run un-tiled (the merge then
        just relabels contiguously).
        """
        import numpy as np

        self._ensure_model_loaded()

        # Plan tiles per sample; collect every crop into one flat batch
        # (samples arrive already uint8 from _preprocess; the later loops only
        # need the full-image shape, so no re-allocation of large plate images).
        plans: list[list[_Tile]] = []
        full_shapes: list[tuple[int, int]] = []
        flat_crops: list[np.ndarray] = []
        for sample in batch:
            arr = sample
            full_shapes.append((arr.shape[0], arr.shape[1]))
            tiles = _plan_tiles(
                    (arr.shape[0], arr.shape[1]), self.tile_px, self.tile_overlap
            )
            plans.append(tiles)
            for t in tiles:
                flat_crops.append(arr[t.y0:t.y1, t.x0:t.x1])

        # One true-batch forward over every crop.
        crop_objmaps = self._forward_tiles(flat_crops) if flat_crops else []

        # Group the tile-local objmaps by sample — no offsetting here; the
        # merge maps each instance into full-image coordinates itself.
        per_sample_local: list[list[np.ndarray]] = [[] for _ in batch]
        cursor = 0
        for s_idx in range(len(batch)):
            for _t in plans[s_idx]:
                per_sample_local[s_idx].append(crop_objmaps[cursor])
                cursor += 1

        results: list[np.ndarray] = []
        for s_idx in range(len(batch)):
            tile_objmaps = per_sample_local[s_idx]
            if not tile_objmaps:
                results.append(np.zeros(full_shapes[s_idx], dtype=np.uint16))
            else:
                results.append(
                        assign_by_centroid_core(
                                plans[s_idx], tile_objmaps, full_shapes[s_idx]
                        )
                )
        return results


# Expose the class docstring on .apply() for Sphinx autodoc
Sam3.apply.__doc__ = Sam3.__doc__
