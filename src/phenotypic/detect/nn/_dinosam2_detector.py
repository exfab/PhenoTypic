"""DinoSam2Detector — SAM2 proposals + DINOv2 feature scoring (Spec 2a, Task 6).

Training-free instance detector: run SAM2's automatic mask generator for
class-agnostic proposals, pool DINO patch features inside each proposal, score
each proposal by cosine similarity of its pooled feature to a foreground
prototype, drop background-like proposals, merge near-duplicates by IoU, and
paint the survivors largest-first into a labelled ``objmap``.

The recipe is a clean-room reimplementation of the training-free composition
described in *"No time to train!"* (arXiv:2507.02798) — algorithm only, no
vendored code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, List

from pydantic import PrivateAttr

from phenotypic.abc_ import GpuDetector
from phenotypic.detect.nn._checkpoint_manager import Device, Sam2ModelSize
from phenotypic.sdk_.typing_ import (
    DinoSize,
    DinoVersion,
    GpuInputLayer,
    GpuOutputKind,
    TuneSpec,
)

if TYPE_CHECKING:
    import numpy as np


# ---------------------------------------------------------------------------
# Clean-room recipe helpers (pure numpy — driven by synthetic-feature tests)
# ---------------------------------------------------------------------------


def _score_by_prototype(
    features: "np.ndarray", prototype: "np.ndarray"
) -> "np.ndarray":
    """Cosine similarity of each proposal feature to the foreground prototype.

    Args:
        features: ``(N, D)`` pooled per-proposal feature vectors.
        prototype: ``(D,)`` foreground prototype (mean of high-confidence
            proposal features).

    Returns:
        ``(N,)`` cosine-similarity scores in ``[-1, 1]``.
    """
    import numpy as np

    feats = np.asarray(features, dtype=np.float64)
    proto = np.asarray(prototype, dtype=np.float64)
    feat_norm = np.linalg.norm(feats, axis=1)
    proto_norm = np.linalg.norm(proto)
    denom = feat_norm * proto_norm
    safe = np.where(denom == 0, 1.0, denom)
    scores = (feats @ proto) / safe
    scores[denom == 0] = 0.0
    return scores


def _iou_bool(mask_a: "np.ndarray", mask_b: "np.ndarray") -> float:
    """Intersection-over-union of two boolean masks (0.0 if both empty)."""
    inter = int((mask_a & mask_b).sum())
    if inter == 0:
        return 0.0
    union = int((mask_a | mask_b).sum())
    return inter / union if union else 0.0


def _merge_by_iou(
    masks: List["np.ndarray"], iou_thresh: float
) -> List["np.ndarray"]:
    """Greedy IoU dedup of boolean proposal masks, largest-first.

    Args:
        masks: Boolean proposal masks.
        iou_thresh: IoU above which a later (smaller) mask is dropped as a
            near-duplicate of an already-kept mask.

    Returns:
        The surviving masks, ordered largest-first.
    """
    ordered = sorted(masks, key=lambda m: int(m.sum()), reverse=True)
    kept: list = []
    for cand in ordered:
        if any(_iou_bool(cand, k) > iou_thresh for k in kept):
            continue
        kept.append(cand)
    return kept


def _assemble_objmap(
    proposals: List["np.ndarray"],
    scores: "np.ndarray",
    similarity_thresh: float,
    merge_iou_thresh: float,
) -> "np.ndarray":
    """Filter, merge, and paint scored proposals into a uint16 objmap.

    Proposals scoring below ``similarity_thresh`` are dropped as background;
    survivors are IoU-merged and painted largest-first (smaller colonies
    overwrite at overlaps, preserving small-colony identity).

    Args:
        proposals: Boolean proposal masks (all the same shape).
        scores: ``(N,)`` cosine-to-prototype scores aligned with *proposals*.
        similarity_thresh: Minimum score to treat a proposal as foreground.
        merge_iou_thresh: IoU above which two survivors are deduplicated.

    Returns:
        A uint16 objmap with contiguous labels ``1..N``.
    """
    import numpy as np

    if not proposals:
        raise ValueError("_assemble_objmap requires at least one proposal")
    shape = proposals[0].shape

    scores = np.asarray(scores)
    foreground = [
        p for p, s in zip(proposals, scores) if s >= similarity_thresh
    ]
    if not foreground:
        return np.zeros(shape, dtype=np.uint16)

    kept = _merge_by_iou(foreground, merge_iou_thresh)

    max_labels = int(np.iinfo(np.uint16).max)
    if len(kept) > max_labels:
        import warnings

        warnings.warn(
            f"DinoSam2 kept {len(kept)} proposals, exceeding uint16 range. "
            f"Only the first {max_labels} (largest) will be labeled.",
            UserWarning,
            stacklevel=2,
        )
        kept = kept[:max_labels]

    objmap = np.zeros(shape, dtype=np.uint16)
    for idx, mask in enumerate(kept, start=1):
        objmap[mask.astype(bool)] = idx
    return objmap


class DinoSam2Detector(GpuDetector):
    """Detect colonies with SAM2 proposals scored by DINOv2 features.

    A **training-free** instance detector that composes two ungated foundation
    models: SAM2's automatic mask generator produces class-agnostic mask
    proposals, and a DINOv2 backbone supplies dense patch features.  Each
    proposal's pooled DINO feature is scored by cosine similarity to a
    foreground prototype (the mean feature of the high-confidence proposals);
    background-like proposals are dropped, near-duplicates merged by IoU, and
    the survivors painted largest-first into a labelled object map.

    The recommended configuration is **fully ungated and permissive** — SAM2
    (Apache-2.0) plus DINOv2 (Apache-2.0). ``dino_version=3`` selects the gated
    DINOv3 backbone as an explicit opt-in; that load path routes the snapshot
    pull through ``Dinov3CheckpointManager`` (honouring the DINOv3-License
    acceptance gate) before loading the backbone.

    Because the recipe runs SAM2's per-image AMG, it is **not** batchable
    (``supports_batching=False``); the engine fills the GPU via worker packing.

    The models are loaded lazily on the first ``apply()`` call, so the detector
    serialises and round-trips through JSON without a GPU, ``torch``, ``sam2``,
    or ``transformers`` installed.

    Args:
        dino_version: DINO backbone generation.  ``2`` = DINOv2 (Apache,
            ungated, default); ``3`` = DINOv3 (gated opt-in — routes through the
            DINOv3-License acceptance gate).
        dino_size: DINO backbone size (``"small"``/``"base"``/``"large"``),
            mapped with *dino_version* to the Hugging Face model id.
        sam2_model_size: SAM2 variant for the proposal generator
            (``"tiny"`` … ``"large"``).
        similarity_thresh: Minimum cosine-to-prototype score for a proposal to
            be kept as foreground.  Raise to suppress agar/background masks;
            lower to retain faint colonies.  Default 0.5.
        merge_iou_thresh: IoU above which two surviving proposals are merged as
            duplicates (fixes SAM2 over-segmentation).  Default 0.7.
        min_proposal_area: Minimum proposal area in pixels (forwarded to SAM2's
            mask generator).  Default 100.
        device: PyTorch device for inference.  ``"auto"`` probes accelerators
            and raises ``RuntimeError`` if none is found.

    Returns:
        Image: Input image with ``objmap`` set to a labelled instance map and
        ``objmask`` to the derived binary mask.

    Raises:
        ImportError: If ``sam2`` / ``transformers`` / ``torch`` are not
            installed.  Install with ``pip install phenotypic[foundation]``.
        RuntimeError: If ``device="auto"`` and no accelerator is available, or
            (``dino_version=3``) the gated DINOv3 license was not accepted /
            no token is present.

    Best For:
        * Ungated, license-clean instance detection where SAM2 over-segments
          and a feature-similarity prior cleans up the proposals.

    Consider Also:
        * :class:`~phenotypic.detect.nn.Sam2Detector` for raw SAM2 proposals
          without DINO scoring.
        * :class:`~phenotypic.detect.nn.Sam3Detector` for text-prompted,
          true-batch instance detection (gated weights).

    References:
        [1] L. Karazija et al., "No time to train! Training-Free Reference-Based
        Instance Segmentation," *arXiv:2507.02798*, 2025 (recipe referent;
        clean-room reimplementation).
        [2] M. Oquab et al., "DINOv2: Learning Robust Visual Features without
        Supervision," *arXiv:2304.07193*, 2023.

    See Also:
        :doc:`/how_to/pages/gpu_detection_setup`
            Installation, backbone selection, and SLURM deployment.

    Examples:
        Construct a detector and confirm the ungated DINOv2 default (no GPU or
        weights required):

        >>> from phenotypic.detect.nn import DinoSam2Detector
        >>> det = DinoSam2Detector()
        >>> det.dino_version
        2
        >>> det._hf_dino_id()
        'facebook/dinov2-base'

        Build a pipeline and serialise it to JSON (round-trips without GPU
        dependencies):

        >>> from phenotypic import ImagePipeline
        >>> pipe = ImagePipeline(ops=[DinoSam2Detector(dino_size="large")])
        >>> restored = ImagePipeline.from_json(pipe.to_json())
        >>> type(restored.get_ops()["DinoSam2Detector"])
        <class 'phenotypic.detect.nn._dinosam2_detector.DinoSam2Detector'>
    """

    # Capabilities — instance-native, RGB, per-image (SAM2 AMG bounds batching).
    input_layer: GpuInputLayer = "rgb"
    output_kind: GpuOutputKind = "instance"
    supports_batching: bool = False

    dino_version: DinoVersion = 2
    dino_size: DinoSize = "base"
    sam2_model_size: Sam2ModelSize = "tiny"
    similarity_thresh: Annotated[float, TuneSpec(0.0, 1.0)] = 0.5
    merge_iou_thresh: Annotated[float, TuneSpec(0.0, 1.0)] = 0.7
    min_proposal_area: Annotated[int, TuneSpec(0, 500)] = 100
    device: Device = "auto"

    # Lazy runtime state — PrivateAttr → skipped by serialization.
    _generator: Any = PrivateAttr(default=None)
    _dino_model: Any = PrivateAttr(default=None)
    _dino_processor: Any = PrivateAttr(default=None)
    _device: Any = PrivateAttr(default=None)

    def _hf_dino_id(self) -> str:
        """Map ``(dino_version, dino_size)`` to the Hugging Face backbone id.

        Returns:
            ``"facebook/dinov2-{size}"`` for v2 (ungated), or the gated
            ``"facebook/dinov3-vit{s|b|l}16-pretrain-lvd1689m"`` id for v3.
        """
        from phenotypic.detect.nn._dino_support import hf_dino_id

        return hf_dino_id(self.dino_version, self.dino_size)

    def _ensure_model_loaded(self) -> None:
        """Build the SAM2 AMG + DINO backbone on first use (idempotent).

        Rebuilds the SAM2 ``SAM2AutomaticMaskGenerator`` via the shared
        ``build_sam2_generator`` helper (C1 — there is no public accessor for
        another detector's generator) and loads the DINO backbone through
        ``transformers.AutoModel``. ``dino_version=3`` routes the gated DINOv3
        snapshot pull through :class:`Dinov3CheckpointManager` (honouring the
        license-acceptance gate) before loading the backbone.
        """
        if getattr(self, "_generator", None) is not None:
            return

        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError:
            raise ImportError(
                "DinoSam2Detector requires transformers (>=5.2.0). "
                "Install with: pip install phenotypic[foundation]"
            ) from None

        from phenotypic.detect.nn._checkpoint_manager import (
            Dinov3CheckpointManager,
            resolve_device,
        )
        from phenotypic.detect.nn._sam2_detector import build_sam2_generator

        # DINOv3 is gated — accept + pre-stage the snapshot before the load.
        if self.dino_version == 3:
            Dinov3CheckpointManager(size=self.dino_size).download()

        self._device = resolve_device(self.device)
        self._generator = build_sam2_generator(
            self.sam2_model_size,
            device=self._device,
            min_mask_region_area=self.min_proposal_area,
        )
        dino_id = self._hf_dino_id()
        self._dino_model = AutoModel.from_pretrained(dino_id).to(self._device)
        self._dino_processor = AutoImageProcessor.from_pretrained(dino_id)

    # ------------------------------------------------------------------
    # Per-image recipe
    # ------------------------------------------------------------------

    def _infer_one(self, sample: Any) -> "np.ndarray":
        """Run the SAM2-proposals + DINO-scoring recipe on one RGB sample.

        Returns a uint16 labelled objmap. SAM2 proposals are scored by cosine
        similarity of their pooled DINO feature to the foreground prototype
        (mean of the highest-IoU proposals), filtered by
        ``similarity_thresh``, IoU-merged, and painted largest-first.
        """
        import numpy as np

        self._ensure_model_loaded()

        rgb = self._to_uint8(sample)
        raw = self._generator.generate(rgb)  # type: ignore[attr-defined]
        if not raw:
            return np.zeros(rgb.shape[:2], dtype=np.uint16)

        proposals = [m["segmentation"].astype(bool) for m in raw]
        # Shared, register-token-aware dense features (C1 fix lives in one
        # place — _dino_support.extract_patch_features — not duplicated here).
        from phenotypic.detect.nn._dino_support import (
            extract_patch_features,
            pool_prototype,
        )

        dense = extract_patch_features(
            self._dino_model, self._dino_processor, rgb, device=self._device
        )
        features = np.stack(
            [pool_prototype(dense, p).astype(np.float64) for p in proposals]
        )
        prototype = self._foreground_prototype(features, raw)
        scores = _score_by_prototype(features, prototype)
        return _assemble_objmap(
            proposals,
            scores,
            similarity_thresh=self.similarity_thresh,
            merge_iou_thresh=self.merge_iou_thresh,
        )

    @staticmethod
    def _to_uint8(sample: "np.ndarray") -> "np.ndarray":
        """Coerce an ``(H, W, 3)`` sample to uint8 for SAM2 / the DINO processor."""
        import numpy as np

        rgb = sample
        if rgb.dtype != np.uint8:
            max_val = rgb.max()
            if max_val > 0:
                rgb = (rgb / max_val * 255).astype(np.uint8)
            else:
                rgb = np.zeros(rgb.shape, dtype=np.uint8)
        return rgb

    @staticmethod
    def _foreground_prototype(
        features: "np.ndarray", raw_masks: List[dict]
    ) -> "np.ndarray":
        """Build the foreground prototype from the high-confidence proposals.

        Uses SAM2's per-proposal ``predicted_iou`` as the confidence signal;
        the prototype is the mean feature of the proposals at or above the
        median IoU (falls back to all proposals if IoU is unavailable).

        Args:
            features: ``(N, D)`` pooled proposal features.
            raw_masks: SAM2 mask dicts (carry ``predicted_iou``).

        Returns:
            ``(D,)`` foreground prototype vector.
        """
        import numpy as np

        ious = np.array(
            [float(m.get("predicted_iou", 1.0)) for m in raw_masks],
            dtype=np.float64,
        )
        if ious.size == 0:
            return features.mean(axis=0)
        threshold = float(np.median(ious))
        high = features[ious >= threshold]
        if high.size == 0:
            high = features
        return high.mean(axis=0)


# Expose the class docstring on .apply() for Sphinx autodoc
DinoSam2Detector.apply.__doc__ = DinoSam2Detector.__doc__
