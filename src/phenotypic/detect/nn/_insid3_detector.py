"""Insid3Detector — one-shot in-context semantic detector (Spec 2b, Task 4).

Faithful clean-room reimplementation of INSID3 (``github.com/visinf/INSID3``,
Apache-2.0): training-free **in-context** segmentation on a frozen DINOv3
backbone. INSID3's defining step is **positional-bias removal** — DINOv3 patch
features carry a stable low-dimensional positional component that, left in,
makes patches match on *where* they are rather than *what* they are. INSID3
estimates that component with SVD and projects features onto its orthogonal
complement BEFORE prototype matching.

Faithful method (from the upstream code), per image:

1. Extract dense patch features ``X ∈ (Hp, Wp, D)`` from the frozen DINO ViT
   (CLS + register tokens dropped — the C1 fix in ``_dino_support``).
2. **Positional basis** ``U_k``: arrange features as ``E ∈ (D, N)`` and take
   ``U, _, _ = svd(E)``; keep the top ``svd_components`` left singular vectors
   (``U[:, :k]`` — the dominant directions shared across positions, i.e. the
   positional subspace on low-semantic content).
   ``_build_positional_basis`` in the upstream::

       E = rearrange(noise_fmaps, 'b c h w -> c (b h w)')
       U, _, _ = torch.linalg.svd(E, full_matrices=False)
       return U[:, :svd_components]

3. **Debias** by projecting onto the orthogonal complement and L2-normalising::

       P_perp = I - U_k @ U_k.T
       X_deb  = P_perp @ X ; X_deb = normalize(X_deb)

4. **Prototype**: masked-mean the debiased reference features over the reference
   mask, L2-normalise → one in-context prototype.
5. **Match**: cosine (dot product of normalised debiased features) of every
   query patch to the prototype, threshold at ``similarity_thresh`` → boolean
   ``objmask`` (upsampled to the full image; tiled for large plates).

This emits ``output_kind="semantic"`` (writes ``image.objmask``), which the
repo's downstream watershed instances (Spec 1 §8). DINOv3 is gated; a DINOv2
opt-in (``dino_version=2``) runs gate-free for testing — the debias is a
near-no-op on DINOv2 (which has no register tokens) but the path is identical.

Attribution: clean-room from INSID3 (Apache-2.0); no upstream code vendored.
Built with DINOv3 when ``dino_version=3`` (per the DINOv3 License §1.b.i).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, List, Optional

from pydantic import PrivateAttr

from phenotypic.abc_ import GpuDetector
from phenotypic.detect.nn._checkpoint_manager import Device
from phenotypic.tools_.typing_ import (
    DinoSize,
    DinoVersion,
    GpuInputLayer,
    GpuOutputKind,
    TuneSpec,
)

if TYPE_CHECKING:
    import numpy as np


# ---------------------------------------------------------------------------
# Faithful positional-bias removal (clean-room from INSID3, Apache-2.0).
# Pure numpy so it is driven by synthetic-feature unit tests (no model).
# ---------------------------------------------------------------------------


def positional_basis(
    features: "np.ndarray", n_components: int
) -> "np.ndarray":
    """Estimate the positional subspace basis via SVD (INSID3's core step).

    Arrange dense features as ``E ∈ (D, N)`` (channels × patches) and take the
    top ``n_components`` left singular vectors — the dominant directions shared
    across spatial positions, i.e. the positional/global component on
    low-semantic content (mirrors the upstream
    ``U, _, _ = svd(rearrange(fmaps, 'b c h w -> c (b h w)')); U[:, :k]``).

    Args:
        features: ``(Hp, Wp, D)`` dense patch features to estimate the basis
            from (INSID3 uses a low-semantic / noise input; the bundled
            detector uses the reference image's own features, whose dominant
            low-rank component is the positional bias).
        n_components: Number of leading singular vectors to keep (``0`` → an
            empty basis, making the debias an identity).

    Returns:
        ``(D, n_components)`` orthonormal positional basis ``U_k``.
    """
    import numpy as np

    feats = np.asarray(features, dtype=np.float64)
    hp, wp, d = feats.shape
    k = max(0, int(n_components))
    if k == 0:
        return np.zeros((d, 0), dtype=np.float32)
    e = feats.reshape(hp * wp, d).T  # (D, N) — channels × patches
    u, _s, _vt = np.linalg.svd(e, full_matrices=False)
    return u[:, :k].astype(np.float32)


def debias_features(
    features: "np.ndarray", basis: "np.ndarray"
) -> "np.ndarray":
    """Project features onto the orthogonal complement of the positional basis.

    ``P_perp = I - U_k U_kᵀ`` ; ``X_deb = P_perp X`` ; then L2-normalise each
    patch (mirrors INSID3's ``_debias_features``). An empty basis is the
    identity (the debias degenerates to a no-op, e.g. ``svd_components=0`` or a
    near-no-op on DINOv2).

    Args:
        features: ``(Hp, Wp, D)`` dense patch features.
        basis: ``(D, k)`` positional basis from :func:`positional_basis`.

    Returns:
        ``(Hp, Wp, D)`` debiased, L2-normalised features.
    """
    import numpy as np

    feats = np.asarray(features, dtype=np.float64)
    hp, wp, d = feats.shape
    flat = feats.reshape(hp * wp, d)
    b = np.asarray(basis, dtype=np.float64)
    if b.size == 0:
        deb = flat
    else:
        p_perp = np.eye(d) - b @ b.T  # (D, D)
        deb = flat @ p_perp.T  # (N, D)
    norms = np.linalg.norm(deb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    deb = deb / norms
    return deb.reshape(hp, wp, d).astype(np.float32)


class Insid3Detector(GpuDetector):
    """Detect colonies by INSID3 one-shot in-context semantic segmentation.

    A **training-free, one-shot** semantic detector on a frozen DINOv3 backbone.
    Given a single annotated **reference image + reference mask**, INSID3 pools
    an in-context prototype from the reference and cosine-matches every query
    patch to it — but first removes DINOv3's **positional bias** (the dominant
    low-rank component of patch features, estimated by SVD and projected out)
    so patches match on appearance, not position. The result is a boolean
    ``objmask``; the repo's downstream watershed (``SeparateObjects``) turns it
    into instances.

    Because INSID3 emits a **semantic** mask (``output_kind="semantic"``), it
    writes ``image.objmask`` (which auto-labels into the shared ``objmap``
    backend exactly like a threshold detector — Spec 1 §8), not its own
    instance labels.

    A **curated colony exemplar** (a reference RGB crop + its mask, rendered
    once from :func:`phenotypic.data.load_synth_yeast_plate`) ships with the
    package and is the **default** ``reference_image``/``reference_mask`` so the
    detector works out of the box; override with your own annotated reference.

    The model is loaded lazily on the first ``apply()`` call, so the detector
    serialises and round-trips through JSON without a GPU, ``torch``, or
    ``transformers`` installed.

    **Gated weights.** ``dino_version=3`` (the default — INSID3 is DINOv3-native)
    pulls the gated DINOv3 backbone (accept the DINOv3 License at
    ``https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m`` and
    ``uv run hf auth login``; see :doc:`/how_to/pages/gpu_detection_setup`).
    ``dino_version=2`` selects the ungated DINOv2 backbone (gate-free) — the
    positional debias is a near-no-op there, but the path is otherwise identical.

    Args:
        reference_image: Path to the in-context reference RGB image. Defaults to
            the bundled curated colony exemplar.
        reference_mask: Path to the reference's binary mask (foreground = the
            target). Defaults to the bundled exemplar's mask.
        dino_version: DINO backbone generation. ``3`` = DINOv3 (gated, INSID3's
            native backbone, default); ``2`` = DINOv2 (Apache, ungated opt-in).
        dino_size: DINO backbone size (``"small"``/``"base"``/``"large"``).
        similarity_thresh: Cosine-similarity cutoff binarising the match map.
            Raise to keep only strong matches; lower to catch faint colonies.
            Default 0.5.
        svd_components: Number of leading SVD directions removed as the
            positional component (INSID3's debiasing strength). Default 4
            (matches DINOv3's register-token count). ``0`` disables the debias.
        tile_px: Nominal tile size in pixels for large-plate tiling; images that
            fit one tile run un-tiled. Default 1024 (INSID3's default
            resolution).
        tile_overlap: Fractional overlap between neighbouring tiles. Default
            0.15.
        device: PyTorch device for inference. ``"auto"`` probes accelerators and
            raises ``RuntimeError`` if none is found.

    Returns:
        Image: Input image with ``objmask`` set to the semantic foreground mask
        (and ``objmap`` auto-labelled from it).

    Raises:
        ImportError: If ``transformers`` / ``torch`` are not installed. Install
            with ``pip install phenotypic[foundation]``.
        ValueError: If ``reference_image``/``reference_mask`` are unset.
        RuntimeError: If ``device="auto"`` and no accelerator is available, or
            (``dino_version=3``) the gated DINOv3 license was not accepted.

    Best For:
        * One-shot transfer from a single hand-annotated plate to a batch of
          similar plates, without tuning an intensity threshold.

    Consider Also:
        * :class:`~phenotypic.detect.nn.FssDinoDetector` for a **few-shot**
          (support-set) semantic detector with k prototypes + Gram refinement.
        * :class:`~phenotypic.detect.nn.DinoSam2Detector` for an instance-native
          ungated detector.

    References:
        [1] visinf, "INSID3: In-Context Segmentation with DINOv3," Apache-2.0,
        ``https://github.com/visinf/INSID3`` (clean-room reimplementation).
        [2] M. Oquab et al., "DINOv2," *arXiv:2304.07193*, 2023.

    See Also:
        :doc:`/how_to/pages/gpu_detection_setup`
            Installation, gated-weight acceptance, and the exemplar interface.

    Examples:
        Construct a detector and inspect its semantic capability (no GPU or
        weights required):

        >>> from phenotypic.detect.nn import Insid3Detector
        >>> det = Insid3Detector()
        >>> det.output_kind
        'semantic'
        >>> det.dino_version
        3

        Build a pipeline and serialise it to JSON (round-trips without GPU
        dependencies):

        >>> from phenotypic import ImagePipeline
        >>> pipe = ImagePipeline(ops=[Insid3Detector(similarity_thresh=0.6)])
        >>> restored = ImagePipeline.from_json(pipe.to_json())
        >>> type(restored.get_ops()["Insid3Detector"])
        <class 'phenotypic.detect.nn._insid3_detector.Insid3Detector'>
    """

    # Capabilities — in-context, semantic, per-image (one reference forward).
    input_layer: GpuInputLayer = "rgb"
    output_kind: GpuOutputKind = "semantic"
    supports_batching: bool = False

    reference_image: Optional[Path] = None
    reference_mask: Optional[Path] = None
    dino_version: DinoVersion = 3  # INSID3 is DINOv3-native (gated)
    dino_size: DinoSize = "base"
    similarity_thresh: Annotated[float, TuneSpec(0.0, 1.0)] = 0.5
    # INSID3's positional-debias strength (leading SVD directions removed).
    # Structural to the method (≈ DINOv3 register-token count), not a search
    # window — declare intent-to-tune without a fabricated range.
    svd_components: Annotated[int, TuneSpec()] = 4

    # Tiling (shared with the instance detectors via _tiling).
    tile_px: Annotated[int, TuneSpec(512, 2048)] = 1024
    tile_overlap: Annotated[float, TuneSpec(0.0, 0.4)] = 0.15

    device: Device = "auto"

    # Lazy runtime state — PrivateAttr → skipped by serialization.
    _model: Any = PrivateAttr(default=None)
    _processor: Any = PrivateAttr(default=None)
    _device: Any = PrivateAttr(default=None)
    _prototype: Any = PrivateAttr(default=None)
    _basis: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Fill in the bundled colony exemplar default when no reference is set.

        Done post-init (not as a field default) so the default is the real
        on-disk bundled path resolved at construction, while an explicit
        ``reference_image=None`` (e.g. the validation test) stays ``None``.
        """
        if (
            "reference_image" not in self.model_fields_set
            and "reference_mask" not in self.model_fields_set
        ):
            from phenotypic._assets import colony_exemplar_paths

            rgb_path, mask_path = colony_exemplar_paths()
            object.__setattr__(self, "reference_image", rgb_path)
            object.__setattr__(self, "reference_mask", mask_path)

    # ------------------------------------------------------------------
    # Lazy load + in-context prototype caching
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Load the frozen DINO backbone and cache the in-context prototype.

        Validates the reference pair is set, loads the backbone (gated DINOv3
        pre-staged via ``Dinov3CheckpointManager``), extracts + debiases the
        reference features, and masked-mean-pools the prototype over the
        reference mask. Idempotent.
        """
        if self._prototype is not None:
            return
        if self.reference_image is None or self.reference_mask is None:
            raise ValueError(
                "Insid3Detector requires a reference_image and reference_mask "
                "(an in-context reference RGB + its foreground mask)."
            )

        import numpy as np

        from phenotypic.detect.nn._checkpoint_manager import resolve_device
        from phenotypic.detect.nn._dino_support import (
            extract_patch_features,
            load_dino_backbone,
            pool_prototype,
        )

        self._device = resolve_device(self.device)
        self._model, self._processor = load_dino_backbone(
            self.dino_version, self.dino_size, self._device
        )

        ref_rgb = self._read_rgb(self.reference_image)
        ref_mask = self._read_mask(self.reference_mask)
        ref_feats = extract_patch_features(
            self._model, self._processor, ref_rgb, device=self._device
        )
        # INSID3: estimate the positional basis (here from the reference's own
        # features — its dominant low-rank component is the positional bias),
        # then debias before pooling the in-context prototype.
        self._basis = positional_basis(ref_feats, self.svd_components)
        ref_deb = debias_features(ref_feats, self._basis)
        proto = pool_prototype(ref_deb, ref_mask)
        # L2-normalise the prototype (INSID3 normalises the prototype too).
        nrm = float(np.linalg.norm(proto))
        if nrm > 0:
            proto = (proto / nrm).astype(np.float32)
        self._prototype = proto

    @staticmethod
    def _read_rgb(path: Path) -> "np.ndarray":
        """Read a reference RGB image as ``(H, W, 3)`` uint8."""
        import numpy as np
        from skimage.io import imread

        arr = np.asarray(imread(str(path)))
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:
            mx = arr.max()
            arr = (arr / mx * 255).astype(np.uint8) if mx > 0 else arr.astype(np.uint8)
        return arr

    @staticmethod
    def _read_mask(path: Path) -> "np.ndarray":
        """Read a reference mask as a ``(H, W)`` boolean array."""
        import numpy as np
        from skimage.io import imread

        arr = np.asarray(imread(str(path)))
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr > (0.5 * float(arr.max()) if arr.max() > 1 else 0.5)

    # ------------------------------------------------------------------
    # Per-image inference (semantic; tiled for large plates)
    # ------------------------------------------------------------------

    def _infer_one(self, sample: Any) -> "np.ndarray":
        """Cosine-match a query image to the in-context prototype → objmask.

        Extracts + debiases query patch features, cosine-matches them to the
        cached prototype, upsamples and thresholds to a boolean mask. Large
        plates are tiled (semantic union stitch).
        """
        import numpy as np

        self._ensure_model_loaded()
        rgb = self._to_uint8(sample)

        from phenotypic.detect.nn._tiling import (
            _plan_tiles,
            stitch_semantic_tiles,
        )

        tiles = _plan_tiles(
            (rgb.shape[0], rgb.shape[1]), self.tile_px, self.tile_overlap
        )
        if len(tiles) == 1:
            return self._match_crop(rgb)

        tile_masks: List[np.ndarray] = []
        for t in tiles:
            crop = rgb[t.y0:t.y1, t.x0:t.x1]
            tile_masks.append(self._match_crop(crop))
        return stitch_semantic_tiles(
            tiles, tile_masks, (rgb.shape[0], rgb.shape[1])
        )

    def _match_crop(self, rgb: "np.ndarray") -> "np.ndarray":
        """Debias + cosine-match one crop to the prototype → boolean mask."""
        from phenotypic.detect.nn._dino_support import (
            cosine_match_to_mask,
            extract_patch_features,
        )

        feats = extract_patch_features(
            self._model, self._processor, rgb, device=self._device
        )
        feats_deb = debias_features(feats, self._basis)
        return cosine_match_to_mask(
            feats_deb,
            self._prototype,
            thresh=self.similarity_thresh,
            out_shape=(rgb.shape[0], rgb.shape[1]),
        )

    @staticmethod
    def _to_uint8(sample: "np.ndarray") -> "np.ndarray":
        """Coerce an ``(H, W, 3)`` sample to uint8 for the DINO processor."""
        import numpy as np

        rgb = sample
        if rgb.dtype != np.uint8:
            max_val = rgb.max()
            if max_val > 0:
                rgb = (rgb / max_val * 255).astype(np.uint8)
            else:
                rgb = np.zeros(rgb.shape, dtype=np.uint8)
        return rgb


# Expose the class docstring on .apply() for Sphinx autodoc
Insid3Detector.apply.__doc__ = Insid3Detector.__doc__
