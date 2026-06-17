"""FssDinoDetector — few-shot semantic detector (Spec 2b, Task 5).

Clean-room reimplementation **from the paper only** of FSSDINO — Zakir & Ho,
"Revealing the Semantic Selection Gap in DINOv3 through Training-Free Few-Shot
Segmentation," arXiv:2602.07550 (paper CC BY-NC-SA; the reference code repo is
all-rights-reserved and is NOT vendored). Training-free few-shot semantic
segmentation on a frozen DINO backbone.

Faithful method (paper §3–4), with the binary colony case treated as two
classes ``c ∈ {background=0, foreground=1}``:

1. **Class features.** For each class ``c`` gather all support patch features
   under the class mask across the support set: ``X^c = {f ∈ ℝ^d}`` (the
   foreground mask for c=1, its complement for c=0).
2. **Class-specific prototypes.** ``P^c = Cluster(X^c, n_c) ∈ ℝ^{n_c×d}`` via
   **k-means with cosine distance**; the paper sets ``n_c = 5``.
3. **Gram matrix** (style/co-occurrence consistency) over the L2-normalised
   class features: ``G^c = (1/N_c) Σ_{f̃ ∈ X^c} f̃ f̃ᵀ ∈ ℝ^{d×d}``.
4. **Query maps.** For a query's dense features ``F^q``:
   - per-prototype cosine ``S^c_i(u,v) = F^q(u,v)·p^c_i / (‖F^q‖‖p^c_i‖)``,
   - Gram refinement ``Q̂ = G^c Q`` then channel energy
     ``S^c_gram(u,v) = Σ_j Q_j(u,v) Q̂_j(u,v)`` (``Q`` = ``F^q`` reshaped to
     ``ℝ^{d×hw}``).
   Collect ``S^c = {S^c_1, …, S^c_{n_c}, S^c_gram}``, bilinearly upsample all,
   then ``S̃^c_mean = mean(S^c)``, ``S̃^c_max = max(S^c)`` and
   ``S̃^c_score = S̃^c_mean ⊙ S̃^c_max`` (Hadamard product; each map min-max
   normalised to [0,1] first).
5. **Assignment.** ``ŷ(u,v) = argmax_c S̃^c_score(u,v)`` → foreground where the
   fg score beats the bg score.

**Layer selection (the paper's central finding).** Intermediate layers often
beat the last by 6–13 mIoU, but no unsupervised heuristic reliably selects them
(the "Semantic Selection Gap"); the paper recommends the **last layer as the
safe default**. ``feature_layer`` (``-1`` = last) exposes the knob via
``output_hidden_states``.

**Deviation note (documented).** The paper assumes ground-truth-labelled
multi-class support. For the unsupervised binary colony case the detector
derives the background class from the mask complement and adds an optional
``similarity_thresh`` floor on the fg score (a knob, default 0.5) on top of the
paper's ``argmax`` — neither alters the paper's prototype/Gram math.

Emits ``output_kind="semantic"`` (writes ``image.objmask``; the repo's
downstream watershed instances it — Spec 1 §8). DINOv2 is the default (ungated,
gate-free); DINOv3 is an opt-in (gated).

Attribution: clean-room from arXiv:2602.07550 (CC BY-NC-SA); NO code vendored.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, List

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
# Faithful FSSDINO algorithm pieces (clean-room from arXiv:2602.07550).
# Pure numpy / scikit-learn so they are driven by synthetic-feature tests.
# ---------------------------------------------------------------------------


def _l2_normalize_rows(x: "np.ndarray") -> "np.ndarray":
    """L2-normalise each row, guarding zero rows."""
    import numpy as np

    arr = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def cluster_prototypes(
    features: "np.ndarray", n_clusters: int
) -> "np.ndarray":
    """Cluster class features into ``n_c`` prototypes (k-means, cosine distance).

    Features are L2-normalised so Euclidean k-means on the sphere approximates
    cosine-distance clustering (paper Eq. 2: ``P^c = Cluster(X^c, n_c)``); the
    returned cluster centres are re-normalised to unit length. ``n_clusters`` is
    capped at the number of available samples.

    Args:
        features: ``(N, D)`` class patch features.
        n_clusters: Target number of prototypes (paper default 5).

    Returns:
        ``(k, D)`` unit-norm prototypes (``k = min(n_clusters, N)``).
    """
    import numpy as np

    feats = np.asarray(features, dtype=np.float64)
    if feats.shape[0] == 0:
        return np.zeros((0, feats.shape[1] if feats.ndim == 2 else 0))
    normed = _l2_normalize_rows(feats)
    k = max(1, min(int(n_clusters), normed.shape[0]))
    if k == 1:
        centers = normed.mean(axis=0, keepdims=True)
    else:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        km.fit(normed)
        centers = km.cluster_centers_
    return _l2_normalize_rows(centers).astype(np.float32)


def gram_matrix(features: "np.ndarray") -> "np.ndarray":
    """Mean normalised-feature outer product — FSSDINO's class Gram matrix.

    ``G^c = (1/N_c) Σ_{f̃ ∈ X^c} f̃ f̃ᵀ`` over the L2-normalised class features
    (paper §4). Captures inter-channel co-occurrence / style consistency.

    Args:
        features: ``(N, D)`` class patch features.

    Returns:
        ``(D, D)`` Gram matrix (zeros if no features).
    """
    import numpy as np

    feats = np.asarray(features, dtype=np.float64)
    if feats.shape[0] == 0:
        d = feats.shape[1] if feats.ndim == 2 else 0
        return np.zeros((d, d), dtype=np.float32)
    normed = _l2_normalize_rows(feats)
    g = (normed.T @ normed) / normed.shape[0]
    return g.astype(np.float32)


def gram_score_map(
    query_features: "np.ndarray", gram: "np.ndarray"
) -> "np.ndarray":
    """Gram-refined channel-energy score map (paper §4).

    ``Q̂ = G^c Q`` (refine the query channels by the class Gram), then per-patch
    energy ``S^c_gram(u,v) = Σ_j Q_j(u,v) Q̂_j(u,v)``.

    Args:
        query_features: ``(Hp, Wp, D)`` query dense features (``Q`` is this
            reshaped to ``ℝ^{d×hw}``).
        gram: ``(D, D)`` class Gram matrix.

    Returns:
        ``(Hp, Wp)`` Gram-energy score map.
    """
    import numpy as np

    feats = np.asarray(query_features, dtype=np.float64)
    hp, wp, d = feats.shape
    q = feats.reshape(hp * wp, d)  # (N, D)
    g = np.asarray(gram, dtype=np.float64)
    q_hat = q @ g.T  # (N, D) — Q̂ = G Q (row-wise)
    energy = np.einsum("nd,nd->n", q, q_hat)  # Σ_j Q_j Q̂_j
    return energy.reshape(hp, wp).astype(np.float32)


def _cosine_map(
    query_features: "np.ndarray", prototype: "np.ndarray"
) -> "np.ndarray":
    """Per-patch cosine similarity of query features to one prototype."""
    import numpy as np

    feats = np.asarray(query_features, dtype=np.float64)
    hp, wp, d = feats.shape
    flat = feats.reshape(hp * wp, d)
    fn = np.linalg.norm(flat, axis=1)
    pn = np.linalg.norm(prototype)
    denom = fn * pn
    safe = np.where(denom == 0, 1.0, denom)
    sim = (flat @ np.asarray(prototype, dtype=np.float64)) / safe
    sim[denom == 0] = 0.0
    return sim.reshape(hp, wp).astype(np.float32)


def _min_max_normalize(m: "np.ndarray") -> "np.ndarray":
    """Min-max normalise a map to ``[0, 1]`` (constant map → zeros)."""
    import numpy as np

    arr = np.asarray(m, dtype=np.float64)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def combine_score_maps(maps: List["np.ndarray"]) -> "np.ndarray":
    """Combine per-class score maps: ``S̃_mean ⊙ S̃_max`` (paper §4).

    Each map is min-max normalised to ``[0,1]``; the class score is the
    elementwise (Hadamard) product of their mean and their max.

    Args:
        maps: The class's score maps ``{S^c_1, …, S^c_{n_c}, S^c_gram}``, all
            the same shape.

    Returns:
        ``(Hp, Wp)`` combined class score in ``[0, 1]``.
    """
    import numpy as np

    if not maps:
        raise ValueError("combine_score_maps requires at least one map")
    stack = np.stack([_min_max_normalize(m) for m in maps], axis=0)
    s_mean = stack.mean(axis=0)
    s_max = stack.max(axis=0)
    return (s_mean * s_max).astype(np.float32)  # Hadamard product


def assign_foreground(
    fg_score: "np.ndarray", bg_score: "np.ndarray", similarity_thresh: float
) -> "np.ndarray":
    """Per-pixel foreground assignment: ``argmax_c`` + a fg-score floor.

    ``ŷ(u,v) = argmax_c S̃^c_score`` (paper §4) restricted to the binary case:
    foreground where the fg class score exceeds the bg class score. The
    ``similarity_thresh`` floor (the documented deviation) additionally requires
    ``fg_score > similarity_thresh`` so a knob exists for the unsupervised case.

    Args:
        fg_score: ``(Hp, Wp)`` combined foreground class score (in [0, 1]).
        bg_score: ``(Hp, Wp)`` combined background class score (in [0, 1]).
        similarity_thresh: Minimum fg score (a floor on top of the argmax).

    Returns:
        ``(Hp, Wp)`` boolean foreground mask.
    """
    import numpy as np

    fg = np.asarray(fg_score, dtype=np.float64)
    bg = np.asarray(bg_score, dtype=np.float64)
    return ((fg > bg) & (fg > float(similarity_thresh))).astype(bool)


class FssDinoDetector(GpuDetector):
    """Detect colonies by FSSDINO few-shot semantic segmentation.

    A **training-free, few-shot** semantic detector on a frozen DINO backbone.
    From a small **support set** (annotated RGB images + masks), FSSDINO builds
    ``n_clusters`` class-specific **prototypes** (k-means, cosine) plus a class
    **Gram matrix** (channel co-occurrence), then scores each query patch by
    cosine to the prototypes *and* a Gram-refined channel energy, combines the
    maps (mean ⊙ max), and assigns foreground by ``argmax`` over the foreground
    vs background classes. The result is a boolean ``objmask``; the repo's
    downstream watershed (``SeparateObjects``) turns it into instances.

    Because FSSDINO emits a **semantic** mask (``output_kind="semantic"``), it
    writes ``image.objmask`` (auto-labelled into the shared ``objmap`` backend
    like a threshold detector — Spec 1 §8), not its own instance labels.

    A **curated colony exemplar** (a reference RGB crop + its mask, rendered
    once from :func:`phenotypic.data.load_synth_yeast_plate`) ships with the
    package and is the **default** single-shot support set, so the detector
    works out of the box; pass your own ``support_images``/``support_masks`` for
    a true few-shot set.

    The model is loaded lazily on the first ``apply()`` call, so the detector
    serialises and round-trips through JSON without a GPU, ``torch``, or
    ``transformers`` installed.

    **Backbone.** ``dino_version=2`` (DINOv2, Apache, ungated) is the default and
    runs gate-free. ``dino_version=3`` selects the gated DINOv3 backbone
    (accept the DINOv3 License + ``hf auth login`` — see
    :doc:`/how_to/pages/gpu_detection_setup`).

    **Layer selection.** FSSDINO's central finding (the "Semantic Selection
    Gap") is that intermediate layers often carry stronger semantics than the
    last, but no unsupervised heuristic reliably picks them — so the paper
    recommends the **last layer as the safe default** (``feature_layer=-1``).

    Args:
        support_images: Paths to the support RGB images. Defaults to the bundled
            curated colony exemplar (a one-shot support set).
        support_masks: Paths to each support image's binary foreground mask
            (aligned with *support_images*). Defaults to the bundled exemplar's
            mask.
        n_clusters: Prototypes per class (k-means; paper default 5).
        feature_layer: Transformer hidden-state index for the dense features
            (``-1`` = last layer, the paper's safe default; intermediate layers
            may do better but are not reliably selectable unsupervised).
        dino_version: DINO backbone generation. ``2`` = DINOv2 (Apache, ungated,
            default); ``3`` = DINOv3 (gated opt-in).
        dino_size: DINO backbone size (``"small"``/``"base"``/``"large"``).
        similarity_thresh: Foreground-score floor on top of the fg-vs-bg
            ``argmax`` (the documented binary-case deviation). Default 0.5.
        tile_px: Nominal tile size in pixels for large-plate tiling; images that
            fit one tile run un-tiled. Default 512 (FSSDINO's default
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
        ValueError: If the support set is empty or the image/mask lists differ
            in length.
        RuntimeError: If ``device="auto"`` and no accelerator is available, or
            (``dino_version=3``) the gated DINOv3 license was not accepted.

    Best For:
        * Few-shot transfer from a handful of annotated plates to a batch of
          similar plates, capturing intra-class appearance variety via the k
          prototypes.

    Consider Also:
        * :class:`~phenotypic.detect.nn.Insid3Detector` for a **one-shot**
          in-context semantic detector with explicit positional-bias removal.
        * :class:`~phenotypic.detect.nn.DinoSam2Detector` for an instance-native
          ungated detector.

    References:
        [1] H. M. Zakir and E. T. W. Ho, "Revealing the Semantic Selection Gap
        in DINOv3 through Training-Free Few-Shot Segmentation,"
        *arXiv:2602.07550*, 2026 (CC BY-NC-SA; clean-room reimplementation, no
        code vendored).
        [2] M. Oquab et al., "DINOv2," *arXiv:2304.07193*, 2023.

    See Also:
        :doc:`/how_to/pages/gpu_detection_setup`
            Installation, backbone selection, and the support-set interface.

    Examples:
        Construct a detector and confirm the ungated DINOv2 default + last-layer
        recommendation (no GPU or weights required):

        >>> from phenotypic.detect.nn import FssDinoDetector
        >>> det = FssDinoDetector()
        >>> det.output_kind
        'semantic'
        >>> det.dino_version
        2
        >>> det.feature_layer
        -1

        Build a pipeline and serialise it to JSON (round-trips without GPU
        dependencies):

        >>> from phenotypic import ImagePipeline
        >>> pipe = ImagePipeline(ops=[FssDinoDetector(n_clusters=3)])
        >>> restored = ImagePipeline.from_json(pipe.to_json())
        >>> type(restored.get_ops()["FssDinoDetector"])
        <class 'phenotypic.detect.nn._fssdino_detector.FssDinoDetector'>
    """

    # Capabilities — few-shot, semantic, per-image.
    input_layer: GpuInputLayer = "rgb"
    output_kind: GpuOutputKind = "semantic"
    supports_batching: bool = False

    support_images: List[Path] = []
    support_masks: List[Path] = []
    n_clusters: Annotated[int, TuneSpec(1, 20)] = 5  # paper sets n_c = 5
    # FSSDINO layer-selection: -1 (last layer) is the paper's safe default.
    # The optimal intermediate layer is scene-dependent (the "Semantic
    # Selection Gap") → declare intent-to-tune without a fixed window.
    feature_layer: Annotated[int, TuneSpec()] = -1
    dino_version: DinoVersion = 2  # DINOv2 default, ungated (Spec §4.4)
    dino_size: DinoSize = "base"
    similarity_thresh: Annotated[float, TuneSpec(0.0, 1.0)] = 0.5

    # Tiling (shared with the instance detectors via _tiling).
    tile_px: Annotated[int, TuneSpec(512, 2048)] = 512
    tile_overlap: Annotated[float, TuneSpec(0.0, 0.4)] = 0.15

    device: Device = "auto"

    # Lazy runtime state — PrivateAttr → skipped by serialization.
    _model: Any = PrivateAttr(default=None)
    _processor: Any = PrivateAttr(default=None)
    _device: Any = PrivateAttr(default=None)
    _fg_prototypes: Any = PrivateAttr(default=None)
    _bg_prototypes: Any = PrivateAttr(default=None)
    _fg_gram: Any = PrivateAttr(default=None)
    _bg_gram: Any = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Fill in the bundled colony exemplar default when no support is set."""
        if (
            "support_images" not in self.model_fields_set
            and "support_masks" not in self.model_fields_set
        ):
            from phenotypic._assets import colony_exemplar_paths

            rgb_path, mask_path = colony_exemplar_paths()
            object.__setattr__(self, "support_images", [rgb_path])
            object.__setattr__(self, "support_masks", [mask_path])

    # ------------------------------------------------------------------
    # Lazy load + prototype/Gram caching
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Load the frozen DINO backbone and cache the fg/bg prototypes + Gram.

        Validates the support set, loads the backbone (gated DINOv3 pre-staged),
        extracts per-support dense features at ``feature_layer``, gathers
        foreground / background patch features across the support set, and
        builds the ``n_clusters`` class prototypes + class Gram matrices.
        Idempotent.
        """
        if self._fg_prototypes is not None:
            return
        if not self.support_images or not self.support_masks:
            raise ValueError(
                "FssDinoDetector requires a non-empty support set "
                "(support_images + support_masks)."
            )
        if len(self.support_images) != len(self.support_masks):
            raise ValueError(
                "FssDinoDetector support_images and support_masks must have "
                f"equal length ({len(self.support_images)} != "
                f"{len(self.support_masks)})."
            )

        import numpy as np

        from phenotypic.detect.nn._checkpoint_manager import resolve_device
        from phenotypic.detect.nn._dino_support import (
            align_mask_to_grid,
            extract_reference_features,
            load_dino_backbone,
        )

        self._device = resolve_device(self.device)
        self._model, self._processor = load_dino_backbone(
            self.dino_version, self.dino_size, self._device
        )

        fg_feats: list[np.ndarray] = []
        bg_feats: list[np.ndarray] = []
        for img_path, mask_path in zip(self.support_images, self.support_masks):
            rgb = self._read_rgb(img_path)
            mask = self._read_mask(mask_path)
            # W4: capture the processed geometry so a (possibly non-square)
            # support mask aligns through the same resize as the image.
            dense, proc_hw = extract_reference_features(
                self._model,
                self._processor,
                rgb,
                device=self._device,
                layer=self.feature_layer,
            )
            hp, wp, _d = dense.shape
            grid_mask = align_mask_to_grid(mask, proc_hw, (hp, wp))
            flat = dense.reshape(hp * wp, dense.shape[-1])
            fg_idx = grid_mask.reshape(-1)
            fg_feats.append(flat[fg_idx])
            bg_feats.append(flat[~fg_idx])

        fg_all = (
            np.concatenate(fg_feats, axis=0)
            if any(f.shape[0] for f in fg_feats)
            else np.zeros((0, 1), np.float32)
        )
        bg_all = (
            np.concatenate(bg_feats, axis=0)
            if any(f.shape[0] for f in bg_feats)
            else np.zeros((0, 1), np.float32)
        )
        self._fg_prototypes = cluster_prototypes(fg_all, self.n_clusters)
        self._bg_prototypes = cluster_prototypes(bg_all, self.n_clusters)
        self._fg_gram = gram_matrix(fg_all)
        self._bg_gram = gram_matrix(bg_all)

    @staticmethod
    def _read_rgb(path: Path) -> "np.ndarray":
        """Read a support RGB image as ``(H, W, 3)`` uint8."""
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
        """Read a support mask as a ``(H, W)`` boolean array."""
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
        """Score a query against the fg/bg classes → boolean objmask (tiled)."""

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
            return self._segment_crop(rgb)

        tile_masks: List[np.ndarray] = []
        for t in tiles:
            crop = rgb[t.y0:t.y1, t.x0:t.x1]
            tile_masks.append(self._segment_crop(crop))
        return stitch_semantic_tiles(
            tiles, tile_masks, (rgb.shape[0], rgb.shape[1])
        )

    def _segment_crop(self, rgb: "np.ndarray") -> "np.ndarray":
        """Run the FSSDINO scoring on one crop → full-res boolean mask."""
        import numpy as np
        from skimage.transform import resize

        from phenotypic.detect.nn._dino_support import (
            extract_hidden_layer_features,
        )

        dense = extract_hidden_layer_features(
            self._model,
            self._processor,
            rgb,
            device=self._device,
            layer=self.feature_layer,
        )
        fg_score = self._class_score(dense, self._fg_prototypes, self._fg_gram)
        bg_score = self._class_score(dense, self._bg_prototypes, self._bg_gram)
        # Argmax over classes (+ fg floor) on the patch grid, then upsample.
        grid_mask = assign_foreground(fg_score, bg_score, self.similarity_thresh)
        full = resize(
            grid_mask.astype(np.float32),
            (rgb.shape[0], rgb.shape[1]),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        )
        return (full > 0.5).astype(bool)

    def _class_score(
        self,
        dense: "np.ndarray",
        prototypes: "np.ndarray",
        gram: "np.ndarray",
    ) -> "np.ndarray":
        """Combined class score map ``S̃^c_score`` for one class (paper §4)."""
        import numpy as np

        maps: list[np.ndarray] = [
            _cosine_map(dense, p) for p in np.asarray(prototypes)
        ]
        maps.append(gram_score_map(dense, gram))
        if not maps:
            return np.zeros(dense.shape[:2], np.float32)
        return combine_score_maps(maps)

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
FssDinoDetector.apply.__doc__ = FssDinoDetector.__doc__
