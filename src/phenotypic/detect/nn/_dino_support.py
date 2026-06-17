"""Shared frozen-DINO backbone + prototype-match core (Spec 2b, Task 2).

INSID3 and FSSDINO are thin specialisations of one training-free recipe on a
**frozen DINO** backbone: extract dense patch features, pool an exemplar
prototype from a reference/support mask, cosine-match query patches, upsample,
and threshold to a boolean ``objmask``. This module centralises that core
(clean-room, generic) so the two detectors stay small.

It also owns the **single correct** DINO dense-feature extraction — the C1 fix.
DINOv3's ``last_hidden_state`` is ``(B, 1 + num_register_tokens + Hp*Wp, D)``
with ``config.num_register_tokens == 4`` (0 for DINOv2). Slicing only the CLS
token (``[:, 1:, :]``) leaves the 4 register tokens contaminating the patch
grid; we slice ``[:, 1 + num_register_tokens:, :]`` here, in one place, and
``DinoSam2Detector`` calls it too (its buggy private copy is deleted).

All heavy imports (``torch``, ``transformers``) are **lazy** so detectors
construct and serialise without them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Tuple

if TYPE_CHECKING:
    import numpy as np

    from phenotypic.tools_.typing_ import DinoSize, DinoVersion


# ---------------------------------------------------------------------------
# Backbone id mapping (single source of truth for both DINO versions)
# ---------------------------------------------------------------------------

#: DINOv3 size → gated Hugging Face repo id (LVD-1689M ViT-{S,B,L}/16).
_DINOV3_SIZE_TO_REPO: dict[str, str] = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


def hf_dino_id(dino_version: "DinoVersion", dino_size: "DinoSize") -> str:
    """Map ``(dino_version, dino_size)`` to the Hugging Face backbone id.

    Args:
        dino_version: ``2`` (DINOv2, Apache/ungated) or ``3`` (DINOv3, gated).
        dino_size: ``"small"`` / ``"base"`` / ``"large"``.

    Returns:
        ``"facebook/dinov2-{size}"`` for v2, or
        ``"facebook/dinov3-vit{s|b|l}16-pretrain-lvd1689m"`` for v3.
    """
    if dino_version == 2:
        return f"facebook/dinov2-{dino_size}"
    return _DINOV3_SIZE_TO_REPO[dino_size]


# ---------------------------------------------------------------------------
# Frozen-DINO backbone load (lazy transformers; gated v3 pre-staged)
# ---------------------------------------------------------------------------


def load_dino_backbone(
    dino_version: "DinoVersion",
    dino_size: "DinoSize",
    device: str,
) -> Tuple[Any, Any]:
    """Load a frozen DINO backbone ``(model, processor)`` on ``device``.

    Lazy-imports ``transformers`` so callers construct/serialise without it.
    ``dino_version=3`` is gated: the snapshot is pre-staged through
    :class:`~phenotypic.detect.nn._checkpoint_manager.Dinov3CheckpointManager`
    (honouring the license-acceptance gate) before the ``AutoModel`` load.

    Args:
        dino_version: ``2`` (DINOv2, ungated) or ``3`` (DINOv3, gated).
        dino_size: ``"small"`` / ``"base"`` / ``"large"``.
        device: Resolved PyTorch device string (e.g. ``"cuda"``, ``"cpu"``).

    Returns:
        ``(model, processor)`` — an ``AutoModel`` on ``device`` (eval mode) and
        its ``AutoImageProcessor``.

    Raises:
        ImportError: If ``transformers`` is not installed.
        RuntimeError: For ``dino_version=3``, if the gated license was not
            accepted or no Hugging Face token is present.
    """
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError:
        raise ImportError(
            "Frozen-DINO detectors require transformers (>=5.2.0). "
            "Install with: pip install phenotypic[foundation]"
        ) from None

    if dino_version == 3:
        from phenotypic.detect.nn._checkpoint_manager import (
            Dinov3CheckpointManager,
        )

        Dinov3CheckpointManager(size=dino_size).download()

    dino_id = hf_dino_id(dino_version, dino_size)
    model = AutoModel.from_pretrained(dino_id).to(device)
    model.eval()
    processor = AutoImageProcessor.from_pretrained(dino_id)
    return model, processor


# ---------------------------------------------------------------------------
# C1 — register-token-aware patch-grid reshape
# ---------------------------------------------------------------------------


def reshape_patch_tokens(
    tokens: "np.ndarray",
    grid_hw: Tuple[int, int],
    num_register_tokens: int,
) -> "np.ndarray":
    """Reshape a ViT token sequence to the ``(Hp, Wp, D)`` patch grid.

    DINOv3's ``last_hidden_state`` is ``(1 + num_register_tokens + Hp*Wp, D)``
    (``config.num_register_tokens == 4``); DINOv2 has none. Dropping only the
    CLS token (``tokens[1:]``) leaves the register tokens contaminating the
    grid — **the C1 bug**. This drops CLS *and* the register tokens, then
    reshapes the remaining patch tokens row-major.

    Args:
        tokens: ``(n_tokens, D)`` sequence for ONE image (CLS first, then
            ``num_register_tokens`` registers, then ``Hp*Wp`` patch tokens).
        grid_hw: ``(Hp, Wp)`` patch grid inferred from the processed image.
        num_register_tokens: ``config.num_register_tokens`` (0 for DINOv2).

    Returns:
        ``(Hp, Wp, D)`` dense patch features in row-major patch order.
    """
    import numpy as np

    hp, wp = int(grid_hw[0]), int(grid_hw[1])
    start = 1 + int(num_register_tokens)
    patch = np.asarray(tokens)[start:]
    n_expected = hp * wp
    if patch.shape[0] != n_expected:
        # Defensive: if the inferred grid disagrees with the token count, fall
        # back to a square grid over whatever patch tokens remain. (Should not
        # happen once num_register_tokens is honoured.)
        side = int(round(patch.shape[0] ** 0.5))
        hp, wp = side, patch.shape[0] // side
        patch = patch[: hp * wp]
    return patch.reshape(hp, wp, patch.shape[-1])


def extract_patch_features(
    model: Any, processor: Any, rgb_uint8: "np.ndarray", *, device: str
) -> "np.ndarray":
    """Run a frozen DINO ViT on one RGB image → dense ``(Hp, Wp, D)`` features.

    Drops the CLS **and** register tokens (the C1 fix) via
    :func:`reshape_patch_tokens`, inferring the patch grid from the processed
    pixel geometry and ``config.patch_size``.

    Args:
        model: A loaded ``AutoModel`` (DINOv2/DINOv3) on ``device``.
        processor: The matching ``AutoImageProcessor``.
        rgb_uint8: ``(H, W, 3)`` uint8 RGB image.
        device: Resolved PyTorch device string.

    Returns:
        ``(Hp, Wp, D)`` float32 dense patch features.
    """
    import numpy as np
    import torch

    inputs = processor(images=rgb_uint8, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    tokens = outputs.last_hidden_state[0].detach().cpu().numpy().astype(np.float32)

    pixel = inputs["pixel_values"]
    in_h, in_w = int(pixel.shape[-2]), int(pixel.shape[-1])
    patch = int(getattr(model.config, "patch_size", 16))
    grid_hw = (in_h // patch, in_w // patch)
    n_reg = int(getattr(model.config, "num_register_tokens", 0))
    return reshape_patch_tokens(tokens, grid_hw, n_reg)


def extract_hidden_layer_features(
    model: Any,
    processor: Any,
    rgb_uint8: "np.ndarray",
    *,
    device: str,
    layer: int,
) -> "np.ndarray":
    """Like :func:`extract_patch_features` but from an intermediate layer.

    Uses ``output_hidden_states=True`` and selects ``hidden_states[layer]``
    (FSSDINO's layer-selection finding — intermediate layers carry stronger
    semantics than the last). ``layer=-1`` is the last layer.

    Args:
        model: A loaded ``AutoModel`` (DINOv2/DINOv3) on ``device``.
        processor: The matching ``AutoImageProcessor``.
        rgb_uint8: ``(H, W, 3)`` uint8 RGB image.
        device: Resolved PyTorch device string.
        layer: Index into ``hidden_states`` (0 = embeddings, 1..L = blocks;
            negative indexes from the end). FSSDINO recommends the last block.

    Returns:
        ``(Hp, Wp, D)`` float32 dense patch features from ``hidden_states[layer]``.
    """
    import numpy as np
    import torch

    inputs = processor(images=rgb_uint8, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer]
    tokens = hidden[0].detach().cpu().numpy().astype(np.float32)

    pixel = inputs["pixel_values"]
    in_h, in_w = int(pixel.shape[-2]), int(pixel.shape[-1])
    patch = int(getattr(model.config, "patch_size", 16))
    grid_hw = (in_h // patch, in_w // patch)
    n_reg = int(getattr(model.config, "num_register_tokens", 0))
    return reshape_patch_tokens(tokens, grid_hw, n_reg)


# ---------------------------------------------------------------------------
# Mask ↔ patch-grid alignment (W4)
# ---------------------------------------------------------------------------


def resize_mask_to_grid(
    mask: "np.ndarray", grid_hw: Tuple[int, int]
) -> "np.ndarray":
    """Downsample a full-resolution boolean mask onto the patch grid.

    Nearest-neighbour (``order=0``) so labels are not interpolated. The caller
    is responsible for first aligning the mask to the same geometry the image
    went through the processor (W4); this is the final patch-grid step.

    Args:
        mask: ``(H, W)`` boolean (or 0/1) mask.
        grid_hw: ``(Hp, Wp)`` target patch grid.

    Returns:
        ``(Hp, Wp)`` boolean mask.
    """
    import numpy as np
    from skimage.transform import resize

    hp, wp = int(grid_hw[0]), int(grid_hw[1])
    small = (
        resize(
            np.asarray(mask, dtype=np.float32),
            (hp, wp),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        )
        > 0.5
    )
    return small.astype(bool)


# ---------------------------------------------------------------------------
# Prototype pooling + cosine match (the shared core)
# ---------------------------------------------------------------------------


def pool_prototype(
    features: "np.ndarray", mask: "np.ndarray"
) -> "np.ndarray":
    """Masked-mean an exemplar prototype from dense patch features.

    The mask is resized to the patch grid (W4 — handles a full-resolution
    exemplar mask), then the foreground patch features are averaged.

    Args:
        features: ``(Hp, Wp, D)`` dense patch features.
        mask: ``(Hm, Wm)`` boolean mask (any resolution; resized to the grid).

    Returns:
        ``(D,)`` mean-pooled prototype (zero vector if the mask is empty —
        the fail-safe degenerate case).
    """
    import numpy as np

    feats = np.asarray(features, dtype=np.float32)
    hp, wp, d = feats.shape
    small = resize_mask_to_grid(np.asarray(mask), (hp, wp))
    idx = small.reshape(-1)
    flat = feats.reshape(hp * wp, d)
    fg = flat[idx]
    if fg.shape[0] == 0:
        return np.zeros(d, dtype=np.float32)
    return fg.mean(axis=0).astype(np.float32)


def cosine_similarity_map(
    features: "np.ndarray", prototype: "np.ndarray"
) -> "np.ndarray":
    """Per-patch cosine similarity of dense features to a prototype.

    Args:
        features: ``(Hp, Wp, D)`` dense patch features.
        prototype: ``(D,)`` prototype vector.

    Returns:
        ``(Hp, Wp)`` cosine similarities in ``[-1, 1]`` (0 where either the
        patch feature or the prototype is a zero vector — the fail-safe).
    """
    import numpy as np

    feats = np.asarray(features, dtype=np.float64)
    proto = np.asarray(prototype, dtype=np.float64)
    hp, wp, d = feats.shape
    flat = feats.reshape(hp * wp, d)
    fn = np.linalg.norm(flat, axis=1)
    pn = np.linalg.norm(proto)
    denom = fn * pn
    safe = np.where(denom == 0, 1.0, denom)
    sim = (flat @ proto) / safe
    sim[denom == 0] = 0.0
    return sim.reshape(hp, wp).astype(np.float32)


def cosine_match_to_mask(
    features: "np.ndarray",
    prototype: "np.ndarray",
    thresh: float,
    out_shape: Tuple[int, int],
) -> "np.ndarray":
    """Cosine-match query patches to a prototype → an upsampled boolean mask.

    Computes the per-patch cosine map, bilinearly upsamples it to
    ``out_shape``, and thresholds at ``thresh``. A zero prototype (empty
    reference mask) yields an all-False mask (fail-safe).

    Args:
        features: ``(Hp, Wp, D)`` query dense patch features.
        prototype: ``(D,)`` exemplar prototype.
        thresh: Cosine-similarity cutoff (foreground where ``sim > thresh``).
        out_shape: ``(H, W)`` of the full-resolution output mask.

    Returns:
        ``(H, W)`` boolean ``objmask``.
    """
    import numpy as np
    from skimage.transform import resize

    proto = np.asarray(prototype, dtype=np.float64)
    if not np.any(proto):  # degenerate zero-prototype → fail safe
        return np.zeros(out_shape, dtype=bool)

    sim = cosine_similarity_map(features, proto)
    sim_full = resize(
        sim.astype(np.float32),
        out_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=False,
    )
    return (sim_full > thresh).astype(bool)
