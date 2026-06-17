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
