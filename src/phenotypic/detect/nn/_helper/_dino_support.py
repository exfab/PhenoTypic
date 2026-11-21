"""Shared frozen-DINO backbone + prototype-match core (Spec 2b, Task 2).

INSID3 and FSSDINO are thin specialisations of one training-free recipe on a
**frozen DINO** backbone: extract dense patch features, pool an exemplar
prototype from a reference/support mask, cosine-match query patches, upsample,
and threshold to a boolean ``objmask``. This module centralises that core
(clean-room, generic) so the two detectors stay small.

It also owns the **single correct** DINO dense-feature extraction — the C1 fix.
DINOv3's ``last_hidden_state`` is ``(B, 1 + num_register_tokens + Hp*Wp, D)``
with ``config.num_register_tokens == 4`` (0 for the ``facebook/dinov2-*``
checkpoints; ``dinov2-with-registers`` has 4). Slicing only the CLS
token (``[:, 1:, :]``) leaves the 4 register tokens contaminating the patch
grid; we slice ``[:, 1 + num_register_tokens:, :]`` here, in one place, and
``DinoSam2Detector`` calls it too (its buggy private copy is deleted).

It is likewise the single **resize policy**: every extract call pins the model
input to the tile's native geometry (:data:`NATIVE_PROCESSOR_KWARGS`) instead of
the checkpoint's 224-px classification preset, and every grid↔image mapping goes
through the *covered* extent ``hp*patch x wp*patch`` (:func:`covered_hw`,
:func:`upsample_grid_to_image`) rather than the full tile.

All heavy imports (``torch``, ``transformers``) are **lazy** so detectors
construct and serialise without them.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Collection
from typing import TYPE_CHECKING, Any, List, Literal, Tuple

if TYPE_CHECKING:
    import numpy as np

    from phenotypic.detect.nn._helper._tiling import _Tile
    from phenotypic.sdk_.typing_ import ComputeDType, DinoVersion, EmbeddingDinoSize


# ---------------------------------------------------------------------------
# Backbone id mapping (single source of truth for both DINO versions)
# ---------------------------------------------------------------------------

#: DINOv3 size → gated Hugging Face repo id (LVD-1689M ViT-{S,B,L}/16).
_DINOV3_SIZE_TO_REPO: dict[str, str] = {
    "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}


#: Processor kwargs that pin the model input to the tile's native geometry.
#: Without these, ``AutoImageProcessor`` applies the checkpoint's *classification*
#: preset — DINOv2 resizes shortest-edge to 256 then center-crops to 224; DINOv3
#: resizes squarely to 224 — so every tile reaches the ViT at 224x224 regardless
#: of ``tile_px``. Under these kwargs the patch grid is ``(h // patch, w // patch)``
#: and native px per patch is exactly ``patch_size``.
NATIVE_PROCESSOR_KWARGS: dict[str, bool] = {
    "do_resize": False,
    "do_center_crop": False,
}


def backbone_patch_size(model: Any) -> int:
    """Return the loaded backbone's ViT patch size, refusing to guess.

    Every geometry helper here needs the patch size, and getting it wrong is a
    **silent** sub-patch scale error rather than a crash. A ``getattr(...,
    "patch_size", 14)``-style fallback is therefore worse than useless: DINOv2 is
    patch-14 and DINOv3 is patch-16, so a per-call-site default can disagree with
    itself within one run (the detector guessing 14 while this module guesses
    16). Read it from the model, or fail loudly.

    Args:
        model: A loaded ``AutoModel`` (DINOv2/DINOv3).

    Returns:
        ``model.config.patch_size``.

    Raises:
        ValueError: If the backbone exposes no ``config.patch_size``.
    """
    patch = getattr(getattr(model, "config", None), "patch_size", None)
    if patch is None:
        raise ValueError(
            f"{type(model).__name__} exposes no config.patch_size; the patch "
            "grid cannot be derived and guessing it would silently rescale "
            "every mask. Load a DINOv2/DINOv3 backbone."
        )
    return int(patch)


def patch_grid_hw(pixel_hw: Tuple[int, int], patch: int) -> Tuple[int, int]:
    """Return the ``(Hp, Wp)`` patch grid a ViT produces for ``pixel_hw``.

    Patch embedding is a stride-``patch`` convolution, so it floors: a
    non-multiple input silently drops up to ``patch - 1`` pixels off the bottom
    and right. Use :func:`covered_hw` for the extent the grid actually spans.

    Args:
        pixel_hw: ``(H, W)`` of the tensor handed to the model.
        patch: ``model.config.patch_size`` (14 for DINOv2, 16 for DINOv3).

    Returns:
        ``(Hp, Wp)`` patch-grid shape.
    """
    return (int(pixel_hw[0]) // int(patch), int(pixel_hw[1]) // int(patch))


def covered_hw(grid_hw: Tuple[int, int], patch: int) -> Tuple[int, int]:
    """Return the pixel extent a ``grid_hw`` patch grid actually covers.

    ``(hp * patch, wp * patch)`` — never the original ``(H, W)``. Mapping a grid
    onto ``(H, W)`` instead introduces a scale error of up to
    ``(patch - 1) / H`` (2.0% vertically for a 600-px tile at patch 14).

    Args:
        grid_hw: ``(Hp, Wp)`` patch grid.
        patch: ``model.config.patch_size``.

    Returns:
        ``(Hp * patch, Wp * patch)``.
    """
    return (int(grid_hw[0]) * int(patch), int(grid_hw[1]) * int(patch))


def upsample_grid_to_image(
    grid: "np.ndarray",
    image_hw: Tuple[int, int],
    patch: int,
    *,
    order: int = 0,
) -> "np.ndarray":
    """Upsample a patch grid to full resolution through its covered extent.

    The grid spans ``covered_hw(grid.shape, patch)``, **not** ``image_hw``: a
    stride-``patch`` conv floors, so up to ``patch - 1`` rows/columns at the
    bottom/right were never seen by the model. Resizing straight to ``image_hw``
    stretches the grid over pixels it does not describe (a 2.0% vertical scale
    error for a 600-px tile at patch 14). Resize to the covered extent, then
    edge-pad the remainder.

    Args:
        grid: ``(Hp, Wp)`` patch-grid array (boolean mask or float score map).
        image_hw: ``(H, W)`` of the tile the grid came from.
        patch: ``model.config.patch_size``.
        order: Interpolation order — ``0`` (nearest) for masks, ``1`` for
            score maps.

    Returns:
        ``(H, W)`` array with ``grid``'s dtype semantics preserved.
    """
    import numpy as np
    from skimage.transform import resize

    h, w = int(image_hw[0]), int(image_hw[1])
    arr = np.asarray(grid)
    ch, cw = covered_hw(arr.shape[:2], patch)
    is_bool = arr.dtype == bool

    covered = resize(
        arr.astype(np.float32),
        (ch, cw),
        order=order,
        preserve_range=True,
        anti_aliasing=False,
    )
    if (ch, cw) != (h, w):
        covered = np.pad(
            covered, ((0, max(0, h - ch)), (0, max(0, w - cw))), mode="edge"
        )[:h, :w]
    return (covered > 0.5) if is_bool else covered


def hf_dino_id(dino_version: "DinoVersion", dino_size: "EmbeddingDinoSize") -> str:
    """Map ``(dino_version, dino_size)`` to the Hugging Face backbone id.

    Args:
        dino_version: ``2`` (DINOv2, Apache/ungated) or ``3`` (DINOv3, gated).
        dino_size: ``"small"`` / ``"base"`` / ``"large"``, or ``"giant"``
            (ViT-g/14) for DINOv2 only.

    Returns:
        ``"facebook/dinov2-{size}"`` for v2 (``giant`` is
        ``"facebook/dinov2-giant"``), or
        ``"facebook/dinov3-vit{s|b|l}16-pretrain-lvd1689m"`` for v3.

    Raises:
        ValueError: For a DINOv3 size it does not have (``giant``).
    """
    if dino_version == 2:
        return f"facebook/dinov2-{dino_size}"
    if dino_size not in _DINOV3_SIZE_TO_REPO:
        raise ValueError(
            f"DINOv3 has no {dino_size!r} backbone; it has "
            f"{', '.join(repr(size) for size in _DINOV3_SIZE_TO_REPO)}"
        )
    return _DINOV3_SIZE_TO_REPO[dino_size]


# ---------------------------------------------------------------------------
# Frozen-DINO backbone load (lazy transformers; gated v3 pre-staged)
# ---------------------------------------------------------------------------


def load_dino_backbone(
    dino_version: "DinoVersion",
    dino_size: "EmbeddingDinoSize",
    device: str,
    *,
    revision: str | None = None,
    dtype: "ComputeDType" = "float32",
) -> Tuple[Any, Any]:
    """Load a frozen DINO backbone ``(model, processor)`` on ``device``.

    Lazy-imports ``transformers`` so callers construct/serialise without it.
    ``dino_version=3`` is gated: the snapshot is pre-staged through
    :class:`~phenotypic.detect.nn._helper._checkpoint_manager.Dinov3CheckpointManager`
    (honouring the license-acceptance gate) before the ``AutoModel`` load.

    Args:
        dino_version: ``2`` (DINOv2, ungated) or ``3`` (DINOv3, gated).
        dino_size: ``"small"`` / ``"base"`` / ``"large"``, or ``"giant"``
            for DINOv2.
        device: Resolved PyTorch device string (e.g. ``"cuda"``, ``"cpu"``).
        revision: The Hugging Face revision (a commit, branch or tag) to load,
            passed to both ``from_pretrained`` calls. ``None`` loads the hub's
            default branch. :func:`resolved_revision` reports the commit that
            was actually loaded. Weights load from ``.safetensors`` only,
            whatever the revision, so a ref holding only pickle weights is
            refused rather than unpickled.
        dtype: The torch dtype the weights load in, ``from_pretrained``'s
            ``dtype=`` (#282 H7). The model is loaded in it on the host and
            then moved to *device*; no ``device_map``, which would need
            ``accelerate``. ``"float32"`` is what a load without it gives.

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
        from phenotypic.detect.nn._helper._checkpoint_manager import (
            Dinov3CheckpointManager,
        )

        # Never prompt inside a pipeline: a batch worker has no terminal, so
        # input() would raise EOFError or hang (spec §10.3, F10).
        Dinov3CheckpointManager(size=dino_size).download(interactive=False)

    dino_id = hf_dino_id(dino_version, dino_size)
    # safetensors only: any ref (a community refs/pr/N, say) may carry pickle
    # .bin weights, and unpickling runs code. Every facebook/dinov2-* and
    # dinov3-*-lvd1689m repo ships model.safetensors. Never trust_remote_code.
    model = AutoModel.from_pretrained(
        dino_id, revision=revision, use_safetensors=True, dtype=dtype
    ).to(device)
    model.eval()
    processor = AutoImageProcessor.from_pretrained(dino_id, revision=revision)
    return model, processor


@functools.lru_cache(maxsize=4)
def load_dino_processor(
    dino_version: "DinoVersion",
    dino_size: "EmbeddingDinoSize",
    *,
    revision: str | None = None,
) -> Any:
    """Load only the ``AutoImageProcessor`` of a DINO backbone: no model.

    What normalises a ``masked_dense`` tile on the CPU (#282): the processor
    :func:`load_dino_backbone` would return for the same arguments, without
    holding the weights in memory. Cached, one per ``(version, size,
    revision)`` per process. ``dino_version=3`` is gated and pre-staged
    exactly as :func:`load_dino_backbone` does it, so the license gate holds
    here too.

    Args:
        dino_version: ``2`` (DINOv2, ungated) or ``3`` (DINOv3, gated).
        dino_size: As :func:`load_dino_backbone`.
        revision: The Hugging Face revision, as :func:`load_dino_backbone`.

    Returns:
        The backbone's ``AutoImageProcessor``.

    Raises:
        ImportError: If ``transformers`` is not installed.
        RuntimeError: For ``dino_version=3``, if the gated license was not
            accepted or no Hugging Face token is present.
    """
    try:
        from transformers import AutoImageProcessor
    except ImportError:
        raise ImportError(
            "Frozen-DINO detectors require transformers (>=5.2.0). "
            "Install with: pip install phenotypic[foundation]"
        ) from None

    if dino_version == 3:
        from phenotypic.detect.nn._helper._checkpoint_manager import (
            Dinov3CheckpointManager,
        )

        Dinov3CheckpointManager(size=dino_size).download(interactive=False)
    return AutoImageProcessor.from_pretrained(
        hf_dino_id(dino_version, dino_size), revision=revision
    )


#: A full Hugging Face commit: 40 lowercase hex.
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")
#: The snapshot folder of a Hugging Face cache path, ``.../snapshots/<commit>/...``.
_SNAPSHOT_RE = re.compile(r"(?:^|[/\\])snapshots[/\\]([0-9a-f]{40})(?:[/\\]|$)")


def resolved_revision(model: Any, revision: str | None = None) -> str | None:
    """The Hugging Face commit a loaded backbone's weights came from.

    Tried in order, the first answer wins:

    1. ``model.config._commit_hash``, which ``from_pretrained`` records on
       transformers before 5.18. From 5.18 (huggingface-hub 2.x) it is no
       longer set, so the rest is the fallback that makes the revision
       independent of it (#281).
    2. *revision*, when it is a full 40-hex commit: ``from_pretrained``
       loaded that pin or raised, so the pin is the commit.
    3. A ``.../snapshots/<commit>/`` component of the config's or the
       model's ``name_or_path``: a backbone loaded from a cache snapshot
       directory.
    4. The local Hugging Face cache: the snapshot that *revision* (``main``
       when ``None``) resolves to for the repo ``name_or_path``, through
       ``huggingface_hub.try_to_load_from_cache``. It reads only the cache's
       ``refs/`` and ``snapshots/``, so it works offline and never reads or
       sends a token. ``from_pretrained`` just loaded from that cache, and
       updated the ref when it fetched, so it is the snapshot in use.

    Args:
        model: A loaded ``AutoModel``.
        revision: The revision it was loaded with, as passed to
            ``from_pretrained``.

    Returns:
        The 40-hex commit, or ``None`` when no source names one (a model
        built from a plain local directory, say). Never fabricated.
    """
    config = getattr(model, "config", None)
    commit = getattr(config, "_commit_hash", None)
    if commit:
        return str(commit)
    if isinstance(revision, str) and _COMMIT_RE.fullmatch(revision):
        return revision
    paths = [
        getattr(config, "name_or_path", None),
        getattr(config, "_name_or_path", None),
        getattr(model, "name_or_path", None),
    ]
    sources = [str(path) for path in paths if isinstance(path, str) and path]
    for source in sources:
        snapshot = _SNAPSHOT_RE.search(source)
        if snapshot:
            return snapshot.group(1)
    for source in dict.fromkeys(sources):
        commit = _cached_snapshot_commit(source, revision)
        if commit is not None:
            return commit
    return None


def _cached_snapshot_commit(repo_id: str, revision: str | None) -> str | None:
    """The commit *revision* resolves to for *repo_id* in the local Hugging Face cache.

    Offline and token-free: ``try_to_load_from_cache`` only reads the
    cache's ``refs/<revision>`` and ``snapshots/<commit>/config.json``.
    ``None`` when huggingface_hub is missing, *repo_id* is not a hub id, or
    the cache does not hold it.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    try:
        cached = try_to_load_from_cache(repo_id, "config.json", revision=revision)
    except Exception:  # noqa: BLE001 -- an id that is not a repo id, say
        return None
    if not isinstance(cached, str):
        return None
    snapshot = _SNAPSHOT_RE.search(cached)
    return snapshot.group(1) if snapshot else None


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
    (``config.num_register_tokens == 4``); the ``facebook/dinov2-*``
    checkpoints have none (``dinov2-with-registers`` has 4). Dropping only the
    CLS token (``tokens[1:]``) leaves the register tokens contaminating the
    grid — **the C1 bug**. This drops CLS *and* the register tokens, then
    reshapes the remaining patch tokens row-major.

    Args:
        tokens: ``(n_tokens, D)`` sequence for ONE image (CLS first, then
            ``num_register_tokens`` registers, then ``Hp*Wp`` patch tokens).
        grid_hw: ``(Hp, Wp)`` patch grid inferred from the processed image.
        num_register_tokens: ``config.num_register_tokens`` (0 for the
            ``facebook/dinov2-*`` checkpoints, 4 for DINOv3).

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
    pixel geometry and ``config.patch_size``. The image is fed at its native
    geometry (:data:`NATIVE_PROCESSOR_KWARGS`), not the checkpoint's 224-px
    classification preset.

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

    inputs = processor(
        images=rgb_uint8, return_tensors="pt", **NATIVE_PROCESSOR_KWARGS
    ).to(device)
    # inference_mode, as spec E3 asks of the embedder: every caller (DinoSam2,
    # InSID3; the embedder through patch_features_from_pixels) takes the
    # features as numpy, so none needs autograd.
    with torch.inference_mode():
        outputs = model(**inputs)
    tokens = outputs.last_hidden_state[0].detach().cpu().numpy().astype(np.float32)
    return _tokens_to_patch_grid(model, tokens, inputs["pixel_values"].shape[-2:])


def patch_features_from_pixels(
    model: Any, pixel_values: "np.ndarray", *, device: str
) -> "np.ndarray":
    """Run a frozen DINO ViT on one processed tile → dense ``(Hp, Wp, D)`` features.

    The second half of :func:`extract_patch_features`, for a tile its
    processor has already normalised (``processor(images=tile,
    return_tensors="np", **NATIVE_PROCESSOR_KWARGS)["pixel_values"][0]``), so
    the normalisation can run on the CPU apart from the model (#282). Given
    the processor's own pixels, the features are
    :func:`extract_patch_features`'s, bit for bit.

    Args:
        model: A loaded ``AutoModel`` (DINOv2/DINOv3) on ``device``.
        pixel_values: ``(3, h, w)`` (or ``(1, 3, h, w)``) float pixels, cast
            to the backbone's parameter dtype before the forward pass.
        device: Resolved PyTorch device string.

    Returns:
        ``(Hp, Wp, D)`` float32 dense patch features.
    """
    import numpy as np
    import torch

    pixels = torch.from_numpy(np.ascontiguousarray(pixel_values)).to(device)
    if pixels.ndim == 3:
        pixels = pixels[None]
    # A reduced-precision backbone (#282 H7) takes its own dtype in and is
    # read out in float32; for a float32 backbone both casts are no-ops.
    pixels = pixels.to(backbone_dtype(model))
    with torch.inference_mode():
        outputs = model(pixel_values=pixels)
    tokens = outputs.last_hidden_state[0].detach().to(torch.float32).cpu().numpy()
    return _tokens_to_patch_grid(model, tokens, pixels.shape[-2:])


def _tokens_to_patch_grid(
    model: Any, tokens: "np.ndarray", pixel_hw: Tuple[int, int]
) -> "np.ndarray":
    """One image's ``(n_tokens, D)`` tokens as its ``(Hp, Wp, D)`` patch grid.

    The grid is inferred from the processed pixel geometry *pixel_hw* and
    ``config.patch_size``; the CLS and register tokens are dropped
    (:func:`reshape_patch_tokens`).
    """
    in_h, in_w = int(pixel_hw[0]), int(pixel_hw[1])
    patch = backbone_patch_size(model)
    grid_hw = patch_grid_hw((in_h, in_w), patch)
    n_reg = int(getattr(model.config, "num_register_tokens", 0))
    return reshape_patch_tokens(tokens, grid_hw, n_reg)


def backbone_device_name(device: str) -> str | None:
    """The GPU's name for a resolved *device*, for an entry's ``model.device_name``.

    ``torch.cuda.get_device_name()`` of the current CUDA device when *device*
    is ``"cuda"``, ``None`` on the CPU (#281: the device is provenance).
    """
    if device != "cuda":
        return None
    import torch

    return str(torch.cuda.get_device_name())


def compute_block(
    *,
    device: str,
    stage: Literal["inline", "stage4"],
    physical_batch: int | Collection[int],
) -> dict[str, Any]:
    """Where and how an embedding entry was computed: its ``compute`` block (#282 §4.5).

    The **one producer** of every device field an entry records (#282 O4):
    ``MeasureDinoEmbedding.entry_descriptor_for`` copies this block's
    ``device`` and ``device_name`` into ``model.device`` /
    ``model.device_name``, so the two places cannot disagree. Provenance
    only: the deliverables group descriptors without it.

    Args:
        device: The resolved device that ran the backbone, ``"cuda"`` or
            ``"cpu"``.
        stage: ``"inline"`` (the measure core) or ``"stage4"`` (the GPU stage).
        physical_batch: Inputs per forward pass: one size, or every size the
            entry's passes ran at (``BatchEncoder.drain``'s
            ``physical_batches``), which differ when an OOM halved the batch
            part-way through the image (#282 I1).

    Returns:
        ``{device, device_name, capability, torch, cuda, physical_batch,
        stage}``: ``device_name`` from :func:`backbone_device_name`;
        ``capability`` the GPU's ``"<major>.<minor>"`` compute capability on
        CUDA and ``None`` on the CPU; ``torch`` the torch version; ``cuda``
        ``torch.version.cuda`` on CUDA (``None`` on the CPU, which ran none).
        ``physical_batch`` is the one size when there is one. When the
        entry's rows ran at several, it is ``None`` and the block gains
        ``physical_batches``, every size largest first (e.g. ``[64, 32]``),
        so no entry states a single pass shape it does not have. An entry
        that ran no pass (no regions) records ``physical_batch: None``.

    Raises:
        ValueError: If a size is not a positive integer.
    """
    import torch

    sizes = [physical_batch] if isinstance(physical_batch, int) else list(physical_batch)
    if any(isinstance(m, bool) or int(m) != m or m < 1 for m in sizes):
        raise ValueError(f"physical batch sizes must be positive integers, got {physical_batch!r}")
    distinct = sorted({int(m) for m in sizes}, reverse=True)
    on_cuda = device == "cuda"
    capability = None
    if on_cuda:
        major, minor = torch.cuda.get_device_capability()
        capability = f"{int(major)}.{int(minor)}"
    block: dict[str, Any] = {
        "device": device,
        "device_name": backbone_device_name(device),
        "capability": capability,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda if on_cuda else None,
        "physical_batch": distinct[0] if len(distinct) == 1 else None,
        "stage": stage,
    }
    if len(distinct) > 1:
        block["physical_batches"] = distinct
    return block


def backbone_dtype(model: Any) -> Any:
    """The dtype of *model*'s parameters, ``torch.float32`` for a model with none.

    What a backbone's input is cast to before its forward pass (#282 H7): a
    bfloat16 or float16 model takes inputs of its own dtype.
    """
    import torch

    parameters = getattr(model, "parameters", None)
    first = next(iter(parameters()), None) if callable(parameters) else None
    return first.dtype if first is not None else torch.float32


def resolve_backbone_device(device: Literal["auto", "cpu", "cuda"]) -> str:
    """Resolve an embedder's ``device`` to a PyTorch device string.

    Unlike :func:`~phenotypic.detect.nn._helper._checkpoint_manager.resolve_device`,
    ``"auto"`` falls back to the CPU **silently**: an embedder is meant to run
    on CPU nodes (#281 spec E3), so the fallback is not a degraded mode worth
    a warning per image.

    Args:
        device: ``"auto"``, ``"cpu"`` or ``"cuda"``.

    Returns:
        ``"cuda"`` or ``"cpu"``.

    Raises:
        RuntimeError: If ``"cuda"`` is asked for and CUDA is not available.
    """
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' requested but CUDA is not available")
    return device


def embed_crops(
    model: Any,
    processor: Any,
    crops: "np.ndarray",
    *,
    pooling: Literal["crop_cls", "crop_mean"],
    device: str,
    batch_size: int,
) -> "np.ndarray":
    """Embed square RGB crops with a frozen DINO ViT, one vector per crop.

    The batched counterpart of :func:`extract_patch_features`, for crops that
    are already the model's input size. Each crop is scaled and normalized
    with the processor's own ``rescale_factor``, ``image_mean`` and
    ``image_std`` and is **never resized**: the caller has resized it to a
    multiple of the patch size. Batches run under ``torch.inference_mode()``,
    in the backbone's parameter dtype, and are pooled in float32.

    Args:
        model: A loaded ``AutoModel`` (DINOv2/DINOv3) on ``device``, in eval
            mode.
        processor: The matching ``AutoImageProcessor``; only its
            normalization constants are read.
        crops: ``(N, crop_px, crop_px, 3)`` uint8 RGB crops.
        pooling: ``"crop_cls"`` takes the CLS token; ``"crop_mean"`` averages
            the patch tokens, with the CLS and register tokens dropped by
            :func:`reshape_patch_tokens`.
        device: Resolved PyTorch device string.
        batch_size: Crops per forward pass.

    Returns:
        ``(N, hidden_size)`` float32 vectors, in crop order.

    Raises:
        ValueError: If *crops* is not ``(N, H, W, 3)`` uint8, *pooling* is not
            a crop pooling, or *batch_size* is not positive.
    """
    import numpy as np
    import torch

    crops = np.asarray(crops)
    if crops.ndim != 4 or crops.shape[-1] != 3 or crops.dtype != np.uint8:
        raise ValueError(
            f"crops must be (N, H, W, 3) uint8, got {crops.shape} {crops.dtype}"
        )
    if pooling not in ("crop_cls", "crop_mean"):
        raise ValueError(f"pooling must be 'crop_cls' or 'crop_mean', got {pooling!r}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    patch = backbone_patch_size(model)
    grid_hw = patch_grid_hw(crops.shape[1:3], patch)
    n_reg = int(getattr(model.config, "num_register_tokens", 0) or 0)
    scale = float(getattr(processor, "rescale_factor", 1.0 / 255.0))
    mean = torch.tensor(processor.image_mean, dtype=torch.float32, device=device)
    std = torch.tensor(processor.image_std, dtype=torch.float32, device=device)
    mean, std = mean.view(1, 3, 1, 1), std.view(1, 3, 1, 1)

    vectors: List["np.ndarray"] = []
    # Normalised in float32, then cast to the backbone's own dtype (#282 H7);
    # the tokens are read out and pooled in float32. For a float32 backbone
    # the cast is a no-op, so its bytes are unchanged.
    compute = backbone_dtype(model)
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(crops[start : start + batch_size])
            ).to(device)
            pixels = (batch.permute(0, 3, 1, 2).to(torch.float32) * scale - mean) / std
            tokens = model(pixel_values=pixels.to(compute)).last_hidden_state
            tokens = tokens.detach().to(torch.float32).cpu().numpy()
            if pooling == "crop_cls":
                vectors.append(tokens[:, 0, :])
            else:
                vectors.append(
                    np.stack(
                        [
                            reshape_patch_tokens(one, grid_hw, n_reg).mean(axis=(0, 1))
                            for one in tokens
                        ]
                    )
                )
    if not vectors:
        hidden = int(model.config.hidden_size)
        return np.zeros((0, hidden), dtype=np.float32)
    return np.concatenate(vectors, axis=0).astype(np.float32, copy=False)


def extract_reference_features(
    model: Any,
    processor: Any,
    rgb_uint8: "np.ndarray",
    *,
    device: str,
    layer: int | None = None,
) -> Tuple["np.ndarray", Tuple[int, int]]:
    """Dense features + the **processed** pixel geometry (for W4 mask alignment).

    Like :func:`extract_patch_features` (or :func:`extract_hidden_layer_features`
    when ``layer`` is given) but also returns ``(H_proc, W_proc)`` — the
    geometry the processor handed the model — so the caller can align a
    reference/support mask through the same path via :func:`align_mask_to_grid`.
    Under :data:`NATIVE_PROCESSOR_KWARGS` this equals the input ``(H, W)``.

    Args:
        model: A loaded ``AutoModel`` on ``device``.
        processor: The matching ``AutoImageProcessor``.
        rgb_uint8: ``(H, W, 3)`` uint8 RGB image.
        device: Resolved PyTorch device string.
        layer: Optional ``hidden_states`` index (``None`` → ``last_hidden_state``).

    Returns:
        ``((Hp, Wp, D) features, (H_proc, W_proc))``.
    """
    import numpy as np
    import torch

    inputs = processor(
        images=rgb_uint8, return_tensors="pt", **NATIVE_PROCESSOR_KWARGS
    ).to(device)
    with torch.no_grad():
        if layer is None:
            outputs = model(**inputs)
            hidden = outputs.last_hidden_state
        else:
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[layer]
    tokens = hidden[0].detach().cpu().numpy().astype(np.float32)

    pixel = inputs["pixel_values"]
    proc_hw = (int(pixel.shape[-2]), int(pixel.shape[-1]))
    patch = backbone_patch_size(model)
    grid_hw = patch_grid_hw(proc_hw, patch)
    n_reg = int(getattr(model.config, "num_register_tokens", 0))
    feats = reshape_patch_tokens(tokens, grid_hw, n_reg)
    return feats, proc_hw


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
    semantics than the last). ``layer=-1`` is the last layer. The image is fed
    at its native geometry (:data:`NATIVE_PROCESSOR_KWARGS`).

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

    inputs = processor(
        images=rgb_uint8, return_tensors="pt", **NATIVE_PROCESSOR_KWARGS
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[layer]
    tokens = hidden[0].detach().cpu().numpy().astype(np.float32)

    pixel = inputs["pixel_values"]
    in_h, in_w = int(pixel.shape[-2]), int(pixel.shape[-1])
    patch = backbone_patch_size(model)
    grid_hw = patch_grid_hw((in_h, in_w), patch)
    n_reg = int(getattr(model.config, "num_register_tokens", 0))
    return reshape_patch_tokens(tokens, grid_hw, n_reg)


# ---------------------------------------------------------------------------
# Mask ↔ patch-grid alignment (W4)
# ---------------------------------------------------------------------------


def resize_mask_to_grid(
    mask: "np.ndarray", grid_hw: Tuple[int, int], patch: int
) -> "np.ndarray":
    """Downsample a full-resolution boolean mask onto the patch grid.

    Nearest-neighbour (``order=0``) so labels are not interpolated. The mask is
    first cropped to ``covered_hw(grid_hw, patch)`` — the extent the grid
    actually describes — so the truncated bottom/right remainder cannot leak
    into a patch. The caller is responsible for first aligning the mask to the
    same geometry the image went through the processor (W4); this is the final
    patch-grid step.

    ``patch`` is **required**. It was once optional, defaulting to mapping the
    whole mask, and omitting it produced a silent sub-patch scale error rather
    than a crash — a bug that reached both the query path and, later, the
    exemplar path (where the buggy and corrected prototypes had cosine 0.44).
    Get the value from :func:`backbone_patch_size`.

    Args:
        mask: ``(H, W)`` boolean (or 0/1) mask.
        grid_hw: ``(Hp, Wp)`` target patch grid.
        patch: ``model.config.patch_size``.

    Returns:
        ``(Hp, Wp)`` boolean mask.
    """
    import numpy as np
    from skimage.transform import resize

    hp, wp = int(grid_hw[0]), int(grid_hw[1])
    ch, cw = covered_hw((hp, wp), patch)
    arr = np.asarray(mask)[:ch, :cw]
    small = (
        resize(
            arr.astype(np.float32),
            (hp, wp),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        )
        > 0.5
    )
    return small.astype(bool)


def align_mask_to_grid(
    mask: "np.ndarray",
    proc_hw: Tuple[int, int],
    grid_hw: Tuple[int, int],
    patch: int,
) -> "np.ndarray":
    """Align a full-resolution mask to the patch grid via the processor geometry.

    W4: the image reaches the model at ``proc_hw`` (under
    :data:`NATIVE_PROCESSOR_KWARGS` that is its own ``(H, W)``; a resizing
    processor may make it something else), then is patchified to ``grid_hw``.
    The mask must follow the **same** path — resize to ``proc_hw``, then
    nearest-downsample to the grid — so a non-square exemplar's mask lines up
    with its features.

    Because patchification floors, the grid only covers
    ``covered_hw(grid_hw, patch)`` of ``proc_hw``. Pass ``patch`` so the
    truncated bottom/right remainder is cropped rather than squeezed into the
    last patch row/column.

    Args:
        mask: ``(Hm, Wm)`` full-resolution boolean (or 0/1) mask.
        proc_hw: ``(H_proc, W_proc)`` processed pixel geometry
            (``inputs.pixel_values.shape[-2:]``).
        grid_hw: ``(Hp, Wp)`` patch grid.
        patch: ``model.config.patch_size`` (**required** — omitting it used to
            be a silent scale error; see :func:`resize_mask_to_grid`).

    Returns:
        ``(Hp, Wp)`` boolean mask aligned with the patch features.
    """
    import numpy as np
    from skimage.transform import resize

    ph, pw = int(proc_hw[0]), int(proc_hw[1])
    processed = (
        resize(
            np.asarray(mask, dtype=np.float32),
            (ph, pw),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        )
        > 0.5
    )
    return resize_mask_to_grid(processed, grid_hw, patch)


# ---------------------------------------------------------------------------
# Prototype pooling + cosine match (the shared core)
# ---------------------------------------------------------------------------


def pool_prototype(
    features: "np.ndarray",
    mask: "np.ndarray",
    *,
    patch: int,
    proc_hw: Tuple[int, int] | None = None,
) -> "np.ndarray":
    """Masked-mean an exemplar prototype from dense patch features.

    The mask is downsampled to the patch grid (W4 — handles a full-resolution
    and/or non-square exemplar mask), then the foreground patch features are
    averaged. When ``proc_hw`` is given the mask is aligned through the
    processor's geometry first (:func:`align_mask_to_grid`); otherwise it is
    resized straight to the grid.

    Args:
        features: ``(Hp, Wp, D)`` dense patch features.
        mask: ``(Hm, Wm)`` boolean mask (any resolution; resized to the grid).
        patch: ``model.config.patch_size`` (**required**). The mask is cropped
            to the extent the grid actually covers before downsampling, so the
            truncated bottom/right remainder cannot contaminate the prototype.
            Omitting it was once permitted and silently corrupted the exemplar
            prototype; see :func:`resize_mask_to_grid`.
        proc_hw: Optional ``(H_proc, W_proc)`` processed geometry for the W4
            two-step alignment (from :func:`extract_reference_features`).

    Returns:
        ``(D,)`` mean-pooled prototype (zero vector if the mask is empty —
        the fail-safe degenerate case).
    """
    import numpy as np

    feats = np.asarray(features, dtype=np.float32)
    hp, wp, d = feats.shape
    if proc_hw is None:
        small = resize_mask_to_grid(np.asarray(mask), (hp, wp), patch)
    else:
        small = align_mask_to_grid(np.asarray(mask), proc_hw, (hp, wp), patch)
    idx = small.reshape(-1)
    flat = feats.reshape(hp * wp, d)
    fg = flat[idx]
    if fg.shape[0] == 0:
        return np.zeros(d, dtype=np.float32)
    return fg.mean(axis=0).astype(np.float32)


def pool_prototype_tiled(
    dense_by_tile: List["np.ndarray"],
    tiles: List["_Tile"],
    mask: "np.ndarray",
    patch: int,
) -> "np.ndarray":
    """Pool a full-image mask's prototype from per-tile dense features.

    Pooling every proposal against a **whole-plate** patch grid is a silent
    no-op: at ``patch=14`` a 4000x3000 plate fed to the ViT at its 224-px
    classification preset yields a 16x16 grid, i.e. ~250x187 native px per
    patch, so a 30 px colony spans 0.16 x 0.12 patches. It rounds to empty in
    :func:`resize_mask_to_grid`, :func:`pool_prototype` hits its zero-vector
    fail-safe, and *every* proposal gets the same zero prototype — the cosine
    scores become constant and the DINO half of the recipe does no work.

    Tiling restores sub-patch resolution. The mask is assigned to the tile whose
    core contains its centroid
    (:func:`~phenotypic.detect.nn._helper._tiling.owning_tile_index`), cropped to that
    tile, and pooled against that tile's feature grid, where the colony now
    spans ``30 / patch`` patches.

    Args:
        dense_by_tile: One ``(Hp, Wp, D)`` feature grid per tile, aligned with
            *tiles*.
        tiles: Crop rectangles in full-image coordinates (from
            :func:`~phenotypic.detect.nn._helper._tiling._plan_tiles`). A single
            full-extent tile is fine — the un-tiled path.
        mask: ``(H, W)`` boolean mask in full-image coordinates.
        patch: ``model.config.patch_size`` (14 for DINOv2, 16 for DINOv3).

    Returns:
        ``(D,)`` mean-pooled prototype (zero vector if *mask* is empty).
    """
    import numpy as np

    from phenotypic.detect.nn._helper._tiling import owning_tile_index

    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if ys.size == 0:
        return np.zeros(dense_by_tile[0].shape[-1], dtype=np.float32)
    idx = owning_tile_index(tiles, (float(ys.mean()), float(xs.mean())))
    tile = tiles[idx]
    local = np.asarray(mask)[tile.y0:tile.y1, tile.x0:tile.x1]
    return pool_prototype(dense_by_tile[idx], local, patch=patch)


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
    patch: int,
) -> "np.ndarray":
    """Cosine-match query patches to a prototype → an upsampled boolean mask.

    Computes the per-patch cosine map, upsamples it through the extent the grid
    actually covers (:func:`upsample_grid_to_image`), and thresholds at
    ``thresh``. A zero prototype (empty reference mask) yields an all-False mask
    (fail-safe).

    Args:
        features: ``(Hp, Wp, D)`` query dense patch features.
        prototype: ``(D,)`` exemplar prototype.
        thresh: Cosine-similarity cutoff (foreground where ``sim > thresh``).
        out_shape: ``(H, W)`` of the full-resolution output mask.
        patch: ``model.config.patch_size`` (**required**). Stretching the score
            map over ``out_shape`` instead would displace every object by up to
            ``(patch - 1) / H``; see :func:`resize_mask_to_grid`.

    Returns:
        ``(H, W)`` boolean ``objmask``.
    """
    import numpy as np

    proto = np.asarray(prototype, dtype=np.float64)
    if not np.any(proto):  # degenerate zero-prototype → fail safe
        return np.zeros(out_shape, dtype=bool)

    sim = cosine_similarity_map(features, proto)
    sim_full = upsample_grid_to_image(
        sim.astype(np.float32), out_shape, patch, order=1
    )
    return (sim_full > thresh).astype(bool)
