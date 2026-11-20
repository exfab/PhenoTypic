"""Stage 4's batched backbone: the physical batch, determinism, and the encoder (#282).

Stage 4 runs every ``MeasureDinoEmbedding`` of a run on GPUs, with one
resident model per shard and crops from many images in each forward pass
(spec §4.4). This module holds what that needs apart from the CLI:

- :class:`BackboneKey`, the identity of one model load, and its
  backbone-major order (H10);
- the **physical batch** rule, :func:`physical_batch_size`, a function of the
  GPU's total memory, the backbone and ``crop_px`` only, never of free memory;
- :func:`configure_determinism`, the settings Stage 4 runs under, and
  :func:`check_cublas_workspace`, its cuBLAS half.

The rule's constants are copied from the P0 logic-validation script
``docs/superpowers/logic_validation_scripts/2026-11-12-gpu-embedding-staging/physical_batch_size.py``,
and :data:`MAX_IN_FLIGHT_OUTPUT_BYTES` from ``stage4_memory.py`` beside it;
``tests/unit/detect/nn/test_batch_size.py`` holds them to the scripts.

torch is imported inside functions only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Final

# ---------------------------------------------------------------------------
# The physical batch size (spec §4.4; P0 T1 physical_batch_size.py)
# ---------------------------------------------------------------------------

#: The share of a GPU's total memory Stage 4 plans to use.
MEMORY_FRACTION: Final[float] = 0.8
#: Bytes left for the CUDA context, allocator slack and the pooled outputs.
RESERVE_BYTES: Final[int] = 1 << 30
#: Activation bytes per token per hidden unit, in units of the compute dtype,
#: beside the ``heads * T**2`` attention matrix (:func:`per_sample_bytes`).
ACTIVATION_FACTOR: Final[int] = 12

#: The most output bytes the encoder holds for unfinished images before it
#: flushes its partial queues: two DINOv3-7B ``masked_dense`` feature grids of
#: a 4012 x 6016 plate (``(4012 // 16) * (6016 // 16) * 4096 * 4`` bytes each,
#: about 1.54 GB). Sized for 7B and so conservative for giant (about 0.75 GB a
#: grid). A crop pipeline holds ``N * D * 4`` bytes an image and never reaches
#: it. From ``stage4_memory.py``.
MAX_IN_FLIGHT_OUTPUT_BYTES: Final[int] = 2 * (4012 // 16) * (6016 // 16) * 4096 * 4

# ---------------------------------------------------------------------------
# cuBLAS determinism
# ---------------------------------------------------------------------------

#: The variable cuBLAS reads, once, when the process creates its first handle.
CUBLAS_WORKSPACE_VARIABLE: Final[str] = "CUBLAS_WORKSPACE_CONFIG"
#: The two values cuBLAS documents as deterministic; a user's own ``:16:8`` is kept.
DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS: Final[tuple[str, str]] = (":4096:8", ":16:8")
#: What the CLI and the Stage-4 SLURM prelude set when the variable is unset.
CUBLAS_WORKSPACE_DEFAULT: Final[str] = DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS[0]


@dataclass(frozen=True)
class BackboneKey:
    """One model load: what Stage 4 loads once and shares between embedders (H10).

    Embedders with equal keys load the same weights in the same dtype, so
    one resident model serves them all; pooling, ``crop_px`` and the region
    are the embedders' own and not part of it.

    Attributes:
        version: The DINO version, 2 or 3.
        size: ``"small"``, ``"base"``, ``"large"`` or ``"giant"``.
        revision: The Hugging Face revision, or ``None`` for the default branch.
        compute_dtype: ``"float32"``, ``"bfloat16"`` or ``"float16"``.
    """

    version: int
    size: str
    revision: str | None
    compute_dtype: str

    def sort_key(self) -> tuple[int, str, str, str]:
        """The backbone-major order (H10): ``revision or ""``, since ``None`` and a string do not compare."""
        return (self.version, self.size, self.revision or "", self.compute_dtype)


def tokens_for(crop_px: int, patch: int, registers: int) -> int:
    """Tokens a ViT sees for a ``crop_px`` square crop: the patches, the CLS and the registers.

    The patch embedding floors, so ``crop_px`` need not be a multiple of
    *patch* here (the embedder refuses one that is not).
    """
    if crop_px < 1 or patch < 1 or registers < 0:
        raise ValueError(
            f"crop_px and patch must be positive and registers not negative, got "
            f"crop_px={crop_px}, patch={patch}, registers={registers}"
        )
    return (crop_px // patch) ** 2 + 1 + registers


def per_sample_bytes(*, tokens: int, hidden: int, heads: int, dtype_bytes: int) -> int:
    """Peak activation bytes of one input in a forward pass.

    ``dtype_bytes * (ACTIVATION_FACTOR * tokens * hidden + heads * tokens**2)``:
    the per-token activations of a block and one attention matrix per head.
    """
    return dtype_bytes * (ACTIVATION_FACTOR * tokens * hidden + heads * tokens * tokens)


def _pow2floor(value: float) -> int:
    """The largest power of two ``<= value``; 0 below 1."""
    return 0 if value < 1 else 1 << (int(value).bit_length() - 1)


def physical_batch_size(
    *, batch_size: int, gpu_total_bytes: int, weight_bytes: int, per_sample: int
) -> int:
    """The inputs per forward pass on a GPU of *gpu_total_bytes* (spec §4.4).

    ``m = min(batch_size, pow2floor((MEMORY_FRACTION * gpu_total_bytes
    - weight_bytes - RESERVE_BYTES) / per_sample))``. It reads the GPU's
    **total** memory, never its free memory, so every shard on one GPU model
    computes the same ``m`` for one backbone and ``crop_px``: vectors are
    promised batch-invariant within one GPU model and physical batch (S1).

    Args:
        batch_size: The embedder's ``batch_size``, the cap.
        gpu_total_bytes: The GPU's total memory.
        weight_bytes: The backbone's parameters in the compute dtype.
        per_sample: :func:`per_sample_bytes` of one input.

    Returns:
        ``m``, at least 1: a power of two unless *batch_size* caps it.

    Raises:
        ValueError: If the weights do not fit in the budget, one sample does
            not fit beside them, or an argument is not positive (*weight_bytes*
            may be 0).
    """
    if batch_size < 1 or gpu_total_bytes < 1 or per_sample < 1 or weight_bytes < 0:
        raise ValueError(
            "batch_size, gpu_total_bytes and per_sample must be positive and "
            f"weight_bytes not negative, got batch_size={batch_size}, "
            f"gpu_total_bytes={gpu_total_bytes}, weight_bytes={weight_bytes}, "
            f"per_sample={per_sample}"
        )
    budget = MEMORY_FRACTION * gpu_total_bytes - weight_bytes - RESERVE_BYTES
    gib = float(1 << 30)
    if budget <= 0:
        raise ValueError(
            f"the backbone ({weight_bytes / gib:.2f} GiB of weights) does not fit on a "
            f"{gpu_total_bytes / gib:.1f} GiB GPU: Stage 4 plans for "
            f"{MEMORY_FRACTION:.0%} of it less {RESERVE_BYTES / gib:.0f} GiB reserved. "
            "Use a smaller size, compute_dtype='bfloat16', or a larger GPU"
        )
    fit = _pow2floor(budget / per_sample)
    if fit < 1:
        raise ValueError(
            f"one input ({per_sample / gib:.2f} GiB of activations) does not fit in "
            f"the {budget / gib:.2f} GiB left beside the weights on a "
            f"{gpu_total_bytes / gib:.1f} GiB GPU. Use a smaller crop_px or "
            "compute_dtype='bfloat16'"
        )
    return min(batch_size, fit)


# ---------------------------------------------------------------------------
# Determinism (README Global Constraints; spec §4.4)
# ---------------------------------------------------------------------------


def check_cublas_workspace() -> None:
    """Refuse a ``CUBLAS_WORKSPACE_CONFIG`` that does not make cuBLAS deterministic.

    The value must be ``":4096:8"`` or ``":16:8"``. cuBLAS reads it once, when
    the process creates its first handle, so setting it here would be too
    late: the CLI and the Stage-4 SLURM prelude set it, and this only checks.

    Raises:
        RuntimeError: Naming the variable, its value and both accepted values,
            if it is unset or another value.
    """
    value = os.environ.get(CUBLAS_WORKSPACE_VARIABLE)
    if value in DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS:
        return
    found = "unset" if value is None else f"{value!r}"
    accepted, alternative = DETERMINISTIC_CUBLAS_WORKSPACE_CONFIGS
    raise RuntimeError(
        f"{CUBLAS_WORKSPACE_VARIABLE} is {found}; deterministic cuBLAS needs "
        f"{accepted!r} or {alternative!r}, set before the process first uses "
        f"CUDA (export {CUBLAS_WORKSPACE_VARIABLE}={CUBLAS_WORKSPACE_DEFAULT})"
    )


def configure_determinism(*, device: str | None = None) -> dict[str, Any]:
    """Apply the settings Stage 4 runs under, and return them for the shard log.

    - ``torch.backends.cudnn.deterministic = True``, ``benchmark = False``;
    - TF32 off for matmul and cuDNN (DINO's patch embedding is a conv);
    - ``torch.set_float32_matmul_precision("highest")``;
    - ``torch.use_deterministic_algorithms(True, warn_only=True)``;
    - on CUDA, a deterministic ``CUBLAS_WORKSPACE_CONFIG``
      (:func:`check_cublas_workspace`), checked before anything is applied.

    Idempotent. Call it once per Stage-4 process, before the first load.

    Args:
        device: ``"cuda"`` or ``"cpu"``, the device Stage 4 runs on. ``None``
            infers it from ``torch.cuda.is_available()``. The cuBLAS check
            applies on CUDA only.

    Returns:
        The settings as torch reports them after applying them, and the
        cuBLAS variable's value (``None`` when unset).

    Raises:
        RuntimeError: As :func:`check_cublas_workspace`, on CUDA.
    """
    import torch

    on_cuda = torch.cuda.is_available() if device is None else device == "cuda"
    if on_cuda:
        check_cublas_workspace()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_algorithms_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cublas_workspace_config": os.environ.get(CUBLAS_WORKSPACE_VARIABLE),
    }
