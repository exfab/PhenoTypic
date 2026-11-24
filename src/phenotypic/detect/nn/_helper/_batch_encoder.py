"""Stage 4's batched backbone: the physical batch, determinism, and the encoder (#282).

Stage 4 runs every ``MeasureDinoEmbedding`` of a run on GPUs, with one
resident model per shard and crops from many images in each forward pass
(spec §4.4). This module holds what that needs apart from the CLI:

- :class:`BackboneKey`, the identity of one model load, and its
  backbone-major order (H10);
- the **physical batch** rule, :func:`physical_batch_size`, a function of the
  GPU's total memory, the backbone and ``crop_px`` only, never of free memory,
  and :func:`physical_batch_for`, its one producer from a loaded backbone;
- :func:`configure_determinism`, the settings Stage 4 runs under, and
  :func:`check_cublas_workspace`, its cuBLAS half;
- :class:`BatchEncoder`, which runs one backbone over regions from many
  images at a fixed physical shape, and :class:`Stage4OutOfMemory`.

The rule's constants are copied from the P0 logic-validation script
``docs/superpowers/logic_validation_scripts/2026-11-12-gpu-embedding-staging/physical_batch_size.py``,
and :data:`MAX_IN_FLIGHT_OUTPUT_BYTES` from ``stage4_memory.py`` beside it;
``tests/unit/detect/nn/test_batch_size.py`` holds them to the scripts.

torch is imported inside functions only.
"""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable, Hashable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, NamedTuple, get_args

import numpy as np

if TYPE_CHECKING:
    from phenotypic.abc_ import BackboneInputs, RegionEncoder
    from phenotypic.sdk_.typing_ import ComputeDType, EmbeddingDinoSize

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
#: about 1.54 GB). Sized for 7B, which #282 deferred (U11), and so conservative
#: for giant (about 0.75 GB a grid). A crop pipeline holds ``N * D * 4`` bytes
#: an image and never reaches it. From ``stage4_memory.py``.
#:
#: It counts **outputs only**. Not counted: the queued input rows (about
#: 290 MB of float32 tiles for one 4012 x 6016 plate's ``masked_dense``
#: ticket), and finished tickets the caller has not drained, so the caller
#: drains after every :meth:`BatchEncoder.submit` (#282 review M15).
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
    size: EmbeddingDinoSize
    revision: str | None
    compute_dtype: ComputeDType

    def sort_key(self) -> tuple[int, str, str, str]:
        """The backbone-major order (H10): ``revision or ""``, since ``None`` and a string do not compare."""
        return (self.version, self.size, self.revision or "", self.compute_dtype)


def tokens_for(crop_px: int | tuple[int, int], patch: int, registers: int) -> int:
    """Tokens a ViT sees for one input: the patches, the CLS and the registers.

    *crop_px* is a square crop's side, or an input's ``(h, w)`` (a
    ``masked_dense`` tile). The patch embedding floors, so a side need not be
    a multiple of *patch* here (the embedder refuses a ``crop_px`` that is not).
    """
    h, w = (crop_px, crop_px) if isinstance(crop_px, int) else crop_px
    if h < 1 or w < 1 or patch < 1 or registers < 0:
        raise ValueError(
            f"crop_px and patch must be positive and registers not negative, got "
            f"crop_px={crop_px}, patch={patch}, registers={registers}"
        )
    return (h // patch) * (w // patch) + 1 + registers


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


def physical_batch_inputs(
    model: Any, *, crop_px: int | tuple[int, int], compute_dtype: ComputeDType
) -> dict[str, int]:
    """What :func:`physical_batch_for` feeds the rule, read from a loaded backbone (#282 I3).

    - ``weight_bytes``: the loaded parameters' exact bytes, never a table's
      nominal size;
    - ``tokens``: :func:`tokens_for` the geometry, with
      :func:`backbone_patch_size` and ``config.num_register_tokens``;
    - ``per_sample_bytes``: :func:`per_sample_bytes` with
      ``config.hidden_size`` and ``config.num_attention_heads``, in units of
      *compute_dtype*.

    Args:
        model: The loaded backbone, its parameters in *compute_dtype*.
        crop_px: The input's side (a crop), or its ``(h, w)`` (a tile).
        compute_dtype: The dtype *model*'s parameters are in.

    Returns:
        ``{"weight_bytes", "per_sample_bytes", "tokens"}``.

    Raises:
        ValueError: If *model*'s parameters are not in *compute_dtype*.
    """
    import torch

    from phenotypic.detect.nn._helper._dino_support import (
        backbone_dtype,
        backbone_patch_size,
    )

    dtype = getattr(torch, compute_dtype)
    loaded = backbone_dtype(model)
    if loaded != dtype:
        raise ValueError(
            f"the backbone's parameters are {loaded}, not compute_dtype={compute_dtype!r}; "
            "its weight and activation bytes would be misstated"
        )
    config = model.config
    tokens = tokens_for(
        crop_px, backbone_patch_size(model), int(getattr(config, "num_register_tokens", 0) or 0)
    )
    return {
        "weight_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
        "per_sample_bytes": per_sample_bytes(
            tokens=tokens,
            hidden=int(config.hidden_size),
            heads=int(config.num_attention_heads),
            dtype_bytes=dtype.itemsize,
        ),
        "tokens": tokens,
    }


def physical_batch_for(
    model: Any,
    *,
    crop_px: int | tuple[int, int],
    batch_size: int,
    compute_dtype: ComputeDType,
    gpu_total_bytes: int,
) -> int:
    """``m`` for one input geometry of a loaded backbone: the rule's one producer (#282 I3).

    Every caller that needs ``m`` (Stage 4's worker, the P2 T4
    batch-invariance job) calls this, so the ``m`` T4 measured is the ``m``
    production runs at: :func:`physical_batch_size` of
    :func:`physical_batch_inputs`.

    Args:
        model: The loaded backbone, its parameters in *compute_dtype*.
        crop_px: The input's side (a crop), or its ``(h, w)`` (a tile).
        batch_size: The cap: the embedder's ``batch_size``, or the smallest
            of the embedders that share this geometry.
        compute_dtype: The dtype *model*'s parameters are in.
        gpu_total_bytes: The GPU's **total** memory, e.g.
            ``torch.cuda.get_device_properties(index).total_memory``; never
            its free memory.

    Returns:
        ``m``.

    Raises:
        ValueError: As :func:`physical_batch_inputs` and
            :func:`physical_batch_size`.
    """
    inputs = physical_batch_inputs(model, crop_px=crop_px, compute_dtype=compute_dtype)
    return physical_batch_size(
        batch_size=batch_size,
        gpu_total_bytes=gpu_total_bytes,
        weight_bytes=inputs["weight_bytes"],
        per_sample=inputs["per_sample_bytes"],
    )


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

    The settings are **process-global and never restored** (#282 review M13):
    everything that runs after this in the same process runs with
    deterministic algorithms (``warn_only``), cuDNN benchmark off and TF32
    off. A SLURM shard is its own process, so that is the whole shard; a local
    Stage 4 runs in the CLI process, so the rest of that CLI process runs under
    them too.

    Args:
        device: ``"cuda"``, ``"cuda:N"`` or ``"cpu"``, the device Stage 4
            runs on. ``None`` infers it from ``torch.cuda.is_available()``.
            The cuBLAS check applies on CUDA only.

    Returns:
        The settings as torch reports them after applying them, and the
        cuBLAS variable's value (``None`` when unset).

    Raises:
        RuntimeError: As :func:`check_cublas_workspace`, on CUDA.
    """
    import torch

    on_cuda = torch.cuda.is_available() if device is None else torch.device(device).type == "cuda"
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


# ---------------------------------------------------------------------------
# The encoder (spec §4.4)
# ---------------------------------------------------------------------------


#: One queued row: its ticket, its index in that ticket's inputs, its pixels.
_Row = tuple[Hashable, int, np.ndarray]
#: A queue's geometry: ``(kind, h, w, pooling)``.
_QueueKey = tuple[str, int, int, str]


class Stage4OutOfMemory(RuntimeError):
    """The backbone ran out of GPU memory and a queue's physical batch cannot halve again.

    Raised by a forward pass at ``m = 1``, or after that queue's ``m`` has
    already halved once (spec §4.8). Running out is the node's state, never
    the image's: the shard exits non-zero, and the next round retries its
    images. The pass's rows are back in their queue.
    """


class DrainedTicket(NamedTuple):
    """A finished ticket, as :meth:`BatchEncoder.drain` yields it.

    Attributes:
        ticket: The caller's key, as submitted.
        outputs: Shaped as :meth:`~phenotypic.abc_.RegionEncoder.run` returns
            them: for crops one ``(N, D)`` float32 array, for tiles one
            ``(h // p, w // p, D)`` float32 grid per tile.
        physical_batches: Every physical batch size a pass that held one of
            the ticket's rows ran at, largest first: one size, unless an OOM
            halved a queue part-way through the ticket or its rows sat in
            queues of different ``m`` (#282 I1). Empty for a zero-row
            submission, which ran no pass.
    """

    ticket: Hashable
    outputs: list[Any]
    physical_batches: tuple[int, ...]


@dataclass
class _Ticket:
    """What the encoder holds for one submission until :meth:`BatchEncoder.drain` yields it."""

    outputs: list[Any]
    remaining: int
    output_bytes: int
    physical_batches: set[int] = field(default_factory=set)


def _is_device_oom(exc: BaseException) -> bool:
    """torch's (GPU) ``OutOfMemoryError``: :func:`~phenotypic.abc_.is_out_of_memory` without
    the host ``MemoryError``, which ends the shard (#282 review M7)."""
    from phenotypic.abc_._measure_embedding import is_out_of_memory

    return not isinstance(exc, MemoryError) and is_out_of_memory(exc)


class BatchEncoder:
    """One resident backbone over regions from many images, at a fixed physical shape (spec §4.4).

    Rows are queued by geometry, ``(kind, h, w, pooling)``, one queue per
    key. **Every forward pass of a queue has exactly that queue's** ``m``
    inputs: a full queue runs at once, and :meth:`flush` zero-pads the
    partial ones. ``m`` is per queue (#282 I2): embedders that share one
    model load (:class:`BackboneKey`) may differ in ``crop_px``, and a 448-px
    crop costs about four times a 224-px one, so *physical_batch* may be a
    rule of the geometry, normally :func:`physical_batch_for` of the model.
    Crops are normalised as
    :func:`~phenotypic.detect.nn._helper._dino_support.embed_crops`
    normalises them; ``masked_dense`` tiles arrive normalised by the HF
    processor (P1 T2). Both are cast to ``compute_dtype`` and run under
    ``torch.inference_mode()``; the CLS token, the mean patch token and the
    patch grids are read out in float32 and pooled as the inline encoder
    pools them, so on one physical batch a row's vector is the inline one
    whenever the backbone's rows do not depend on their pass (S1).

    A submission is a *ticket*, the caller's key (e.g. ``(image, key)``);
    :meth:`drain` yields each ticket as soon as all its rows are done, in
    completion order, with every physical batch size its rows ran at
    (:class:`DrainedTicket`), and drops it. So a finished image never waits
    behind another image's partial batch, and its ``compute`` block can
    state the pass shape it has.

    **In-flight bound.** The output bytes held for unfinished tickets
    (:attr:`held_bytes`) are bounded by *max_in_flight_bytes*: when a
    submission takes them over, every partial queue is flushed. A crop
    ticket holds ``N * D * 4`` bytes, so crop pipelines never reach the
    default bound and keep full batches; a ``masked_dense`` ticket holds its
    feature grid, so only dense grids flush early (re-review item 4). The
    bound counts outputs only, not queued input rows, and not finished
    tickets the caller has not drained: drain after every :meth:`submit`.

    **Out of memory.** A pass that runs out of GPU memory halves its queue's
    ``m`` once, for the encoder's life, and puts its rows back at the head of
    that queue, so no row is lost and every later pass of that queue has the
    halved size; other queues keep their ``m``. A run started by
    :meth:`submit` then returns, leaving the rows for the next :meth:`submit`
    or :meth:`flush`; :meth:`flush` retries them at once. A pass that runs
    out of memory when its queue cannot halve raises
    :class:`Stage4OutOfMemory`, which poisons the encoder (below): the shard
    fails, and the next round starts a fresh encoder.

    **Any other error in a forward pass** (a driver fault, a CUDA error that
    is not an ``OutOfMemoryError``, a dtype mismatch) also puts the pass's
    rows back, so :attr:`held_bytes` and every ticket stay as they were, and
    then **poisons** the encoder: the error propagates, and every later
    :meth:`submit`, :meth:`flush` and :meth:`drain` raises ``RuntimeError``
    naming it (#282 I6). The caller fails the shard rather than continue
    with images whose rows a failed pass held.

    Args:
        model: The loaded backbone, on *device*, in *compute_dtype*.
        processor: Its processor; only ``rescale_factor``, ``image_mean`` and
            ``image_std`` are read, for crops.
        device: The device *model* is on, e.g. ``"cuda"`` or ``"cpu"``.
        compute_dtype: The dtype *model*'s parameters are in.
        physical_batch: ``m`` for every geometry, or a rule
            ``(h, w) -> m`` called once per queue, at its first row. A caller
            whose embedders share a geometry but differ in ``batch_size``
            passes the smallest cap for it.
        max_in_flight_bytes: The bound on :attr:`held_bytes`.

    Raises:
        ValueError: If *physical_batch* (or a value of its rule) or
            *max_in_flight_bytes* is not positive, *compute_dtype* is not one
            of the three, or *model*'s parameters are in another dtype.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        device: str,
        compute_dtype: ComputeDType,
        physical_batch: int | Callable[[int, int], int],
        max_in_flight_bytes: int = MAX_IN_FLIGHT_OUTPUT_BYTES,
    ) -> None:
        import torch

        from phenotypic.detect.nn._helper._dino_support import (
            backbone_dtype,
            backbone_patch_size,
        )

        if callable(physical_batch):
            rule: Callable[[int, int], int] = physical_batch
        else:
            fixed = int(physical_batch)
            if fixed < 1:
                raise ValueError(f"physical_batch must be positive, got {physical_batch}")
            rule = lambda h, w: fixed  # noqa: E731
        if max_in_flight_bytes < 1:
            raise ValueError(f"max_in_flight_bytes must be positive, got {max_in_flight_bytes}")
        from phenotypic.sdk_.typing_ import ComputeDType

        dtypes = get_args(ComputeDType)
        if compute_dtype not in dtypes:
            raise ValueError(f"compute_dtype must be one of {dtypes}, got {compute_dtype!r}")
        dtype = getattr(torch, compute_dtype)
        loaded = backbone_dtype(model)
        if loaded != dtype:
            raise ValueError(
                f"the backbone's parameters are {loaded}, not compute_dtype={compute_dtype!r}; "
                "load it in the compute dtype"
            )
        self._model, self._device, self._dtype = model, device, dtype
        self._on_cuda = torch.device(device).type == "cuda"
        self._patch = backbone_patch_size(model)
        self._registers = int(getattr(model.config, "num_register_tokens", 0) or 0)
        self._hidden = int(model.config.hidden_size)
        # The crop normalisation, exactly as embed_crops builds it.
        self._scale = float(getattr(processor, "rescale_factor", 1.0 / 255.0))
        mean = torch.tensor(processor.image_mean, dtype=torch.float32, device=device)
        std = torch.tensor(processor.image_std, dtype=torch.float32, device=device)
        self._mean, self._std = mean.view(1, 3, 1, 1), std.view(1, 3, 1, 1)

        self._rule = rule
        self._m: dict[_QueueKey, int] = {}
        self._halved: set[_QueueKey] = set()
        self._failure: BaseException | None = None
        self._max_in_flight_bytes = int(max_in_flight_bytes)
        self._queues: dict[_QueueKey, deque[_Row]] = {}
        self._tickets: dict[Hashable, _Ticket] = {}
        self._done: deque[Hashable] = deque()
        self._held_bytes = 0

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #

    @property
    def physical_batches(self) -> dict[_QueueKey, int]:
        """Each queue's ``m`` so far, ``{(kind, h, w, pooling): m}`` (halved after an OOM)."""
        return dict(self._m)

    @property
    def halved(self) -> bool:
        """Whether any queue's ``m`` has halved after an OOM."""
        return bool(self._halved)

    @property
    def held_bytes(self) -> int:
        """The output bytes held for unfinished tickets: what the in-flight bound counts."""
        return self._held_bytes

    @property
    def max_in_flight_bytes(self) -> int:
        """The bound on :attr:`held_bytes`."""
        return self._max_in_flight_bytes

    # ------------------------------------------------------------------ #
    # Submit, flush, drain
    # ------------------------------------------------------------------ #

    def submit(self, ticket: Hashable, inputs: BackboneInputs) -> None:
        """Queue *inputs*' rows under *ticket*, and run every queue that is full.

        A zero-row submission completes at once, with no forward pass. When
        the output bytes held for unfinished tickets then exceed the
        in-flight bound, every partial queue is flushed.

        Args:
            ticket: The caller's key for these outputs, not already held.
            inputs: ``"crops"`` (``crop_cls`` / ``crop_mean``) or ``"tiles"``
                (``patches``), as :meth:`MeasureDinoEmbedding.prepare_regions`
                builds them.

        Raises:
            ValueError: If *ticket* is held (submitted and not yet drained),
                *inputs* are not a backbone's, or the physical-batch rule
                gives a new geometry no positive ``m``.
            Stage4OutOfMemory: As :meth:`flush`, from a pass this runs.
            RuntimeError: If the encoder is poisoned (see the class).
        """
        self._refuse_if_failed()
        if ticket in self._tickets:
            raise ValueError(f"ticket {ticket!r} is already held; drain it before reusing it")
        entry, rows = self._admit(inputs)
        for key in dict.fromkeys(key for key, _, _ in rows):
            if key not in self._m:
                self._m[key] = self._physical_batch_of(key)
        self._tickets[ticket] = entry
        if entry.remaining == 0:
            self._done.append(ticket)
            return
        self._held_bytes += entry.output_bytes
        for key, index, pixels in rows:
            self._queues.setdefault(key, deque()).append((ticket, index, pixels))
        self._run_queues(partial=False)
        if self._held_bytes > self._max_in_flight_bytes:
            self.flush()

    def flush(self) -> None:
        """Run every queue to empty, the last pass of each zero-padded to exactly its ``m``.

        A pass that runs out of memory halves its queue's ``m`` and its rows
        are retried at once, at the halved size.

        Raises:
            Stage4OutOfMemory: If a pass runs out of GPU memory and its
                queue's ``m`` cannot halve (it already has, or it is 1);
                the encoder is then poisoned.
            RuntimeError: If the encoder is poisoned (see the class).
        """
        self._refuse_if_failed()
        self._run_queues(partial=True)

    def drain(self) -> Iterator[DrainedTicket]:
        """Yield each finished ticket, in completion order, and drop it.

        Runs nothing: a ticket with rows still queued stays held.

        Returns:
            An iterator of :class:`DrainedTicket`.

        Raises:
            RuntimeError: If the encoder is poisoned (see the class), at the
                call, before anything is yielded.
        """
        self._refuse_if_failed()
        return self._drain()

    def as_region_encoder(self) -> RegionEncoder:
        """A synchronous :class:`~phenotypic.abc_.RegionEncoder` on this encoder.

        Its ``run`` submits, flushes and returns that one ticket's outputs,
        so :meth:`MeasureEmbedding.encode` can run on the batched path.
        """
        return _SyncRegionEncoder(self)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _refuse_if_failed(self) -> None:
        failure = self._failure
        if failure is not None:
            raise RuntimeError(
                "this BatchEncoder is unusable: a forward pass failed with "
                f"{type(failure).__name__}: {failure}. The pass's rows are back in "
                "their queue, but the encoder never retries them; fail the shard, and "
                "the next round re-embeds its images"
            ) from failure

    def _drain(self) -> Iterator[DrainedTicket]:
        while self._done:
            ticket = self._done.popleft()
            entry = self._tickets.pop(ticket)
            yield DrainedTicket(
                ticket, entry.outputs, tuple(sorted(entry.physical_batches, reverse=True))
            )

    def _physical_batch_of(self, key: _QueueKey) -> int:
        """A new queue's ``m``, from the rule, checked positive."""
        _, h, w, _ = key
        m = self._rule(h, w)
        if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 1:
            raise ValueError(f"the physical batch of a {h} x {w} input must be a positive int, got {m!r}")
        return int(m)

    def _halve(self, key: _QueueKey) -> bool:
        """Halve *key*'s ``m``, once for the encoder's life; ``False`` if it has, or ``m`` is 1."""
        if key in self._halved or self._m[key] <= 1:
            return False
        self._m[key] //= 2
        self._halved.add(key)
        return True

    def _admit(self, inputs: BackboneInputs) -> tuple[_Ticket, list[tuple[_QueueKey, int, np.ndarray]]]:
        """A ticket for *inputs*, with its preallocated outputs, and its rows by queue."""
        if inputs.kind == "crops":
            if inputs.pooling not in ("crop_cls", "crop_mean"):
                raise ValueError(f"crops pool by 'crop_cls' or 'crop_mean', got {inputs.pooling!r}")
            (crops,) = inputs.arrays
            crops = np.asarray(crops)
            if crops.ndim != 4 or crops.shape[-1] != 3 or crops.dtype != np.uint8:
                raise ValueError(f"crops must be (N, H, W, 3) uint8, got {crops.shape} {crops.dtype}")
            n, h, w = crops.shape[:3]
            vectors = np.zeros((n, self._hidden), dtype=np.float32)
            crop_key: _QueueKey = ("crops", int(h), int(w), inputs.pooling)
            return (
                _Ticket(outputs=[vectors], remaining=n, output_bytes=vectors.nbytes),
                [(crop_key, i, crops[i]) for i in range(n)],
            )
        if inputs.kind == "tiles":
            if inputs.pooling != "patches":
                raise ValueError(f"tiles pool by 'patches', got {inputs.pooling!r}")
            rows: list[tuple[_QueueKey, int, np.ndarray]] = []
            grid_bytes = 0
            for i, tile in enumerate(inputs.arrays):
                tile = np.asarray(tile)
                if tile.ndim == 4 and tile.shape[0] == 1:
                    tile = tile[0]
                if tile.ndim != 3 or tile.shape[0] != 3 or tile.dtype != np.float32:
                    raise ValueError(f"a tile must be (3, h, w) float32, got {tile.shape} {tile.dtype}")
                h, w = int(tile.shape[1]), int(tile.shape[2])
                grid_bytes += (h // self._patch) * (w // self._patch) * self._hidden * 4
                rows.append((("tiles", h, w, "patches"), i, tile))
            outputs: list[Any] = [None] * len(rows)
            return _Ticket(outputs=outputs, remaining=len(rows), output_bytes=grid_bytes), rows
        raise ValueError(f"a DINO backbone does not run {inputs.kind!r} inputs")

    def _run_queues(self, *, partial: bool) -> None:
        """Run full queues (and, with *partial*, every queue) until none is left to run.

        After an out-of-memory pass that halved a queue's ``m``, a run
        started by :meth:`submit` (``partial=False``) returns; :meth:`flush`
        goes on.
        """
        while True:
            key = next(
                (k for k, q in self._queues.items() if len(q) >= self._m[k] or (partial and q)),
                None,
            )
            if key is None:
                return
            queue = self._queues[key]
            batch = [queue.popleft() for _ in range(min(self._m[key], len(queue)))]
            if not queue:
                del self._queues[key]
            if not self._forward(key, batch) and not partial:
                return

    def _forward(self, key: _QueueKey, batch: list[_Row]) -> bool:
        """One pass of exactly the queue's ``m``; ``False`` if it ran out of memory and was re-queued."""
        m = self._m[key]
        try:
            tokens = self._tokens_of(key, batch, m)
        except Exception as exc:
            # Every row goes back at the head of its queue, in order: no row is lost.
            self._queues.setdefault(key, deque()).extendleft(reversed(batch))
            if not _is_device_oom(exc):
                self._failure = exc
                raise
            if not self._halve(key):
                halved = ", after halving once" if key in self._halved else ""
                failure = Stage4OutOfMemory(
                    f"the backbone ran out of GPU memory at physical batch {m}{halved}; "
                    "the shard stops and the next round retries its images"
                )
                self._failure = failure
                raise failure from exc
        else:
            self._route(key, batch, tokens, m)
            return True
        # Released here, not inside the except: by now the traceback, and the
        # failed pass's tensors in its frames, are gone, so the cache can free them.
        if self._on_cuda:
            import torch

            torch.cuda.empty_cache()
        return False

    def _tokens_of(self, key: _QueueKey, batch: list[_Row], m: int) -> np.ndarray:
        """The backbone's float32 tokens for *batch*, zero-padded to *m* inputs."""
        import torch

        kind, h, w, _ = key
        n = len(batch)
        with torch.inference_mode():
            pixels = torch.zeros((m, 3, h, w), dtype=torch.float32, device=self._device)
            stacked = torch.from_numpy(np.stack([row for _, _, row in batch])).to(self._device)
            if kind == "crops":
                stacked = (stacked.permute(0, 3, 1, 2).to(torch.float32) * self._scale - self._mean) / self._std
            pixels[:n] = stacked
            del stacked
            tokens = self._model(pixel_values=pixels.to(self._dtype)).last_hidden_state
            return tokens.detach().to(torch.float32).cpu().numpy()

    def _route(self, key: _QueueKey, batch: list[_Row], tokens: np.ndarray, m: int) -> None:
        """Put each row's pooled output in its ticket, pooled as the inline encoder pools it."""
        from phenotypic.detect.nn._helper._dino_support import (
            patch_grid_hw,
            reshape_patch_tokens,
        )

        _, h, w, pooling = key
        grid_hw = patch_grid_hw((h, w), self._patch)
        for position, (ticket, index, _) in enumerate(batch):
            entry = self._tickets[ticket]
            entry.physical_batches.add(m)
            one = tokens[position]
            if pooling == "crop_cls":
                entry.outputs[0][index] = one[0]
            elif pooling == "crop_mean":
                entry.outputs[0][index] = reshape_patch_tokens(one, grid_hw, self._registers).mean(
                    axis=(0, 1)
                )
            else:
                # A copy, so the pass's token array is not kept alive by a view.
                entry.outputs[index] = np.array(reshape_patch_tokens(one, grid_hw, self._registers))
            entry.remaining -= 1
            if entry.remaining == 0:
                self._held_bytes -= entry.output_bytes
                self._done.append(ticket)


class _SyncRegionEncoder:
    """:meth:`BatchEncoder.as_region_encoder`: one ticket at a time, run to completion."""

    def __init__(self, encoder: BatchEncoder) -> None:
        self._encoder = encoder

    def run(self, inputs: BackboneInputs) -> list[np.ndarray]:
        encoder = self._encoder
        if encoder._tickets:
            raise RuntimeError("the synchronous adaptor needs an encoder that holds no other ticket")
        ticket = object()
        encoder.submit(ticket, inputs)
        encoder.flush()
        ((done, outputs, _),) = list(encoder.drain())
        assert done is ticket
        return outputs
