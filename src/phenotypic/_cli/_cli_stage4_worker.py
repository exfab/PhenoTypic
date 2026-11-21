"""The Stage-4 worker: every ``MeasureDinoEmbedding`` of a staged run, on a GPU (#282).

Two halves.

**The device half** (#282 P4 T1), which the CLI also calls at startup for a
local ``--embed-stage gpu``: :func:`stage4_device` and
:func:`check_cuda_usable`. Inside Stage 4 an embedder's device is ``cuda``; a
missing or unusable CUDA fails the stage and never falls back to the CPU
(spec §4.4, §4.8). The one escape is the test-only
``PHENOTYPIC_STAGE4_DEVICE=cpu``.

**The loop** (#282 P4 T3, spec §4.2, §4.4): :func:`run_stage4` over a list
of images that finished Stages 1-3, and :func:`run_local_stage4`, its local
caller. Per image, in this order and never another (spec §4.2):

1. one **signal** per embedder key, written as the key's vectors drain
   (``write_stage4_signal``), computed against the store root the loader read;
2. the **store transaction**, ``replace_run_embeddings``, root last, fenced
   on the signals' root;
3. the **record**, ``publish_image_success``, published once;
4. the signals **consumed**, ``delete_stage4_signals``.

So nothing is rewritten after the record that certifies it, and a crash
between any two of them resumes from what is on disk: valid signals are
committed without re-embedding, a committed store is published without
re-committing, and a published image's leftover signals are swept
(:func:`sweep_orphan_stage4_signals`).

The four durable writes are looked up as this module's globals at each call,
so a crash test can wrap one (``run_cli_subprocess(crash_after=)``).

torch is imported inside functions only.
"""

from __future__ import annotations

import logging
import os
import re
import traceback as traceback_module
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal
from uuid import uuid4

from phenotypic.sdk_ import (
    CommitGuard,
    event_log_path,
    publication_commit,
    journal_table_run,
    replace_run_embeddings,
    source_image_stem,
    zarr_store_path,
)

from ._cli_completion import (
    image_data_artifact,
    image_measurement_artifact,
    publish_image_success,
)
from ._cli_stage4_signal import (
    Stage4Signal,
    classify_stage4_signals,
    delete_stage4_signal_keys,
    delete_stage4_signal_temps,
    delete_stage4_signals,
    write_stage4_signal,
)

if TYPE_CHECKING:
    from phenotypic import ImagePipeline
    from phenotypic.abc_ import PreparedRegions
    from phenotypic.detect.nn._helper._batch_encoder import BackboneKey, BatchEncoder
    from phenotypic.measure import MeasureDinoEmbedding
    from phenotypic.sdk_._embedding_tables import EmbeddingPayload
    from phenotypic.sdk_.typing_ import ImageTypeName

    from ._cli_output_manager import OutputManager
    from ._cli_staged_orchestration import StagedManifestEntry
    from ._cli_types import Dataset, ExecutionConfig

logger = logging.getLogger(__name__)

#: Test-only: ``cpu`` runs Stage 4 on the CPU, so a CPU node can exercise it.
STAGE4_DEVICE_ENV: Final[str] = "PHENOTYPIC_STAGE4_DEVICE"

_ARCH = re.compile(r"^(sm|compute)_(\d+)(\d)[a-z]?$")


class Stage4DeviceError(RuntimeError):
    """Stage 4 has no usable CUDA device: none visible, or no kernels for it."""


def stage4_device() -> str:
    """The device Stage 4 runs on: ``"cuda"``, or ``"cpu"`` under the test escape.

    With ``"cuda"``, :func:`check_cuda_usable` runs first.

    Raises:
        Stage4DeviceError: As :func:`check_cuda_usable`.
    """
    if os.environ.get(STAGE4_DEVICE_ENV) == "cpu":
        return "cpu"
    check_cuda_usable()
    return "cuda"


def check_cuda_usable() -> None:
    """Refuse a process whose torch cannot run on GPU 0.

    Usable means torch is installed, ``torch.cuda.is_available()``, and this
    torch build carries kernels for device 0's compute capability
    (:func:`has_kernels_for`). The last is the P100 case (spec §4.8): such a
    node passes ``is_available()``, and the job would otherwise die at the
    first forward pass, after all its CPU work, with "no kernel image is
    available".

    Raises:
        Stage4DeviceError: Naming what is missing: torch, a CUDA device, or
            kernels for the device (with its name and the build's arch list).
    """
    try:
        import torch
    except ImportError as exc:
        raise Stage4DeviceError(
            "torch is not installed, so no CUDA device is usable (install the "
            "foundation extra)"
        ) from exc
    if not torch.cuda.is_available():
        raise Stage4DeviceError(
            "no CUDA device is usable (torch.cuda.is_available() is False)"
        )
    major, minor = torch.cuda.get_device_capability(0)
    arches = list(torch.cuda.get_arch_list())
    if not has_kernels_for(major, minor, arches):
        name = torch.cuda.get_device_name(0)
        raise Stage4DeviceError(
            f"CUDA device 0 ({name}, sm_{major}{minor}) has no kernels in this "
            f"torch build (built for {' '.join(arches) or 'no architecture'})"
        )


def has_kernels_for(major: int, minor: int, arches: Sequence[str]) -> bool:
    """Whether a torch build for *arches* runs on a ``sm_<major><minor>`` device.

    A ``sm_XY`` binary runs on any device of the same major version and a
    minor version at least ``Y`` (CUDA binary compatibility), so an Ada GPU
    (``sm_89``) runs a build that lists ``sm_86`` and no ``sm_89``, which
    is how the cluster's torch 2.11 + CUDA 13 wheel is built. A
    ``compute_XY`` entry is PTX, which the driver compiles for any device of
    capability ``XY`` or newer. An exact ``sm_<major><minor> in arches`` test
    would refuse every Ada GPU, the ``exfab`` partition's included.

    Args:
        major: Device compute capability, major.
        minor: Device compute capability, minor.
        arches: ``torch.cuda.get_arch_list()``, e.g. ``["sm_86", "sm_90"]``.
    """
    for arch in arches:
        match = _ARCH.match(arch)
        if match is None:
            continue
        kind, arch_major, arch_minor = match[1], int(match[2]), int(match[3])
        if kind == "sm" and arch_major == major and arch_minor <= minor:
            return True
        if kind == "compute" and (arch_major, arch_minor) <= (major, minor):
            return True
    return False




# ---------------------------------------------------------------------------
# The loop (#282 P4 T3; spec §4.2, §4.4, §4.8)
# ---------------------------------------------------------------------------

#: What ``gpu_total_bytes`` reads under the CPU test escape, which has no GPU
#: to ask: a fixed total, so the physical-batch rule still runs its one
#: producer (``physical_batch_for``) and gives the same ``m`` on any node.
_CPU_TEST_GPU_TOTAL_BYTES: Final[int] = 16 << 30


class Stage4ShardError(RuntimeError):
    """The shard (or the local Stage 4) cannot go on; no image is to blame.

    A failed forward pass, which poisons the encoder (#282 P1/P2 review I6),
    a backbone whose geometry is not its embedders' (Step 3a), a physical
    batch that does not fit the GPU, or a refused model licence. Like an OOM
    or a missing CUDA, it is node or model state: nothing is journalled, every
    image whose keys did not all commit stays pending, and the next round (or
    the same command again) re-embeds it with a fresh encoder (spec §4.8, U7).
    """


@dataclass
class Stage4Outcome:
    """What one :func:`run_stage4` call did.

    Attributes:
        published: Identities (``StagedManifestEntry.identity``) whose record
            this call published.
        failed: Identity -> error type, for each image this call failed as a
            per-image ``stage4`` failure (journalled when terminal).
        pending: Identities left for another round: their store root changed
            under the call (spec §4.8; never terminal).
        signals_written: How many Stage-4 signals this call wrote.
        physical_batches: Each loaded backbone's
            ``BatchEncoder.physical_batches`` when its loop ended: the ``m``
            of every queue, ``(kind, h, w, pooling)``, halved where an OOM
            halved it.
    """

    published: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    pending: list[str] = field(default_factory=list)
    signals_written: int = 0
    physical_batches: dict[BackboneKey, dict[tuple[str, int, int, str], int]] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class LoadedImage:
    """One image as a loader worker prepared it: every pending key of one backbone.

    Attributes:
        identity: ``StagedManifestEntry.identity``.
        root_sha256: sha256 of the store root ``zarr.json`` read **before**
            the image was loaded: the root its signals are bound to.
        prepared: ``{key: PreparedRegions}``, materialised (Step 3d).
    """

    identity: str
    root_sha256: str
    prepared: dict[str, PreparedRegions]


@dataclass(frozen=True)
class FailedImage:
    """A per-image error raised inside a loader worker, returned rather than raised.

    It crosses the process boundary as data, so the loader, the shard and
    the other images go on (README Review Focus 2).

    Attributes:
        identity: ``StagedManifestEntry.identity``.
        error_type: The exception's class name.
        message: ``str()`` of it.
        traceback: Its formatted traceback, from the worker.
        error: The exception itself, when it survives a pickle round trip;
            else ``None``.
        node_state: Whether it is the node's state (out of memory), never
            the image's: the stage then fails and journals nothing.
    """

    identity: str
    error_type: str
    message: str
    traceback: str
    error: BaseException | None = None
    node_state: bool = False

    @classmethod
    def of(cls, identity: str, exc: BaseException) -> FailedImage:
        """The failure of *identity* by *exc*, with *exc* kept only if it pickles."""
        import pickle

        from phenotypic.abc_._measure_embedding import is_out_of_memory

        kept: BaseException | None = exc
        try:
            pickle.loads(pickle.dumps(exc))
        except Exception:
            kept = None
        return cls(
            identity=identity,
            error_type=type(exc).__name__,
            message=str(exc),
            traceback="".join(traceback_module.format_exception(exc)),
            error=kept,
            node_state=is_out_of_memory(exc),
        )

    def exception(self) -> BaseException:
        """The exception to journal: the worker's own, or one carrying its type and message."""
        if self.error is not None:
            return self.error
        return Stage4ImageError(f"{self.error_type}: {self.message}")


class Stage4ImageError(ValueError):
    """An image's Stage-4 failure whose own exception could not cross a process boundary,
    or whose payload is bound to another image than the store holds."""


def _identity_collate(item: Any) -> Any:
    """The loader's ``collate_fn``: an item is one image, handed over as it is (plan review I3)."""
    return item


class Stage4ImageDataset:
    """One backbone's loader items: each image's pending keys, prepared on the CPU (spec §4.4).

    A map-style dataset for ``torch.utils.data.DataLoader``. Item *i* opens
    one store read-only, reads its root sha first, loads the image and calls
    each pending key's ``prepare_regions(image, processor=...)`` with its
    default ``materialise=True`` (Step 3d): never the inline path's deferred
    crops, which hold the plate's pixels and must not be pickled. A per-image
    error is returned as a :class:`FailedImage`, never raised through the
    loader.

    Everything it holds pickles, so ``spawn`` workers rebuild it; the
    processor of a ``masked_dense`` backbone is loaded in the main process and
    pickled in (re-review item 2c), so a worker loads nothing.

    Args:
        output_dir: Run output root.
        image_type: ``"Image"`` or ``"GridImage"``.
        work: ``[(entry, keys)]``: each image and the keys of this backbone it
            still needs, in ``meas`` order.
        embedders: ``{key: embedder}`` for this backbone.
        processors: ``{backbone: processor}`` when an embedder of this
            backbone pools ``masked_dense``; empty otherwise.
        backbone: This loader's backbone.
    """

    def __init__(
        self,
        output_dir: Path,
        image_type: ImageTypeName,
        work: Sequence[tuple[StagedManifestEntry, tuple[str, ...]]],
        embedders: Mapping[str, MeasureDinoEmbedding],
        processors: Mapping[BackboneKey, Any],
        backbone: BackboneKey,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.image_type = image_type
        self.work = list(work)
        self.embedders = dict(embedders)
        self.processors = dict(processors)
        self.backbone = backbone

    def __len__(self) -> int:
        return len(self.work)

    def __getitem__(self, index: int) -> LoadedImage | FailedImage:
        from phenotypic import GridImage, Image

        entry, keys = self.work[index]
        try:
            store = zarr_store_path(self.output_dir, entry.dataset, entry.stem)
            root = _live_root_sha256(store)
            if root is None:
                raise FileNotFoundError(f"{store} has no root zarr.json")
            image_class = GridImage if self.image_type == "GridImage" else Image
            image = image_class.load_zarr(store)  # read-only: nothing here writes
            image.name = entry.stem
            processor = self.processors.get(self.backbone)
            prepared = {
                key: self.embedders[key].prepare_regions(image, processor=processor)
                for key in keys
            }
        except Exception as exc:
            return FailedImage.of(entry.identity, exc)
        return LoadedImage(identity=entry.identity, root_sha256=root, prepared=prepared)


def _stage4_loader(dataset: Stage4ImageDataset, num_workers: int) -> Any:
    """One backbone's ``DataLoader`` (plan review I3, re-review item 2a).

    ``batch_size=None`` with an identity ``collate_fn``, so an item is one
    image and ndarrays stay ndarrays; default in-order iteration; with
    workers, ``spawn`` (CUDA is live and the commit thread exists when they
    start), not persistent (no per-backbone state survives), and
    ``prefetch_factor=2``. ``num_workers=0`` prepares in this process.
    """
    import torch.utils.data as torch_data

    kwargs: dict[str, Any] = {
        "batch_size": None,
        "shuffle": False,
        "collate_fn": _identity_collate,
        "num_workers": num_workers,
    }
    if num_workers > 0:
        kwargs.update(
            multiprocessing_context="spawn", persistent_workers=False, prefetch_factor=2
        )
    return torch_data.DataLoader(dataset, **kwargs)


def plan_stage4(
    pipeline: ImagePipeline, *, revisions: Mapping[str, str | None] | None = None
) -> list[tuple[BackboneKey, list[tuple[str, MeasureDinoEmbedding]]]]:
    """The pipeline's embedders, grouped by model load, in backbone-major order (H10).

    Embedders whose :class:`BackboneKey` is equal share one load. A
    ``revision=None`` embedder takes ``revisions[hf_id]`` into its key (U5, P5
    T2), but the embedder itself is not copied, so its
    ``producer.params_sha256`` stays the inline one.

    Args:
        pipeline: The run's pipeline.
        revisions: ``{hf_id: commit}`` resolved at the run's first submission.

    Returns:
        ``[(backbone, [(meas key, embedder), ...]), ...]``, sorted by
        ``BackboneKey.sort_key()``, each group in ``meas`` order.

    Raises:
        ValueError: If an in-scope embedder is not a ``MeasureDinoEmbedding``
            (R2 refuses that before a run starts).
    """
    import dataclasses

    from phenotypic.detect.nn._helper._dino_support import hf_dino_id
    from phenotypic.measure import MeasureDinoEmbedding

    from ._cli_embed_stage import pipeline_embedders

    groups: dict[BackboneKey, list[tuple[str, MeasureDinoEmbedding]]] = {}
    for meas_key, embedder in pipeline_embedders(pipeline):
        if not isinstance(embedder, MeasureDinoEmbedding):
            raise ValueError(
                f"meas {meas_key!r} ({type(embedder).__name__}) has no GPU Stage-4 encoder"
            )
        key = embedder.backbone_key()
        if key.revision is None and revisions:
            resolved = revisions.get(hf_dino_id(embedder.version, key.size))
            if resolved:
                key = dataclasses.replace(key, revision=resolved)
        groups.setdefault(key, []).append((meas_key, embedder))
    return sorted(groups.items(), key=lambda item: item[0].sort_key())


def _live_root_sha256(store: Path) -> str | None:
    """sha256 of *store*'s root ``zarr.json`` bytes as they are now; ``None`` without one."""
    import hashlib

    try:
        return hashlib.sha256((Path(store) / "zarr.json").read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def _binds_of(payload: EmbeddingPayload) -> dict[str, Any]:
    """A payload descriptor's ``binds`` (empty when it has none)."""
    binds = payload.descriptor.get("binds")
    return dict(binds) if isinstance(binds, Mapping) else {}


def _series_of(payload: EmbeddingPayload) -> str:
    """The series a payload's regions were read from (its descriptor's ``input.series``)."""
    source = payload.descriptor.get("input")
    if not isinstance(source, Mapping) or not isinstance(source.get("series"), str):
        raise ValueError(f"the {payload.key} payload's descriptor names no input series")
    return str(source["series"])


def _level0(group: Any, path: str) -> Any:
    """The level-0 array at store-relative *path*, read in full through zarr-python."""
    import numpy as np

    return np.asarray(group[f"{path}/0"][...])


def _live_store_digests(
    store: Path, payloads: Sequence[EmbeddingPayload]
) -> tuple[str | None, dict[str, str]]:
    """``(objmap_sha256, array_sha256)`` read from the **live** store (plan review I11).

    The level-0 objmap (when a payload binds one) and each level-0 series a
    grid payload binds, read through zarr-python, never from the payloads'
    own ``binds``, so a payload computed from another image cannot certify
    itself.
    """
    import zarr

    from phenotypic.sdk_ import ngff_
    from phenotypic.sdk_._measurement_tables import array_digest, objmap_digest

    binds = [_binds_of(payload) for payload in payloads]
    series = {
        _series_of(payload) for payload, bound in zip(payloads, binds) if "array_sha256" in bound
    }
    group = zarr.open_group(str(store), mode="r")
    objmap_sha256: str | None = None
    if any("objmap_sha256" in bound for bound in binds):
        labels = ngff_.read_phenotypic_attributes(store).get(ngff_.PhenotypicAttr.LABELS) or {}
        path = labels.get(ngff_.OBJMAP_LABEL) if isinstance(labels, Mapping) else None
        if path:
            objmap_sha256 = objmap_digest(_level0(group, str(path)))
    arrays = {name: array_digest(_level0(group, name)) for name in sorted(series)}
    return objmap_sha256, arrays


def _bound_elsewhere(
    payloads: Mapping[str, EmbeddingPayload],
    objmap_sha256: str | None,
    array_sha256: Mapping[str, str],
) -> list[str]:
    """The keys whose ``binds`` do not match the live store's digests."""
    wrong = []
    for key, payload in payloads.items():
        bound = _binds_of(payload)
        if "objmap_sha256" in bound and bound["objmap_sha256"] != objmap_sha256:
            wrong.append(key)
        elif "array_sha256" in bound:
            if bound["array_sha256"] != array_sha256.get(_series_of(payload)):
                wrong.append(key)
    return wrong


def _gpu_total_bytes(device: str) -> int:
    """The GPU's **total** memory (never its free memory); a fixed total on the CPU escape."""
    import torch

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        return _CPU_TEST_GPU_TOTAL_BYTES
    index = torch_device.index if torch_device.index is not None else torch.cuda.current_device()
    return int(torch.cuda.get_device_properties(index).total_memory)


def _takes_geometry(embedder: MeasureDinoEmbedding, h: int, w: int) -> bool:
    """Whether *embedder*'s inputs can have shape ``h x w``: its crop, or one of its tiles."""
    if embedder.pooling == "masked_dense":
        return h <= embedder.crop_px and w <= embedder.crop_px
    return (h, w) == (embedder.crop_px, embedder.crop_px)


def _physical_batch_rule(
    model: Any,
    backbone: BackboneKey,
    embedders: Sequence[MeasureDinoEmbedding],
    device: str,
) -> Callable[[int, int], int]:
    """One ``m`` per queue geometry, from the rule's one producer (Step 3c).

    ``physical_batch_for(model, crop_px=(h, w), ...)`` capped by the smallest
    ``batch_size`` of the backbone's embedders whose inputs have that
    geometry (a crop of that side, or a ``masked_dense`` tile no larger than
    its ``crop_px``), on the GPU's total memory. So the ``m`` P2 T4 measured is
    the ``m`` this runs at.
    """
    from phenotypic.detect.nn._helper._batch_encoder import physical_batch_for

    total = _gpu_total_bytes(device)
    every_cap = min(embedder.batch_size for embedder in embedders)

    def rule(h: int, w: int) -> int:
        caps = [e.batch_size for e in embedders if _takes_geometry(e, h, w)]
        return physical_batch_for(
            model,
            crop_px=(h, w),
            batch_size=min(caps) if caps else every_cap,
            compute_dtype=backbone.compute_dtype,
            gpu_total_bytes=total,
        )

    return rule


def _require_licences(plan: Sequence[tuple[BackboneKey, Any]]) -> None:
    """The DINOv3 licence, refused before any load (spec §4.5: again at shard start)."""
    if not any(int(backbone.version) == 3 for backbone, _ in plan):
        return
    from phenotypic.detect.nn._helper._checkpoint_manager import (
        Dinov3CheckpointManager,
        require_license_acceptance,
    )

    try:
        require_license_acceptance(
            Dinov3CheckpointManager.license_key,
            Dinov3CheckpointManager.license_name,
            Dinov3CheckpointManager.license_url,
            interactive=False,
        )
    except RuntimeError as exc:
        raise Stage4ShardError(f"Stage 4 loads no DINOv3 backbone: {exc}") from exc


Stage4Action = Literal["commit", "publish", "embed"]


@dataclass
class _ImageWork:
    """One image's Stage-4 state in this call.

    ``payloads`` and ``reused`` hold the image's vectors only until it is
    done: :meth:`release` drops them once its commit finishes (or it fails),
    so a call's memory does not grow with the shard (review I3).
    """

    entry: StagedManifestEntry
    run_id: str
    action: Stage4Action
    reused: dict[str, Stage4Signal] = field(default_factory=dict)
    todo: list[str] = field(default_factory=list)
    payloads: dict[str, EmbeddingPayload] = field(default_factory=dict)
    roots: set[str] = field(default_factory=set)
    failed: bool = False

    def release(self) -> None:
        """Drop this image's payloads and reused signals; nothing reads them again."""
        self.payloads.clear()
        self.reused.clear()


@dataclass(frozen=True)
class _Stage4Context:
    """What every durable step of one :func:`run_stage4` call needs."""

    output_dir: Path
    output_manager: OutputManager
    lifecycle_epoch: str
    active_check: Callable[[], None] | None
    commit_guard: CommitGuard | None
    record_lock: CommitGuard | None = None

    def check(self) -> None:
        """The fence check that precedes every durable step."""
        if self.active_check is not None:
            self.active_check()

    @property
    def event_log(self) -> Path:
        return event_log_path(self.output_dir)


def _event(ctx: _Stage4Context, entry: StagedManifestEntry, status: str, error: str = "") -> None:
    """A ``stage4``-tagged event for *entry* (``started`` / ``completed`` / ``failed``)."""
    from ._cli_update_state import append_completion_event, append_event
    from ._stages import STAGE_EMBED

    ctx.check()
    if status == "started":
        append_event(
            ctx.event_log, entry.dataset, entry.image_name, "started",
            stage=STAGE_EMBED, commit_guard=ctx.commit_guard,
        )
        return
    append_completion_event(
        ctx.event_log, entry.dataset, entry.image_name, status,  # type: ignore[arg-type]
        error, stage=STAGE_EMBED, commit_guard=ctx.commit_guard,
    )


def _fail_image(
    ctx: _Stage4Context, entry: StagedManifestEntry, exc: BaseException, traceback_text: str = ""
) -> str:
    """Fail *entry* as a per-image ``stage4`` failure: journal it, drop its signals.

    Terminal when the exception is (``append_terminal_failure`` refuses an
    out-of-memory one). Nothing is committed and nothing is published.

    Returns:
        The error type, for :attr:`Stage4Outcome.failed`.
    """
    from ._cli_failure_tracker import append_terminal_failure
    from ._stages import STAGE_EMBED

    logger.warning(
        "Stage 4 failed %s/%s: %s: %s", entry.dataset, entry.image_name, type(exc).__name__, exc
    )
    ctx.check()
    with publication_commit(ctx.record_lock):
        append_terminal_failure(
            ctx.output_dir,
            work_id=entry.work_id,
            dataset=entry.dataset,
            relative_image_path=entry.relative_image_path or entry.image_name,
            failed_stage=STAGE_EMBED,
            exception=exc,  # type: ignore[arg-type]
            attempt_id=entry.attempt_id or uuid4().hex,
            lifecycle_epoch=ctx.lifecycle_epoch,
            traceback=traceback_text or "".join(traceback_module.format_exception(exc)),
            commit_guard=ctx.commit_guard,
        )
    ctx.check()
    delete_stage4_signals(ctx.output_dir, entry.dataset, entry.stem, commit_guard=ctx.commit_guard)
    _event(ctx, entry, "failed", f"{type(exc).__name__}: {exc}")
    return type(exc).__name__


def _store_holds_fresh_embeddings(
    store: Path, run: Any, embedders: Mapping[str, MeasureDinoEmbedding]
) -> bool:
    """Whether *run* already holds exactly these keys, fresh, verified, from these embedders.

    The "store promoted but no record" state (spec §4.6): every key's entry
    reads (``read_embedding_entry``: present, its sha256, not stale), its
    ``producer.params_sha256`` is the embedder's, and its ``compute_dtype``
    is the embedder's (an entry without one reads ``COMPUTE_DTYPE_DEFAULT``).
    """
    from phenotypic.sdk_._embedding_tables import read_embedding_entry
    from phenotypic.sdk_.typing_ import COMPUTE_DTYPE_DEFAULT

    block = run.descriptor.get("embeddings")
    if not isinstance(block, Mapping) or set(block) != set(embedders):
        return False
    for key, embedder in embedders.items():
        entry = block[key]
        if not isinstance(entry, Mapping):
            return False
        producer = entry.get("producer")
        if not isinstance(producer, Mapping) or producer.get("params_sha256") != embedder._params_sha256():
            return False
        if entry.get("compute_dtype", COMPUTE_DTYPE_DEFAULT) != embedder.compute_dtype:
            return False
        try:
            read_embedding_entry(store, key, entry)
        except Exception:
            return False
    return True


def _classify_image(
    ctx: _Stage4Context,
    entry: StagedManifestEntry,
    embedders: Mapping[str, MeasureDinoEmbedding],
) -> _ImageWork | str:
    """What *entry* needs (spec §4.4 loop step 1, §4.6), or ``"pending"``.

    Its signals are classified per key against the live root, and the stale
    ones deleted (plan review C5). Every key valid: commit only. The store
    already holding every key fresh: publish only. Otherwise the keys without
    a valid signal go to the loader (plan review C4).
    """
    store = zarr_store_path(ctx.output_dir, entry.dataset, entry.stem)
    live = _live_root_sha256(store)
    run = journal_table_run(store) if live is not None else None
    if live is None or run is None:
        return "pending"
    keys = list(embedders)
    state = classify_stage4_signals(
        ctx.output_dir, entry.dataset, entry.stem, keys,
        work_id=entry.work_id, run_id=run.run_id, live_root_sha256=live,
    )
    _event(ctx, entry, "started")
    if state.stale:
        ctx.check()
        delete_stage4_signal_keys(
            ctx.output_dir, entry.dataset, entry.stem, state.stale, commit_guard=ctx.commit_guard
        )
    if state.valid == set(keys):
        return _ImageWork(entry, run.run_id, "commit", reused=dict(state.signals))
    if _store_holds_fresh_embeddings(store, run, embedders):
        return _ImageWork(entry, run.run_id, "publish")
    return _ImageWork(
        entry, run.run_id, "embed",
        reused=dict(state.signals), todo=[key for key in keys if key not in state.valid],
    )


CommitStatus = tuple[Literal["published", "pending", "failed"], str]


def _publish(ctx: _Stage4Context, work: _ImageWork) -> CommitStatus:
    """The record, then the signals consumed (spec §4.2 steps 2-3)."""
    entry = work.entry
    data_key, data_path = image_data_artifact(
        ctx.output_dir, ctx.output_manager, entry.dataset, entry.stem
    )
    artifacts = {
        "measurements": image_measurement_artifact(ctx.output_dir, entry.dataset, entry.stem),
        data_key: data_path,
    }
    if ctx.output_manager.save_overlays:
        artifacts["overlay"] = ctx.output_manager.get_output_path(
            entry.dataset, "overlays", entry.stem
        )
    ctx.check()
    with publication_commit(ctx.record_lock):
        publish_image_success(
            ctx.output_dir,
            work_id=entry.work_id,
            dataset=entry.dataset,
            relative_image_path=entry.relative_image_path or entry.image_name,
            image_stem=entry.stem,
            mode="full",
            attempt_id=entry.attempt_id or uuid4().hex,
            lifecycle_epoch=ctx.lifecycle_epoch,
            artifacts=artifacts,
            commit_guard=ctx.commit_guard,
        )
    ctx.check()
    delete_stage4_signals(ctx.output_dir, entry.dataset, entry.stem, commit_guard=ctx.commit_guard)
    _event(ctx, entry, "completed")
    return "published", ""


def _leave_pending(ctx: _Stage4Context, entry: StagedManifestEntry, why: str) -> CommitStatus:
    """Drop *entry*'s signals and leave it for another round (a root change: never terminal)."""
    logger.warning("Stage 4 left %s/%s pending: %s", entry.dataset, entry.image_name, why)
    ctx.check()
    delete_stage4_signals(ctx.output_dir, entry.dataset, entry.stem, commit_guard=ctx.commit_guard)
    return "pending", why


def _commit_image(ctx: _Stage4Context, work: _ImageWork) -> CommitStatus:
    """Commit one image whose signals are all valid: store, then record, then signals.

    Runs on the one commit thread, so per-image order holds by construction
    (plan review I5). However it ends, the image's payloads and reused
    signals are released (:meth:`_ImageWork.release`, review I3).

    1. The live root must be the signals' root; otherwise they are deleted
       and the image stays pending (spec §4.8, plan review C5).
    2. The digests come from the **live** store (plan review I11). A payload
       bound to another image fails the image; nothing is committed or
       published (never a published-then-dropped entry).
    3. ``replace_run_embeddings``, fenced on the signals' root and durable
       as the run's ``--durable-writes`` says (``output_manager``); a
       ``StoreRootChangedError`` leaves the image pending.
    4. The record, then the signals consumed (:func:`_publish`).
    """
    try:
        return _commit_image_once(ctx, work)
    finally:
        work.release()


def _commit_image_once(ctx: _Stage4Context, work: _ImageWork) -> CommitStatus:
    """:func:`_commit_image`'s body: the four steps above, in order."""
    from phenotypic.sdk_._measurement_tables import StoreRootChangedError

    if work.action == "publish":
        return _publish(ctx, work)
    entry = work.entry
    store = zarr_store_path(ctx.output_dir, entry.dataset, entry.stem)
    payloads = {key: signal.payload for key, signal in work.reused.items()}
    payloads.update(work.payloads)
    roots = {signal.root_sha256 for signal in work.reused.values()} | work.roots
    ctx.check()
    live = _live_root_sha256(store)
    if len(roots) != 1 or live not in roots:
        return _leave_pending(ctx, entry, "its store root changed since its signals were computed")
    (root,) = roots
    try:
        objmap_sha256, array_sha256 = _live_store_digests(store, list(payloads.values()))
    except Exception as exc:
        return "failed", _fail_image(ctx, entry, exc)
    wrong = _bound_elsewhere(payloads, objmap_sha256, array_sha256)
    if wrong:
        return "failed", _fail_image(
            ctx,
            entry,
            Stage4ImageError(
                f"the Stage-4 payloads {wrong} are bound to another objmap or array "
                "than the store holds"
            ),
        )
    ctx.check()
    try:
        replace_run_embeddings(
            store,
            work.run_id,
            payloads,
            expected_root_sha256=root,
            objmap_sha256=objmap_sha256,
            array_sha256=array_sha256,
            # The run's --durable-writes tri-state, from the OutputManager like
            # every other store writer (``_cli/CLAUDE.md`` "Durability"); None
            # is auto-detect, resolved in ``ngff_._resolve_durability`` only.
            durable=ctx.output_manager.durable_writes,
            commit_guard=ctx.commit_guard,
        )
    except StoreRootChangedError as exc:
        return _leave_pending(ctx, entry, str(exc))
    return _publish(ctx, work)


def _raise_commit_failure(futures: Sequence[Future[CommitStatus]]) -> None:
    """Re-raise the first commit that raised (a fence, an I/O error): the stage stops."""
    for future in futures:
        if future.done() and future.exception() is not None:
            future.result()


def _encoder_call(call: Callable[[], Any]) -> Any:
    """Run one ``BatchEncoder`` call; any failure but OOM ends the shard (Step 3e).

    A failed pass poisons the encoder (#282 P1/P2 review I6): it holds rows
    of several images, so no image is failed for it, and Stage 4 never goes
    on with that encoder.
    """
    from phenotypic.detect.nn._helper._batch_encoder import Stage4OutOfMemory

    try:
        return call()
    except (Stage4OutOfMemory, MemoryError):
        raise
    except Exception as exc:
        raise Stage4ShardError(
            f"a Stage-4 forward pass failed ({type(exc).__name__}: {exc}); the "
            "images it held stay pending for the next attempt"
        ) from exc


class _BackboneStream:
    """One backbone's pass over its images: submit, drain, signal (spec §4.4 loop step 3).

    A key's signal is written as its ticket drains, so a walltime loses at
    most the batches in flight. An image whose last pending key is signalled
    goes to *on_complete* (the commit pool). An error of the image's own
    (from ``vectors_from`` or ``payload_from``, a loader ``FailedImage``)
    fails that image; any ``BatchEncoder`` error ends the shard
    (:func:`_encoder_call`).
    """

    def __init__(
        self,
        ctx: _Stage4Context,
        encoder: BatchEncoder,
        model: Any,
        embedders: Mapping[str, MeasureDinoEmbedding],
        works: Sequence[_ImageWork],
        *,
        device: str,
        outcome: Stage4Outcome,
        on_complete: Callable[[_ImageWork], None],
    ) -> None:
        self.ctx, self.encoder, self.model = ctx, encoder, model
        self.embedders = embedders
        self.by_identity = {work.entry.identity: work for work in works}
        self.device, self.outcome, self.on_complete = device, outcome, on_complete
        self.held: dict[tuple[str, str], tuple[PreparedRegions, str]] = {}

    def run(self, loader: Any) -> None:
        """Stream every loader item through the encoder, then flush and drain the rest."""
        import numpy as np

        for item in loader:
            if isinstance(item, FailedImage):
                self._loader_failure(item)
                continue
            work = self.by_identity[item.identity]
            for key, prepared in item.prepared.items():
                if work.failed:
                    break
                if prepared.n_regions == 0:
                    # Nothing reaches the GPU for an image with no regions
                    # (README Review Focus 1); it still commits an empty table.
                    self._signal(work, key, prepared, np.zeros((0, 0), np.float32), (), item.root_sha256)
                    continue
                self.held[(item.identity, key)] = (prepared, item.root_sha256)
                ticket = (item.identity, key)
                _encoder_call(lambda: self.encoder.submit(ticket, prepared.inputs))
                self._drain()
        _encoder_call(self.encoder.flush)
        self._drain()

    def _fail(self, work: _ImageWork, exc: BaseException, traceback_text: str = "") -> None:
        work.failed = True
        work.release()  # its signalled keys are never committed (review I3)
        self.outcome.failed[work.entry.identity] = _fail_image(ctx=self.ctx, entry=work.entry, exc=exc,
                                                                traceback_text=traceback_text)

    def _loader_failure(self, item: FailedImage) -> None:
        if item.node_state:
            raise MemoryError(
                f"a Stage-4 loader ran out of memory preparing {item.identity!r}: {item.message}"
            )
        work = self.by_identity[item.identity]
        if not work.failed:
            self._fail(work, item.exception(), item.traceback)

    def _drain(self) -> None:
        from phenotypic.abc_._measure_embedding import is_out_of_memory

        for drained in _encoder_call(self.encoder.drain):
            identity, key = drained.ticket
            prepared, root = self.held.pop((identity, key))
            work = self.by_identity[identity]
            if work.failed:
                continue
            try:
                vectors = self.embedders[key].vectors_from(prepared, drained.outputs)
            except Exception as exc:
                if is_out_of_memory(exc):
                    raise
                self._fail(work, exc)
                continue
            self._signal(work, key, prepared, vectors, drained.physical_batches, root)

    def _signal(
        self,
        work: _ImageWork,
        key: str,
        prepared: PreparedRegions,
        vectors: Any,
        sizes: Sequence[int],
        root: str,
    ) -> None:
        """Turn one drained key into its payload and write its signal."""
        from phenotypic.abc_._measure_embedding import is_out_of_memory
        from phenotypic.detect.nn._helper._dino_support import compute_block

        if work.failed:
            return
        embedder = self.embedders[key]
        try:
            compute = compute_block(device=self.device, stage="stage4", physical_batch=sizes)
            payload = embedder.payload_from(
                prepared, vectors, embedder.entry_descriptor_for(self.model, compute)
            )
        except Exception as exc:
            if is_out_of_memory(exc):
                raise
            self._fail(work, exc)
            return
        self.ctx.check()
        write_stage4_signal(
            self.ctx.output_dir, work.entry.dataset, work.entry.stem, payload,
            work_id=work.entry.work_id, run_id=work.run_id, root_sha256=root,
            commit_guard=self.ctx.commit_guard,
        )
        self.outcome.signals_written += 1
        work.payloads[key] = payload
        work.roots.add(root)
        work.todo.remove(key)
        if not work.todo:
            self.on_complete(work)


def _embed_backbone(
    ctx: _Stage4Context,
    backbone: BackboneKey,
    group: Sequence[tuple[str, MeasureDinoEmbedding]],
    works: Sequence[_ImageWork],
    *,
    device: str,
    image_type: ImageTypeName,
    num_workers: int,
    outcome: Stage4Outcome,
    on_complete: Callable[[_ImageWork], None],
) -> None:
    """Load one backbone once and run its images through one ``BatchEncoder`` (H10).

    Loaded through the measurer module's ``load_dino_backbone`` (so a test
    stub applies; never ``_cached_backbone``) in ``key.compute_dtype``, its
    geometry checked for every embedder of the backbone before the first
    pass (Step 3a), its ``m`` per queue geometry (Step 3c), then streamed
    (:class:`_BackboneStream`). The model is dropped before the next load.
    """
    from phenotypic.detect.nn._helper._batch_encoder import BatchEncoder
    from phenotypic.measure import _measure_dino_embedding as dino_module

    embedders = {embedder.payload_key(): embedder for _, embedder in group}
    model: Any
    processor: Any
    model, processor = dino_module.load_dino_backbone(
        backbone.version,  # type: ignore[arg-type]  # BackboneKey.version is 2 or 3
        backbone.size, device, revision=backbone.revision, dtype=backbone.compute_dtype,
    )
    encoder: BatchEncoder | None = None
    try:
        for embedder in embedders.values():
            try:
                embedder._check_geometry(model)
            except ValueError as exc:
                raise Stage4ShardError(f"Stage 4 cannot run this backbone: {exc}") from exc
        encoder = BatchEncoder(
            model, processor, device=device, compute_dtype=backbone.compute_dtype,
            physical_batch=_physical_batch_rule(model, backbone, list(embedders.values()), device),
        )
        dense = any(e.pooling == "masked_dense" for e in embedders.values())
        dataset = Stage4ImageDataset(
            ctx.output_dir,
            image_type,
            [(work.entry, tuple(k for k in work.todo if k in embedders)) for work in works],
            embedders,
            {backbone: processor} if dense else {},
            backbone,
        )
        _BackboneStream(
            ctx, encoder, model, embedders, works,
            device=device, outcome=outcome, on_complete=on_complete,
        ).run(_stage4_loader(dataset, num_workers))
        outcome.physical_batches[backbone] = encoder.physical_batches
    finally:
        # One model resident at a time (H10): drop every reference before the next load.
        encoder = model = processor = None
        import torch

        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()


def run_stage4(
    pipeline_path: Path,
    output_dir: Path,
    items: Sequence[StagedManifestEntry],
    *,
    image_type: ImageTypeName,
    epoch: str | None,
    republish: bool = False,
    revisions: Mapping[str, str | None] | None = None,
    num_workers: int,
    output_manager: OutputManager,
    active_check: Callable[[], None] | None = None,
    commit_guard: CommitGuard | None = None,
    record_lock: CommitGuard | None = None,
) -> Stage4Outcome:
    """Embed, commit and publish every image of *items* (spec §4.4).

    *items* are images that finished Stages 1-3 (``stage4_status ==
    "pending"``). The loop:

    1. the DINOv3 licence, before any load; then ``stage4_device()`` and
       ``configure_determinism(device=...)``, the device passed explicitly
       (Step 3b);
    2. per image, in order: its signals classified per key against the live
       root (stale ones deleted), then commit only, publish only, or embed
       the keys without a valid signal (:func:`_classify_image`);
    3. per backbone, in plan order: one load, one loader, one encoder, a
       signal per key as it drains (:func:`_embed_backbone`);
    4. one commit thread, overlapping the GPU work, commits each image whose
       signals are all valid (:func:`_commit_image`).

    Every durable step is preceded by *active_check*, and every durable
    write runs under *commit_guard*.

    Args:
        pipeline_path: The run's pipeline file.
        output_dir: Run output root.
        items: The images, as manifest entries.
        image_type: ``"Image"`` or ``"GridImage"``.
        epoch: The lifecycle epoch records and failures are stamped with;
            ``None`` stamps ``"local-unfenced"``.
        republish: Measure mode's republish Stage 4 (P6 T1); not yet built.
        revisions: ``{hf_id: commit}`` for unpinned embedders (U5, P5 T2).
        num_workers: DataLoader workers; 0 prepares in this process.
        output_manager: The run's output manager (artifact paths).
        active_check: The fence check; raises when the epoch is fenced.
        commit_guard: The publication guard, threaded to every durable write
            (the SLURM lifecycle's).
        record_lock: Held around each record and each failure journal
            entry, outermost: the local epoch ownership, which the local
            strategies take around theirs. It reads the event log under its
            lock, so it is never a *commit_guard* (an event append holds that
            lock when it calls its guard).

    Returns:
        The :class:`Stage4Outcome`.

    Raises:
        Stage4DeviceError: No usable CUDA (nothing journalled).
        Stage4OutOfMemory: A queue cannot halve again (nothing journalled).
        MemoryError: The host ran out (nothing journalled).
        Stage4ShardError: As the class says (nothing journalled).
        NotImplementedError: For *republish* (P6 T1).
        Exception: Whatever *active_check* or *commit_guard* raise.
    """
    from phenotypic import ImagePipeline
    from phenotypic.detect.nn._helper._batch_encoder import configure_determinism

    if republish:
        raise NotImplementedError("republish-mode Stage 4 is built in #282 P6 T1")
    outcome = Stage4Outcome()
    if not items:
        return outcome
    plan = plan_stage4(ImagePipeline.from_json(pipeline_path), revisions=revisions)
    embedders = {
        embedder.payload_key(): embedder for _, group in plan for _, embedder in group
    }
    _require_licences(plan)
    device = stage4_device()
    configure_determinism(device=device)
    ctx = _Stage4Context(
        output_dir=Path(output_dir),
        output_manager=output_manager,
        lifecycle_epoch=epoch or "local-unfenced",
        active_check=active_check,
        commit_guard=commit_guard,
        record_lock=record_lock,
    )
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stage4-commit")
    futures: list[Future[CommitStatus]] = []
    identities: list[str] = []

    def _submit(work: _ImageWork) -> None:
        futures.append(pool.submit(_commit_image, ctx, work))
        identities.append(work.entry.identity)

    try:
        to_embed: list[_ImageWork] = []
        for entry in items:
            ctx.check()
            work = _classify_image(ctx, entry, embedders)
            if isinstance(work, str):
                outcome.pending.append(entry.identity)
            elif work.action == "embed":
                to_embed.append(work)
            else:
                _submit(work)
        for backbone, group in plan:
            keys = {embedder.payload_key() for _, embedder in group}
            works = [w for w in to_embed if not w.failed and keys.intersection(w.todo)]
            if not works:
                continue
            _raise_commit_failure(futures)
            _embed_backbone(
                ctx, backbone, group, works, device=device, image_type=image_type,
                num_workers=num_workers, outcome=outcome, on_complete=_submit,
            )
    finally:
        pool.shutdown(wait=True)
    for identity, future in zip(identities, futures):
        status, detail = future.result()
        if status == "published":
            outcome.published.append(identity)
        elif status == "pending":
            outcome.pending.append(identity)
        else:
            outcome.failed[identity] = detail
    return outcome


def sweep_orphan_stage4_signals(
    output_dir: Path,
    inventory: Mapping[str, Sequence[str]],
    work_ids: Mapping[str, Mapping[str, str]],
    *,
    commit_guard: CommitGuard | None = None,
) -> int:
    """Remove the leftover signals of every published image of the whole inventory (I7).

    A crash between an image's record and its signals' consumption leaves
    signals no loop will reach, because a done image never enters Stage 4.
    So the sweep walks the **whole** inventory: an image with signal files or
    signal temp files (``.<stem>.parquet.*.tmp``, a crash mid-write) and a
    valid record loses them. A pending image's signals are never touched.

    Args:
        output_dir: Run output root.
        inventory: ``{dataset: [image name, ...]}``.
        work_ids: ``{dataset: {image name: work id}}``.
        commit_guard: Guards each removal.

    Returns:
        How many images lost leftovers.
    """
    from ._cli_completion import valid_image_success
    from ._cli_stage4_signal import stage4_signal_stems

    swept = 0
    for dataset, images in inventory.items():
        leftovers = stage4_signal_stems(output_dir, dataset)
        if not leftovers:
            continue
        for image in images:
            stem = source_image_stem(Path(image))
            work_id = (work_ids.get(dataset) or {}).get(image)
            if stem not in leftovers or not work_id:
                continue
            if not valid_image_success(
                output_dir, dataset=dataset, image_stem=stem, work_id=work_id
            ):
                continue
            removed = delete_stage4_signals(output_dir, dataset, stem, commit_guard=commit_guard)
            removed += delete_stage4_signal_temps(output_dir, dataset, stem, commit_guard=commit_guard)
            swept += bool(removed)
    return swept


def split_stage4_pending(
    datasets: Sequence[Dataset],
    output_dir: Path,
    entries: Mapping[tuple[str, Path], StagedManifestEntry],
) -> tuple[list[Dataset], list[StagedManifestEntry]]:
    """Split a deferring run's worklist into the images that still need the core and Stage 4's.

    Content-defined (spec §4.6), from each image's ``stage4_status``:

    - ``"not_ready"``: Stages 1-3 have work left, so the image goes to the
      core (the single-pass core, or the staged stages);
    - ``"pending"``: Stages 1-3 are done and only Stage 4 is left. The image
      skips the core, which would re-measure it and drop its Stage-4 signals
      (plan review C1), and is one of the returned Stage-4 items;
    - ``"done"``: its record is valid; it needs neither.

    The no-detector continuation's planner (#282 P4 T4), the counterpart of
    :func:`~phenotypic._cli._cli_staged_resume.build_staged_resume_plan` for a
    run without a ``GpuDetector``; P5 T2 reuses it for the SLURM one.

    It hashes no input (review I4): *entries* carry each image's one
    identity for the invocation, computed once by the caller
    (:func:`stage4_entries`, or the identities a strategy already holds).

    Args:
        datasets: The worklist.
        output_dir: Run output root.
        entries: ``{(dataset name, image path): manifest entry}`` for every
            image of *datasets*.

    Returns:
        ``(core datasets, Stage-4 items)``: the datasets with only their
        ``"not_ready"`` images (a dataset left empty is dropped), and every
        ``"pending"`` image as a manifest entry, in worklist order.
    """
    from ._cli_staged_resume import stage4_status
    from ._cli_types import Dataset

    core: list[Dataset] = []
    pending: list[StagedManifestEntry] = []
    for dataset in datasets:
        images = []
        for image in dataset.images:
            entry = entries[(dataset.name, image)]
            status = stage4_status(output_dir, dataset.name, entry.stem, entry.work_id)
            if status == "not_ready":
                images.append(image)
            elif status == "pending":
                pending.append(entry)
        if images:
            core.append(
                Dataset(
                    name=dataset.name,
                    images=images,
                    input_dir=dataset.input_dir,
                    output_dir=dataset.output_dir,
                )
            )
    return core, pending


def stage4_entries(
    config: ExecutionConfig,
    tasks: Sequence[tuple[Dataset, Path]],
    *,
    n_jobs: int = 1,
) -> dict[tuple[str, Path], StagedManifestEntry]:
    """Each task's manifest entry under its **one** identity for this invocation (review I4).

    ``work_identity_for_image`` hashes the input; this is the only place a
    deferring local run without a ``GpuDetector`` computes it. The core is
    then handed the same identity (never recomputing it), and so is Stage 4,
    after Stages 1-3. Threads, not processes: ``hashlib`` releases the GIL.

    Args:
        config: The run's configuration.
        tasks: ``[(dataset, image path)]``.
        n_jobs: joblib's ``n_jobs`` for the hashing threads.

    Returns:
        ``{(dataset name, image path): entry}``.
    """
    from joblib import Parallel, delayed

    from ._cli_failure_tracker import work_identity_for_image
    from ._cli_staged_slurm import manifest_entry_for_identity

    identities = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(work_identity_for_image)(config, dataset.name, image) for dataset, image in tasks
    )
    return {
        (dataset.name, image): manifest_entry_for_identity(dataset.name, image, identity)
        for (dataset, image), identity in zip(tasks, identities)
    }


def sweep_run_orphan_stage4_signals(output_dir: Path) -> int:
    """:func:`sweep_orphan_stage4_signals` over the run's whole recorded inventory.

    The inventory and the work ids are ``processing_state.json``'s
    ``work_ids``. Called by a local Stage 4 before it starts, and by the CLI
    before a finished continuation exits (``"All images already
    processed!"``), which never reaches Stage 4 (plan review I7).

    Returns:
        How many images lost leftovers (``0`` without a processing state).
    """
    from ._cli_state_management import load_processing_state

    state = load_processing_state(output_dir)
    work_ids = (state.config.get("work_ids") if state is not None else None) or {}
    if not isinstance(work_ids, Mapping):
        return 0
    return sweep_orphan_stage4_signals(
        output_dir, {dataset: list(images) for dataset, images in work_ids.items()}, work_ids
    )


def _local_record_lock(config: ExecutionConfig, output_dir: Path) -> CommitGuard:
    """The local epoch ownership each local record and failure is published under."""
    from ._cli_execution_strategies import _local_epoch_ownership

    return lambda: _local_epoch_ownership(config, output_dir)


def run_local_stage4(
    config: ExecutionConfig,
    output_manager: OutputManager,
    output_dir: Path,
    datasets: Sequence[Dataset],
    entries: Mapping[tuple[str, Path], StagedManifestEntry],
) -> dict[str, tuple[int, int]]:
    """Run Stage 4 in this process over *datasets*: ``{dataset: (published, failed)}``.

    Sweeps the whole inventory's orphan signals first, then hands every image
    of *datasets* whose ``stage4_status`` is ``"pending"`` to
    :func:`run_stage4`, except one with an exact terminal failure unless
    ``--retry-failures`` (a finished continuation hands this the whole
    inventory, the failed image included; its Stage-4 failure kept
    ``stages.stage3``, so it reads ``"pending"``), with ``resolve_local_worker_count(config.n_jobs, n) - 1``
    loader workers, so ``--njobs`` is honoured and ``--njobs 1`` prepares
    in-process (re-review item 2a). Every image it is handed ends published
    or journalled (plan review C2), or this raises.

    Args:
        config: The run's configuration.
        output_manager: The run's output manager.
        output_dir: Run output root.
        datasets: The worklist.
        entries: ``{(dataset name, image path): manifest entry}`` for every
            image of *datasets*, built from the identity Stages 1-3 ran
            under; Stage 4 never recomputes a work id (review I4).

    Raises:
        click.ClickException: Exit ``1``, after the commit pool drains and
            journalling no image: no usable CUDA, a GPU or host out of
            memory, a failed forward pass or another shard failure, or any
            image left pending (its store changed during the run).
    """
    import click

    from phenotypic.detect.nn._helper._batch_encoder import Stage4OutOfMemory

    from ._cli_failure_tracker import terminal_failure_index
    from ._cli_utils import resolve_local_worker_count

    sweep_run_orphan_stage4_signals(output_dir)
    _core, items = split_stage4_pending(datasets, output_dir, entries)
    if not config.retry_failures:
        terminal = terminal_failure_index(output_dir)
        items = [entry for entry in items if entry.work_id not in terminal]
    if not items:
        return {}
    num_workers = resolve_local_worker_count(config.n_jobs, len(items)) - 1
    try:
        outcome = run_stage4(
            config.pipeline_json,
            output_dir,
            items,
            image_type=config.image_type,
            epoch=config.processing_generation,
            num_workers=num_workers,
            output_manager=output_manager,
            record_lock=_local_record_lock(config, output_dir),
        )
    except (Stage4DeviceError, Stage4OutOfMemory, Stage4ShardError, MemoryError) as exc:
        raise click.ClickException(
            f"Stage 4 stopped: {exc}. No image was recorded as failed; run the "
            "same command again"
        ) from exc
    if outcome.pending:
        raise click.ClickException(
            f"{len(outcome.pending)} image(s) still need Stage 4 because their store "
            "changed during the run; run the same command again"
        )
    counts = {dataset.name: [0, 0] for dataset in datasets}
    by_identity = {entry.identity: entry.dataset for entry in items}
    for identity in outcome.published:
        counts[by_identity[identity]][0] += 1
    for identity in outcome.failed:
        counts[by_identity[identity]][1] += 1
    return {name: (published, failed) for name, (published, failed) in counts.items()}
