"""The Stage-4 signal: one image's embedding for one key, before the store commit.

Stage 4 reads a finished store read-only, computes each embedder key's
payload on a GPU, and writes it here, one atomic Parquet per key::

    <output>/.phenotypic/progress/stage4_payload/<ds>/<model>/<layer>/<region>/<stem>.parquet

The table is :func:`~phenotypic.sdk_._embedding_tables.embedding_arrow_table`
of the payload -- the same table the store commit later writes -- and its
Parquet key-value metadata carries, under :data:`SIGNAL_METADATA_KEY`, a JSON
footer ``{work_id, run_id, root_sha256, key, descriptor}``. The order per image
is fixed (spec §4.2): signals, then the store transaction
(``replace_run_embeddings``), then the record, then the signals are consumed.

**The validity rule** (plan review C4, C5), stated once, in
:func:`classify_stage4_signals`. A key's signal is *valid* only when its file

* reads, as a table whose schema fits its footer's descriptor;
* is fenced to this ``work_id`` and this ``run_id``;
* names, in its footer, the key whose directory it sits in; and
* was computed against the store root that is live **now**
  (``root_sha256 == live_root_sha256``).

A file present but failing any of these is *stale*: the caller deletes it
(:func:`delete_stage4_signal_keys`) and re-embeds that key. An absent file is
neither. Validity is **per key**, so a round interrupted after one backbone
keeps that backbone's keys and embeds only the rest; and it is against the
**live** root, so a signal computed before a reprocess is never committed onto
the reprocessed store.

Like the Stage-2 token, the signal is resume state and lives under
``progress/``, never in the store. ``--restart`` clears it with ``progress/``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from phenotypic.sdk_ import (
    DIR_STAGE4_PAYLOAD,
    PARQUET_WRITE_OPTIONS,
    CommitGuard,
    atomic_write_with_writer,
    progress_dir,
    publication_commit,
)

if TYPE_CHECKING:
    import pyarrow as pa  # type: ignore[import-untyped]

    from phenotypic.sdk_._embedding_tables import EmbeddingPayload

logger = logging.getLogger(__name__)

#: The Parquet key-value metadata key holding a signal's JSON footer.
SIGNAL_METADATA_KEY: Final[bytes] = b"phenotypic.stage4"

#: The footer's string fields; ``descriptor`` is the one mapping.
_FOOTER_STRING_FIELDS: Final[tuple[str, ...]] = ("work_id", "run_id", "root_sha256", "key")

_SIGNAL_SUFFIX: Final[str] = ".parquet"


@dataclass(frozen=True)
class Stage4Signal:
    """A signal read back and fenced to the caller's work id and run.

    Attributes:
        payload: The key's payload, as Stage 4 computed it.
        root_sha256: sha256 of the store root ``zarr.json`` the payload was
            computed against: the commit's ``expected_root_sha256``.
    """

    payload: EmbeddingPayload
    root_sha256: str


@dataclass(frozen=True)
class Stage4SignalState:
    """One image's signals, classified per key against the live root.

    Attributes:
        valid: Keys whose signal can be committed as it stands.
        stale: Keys whose file is present but must be deleted and re-embedded:
            unreadable, fenced to another work id or run, filed under another
            key, or computed against another root.
        signals: The valid keys' signals, as :func:`classify_stage4_signals`
            read them, so a commit reads each file once (the classification
            reads every file in full anyway). Not part of equality.
    """

    valid: frozenset[str]
    stale: frozenset[str]
    signals: Mapping[str, Stage4Signal] = field(default_factory=dict, compare=False)


def _stage4_dataset_dir(output_dir: Path, dataset: str) -> Path:
    return progress_dir(output_dir) / DIR_STAGE4_PAYLOAD / dataset


def stage4_signal_path(output_dir: Path, dataset: str, key: str, stem: str) -> Path:
    """``<output>/.phenotypic/progress/stage4_payload/<ds>/<model>/<layer>/<region>/<stem>.parquet``.

    The key is parsed by
    :func:`~phenotypic.sdk_._table_runs.split_embedding_key`, the one key
    parser, so a key the store would refuse has no signal path either.

    Raises:
        ValueError: If *key* is not ``"<model>/<layer>/<region>"``.
    """
    from phenotypic.sdk_._table_runs import split_embedding_key

    model, layer, region = split_embedding_key(key)
    return _stage4_dataset_dir(output_dir, dataset) / model / layer / region / f"{stem}{_SIGNAL_SUFFIX}"


def _key_of_signal_path(path: Path) -> str:
    """The key whose directory *path* sits in (the inverse of :func:`stage4_signal_path`)."""
    return "/".join(path.parts[-4:-1])


def write_stage4_signal(
    output_dir: Path,
    dataset: str,
    stem: str,
    payload: EmbeddingPayload,
    *,
    work_id: str,
    run_id: str,
    root_sha256: str,
    commit_guard: CommitGuard | None = None,
) -> Path:
    """Atomically write *payload* as this image's signal for ``payload.key``.

    The table goes to a dot-prefixed sibling temp file, is fsynced, and is
    renamed over the final path under *commit_guard*, so a reader sees no
    file or the whole signal, and a fenced epoch publishes nothing. A signal
    already at the path (a stale one the caller kept) is replaced.

    Args:
        output_dir: Run output root.
        dataset: Dataset name.
        stem: Image stem.
        payload: The key's payload.
        work_id: The image's work id, the first half of the fence.
        run_id: The table run the payload belongs to, the second half.
        root_sha256: sha256 of the store root the payload was computed against.
        commit_guard: Guards the rename.

    Returns:
        The signal's path.

    Raises:
        ValueError: If the payload's frame does not fit its descriptor.
    """
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    from phenotypic.sdk_._embedding_tables import embedding_arrow_table

    final = stage4_signal_path(output_dir, dataset, payload.key, stem)
    footer = {
        "work_id": work_id,
        "run_id": run_id,
        "root_sha256": root_sha256,
        "key": payload.key,
        "descriptor": dict(payload.descriptor),
    }
    table = embedding_arrow_table(payload).replace_schema_metadata(
        {SIGNAL_METADATA_KEY: json.dumps(footer, sort_keys=True).encode("utf-8")}
    )

    def _write(temp_path: str) -> None:
        pq.write_table(table, temp_path, **PARQUET_WRITE_OPTIONS)

    atomic_write_with_writer(final, _write, commit_guard=commit_guard)
    return final


def _unreadable(path: Path, reason: str) -> None:
    logger.warning("Ignoring the unreadable Stage-4 signal %s: %s", path, reason)


def _read_footer(table: pa.Table) -> dict[str, Any] | str:
    """The footer of *table*, or the reason it has none a reader can use."""
    raw = (table.schema.metadata or {}).get(SIGNAL_METADATA_KEY)
    if raw is None:
        return f"no {SIGNAL_METADATA_KEY.decode()} footer"
    try:
        footer = json.loads(raw)
    except ValueError as exc:
        return f"its footer is not JSON ({exc})"
    if not isinstance(footer, dict):
        return "its footer is not a JSON object"
    for name in _FOOTER_STRING_FIELDS:
        if not isinstance(footer.get(name), str):
            return f"its footer has no string {name!r}"
    if not isinstance(footer.get("descriptor"), dict):
        return "its footer has no descriptor object"
    return footer


def _payload_from_table(table: pa.Table, footer: dict[str, Any]) -> EmbeddingPayload | str:
    """Rebuild the payload *table* holds, or the reason the table does not fit *footer*."""
    import numpy as np
    import pandas as pd
    import pyarrow as pa  # type: ignore[import-untyped]

    from phenotypic.sdk_._embedding_tables import EMBEDDING_COLUMN, EmbeddingPayload

    descriptor = footer["descriptor"]
    dim = descriptor.get("dim")
    if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
        return f"its descriptor's dim is {dim!r}"
    names = table.column_names
    if not names or names[-1] != EMBEDDING_COLUMN:
        return f"its last column is not {EMBEDDING_COLUMN!r}"
    vectors = table.column(EMBEDDING_COLUMN)
    if vectors.type != pa.list_(pa.float16(), dim) or vectors.null_count:
        return f"its {EMBEDDING_COLUMN!r} column is {vectors.type}, not FixedSizeList<float16>[{dim}]"
    keys = names[:-1]
    if any(table.schema.field(name).type != pa.int64() for name in keys):
        return "a key column is not int64"
    flat = vectors.combine_chunks().flatten().to_numpy(zero_copy_only=False)
    rows = np.asarray(flat, dtype=np.float16).reshape(table.num_rows, dim)
    frame = pd.DataFrame({name: table.column(name).to_numpy() for name in keys})
    frame[EMBEDDING_COLUMN] = pd.Series(list(rows), index=frame.index, dtype=object)
    try:
        return EmbeddingPayload(key=footer["key"], frame=frame, descriptor=descriptor)
    except ValueError as exc:
        return str(exc)


def read_stage4_signal(path: Path, *, work_id: str, run_id: str) -> Stage4Signal | None:
    """Read the signal at *path*, fenced to *work_id* and *run_id*.

    Args:
        path: A path from :func:`stage4_signal_path`.
        work_id: The image's current work id.
        run_id: The table run being committed.

    Returns:
        The signal, or ``None`` when the file is absent, unreadable, or fenced
        to another work id or run. An unreadable file -- not Parquet, no
        footer, a footer naming another key than its directory's, a table that
        does not fit its descriptor -- logs one warning naming it. The root is
        **not** checked here: :func:`classify_stage4_signals` compares it with
        the live root, and the commit uses it as ``expected_root_sha256``.
    """
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    path = Path(path)
    try:
        table = pq.read_table(path)
    except FileNotFoundError:
        return None
    except (pa.ArrowException, OSError) as exc:
        _unreadable(path, str(exc))
        return None
    footer = _read_footer(table)
    if isinstance(footer, str):
        _unreadable(path, footer)
        return None
    if footer["key"] != _key_of_signal_path(path):
        _unreadable(path, f"its footer names the key {footer['key']!r}")
        return None
    if footer["work_id"] != work_id or footer["run_id"] != run_id:
        logger.debug("Stage-4 signal %s is fenced to another work id or run", path)
        return None
    payload = _payload_from_table(table, footer)
    if isinstance(payload, str):
        _unreadable(path, payload)
        return None
    return Stage4Signal(payload=payload, root_sha256=footer["root_sha256"])


def classify_stage4_signals(
    output_dir: Path,
    dataset: str,
    stem: str,
    keys: Iterable[str],
    *,
    work_id: str,
    run_id: str,
    live_root_sha256: str,
) -> Stage4SignalState:
    """Classify each of *keys*' signals as valid, stale, or absent.

    The one validity rule (module docstring): valid when
    :func:`read_stage4_signal` returns a signal and its root is
    *live_root_sha256*; stale when a file is present but either fails; absent
    keys are in neither set. Every file is read in full, so "valid" means the
    commit's own read will succeed.

    Args:
        output_dir: Run output root.
        dataset: Dataset name.
        stem: Image stem.
        keys: The image's in-scope embedder keys.
        work_id: The image's current work id.
        run_id: The table run being committed.
        live_root_sha256: sha256 of the store root ``zarr.json`` as it is now.

    Returns:
        The two key sets, and the valid keys' signals.
    """
    signals: dict[str, Stage4Signal] = {}
    stale: set[str] = set()
    for key in keys:
        path = stage4_signal_path(output_dir, dataset, key, stem)
        if not path.is_file():
            continue
        signal = read_stage4_signal(path, work_id=work_id, run_id=run_id)
        if signal is not None and signal.root_sha256 == live_root_sha256:
            signals[key] = signal
        else:
            stale.add(key)
    return Stage4SignalState(
        valid=frozenset(signals), stale=frozenset(stale), signals=signals
    )


def stage4_signals_complete(state: Stage4SignalState, keys: Iterable[str]) -> bool:
    """Whether every one of *keys*, and nothing else, has a valid signal."""
    return state.valid == set(keys)


def _unlink_signal(path: Path, commit_guard: CommitGuard | None) -> int:
    """Remove one signal under *commit_guard*; 1 if a file was removed."""
    if not path.is_file():
        return 0
    with publication_commit(commit_guard):
        try:
            path.unlink()
        except FileNotFoundError:
            return 0
    return 1


def delete_stage4_signal_keys(
    output_dir: Path,
    dataset: str,
    stem: str,
    keys: Iterable[str],
    *,
    commit_guard: CommitGuard | None = None,
) -> int:
    """Remove this image's signals for *keys* only (the stale ones, before re-embedding).

    Returns:
        How many files were removed.
    """
    return sum(
        _unlink_signal(stage4_signal_path(output_dir, dataset, key, stem), commit_guard)
        for key in keys
    )


def delete_stage4_signals(
    output_dir: Path,
    dataset: str,
    stem: str,
    *,
    commit_guard: CommitGuard | None = None,
) -> int:
    """Remove this image's signal under every key directory of *dataset*.

    Walks the ``<model>/<layer>/<region>`` directories that exist rather than
    taking keys, so it also consumes signals for keys the current pipeline no
    longer has. The stem is joined, never globbed, so a stem holding ``[`` or
    ``*`` removes only its own files.

    Returns:
        How many files were removed.
    """
    return sum(
        _unlink_signal(region / f"{stem}{_SIGNAL_SUFFIX}", commit_guard)
        for region in _key_dirs(output_dir, dataset)
    )


def _key_dirs(output_dir: Path, dataset: str) -> list[Path]:
    """Every ``<model>/<layer>/<region>`` directory of *dataset*'s signals, sorted."""
    base = _stage4_dataset_dir(output_dir, dataset)
    if not base.is_dir():
        return []
    return [
        region
        for model in sorted(p for p in base.iterdir() if p.is_dir())
        for layer in sorted(p for p in model.iterdir() if p.is_dir())
        for region in sorted(p for p in layer.iterdir() if p.is_dir())
    ]


def delete_stage4_signal_temps(
    output_dir: Path,
    dataset: str,
    stem: str,
    *,
    commit_guard: CommitGuard | None = None,
) -> int:
    """Remove this image's leftover signal temp files, ``.<stem>.parquet.*.tmp``.

    :func:`write_stage4_signal` writes through a dot-prefixed sibling temp
    file (``atomic_write_with_writer``) that a crash before its rename
    leaves behind. Call this only for an image no writer can still be
    writing, e.g. one whose record is valid (the orphan sweep); a live
    writer's temp file is otherwise indistinguishable. Matched by prefix and
    suffix, never by a glob of the stem.

    Returns:
        How many files were removed.
    """
    prefix, suffix = f".{stem}{_SIGNAL_SUFFIX}.", ".tmp"
    return sum(
        _unlink_signal(path, commit_guard)
        for region in _key_dirs(output_dir, dataset)
        for path in sorted(region.iterdir())
        if path.name.startswith(prefix) and path.name.endswith(suffix)
    )


def stage4_signal_stems(output_dir: Path, dataset: str) -> set[str]:
    """The stems of *dataset* with a signal file or a signal temp file under any key.

    One directory listing per key directory, so a sweep over a large
    inventory stats each image's files once rather than listing every key
    directory per image. A temp file ``.<stem>.parquet.<random>.tmp`` names its
    stem before the last ``.parquet.``.
    """
    stems: set[str] = set()
    for region in _key_dirs(output_dir, dataset):
        for path in region.iterdir():
            name = path.name
            if name.startswith(".") and name.endswith(".tmp") and f"{_SIGNAL_SUFFIX}." in name:
                stems.add(name[1 : name.rindex(f"{_SIGNAL_SUFFIX}.")])
            elif not name.startswith(".") and name.endswith(_SIGNAL_SUFFIX):
                stems.add(name[: -len(_SIGNAL_SUFFIX)])
    return stems


def stage4_signal_mtimes(
    output_dir: Path, dataset: str, stems: Iterable[str]
) -> dict[str, dict[str, int]]:
    """``{stem: {key: st_mtime_ns}}`` of each of *stems*' signal files, under any key.

    The SLURM controller's progress probe (#282 P5 T1): one walk of
    *dataset*'s key directories, then **one** ``stat`` per stem and key
    directory. No file is opened and no Parquet footer is read -- at 33k
    images and three keys that would be about 100k opens per transition.
    Validity against the live root is not checked here; the Stage-4 worker
    deletes a stale signal before re-embedding its key, so a rewritten
    signal shows as a new mtime. A stem with no signal maps to ``{}``.
    """
    key_dirs = _key_dirs(output_dir, dataset)
    found: dict[str, dict[str, int]] = {}
    for stem in stems:
        mtimes: dict[str, int] = {}
        for region in key_dirs:
            path = region / f"{stem}{_SIGNAL_SUFFIX}"
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            mtimes[_key_of_signal_path(path)] = stat.st_mtime_ns
        found[stem] = mtimes
    return found
