"""The run's embedding deliverables (#281 spec E5, D6a, D7, D9).

``finalize_run`` aggregates each image's embedding tables, as it aggregates
its measurement table, from **the run folder the image's record certifies**
(spec D6a), into one file per entry key::

    deliverables/embeddings/<model>/<layer>/<region>.parquet
    deliverables/embeddings/manifest.json

where ``<layer>`` is ``rgb`` or ``gray`` and ``<region>`` is ``objects`` or
``grid``, either optionally ``-<name>``: the entry key
``<model>/<layer>/<region>``
(:func:`~phenotypic.sdk_._table_runs.split_embedding_key`).

Each file holds the master's intrinsic identity (``Metadata_Dataset``,
``Metadata_ImageName``) plus the table's own key columns,
``descriptor_index`` and ``embedding``, so a row joins to the master on
``(Metadata_Dataset, Metadata_ImageName, Object_Label)``. These files are
Parquet only: never joined to metadata, never in the master, the mirror, a
split or any TSV (spec E5).

**One key, several descriptors** (review I2, user decision Q1). A key's
stores may disagree on the entry descriptor -- the entry minus ``binds``,
``files`` and ``stale``, and minus the model's ``device`` and
``device_name``: another resolved revision (``revision=None`` resolves the
hub's latest per process), another pooling, layer or producer parameter set
(a partly re-measured tree). The device is provenance, not part of the
descriptor (#281, user decision 2026-11-19): a CPU and a GPU embedding
differ by about 1e-6 in cosine distance and two GPU models by about one
float16 bit, so a run that ran on both publishes them as one descriptor,
with no warning; each store's own entry still records its device.
Finalization still publishes,
logs one warning per such key naming each descriptor with its store count
and a few stores, and tags every row with ``descriptor_index``, an
``int16`` index into the key's manifest ``descriptors`` list. That list is
in order of first appearance over the sorted store paths, so it and the
indices are byte-stable. The column is there with a single descriptor too
(``0`` everywhere), so the schema never depends on the data.

**Two phases, so a failure changes no deliverable** (review I1).
:func:`stage_embedding_deliverables` runs **before** finalization writes
anything, the master included. It reads every non-stale entry once --
present, inside its store, the bytes its sha256 records, a
``FixedSizeList<float16>[dim]`` table, one schema per key -- and streams each
key into a hidden temporary file beside its target, one
``pq.ParquetWriter`` per key and one row group per store, so memory holds one
store's table at a time (about 5 GB of float16 over a 33k-store run
otherwise). Any failure deletes every temporary and raises.
:meth:`StagedEmbeddingDeliverables.promote` runs after the master and the
mirror: each staged file replaces its target, then ``manifest.json`` is
written **last**, then the files a previous finalization wrote and this one
does not are removed. :func:`aggregate_embedding_tables` is the two in one.

**Two failure rules, and why they differ** (spec D7, D9, E5):

- A **stale** entry -- its ``stale`` is anything but ``False``
  (:func:`~phenotypic.sdk_._embedding_tables.embedding_entry_is_stale`) -- is
  skipped with a warning naming the store, and counted. It describes nothing
  the run certifies, so leaving it out is the correct output rather than a
  degraded one. No forward CLI path brings one to finalization (a write that
  changes the objmap rewrites the root, which voids the image's record), so
  this is a defence, not a state users see.
- An entry that is **not stale but cannot be read** fails finalization, as a
  master that cannot be built does. Every such store is named in the one
  error, not only the first. So does an entry whose key is not
  ``"<model>/<layer>/<region>"`` (a store root edited by hand), stale or not:
  the key names the deliverable's file, and it is the rule an
  ``EmbeddingPayload`` is refused on when built
  (:func:`~phenotypic.sdk_._table_runs.split_embedding_key`). It never revokes measurement authority: the
  image's record still verifies and ``resolve_run_state`` still reads it
  ``verified`` (D9), so the remedy is a same-pipeline ``--mode measure``
  over the whole run, which re-embeds (and re-measures) every image without
  detecting again (user decision Q2: there is no scoped remedy).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pyarrow as pa  # type: ignore[import-untyped]

    from phenotypic.sdk_ import CommitGuard, TableRun

logger = logging.getLogger(__name__)

#: ``deliverables/embeddings/manifest.json``'s own format version.
EMBEDDINGS_MANIFEST_SCHEMA_VERSION: Final[int] = 1

#: The aggregate proof's ``required_outputs`` name of ``manifest.json``; each
#: published table is ``embeddings/<model>/<layer>/<region>`` (review M1a).
EMBEDDINGS_MANIFEST_PROOF_NAME: Final[str] = "embeddings_manifest"

#: Entry fields that describe one store's file rather than the table's kind;
#: the manifest's ``descriptors`` leave them out (spec D3).
_PER_STORE_FIELDS: Final[frozenset[str]] = frozenset({"binds", "files", "stale"})

#: ``model`` fields that record where an entry ran, not what it is: provenance
#: (#281, user decision 2026-11-19). The manifest's ``descriptors``, the
#: grouping and ``descriptor_index`` leave them out.
_PROVENANCE_MODEL_FIELDS: Final[frozenset[str]] = frozenset({"device", "device_name"})

#: The largest ``descriptor_index`` an ``int16`` holds.
_MAX_DESCRIPTOR_INDEX: Final[int] = 2**15 - 1

#: How many stores a warning or an error names before summarizing the rest.
_EXAMPLE_STORES: Final[int] = 3
_NAMED_UNREADABLE: Final[int] = 10


def _entries(run: "TableRun") -> Mapping[str, object]:
    """*run*'s ``embeddings`` block, or an empty mapping."""
    from phenotypic.sdk_._table_runs import EMBEDDINGS_KEY

    block = run.descriptor.get(EMBEDDINGS_KEY)
    return block if isinstance(block, Mapping) else {}


def _relative_to_output(output_dir: Path, store: Path) -> Path:
    """*store* relative to *output_dir*, on the **unresolved** paths (review I3).

    Never through ``Path.resolve``: a ``results/`` that is a symlink to
    another disk resolves outside the output, though the store is the run's.
    The resolved output root is tried as well, because a record's artifact
    path is joined onto it.
    """
    absolute = Path(os.path.abspath(store))
    for base in (Path(os.path.abspath(output_dir)), Path(output_dir).resolve()):
        try:
            return absolute.relative_to(base)
        except ValueError:
            continue
    raise ValueError(f"{store} is not under the output {output_dir}")


def _store_identity(output_dir: Path, store: Path) -> tuple[str, str, str]:
    """``(store relative to output_dir, dataset, image name)`` of a run's store.

    The image name is the store's stem, the name the master gives a row read
    from an embedded table (``add_metadata_image_name_from_filename``).

    Raises:
        ValueError: If *store* is not ``results/<ds>/zarr/<stem>.ome.zarr``
            under *output_dir*.
    """
    from phenotypic.sdk_ import store_stem, zarr_store_path

    relative = _relative_to_output(output_dir, store)
    stem = store_stem(Path(store))
    dataset = relative.parts[1] if len(relative.parts) == 4 else ""
    expected = zarr_store_path(output_dir, dataset, stem) if dataset else None
    if (
        expected is None
        or _relative_to_output(output_dir, expected) != relative
        or not os.path.samefile(expected, store)
    ):
        raise ValueError(
            f"{store} is not a store of this run: expected "
            "results/<dataset>/zarr/<stem>.ome.zarr under the output"
        )
    return relative.as_posix(), dataset, stem


def _with_identity(
    table: "pa.Table", dataset: str, image: str, descriptor_index: int
) -> "pa.Table":
    """Prepend ``Metadata_Dataset`` and ``Metadata_ImageName`` to *table*, and
    insert ``descriptor_index`` just before ``embedding``."""
    import pyarrow as pa  # type: ignore[import-untyped]

    from phenotypic.schema import EXPERIMENT, IMAGE
    from phenotypic.sdk_._embedding_tables import DESCRIPTOR_INDEX_COLUMN, EMBEDDING_COLUMN

    table = table.replace_schema_metadata(None)
    rows = table.num_rows
    table = table.add_column(
        table.schema.get_field_index(EMBEDDING_COLUMN),
        pa.field(DESCRIPTOR_INDEX_COLUMN, pa.int16(), nullable=False),
        pa.array([descriptor_index] * rows, type=pa.int16()),
    )
    for name, value in ((str(IMAGE.IMAGE_NAME), image), (str(EXPERIMENT.DATASET), dataset)):
        table = table.add_column(
            0,
            pa.field(name, pa.string(), nullable=False),
            pa.array([value] * rows, type=pa.string()),
        )
    return table


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class _Unreadable:
    """One non-stale entry staging could not read."""

    relative: str
    key: str
    reason: str


def _unreadable_error(output_dir: Path, failures: Sequence[_Unreadable]) -> RuntimeError:
    """The finalization error naming every unreadable store and the remedy (Q2)."""
    named = "; ".join(
        f"{failure.relative} ({failure.key}): {failure.reason}"
        for failure in failures[:_NAMED_UNREADABLE]
    )
    if len(failures) > _NAMED_UNREADABLE:
        named += f"; and {len(failures) - _NAMED_UNREADABLE} more"
    return RuntimeError(
        f"{len(failures)} embedding table(s) cannot be read: {named}. "
        "Measurement authority is unaffected (embeddings are additive), but "
        "finalization cannot publish deliverables/embeddings/ without them, "
        "so it published nothing. To rewrite them, re-run the same pipeline "
        f"with --mode measure --output {output_dir}: that re-embeds and "
        "re-measures every image of the run (it never detects again), then "
        "finalizes."
    )


def _target(output_dir: Path, key: str) -> Path:
    """``deliverables/embeddings/<model>/<layer>/<region>.parquet`` for entry *key*."""
    from phenotypic.sdk_ import embeddings_dir
    from phenotypic.sdk_._table_runs import split_embedding_key

    model, layer, region = split_embedding_key(key)
    return embeddings_dir(output_dir) / model / layer / f"{region}.parquet"


@dataclass
class _KeyPlan:
    """What one entry key will publish: its readable runs and its stale stores."""

    key: str
    readable: list["TableRun"] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)
    #: canonical JSON -> each distinct shared descriptor (no per-store fields),
    #: in order of first appearance over the sorted stores: the manifest's
    #: ``descriptors`` order, which ``descriptor_index`` indexes.
    descriptors: dict[str, Mapping[str, object]] = field(default_factory=dict)
    #: store relative path -> its canonical shared descriptor.
    descriptor_of: dict[str, str] = field(default_factory=dict)

    def descriptor_index(self, relative: str) -> int:
        """The position of *relative*'s descriptor in :attr:`descriptors`."""
        return list(self.descriptors).index(self.descriptor_of[relative])


def _plan_keys(
    runs: Sequence["TableRun"], identities: Mapping[Path, tuple[str, str, str]]
) -> tuple[list[_KeyPlan], list[_Unreadable]]:
    """Split each key's entries into readable runs and stale stores, in run order.

    A key that is not ``"<model>/<layer>/<region>"``
    (:func:`~phenotypic.sdk_._table_runs.split_embedding_key`, the rule an
    ``EmbeddingPayload`` is refused on when built) gets no plan: it names the
    deliverable's file, so each of its entries, stale or not, is returned as
    unreadable instead.

    Returns:
        ``(plans, unreadable)``: the plans, sorted by key, and the entries
        whose key is refused, in run order.
    """
    from phenotypic.sdk_._embedding_tables import embedding_entry_is_stale
    from phenotypic.sdk_._table_runs import split_embedding_key

    plans: dict[str, _KeyPlan] = {}
    unreadable: list[_Unreadable] = []
    for run in runs:
        for raw_key, entry in _entries(run).items():
            key = str(raw_key)
            relative = identities[run.store][0]
            try:
                split_embedding_key(key)
            except ValueError as exc:
                unreadable.append(
                    _Unreadable(relative, key, f"its key is not <model>/<layer>/<region>: {exc}")
                )
                continue
            plan = plans.setdefault(key, _KeyPlan(key))
            if not isinstance(entry, Mapping):
                # Not stale, but not readable either: staging names it unreadable.
                plan.readable.append(run)
                continue
            if embedding_entry_is_stale(entry):
                logger.warning(
                    "Skipping the stale embedding table %r of %s: the store no "
                    "longer holds the objmap or array its vectors were computed "
                    "from (spec D7). Re-measure the image to refresh it.",
                    key,
                    relative,
                )
                plan.stale.append(relative)
                continue
            plan.readable.append(run)
            shared = _shared_descriptor(entry)
            canonical = _canonical(shared)
            plan.descriptors.setdefault(canonical, shared)
            plan.descriptor_of[relative] = canonical
    return [plans[key] for key in sorted(plans)], unreadable


def _shared_descriptor(entry: Mapping[str, object]) -> dict[str, object]:
    """*entry* as its key's stores share it: no per-store fields, no device.

    The entry minus :data:`_PER_STORE_FIELDS`, and its ``model`` minus
    :data:`_PROVENANCE_MODEL_FIELDS`, so stores that differ only in the
    device they ran on share one descriptor.
    """
    shared = {n: v for n, v in entry.items() if n not in _PER_STORE_FIELDS}
    model = shared.get("model")
    if isinstance(model, Mapping):
        shared["model"] = {
            n: v for n, v in model.items() if n not in _PROVENANCE_MODEL_FIELDS
        }
    return shared


def _short(value: object) -> str:
    """The first 8 characters of a digest or revision, or ``none``."""
    return str(value)[:8] if value else "none"


def _descriptor_summary(descriptor: Mapping[str, object]) -> str:
    """``revision <sha8>, pooling <p>, params <sha8>``: one descriptor, briefly."""
    model = descriptor.get("model")
    producer = descriptor.get("producer")
    revision = model.get("revision") if isinstance(model, Mapping) else None
    params = producer.get("params_sha256") if isinstance(producer, Mapping) else None
    return (
        f"revision {_short(revision)}, pooling {descriptor.get('pooling')}, "
        f"params {_short(params)}"
    )


def _warn_on_mixed_descriptors(plan: _KeyPlan) -> None:
    """Log one warning when *plan*'s stores carry more than one descriptor (Q1).

    Names each descriptor by its manifest index, briefly (revision, pooling,
    producer parameter digest; plus a digest of the whole descriptor when
    two would otherwise read alike), with its store count and the first few
    stores. Publication goes on: every row's ``descriptor_index`` says which
    descriptor its vector came from.

    Raises:
        RuntimeError: If the key has more descriptors than an ``int16``
            index can name.
    """
    if len(plan.descriptors) <= 1:
        return
    if len(plan.descriptors) > _MAX_DESCRIPTOR_INDEX + 1:
        raise RuntimeError(
            f"embedding key {plan.key!r} has {len(plan.descriptors)} distinct "
            "descriptors, more than descriptor_index (int16) can name"
        )
    stores: dict[str, list[str]] = {canonical: [] for canonical in plan.descriptors}
    for relative, canonical in plan.descriptor_of.items():
        stores[canonical].append(relative)
    summaries = [_descriptor_summary(d) for d in plan.descriptors.values()]
    if len(set(summaries)) < len(summaries):
        summaries = [
            f"{summary}, descriptor {hashlib.sha256(canonical.encode()).hexdigest()[:8]}"
            for summary, canonical in zip(summaries, plan.descriptors)
        ]
    parts = []
    for index, (canonical, summary) in enumerate(zip(plan.descriptors, summaries)):
        held = sorted(stores[canonical])
        examples = ", ".join(held[:_EXAMPLE_STORES]) + (
            ", ..." if len(held) > _EXAMPLE_STORES else ""
        )
        noun = "store" if len(held) == 1 else "stores"
        parts.append(f"[{index}] {summary}: {len(held)} {noun} ({examples})")
    logger.warning(
        "Embedding key %r mixes %d entry descriptors across its stores, so its "
        "vectors are not all comparable. Publishing all of them; each row's "
        "descriptor_index names its descriptor in manifest.json: %s. Pin "
        "the embedder's revision to a commit, and re-measure the whole run "
        "with one pipeline, for one descriptor per run.",
        plan.key,
        len(plan.descriptors),
        "; ".join(parts),
    )


@dataclass
class StagedEmbeddingDeliverables:
    """Embedding deliverables validated and written to temporaries, not yet published.

    Attributes:
        output_dir: Run output root.
        counts: ``{key: {"rows", "stale_skipped"}}``, sorted by key; what
            ``build_run_metrics(embeddings=)`` takes. Empty when no run embeds.
        staged: Each published target and the temporary that will replace it.
        manifest: The ``manifest.json`` document, or ``None`` when no run
            embeds.
    """

    output_dir: Path
    counts: dict[str, dict[str, int]]
    staged: dict[Path, Path]
    manifest: dict[str, object] | None
    _done: bool = False

    def promote(self, *, commit_guard: "CommitGuard | None" = None) -> None:
        """Publish: each staged file over its target, ``manifest.json`` last, then prune.

        Files under ``deliverables/embeddings/`` that a previous finalization
        wrote (``*.parquet`` and ``manifest.json``) and this one does not are
        deleted, and directories left empty are removed. Any other file there
        is left alone (review M8).
        """
        from phenotypic.sdk_ import embeddings_manifest_path
        from phenotypic.sdk_._atomic_io import publication_commit
        from phenotypic.sdk_._run_metrics import write_canonical_json

        if self._done:
            raise RuntimeError("embedding deliverables already promoted or discarded")
        published: set[Path] = set()
        try:
            for target, temp in sorted(self.staged.items()):
                with open(temp, "r+b") as handle:
                    os.fsync(handle.fileno())
                with publication_commit(commit_guard):
                    os.replace(temp, target)
                published.add(target)
            if self.manifest is not None:
                manifest_path = embeddings_manifest_path(self.output_dir)
                write_canonical_json(manifest_path, self.manifest, commit_guard=commit_guard)
                published.add(manifest_path)
            _remove_unpublished(self.output_dir, published, commit_guard=commit_guard)
        finally:
            self.discard()

    def discard(self) -> None:
        """Delete every temporary not yet promoted, and any directory left empty."""
        if self._done:
            return
        self._done = True
        for temp in self.staged.values():
            try:
                os.unlink(temp)
            except FileNotFoundError:
                pass
        _prune_empty_directories(self.output_dir)


def _prune_empty_directories(output_dir: Path) -> None:
    from phenotypic.sdk_ import embeddings_dir

    root = embeddings_dir(output_dir)
    if not root.is_dir():
        return
    for directory in sorted(
        (p for p in root.rglob("*") if p.is_dir()), key=lambda p: -len(p.parts)
    ):
        if not any(directory.iterdir()):
            directory.rmdir()
    if not any(root.iterdir()):
        root.rmdir()


def _remove_unpublished(
    output_dir: Path, published: set[Path], *, commit_guard: "CommitGuard | None"
) -> None:
    """Delete the finalization-written files this finalization did not publish."""
    from phenotypic.sdk_ import embeddings_dir, embeddings_manifest_path
    from phenotypic.sdk_._atomic_io import publication_commit

    root = embeddings_dir(output_dir)
    if not root.is_dir():
        return
    # Finalization writes only <model>/<layer>/<region>.parquet, so a user's
    # file anywhere else under embeddings/ is never ours.
    ours = [*root.glob("*/*/*.parquet"), embeddings_manifest_path(output_dir)]
    with publication_commit(commit_guard):
        for path in sorted(ours):
            if path not in published and path.is_file():
                path.unlink()
        _prune_empty_directories(output_dir)


def _stage_key(
    output_dir: Path,
    plan: _KeyPlan,
    identities: Mapping[Path, tuple[str, str, str]],
    unreadable: list[_Unreadable],
) -> tuple[Path, Path, int]:
    """Stream *plan*'s readable tables into a temporary beside the key's target.

    An entry that cannot be read is appended to *unreadable* and skipped, so
    the caller can name every such store at once and then publish nothing.

    Returns:
        ``(target, temporary, rows)``.

    Raises:
        RuntimeError: If two stores' tables differ in schema. The temporary
            is removed first.
    """
    import tempfile

    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    from phenotypic.sdk_._atomic_io import PARQUET_WRITE_OPTIONS
    from phenotypic.sdk_._embedding_tables import (
        EmbeddingTableUnreadableError,
        read_embedding_entry,
    )

    target = _target(output_dir, plan.key)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
    )
    temp = Path(handle.name)
    handle.close()
    rows = 0
    writer: pq.ParquetWriter | None = None
    try:
        for run in plan.readable:
            relative, dataset, image = identities[run.store]
            try:
                table = read_embedding_entry(run.store, plan.key, _entries(run).get(plan.key))
            except EmbeddingTableUnreadableError as exc:
                unreadable.append(_Unreadable(relative, plan.key, exc.reason))
                continue
            table = _with_identity(table, dataset, image, plan.descriptor_index(relative))
            if writer is None:
                writer = pq.ParquetWriter(
                    temp,
                    table.schema,
                    compression=PARQUET_WRITE_OPTIONS["compression"],
                    compression_level=PARQUET_WRITE_OPTIONS["compression_level"],
                )
            elif not table.schema.equals(writer.schema, check_metadata=False):
                raise RuntimeError(
                    f"embedding table {plan.key!r} of {relative} has schema "
                    f"{table.schema}, unlike the other stores' {writer.schema}; "
                    "one deliverable cannot hold both, so finalization "
                    "published nothing"
                )
            # One row group per store: memory is one store's table.
            writer.write_table(table, row_group_size=max(table.num_rows, 1))
            rows += table.num_rows
        if writer is not None:
            writer.close()
            writer = None
    except BaseException:
        if writer is not None:
            writer.close()
        temp.unlink(missing_ok=True)
        raise
    return target, temp, rows


def stage_embedding_deliverables(
    output_dir: Path, runs: Sequence["TableRun"]
) -> StagedEmbeddingDeliverables:
    """Validate and stage ``deliverables/embeddings/`` from each image's certified run.

    Phase one of :func:`aggregate_embedding_tables`; ``finalize_run`` calls
    it **before** it writes the master. Nothing under ``deliverables/`` is
    replaced or removed here: every key goes to a hidden temporary beside
    its target, and on any failure every temporary is deleted before the
    error propagates.

    For each entry key any run declares, the rows of every readable run's
    table are streamed, store by store in store-path order, with
    ``Metadata_Dataset`` and ``Metadata_ImageName`` prepended and
    ``descriptor_index`` before ``embedding``. The manifest records, per key,
    the file's path, row count and sha256, how many stores it holds, the
    stores skipped as stale, and the distinct entry descriptors (each store's
    ``binds``, ``files`` and ``stale`` left out) in order of first
    appearance, the order ``descriptor_index`` indexes. A key with more than
    one descriptor is published with a warning (review Q1). No wall-clock
    value is written, so identical runs give identical bytes (spec C11).

    Args:
        output_dir: Run output root.
        runs: Each image's certified run folder (spec D6a): the run its record
            certifies, **never** one chosen by pipeline sha. A run without
            embeddings contributes nothing.

    Returns:
        The staged deliverables; publish them with
        :meth:`StagedEmbeddingDeliverables.promote` or drop them with
        :meth:`~StagedEmbeddingDeliverables.discard`.

    Raises:
        RuntimeError: If any non-stale entry cannot be read, or any entry's
            key is not ``"<model>/<layer>/<region>"`` (its message names every
            such store and the whole-run ``--mode measure`` remedy), or two
            stores' tables for one key have different schemas.
        ValueError: If a run's store is not a store of this run.
    """
    from phenotypic.sdk_ import deliverables_dir
    from phenotypic.sdk_._embedding_tables import _file_sha256

    output_dir = Path(output_dir)
    with_embeddings = [run for run in runs if _entries(run)]
    identities = {run.store: _store_identity(output_dir, run.store) for run in with_embeddings}
    ordered = sorted(with_embeddings, key=lambda run: identities[run.store][0])
    # Entries under a refused key are unreadable from the start; the other
    # keys are still staged, so the one error names every unreadable store.
    plans, unreadable = _plan_keys(ordered, identities)
    for plan in plans:
        _warn_on_mixed_descriptors(plan)

    staged = StagedEmbeddingDeliverables(output_dir, {}, {}, None)
    entries: dict[str, object] = {}
    try:
        for plan in plans:
            # A key whose every entry is stale publishes no file.
            path: str | None = None
            sha256: str | None = None
            rows = 0
            if plan.readable:
                target, temp, rows = _stage_key(output_dir, plan, identities, unreadable)
                staged.staged[target] = temp
                path = target.relative_to(deliverables_dir(output_dir)).as_posix()
                sha256 = _file_sha256(temp)
            staged.counts[plan.key] = {"rows": rows, "stale_skipped": len(plan.stale)}
            entries[plan.key] = {
                "path": path,
                "rows": rows,
                "sha256": sha256,
                "stores": len(plan.readable),
                "stale_skipped": sorted(plan.stale),
                "descriptors": list(plan.descriptors.values()),
            }
        if unreadable:
            raise _unreadable_error(output_dir, unreadable)
    except BaseException:
        staged.discard()
        raise
    if plans:
        staged.manifest = {
            "schema_version": EMBEDDINGS_MANIFEST_SCHEMA_VERSION,
            "entries": entries,
        }
    return staged


def aggregate_embedding_tables(
    output_dir: Path,
    runs: Sequence["TableRun"],
    *,
    commit_guard: "CommitGuard | None" = None,
) -> dict[str, dict[str, int]]:
    """Stage and publish ``deliverables/embeddings/`` in one call (spec E5).

    :func:`stage_embedding_deliverables`, then
    :meth:`StagedEmbeddingDeliverables.promote`. ``finalize_run`` calls the
    two phases separately, so that the validation runs before the master is
    written; this is the one-call form for anything else.

    Returns:
        ``{key: {"rows": n, "stale_skipped": k}}``, sorted by key. Empty when
        no run has embeddings.

    Raises:
        RuntimeError, ValueError: As :func:`stage_embedding_deliverables`; no
            deliverable has changed when either propagates.
    """
    staged = stage_embedding_deliverables(output_dir, runs)
    staged.promote(commit_guard=commit_guard)
    return staged.counts


def published_embedding_files(output_dir: Path) -> dict[str, Path]:
    """The embedding deliverables a finalization published, for the aggregate proof.

    ``manifest.json`` under :data:`EMBEDDINGS_MANIFEST_PROOF_NAME`, and each
    table the manifest lists under ``embeddings/<model>/<layer>/<region>`` (review
    M1a). Empty when there is no manifest: a run that embeds nothing
    declares nothing. No GUI writer touches these files, so the proof
    enforces them.

    Raises:
        ValueError: If the manifest is not a manifest this build writes.
    """
    from phenotypic.sdk_ import deliverables_dir, embeddings_manifest_path

    manifest_path = embeddings_manifest_path(output_dir)
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries") if isinstance(manifest, dict) else None
    if not isinstance(entries, dict):
        raise ValueError(f"{manifest_path} lists no entries")
    files = {EMBEDDINGS_MANIFEST_PROOF_NAME: manifest_path}
    for key, entry in sorted(entries.items()):
        path = entry.get("path") if isinstance(entry, dict) else None
        if isinstance(path, str):
            files[f"embeddings/{key}"] = deliverables_dir(output_dir) / path
    return files


def embedding_table_schemas(output_dir: Path) -> dict[str, dict[str, object]]:
    """Describe each published embedding deliverable for ``schema.json`` (spec C12).

    Read back from what :meth:`StagedEmbeddingDeliverables.promote` just
    published: the manifest names the files and their producer, and each
    file's footer gives its columns. Key and identity columns are rendered
    as polars dtypes, as every other ``schema.json`` table is.

    Returns:
        ``{"embeddings/<model>/<layer>/<region>": {"columns", "dim", "model",
        "producer"}}``, the shape
        :func:`~phenotypic.util._measurement_outputs.build_output_schema`
        takes as ``embedding_tables``. Empty when nothing is published.
    """
    import polars as pl
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    from phenotypic.sdk_ import deliverables_dir, embeddings_manifest_path
    from phenotypic.sdk_._embedding_tables import EMBEDDING_COLUMN

    manifest_path = embeddings_manifest_path(output_dir)
    if not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tables: dict[str, dict[str, object]] = {}
    for key, entry in sorted(manifest.get("entries", {}).items()):
        if entry.get("path") is None or not entry.get("descriptors"):
            continue
        path = deliverables_dir(output_dir) / entry["path"]
        schema = pq.read_schema(path)
        frame = pl.from_arrow(schema.empty_table().drop_columns([EMBEDDING_COLUMN]))
        assert isinstance(frame, pl.DataFrame)
        keys = frame.schema
        columns: dict[str, object] = {}
        for name in schema.names:
            columns[name] = EMBEDDING_COLUMN if name == EMBEDDING_COLUMN else keys[name]
        descriptor = entry["descriptors"][0]
        tables[f"embeddings/{key}"] = {
            "columns": columns,
            "dim": schema.field(EMBEDDING_COLUMN).type.list_size,
            "model": key.split("/")[0],
            "producer": descriptor["producer"]["class"],
        }
    return tables


__all__ = [
    "EMBEDDINGS_MANIFEST_PROOF_NAME",
    "EMBEDDINGS_MANIFEST_SCHEMA_VERSION",
    "StagedEmbeddingDeliverables",
    "aggregate_embedding_tables",
    "embedding_table_schemas",
    "published_embedding_files",
    "stage_embedding_deliverables",
]
