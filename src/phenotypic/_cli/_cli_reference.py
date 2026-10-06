"""Reference metadata for CLI runs: plan once at startup, read in every worker.

The main process resolves every input image's references (the run preflight
calls the same planner without hashing). Startup writes the result to
``.phenotypic/reference_manifest.json``; each worker core enters
:func:`worker_reference_context` around its apply call, so no worker needs a
new argument and SLURM workers need nothing but the run root.

This module imports only the standard library at module level.
"""

from __future__ import annotations

import functools
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Sequence

if TYPE_CHECKING:  # pragma: no cover
    from phenotypic import ImagePipeline, ReferenceContext

    from ._cli_preflight import RunMode
    from ._cli_types import Dataset, ExecutionConfig

MANIFEST_SCHEMA_VERSION = 1
#: The digest of an image the manifest does not know (it arrived after
#: startup). An image whose planning *failed* gets ``"unplanned:<reason>"``
#: instead (:func:`_unplanned_digest`). Neither is hex, so neither ever equals
#: a planned image's digest.
UNPLANNED_DIGEST = "unplanned"


class ReferencePlanStaleError(RuntimeError):
    """The run's reference plan changed under a worker; the image is not at fault.

    Raised when the table's bytes no longer match the manifest, or when the
    manifest's digest for an image no longer matches the one its work-id was
    computed from (a later invocation re-planned it). Deliberately not a
    ``ReferenceContextError``: every apply site lets it through rather than
    wrapping it as a per-image scientific failure, so no terminal record is
    written and the next run simply re-attempts the image.
    """


@dataclass(frozen=True)
class ReferencePin:
    """The reference digest an image's work-id was computed from.

    Handed to :func:`worker_reference_context` so the context and the work-id
    the result is published under always describe the same plan.

    Attributes:
        image_stem: The image's ``source_image_stem``.
        digest: Its digest at identity time (``None`` without a manifest).
    """

    image_stem: str
    digest: str | None


def _unplanned_digest(reason: str, values: dict[str, str] | None = None) -> str:
    """``"unplanned:<reason>"``, plus the looked-up values when there are any.

    A changed reason, or a changed value for the same reason (a blank renamed
    from one missing file to another), changes the work-id, so a terminal
    failure recorded for the old cause never pins the image (review F4).
    """
    from phenotypic.sdk_._digests import canonical_digest

    marker = f"{UNPLANNED_DIGEST}:{reason}"
    return f"{marker}:{canonical_digest(values)}" if values else marker


@dataclass(frozen=True)
class ReferencePlan:
    """Every input image's reference resolution, classified.

    Attributes:
        total_images: Input images planned, across every dataset.
        unmatched: ``"<dataset>/<stem>"`` labels with no table row.
        ambiguous: Labels whose rows are empty or disagree for a column.
        self_referenced: Labels that name themselves as a reference image.
        unresolved: Labels naming a reference image that matches no single
            file in the dataset's input directory.
        images_by_dataset: ``{dataset: {name as written: absolute path}}``.
        digests: ``{dataset: {stem: digest}}``; empty without hashing.
        unplanned: ``{dataset: {stem: "unplanned:<reason>[:<values digest>]"}}``
            for every image in the four failure lists.
    """

    total_images: int
    unmatched: tuple[str, ...]
    ambiguous: tuple[str, ...]
    self_referenced: tuple[str, ...]
    unresolved: tuple[str, ...]
    images_by_dataset: dict[str, dict[str, str]]
    digests: dict[str, dict[str, str]]
    unplanned: dict[str, dict[str, str]]


def _union(groups: Iterable[tuple[str, ...]]) -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for columns in groups:
        for column in columns:
            seen.setdefault(column, None)
    return tuple(seen)


def plan_references(
    context: "ReferenceContext",
    pipeline: "ImagePipeline",
    datasets: Sequence["Dataset"],
    *,
    hash_images: bool,
    operations: Sequence[Any] | None = None,
) -> ReferencePlan:
    """Resolve every input image's reference values and reference images.

    Args:
        context: The run's reference table (dataset and root are set per dataset).
        pipeline: The run's pipeline; its ``reference_columns()`` decide what
            is looked up and which values are resolved to files.
        datasets: The scanned inputs.
        hash_images: Hash reference-image files for the per-image digests.
            The preflight passes ``False`` (headers-only contract); startup
            passes ``True``.
        operations: The reference-metadata operations the run executes, when
            narrower than the whole pipeline (the preflight passes those
            :func:`~._cli_preflight.operations_in_scope` keeps). ``None``
            plans for every one in *pipeline*.

    Returns:
        The classified plan. An image appears in at most one failure list.

    Raises:
        ReferenceTableError: A planned column is not in the table.
    """
    from phenotypic._core import _reference_context
    from phenotypic._core._reference_context import (
        ReferenceImageError,
        ReferenceLookupError,
    )
    from phenotypic.sdk_._digests import canonical_digest
    from phenotypic.sdk_._io_constants import source_image_stem

    if operations is None:
        value_columns = _union(pipeline.reference_columns().values())
        image_columns = set(
            _union(pipeline.reference_columns(images_only=True).values())
        )
    else:
        value_columns = _union(op._ref_columns() for op in operations)
        image_columns = set(_union(op._ref_image_columns() for op in operations))
    unmatched: list[str] = []
    ambiguous: list[str] = []
    self_referenced: list[str] = []
    unresolved: list[str] = []
    images_by_dataset: dict[str, dict[str, str]] = {}
    digests: dict[str, dict[str, str]] = {}
    unplanned: dict[str, dict[str, str]] = {}
    sha_cache: dict[str, str] = {}
    total = 0
    for dataset in datasets:
        scoped = context.narrow(dataset=dataset.name, image_root=dataset.input_dir)
        resolved = images_by_dataset.setdefault(dataset.name, {})
        dataset_digests = digests.setdefault(dataset.name, {})
        dataset_unplanned = unplanned.setdefault(dataset.name, {})
        for image_path in dataset.images:
            total += 1
            # The name Image.imread gives this input ("x.ome.zarr" -> "x").
            stem = source_image_stem(Path(image_path))
            label = f"{dataset.name}/{stem}"
            try:
                values = scoped.lookup(stem, value_columns)
            except ReferenceLookupError as exc:
                reason = "unmatched" if exc.reason == "unmatched" else "ambiguous"
                (unmatched if reason == "unmatched" else ambiguous).append(label)
                dataset_unplanned[stem] = _unplanned_digest(reason)
                continue
            image_shas: dict[str, str] = {}
            failure: list[str] | None = None
            for column in value_columns:
                if column not in image_columns:
                    continue
                name = values[column]
                if name == stem:
                    failure = self_referenced
                    break
                try:
                    target = scoped.resolve_image(name)
                except ReferenceImageError:
                    failure = unresolved
                    break
                if not isinstance(target, Path):
                    # An in-memory ``images`` entry (never the CLI's): no file
                    # to record or hash.
                    continue
                # Same rule as SubtractBlank: "t04.tif" for image "t04" resolves
                # to the frame's own file and is a self-reference too.
                if source_image_stem(target) == stem:
                    failure = self_referenced
                    break
                key = str(target.resolve())
                resolved[name] = key
                if hash_images:
                    if key not in sha_cache:
                        # Looked up on the module so a test can prove the
                        # preflight path never reaches it.
                        sha_cache[key] = _reference_context.reference_file_digest(
                            Path(key)
                        )
                    image_shas[name] = sha_cache[key]
            if failure is not None:
                failure.append(label)
                reason = "self" if failure is self_referenced else "unresolved"
                dataset_unplanned[stem] = _unplanned_digest(reason, values)
                continue
            if hash_images:
                dataset_digests[stem] = canonical_digest(
                    {"values": values, "images": image_shas}
                )
    return ReferencePlan(
        total_images=total,
        unmatched=tuple(unmatched),
        ambiguous=tuple(ambiguous),
        self_referenced=tuple(self_referenced),
        unresolved=tuple(unresolved),
        images_by_dataset=images_by_dataset,
        digests=digests,
        unplanned=unplanned,
    )


def reference_operations_with_paths(
    pipeline: "ImagePipeline", mode: "RunMode"
) -> list[tuple[tuple[str, ...], Any]]:
    """``(path, operation)`` for each operation a run in *mode* executes that
    reads at least one reference column, in depth-first order."""
    from phenotypic.abc_._ref_metadata import RefMetadata

    from ._cli_preflight import operations_run_in_mode

    return [
        (path, operation)
        for path, operation in operations_run_in_mode(pipeline, mode)
        if isinstance(operation, RefMetadata) and operation._ref_columns()
    ]


def reference_operations_in_scope(
    pipeline: "ImagePipeline", mode: "RunMode"
) -> list[Any]:
    """The reference-metadata operations a run in *mode* executes.

    The preflight's mode walk (:func:`~._cli_preflight.operations_run_in_mode`),
    kept to operations that read at least one column: a ``SubtractBlank``
    inside a measurer never runs in ``process`` mode, so a process run never
    needs the column it reads.
    """
    return [op for _, op in reference_operations_with_paths(pipeline, mode)]


def reference_operation_paths(pipeline: "ImagePipeline", mode: "RunMode") -> list[str]:
    """Tree paths (``/``-joined) of :func:`reference_operations_in_scope`."""
    return ["/".join(path) for path, _ in reference_operations_with_paths(pipeline, mode)]


def publish_reference_inputs(
    config: "ExecutionConfig", datasets: Sequence["Dataset"], output_dir: Path
) -> None:
    """Snapshot the table (process mode) and publish the run's reference manifest.

    Must run before the invocation computes any work-id, because work-ids read
    the manifest's per-image digests. Removes a stale manifest when the
    operations this mode runs read no reference metadata. Measure mode returns
    without touching the manifest: it applies no operation, and may run beside
    live forward workers that read it.

    In process mode the manifest names the snapshot, never the user's file,
    and ``config.metadata_csv`` is left as given: the run identity and the
    processing state digest it exactly as before this feature.

    Raises:
        ReferenceTableError: The pipeline reads reference metadata and the run
            has no table, or the table lacks a planned column.
    """
    from phenotypic import ImagePipeline
    from phenotypic._core._reference_context import ReferenceContext, ReferenceTableError

    from ._cli_preflight import run_mode_of

    if config.measure_only:
        return
    pipeline = ImagePipeline.from_json(config.pipeline_json)
    operations = reference_operations_in_scope(pipeline, run_mode_of(config))
    if not operations:
        remove_reference_manifest(output_dir)
        return
    if config.process_only_layer is not None:
        table_path = snapshot_reference_metadata(output_dir, config.metadata_csv)
    else:
        table_path = resolve_reference_table_path(config, output_dir)
    if table_path is None:
        raise ReferenceTableError(
            "The pipeline reads reference metadata but the run has no table; pass --metadata"
        )
    context = ReferenceContext(table_path)
    plan = plan_references(
        context, pipeline, datasets, hash_images=True, operations=operations
    )
    write_reference_manifest(
        output_dir,
        plan=plan,
        table_path=table_path,
        table_sha256=context.table_sha256 or "",
        read_kwargs=input_read_kwargs(config),
    )


def resolve_reference_table_path(
    config: "ExecutionConfig", output_dir: Path | None
) -> Path | None:
    """The table a run's reference ops read: ``--metadata``, else the run's snapshot.

    Measure mode applies no operation and so reads no table. Full mode falls
    back to ``deliverables/metadata.csv``, process mode to
    ``.phenotypic/reference_metadata.csv``.
    """
    if config.measure_only:
        return None
    if config.metadata_csv is not None:
        return Path(config.metadata_csv)
    if output_dir is None:
        return None
    snapshot = reference_table_snapshot_path(
        output_dir, process_mode=config.process_only_layer is not None
    )
    return snapshot if snapshot.is_file() else None


def reference_table_snapshot_path(output_dir: Path, *, process_mode: bool) -> Path:
    """The snapshot a run without ``--metadata`` falls back to.

    Full mode: ``deliverables/metadata.csv``; process mode:
    ``.phenotypic/reference_metadata.csv``. Whether it exists is the caller's
    question.
    """
    from phenotypic.sdk_._io_constants import (
        metadata_csv_deliverable_path,
        reference_metadata_snapshot_path,
    )

    if process_mode:
        return reference_metadata_snapshot_path(output_dir)
    return metadata_csv_deliverable_path(output_dir)


def input_read_kwargs(config: "ExecutionConfig") -> dict[str, Any]:
    """``Image.imread`` kwargs for reference images: the inputs' reader settings."""
    return {"bit_depth": config.bit_depth} if config.bit_depth else {}


def snapshot_reference_metadata(output_dir: Path, source: Path | None) -> Path | None:
    """Byte-copy *source* to the process-mode snapshot; reuse it when *source* is None.

    Args:
        output_dir: The run's ``--output``.
        source: The ``--metadata`` table, or ``None`` on a continuation.

    Returns:
        The snapshot, or ``None`` when there is no source and no snapshot.

    Raises:
        ValueError: *source* does not parse as CSV (pandas' ``ParserError``
            or ``EmptyDataError``); an existing snapshot is left untouched.
    """
    import io

    import pandas as pd

    from phenotypic.sdk_._atomic_io import atomic_write_bytes
    from phenotypic.sdk_._io_constants import reference_metadata_snapshot_path

    destination = reference_metadata_snapshot_path(output_dir)
    if source is None:
        return destination if destination.is_file() else None
    payload = Path(source).read_bytes()
    # Never replace a valid snapshot with unparseable bytes; pandas refuses an
    # unterminated quote that Polars reads as a header (as in
    # phenotypicCLI._snapshot_metadata_csv).
    pd.read_csv(io.BytesIO(payload))
    if destination.is_file() and destination.read_bytes() == payload:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(destination, payload)
    if destination.read_bytes() != payload:
        raise OSError(f"Reference metadata snapshot verification failed: {destination}")
    return destination


def write_reference_manifest(
    output_dir: Path,
    *,
    plan: ReferencePlan,
    table_path: Path,
    table_sha256: str,
    read_kwargs: dict[str, Any],
) -> Path:
    """Atomically publish the run's reference manifest."""
    from phenotypic.sdk_._atomic_io import atomic_write_json
    from phenotypic.sdk_._io_constants import reference_manifest_path

    path = reference_manifest_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets = {
        name: {
            "images": plan.images_by_dataset.get(name, {}),
            # An image whose planning failed is pinned to why (review F4).
            "digests": {**plan.unplanned.get(name, {}), **plan.digests.get(name, {})},
        }
        for name in sorted(
            set(plan.images_by_dataset) | set(plan.digests) | set(plan.unplanned)
        )
    }
    atomic_write_json(
        path,
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "table": str(Path(table_path).resolve()),
            "table_sha256": table_sha256,
            "read_kwargs": read_kwargs,
            "datasets": datasets,
        },
    )
    _MANIFEST_CACHE.pop(str(path), None)
    return path


#: Parsed manifests keyed by path, stamped with ``(st_ino, st_mtime_ns,
#: st_size)``. A 34,500-image manifest is ~3 MB and parses in ~19 ms (review
#: M2 probe), and work-ids are computed per image in several startup passes
#: and per image in every worker -- re-parsing each time would cost minutes
#: per pass. The inode is in the stamp because every publication is an atomic
#: replace (a new file): a re-plan that changes a blank name to one of the
#: same length keeps the size, and two writes inside one timestamp tick keep
#: the mtime.
_MANIFEST_CACHE: dict[str, tuple[tuple[int, int, int], dict]] = {}


def _parse_manifest(text: str) -> dict:
    return json.loads(text)


def read_reference_manifest(output_dir: Path) -> dict | None:
    """The run's reference manifest, or ``None`` when the run needs none."""
    from phenotypic.sdk_._io_constants import reference_manifest_path

    path = reference_manifest_path(output_dir)
    try:
        stat = os.stat(path)
    except FileNotFoundError:
        _MANIFEST_CACHE.pop(str(path), None)
        return None
    stamp = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
    cached = _MANIFEST_CACHE.get(str(path))
    if cached is None or cached[0] != stamp:
        cached = (stamp, _parse_manifest(path.read_text(encoding="utf-8")))
        _MANIFEST_CACHE[str(path)] = cached
    return cached[1]


def remove_reference_manifest(output_dir: Path) -> None:
    """Delete a stale manifest (the pipeline no longer reads reference metadata)."""
    from phenotypic.sdk_._io_constants import reference_manifest_path

    path = reference_manifest_path(output_dir)
    path.unlink(missing_ok=True)
    _MANIFEST_CACHE.pop(str(path), None)


@functools.lru_cache(maxsize=4)
def _manifest_base_context(
    table: str, table_sha256: str, read_kwargs_json: str
) -> "ReferenceContext":
    """One parsed table per process, refused if its bytes moved since planning."""
    from phenotypic._core._reference_context import ReferenceContext

    context = ReferenceContext(Path(table), read_kwargs=json.loads(read_kwargs_json))
    if context.table_sha256 != table_sha256:
        raise ReferencePlanStaleError(
            f"Reference table {table} changed since this run planned its references; "
            f"run the same command again to re-plan"
        )
    return context


@contextmanager
def worker_reference_context(
    output_dir: Path,
    dataset_name: str | None,
    *,
    pin: ReferencePin | None = None,
) -> Iterator["ReferenceContext | None"]:
    """Activate the run's ReferenceContext for one image of *dataset_name*.

    Yields ``None`` (and activates nothing) when the run has no manifest. The
    context resolves only the reference images the manifest planned for the
    dataset; it has no ``image_root``.

    Args:
        output_dir: The run root.
        dataset_name: The image's dataset.
        pin: The digest the image's work-id was computed from. Checked against
            the same manifest read the context is built from, so a later
            invocation re-planning the image between the worker's identity
            and its apply is refused rather than published under a work-id
            that describes a different plan (review F1). ``None`` checks
            nothing.

    Raises:
        ValueError: The run has a manifest and *dataset_name* is ``None``.
        ReferencePlanStaleError: The table's bytes no longer match the
            manifest, or the manifest's digest for the pinned image changed.
    """
    manifest = read_reference_manifest(output_dir)
    if pin is not None:
        current = (
            None
            if manifest is None or dataset_name is None
            else _digest_in(manifest, dataset_name, pin.image_stem)
        )
        if current != pin.digest:
            raise ReferencePlanStaleError(
                f"The reference plan for {dataset_name}/{pin.image_stem} changed "
                f"after this worker computed its work identity (another invocation "
                f"re-planned the run); run the same command again"
            )
    if manifest is None:
        yield None
        return
    if dataset_name is None:
        raise ValueError("A run with reference metadata needs the image's dataset name")
    base = _manifest_base_context(
        manifest["table"],
        manifest["table_sha256"],
        json.dumps(manifest["read_kwargs"], sort_keys=True),
    )
    images = manifest["datasets"].get(dataset_name, {}).get("images", {})
    with base.narrow(dataset=dataset_name, images=images) as context:
        yield context


def reference_digest_for(
    output_dir: Path | None, dataset_name: str, image_stem: str
) -> str | None:
    """The image's reference digest for its work-id; ``None`` when the run has none.

    An image whose planning failed gets its recorded ``"unplanned:<reason>"``
    marker; one the manifest does not know at all gets
    :data:`UNPLANNED_DIGEST`.
    """
    if output_dir is None:
        return None
    manifest = read_reference_manifest(output_dir)
    if manifest is None:
        return None
    return _digest_in(manifest, dataset_name, image_stem)


def _digest_in(manifest: dict, dataset_name: str, image_stem: str) -> str:
    dataset = manifest["datasets"].get(dataset_name, {})
    return dataset.get("digests", {}).get(image_stem, UNPLANNED_DIGEST)
