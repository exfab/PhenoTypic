"""The run summary and column schema a DVC stage reads (#281 spec C4, C12).

``deliverables/metrics.json`` and ``deliverables/schema.json`` are
**deliverables, not tracked state**: ``finalize_run`` writes them from what it
already holds, and nothing in PhenoTypic reads either back. Both are
byte-stable -- no wall-clock or host value, sorted keys, stable list order --
so identical runs give identical bytes and ``dvc metrics diff`` is meaningful.

This module holds the pure builders and the one canonical JSON writer. The
counts are gathered by the writer in ``_cli`` (INV-LAYER: ``sdk_`` never
imports ``_cli``); the column schema is built by
:func:`phenotypic.util._measurement_outputs.build_output_schema`, beside the
registry the deliverables README reads.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Final

from ._atomic_io import CommitGuard, atomic_write_json

#: ``metrics.json``'s own format version. Bumped when a key changes meaning or
#: is removed; adding the optional ``embeddings`` block (P4) is not a bump.
RUN_METRICS_SCHEMA_VERSION: Final[int] = 1


def shared_pipeline_sha256(digests: Iterable[str | None]) -> str | None:
    """Return the pipeline digest every aggregated table shares, else ``None``.

    Spec C4: ``metrics.json``'s ``pipeline_sha256`` is the digest of the
    pipeline that produced **every** table the finalizer aggregated, and
    ``null`` when they differ. It is never the run config's digest, which
    still names the full run's pipeline after ``--mode measure`` (D6a).

    An unknown digest (``None``) is never treated as agreeing with a known
    one, and no digests at all is ``None``: a value is reported only when it
    is known for every table.

    Args:
        digests: One pipeline digest per aggregated table, ``None`` where the
            table's producing pipeline is not recorded.

    Returns:
        The single shared digest, or ``None``.

    Examples:
        >>> from phenotypic.sdk_._run_metrics import shared_pipeline_sha256
        >>> shared_pipeline_sha256(["a" * 64, "a" * 64]) == "a" * 64
        True
        >>> shared_pipeline_sha256(["a" * 64, "b" * 64]) is None
        True
        >>> shared_pipeline_sha256(["a" * 64, None]) is None
        True
        >>> shared_pipeline_sha256([]) is None
        True
    """
    distinct = set(digests)
    if len(distinct) != 1:
        return None
    (only,) = distinct
    return only


def build_run_metrics(
    *,
    phenotypic_version: str,
    pipeline_sha256: str | None,
    accepted: int,
    succeeded: int,
    failed: int,
    measured_objects: int,
    metadata_phantoms: int,
    embeddings: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    """Build the ``metrics.json`` payload.

    Args:
        phenotypic_version: The installed PhenoTypic version.
        pipeline_sha256: The digest every aggregated table shares
            (:func:`shared_pipeline_sha256`), or ``None``.
        accepted: Images the run accepted (its inventory).
        succeeded: Images whose measurements the finalizer aggregated.
        failed: Accepted images with a terminal failure and no success.
        measured_objects: Rows of the master: every measured object.
        metadata_phantoms: Mirror rows for a metadata identity that matched
            no measured object (``QC_MetadataOnly`` true).
        embeddings: Per embedding entry id, ``{"rows", "stale_skipped"}``.
            Filled by the embedding deliverables (P4); the key is absent
            when it is ``None``.

    Returns:
        The payload, ready for :func:`write_canonical_json`.

    Examples:
        >>> from phenotypic.sdk_._run_metrics import build_run_metrics
        >>> m = build_run_metrics(phenotypic_version="0.21.0", pipeline_sha256=None,
        ...                       accepted=3, succeeded=2, failed=1,
        ...                       measured_objects=180, metadata_phantoms=0)
        >>> m["images"]
        {'accepted': 3, 'succeeded': 2, 'failed': 1}
        >>> "embeddings" in m
        False
    """
    metrics: dict[str, Any] = {
        "schema_version": RUN_METRICS_SCHEMA_VERSION,
        "phenotypic_version": phenotypic_version,
        "pipeline_sha256": pipeline_sha256,
        "images": {
            "accepted": int(accepted),
            "succeeded": int(succeeded),
            "failed": int(failed),
        },
        "objects": {"measured": int(measured_objects)},
        "metadata": {"phantoms": int(metadata_phantoms)},
    }
    if embeddings is not None:
        metrics["embeddings"] = {
            str(entry): {key: int(value) for key, value in counts.items()}
            for entry, counts in embeddings.items()
        }
    return metrics


def write_canonical_json(
    path: Path,
    obj: Mapping[str, Any] | list[Any],
    *,
    commit_guard: CommitGuard | None = None,
) -> None:
    """Atomically write *obj* as canonical JSON: the byte-stable form.

    ``json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False) + "\\n"``,
    UTF-8, published by rename. The formatting is spelled out rather than
    inherited from :func:`atomic_write_json`'s defaults, because byte
    stability is this writer's contract and a default can change.

    Args:
        path: Destination file.
        obj: A JSON-serializable mapping or list.
        commit_guard: Publication guard threaded to the write.
    """
    atomic_write_json(
        path,
        obj,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        commit_guard=commit_guard,
    )


__all__ = [
    "RUN_METRICS_SCHEMA_VERSION",
    "build_run_metrics",
    "shared_pipeline_sha256",
    "write_canonical_json",
]
