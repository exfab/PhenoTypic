"""Runtime registry and dynamic resolver for named analysis tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd
from phenotypic.sdk_._file_locking import exclusive_path_lock

from ._analysis_artifacts import (
    AnalysisArtifactPaths,
    AnalysisManifestEntry,
    AnalysisManifestError,
    named_analysis_paths,
    read_analysis_manifest,
    recover_analysis_publication,
    resolve_manifest_artifact_path,
    validate_analysis_id,
)

_LEGACY_ANALYSIS_PARQUET = "analysis.parquet"


@runtime_checkable
class AnalysisInputLike(Protocol):
    """Structural input accepted by :meth:`AnalysisRegistry.resolve`."""

    @property
    def analysis_id(self) -> str:
        """Return the stable analysis identifier."""
        ...


class AnalysisNotFoundError(KeyError):
    """Raised when a requested analysis is unavailable from every valid source."""


@dataclass(frozen=True)
class AnalysisResult:
    """Current runtime result for one analysis producer and publication."""

    analysis_id: str
    table: pd.DataFrame
    producer: Any = None
    artifacts: AnalysisArtifactPaths | None = None
    manifest_entry: AnalysisManifestEntry | None = None


class AnalysisRegistry:
    """Map stable analysis IDs to current tables and persisted artifacts.

    The registry deliberately does not cache tables loaded from disk. Each persisted
    refresh re-reads the authoritative manifest and verifies the selected Parquet
    checksum. Explicitly registered in-memory results take precedence and preserve
    producer/table identity for same-run plotting.

    Args:
        deliverables_base: Optional default deliverables directory for persisted
            resolution.
    """

    def __init__(self, deliverables_base: Path | None = None) -> None:
        self._deliverables_base = (
            None if deliverables_base is None else Path(deliverables_base)
        )
        self._analyses: dict[str, AnalysisResult] = {}

    def register(
        self,
        analysis_id: str,
        table: pd.DataFrame,
        *,
        producer: Any = None,
        artifacts: AnalysisArtifactPaths | None = None,
        manifest_entry: AnalysisManifestEntry | None = None,
    ) -> AnalysisResult:
        """Register or replace the current in-memory result for an analysis ID.

        Args:
            analysis_id: Stable safe analysis identity.
            table: Current analysis table. The exact object is retained.
            producer: Optional exact producer instance that populated the table.
            artifacts: Optional paths for the persisted generation.
            manifest_entry: Optional authoritative persisted manifest entry.

        Returns:
            The immutable registration record.

        Raises:
            TypeError: If ``table`` is not a pandas DataFrame.
            ValueError: If ``analysis_id`` is unsafe.
        """
        safe_id = validate_analysis_id(analysis_id)
        if not isinstance(table, pd.DataFrame):
            raise TypeError("analysis table must be a pandas DataFrame")
        registration = AnalysisResult(
            analysis_id=safe_id,
            table=table,
            producer=producer,
            artifacts=artifacts,
            manifest_entry=manifest_entry,
        )
        self._analyses[safe_id] = registration
        return registration

    def get(self, analysis_id: str) -> AnalysisResult | None:
        """Return an in-memory registration, or ``None`` when absent."""
        return self._analyses.get(validate_analysis_id(analysis_id))

    def resolve(
        self,
        input_ref: str | AnalysisInputLike,
        *,
        deliverables_base: Path | None = None,
    ) -> pd.DataFrame:
        """Resolve an analysis table in memory first, then through its manifest.

        A legacy ``analysis.parquet`` is consulted only when no manifest exists. A
        present manifest is authoritative, including when it does not contain the
        requested ID.

        Args:
            input_ref: Safe analysis ID or an ``AnalysisInput``-like object exposing
                ``analysis_id``.
            deliverables_base: Per-call persisted bundle override.

        Returns:
            The current pandas analysis table.

        Raises:
            AnalysisNotFoundError: If the ID has no valid source.
            AnalysisManifestError: If the authoritative manifest or artifact is
                malformed, missing, or fails checksum validation.
        """
        analysis_id = _analysis_id_from_input(input_ref)
        registered = self._analyses.get(analysis_id)
        if registered is not None:
            return registered.table

        base = self._resolve_deliverables_base(deliverables_base)
        if base is None:
            raise self._not_found(analysis_id, None)
        with exclusive_path_lock(base / ".analysis-artifacts.lock"):
            # The first publication may not have a manifest yet. Waiting on the
            # generation lock before reading prevents a false legacy/not-found
            # result and gives interrupted writers a recovery point.
            recover_analysis_publication(base)
            manifest = read_analysis_manifest(base)
            if manifest is not None:
                entry = manifest.analyses.get(analysis_id)
                if entry is None:
                    raise self._not_found(
                        analysis_id,
                        None,
                        available_ids=tuple(
                            sorted({*self._analyses, *manifest.analyses})
                        ),
                    )
                # Verify both mirrors and consume Parquet under the same lock.
                resolve_manifest_artifact_path(
                    base, analysis_id, entry, artifact="csv"
                )
                parquet_path = resolve_manifest_artifact_path(
                    base, analysis_id, entry, artifact="parquet"
                )
                return pd.read_parquet(parquet_path)

            legacy_path = _safe_legacy_path(base)
            if legacy_path is not None:
                return pd.read_parquet(legacy_path)
            raise self._not_found(
                analysis_id,
                None,
                available_ids=tuple(sorted(self._analyses)),
            )

    def available_analysis_ids(
        self, *, deliverables_base: Path | None = None
    ) -> tuple[str, ...]:
        """Return sorted IDs available in memory or the current manifest."""
        ids = set(self._analyses)
        base = self._resolve_deliverables_base(deliverables_base)
        if base is not None:
            with exclusive_path_lock(base / ".analysis-artifacts.lock"):
                recover_analysis_publication(base)
                manifest = read_analysis_manifest(base)
                if manifest is not None:
                    ids.update(manifest.analyses)
        return tuple(sorted(ids))

    def artifact_paths(
        self, analysis_id: str, *, deliverables_base: Path | None = None
    ) -> AnalysisArtifactPaths:
        """Return ID-derived artifact paths for a configured deliverables base."""
        safe_id = validate_analysis_id(analysis_id)
        base = self._resolve_deliverables_base(deliverables_base)
        if base is None:
            raise ValueError(
                "deliverables_base is required to resolve artifact paths"
            )
        return named_analysis_paths(base, safe_id)

    def _resolve_deliverables_base(self, override: Path | None) -> Path | None:
        return (
            Path(override) if override is not None else self._deliverables_base
        )

    def _not_found(
        self,
        analysis_id: str,
        deliverables_base: Path | None,
        *,
        available_ids: tuple[str, ...] | None = None,
    ) -> AnalysisNotFoundError:
        available = (
            available_ids
            if available_ids is not None
            else self.available_analysis_ids(deliverables_base=deliverables_base)
        )
        rendered = ", ".join(available) if available else "none"
        return AnalysisNotFoundError(
            f"analysis {analysis_id!r} is unavailable; available analysis IDs: "
            f"{rendered}"
        )


def _analysis_id_from_input(input_ref: str | AnalysisInputLike) -> str:
    if isinstance(input_ref, str):
        return validate_analysis_id(input_ref)
    try:
        analysis_id = input_ref.analysis_id
    except AttributeError as exc:
        raise TypeError(
            "analysis input must be a string or expose an analysis_id"
        ) from exc
    return validate_analysis_id(analysis_id)


def _safe_legacy_path(deliverables_base: Path) -> Path | None:
    base = Path(deliverables_base).resolve()
    candidate = base / _LEGACY_ANALYSIS_PARQUET
    if not candidate.exists():
        return None
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise AnalysisManifestError(
            f"legacy analysis artifact is unreadable: {candidate}"
        ) from exc
    if resolved.parent != base:
        raise AnalysisManifestError(
            "legacy analysis artifact escapes the deliverables directory"
        )
    return resolved
