"""Tests for dynamic in-memory and persisted analysis resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from phenotypic.plotting._analysis_artifacts import (
    AnalysisArtifactIntegrityError,
    AnalysisManifestEntry,
    build_analysis_manifest_entry,
    file_sha256,
    named_analysis_paths,
    publish_analysis_manifest_entry,
    write_analysis_publication_journal,
)
from phenotypic.plotting._analysis_registry import (
    AnalysisNotFoundError,
    AnalysisRegistry,
)
from phenotypic.plotting._bindings import AnalysisInput


@dataclass(frozen=True)
class _Input:
    analysis_id: str


def test_analysis_input_uses_safe_analysis_id_contract() -> None:
    assert (
        AnalysisInput(analysis_id="LinearLagModel").analysis_id
        == "LinearLagModel"
    )

    with pytest.raises(ValidationError, match="analysis_id must be"):
        AnalysisInput(analysis_id="../LinearLagModel")


def _publish_table(
    deliverables: Path, analysis_id: str, table: pd.DataFrame
) -> None:
    paths = named_analysis_paths(deliverables, analysis_id)
    table.to_csv(paths.csv, index=False)
    table.to_parquet(paths.parquet, index=False)
    entry = build_analysis_manifest_entry(
        analysis_id=analysis_id,
        producer_class=analysis_id,
        csv_path=paths.csv,
        parquet_path=paths.parquet,
        rows=len(table),
        columns=list(table.columns),
    )
    publish_analysis_manifest_entry(deliverables, analysis_id, entry)


def test_registry_returns_exact_current_table_before_persisted_artifact(
    tmp_path: Path,
) -> None:
    persisted = pd.DataFrame({"value": [1]})
    current = pd.DataFrame({"value": [2]})
    _publish_table(tmp_path, "Model", persisted)
    producer = object()
    registry = AnalysisRegistry(tmp_path)
    registration = registry.register("Model", current, producer=producer)

    resolved = registry.resolve(_Input("Model"))

    assert resolved is current
    assert registration.table is current
    assert registration.producer is producer
    assert registration.manifest_entry is None


def test_registry_reads_manifest_selected_parquet(tmp_path: Path) -> None:
    expected = pd.DataFrame({"strain": ["A", "B"], "lag": [1.5, 2.0]})
    _publish_table(tmp_path, "LinearLagModel", expected)

    resolved = AnalysisRegistry(tmp_path).resolve("LinearLagModel")

    pd.testing.assert_frame_equal(resolved, expected)


def test_persisted_resolution_revalidates_checksum_on_every_refresh(
    tmp_path: Path,
) -> None:
    expected = pd.DataFrame({"value": [1]})
    _publish_table(tmp_path, "Model", expected)
    registry = AnalysisRegistry(tmp_path)
    pd.testing.assert_frame_equal(registry.resolve("Model"), expected)
    named_analysis_paths(tmp_path, "Model").parquet.write_bytes(b"corrupt")

    with pytest.raises(
        AnalysisArtifactIntegrityError, match="checksum mismatch"
    ):
        registry.resolve("Model")


def test_persisted_resolution_rejects_mixed_csv_parquet_generation(
    tmp_path: Path,
) -> None:
    expected = pd.DataFrame({"value": [1]})
    _publish_table(tmp_path, "Model", expected)
    named_analysis_paths(tmp_path, "Model").csv.write_text(
        "value\n2\n", encoding="utf-8"
    )

    with pytest.raises(
        AnalysisArtifactIntegrityError, match="csv checksum mismatch"
    ):
        AnalysisRegistry(tmp_path).resolve("Model")


def test_registry_recovers_interrupted_artifact_publication(
    tmp_path: Path,
) -> None:
    previous = pd.DataFrame({"value": [1]})
    replacement = pd.DataFrame({"value": [2]})
    _publish_table(tmp_path, "Model", previous)
    paths = named_analysis_paths(tmp_path, "Model")
    token = "a" * 32
    staged_csv = tmp_path / f".{paths.csv.name}.{token}.staged"
    staged_parquet = tmp_path / f".{paths.parquet.name}.{token}.staged"
    backup_csv = tmp_path / f".{paths.csv.name}.{token}.backup"
    replacement.to_csv(staged_csv, index=False)
    replacement.to_parquet(staged_parquet, index=False)
    entry = AnalysisManifestEntry(
        producer_class="Model",
        csv=paths.csv.name,
        parquet=paths.parquet.name,
        rows=len(replacement),
        columns=tuple(replacement.columns),
        csv_sha256=file_sha256(staged_csv),
        parquet_sha256=file_sha256(staged_parquet),
    )
    write_analysis_publication_journal(
        tmp_path,
        analysis_id="Model",
        token=token,
        old_csv_exists=True,
        old_parquet_exists=True,
        entry=entry,
    )
    # Simulate process loss after replacing only the CSV mirror.
    os.replace(paths.csv, backup_csv)
    os.replace(staged_csv, paths.csv)

    resolved = AnalysisRegistry(tmp_path).resolve("Model")

    pd.testing.assert_frame_equal(resolved, previous)
    assert paths.csv.read_text(encoding="utf-8").startswith("value\n1")
    assert not (tmp_path / ".analysis-publication.json").exists()
    assert not backup_csv.exists()
    assert not staged_parquet.exists()


def test_present_manifest_prevents_legacy_fallback(tmp_path: Path) -> None:
    _publish_table(tmp_path, "Available", pd.DataFrame({"value": [1]}))
    pd.DataFrame({"legacy": [2]}).to_parquet(tmp_path / "analysis.parquet")

    with pytest.raises(AnalysisNotFoundError, match="Available"):
        AnalysisRegistry(tmp_path).resolve("Missing")


def test_legacy_parquet_is_used_only_without_manifest(tmp_path: Path) -> None:
    expected = pd.DataFrame({"legacy": [2]})
    expected.to_parquet(tmp_path / "analysis.parquet", index=False)

    resolved = AnalysisRegistry(tmp_path).resolve("LinearLagModel")

    pd.testing.assert_frame_equal(resolved, expected)


def test_missing_analysis_lists_memory_and_manifest_ids(
    tmp_path: Path,
) -> None:
    _publish_table(tmp_path, "Persisted", pd.DataFrame({"value": [1]}))
    registry = AnalysisRegistry(tmp_path)
    registry.register("Current", pd.DataFrame({"value": [2]}))

    with pytest.raises(AnalysisNotFoundError) as exc_info:
        registry.resolve("Missing")

    message = str(exc_info.value)
    assert "Current" in message
    assert "Persisted" in message


def test_registry_rejects_non_dataframe_registration() -> None:
    registry = AnalysisRegistry()

    with pytest.raises(TypeError, match="pandas DataFrame"):
        registry.register("Model", {"value": [1]})  # type: ignore[arg-type]
