"""Tests for named analysis paths and manifest integrity."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from phenotypic.plotting._pipeline._analysis_artifacts import (
    AnalysisArtifactIntegrityError,
    AnalysisManifest,
    AnalysisManifestEntry,
    AnalysisManifestError,
    analysis_manifest_path,
    build_analysis_manifest_entry,
    named_analysis_paths,
    publish_analysis_manifest_entry,
    read_analysis_manifest,
    resolve_manifest_artifact_path,
    validate_analysis_id,
)


@pytest.mark.parametrize(
    "analysis_id",
    ["A", "LinearLagModel", "model_2", "model-v2", "model.v2", "A" * 128],
)
def test_validate_analysis_id_accepts_safe_ascii_stems(
    analysis_id: str,
) -> None:
    assert validate_analysis_id(analysis_id) == analysis_id


@pytest.mark.parametrize(
    "analysis_id",
    [
        "",
        "1model",
        ".",
        "..",
        "a/b",
        "a\\b",
        "a b",
        "a\tmodel",
        "mødel",
        "A" * 129,
    ],
)
def test_validate_analysis_id_rejects_unsafe_stems(analysis_id: str) -> None:
    with pytest.raises(ValueError, match="analysis_id must be"):
        validate_analysis_id(analysis_id)


@pytest.mark.parametrize(
    "analysis_id",
    [
        "Measurements",
        "MASTER_MEASUREMENTS",
        "Metadata",
        "Error_Analysis",
        "Verified",
    ],
)
def test_validate_analysis_id_rejects_canonical_table_collisions(
    analysis_id: str,
) -> None:
    with pytest.raises(ValueError, match="canonical deliverables table"):
        validate_analysis_id(analysis_id)


def test_named_paths_are_derived_from_analysis_id(tmp_path: Path) -> None:
    paths = named_analysis_paths(tmp_path, "LinearLagModel")

    assert paths.csv == tmp_path / "LinearLagModel.csv"
    assert paths.parquet == tmp_path / "LinearLagModel.parquet"
    assert paths.manifest == tmp_path / "analysis_manifest.json"


def test_manifest_entry_round_trip_and_checksums(tmp_path: Path) -> None:
    paths = named_analysis_paths(tmp_path, "LinearLagModel")
    paths.csv.write_text("value\n1\n", encoding="utf-8")
    paths.parquet.write_bytes(b"parquet-generation-one")
    entry = build_analysis_manifest_entry(
        analysis_id="LinearLagModel",
        producer_class="LinearLagModel",
        csv_path=paths.csv,
        parquet_path=paths.parquet,
        rows=1,
        columns=["value"],
    )

    publish_analysis_manifest_entry(tmp_path, "LinearLagModel", entry)
    loaded = read_analysis_manifest(tmp_path)

    assert loaded == AnalysisManifest(analyses={"LinearLagModel": entry})
    payload = json.loads(analysis_manifest_path(tmp_path).read_text())
    assert payload["schema_version"] == 1
    assert len(payload["analyses"]["LinearLagModel"]["csv_sha256"]) == 64
    assert (
        resolve_manifest_artifact_path(
            tmp_path, "LinearLagModel", entry, artifact="parquet"
        )
        == paths.parquet.resolve()
    )


def test_manifest_update_retains_other_analysis_entries(
    tmp_path: Path,
) -> None:
    for analysis_id in ("First", "Second"):
        paths = named_analysis_paths(tmp_path, analysis_id)
        paths.csv.write_text("x\n1\n", encoding="utf-8")
        paths.parquet.write_bytes(analysis_id.encode())
        entry = build_analysis_manifest_entry(
            analysis_id=analysis_id,
            producer_class=analysis_id,
            csv_path=paths.csv,
            parquet_path=paths.parquet,
            rows=1,
            columns=["x"],
        )
        publish_analysis_manifest_entry(tmp_path, analysis_id, entry)

    loaded = read_analysis_manifest(tmp_path)

    assert loaded is not None
    assert tuple(sorted(loaded.analyses)) == ("First", "Second")


def test_concurrent_manifest_publications_retain_every_entry(
    tmp_path: Path,
) -> None:
    entries: dict[str, AnalysisManifestEntry] = {}
    for index in range(8):
        analysis_id = f"Model{index}"
        paths = named_analysis_paths(tmp_path, analysis_id)
        paths.csv.write_text(f"x\n{index}\n", encoding="utf-8")
        paths.parquet.write_bytes(f"parquet-{index}".encode())
        entries[analysis_id] = build_analysis_manifest_entry(
            analysis_id=analysis_id,
            producer_class=analysis_id,
            csv_path=paths.csv,
            parquet_path=paths.parquet,
            rows=1,
            columns=["x"],
        )

    with ThreadPoolExecutor(max_workers=len(entries)) as executor:
        futures = [
            executor.submit(
                publish_analysis_manifest_entry,
                tmp_path,
                analysis_id,
                entry,
            )
            for analysis_id, entry in entries.items()
        ]
        for future in futures:
            future.result()

    loaded = read_analysis_manifest(tmp_path)
    assert loaded is not None
    assert set(loaded.analyses) == set(entries)


def test_manifest_rejects_traversal_even_with_valid_analysis_key(
    tmp_path: Path,
) -> None:
    checksum = "0" * 64
    payload = {
        "schema_version": 1,
        "analyses": {
            "LinearLagModel": {
                "class": "LinearLagModel",
                "csv": "../LinearLagModel.csv",
                "parquet": "LinearLagModel.parquet",
                "rows": 0,
                "columns": [],
                "csv_sha256": checksum,
                "parquet_sha256": checksum,
            }
        },
    }
    analysis_manifest_path(tmp_path).write_text(json.dumps(payload))

    with pytest.raises(
        AnalysisManifestError, match="must use artifact filename"
    ):
        read_analysis_manifest(tmp_path)


def test_manifest_checksum_mismatch_is_fatal(tmp_path: Path) -> None:
    paths = named_analysis_paths(tmp_path, "Model")
    paths.csv.write_text("x\n1\n", encoding="utf-8")
    paths.parquet.write_bytes(b"before")
    entry = build_analysis_manifest_entry(
        analysis_id="Model",
        producer_class="Model",
        csv_path=paths.csv,
        parquet_path=paths.parquet,
        rows=1,
        columns=["x"],
    )
    paths.parquet.write_bytes(b"after")

    with pytest.raises(
        AnalysisArtifactIntegrityError, match="checksum mismatch"
    ):
        resolve_manifest_artifact_path(tmp_path, "Model", entry)


def test_manifest_entry_requires_exact_fields() -> None:
    with pytest.raises(AnalysisManifestError, match="invalid fields"):
        AnalysisManifestEntry.from_mapping("Model", {})


def test_manifest_rejects_case_insensitive_analysis_id_collisions() -> None:
    checksum = "0" * 64

    def entry(analysis_id: str) -> dict[str, object]:
        return {
            "class": analysis_id,
            "csv": f"{analysis_id}.csv",
            "parquet": f"{analysis_id}.parquet",
            "rows": 0,
            "columns": [],
            "csv_sha256": checksum,
            "parquet_sha256": checksum,
        }

    with pytest.raises(AnalysisManifestError, match="case-insensitively"):
        AnalysisManifest.from_mapping(
            {
                "schema_version": 1,
                "analyses": {"Model": entry("Model"), "model": entry("model")},
            }
        )
