"""Tests for named analysis paths and manifest integrity."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import polars as pl
import pytest

from phenotypic.plotting._pipeline._analysis_artifacts import (
    ANALYSIS_MANIFEST_SCHEMA_VERSION,
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
    write_analysis_manifest,
)
from phenotypic.sdk_ import write_tsv


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

    assert paths.tsv == tmp_path / "LinearLagModel.tsv"
    assert paths.csv == tmp_path / "LinearLagModel.csv"
    assert paths.parquet == tmp_path / "LinearLagModel.parquet"
    assert paths.manifest == tmp_path / "analysis_manifest.json"


def test_manifest_entry_round_trip_and_checksums(tmp_path: Path) -> None:
    paths = named_analysis_paths(tmp_path, "LinearLagModel")
    paths.tsv.write_text("value\n1\n", encoding="utf-8")
    paths.parquet.write_bytes(b"parquet-generation-one")
    entry = build_analysis_manifest_entry(
        analysis_id="LinearLagModel",
        producer_class="LinearLagModel",
        tsv_path=paths.tsv,
        parquet_path=paths.parquet,
        rows=1,
        columns=["value"],
    )

    publish_analysis_manifest_entry(tmp_path, "LinearLagModel", entry)
    loaded = read_analysis_manifest(tmp_path)

    assert loaded == AnalysisManifest(analyses={"LinearLagModel": entry})
    payload = json.loads(analysis_manifest_path(tmp_path).read_text())
    assert payload["schema_version"] == 2
    assert len(payload["analyses"]["LinearLagModel"]["tsv_sha256"]) == 64
    assert "csv" not in payload["analyses"]["LinearLagModel"]
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
        paths.tsv.write_text("x\n1\n", encoding="utf-8")
        paths.parquet.write_bytes(analysis_id.encode())
        entry = build_analysis_manifest_entry(
            analysis_id=analysis_id,
            producer_class=analysis_id,
            tsv_path=paths.tsv,
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
        paths.tsv.write_text(f"x\n{index}\n", encoding="utf-8")
        paths.parquet.write_bytes(f"parquet-{index}".encode())
        entries[analysis_id] = build_analysis_manifest_entry(
            analysis_id=analysis_id,
            producer_class=analysis_id,
            tsv_path=paths.tsv,
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


@pytest.mark.parametrize(
    ("version", "table_key", "name"),
    [(1, "csv", "../LinearLagModel.csv"), (2, "tsv", "../LinearLagModel.tsv")],
)
def test_manifest_rejects_traversal_even_with_valid_analysis_key(
    tmp_path: Path, version: int, table_key: str, name: str
) -> None:
    checksum = "0" * 64
    payload = {
        "schema_version": version,
        "analyses": {
            "LinearLagModel": {
                "class": "LinearLagModel",
                table_key: name,
                "parquet": "LinearLagModel.parquet",
                "rows": 0,
                "columns": [],
                f"{table_key}_sha256": checksum,
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
    paths.tsv.write_text("x\n1\n", encoding="utf-8")
    paths.parquet.write_bytes(b"before")
    entry = build_analysis_manifest_entry(
        analysis_id="Model",
        producer_class="Model",
        tsv_path=paths.tsv,
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


def test_analysis_manifest_records_tsv(tmp_path):
    write_tsv(pl.DataFrame({"a": [1]}), tmp_path / "Fit.tsv")
    pl.DataFrame({"a": [1]}).write_parquet(tmp_path / "Fit.parquet")
    entry = build_analysis_manifest_entry(analysis_id="Fit", producer_class="Fit",
                                          tsv_path=tmp_path / "Fit.tsv",
                                          parquet_path=tmp_path / "Fit.parquet", rows=1, columns=["a"])
    write_analysis_manifest(tmp_path, AnalysisManifest(analyses={"Fit": entry}))
    assert read_analysis_manifest(tmp_path).analyses["Fit"].tsv == "Fit.tsv"


def test_old_manifest_with_csv_fields_still_reads(tmp_path):
    old_entry = {"class": "Fit", "csv": "Fit.csv", "parquet": "Fit.parquet", "rows": 1,
                 "columns": ["a"], "csv_sha256": "0" * 64, "parquet_sha256": "0" * 64}
    payload = {"schema_version": ANALYSIS_MANIFEST_SCHEMA_VERSION - 1, "analyses": {"Fit": old_entry}}
    (tmp_path / "analysis_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    assert read_analysis_manifest(tmp_path).analyses["Fit"].tsv == "Fit.csv"


def test_a_version_one_entry_is_carried_forward_by_a_new_publication(tmp_path):
    """A v1 entry survives a v2 rewrite, still naming (and verifying) its CSV."""
    (tmp_path / "Old.csv").write_text("a\n1\n", encoding="utf-8")
    pl.DataFrame({"a": [1]}).write_parquet(tmp_path / "Old.parquet")
    from phenotypic.plotting._pipeline._analysis_artifacts import file_sha256

    old_entry = {
        "class": "Old", "csv": "Old.csv", "parquet": "Old.parquet", "rows": 1,
        "columns": ["a"], "csv_sha256": file_sha256(tmp_path / "Old.csv"),
        "parquet_sha256": file_sha256(tmp_path / "Old.parquet"),
    }
    analysis_manifest_path(tmp_path).write_text(
        json.dumps({"schema_version": 1, "analyses": {"Old": old_entry}}), encoding="utf-8"
    )
    write_tsv(pl.DataFrame({"a": [2]}), tmp_path / "New.tsv")
    pl.DataFrame({"a": [2]}).write_parquet(tmp_path / "New.parquet")
    new_entry = build_analysis_manifest_entry(
        analysis_id="New", producer_class="New", tsv_path=tmp_path / "New.tsv",
        parquet_path=tmp_path / "New.parquet", rows=1, columns=["a"],
    )

    publish_analysis_manifest_entry(tmp_path, "New", new_entry)

    payload = json.loads(analysis_manifest_path(tmp_path).read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["analyses"]["Old"]["tsv"] == "Old.csv"
    loaded = read_analysis_manifest(tmp_path)
    assert loaded.analyses["New"].tsv == "New.tsv"
    old = loaded.analyses["Old"]
    assert resolve_manifest_artifact_path(tmp_path, "Old", old, artifact="tsv") == (
        tmp_path / "Old.csv"
    ).resolve()


def test_a_version_one_entry_must_name_a_csv(tmp_path):
    entry = {"class": "Fit", "csv": "Fit.tsv", "parquet": "Fit.parquet", "rows": 1,
             "columns": ["a"], "csv_sha256": "0" * 64, "parquet_sha256": "0" * 64}
    with pytest.raises(AnalysisManifestError, match="must use artifact filename"):
        AnalysisManifest.from_mapping({"schema_version": 1, "analyses": {"Fit": entry}})


def test_a_version_one_entry_may_not_mix_in_tsv_fields():
    entry = {"class": "Fit", "csv": "Fit.csv", "tsv": "Fit.tsv", "parquet": "Fit.parquet",
             "rows": 1, "columns": ["a"], "csv_sha256": "0" * 64, "parquet_sha256": "0" * 64}
    with pytest.raises(AnalysisManifestError, match="mixes"):
        AnalysisManifest.from_mapping({"schema_version": 1, "analyses": {"Fit": entry}})


@pytest.mark.parametrize("version", [0, 3, True, "2"])
def test_an_unknown_manifest_version_is_refused(version):
    with pytest.raises(AnalysisManifestError, match="unsupported analysis manifest"):
        AnalysisManifest.from_mapping({"schema_version": version, "analyses": {}})


def test_resolve_refuses_the_retired_csv_artifact_kind(tmp_path):
    write_tsv(pl.DataFrame({"a": [1]}), tmp_path / "Fit.tsv")
    pl.DataFrame({"a": [1]}).write_parquet(tmp_path / "Fit.parquet")
    entry = build_analysis_manifest_entry(analysis_id="Fit", producer_class="Fit",
                                          tsv_path=tmp_path / "Fit.tsv",
                                          parquet_path=tmp_path / "Fit.parquet", rows=1, columns=["a"])
    with pytest.raises(ValueError, match="'tsv' or 'parquet'"):
        resolve_manifest_artifact_path(tmp_path, "Fit", entry, artifact="csv")
