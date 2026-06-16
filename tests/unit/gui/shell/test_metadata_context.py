"""Unit tests for the shell-owned metadata CSV context."""
from __future__ import annotations

from pathlib import Path

from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.schema import METADATA


def _write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_metadata_payload_accepts_in_sandbox_csv(tmp_path: Path) -> None:
    from phenotypic.gui.shell._metadata_context import metadata_payload_from_path

    csv_path = _write_csv(
        tmp_path / "layout.csv",
        f"{METADATA.IMAGE_NAME},Treatment\nplate_a,control\nplate_b,stress\n",
    )
    sandbox = SandboxRoot.from_path(tmp_path)

    payload = metadata_payload_from_path(sandbox, csv_path)

    assert payload is not None
    assert payload["abs_path"] == str(csv_path.resolve())
    assert payload["rel_path"] == "layout.csv"
    assert payload["label"] == "layout.csv"
    assert payload["validated"] is True
    assert payload["version"] == 1
    assert payload["has_image_name"] is True
    assert payload["row_count"] == 2
    assert payload["unique_image_names"] is True


def test_metadata_payload_rejects_invalid_files(tmp_path: Path) -> None:
    from phenotypic.gui.shell._metadata_context import metadata_payload_from_path

    sandbox_root = tmp_path / "sandbox"
    sandbox_root.mkdir()
    outside = _write_csv(tmp_path / "outside.csv", "A,B\n1,2\n")
    directory = sandbox_root / "dir"
    directory.mkdir()
    txt = _write_csv(sandbox_root / "layout.txt", "A,B\n1,2\n")
    sandbox = SandboxRoot.from_path(sandbox_root)

    assert metadata_payload_from_path(sandbox, outside) is None
    assert metadata_payload_from_path(sandbox, directory) is None
    assert metadata_payload_from_path(sandbox, txt) is None
    assert metadata_payload_from_path(sandbox, sandbox_root / "missing.csv") is None


def test_metadata_payload_allows_missing_image_name_column(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.shell._metadata_context import metadata_payload_from_path

    csv_path = _write_csv(tmp_path / "layout.csv", "Plate,Treatment\n1,control\n")
    sandbox = SandboxRoot.from_path(tmp_path)

    payload = metadata_payload_from_path(sandbox, csv_path)

    assert payload is not None
    assert payload["has_image_name"] is False
    assert payload["row_count"] == 1
    assert payload["unique_image_names"] is False


def test_metadata_payload_reports_duplicate_image_names(tmp_path: Path) -> None:
    from phenotypic.gui.shell._metadata_context import metadata_payload_from_path

    csv_path = _write_csv(
        tmp_path / "layout.csv",
        f"{METADATA.IMAGE_NAME},Treatment\nplate_a,control\nplate_a,stress\n",
    )
    sandbox = SandboxRoot.from_path(tmp_path)

    payload = metadata_payload_from_path(sandbox, csv_path)

    assert payload is not None
    assert payload["has_image_name"] is True
    assert payload["unique_image_names"] is False


def test_resolve_metadata_csv_rejects_malformed_payloads(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.shell._metadata_context import resolve_metadata_csv

    sandbox = SandboxRoot.from_path(tmp_path)
    csv_path = _write_csv(tmp_path / "layout.csv", "A,B\n1,2\n")

    assert resolve_metadata_csv(sandbox, None) is None
    assert resolve_metadata_csv(sandbox, {"abs_path": str(csv_path)}) is None
    assert (
        resolve_metadata_csv(
            sandbox,
            {
                "version": 999,
                "validated": True,
                "abs_path": str(csv_path),
            },
        )
        is None
    )


def test_read_metadata_row_matches_image_stem(tmp_path: Path) -> None:
    from phenotypic.gui.shell._metadata_context import (
        metadata_payload_from_path,
        read_metadata_row_for_image_stem,
    )

    csv_path = _write_csv(
        tmp_path / "layout.csv",
        f"{METADATA.IMAGE_NAME},Treatment,Replicate\nplate_a,control,1\n",
    )
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = metadata_payload_from_path(sandbox, csv_path)

    result = read_metadata_row_for_image_stem(sandbox, payload, "plate_a")

    assert result.state == "matched"
    assert result.image_stem == "plate_a"
    assert result.row_count == 1
    assert result.rows == [{"Treatment": "control", "Replicate": "1"}]


def test_read_metadata_row_reports_expected_states(tmp_path: Path) -> None:
    from phenotypic.gui.shell._metadata_context import (
        metadata_payload_from_path,
        read_metadata_row_for_image_stem,
    )

    no_key = _write_csv(tmp_path / "no-key.csv", "Treatment\ncontrol\n")
    unique = _write_csv(
        tmp_path / "unique.csv",
        f"{METADATA.IMAGE_NAME},Treatment\nplate_b,control\n",
    )
    sandbox = SandboxRoot.from_path(tmp_path)

    assert read_metadata_row_for_image_stem(sandbox, None, "plate_a").state == "unset"
    assert (
        read_metadata_row_for_image_stem(
            sandbox, metadata_payload_from_path(sandbox, no_key), "plate_a"
        ).state
        == "missing_image_name"
    )
    assert (
        read_metadata_row_for_image_stem(
            sandbox, metadata_payload_from_path(sandbox, unique), "plate_a"
        ).state
        == "no_match"
    )


def test_read_metadata_row_returns_all_matching_colony_rows(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.shell._metadata_context import (
        metadata_payload_from_path,
        read_metadata_row_for_image_stem,
    )

    csv_path = _write_csv(
        tmp_path / "layout.csv",
        (
            f"{METADATA.IMAGE_NAME},Colony,Treatment\n"
            "plate_a,A01,control\n"
            "plate_a,A02,stress\n"
            "plate_b,B01,control\n"
        ),
    )
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = metadata_payload_from_path(sandbox, csv_path)

    result = read_metadata_row_for_image_stem(sandbox, payload, "plate_a")

    assert result.state == "matched"
    assert result.row_count == 2
    assert result.rows == [
        {"Colony": "A01", "Treatment": "control"},
        {"Colony": "A02", "Treatment": "stress"},
    ]
