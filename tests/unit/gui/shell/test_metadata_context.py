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
    assert payload["kind"] == "metadata_csv"
    assert payload["relative_path"] == "layout.csv"
    assert payload["absolute_path_at_selection"] == str(csv_path.resolve())
    assert payload["validation"] == {
        "exists": True,
        "is_file": True,
        "is_csv": True,
        "readable": True,
    }
    assert payload["selected_at"]
    assert payload["sandbox_fingerprint"]
    assert payload["abs_path"] == str(csv_path.resolve())
    assert payload["rel_path"] == "layout.csv"
    assert payload["label"] == "layout.csv"
    assert payload["validated"] is True
    assert payload["version"] == 2
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


def test_metadata_image_identity_supports_current_and_legacy_headers() -> None:
    from phenotypic.gui.shell._metadata_context import (
        resolve_metadata_image_identity,
    )

    columns = [
        str(METADATA.IMAGE_NAME),
        "Metadata_ImageName",
        "Metadata_ImageFileName",
        "ImageName",
    ]
    rows = [
        {
            str(METADATA.IMAGE_NAME): "plates/plate_a.tif",
            "Metadata_ImageName": "plate_a",
            "Metadata_ImageFileName": r"C:\images\plate_a.tiff",
            "ImageName": "plate_a.png",
        }
    ]

    identity = resolve_metadata_image_identity(columns, rows)

    assert identity.state == "resolved"
    assert identity.column == str(METADATA.IMAGE_NAME)
    assert identity.recognized_columns == tuple(columns)
    assert identity.normalized_values == ("plate_a",)


def test_metadata_image_identity_prefers_populated_legacy_column() -> None:
    from phenotypic.gui.shell._metadata_context import (
        resolve_metadata_image_identity,
    )

    columns = [str(METADATA.IMAGE_NAME), "Metadata_ImageFileName"]
    rows = [
        {
            str(METADATA.IMAGE_NAME): "",
            "Metadata_ImageFileName": "plate_a.tif",
        },
        {
            str(METADATA.IMAGE_NAME): "",
            "Metadata_ImageFileName": "plate_b.tif",
        },
    ]

    identity = resolve_metadata_image_identity(columns, rows)

    assert identity.state == "resolved"
    assert identity.column == "Metadata_ImageFileName"
    assert identity.normalized_values == ("plate_a", "plate_b")


def test_metadata_image_identity_is_ambiguous_when_aliases_disagree() -> None:
    from phenotypic.gui.shell._metadata_context import (
        resolve_metadata_image_identity,
    )

    columns = ["Metadata_ImageName", "Metadata_ImageFileName"]
    rows = [
        {
            "Metadata_ImageName": "plate_a",
            "Metadata_ImageFileName": "plate_b.tif",
        }
    ]

    identity = resolve_metadata_image_identity(columns, rows)

    assert identity.state == "ambiguous"
    assert identity.column is None
    assert identity.normalized_values == ()


def test_metadata_payload_recognizes_legacy_image_filename(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.shell._metadata_context import metadata_payload_from_path

    csv_path = _write_csv(
        tmp_path / "layout.csv",
        "Metadata_ImageFileName,Treatment\nplate_a.tif,control\n",
    )
    sandbox = SandboxRoot.from_path(tmp_path)

    payload = metadata_payload_from_path(sandbox, csv_path)

    assert payload is not None
    assert payload["has_image_name"] is True
    assert payload["unique_image_names"] is True


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


def test_v2_metadata_rejects_same_relative_path_in_different_sandbox(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.shell._metadata_context import (
        metadata_csv_label,
        metadata_csv_title,
        metadata_payload_from_path,
        resolve_metadata_csv,
        resolve_metadata_csv_state,
    )

    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    old_root.mkdir()
    new_root.mkdir()
    old_csv = _write_csv(old_root / "layout.csv", "A,B\n1,2\n")
    _write_csv(new_root / "layout.csv", "A,B\n3,4\n")
    old_sandbox = SandboxRoot.from_path(old_root)
    new_sandbox = SandboxRoot.from_path(new_root)
    payload = metadata_payload_from_path(old_sandbox, old_csv)

    resolution = resolve_metadata_csv_state(new_sandbox, payload)

    assert resolution.state == "fingerprint_mismatch"
    assert resolution.path is None
    assert resolution.payload_version == 2
    assert resolve_metadata_csv(new_sandbox, payload) is None
    assert (
        metadata_csv_label(payload, sandbox=new_sandbox)
        == "Previous metadata unavailable in this sandbox"
    )
    assert (
        metadata_csv_title(payload, sandbox=new_sandbox)
        == "Previous metadata unavailable in this sandbox"
    )


def test_v2_metadata_resolution_ignores_diagnostic_absolute_path(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.shell._metadata_context import (
        metadata_payload_from_path,
        resolve_metadata_csv_state,
    )

    csv_path = _write_csv(tmp_path / "layout.csv", "A,B\n1,2\n")
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = metadata_payload_from_path(sandbox, csv_path)
    assert payload is not None
    payload["absolute_path_at_selection"] = "/stale/root/layout.csv"
    payload["abs_path"] = "/stale/root/layout.csv"

    resolution = resolve_metadata_csv_state(sandbox, payload)

    assert resolution.state == "resolved"
    assert resolution.path == csv_path.resolve()


def test_v2_metadata_reports_unavailable_after_csv_is_removed(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.shell._metadata_context import (
        metadata_csv_label,
        metadata_payload_from_path,
        resolve_metadata_csv_state,
    )

    csv_path = _write_csv(tmp_path / "layout.csv", "A,B\n1,2\n")
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = metadata_payload_from_path(sandbox, csv_path)
    descriptor = dict(payload or {})
    csv_path.unlink()

    resolution = resolve_metadata_csv_state(sandbox, payload)

    assert resolution.state == "unavailable"
    assert resolution.path is None
    assert payload == descriptor
    assert (
        metadata_csv_label(payload, sandbox=sandbox)
        == "Previous metadata unavailable in this sandbox"
    )


def test_v1_metadata_payload_reads_without_rewrite(tmp_path: Path) -> None:
    from phenotypic.gui.shell._metadata_context import resolve_metadata_csv_state

    csv_path = _write_csv(tmp_path / "layout.csv", "A,B\n1,2\n")
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = {
        "abs_path": str(csv_path.resolve()),
        "rel_path": "layout.csv",
        "label": "layout.csv",
        "validated": True,
        "version": 1,
        "has_image_name": False,
        "row_count": 1,
        "unique_image_names": False,
    }

    resolution = resolve_metadata_csv_state(sandbox, payload)

    assert resolution.state == "resolved"
    assert resolution.path == csv_path.resolve()
    assert resolution.payload_version == 1
    assert payload["version"] == 1
    assert "sandbox_fingerprint" not in payload


def test_v1_metadata_rejects_malformed_and_inconsistent_paths(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.shell._metadata_context import resolve_metadata_csv_state

    selected_csv = _write_csv(tmp_path / "layout.csv", "A,B\n1,2\n")
    other_csv = _write_csv(tmp_path / "other.csv", "A,B\n3,4\n")
    sandbox = SandboxRoot.from_path(tmp_path)
    payloads = [
        {
            "version": 1,
            "abs_path": "\x00",
            "rel_path": "layout.csv",
            "validated": True,
        },
        {
            "version": 1,
            "abs_path": str(selected_csv.resolve()),
            "rel_path": "lay\x00out.csv",
            "validated": True,
        },
        {
            "version": 1,
            "abs_path": str(other_csv.resolve()),
            "rel_path": "layout.csv",
            "validated": True,
        },
    ]

    resolutions = [
        resolve_metadata_csv_state(sandbox, payload)
        for payload in payloads
    ]

    assert [resolution.state for resolution in resolutions] == [
        "fingerprint_mismatch",
        "fingerprint_mismatch",
        "fingerprint_mismatch",
    ]
    assert all(resolution.path is None for resolution in resolutions)


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


def test_read_metadata_row_matches_legacy_filename_and_strips_extension(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.shell._metadata_context import (
        metadata_payload_from_path,
        read_metadata_row_for_image_stem,
    )

    csv_path = _write_csv(
        tmp_path / "layout.csv",
        "Metadata_ImageFileName,Treatment\nplate_a.tiff,control\n",
    )
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = metadata_payload_from_path(sandbox, csv_path)

    result = read_metadata_row_for_image_stem(
        sandbox,
        payload,
        "plate_a.png",
    )

    assert result.state == "matched"
    assert result.rows == [{"Treatment": "control"}]


def test_read_metadata_row_rejects_conflicting_recognized_columns(
    tmp_path: Path,
) -> None:
    from phenotypic.gui.shell._metadata_context import (
        metadata_payload_from_path,
        read_metadata_row_for_image_stem,
    )

    csv_path = _write_csv(
        tmp_path / "layout.csv",
        (
            "Metadata_ImageName,Metadata_ImageFileName,Treatment\n"
            "plate_a,plate_b.tif,control\n"
        ),
    )
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = metadata_payload_from_path(sandbox, csv_path)

    result = read_metadata_row_for_image_stem(sandbox, payload, "plate_a")

    assert result.state == "ambiguous_image_name"
    assert result.rows == []


def test_read_metadata_csv_table_returns_columns_and_rows(tmp_path: Path) -> None:
    from phenotypic.gui.shell._metadata_context import read_metadata_csv_table

    csv_path = _write_csv(
        tmp_path / "plain.csv",
        "image,media,tp\nplateA,YPD,0h\nplateB,SD,6h\n",
    )

    columns, rows = read_metadata_csv_table(csv_path)

    assert columns == ["image", "media", "tp"]
    assert rows == [
        {"image": "plateA", "media": "YPD", "tp": "0h"},
        {"image": "plateB", "media": "SD", "tp": "6h"},
    ]


def test_read_metadata_csv_table_strips_excel_utf8_bom(tmp_path: Path) -> None:
    # Excel CSV exports carry a UTF-8 BOM. Plain utf-8 would leave a "﻿"
    # on the first header name, so a csv_image_col="image" join would miss
    # ("﻿image" != "image"). utf-8-sig (matching _read_rows) strips it.
    from phenotypic.gui.shell._metadata_context import read_metadata_csv_table

    csv_path = tmp_path / "bom.csv"
    csv_path.write_text(
        "image,media\nplateA,YPD\n", encoding="utf-8-sig"
    )

    columns, rows = read_metadata_csv_table(csv_path)

    assert columns[0] == "image"
    assert "﻿" not in columns[0]
    assert rows == [{"image": "plateA", "media": "YPD"}]
