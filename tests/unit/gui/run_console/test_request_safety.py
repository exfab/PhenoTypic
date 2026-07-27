"""Behavioral tests for Run output and metadata request authority."""
from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic.gui.run_console._request_safety import (
    RunRequestSafetyError,
    build_metadata_preflight,
    confirm_output_target,
    recheck_metadata_selection,
    validate_output_confirmation,
)
from phenotypic.gui.shell._metadata_context import metadata_payload_from_path
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.schema import METADATA


def _one_image_source(root: Path, *, stem: str = "plate_a") -> Path:
    source = root / "images"
    source.mkdir()
    (source / f"{stem}.tif").write_bytes(b"one-image")
    return source


def _metadata_csv(root: Path, *identities: str) -> Path:
    metadata = root / "metadata.csv"
    rows = "\n".join(identities)
    metadata.write_text(
        f"{METADATA.IMAGE_NAME},Treatment\n"
        + "\n".join(f"{identity},control" for identity in identities)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return metadata


def test_typed_relative_output_preserves_nonexistent_canonical_target(
    tmp_path: Path,
) -> None:
    sandbox = SandboxRoot.from_path(tmp_path)
    target = tmp_path / "acceptance" / "new-run"

    receipt = confirm_output_target(
        sandbox,
        "acceptance/new-run",
        project_root=tmp_path / "project",
    )

    assert receipt.canonical_path == str(target)
    assert receipt.relative_path == "acceptance/new-run"
    assert not target.exists()
    assert validate_output_confirmation(
        sandbox,
        "acceptance/new-run",
        receipt.to_json(),
        project_root=tmp_path / "project",
    ) == target


@pytest.mark.parametrize("typed_value", [None, "", "   ", "."])
def test_output_confirmation_rejects_empty_or_sandbox_root(
    tmp_path: Path,
    typed_value: object,
) -> None:
    sandbox = SandboxRoot.from_path(tmp_path)

    with pytest.raises(RunRequestSafetyError):
        confirm_output_target(
            sandbox,
            typed_value,
            project_root=tmp_path / "project",
        )


def test_output_confirmation_rejects_project_root_escape_and_file(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    existing_file = tmp_path / "output.txt"
    existing_file.write_text("not a directory", encoding="utf-8")
    sandbox = SandboxRoot.from_path(tmp_path)

    for target in (project, tmp_path.parent / "escape", existing_file):
        with pytest.raises(RunRequestSafetyError):
            confirm_output_target(
                sandbox,
                str(target),
                project_root=project,
            )


def test_changed_or_silently_substituted_output_fails_stale_receipt(
    tmp_path: Path,
) -> None:
    sandbox = SandboxRoot.from_path(tmp_path)
    receipt = confirm_output_target(
        sandbox,
        "requested/new-run",
        project_root=tmp_path / "project",
    )

    with pytest.raises(RunRequestSafetyError, match="stale"):
        validate_output_confirmation(
            sandbox,
            "silently-substituted",
            receipt.to_json(),
            project_root=tmp_path / "project",
        )


def test_ambient_metadata_is_omitted_without_explicit_include(
    tmp_path: Path,
) -> None:
    source = _one_image_source(tmp_path)
    metadata = _metadata_csv(tmp_path, "plate_a")
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = metadata_payload_from_path(sandbox, metadata)
    preflight = build_metadata_preflight(sandbox, str(source), payload)

    selected = recheck_metadata_selection(
        sandbox,
        input_dir=str(source),
        metadata_payload=payload,
        choice="omit",
        acknowledgement=[],
        preflight_payload=preflight.to_json(),
    )

    assert preflight.compatibility == "compatible"
    assert preflight.matched_source_count == 1
    assert selected is None


def test_compatible_metadata_can_be_explicitly_included(
    tmp_path: Path,
) -> None:
    source = _one_image_source(tmp_path)
    metadata = _metadata_csv(tmp_path, "plate_a")
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = metadata_payload_from_path(sandbox, metadata)
    preflight = build_metadata_preflight(sandbox, str(source), payload)

    selected = recheck_metadata_selection(
        sandbox,
        input_dir=str(source),
        metadata_payload=payload,
        choice="include",
        acknowledgement=[],
        preflight_payload=preflight.to_json(),
    )

    assert selected == metadata


def test_mismatched_metadata_requires_visible_warning_acknowledgement(
    tmp_path: Path,
) -> None:
    source = _one_image_source(tmp_path)
    metadata = _metadata_csv(tmp_path, "unrelated", "unrelated")
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = metadata_payload_from_path(sandbox, metadata)
    preflight = build_metadata_preflight(sandbox, str(source), payload)

    assert preflight.compatibility == "warning"
    assert preflight.unmatched_source_count == 1
    assert preflight.metadata_only_count == 2
    assert preflight.duplicate_identity_count == 1
    with pytest.raises(RunRequestSafetyError, match="Acknowledge"):
        recheck_metadata_selection(
            sandbox,
            input_dir=str(source),
            metadata_payload=payload,
            choice="include",
            acknowledgement=[],
            preflight_payload=preflight.to_json(),
        )
    assert recheck_metadata_selection(
        sandbox,
        input_dir=str(source),
        metadata_payload=payload,
        choice="include",
        acknowledgement=["acknowledge"],
        preflight_payload=preflight.to_json(),
    ) == metadata


@pytest.mark.parametrize("changed_input", ["source", "metadata"])
def test_action_recheck_rejects_changed_source_or_metadata_fingerprint(
    tmp_path: Path,
    changed_input: str,
) -> None:
    source = _one_image_source(tmp_path)
    image = source / "plate_a.tif"
    metadata = _metadata_csv(tmp_path, "plate_a")
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = metadata_payload_from_path(sandbox, metadata)
    preflight = build_metadata_preflight(sandbox, str(source), payload)
    if changed_input == "source":
        image.write_bytes(b"changed-source")
    else:
        metadata.write_text(
            f"{METADATA.IMAGE_NAME},Treatment\nplate_a,changed\n",
            encoding="utf-8",
        )

    with pytest.raises(RunRequestSafetyError, match="changed after preflight"):
        recheck_metadata_selection(
            sandbox,
            input_dir=str(source),
            metadata_payload=payload,
            choice="omit",
            acknowledgement=[],
            preflight_payload=preflight.to_json(),
        )
