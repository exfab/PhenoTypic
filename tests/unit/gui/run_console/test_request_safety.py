"""Behavioral tests for Run output and metadata request authority."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from phenotypic._cli._cli_output_manager import join_metadata
from phenotypic.gui.run_console._request_safety import (
    RunRequestSafetyError,
    build_metadata_preflight,
    confirm_output_target,
    recheck_metadata_selection,
    validate_output_confirmation,
)
from phenotypic.gui.shell._metadata_context import metadata_payload_from_path
from phenotypic.gui.shell._sandbox import SandboxRoot
from phenotypic.schema import EXPERIMENT_METADATA, METADATA, header_to_module


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
    assert preflight.duplicate_key_count == 1
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


def test_preflight_uses_every_source_level_production_join_key(
    tmp_path: Path,
) -> None:
    """A matching image name cannot hide a mismatched dataset join key."""
    source = _one_image_source(tmp_path)
    metadata = tmp_path / "metadata.csv"
    image_key = str(METADATA.IMAGE_NAME)
    dataset_key = str(EXPERIMENT_METADATA.DATASET)
    metadata.write_text(
        f"{image_key},{dataset_key},Treatment\n"
        "plate_a,wrong-dataset,control\n",
        encoding="utf-8",
    )
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = metadata_payload_from_path(sandbox, metadata)

    preflight = build_metadata_preflight(sandbox, str(source), payload)
    production_result = join_metadata(
        pl.DataFrame(
            {
                image_key: ["plate_a"],
                dataset_key: [source.name],
                "Shape_Area": [1.0],
            }
        ),
        metadata,
    )

    assert set(preflight.join_columns) == {image_key, dataset_key}
    assert preflight.compatibility == "warning"
    assert preflight.matched_source_count == 0
    assert preflight.unmatched_source_count == 1
    assert preflight.metadata_only_count == 1
    assert production_result.height == 0


def test_preflight_duplicate_risk_uses_full_production_key_grain(
    tmp_path: Path,
) -> None:
    """Repeated image names are not duplicates when the full keys differ."""
    source = _one_image_source(tmp_path)
    metadata = tmp_path / "metadata.csv"
    image_key = str(METADATA.IMAGE_NAME)
    dataset_key = str(EXPERIMENT_METADATA.DATASET)
    metadata.write_text(
        f"{image_key},{dataset_key},Treatment\n"
        f"plate_a,{source.name},control\n"
        "plate_a,other-dataset,treated\n",
        encoding="utf-8",
    )
    sandbox = SandboxRoot.from_path(tmp_path)
    payload = metadata_payload_from_path(sandbox, metadata)

    preflight = build_metadata_preflight(sandbox, str(source), payload)

    assert set(preflight.join_columns) == {image_key, dataset_key}
    assert preflight.duplicate_key_count == 0
    assert preflight.matched_source_count == 1
    assert preflight.metadata_only_count == 1

    metadata.write_text(
        f"{image_key},{dataset_key},Treatment\n"
        f"plate_a,{source.name},control\n"
        f"plate_a,{source.name},treated\n",
        encoding="utf-8",
    )
    refreshed = build_metadata_preflight(
        sandbox,
        str(source),
        metadata_payload_from_path(sandbox, metadata),
    )
    assert refreshed.duplicate_key_count == 1


def test_preflight_never_calls_measurement_level_join_keys_compatible(
    tmp_path: Path,
) -> None:
    """Grid keys require post-measurement verification and stay a warning."""
    source = _one_image_source(tmp_path)
    metadata = tmp_path / "metadata.csv"
    image_key = str(METADATA.IMAGE_NAME)
    metadata.write_text(
        f"{image_key},Grid_RowNum,Treatment\nplate_a,999,control\n",
        encoding="utf-8",
    )
    sandbox = SandboxRoot.from_path(tmp_path)

    preflight = build_metadata_preflight(
        sandbox,
        str(source),
        metadata_payload_from_path(sandbox, metadata),
    )

    assert preflight.matched_source_count == 1
    assert preflight.compatibility == "warning"
    assert preflight.unverified_join_columns == ("Grid_RowNum",)
    assert any(
        "production join will use" in warning
        for warning in preflight.warnings
    )


@pytest.mark.parametrize(
    "custom_key",
    ["ExternalMeasure_BatchKey", "Metadata_CustomBatchKey"],
)
def test_preflight_warns_for_unregistered_custom_measurement_key(
    tmp_path: Path,
    custom_key: str,
) -> None:
    """An external qualified key can join even when absent from the schema."""
    source = _one_image_source(tmp_path)
    metadata = tmp_path / "metadata.csv"
    image_key = str(METADATA.IMAGE_NAME)
    assert custom_key not in header_to_module()
    metadata.write_text(
        f"{image_key},{custom_key},Treatment\n"
        "plate_a,metadata-value,control\n",
        encoding="utf-8",
    )
    sandbox = SandboxRoot.from_path(tmp_path)

    preflight = build_metadata_preflight(
        sandbox,
        str(source),
        metadata_payload_from_path(sandbox, metadata),
    )
    production_result = join_metadata(
        pl.DataFrame(
            {
                image_key: ["plate_a"],
                custom_key: ["measurement-value"],
                "Shape_Area": [1.0],
            }
        ),
        metadata,
    )

    assert preflight.matched_source_count == 1
    assert preflight.compatibility == "warning"
    assert preflight.unverified_join_columns == (custom_key,)
    assert production_result.height == 0


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
