"""Schema-v2 provenance applications and mutation boundaries."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import phenotypic
from phenotypic import Image, ImagePipeline
from phenotypic._core._provenance import (
    initialize_cli_provenance,
    new_provenance_journal,
    set_provenance_status,
    set_retry_base_length,
    strip_non_reproducible_operation_fields,
    truncate_provenance_to_retry_base,
)
from phenotypic.enhance import BlurGauss


def _image(name: str = "plate.tiff") -> Image:
    pixels = np.arange(20 * 16 * 3, dtype=np.uint8).reshape(20, 16, 3)
    return Image(pixels, name=name)


def _operation(
    sequence: int, *, sigma: float, name: str = "BlurGauss"
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "operation_name": name,
        "operation_class": "phenotypic.enhance._blur_gauss.BlurGauss",
        "phenotypic_version": phenotypic.__version__,
        "parameters": {"sigma": sigma, "axes": [0, 1]},
        "applied_at_utc": "2026-09-01T12:00:00.000Z",
        "duration_seconds": 0.25,
        "pipeline_step_path": ["smooth"],
    }


def test_new_journal_has_only_the_canonical_v2_root_shape() -> None:
    assert new_provenance_journal("/imports/Plate A.TIFF") == {
        "schema_version": 2,
        "status": "complete",
        "original_filename": "Plate A.TIFF",
        "applications": [],
    }


def test_programmatic_pipeline_applications_preserve_history_and_global_sequence(
) -> None:
    pipeline = ImagePipeline(ops={"smooth": BlurGauss(sigma=1.25)})
    source = _image("Plate A.TIFF")

    first = pipeline.apply(source)
    second = pipeline.apply(first)

    journal = second._metadata.provenance_journal
    assert journal["schema_version"] == 2
    assert journal["original_filename"] == "Plate A.TIFF"
    assert [app["kind"] for app in journal["applications"]] == [
        "programmatic",
        "programmatic",
    ]
    assert [app["sequence"] for app in journal["applications"]] == [1, 2]
    assert [app["input_filename"] for app in journal["applications"]] == [
        "Plate A.TIFF",
        "Plate A.TIFF",
    ]
    assert all(
        app["phenotypic_version"] == phenotypic.__version__
        for app in journal["applications"]
    )
    assert [record["sequence"] for record in second.provenance] == [1, 2]
    assert source.provenance == ()
    assert len(first._metadata.provenance_journal["applications"]) == 1


def test_direct_operation_owns_one_programmatic_application() -> None:
    result = BlurGauss(sigma=0.75).apply(_image())

    applications = result._metadata.provenance_journal["applications"]
    assert len(applications) == 1
    assert applications[0]["kind"] == "programmatic"
    assert applications[0]["status"] == "complete"
    assert applications[0]["retry_base_length"] == 0
    assert [record["sequence"] for record in result.provenance] == [1]


def test_apply_with_intermediates_owns_one_programmatic_application() -> None:
    pipeline = ImagePipeline(
        ops={
            "first": BlurGauss(sigma=0.75),
            "second": BlurGauss(sigma=1.25),
        }
    )

    result = pipeline.apply_with_intermediates(_image())

    applications = result.image._metadata.provenance_journal["applications"]
    assert len(applications) == 1
    assert applications[0]["kind"] == "programmatic"
    assert applications[0]["status"] == "complete"
    assert [entry["pipeline_step_path"] for entry in applications[0]["operations"]] == [
        ["first"],
        ["second"],
    ]


def test_cli_initialization_owns_one_typed_application_with_exact_basenames(
    tmp_path: Path,
) -> None:
    pipeline_path = tmp_path / "configs" / "Pipeline Final.JSON"
    pipeline_path.parent.mkdir()
    pipeline_path.write_text("{}", encoding="utf-8")
    image = _image("nested/input/Plate A.TIFF")

    initialize_cli_provenance(
        image,
        pipeline_path,
        kind="full",
        input_filename="/incoming/Plate A.TIFF",
    )

    journal = image._metadata.provenance_journal
    assert journal["original_filename"] == "Plate A.TIFF"
    assert journal["status"] == "in_progress"
    assert journal["applications"] == [
        {
            "sequence": 1,
            "kind": "full",
            "phenotypic_version": phenotypic.__version__,
            "input_filename": "Plate A.TIFF",
            "status": "in_progress",
            "pipeline": {
                "source_path": "Pipeline Final.JSON",
                "sha256": journal["applications"][0]["pipeline"]["sha256"],
            },
            "retry_base_length": 0,
            "operations": [],
        }
    ]


def test_flattened_reads_are_deeply_immutable_across_applications() -> None:
    journal = new_provenance_journal("plate.tiff")
    journal["applications"] = [
        {
            "sequence": 1,
            "kind": "process",
            "phenotypic_version": phenotypic.__version__,
            "input_filename": "plate.tiff",
            "status": "complete",
            "pipeline": None,
            "retry_base_length": 0,
            "operations": [_operation(1, sigma=1.0)],
        },
        {
            "sequence": 2,
            "kind": "full",
            "phenotypic_version": phenotypic.__version__,
            "input_filename": "plate.ome.zarr",
            "status": "complete",
            "pipeline": None,
            "retry_base_length": 0,
            "operations": [_operation(2, sigma=2.0)],
        },
    ]
    image = _image()
    image._metadata.provenance_journal = journal

    assert [entry["sequence"] for entry in image.provenance] == [1, 2]
    with pytest.raises(TypeError, match="read-only"):
        image.provenance[1]["parameters"]["axes"].append(2)
    assert journal["applications"][1]["operations"][0]["parameters"][
        "axes"
    ] == [0, 1]


def test_process_sanitization_recurses_across_every_application() -> None:
    journal = new_provenance_journal("plate.tiff")
    journal["applications"] = [
        {
            "sequence": 1,
            "kind": "process",
            "phenotypic_version": phenotypic.__version__,
            "input_filename": "plate.tiff",
            "status": "complete",
            "pipeline": {
                "source_path": "process.json",
                "sha256": "a" * 64,
            },
            "retry_base_length": 0,
            "operations": [_operation(1, sigma=1.0)],
        },
        {
            "sequence": 2,
            "kind": "full",
            "phenotypic_version": phenotypic.__version__,
            "input_filename": "plate.ome.zarr",
            "status": "complete",
            "pipeline": {
                "source_path": "full.json",
                "sha256": "b" * 64,
            },
            "retry_base_length": 0,
            "operations": [_operation(2, sigma=2.0)],
        },
    ]
    source = deepcopy(journal)

    sanitized = strip_non_reproducible_operation_fields(deepcopy(journal))

    assert [
        application["pipeline"]["source_path"]
        for application in sanitized["applications"]
    ] == ["process.json", "full.json"]
    for application in sanitized["applications"]:
        for operation in application["operations"]:
            assert "applied_at_utc" not in operation
            assert "duration_seconds" not in operation
    assert sanitized["original_filename"] == "plate.tiff"
    assert source == journal


def test_retry_truncation_is_local_to_the_last_application() -> None:
    image = _image()
    journal = new_provenance_journal("plate.tiff")
    first = {
        "sequence": 1,
        "kind": "process",
        "phenotypic_version": phenotypic.__version__,
        "input_filename": "plate.tiff",
        "status": "complete",
        "pipeline": None,
        "retry_base_length": 0,
        "operations": [_operation(1, sigma=1.0)],
    }
    second = {
        "sequence": 2,
        "kind": "full",
        "phenotypic_version": phenotypic.__version__,
        "input_filename": "plate.ome.zarr",
        "status": "staged",
        "pipeline": None,
        "retry_base_length": 1,
        "operations": [
            _operation(2, sigma=2.0),
            _operation(3, sigma=3.0),
        ],
    }
    journal["applications"] = [first, second]
    journal["status"] = "staged"
    image._metadata.provenance_journal = journal

    truncate_provenance_to_retry_base(image)

    assert image._metadata.provenance_journal["applications"][0] == first
    assert [entry["sequence"] for entry in image.provenance] == [1, 2]


@pytest.mark.parametrize(
    "journal",
    [
        {
            "schema_version": 1,
            "status": "complete",
            "pipeline": None,
            "retry_base_length": 0,
            "operations": [],
        },
        {
            "schema_version": 3,
            "status": "complete",
            "original_filename": "plate.tiff",
            "applications": [],
        },
        {
            "schema_version": 2,
            "status": "complete",
            "original_filename": "plate.tiff",
            "applications": [{"kind": "full"}],
        },
    ],
)
def test_every_mutation_seam_refuses_v1_future_and_malformed_v2(
    journal: dict[str, Any],
) -> None:
    image = _image()
    image._metadata.provenance_journal = deepcopy(journal)

    with pytest.raises(ValueError, match="migrat|schema|malformed"):
        set_provenance_status(image, "failed")
    assert image._metadata.provenance_journal == journal


def test_programmatic_apply_refuses_unowned_unfinished_cli_application(
    tmp_path: Path,
) -> None:
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text("{}", encoding="utf-8")
    image = _image()
    initialize_cli_provenance(image, pipeline_path, kind="full")

    with pytest.raises(ValueError, match="cannot start a new provenance application"):
        BlurGauss(sigma=1.0).apply(image, inplace=True)

    assert image._metadata.provenance_journal["applications"][0]["operations"] == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda application: application["pipeline"].update(
            source_path="/private/pipeline.json"
        ),
        lambda application: application["pipeline"].update(
            source_path=r"C:\private\pipeline.json"
        ),
        lambda application: application["operations"].append({"sequence": 1}),
    ],
)
def test_strict_v2_validation_refuses_path_pipeline_and_partial_operation(
    tmp_path: Path,
    mutation: Any,
) -> None:
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_text("{}", encoding="utf-8")
    image = _image()
    initialize_cli_provenance(image, pipeline_path, kind="full")
    application = image._metadata.provenance_journal["applications"][-1]
    mutation(application)

    with pytest.raises(ValueError, match="malformed"):
        set_provenance_status(image, "failed")


def test_retry_base_mutation_also_refuses_v1_without_partial_change() -> None:
    image = _image()
    legacy = {
        "schema_version": 1,
        "status": "complete",
        "pipeline": None,
        "retry_base_length": 0,
        "operations": [],
    }
    image._metadata.provenance_journal = deepcopy(legacy)

    with pytest.raises(ValueError, match="migrat"):
        set_retry_base_length(image, 4)
    assert image._metadata.provenance_journal == legacy
