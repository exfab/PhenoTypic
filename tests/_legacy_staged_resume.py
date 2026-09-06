"""A FROZEN copy of the pre-port, HDF-backed staged resume classifier.

Task 3.4's differential parity test compares the ported classifier against
this. Freezing it is the whole point: a differential test against a classifier
that moves with the code proves nothing, so nothing in here may ever be
"kept in sync" with :mod:`phenotypic._cli._cli_staged_resume`.

Copied verbatim from ``_cli_staged_resume.py`` at commit 3a02bf62 --
``valid_staged_hdf`` (:69), ``staged_hdf_matches_work_id`` (:99),
``classify_staged_image`` (:167) -- plus inlined copies of the two probes it
called that the port removes or repoints: ``sidecar_exists``
(``_cli_sidecar.py``, deleted in Task 3.5) and ``stage3_completion_exists``.

The format-neutral helpers it also calls (``process_only_output_path``,
``valid_image_success``, ``dataset_measurements_dir``, ``progress_dir``) are
imported rather than frozen: they are identical on both sides of the
comparison, so a change to one of them moves both worlds together and parity
still means what it says.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import h5py  # type: ignore[import-untyped]

from phenotypic._cli._cli_process_only import process_only_output_path
from phenotypic.sdk_ import dataset_measurements_dir, progress_dir

ResumeStage = Literal["stage1", "stage2", "stage3", "complete"]


def legacy_hdf_path(output_dir: Path, dataset: str, image_stem: str) -> Path:
    """``<output>/results/<dataset>/hdf/<stem>.h5`` -- frozen, hand-joined.

    Hand-joined on purpose: the fixture that builds the HDF world and this
    frozen classifier must agree with each other and with nothing else.
    """
    return output_dir / "results" / dataset / "hdf" / f"{image_stem}.h5"


def legacy_sidecar_path(output_dir: Path, dataset: str, image_stem: str) -> Path:
    """``<output>/results/<dataset>/objmap/<stem>.npy`` -- frozen."""
    return output_dir / "results" / dataset / "objmap" / f"{image_stem}.npy"


def sidecar_exists(output_dir: Path, dataset: str, image_stem: str) -> bool:
    return legacy_sidecar_path(output_dir, dataset, image_stem).is_file()


def legacy_stage3_marker_path(
    output_dir: Path, dataset: str, image_stem: str
) -> Path:
    """``<output>/.phenotypic/progress/stage3_complete/<ds>/<stem>.json``.

    Hand-joined on purpose, exactly like :func:`legacy_hdf_path` and
    :func:`legacy_sidecar_path`: the fixture that builds the HDF world and this
    frozen classifier must agree with each other and with nothing else.

    Extracted from the body of :func:`stage3_completion_exists` when P3 §6.1
    collapsed the live stage-3 marker into the per-image record. **This is not
    "keeping the frozen classifier in sync"** -- the path it names is unchanged
    and stays unchanged whatever the live module does. It exists because the
    HDF world must now *write* the artifact this function reads: the live
    writer stopped producing this file, so a fixture that still called it would
    leave the frozen side blind on the stage-3 axis, and the parity test would
    report a divergence that is an artifact of the fixture rather than of the
    port.
    """
    return (
        progress_dir(output_dir)
        / "stage3_complete"
        / dataset
        / f"{image_stem}.json"
    )


def stage3_completion_exists(
    output_dir: Path, dataset: str, image_stem: str
) -> bool:
    return legacy_stage3_marker_path(output_dir, dataset, image_stem).is_file()


def valid_staged_hdf(path: Path) -> bool:
    """Return whether *path* contains the image layers Stage 2 requires."""
    try:
        if not path.is_file() or not h5py.is_hdf5(path):
            return False
        with h5py.File(path, "r") as handle:
            schema_version = int(handle.attrs.get("schema_version", 1))
            layers = (
                handle["layers"]
                if schema_version >= 2 and "layers" in handle
                else handle
            )
            detect_name = (
                "detect_mat" if "detect_mat" in layers else "enh_gray"
            )
            names = ("gray", detect_name, "objmap")
            if any(name not in layers for name in names):
                return False
            datasets = [layers[name] for name in names]
            if any(not isinstance(item, h5py.Dataset) for item in datasets):
                return False
            shapes = [item.shape for item in datasets]
            return all(
                len(shape) >= 2 and shape[0] > 0 and shape[1] > 0
                for shape in shapes
            ) and all(shape[:2] == shapes[0][:2] for shape in shapes[1:])
    except (OSError, TypeError, ValueError):
        return False


def staged_hdf_matches_work_id(path: Path, work_id: str) -> bool:
    """Return whether a valid staged HDF is bound to ``work_id``."""
    if not valid_staged_hdf(path):
        return False
    try:
        with h5py.File(path, "r") as handle:
            value = handle.attrs.get("phenotypic_work_id")
            if isinstance(value, bytes):
                value = value.decode("utf-8")
            return value == work_id
    except (OSError, UnicodeDecodeError):
        return False


def classify_staged_image(
    *,
    output_dir: Path,
    dataset: str,
    image: Path,
    input_root: Path,
    process_only_layer: str | None,
    markers_required: bool,
    expected_work_id: str | None = None,
) -> ResumeStage:
    """Return the earliest stage required by one image's durable artifacts."""
    stem = image.stem
    if expected_work_id is not None:
        from phenotypic._cli._cli_completion import valid_image_success

        if valid_image_success(
            output_dir,
            dataset=dataset,
            image_stem=stem,
            work_id=expected_work_id,
        ):
            return "complete"

    if process_only_layer == "objmap":
        terminal = process_only_output_path(
            output_dir, image, input_root, "objmap"
        )
        if terminal.is_file() and expected_work_id is None:
            return "complete"

    hdf = legacy_hdf_path(output_dir, dataset, stem)
    if expected_work_id is not None:
        hdf_valid = staged_hdf_matches_work_id(hdf, expected_work_id)
    else:
        hdf_valid = valid_staged_hdf(hdf)
    if not hdf_valid:
        return "stage1"

    if (
        process_only_layer is None
        and expected_work_id is None
        and stage3_completion_exists(output_dir, dataset, stem)
    ):
        return "complete"
    if (
        process_only_layer is None
        and expected_work_id is not None
        and stage3_completion_exists(output_dir, dataset, stem)
        and (
            dataset_measurements_dir(output_dir, dataset) / f"{stem}.parquet"
        ).is_file()
    ):
        return "stage3"

    sidecar = sidecar_exists(output_dir, dataset, stem)
    parquet = (
        dataset_measurements_dir(output_dir, dataset) / f"{stem}.parquet"
    )
    if (
        process_only_layer is None
        and not markers_required
        and parquet.is_file()
        and not sidecar
    ):
        return "complete"

    return "stage3" if sidecar else "stage2"
