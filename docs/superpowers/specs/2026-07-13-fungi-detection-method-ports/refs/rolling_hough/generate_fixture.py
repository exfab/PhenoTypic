"""Generate all-output fixtures from pinned Clark and FilFinder source files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import tempfile
from typing import Any

import numpy as np

import source_contract_probe


REFERENCE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = REFERENCE_DIRECTORY.parents[5]
FIXTURE_DIRECTORY = REPOSITORY_ROOT / "tests/fixtures/reconnect/rolling_hough"
FIXTURE_PATH = FIXTURE_DIRECTORY / "clark_rht_source.npz"
MANIFEST_PATH = FIXTURE_DIRECTORY / "manifest.json"
CLARK_REVISION = "4d06f9fa4cafe9022011a0bec0315390d7e23c39"
FILFINDER_REVISION = "22539cf2176ad9b717658652e8da749158597f4d"


def _canonical_content_hash(
    fixture: dict[str, np.ndarray], required_keys: list[str]
) -> str:
    """Hash NPZ variables independently of ZIP metadata and key insertion order."""
    digest = hashlib.sha256()
    for key in required_keys:
        value = np.asarray(fixture[key])
        digest.update(key.encode("utf-8") + b"\0")
        digest.update(struct.pack("<Q", value.ndim))
        for extent in value.shape:
            digest.update(struct.pack("<Q", extent))
        if value.dtype.kind in "US":
            digest.update(b"utf8\0")
            for item in value.reshape(-1, order="C"):
                encoded = str(item).encode("utf-8")
                digest.update(struct.pack("<Q", len(encoded)))
                digest.update(encoded)
        else:
            canonical_dtype = value.dtype.newbyteorder("<")
            canonical = np.ascontiguousarray(value.astype(canonical_dtype, copy=False))
            digest.update(canonical_dtype.str.encode("ascii") + b"\0")
            digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _full_pipeline_cases() -> list[tuple[str, np.ndarray, int, int, float]]:
    """Construct deterministic images spanning source preprocessing and masks."""
    horizontal = np.zeros((17, 19), dtype=np.float64)
    horizontal[8, 2:17] = 5.0
    horizontal[5, 4:15] = 1.0

    crossing = np.zeros((19, 19), dtype=np.float64)
    crossing[9, 2:17] = 4.0
    crossing[2:17, 9] = 4.0
    for index in range(3, 16):
        crossing[index, index] += 2.0

    gapped = np.zeros((21, 23), dtype=np.float64)
    gapped[10, 3:20] = 3.0
    gapped[10, 10:13] = 0.0
    gapped[np.arange(4, 17), np.arange(5, 18)] += 1.5

    nonfinite = np.zeros((23, 25), dtype=np.float64)
    nonfinite[1, 2:23] = 4.0
    nonfinite[11, 2:23] = 2.5
    nonfinite[11, 12] = np.nan
    nonfinite[5, 5] = np.inf

    constant = np.full((15, 17), 7.25, dtype=np.float64)

    return [
        ("horizontal-asymmetric", horizontal, 5, 1, 0.60),
        ("crossing-nondefault", crossing, 7, 2, 0.55),
        ("gapped-diagonal", gapped, 7, 1, 0.50),
        ("border-and-nonfinite", nonfinite, 5, 1, 0.70),
        ("constant", constant, 5, 1, 0.70),
    ]


def _source_pipeline_case(
    clark: Any,
    image: np.ndarray,
    window_diameter: int,
    smoothing_radius: int,
    fraction: float,
) -> dict[str, np.ndarray]:
    """Call source helpers and instrument every numerical pipeline stage."""
    clark.BUFFER = False
    clark.PROGRESS = False
    smoothing_mask, window_mask = clark.getMask(
        image, smr=smoothing_radius, wlen=window_diameter
    )
    smoothing_kernel = clark.circ_kern(2 * smoothing_radius + 1)
    correlated = clark.scipy.ndimage.filters.correlate(image, smoothing_kernel)
    smoothed = correlated / np.sum(smoothing_kernel)
    unsharp = image - smoothed
    bitmask = clark.umask(image, radius=smoothing_radius, smr_mask=smoothing_mask)

    theta_count = clark.ntheta_w(window_diameter)
    theta = np.linspace(0.0, np.pi, theta_count, endpoint=False)
    center_lines = clark.all_thetas(window_diameter, theta, True)
    circular_window = clark.circ_kern(window_diameter)
    support_counts = clark.fast_hough(circular_window, center_lines)

    height, width = image.shape
    radius = window_diameter // 2
    raw_counts = np.zeros((height, width, theta_count), dtype=np.int64)
    threshold_residual = np.zeros((height, width, theta_count), dtype=np.float64)
    for row, column in zip(*np.nonzero(window_mask), strict=True):
        window = bitmask[
            row - radius : row + radius + 1,
            column - radius : column + radius + 1,
        ]
        counts = clark.fast_hough(window, center_lines)
        residual = np.true_divide(counts, support_counts) - fraction
        residual *= np.greater_equal(residual, 0.0)
        raw_counts[row, column] = counts
        threshold_residual[row, column] = residual

    valid = np.any(threshold_residual, axis=2)
    raw_response = np.sum(threshold_residual, axis=2)
    derived_orientation = np.full(image.shape, np.nan, dtype=np.float64)
    for row, column in zip(*np.nonzero(valid), strict=True):
        derived_orientation[row, column] = clark.theta_rht(
            threshold_residual[row, column], True
        )

    source_error = ""
    with tempfile.TemporaryDirectory(prefix="a09-clark-window-step-") as temporary:
        source_path = Path(temporary) / "source_output.npz"
        try:
            succeeded = clark.window_step(
                image,
                window_diameter,
                fraction,
                smoothing_radius,
                True,
                smoothing_mask,
                window_mask,
                str(source_path),
                "",
                "",
            )
            if not succeeded:
                raise RuntimeError("pinned Clark window_step returned failure")
            with np.load(source_path, allow_pickle=False) as source_output:
                sparse_hi = np.asarray(source_output["hi"])
                sparse_hj = np.asarray(source_output["hj"])
                sparse_residual = np.asarray(source_output["hthets"])
                source_backprojection = np.asarray(source_output["backproj"])
        except IndexError as error:
            if np.any(valid):
                raise
            source_error = f"{type(error).__name__}: {error}"
            sparse_hi = np.empty(0, dtype=np.int64)
            sparse_hj = np.empty(0, dtype=np.int64)
            sparse_residual = np.empty(0, dtype=np.float64)
            source_backprojection = np.empty((0, 0), dtype=np.float64)

    expected_rows, expected_columns = np.nonzero(valid)
    np.testing.assert_array_equal(sparse_hi, expected_columns)
    np.testing.assert_array_equal(sparse_hj, expected_rows)
    if expected_rows.size:
        np.testing.assert_allclose(
            sparse_residual,
            threshold_residual[expected_rows, expected_columns],
            rtol=0.0,
            atol=0.0,
        )
    else:
        assert sparse_residual.size == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        attempted_backprojection = raw_response / np.max(raw_response)
    if not source_error:
        np.testing.assert_allclose(
            source_backprojection,
            attempted_backprojection,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )

    return {
        "image": image,
        "window_diameter": np.array(window_diameter, dtype=np.int64),
        "smoothing_radius": np.array(smoothing_radius, dtype=np.int64),
        "coherence_fraction": np.array(fraction, dtype=np.float64),
        "smoothing_kernel": smoothing_kernel,
        "smoothing_mask": np.asarray(smoothing_mask, dtype=bool),
        "window_mask": np.asarray(window_mask, dtype=bool),
        "correlated": correlated,
        "smoothed": smoothed,
        "unsharp": unsharp,
        "bitmask": bitmask,
        "theta": theta,
        "circular_window": circular_window,
        "center_lines": center_lines,
        "support_counts": support_counts,
        "raw_counts": raw_counts,
        "threshold_residual": threshold_residual,
        "accepted_bins": threshold_residual > 0.0,
        "valid": valid,
        "raw_response": raw_response,
        "derived_orientation": derived_orientation,
        "source_sparse_hi": sparse_hi,
        "source_sparse_hj": sparse_hj,
        "source_sparse_residual": sparse_residual,
        "source_backprojection": source_backprojection,
        "source_attempted_backprojection": attempted_backprojection,
        "source_error": np.array(source_error),
    }


def _local_window_templates(clark: Any) -> dict[str, np.ndarray]:
    """Capture exact local counts, supports, equality gate, and axial angles."""
    diameter = 11
    radius = diameter // 2
    theta = np.linspace(0.0, np.pi, clark.ntheta_w(diameter), endpoint=False)
    center_lines = clark.all_thetas(diameter, theta, True)
    support = clark.fast_hough(clark.circ_kern(diameter), center_lines)
    templates: dict[str, np.ndarray] = {}
    windows: dict[str, np.ndarray] = {}
    windows["horizontal"] = np.zeros((diameter, diameter), dtype=np.int64)
    windows["horizontal"][radius, :] = 1
    windows["vertical"] = np.zeros((diameter, diameter), dtype=np.int64)
    windows["vertical"][:, radius] = 1
    windows["diagonal"] = np.eye(diameter, dtype=np.int64)
    windows["crossing"] = windows["horizontal"] + windows["vertical"]
    windows["gap"] = windows["horizontal"].copy()
    windows["gap"][radius, radius - 1 : radius + 2] = 0
    windows["circle"] = clark.circ_kern(diameter)

    templates["theta"] = theta
    templates["support_counts"] = support
    for name, window in windows.items():
        counts = clark.fast_hough(window, center_lines)
        equality_residual = np.true_divide(counts, support) - 1.0
        equality_residual *= np.greater_equal(equality_residual, 0.0)
        templates[f"{name}_window"] = window
        templates[f"{name}_counts"] = counts
        templates[f"{name}_fraction_one_residual"] = equality_residual
        templates[f"{name}_source_angle"] = np.array(
            clark.theta_rht(counts.astype(np.float64), True), dtype=np.float64
        )
    return templates


def _filfinder_templates(filfinder: Any) -> dict[str, np.ndarray]:
    """Capture the stable test-only oracle on simple binary skeletons."""
    output: dict[str, np.ndarray] = {}
    for name in ("horizontal", "vertical", "diagonal"):
        mask = np.zeros((25, 25), dtype=bool)
        if name == "horizontal":
            mask[12, 5:20] = True
        elif name == "vertical":
            mask[5:20, 12] = True
        else:
            indices = np.arange(5, 20)
            mask[indices, indices] = True
        theta, response, quantiles = filfinder.rht(mask, radius=5, ntheta=18)
        output[f"{name}_mask"] = mask
        output[f"{name}_theta"] = theta
        output[f"{name}_response"] = response
        output[f"{name}_quantiles"] = np.asarray(quantiles, dtype=np.float64)
    return output


def generate_source_fixture() -> None:
    """Run pinned sources, save all outputs, and write a canonical manifest."""
    clark = source_contract_probe._load_clark_source()
    filfinder = source_contract_probe._load_source_module(
        "pinned_filfinder_fixture",
        REFERENCE_DIRECTORY
        / "source_filfinder"
        / "fil_finder"
        / "rollinghough.py",
    )

    fixture: dict[str, np.ndarray] = {
        "clark_revision": np.array(CLARK_REVISION),
        "filfinder_revision": np.array(FILFINDER_REVISION),
    }
    for case_index, (name, image, diameter, radius, fraction) in enumerate(
        _full_pipeline_cases(), start=1
    ):
        prefix = f"c{case_index:02d}_"
        fixture[prefix + "name"] = np.array(name)
        for field, value in _source_pipeline_case(
            clark, image, diameter, radius, fraction
        ).items():
            fixture[prefix + field] = np.asarray(value)
    for field, value in _local_window_templates(clark).items():
        fixture["local_" + field] = np.asarray(value)
    for field, value in _filfinder_templates(filfinder).items():
        fixture["filfinder_" + field] = np.asarray(value)

    required_keys = sorted(fixture)
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE_PATH, **fixture)
    manifest = {
        "fixture": FIXTURE_PATH.name,
        "canonical_content_sha256": _canonical_content_hash(fixture, required_keys),
        "clark_revision": CLARK_REVISION,
        "filfinder_revision": FILFINDER_REVISION,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "required_keys": required_keys,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    generate_source_fixture()
