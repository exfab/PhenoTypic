"""Verify the A09 source fixture without importing either reference source."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import numpy as np


REFERENCE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = REFERENCE_DIRECTORY.parents[5]
FIXTURE_DIRECTORY = REPOSITORY_ROOT / "tests/fixtures/reconnect/rolling_hough"


def _canonical_hash(fixture: dict[str, np.ndarray], keys: list[str]) -> str:
    """Recompute the manifest's container-independent numeric content hash."""
    digest = hashlib.sha256()
    for key in keys:
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


def verify_source_fixture() -> None:
    """Check fixture integrity and cross-field source invariants exactly."""
    manifest = json.loads(
        (FIXTURE_DIRECTORY / "manifest.json").read_text(encoding="utf-8")
    )
    with np.load(FIXTURE_DIRECTORY / manifest["fixture"], allow_pickle=False) as archive:
        fixture = {key: np.asarray(archive[key]) for key in archive.files}
    required_keys = manifest["required_keys"]
    if sorted(fixture) != required_keys:
        raise AssertionError("A09 fixture key set differs from its manifest")
    if _canonical_hash(fixture, required_keys) != manifest["canonical_content_sha256"]:
        raise AssertionError("A09 fixture canonical content hash mismatch")

    for case_index in range(1, 6):
        prefix = f"c{case_index:02d}_"
        residual = fixture[prefix + "threshold_residual"]
        valid = fixture[prefix + "valid"]
        response = fixture[prefix + "raw_response"]
        orientation = fixture[prefix + "derived_orientation"]
        np.testing.assert_array_equal(valid, np.any(residual, axis=2))
        np.testing.assert_array_equal(
            fixture[prefix + "accepted_bins"], residual > 0.0
        )
        np.testing.assert_allclose(
            response, np.sum(residual, axis=2), rtol=0.0, atol=0.0
        )
        if not np.all(np.isnan(orientation[~valid])):
            raise AssertionError(f"case {case_index}: invalid orientation is not NaN")
        if not np.all(np.isfinite(orientation[valid])):
            raise AssertionError(f"case {case_index}: valid orientation is not finite")

        rows, columns = np.nonzero(valid)
        np.testing.assert_array_equal(
            fixture[prefix + "source_sparse_hi"], columns
        )
        np.testing.assert_array_equal(fixture[prefix + "source_sparse_hj"], rows)
        source_sparse = fixture[prefix + "source_sparse_residual"]
        if rows.size:
            np.testing.assert_allclose(
                source_sparse, residual[rows, columns], rtol=0.0, atol=0.0
            )
            if str(fixture[prefix + "source_error"]) != "":
                raise AssertionError(f"case {case_index}: unexpected source error")
        elif "IndexError" not in str(fixture[prefix + "source_error"]):
            raise AssertionError(f"case {case_index}: missing empty-output source error")

    constant_prefix = "c05_"
    if np.any(fixture[constant_prefix + "bitmask"]):
        raise AssertionError("constant source case must have an empty bitmask")
    if np.any(fixture[constant_prefix + "raw_response"]):
        raise AssertionError("constant source case must have zero raw response")
    if not np.all(
        np.isnan(fixture[constant_prefix + "source_attempted_backprojection"])
    ):
        raise AssertionError("constant source normalization must be all NaN")

    if fixture["local_support_counts"].shape != (23,):
        raise AssertionError("canonical diameter-11 theta count drifted")
    for name in ("horizontal", "vertical", "diagonal"):
        if fixture[f"filfinder_{name}_theta"].shape != (17,):
            raise AssertionError(f"FilFinder {name} endpoint-drop behavior drifted")


if __name__ == "__main__":
    verify_source_fixture()
    print("A09 Rolling Hough source fixture verification passed")
