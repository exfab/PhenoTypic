"""Independently validate the numerical claims behind oriented tensor voting.

This script intentionally does not import phenotypic or Numba. It reconstructs the
pinned MATLAB source with vectorized NumPy voting fields and uses ``eigvalsh`` rather
than the production helper's closed-form decomposition.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
from scipy.io import loadmat


REPO_ROOT = Path(__file__).resolve().parents[4]
FIXTURE_DIRECTORY = REPO_ROOT / "tests/fixtures/reconnect/tensor_voting"


def source_radius(sigma: float) -> int:
    """Reproduce the source's even-round-then-force-odd support rule."""
    doubled = math.sqrt(-math.log(0.01) * sigma**2) * 2.0
    return math.floor(math.ceil(doubled) / 2.0)


def linton_stick_field(
    tangent: np.ndarray,
    sigma: float,
    radius: int,
) -> np.ndarray:
    """Construct one full source field with vectorized matrix operations."""
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    offset_x, row_offset = np.meshgrid(offsets, offsets)
    offset_y = -row_offset
    tangent = tangent / np.linalg.norm(tangent)
    perpendicular = np.array([-tangent[1], tangent[0]])
    local_x = tangent[0] * offset_x + tangent[1] * offset_y
    local_y = perpendicular[0] * offset_x + perpendicular[1] * offset_y
    local_angle = np.arctan2(local_y, local_x)

    base_angle = math.atan2(tangent[1], tangent[0])
    vote_angle = 2.0 * local_angle + base_angle
    vote_normal = np.stack((-np.sin(vote_angle), np.cos(vote_angle)), axis=-1)

    attenuation_angle = np.abs(local_angle)
    attenuation_angle = np.where(
        attenuation_angle > np.pi / 2.0,
        np.pi - attenuation_angle,
        attenuation_angle,
    )
    attenuation_angle *= 4.0
    distance = np.hypot(local_x, local_y)
    nontrivial_arc = (distance != 0.0) & (attenuation_angle != 0.0)
    arc_length = distance.copy()
    arc_length[nontrivial_arc] = (
        attenuation_angle[nontrivial_arc]
        * distance[nontrivial_arc]
        / np.sin(attenuation_angle[nontrivial_arc])
    )
    curvature = np.zeros_like(distance)
    nonzero_distance = distance != 0.0
    curvature[nonzero_distance] = (
        2.0 * np.sin(attenuation_angle[nonzero_distance]) / distance[nonzero_distance]
    )
    curvature_weight = -16.0 * np.log2(0.1) * (sigma - 1.0) / np.pi**2
    decay = np.exp(
        -(arc_length**2 + curvature_weight * curvature**2) / sigma**2
    )
    decay[attenuation_angle > np.pi / 2.0] = 0.0
    return decay[..., None, None] * np.einsum(
        "...i,...j->...ij", vote_normal, vote_normal
    )


def slow_oracle(
    response: np.ndarray,
    theta: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate source fields using explicit 2x2 tensors and ``eigh``."""
    normal = np.stack((-np.sin(theta), np.cos(theta)), axis=-1)
    accumulator = response[..., None, None] * np.einsum(
        "...i,...j->...ij", normal, normal
    )
    accumulator = accumulator.copy()
    height, width = response.shape
    radius = source_radius(sigma)

    # Transposed flat indices reproduce MATLAB find() column-major ordering.
    for flat_index in np.flatnonzero(response.T > 0.0):
        voter_column, voter_row = np.unravel_index(flat_index, (width, height))
        input_tensor = response[voter_row, voter_column] * np.outer(
            normal[voter_row, voter_column], normal[voter_row, voter_column]
        )
        eigenvalues, eigenvectors = np.linalg.eigh(input_tensor)
        vote_amplitude = eigenvalues[1] - eigenvalues[0]
        principal_normal = eigenvectors[:, 1]
        tangent = np.array([-principal_normal[1], principal_normal[0]])
        field = vote_amplitude * linton_stick_field(tangent, sigma, radius)

        source_row_start = max(0, radius - voter_row)
        source_row_stop = min(2 * radius + 1, height + radius - voter_row)
        source_column_start = max(0, radius - voter_column)
        source_column_stop = min(2 * radius + 1, width + radius - voter_column)
        target_row_start = max(0, voter_row - radius)
        target_row_stop = min(height, voter_row + radius + 1)
        target_column_start = max(0, voter_column - radius)
        target_column_stop = min(width, voter_column + radius + 1)
        accumulator[
            target_row_start:target_row_stop,
            target_column_start:target_column_stop,
        ] += field[
            source_row_start:source_row_stop,
            source_column_start:source_column_stop,
        ]

    eigenvalues = np.linalg.eigvalsh(accumulator)
    return accumulator, eigenvalues[..., 1] - eigenvalues[..., 0], eigenvalues[..., 0]


def gamma_bound(term_count: int, magnitude: np.ndarray, factor: float = 64.0) -> np.ndarray:
    """Return a Higham gamma_n sum bound expanded for transcendental evaluations."""
    unit_roundoff = np.finfo(np.float64).eps / 2.0
    gamma = term_count * unit_roundoff / (1.0 - term_count * unit_roundoff)
    return factor * gamma * np.maximum(np.abs(magnitude), 1.0)


def require(condition: bool, message: str) -> None:
    """Raise a visible assertion instead of silently skipping a claim."""
    if not condition:
        raise AssertionError(message)


def canonical_fixture_hash(
    fixture: dict[str, np.ndarray], required_keys: list[str]
) -> str:
    """Hash decoded content, not MATLAB's timestamp-bearing container header."""
    digest = hashlib.sha256()
    for key in sorted(required_keys):
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


def validate_fixture() -> float:
    """Verify fixture completeness, provenance, tensors, and all numeric outputs."""
    manifest_path = FIXTURE_DIRECTORY / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fixture_path = FIXTURE_DIRECTORY / manifest["fixture"]
    fixture = loadmat(fixture_path)
    keys = {key for key in fixture if not key.startswith("__")}
    require(keys == set(manifest["required_keys"]), "fixture key set is incomplete")
    require(
        canonical_fixture_hash(fixture, manifest["required_keys"])
        == manifest["canonical_content_sha256"],
        "fixture canonical content hash does not match manifest",
    )
    require(
        fixture["source_archive_sha256"][0]
        == manifest["source_archive_sha256"],
        "fixture source revision mismatch",
    )

    response = fixture["response"]
    theta = fixture["theta"]
    sigma = float(fixture["sigma"][0, 0])
    normal = np.stack((-np.sin(theta), np.cos(theta)), axis=-1)
    input_tensor = response[..., None, None] * np.einsum(
        "...i,...j->...ij", normal, normal
    )
    require(np.array_equal(normal[..., 0], fixture["normal_x"]), "normal_x moved")
    require(np.array_equal(normal[..., 1], fixture["normal_y"]), "normal_y moved")
    require(
        np.all(np.abs(input_tensor - fixture["input_tensor"]) <= gamma_bound(2, input_tensor)),
        "input rank-1 tensor mapping moved",
    )

    tensor, stick, ball = slow_oracle(response, theta, sigma)
    maximum_error = 0.0
    for actual, expected, label, term_count in (
        (tensor[..., 0, 0], fixture["accumulated_a"], "accumulated_a", 6),
        (tensor[..., 0, 1], fixture["accumulated_b"], "accumulated_b", 6),
        (tensor[..., 1, 1], fixture["accumulated_d"], "accumulated_d", 6),
        (stick, fixture["stick"], "stick", 6),
        (ball, fixture["ball"], "ball", 6),
    ):
        error = np.abs(actual - expected)
        maximum_error = max(maximum_error, float(np.max(error)))
        require(np.all(error <= gamma_bound(term_count, expected)), f"{label} exceeds bound")

    require(
        np.all(np.abs(tensor - fixture["accumulated_tensor"]) <= gamma_bound(6, tensor)),
        "full accumulated tensor moved",
    )
    fixture_eigenvalues = np.stack((fixture["lambda2"], fixture["lambda1"]), axis=-1)
    require(
        np.all(
            np.abs(np.linalg.eigvalsh(fixture["accumulated_tensor"]) - fixture_eigenvalues)
            <= gamma_bound(4, fixture_eigenvalues)
        ),
        "fixture eigenvalues are inconsistent with its tensor",
    )
    e1 = fixture["e1"]
    e2 = fixture["e2"]
    require(
        np.all(np.abs(np.sum(e1 * e2, axis=-1)) <= gamma_bound(2, np.ones(e1.shape[:2]))),
        "fixture eigenvectors are not orthogonal",
    )
    require(
        int(fixture["window_size"][0, 0]) == 2 * source_radius(sigma) + 1,
        "fixture window size moved",
    )
    return maximum_error


def validate_algebraic_and_geometric_controls() -> None:
    """Check exact tensor algebra and independent geometric behavior."""
    tangent = np.array([1.0, 0.0])
    normal = np.array([0.0, 1.0])
    require(
        np.array_equal(2.0 * np.outer(normal, normal), np.array([[0.0, 0.0], [0.0, 2.0]])),
        "collinear 2nn^T identity failed",
    )
    require(
        np.array_equal(np.outer(tangent, tangent) + np.outer(normal, normal), np.eye(2)),
        "orthogonal tensors must sum to I",
    )

    zero_response = np.zeros((5, 5), dtype=np.float64)
    zero_theta = np.zeros_like(zero_response)
    zero_tensor, zero_stick, zero_ball = slow_oracle(zero_response, zero_theta, 2.25)
    require(
        np.array_equal(zero_tensor, np.zeros((5, 5, 2, 2)))
        and np.array_equal(zero_stick, zero_response)
        and np.array_equal(zero_ball, zero_response),
        "zero response must produce exact zero tensor and saliencies",
    )
    positive_response = zero_response.copy()
    positive_response[2, 2] = 1e-100
    _, positive_stick, _ = slow_oracle(positive_response, zero_theta, 1.0)
    require(
        positive_stick[2, 2] > 0.0
        and abs(positive_stick[2, 2] - 2e-100) <= 32.0 * np.finfo(float).eps * 2e-100,
        "zero-to-positive active-token threshold discontinuity moved",
    )

    field = linton_stick_field(tangent, sigma=2.25, radius=5)
    center = field[5, 5]
    require(np.array_equal(center, np.outer(normal, normal)), "self-vote is not nn^T")
    require(
        np.linalg.norm(field[5, 7]) > np.linalg.norm(field[3, 7]),
        "distance/curvature field does not prefer good continuation",
    )

    response = np.zeros((17, 17), dtype=np.float64)
    theta = np.zeros_like(response)
    response[8, 3] = response[8, 13] = 1.0
    _, stick, _ = slow_oracle(response, theta, 3.0)
    require(stick[8, 8] > stick[6, 8], "gap stick response does not beat lateral control")

    crossing = np.zeros((17, 17), dtype=np.float64)
    crossing_theta = np.zeros_like(crossing)
    crossing[8, 3] = crossing[8, 13] = 1.0
    crossing[3, 8] = crossing[13, 8] = 1.0
    crossing_theta[3, 8] = crossing_theta[13, 8] = np.pi / 2.0
    _, crossing_stick, crossing_ball = slow_oracle(crossing, crossing_theta, 3.0)
    require(
        crossing_ball[8, 8] > crossing_stick[8, 8],
        "orthogonal crossing does not produce ball-dominant saliency",
    )

    shifted = slow_oracle(response, theta + np.pi, 3.0)
    baseline = slow_oracle(response, theta, 3.0)
    for original, periodic in zip(baseline, shifted, strict=True):
        require(
            np.all(np.abs(original - periodic) <= gamma_bound(4, original)),
            "pi periodicity failed",
        )

    scaled = slow_oracle(2.75 * response, theta, 3.0)
    for original, scaled_field in zip(baseline, scaled, strict=True):
        require(
            np.all(np.abs(scaled_field - 2.75 * original) <= gamma_bound(8, scaled_field)),
            "positive linearity failed for fixed active mask",
        )

    rotated = slow_oracle(
        np.rot90(response), np.rot90(theta) + np.pi / 2.0, 3.0
    )
    for original, rotated_field in zip(baseline[1:], rotated[1:], strict=True):
        require(
            np.all(
                np.abs(np.rot90(original, axes=(0, 1)) - rotated_field)
                <= gamma_bound(8, rotated_field)
            ),
            "90-degree rotation covariance failed",
        )

    edge_response = np.zeros((11, 11), dtype=np.float64)
    edge_theta = np.zeros_like(edge_response)
    edge_response[0, 0] = 1.0
    edge_tensor, edge_stick, edge_ball = slow_oracle(edge_response, edge_theta, 2.25)
    require(edge_stick[0, 0] == 2.0, "retained input plus cast self-vote moved")
    require(
        np.array_equal(edge_tensor[-1, -1], np.zeros((2, 2)))
        and edge_stick[-1, -1] == 0.0
        and edge_ball[-1, -1] == 0.0,
        "boundary wrapped instead of clipping",
    )


def validate_tensor_voting_claims() -> None:
    """Run every source and independent numerical claim."""
    maximum_error = validate_fixture()
    validate_algebraic_and_geometric_controls()
    print("Tensor-voting numerical validation passed")
    print("Source: MATLAB Central File Exchange 21051 v1.0.0.0")
    print("Runtime fixture: MATLAB R2023b Update 9")
    print("Assumptions: sparse positive tokens, Cartesian axial tangent theta, raw saliency")
    print("Tolerance: Higham gamma_n sum bound plus Weyl-compatible 64x libm allowance")
    print(f"Maximum fixture error: {maximum_error:.3e}")


if __name__ == "__main__":
    validate_tensor_voting_claims()
