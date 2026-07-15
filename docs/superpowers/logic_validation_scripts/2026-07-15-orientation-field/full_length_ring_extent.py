"""Validate the full-length ring boundary invariant independently."""

from __future__ import annotations

import numpy as np


def full_length_outer_radius(
    inner_radius: float,
    object_extent_radius: float,
    ring_width: float,
) -> tuple[float, int]:
    """Return the first complete ring boundary beyond the object extent.

    Args:
        inner_radius: Inoculum exclusion radius in pixels.
        object_extent_radius: Farthest detected object radius in pixels.
        ring_width: Width of every radial ring in pixels.

    Returns:
        The exclusive outer radius and number of full-width rings.
    """
    radial_span = np.nextafter(object_extent_radius, np.inf) - inner_radius
    n_rings = max(1, int(np.ceil(radial_span / ring_width)))
    return inner_radius + n_rings * ring_width, n_rings


def validate_full_length_ring_extent() -> None:
    """Prove extent coverage, equal widths, and symmetry independence."""
    cases = (
        (10.0, 10.1, 8.0),
        (10.0, 18.0, 8.0),
        (17.25, 42.634, 8.0),
        (31.0, 186.9, 8.0),
        (4.5, 100.0, 3.25),
    )
    for inner_radius, object_extent_radius, ring_width in cases:
        outer_radius, n_rings = full_length_outer_radius(
            inner_radius,
            object_extent_radius,
            ring_width,
        )
        assert n_rings >= 1
        assert outer_radius > object_extent_radius
        assert outer_radius - object_extent_radius <= ring_width + 1e-12
        assert np.isclose(
            outer_radius,
            inner_radius + n_rings * ring_width,
            rtol=0.0,
            atol=1e-12,
        )

        object_distances = np.linspace(
            inner_radius,
            object_extent_radius,
            257,
        )
        selector = (
            (object_distances >= inner_radius)
            & (object_distances < outer_radius)
        )
        assert selector.all()

        for symmetric_radius in (
            inner_radius + 0.5,
            outer_radius,
            outer_radius * 2.0,
        ):
            old_symmetric_cap = min(object_extent_radius, symmetric_radius)
            repeated_outer, repeated_n_rings = full_length_outer_radius(
                inner_radius,
                object_extent_radius,
                ring_width,
            )
            assert repeated_outer == outer_radius
            assert repeated_n_rings == n_rings
            if symmetric_radius < object_extent_radius:
                assert old_symmetric_cap < object_extent_radius


if __name__ == "__main__":
    validate_full_length_ring_extent()
    print("PASS: full-length rings cover the object independently of symmetry")
