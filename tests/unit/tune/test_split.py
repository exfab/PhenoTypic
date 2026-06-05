"""4.5p1 B — the robust-eval held-out split (numpy-only, cross-process stable).

B1: dataset identity is an order-independent SHA-256 over plate names, and the
per-dataset sub-seed is derived deterministically from ``(master_seed, identity)``
via ``numpy.random.SeedSequence`` (no global RNG, reproducible across processes).
B2: the 3-tier partition (whole-group / within-group / data-poor skip). B3:
persist + reload + resume reuse.
"""
from __future__ import annotations

import numpy as np

from phenotypic.data import load_synth_yeast_plate
from phenotypic.tune._evaluation._split import (
    _dataset_identity,
    _split_subseed,
)


class _NamedImage:
    """A minimal stand-in carrying ``.name`` and a metadata accessor.

    Only ``name`` + ``metadata.get(col)`` are exercised by the split, so a tiny
    fake keeps these unit tests fast (no full ``Image`` construction / detection).
    """

    def __init__(self, name, group=None, group_key="Metadata_Group"):
        self.name = name
        self._meta = {} if group is None else {group_key: group}

    @property
    def metadata(self):
        return self

    def get(self, key, default=None):
        return self._meta.get(key, default)


def _plates(n, *, groups=None, group_key="Metadata_Group"):
    if groups is None:
        return [_NamedImage(f"plate_{i:02d}") for i in range(n)]
    return [
        _NamedImage(f"plate_{i:02d}", group=groups[i], group_key=group_key)
        for i in range(n)
    ]


# -- B1: identity + sub-seed --------------------------------------------------


def test_dataset_identity_stable():
    a = _plates(5)
    b = list(reversed(_plates(5)))  # same names, different order
    assert _dataset_identity(a) == _dataset_identity(b)
    # Two independent constructions of the same names agree.
    assert _dataset_identity(_plates(5)) == _dataset_identity(_plates(5))


def test_dataset_identity_changes_with_added_plate():
    base = _plates(5)
    grown = _plates(6)
    assert _dataset_identity(base) != _dataset_identity(grown)


def test_subseed_deterministic():
    identity = _dataset_identity(_plates(5))
    s1 = _split_subseed(1234, identity)
    s2 = _split_subseed(1234, identity)
    assert isinstance(s1, np.random.SeedSequence)
    # Same inputs → identical entropy → identical downstream draws.
    assert s1.entropy == s2.entropy
    d1 = np.random.default_rng(s1).integers(0, 1_000_000, size=8)
    d2 = np.random.default_rng(s2).integers(0, 1_000_000, size=8)
    assert np.array_equal(d1, d2)
    # A different master seed → different stream.
    other = _split_subseed(9999, identity)
    assert other.entropy != s1.entropy


def test_subseed_works_on_real_image_names():
    img = load_synth_yeast_plate()
    identity = _dataset_identity([img])
    assert isinstance(identity, str) and len(identity) == 64
    assert isinstance(_split_subseed(0, identity), np.random.SeedSequence)
