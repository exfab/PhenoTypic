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
    Split,
    _dataset_identity,
    _split_subseed,
    derive_split,
    read_split,
    resolve_split,
    write_split,
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


# -- B2: 3-tier partition -----------------------------------------------------


def test_group_whole_heldout():
    # >=2 groups + enough plates → one whole group held out, kind="group".
    groups = ["A", "A", "A", "A", "B", "B", "B", "B"]
    images = _plates(8, groups=groups)
    split = derive_split(
        images, master_seed=0, group_key="Metadata_Group",
        held_out_fraction=0.2, min_heldout_plates=2,
    )
    assert split.kind == "group"
    assert split.group_key == "Metadata_Group"
    assert not split.within_group_caveat
    held_groups = {
        im.get("Metadata_Group") for im in images if im.name in split.held_out
    }
    assert len(held_groups) == 1  # exactly one whole group
    # Held-out plates are precisely that group's plates; calibration is the rest.
    assert set(split.calibration).isdisjoint(split.held_out)
    assert set(split.calibration) | set(split.held_out) == {im.name for im in images}
    assert split.held_out  # non-empty


def test_within_group_heldout_single_group():
    # A single group → fall back to within-group hold-out (flagged weaker).
    groups = ["A"] * 10
    images = _plates(10, groups=groups)
    split = derive_split(
        images, master_seed=0, group_key="Metadata_Group",
        held_out_fraction=0.3, min_heldout_plates=2,
    )
    assert split.kind == "within_group"
    assert split.within_group_caveat is True
    assert len(split.held_out) >= 2
    assert set(split.calibration).isdisjoint(split.held_out)


def test_skip_heldout_data_poor():
    # Fewer than min_heldout_plates → kind="none", all calibration, held_out empty.
    images = _plates(4)
    split = derive_split(
        images, master_seed=0, group_key=None,
        held_out_fraction=0.2, min_heldout_plates=6,
    )
    assert split.kind == "none"
    assert split.held_out == []
    assert split.group_key is None
    assert set(split.calibration) == {im.name for im in images}


def test_no_group_key_uses_within_group_when_data_rich():
    # No grouping metadata but enough plates → within-group hold-out.
    images = _plates(10)
    split = derive_split(
        images, master_seed=0, group_key=None,
        held_out_fraction=0.2, min_heldout_plates=2,
    )
    assert split.kind == "within_group"
    assert split.within_group_caveat is True
    assert len(split.held_out) >= 2


def test_split_is_deterministic():
    groups = ["A", "A", "A", "A", "B", "B", "B", "B"]
    images = _plates(8, groups=groups)
    kwargs = dict(
        master_seed=7, group_key="Metadata_Group",
        held_out_fraction=0.25, min_heldout_plates=2,
    )
    s1 = derive_split(images, **kwargs)
    s2 = derive_split(images, **kwargs)
    assert s1.calibration == s2.calibration
    assert s1.held_out == s2.held_out
    assert s1.kind == s2.kind


def test_split_is_a_frozen_dataclass():
    images = _plates(4)
    split = derive_split(
        images, master_seed=0, group_key=None,
        held_out_fraction=0.2, min_heldout_plates=6,
    )
    assert isinstance(split, Split)
    assert split.dataset_identity == _dataset_identity(images)
    # Frozen: assignment raises.
    try:
        split.kind = "group"  # type: ignore[misc]
    except Exception as exc:  # dataclasses.FrozenInstanceError
        assert "frozen" in type(exc).__name__.lower() or "frozen" in str(exc).lower()
    else:
        raise AssertionError("Split must be a frozen dataclass")


# -- B3: persist + reload + resume reuse --------------------------------------


def test_split_persists_and_reloads(tmp_path):
    groups = ["A", "A", "A", "A", "B", "B", "B", "B"]
    images = _plates(8, groups=groups)
    split = derive_split(
        images, master_seed=3, group_key="Metadata_Group",
        held_out_fraction=0.25, min_heldout_plates=2,
    )
    write_split(tmp_path, split)
    reloaded = read_split(tmp_path)
    assert reloaded == split


def test_read_split_missing_returns_none(tmp_path):
    assert read_split(tmp_path) is None


def test_resume_reuses_persisted_split(tmp_path):
    groups = ["A", "A", "A", "A", "B", "B", "B", "B"]
    images = _plates(8, groups=groups)
    common = dict(
        group_key="Metadata_Group", held_out_fraction=0.25, min_heldout_plates=2,
    )
    first = resolve_split(tmp_path, images, master_seed=1, **common)
    # A second resolve with a DIFFERENT master seed must reuse the persisted split.
    second = resolve_split(tmp_path, images, master_seed=42, **common)
    assert second == first


def test_fresh_run_derives_and_persists(tmp_path):
    images = _plates(8, groups=["A", "A", "A", "A", "B", "B", "B", "B"])
    assert read_split(tmp_path) is None
    split = resolve_split(
        tmp_path, images, master_seed=0, group_key="Metadata_Group",
        held_out_fraction=0.25, min_heldout_plates=2,
    )
    assert read_split(tmp_path) == split
