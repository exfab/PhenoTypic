"""Both branches of ``remove_objects_smaller_than``, whatever is installed.

The helper picks its scikit-image call from the installed signature: ``max_size``
on 0.26+, ``min_size`` before. CI syncs the lock, so only one branch ever runs
there. The other branch -- the one a fresh ``pip install`` takes -- would go
untested, and an off-by-one or a dropped ``connectivity`` in it would pass the
whole suite. Here ``skimage.morphology.remove_small_objects`` is replaced by a
stub with each release's semantics, so both branches run on any release and keep
running after the lock moves.

Each stub records how it was called. That catches a helper that bypasses the
stub (for example by importing the function at module level, which the
monkeypatch cannot reach) instead of letting the test pass against the real
function.
"""

from __future__ import annotations

import numpy as np
import pytest
import skimage.morphology
from scipy import ndimage as ndi

from phenotypic.sdk_ import _skimage_compat as compat

_DEPRECATED = object()


def _remove(ar, too_small_of_sizes, connectivity):
    out = ar.copy()
    if out.dtype == bool:
        structure = ndi.generate_binary_structure(ar.ndim, connectivity)
        ccs, _ = ndi.label(ar, structure)
    else:
        ccs = out
    out[too_small_of_sizes(np.bincount(ccs.ravel()))[ccs]] = 0
    return out


def _make_stub(release: str, calls: list[dict]):
    """A ``remove_small_objects`` with ``release``'s signature and semantics."""
    if release == "0.26":

        def stub(
            ar, min_size=_DEPRECATED, connectivity=1, *, max_size=64, out=None
        ):
            assert min_size is _DEPRECATED, "the 0.26 branch passed min_size"
            calls.append({"max_size": max_size, "connectivity": connectivity})
            if max_size == 0:
                return ar.copy()
            return _remove(ar, lambda sizes: sizes <= max_size, connectivity)

    else:

        def stub(ar, min_size=64, connectivity=1, *, out=None):
            calls.append({"min_size": min_size, "connectivity": connectivity})
            if min_size == 0:
                return ar.copy()
            return _remove(ar, lambda sizes: sizes < min_size, connectivity)

    return stub


@pytest.fixture(params=["0.25", "0.26"])
def stubbed_release(request, monkeypatch):
    """Install one release's stub; yield ``(release, calls)``."""
    calls: list[dict] = []
    monkeypatch.setattr(
        skimage.morphology,
        "remove_small_objects",
        _make_stub(request.param, calls),
    )
    compat._remove_small_objects_takes_max_size.cache_clear()
    yield request.param, calls
    compat._remove_small_objects_takes_max_size.cache_clear()


@pytest.mark.parametrize(
    "min_size", [0, 1, 2, 3, 7, 30, 0.5, 2.5, 6.01, 29.999]
)
@pytest.mark.parametrize("as_bool", [False, True])
@pytest.mark.parametrize("connectivity", [1, 2])
def test_each_branch_matches_the_strictly_smaller_rule(
    stubbed_release, min_size, as_bool, connectivity
) -> None:
    """Oracle: zero every component with ``< min_size`` pixels.

    ``connectivity=2`` is where a dropped argument shows: a boolean mask is
    labelled 8-connected by the oracle and 4-connected by a helper that
    forgot to pass it.
    """
    release, calls = stubbed_release
    assert compat._remove_small_objects_takes_max_size() is (release == "0.26")

    rng = np.random.default_rng(20261104)
    structure = ndi.generate_binary_structure(2, connectivity)
    labels, _ = ndi.label(rng.random((48, 48)) > 0.55, structure=structure)
    small = np.bincount(labels.ravel()) < min_size
    small[0] = False
    expected = labels.copy()
    expected[small[labels]] = 0
    ar = labels > 0 if as_bool else labels

    result = compat.remove_objects_smaller_than(
        ar, min_size, connectivity=connectivity
    )

    np.testing.assert_array_equal(
        result, expected > 0 if as_bool else expected
    )
    threshold = (
        {"max_size": compat._max_size_for(min_size)}
        if release == "0.26"
        else {"min_size": min_size}
    )
    assert calls == [{**threshold, "connectivity": connectivity}]
