# tests/unit/detect/test_filamentous_fungi_regression.py
"""Golden characterization tests pinning FilamentousFungiDetector's objmap.

Two goldens:
- ``FIXTURE_PSEUDO`` (frozen, pre-reconnect-scope): FFD with ``reconnect_scope="pseudo"``
  must reproduce it bit-for-bit — the legacy-behavior lock.
- ``FIXTURE_BRANCHES``: FFD with the default ``reconnect_scope="branches"``.
On the synthetic plate the two may be byte-identical (no reachable disconnected fragments);
that is the correct, reassuring result — branches only adds structure when a bridgeable
fragment exists. The behavioral proof lives in the TwoK severed-hypha test.
"""
from pathlib import Path

import numpy as np

from phenotypic.data import load_synth_filamentous_plate
from phenotypic.detect import FilamentousFungiDetector, OtsuDetector

_FIX = Path(__file__).parent.parent.parent / "fixtures"
FIXTURE_PSEUDO = _FIX / "filamentous_fungi_regression_objmap.npy"
FIXTURE_BRANCHES = _FIX / "filamentous_fungi_regression_objmap_branches.npy"


def _run(reconnect_scope: str) -> np.ndarray:
    image = load_synth_filamentous_plate().copy()
    detector = FilamentousFungiDetector(
        inoculum_detector=OtsuDetector(ignore_zeros=True), reconnect_scope=reconnect_scope,
    )
    return np.asarray(detector.apply(image, inplace=False).objmap[:])


def test_pseudo_scope_matches_legacy_golden():
    # A missing fixture must FAIL loudly, not skip.
    assert FIXTURE_PSEUDO.exists(), f"Legacy golden missing: {FIXTURE_PSEUDO}."
    expected = np.load(FIXTURE_PSEUDO)
    actual = _run("pseudo")
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.array_equal(actual, expected), (
        f"reconnect_scope='pseudo' is NOT bit-identical to the legacy golden: "
        f"{int((actual != expected).sum())} pixels differ"
    )


def test_branches_default_matches_golden():
    assert FIXTURE_BRANCHES.exists(), (
        f"Branches golden missing: {FIXTURE_BRANCHES}. Regenerate with "
        f"`uv run python -m tests.unit.detect.test_filamentous_fungi_regression`."
    )
    expected = np.load(FIXTURE_BRANCHES)
    actual = _run("branches")
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.array_equal(actual, expected), (
        f"objmap changed: {int((actual != expected).sum())} pixels differ"
    )


if __name__ == "__main__":  # regenerate the BRANCHES default only; the pseudo golden is frozen.
    FIXTURE_BRANCHES.parent.mkdir(parents=True, exist_ok=True)
    np.save(FIXTURE_BRANCHES, _run("branches"))
    print(f"wrote {FIXTURE_BRANCHES}")
