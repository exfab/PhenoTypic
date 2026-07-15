# tests/unit/detect/test_filamentous_fungi_regression.py
"""Golden characterization test pinning FilamentousFungiDetector's objmap.

Guards the Phase-3 extraction refactor: the detector's output on a fixed
synthetic plate + fixed config must not change when its Phase 3-5 helpers
move into sdk_.reconnect.
"""
from pathlib import Path

import numpy as np

from phenotypic.data import load_synth_filamentous_plate
from phenotypic.detect import FilamentousFungiDetector, OtsuDetector

FIXTURE = Path(__file__).parent.parent.parent / "fixtures" / "filamentous_fungi_regression_objmap.npy"


def _run_detector() -> np.ndarray:
    image = load_synth_filamentous_plate().copy()
    detector = FilamentousFungiDetector(inoculum_detector=OtsuDetector(ignore_zeros=True))
    result = detector.apply(image, inplace=False)
    return np.asarray(result.objmap[:])


def test_filamentous_fungi_objmap_matches_golden():
    # A missing fixture must FAIL loudly (regenerate via the __main__ block), not skip.
    assert FIXTURE.exists(), (
        f"Golden fixture missing: {FIXTURE}. Regenerate with "
        f"`uv run python -m tests.unit.detect.test_filamentous_fungi_regression`."
    )
    expected = np.load(FIXTURE)
    actual = _run_detector()
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert np.array_equal(actual, expected), (
        f"objmap changed: {int((actual != expected).sum())} pixels differ"
    )


if __name__ == "__main__":  # regeneration entrypoint (run intentionally, then commit the .npy)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.save(FIXTURE, _run_detector())
    print(f"wrote {FIXTURE}")
