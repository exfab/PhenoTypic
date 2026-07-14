"""Public API and optional-import checks for reviewed fungi method ports."""

from __future__ import annotations

import json
import subprocess
import sys


def test_reviewed_fungi_ports_are_publicly_exported() -> None:
    """Every G3-approved API is reachable only through a public package."""
    from phenotypic import analysis, detect
    from phenotypic.sdk_ import reconnect

    expected_reconnect = {
        "ClarkRollingHoughResult",
        "NFAResult",
        "RorpoResult",
        "TrickTrackCAResult",
        "app2_gwdt_cost",
        "binomial_nfa",
        "clark_rolling_hough",
        "grey_weighted_distance",
        "rorpo",
        "tensor_vote",
        "tricktrack_ca",
    }

    assert expected_reconnect == set(reconnect.__all__)
    assert {"PersistencePairsResult", "cubical_persistence"} <= set(
        analysis.__all__
    )
    assert "FilFinderDetector" in detect.__all__
    for name in expected_reconnect:
        assert getattr(reconnect, name).__module__.startswith(
            "phenotypic.sdk_.reconnect."
        )
    assert analysis.cubical_persistence.__module__ == (
        "phenotypic.analysis._cubical_persistence"
    )
    assert detect.FilFinderDetector.__module__ == (
        "phenotypic.detect._filfinder_detector"
    )


def test_public_imports_do_not_load_topology_dependencies() -> None:
    """Optional runtime packages remain absent until an algorithm is executed."""
    script = """
import json
import sys

import phenotypic
before = set(sys.modules)
from phenotypic import analysis, detect
from phenotypic.sdk_ import reconnect
assert analysis.cubical_persistence
assert detect.FilFinderDetector
assert reconnect.rorpo
added = set(sys.modules) - before
blocked = {"astropy", "fil_finder", "gudhi"}
print(json.dumps(sorted(name for name in added if name.split(".")[0] in blocked)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout) == []
