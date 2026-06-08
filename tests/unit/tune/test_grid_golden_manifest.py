from __future__ import annotations

import json
from pathlib import Path

GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "fixtures/tune/grid_golden_manifest.json"
)


def test_golden_exists_and_is_stable():
    assert GOLDEN.exists(), (
        "frozen golden fixture missing from tests/fixtures/tune/ "
        "(it is committed; restore from git history)"
    )
    manifest = json.loads(GOLDEN.read_text(encoding="utf-8"))
    # The Phase-1 GridStrategy regression lock compares against this exact file.
    assert manifest["total_pipelines"] == 6
