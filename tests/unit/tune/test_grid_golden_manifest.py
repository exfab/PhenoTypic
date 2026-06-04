from __future__ import annotations

import json
from pathlib import Path

GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "fixtures/tune/grid_golden_manifest.json"
)


def test_golden_exists_and_is_stable():
    assert GOLDEN.exists(), "run scripts/capture_grid_golden_manifest.py"
    manifest = json.loads(GOLDEN.read_text())
    # The Phase-1 GridStrategy regression lock compares against this exact file.
    assert manifest["total_pipelines"] == 6


def test_golden_matches_fresh_generation_while_sweep_exists():
    # Belt-and-suspenders: while sweep still exists, the committed golden must
    # equal a fresh generation (so we know it wasn't hand-edited). Phase 1's
    # GridStrategy will be locked to this golden after sweep is deleted.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from capture_grid_golden_manifest import build_golden_config

    from phenotypic.sweep import generate_sweep_manifest

    fresh = generate_sweep_manifest(build_golden_config())
    committed = json.loads(GOLDEN.read_text())
    assert json.loads(json.dumps(fresh, sort_keys=True)) == committed
