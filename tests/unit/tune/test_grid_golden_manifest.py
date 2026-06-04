from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "fixtures/tune/grid_golden_manifest.json"
)


def _strip_versions(obj: Any) -> Any:
    """Recursively drop ``version`` keys so the comparison survives bumps.

    ``generate_sweep_manifest`` stamps ``phenotypic.__version__`` into the
    manifest (top level + every pipeline config). The golden's structural
    content is what the Phase-1 lock cares about, not the version string, so
    we exclude ``version`` before asserting equality — otherwise a routine
    version bump would turn this test red with no behavioral change.
    """
    if isinstance(obj, dict):
        return {
            k: _strip_versions(v) for k, v in obj.items() if k != "version"
        }
    if isinstance(obj, list):
        return [_strip_versions(v) for v in obj]
    return obj


def test_golden_exists_and_is_stable():
    assert GOLDEN.exists(), "run scripts/capture_grid_golden_manifest.py"
    manifest = json.loads(GOLDEN.read_text(encoding="utf-8"))
    # The Phase-1 GridStrategy regression lock compares against this exact file.
    assert manifest["total_pipelines"] == 6


def test_golden_matches_fresh_generation_while_sweep_exists():
    # Belt-and-suspenders: while sweep still exists, the committed golden must
    # equal a fresh generation (so we know it wasn't hand-edited). Phase 1's
    # GridStrategy will be locked to this golden after sweep is deleted.
    # Compared modulo the volatile ``version`` stamps (see _strip_versions).
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    from capture_grid_golden_manifest import build_golden_config

    from phenotypic.sweep import generate_sweep_manifest

    fresh = generate_sweep_manifest(build_golden_config())
    committed = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert _strip_versions(json.loads(json.dumps(fresh))) == _strip_versions(
        committed
    )
