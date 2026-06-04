"""One-shot capture of the grid golden manifest (the Phase-1 GridStrategy lock).

Run once, WHILE `phenotypic.sweep` still exists (it is deleted at the end of
tune Phase 1). Writes a frozen `generate_sweep_manifest` output over a
representative conditional (Presence) config. Re-run only to intentionally
regenerate the golden.
"""
from __future__ import annotations

import json
from pathlib import Path

from phenotypic.detect import OtsuDetector
from phenotypic.enhance import GaussianBlur
from phenotypic.sweep import Presence, Sweep, generate_sweep_manifest

GOLDEN = Path(__file__).resolve().parents[1] / (
    "tests/fixtures/tune/grid_golden_manifest.json"
)


def build_golden_config():
    """A conditional config: a Presence (present/absent) + a swept detector."""
    return [
        Presence(GaussianBlur, sigma=(1.0, 2.0)),     # 2 sigmas + absent = 3
        Sweep(OtsuDetector, ignore_zeros=(True, False)),  # 2
    ]  # -> 3 * 2 = 6 pipelines


def main() -> None:
    manifest = generate_sweep_manifest(build_golden_config())
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Wrote {GOLDEN} (total_pipelines={manifest['total_pipelines']})")


if __name__ == "__main__":
    main()
