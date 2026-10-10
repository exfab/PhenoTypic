"""Output-equivalence evidence for the `no_convergence_stop_lines` mutant.

Drives shipped code, so it lives beside the plan (not in logic_validation_scripts/). The mutant
is built in memory from `mutants.json`; no file is touched. It compares `level_lines` of the
shipped kernel against the mutant, and counts the cases in which the mutant actually performed
more refit solves (`np.linalg.solve`) than the original, so a zero difference cannot come from
a code path never reached.

The mutant is killed by a spy test on cost (a converged row is not refit); this script records
that its OUTPUT differs from the original only by rounding (the first extra refit swaps the
`lstsq` coefficients for the normal-equation solution; further rounds are a fixed point).
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
KERNELS = ROOT / "src/phenotypic/enhance/_poly_surface_kernels.py"
TABLE = Path(__file__).with_name("mutants.json")
TOLERANCE = 1e-9


def _load(source: str, name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    exec(compile(source, name, "exec"), module.__dict__)
    return module


def _count_solves(module: types.ModuleType, z: np.ndarray, kwargs: dict) -> tuple[np.ndarray, int]:
    calls = []
    real = np.linalg.solve

    def spy(*args, **kw):
        calls.append(1)
        return real(*args, **kw)

    np.linalg.solve = spy
    try:
        out = module.level_lines(z, **kwargs)
    finally:
        np.linalg.solve = real
    return out, len(calls)


def measure_equivalence() -> int:
    source = KERNELS.read_text()
    mutant = next(m for m in json.loads(TABLE.read_text()) if m["name"] == "no_convergence_stop_lines")
    assert source.count(mutant["old"]) == 1
    original = _load(source, "original")
    mutated = _load(source.replace(mutant["old"], mutant["new"]), "mutated")
    rng = np.random.default_rng(0)
    worst, cases, extra_refit_cases = 0.0, 0, 0
    for _ in range(400):
        height, width = int(rng.integers(2, 40)), int(rng.integers(6, 80))
        order = int(rng.integers(0, min(4, width - 1)))
        z = rng.standard_t(rng.choice([1, 2, 5]), (height, width)) + rng.normal(0, 1, (height, 1))
        for clip_sigma in (1.0, 1.5, 3.0):
            for max_iter in (1, 3, 10, 50):
                kwargs = dict(line_order=order, fit="robust", clip_sigma=clip_sigma, max_iter=max_iter)
                out_o, solves_o = _count_solves(original, z, kwargs)
                out_m, solves_m = _count_solves(mutated, z, kwargs)
                worst = max(worst, float(np.abs(out_o - out_m).max()))
                extra_refit_cases += solves_m > solves_o
                cases += 1
    print(f"{cases} cases; mutant did extra refit solves in {extra_refit_cases}; "
          f"max |original - mutant| = {worst:.3e}")
    return 0 if worst < TOLERANCE and extra_refit_cases > 0 else 1


if __name__ == "__main__":
    sys.exit(measure_equivalence())
