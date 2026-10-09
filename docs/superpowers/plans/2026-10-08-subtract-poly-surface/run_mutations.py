"""Mutation harness for SubtractPolySurface. Drives shipped code, so it lives beside the plan.

Each mutant is ONE exact text replacement that must match exactly once. Before anything is
touched the harness (a) checks every mutant's target and anchor and (b) runs the three test
files unmutated and refuses to continue if that baseline is red. The original bytes are saved
by full path before mutating, written back in `finally`, and the restored file's sha256 is
checked against the original before the next mutant runs.

Usage: run_mutations.py <mutants.json> [<results.json>]
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TESTS = [
    "tests/unit/enhance/test_poly_surface_kernels.py",
    "tests/unit/enhance/test_subtract_poly_surface.py",
    "tests/unit/enhance/test_subtract_poly_surface_gwyddion.py",
]


def _pytest() -> subprocess.CompletedProcess:
    return subprocess.run(
        ["uv", "run", "pytest", *TESTS, "-q", "--no-header", "-p", "no:randomly",
         "-o", "addopts=", "-m", "not slow", "-rfE"],
        cwd=ROOT, capture_output=True, text=True,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_all(mutants: list[dict]) -> list[str]:
    problems = []
    names = [m["name"] for m in mutants]
    if len(set(names)) != len(names):
        problems.append("duplicate mutant names")
    for m in mutants:
        target = ROOT / m["path"]
        if not target.is_file():
            problems.append(f"{m['name']}: target {target} missing")
            continue
        n = target.read_text().count(m["old"])
        if n != 1:
            problems.append(f"{m['name']}: anchor matched {n} times")
        if m["old"] == m["new"]:
            problems.append(f"{m['name']}: old == new")
    return problems


def run_mutant(path: str, old: str, new: str) -> dict:
    target = (ROOT / path).resolve()
    original = target.read_bytes()
    digest = hashlib.sha256(original).hexdigest()
    started = time.monotonic()
    try:
        target.write_bytes(original.decode().replace(old, new).encode())
        proc = _pytest()
        lines = proc.stdout.splitlines()
        failed = [ln for ln in lines if ln.startswith("FAILED")]
        errors = [ln for ln in lines if ln.startswith("ERROR")]
        result = {
            "status": "KILLED" if proc.returncode != 0 else "SURVIVED",
            "returncode": proc.returncode,
            "n_failed": len(failed),
            "killed_by": [ln.split(" - ")[0] for ln in failed],
            "errors": errors[:5],
            "summary": lines[-1] if lines else "",
        }
    finally:
        target.write_bytes(original)
    result["restored_sha_ok"] = _sha(target) == digest
    result["seconds"] = round(time.monotonic() - started, 1)
    return result


def run_all_mutants(mutants_json: Path, results_json: Path | None) -> int:
    mutants = json.loads(mutants_json.read_text())
    problems = validate_all(mutants)
    if problems:
        print("INVALID MUTANT TABLE (nothing modified):", *problems, sep="\n  ")
        return 2
    baseline = _pytest()
    print("BASELINE:", baseline.stdout.splitlines()[-1] if baseline.stdout else "<no output>")
    if baseline.returncode != 0:
        print(baseline.stdout[-3000:], baseline.stderr[-2000:])
        print("BASELINE RED: refusing to report")
        return 3
    results = {}
    for m in mutants:
        results[m["name"]] = run_mutant(m["path"], m["old"], m["new"])
        r = results[m["name"]]
        print(f"{m['name']}: {r['status']} ({r['n_failed']} failed, {r['seconds']}s)", flush=True)
        if not r["restored_sha_ok"]:
            print("RESTORE MISMATCH: aborting")
            break
    if results_json:
        results_json.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
    ok = len(results) == len(mutants) and all(
        r["status"] == "KILLED" and r["restored_sha_ok"] for r in results.values())
    return 0 if ok else 1


if __name__ == "__main__":
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    sys.exit(run_all_mutants(Path(sys.argv[1]), out))
