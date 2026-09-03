"""Capture the goldens that gate the import-laziness refactor.

Deferring an ``import`` from module scope into a function body is invisible to
an ordinary test suite: the library still loads the first time anything calls
the function, so the tests pass whether or not the deferral actually worked --
and equally, a deferral that quietly changed *which object* a name resolves to
would also pass. These goldens make both observable.

Two artifacts are written under ``tests/fixtures/import_laziness/``:

* ``import_surface.json`` -- ``__all__`` and per-name resolved identity for
  every public package, the eager ``sys.modules`` set after importing
  ``phenotypic`` and ``phenotypic.phenotypicCLI``, and what
  ``_find_class_in_phenotypic`` resolves every operation name to.
* ``legacy_migration.json`` -- structure, pixel hashes and (volatile-stripped)
  metadata for each of the six committed legacy HDF layouts converted to
  OME-Zarr. This anchors the ``h5py`` deferral, whose existing tests are all
  differential and would pass a defect present on both sides.

``timings.json`` is appended to as an informational record. It is never
asserted -- import timing is machine- and cache-dependent.

Capture must run **before** any import is moved, against unmodified behaviour.

Usage::

    uv run python scripts/capture_import_laziness_goldens.py
    uv run python scripts/capture_import_laziness_goldens.py --check
    uv run python scripts/capture_import_laziness_goldens.py --timings --label stage-1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make the repository root importable so ``tests.*`` resolves whether the
# script is run from the repo root or elsewhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests import _import_surface as surface  # noqa: E402
from tests import _legacy_store_digest as legacy  # noqa: E402


def _capture_migration() -> dict[str, Any]:
    """Convert every committed legacy layout and digest the results."""
    digests: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        for layout in legacy.LAYOUTS:
            digests[layout] = legacy.migrate_and_digest(
                layout, root / layout / "img.ome.zarr"
            )
    return {"schema_version": legacy.SCHEMA_VERSION, "layouts": digests}


def _measure_timings(label: str) -> dict[str, Any]:
    """Wall time and ``-X importtime`` self-total for the two import targets.

    Informational only. Recorded so each stage's effect is visible in the diff,
    never asserted -- these numbers move with the node, the filesystem cache and
    the interpreter's own startup.
    """
    measurements: dict[str, Any] = {}
    for target, argv in (
        ("import phenotypic", [sys.executable, "-c", "import phenotypic"]),
        (
            "python -m phenotypic --help",
            [sys.executable, "-m", "phenotypic", "--help"],
        ),
    ):
        walls: list[float] = []
        for _ in range(3):
            started = time.perf_counter()
            subprocess.run(
                argv, check=True, capture_output=True, cwd=_REPO_ROOT
            )
            walls.append(time.perf_counter() - started)

        timed = subprocess.run(
            [argv[0], "-X", "importtime", *argv[1:]],
            check=True,
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        )
        self_us = 0
        for line in timed.stderr.splitlines():
            if not line.startswith("import time:"):
                continue
            field = line.removeprefix("import time:").split("|", 1)[0].strip()
            if field.isdigit():
                self_us += int(field)

        measurements[target] = {
            "wall_seconds": [round(value, 3) for value in walls],
            "wall_seconds_min": round(min(walls), 3),
            "importtime_self_seconds": round(self_us / 1e6, 3),
        }
    return {
        "label": label,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "node": subprocess.run(
            ["hostname", "-s"], check=True, capture_output=True, text=True
        ).stdout.strip(),
        "commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=_REPO_ROOT,
        ).stdout.strip(),
        "measurements": measurements,
    }


def _append_timings(entry: dict[str, Any]) -> None:
    """Append one timing record to the informational log."""
    surface.GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    if surface.TIMINGS_LOG.exists():
        history = json.loads(surface.TIMINGS_LOG.read_text(encoding="utf-8"))
    history.append(entry)
    surface.TIMINGS_LOG.write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )


def _report_diff(name: str, expected: Any, actual: Any) -> bool:
    """Print a readable diff for one golden section. Returns True if equal."""
    if expected == actual:
        print(f"  ok       {name}")
        return True
    print(f"  MISMATCH {name}")
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) - set(actual)):
            print(f"             missing: {key}")
        for key in sorted(set(actual) - set(expected)):
            print(f"             added:   {key}")
        for key in sorted(set(expected) & set(actual)):
            if expected[key] != actual[key]:
                print(f"             changed: {key}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare against the committed goldens instead of overwriting them",
    )
    parser.add_argument(
        "--timings",
        action="store_true",
        help="also append a timing record (informational, never asserted)",
    )
    parser.add_argument(
        "--label",
        default="unlabelled",
        help="label for the timing record, e.g. 'stage-0-baseline'",
    )
    args = parser.parse_args()

    print("Capturing import surface ...")
    import_surface = surface.capture()
    print(
        f"  {len(import_surface['exports'])} packages, "
        f"{len(import_surface['class_resolution'])} operation classes"
    )
    for target, entry in import_surface["eager"].items():
        print(
            f"  eager after 'import {target}': "
            f"{len(entry['third_party_roots'])} third-party roots, "
            f"{entry['phenotypic_module_count']} phenotypic modules, "
            f"{entry['total_modules']} modules total"
        )

    print("Capturing legacy migration digests ...")
    migration = _capture_migration()
    print(f"  {len(migration['layouts'])} layouts")

    if args.check:
        print("\nComparing against committed goldens:")
        ok = True
        expected_surface = surface.load_golden()

        # `__all__` and per-name identity are the contract. `dir()` reads
        # __dict__ and does not trigger PEP-562 __getattr__, so it shrinks
        # legitimately whenever a name goes lazy -- reported, never failed.
        for key in ("__all__", "resolved"):
            ok &= _report_diff(
                f"import_surface.exports[{key}]",
                {
                    name: entry.get(key)
                    for name, entry in expected_surface["exports"].items()
                },
                {
                    name: entry.get(key)
                    for name, entry in import_surface["exports"].items()
                },
            )
        for name, entry in import_surface["exports"].items():
            was = set(expected_surface["exports"][name]["dir"])
            now = set(entry["dir"])
            if was != now:
                print(
                    f"  note     {name} dir(): "
                    f"-{sorted(was - now)} +{sorted(now - was)}"
                )

        # The eager set is a ratchet, not an equality: every successful stage
        # shrinks it, so demanding equality would fail on exactly the runs that
        # worked. Growth is the regression.
        for target, entry in import_surface["eager"].items():
            was = expected_surface["eager"][target]
            gained = sorted(
                set(entry["third_party_roots"]) - set(was["third_party_roots"])
            )
            dropped = sorted(
                set(was["third_party_roots"]) - set(entry["third_party_roots"])
            )
            delta = entry["phenotypic_module_count"] - was["phenotypic_module_count"]
            if gained or delta > 0:
                print(f"  REGRESSED import {target}: +{gained}, modules {delta:+d}")
                ok = False
            else:
                print(
                    f"  ok       import {target}: "
                    f"-{dropped or 'no'} third-party, modules {delta:+d}"
                )

        ok &= _report_diff(
            "import_surface.class_resolution",
            expected_surface["class_resolution"],
            import_surface["class_resolution"],
        )
        ok &= _report_diff(
            "legacy_migration.layouts",
            legacy.load_golden()["layouts"],
            migration["layouts"],
        )
        if args.timings:
            _append_timings(_measure_timings(args.label))
            print(f"  timings appended to {surface.TIMINGS_LOG}")
        print("\nPASS" if ok else "\nFAIL")
        return 0 if ok else 1

    surface.write_golden(import_surface)
    legacy.write_golden(migration)
    print(f"\nWrote {surface.IMPORT_SURFACE_GOLDEN}")
    print(f"Wrote {legacy.MIGRATION_GOLDEN}")
    if args.timings:
        _append_timings(_measure_timings(args.label))
        print(f"Appended {surface.TIMINGS_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
