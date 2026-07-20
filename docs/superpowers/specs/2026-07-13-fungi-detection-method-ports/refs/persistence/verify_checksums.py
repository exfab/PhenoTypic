"""Verify the committed A11 reference corpus and oracle artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent


def verify_persistence_reference_checksums() -> None:
    """Exit unsuccessfully if a checksum-pinned reference file drifted."""
    failures: list[str] = []
    for line in (HERE / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        expected, relative_path = line.split(maxsplit=1)
        path = HERE / relative_path
        if not path.is_file():
            failures.append(f"missing: {relative_path}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(
                f"mismatch: {relative_path}: expected {expected}, found {actual}"
            )
    if failures:
        raise SystemExit("\n".join(failures))
    print("A11 persistence reference checksums: PASS")


if __name__ == "__main__":
    verify_persistence_reference_checksums()
