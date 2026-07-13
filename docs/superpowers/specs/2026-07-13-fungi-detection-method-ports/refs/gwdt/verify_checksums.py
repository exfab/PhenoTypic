"""Verify pinned GWDT artifacts independent of checkout line endings."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[6]
MANIFEST = Path(__file__).with_name("CHECKSUMS.sha256")
TEXT_SUFFIXES = {"", ".cpp", ".h", ".json", ".md", ".py", ".txt"}


def verify_checksums() -> None:
    """Verify every manifest entry after normalizing text to upstream LF."""
    for record in MANIFEST.read_text(encoding="utf-8").splitlines():
        expected, relative_path = record.split("  ", maxsplit=1)
        path = ROOT / relative_path
        content = path.read_bytes()
        if path.suffix in TEXT_SUFFIXES:
            content = content.replace(b"\r\n", b"\n")
        actual = hashlib.sha256(content).hexdigest()
        if actual != expected:
            raise AssertionError(
                f"checksum mismatch for {relative_path}: {actual} != {expected}"
            )
        print(f"{relative_path}: OK")


if __name__ == "__main__":
    verify_checksums()
