"""Verify pinned A10 FilFinder evidence and the local source corpus."""

from __future__ import annotations

import hashlib
import pathlib
import sys


REFERENCE_DIR = pathlib.Path(__file__).resolve().parent
REPOSITORY_ROOT = REFERENCE_DIR.parents[5]
EXPECTED_SOURCE_FILE_COUNT = 59
EXPECTED_SOURCE_AGGREGATE_SHA256 = (
    "73db4ddb96269a1602a66f1afdc9d6b036faf79f3d374861cf4544761a590174"
)
TEXT_SUFFIXES = {".json", ".md", ".py", ".rst", ".txt"}
TEXT_FILENAMES = {"PKG-INFO"}


def canonical_bytes(path: pathlib.Path) -> bytes:
    """Normalize line endings only for explicitly classified text evidence."""
    content = path.read_bytes()
    if path.suffix in TEXT_SUFFIXES or path.name in TEXT_FILENAMES:
        return content.replace(b"\r\n", b"\n")
    return content


def sha256(path: pathlib.Path) -> str:
    """Return the canonical SHA-256 digest for one evidence file."""
    return hashlib.sha256(canonical_bytes(path)).hexdigest()


def verify_manifest() -> None:
    """Verify every explicitly pinned evidence file."""
    manifest = REFERENCE_DIR / "CHECKSUMS.sha256"
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        expected, relative = line.split("  ", maxsplit=1)
        actual = sha256(REPOSITORY_ROOT / relative)
        if actual != expected:
            raise AssertionError(
                f"checksum mismatch on manifest line {line_number}: "
                f"{relative}: {actual} != {expected}"
            )


def verify_source_corpus() -> None:
    """Verify the exact extracted source subset and aggregate digest."""
    source_dir = REFERENCE_DIR / "upstream"
    paths = sorted(path for path in source_dir.rglob("*") if path.is_file())
    if len(paths) != EXPECTED_SOURCE_FILE_COUNT:
        raise AssertionError(
            f"source file count changed: {len(paths)} != {EXPECTED_SOURCE_FILE_COUNT}"
        )
    aggregate = hashlib.sha256()
    for path in paths:
        aggregate.update(path.relative_to(source_dir).as_posix().encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(bytes.fromhex(sha256(path)))
    actual = aggregate.hexdigest()
    if actual != EXPECTED_SOURCE_AGGREGATE_SHA256:
        raise AssertionError(
            "source aggregate changed: "
            f"{actual} != {EXPECTED_SOURCE_AGGREGATE_SHA256}"
        )


def verify_filfinder_evidence() -> None:
    """Run every A10 immutable-evidence verification."""
    verify_manifest()
    verify_source_corpus()


if __name__ == "__main__":
    try:
        verify_filfinder_evidence()
    except Exception as error:  # pragma: no cover - command-line failure report
        print(f"A10 FilFinder checksum verification FAILED: {error}", file=sys.stderr)
        raise
    print("A10 FilFinder checksum verification PASSED")
