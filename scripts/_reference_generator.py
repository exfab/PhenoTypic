"""Shared check/write behavior for generated reference scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def write_or_check_generated_file(
    *,
    output_path: Path,
    rendered: str,
    check: bool,
    regenerate_command: str,
) -> int:
    """Write generated content or check that the committed file is current.

    Args:
        output_path: Destination path for the generated document.
        rendered: Generated document content.
        check: If true, compare without writing.
        regenerate_command: Command shown to users when regeneration is needed.

    Returns:
        Process-style exit code.
    """
    if check:
        if not output_path.exists():
            print(
                f"{output_path} does not exist; run "
                f"{regenerate_command} without --check.",
                file=sys.stderr,
            )
            return 1
        existing = output_path.read_text(encoding="utf-8")
        if existing != rendered:
            print(
                f"{output_path} is out of date; regenerate with "
                f"`{regenerate_command}`.",
                file=sys.stderr,
            )
            return 1
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered, encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0
