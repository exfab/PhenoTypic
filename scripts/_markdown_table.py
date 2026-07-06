"""Small Markdown table parsing helpers for repository scripts."""

from __future__ import annotations

_PIPE_ESCAPE = "\\|"
_PIPE_SENTINEL = "\x00PIPE\x00"


def split_markdown_row_cells(row: str) -> list[str]:
    """Split a Markdown table row while honoring escaped pipe characters.

    Args:
        row: Row text with or without the outer table delimiters.

    Returns:
        Trimmed cell values with escaped pipes restored.
    """
    inner = row.strip()
    if inner.startswith("|") and inner.endswith("|"):
        inner = inner[1:-1]
    protected = inner.replace(_PIPE_ESCAPE, _PIPE_SENTINEL)
    return [
        cell.strip().replace(_PIPE_SENTINEL, "|")
        for cell in protected.split("|")
    ]
