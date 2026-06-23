"""Compile a plate-identity pattern over filename stems (spec §5.3, §15.5).

Placeholder syntax (primary): ``{plate}`` (required), optional ``{time}``,
``*`` wildcard, literal text. Compiled to an anchored, non-greedy regex.
Advanced mode: a raw regex with a named ``plate`` group (``time`` optional).
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

__all__ = ["PlateMatch", "PatternError", "parse_plate_identity"]


@dataclass(frozen=True)
class PlateMatch:
    """One stem's extracted plate identity + time (``None`` when unmatched)."""

    stem: str
    plate: str | None
    time: str | None


class PatternError(ValueError):
    """Raised for an invalid plate-identity pattern (placeholder or regex)."""


_TOKEN = re.compile(r"\{plate\}|\{time\}|\*")
_TOKEN_REGEX = {
    "{plate}": "(?P<plate>.+?)",
    "{time}": "(?P<time>.+?)",
    "*": ".*?",
}


def _compile(pattern: str, *, advanced: bool) -> re.Pattern[str]:
    if advanced:
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise PatternError(f"invalid regex: {exc}") from exc
        if "plate" not in compiled.groupindex:
            raise PatternError("pattern must contain a (?P<plate>...) group")
        return compiled

    if "{plate}" not in pattern:
        raise PatternError("pattern must contain {plate}")
    if pattern.count("{plate}") > 1 or pattern.count("{time}") > 1:
        raise PatternError("duplicate {plate}/{time} token")

    parts: list[str] = []
    pos = 0
    for match in _TOKEN.finditer(pattern):
        parts.append(re.escape(pattern[pos : match.start()]))
        parts.append(_TOKEN_REGEX[match.group()])
        pos = match.end()
    parts.append(re.escape(pattern[pos:]))
    try:
        return re.compile("^" + "".join(parts) + "$")
    except re.error as exc:  # pragma: no cover - tokens are well-formed
        raise PatternError(f"could not compile pattern: {exc}") from exc


def parse_plate_identity(
    stems: Iterable[str], pattern: str, *, advanced: bool = False
) -> list[PlateMatch]:
    """Match each stem against ``pattern``; return per-stem plate/time captures.

    Args:
        stems: Filename stems (no directory, no extension).
        pattern: Placeholder (default) or raw-regex (``advanced=True``) pattern.
        advanced: When ``True``, ``pattern`` is a raw regex with a named
            ``plate`` group (``time`` optional).

    Returns:
        One :class:`PlateMatch` per stem (``plate``/``time`` are ``None`` when
        the stem does not match).

    Raises:
        PatternError: When the pattern is structurally invalid.
    """
    compiled = _compile(pattern, advanced=advanced)
    out: list[PlateMatch] = []
    for stem in stems:
        m = compiled.match(stem)
        if m is None:
            out.append(PlateMatch(stem, None, None))
            continue
        groups = m.groupdict()
        out.append(PlateMatch(stem, groups.get("plate"), groups.get("time")))
    return out
