"""FEATURES.md validator — invoked by the pre-commit hook and CI gate.

Behaviour:
    * Parses ``src/phenotypic/gui/FEATURES.md`` as a sequence of markdown tables.
    * For rows with ``Status == ✅ shipping``, resolves ``Test ref`` to a real
      ``path::test`` reference and asserts the file (and named test) exists.
    * Rows with ``Status == 🚧 in progress`` cause failure when ``--strict`` is
      passed (used by the merge-gate CI step).
    * Rows with ``Status == 🔭 planned`` are skipped — Test ref is allowed to
      point at not-yet-existing files.
    * Returns nonzero on any violation; prints the offending row(s) to stderr.

Usage::

    python scripts/check_features_md.py            # pre-commit + per-PR CI
    python scripts/check_features_md.py --strict   # merge gate (rejects 🚧)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURES_MD = REPO_ROOT / "src" / "phenotypic" / "gui" / "FEATURES.md"

STATUS_SHIPPING = "✅ shipping"
STATUS_IN_PROGRESS = "🚧 in progress"
STATUS_PLANNED = "🔭 planned"
MANUAL_REF = "n/a (manual)"

# A row is a markdown table line: ``| cell1 | cell2 | ... |``. The trailing
# newline + optional whitespace before EOL is matched by ``\s*$``.
ROW_RE = re.compile(r"^\|(.+)\|\s*$")
SEPARATOR_RE = re.compile(r"^\|[\s:|-]+\|\s*$")


def parse_tables(text: str) -> tuple[list[dict[str, str]], list[str]]:
    """Parse all markdown tables in ``text``.

    Returns ``(rows, warnings)``. ``rows`` is keyed by header cell. Tables in
    the legend (Status / Meaning) and other non-feature tables are included;
    the caller filters by required columns. ``warnings`` collects malformed
    rows whose column count does not match the header — but only under
    *feature tables* (those with both ``Status`` and ``Test ref`` headers).
    Anywhere else, mismatched cell counts are silently skipped.
    """
    rows: list[dict[str, str]] = []
    warnings: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = ROW_RE.match(lines[i])
        if not m:
            i += 1
            continue
        # Potential header row — must be followed by a separator row.
        if i + 1 >= len(lines) or not SEPARATOR_RE.match(lines[i + 1]):
            i += 1
            continue
        headers = [c.strip() for c in m.group(1).split("|")]
        is_feature_table = "Status" in headers and "Test ref" in headers
        i += 2
        while i < len(lines):
            m2 = ROW_RE.match(lines[i])
            if not m2:
                break
            cells = [c.strip() for c in m2.group(1).split("|")]
            if len(cells) != len(headers):
                if is_feature_table:
                    warnings.append(
                        f"line {i + 1}: malformed feature row "
                        f"({len(cells)} cells vs {len(headers)} headers) "
                        f"— skipped (likely missing or extra '|')"
                    )
                i += 1
                continue
            rows.append(dict(zip(headers, cells)))
            i += 1
    return rows, warnings


def resolve_test_ref(ref: str) -> tuple[Path, str | None] | None:
    """Split a test ref into ``(absolute_path, leaf_test_name_or_None)``.

    Accepts the pytest reference forms used by the FEATURES.md ledger:

    * ``path/to/file.py`` → ``(path, None)`` (file-level)
    * ``path/to/file.py::test_name`` → ``(path, "test_name")``
    * ``path/to/file.py::TestClass::test_method`` → ``(path, "test_method")``;
      the class chain is dropped because the existence check below greps for
      a ``def`` definition and class-based tests still satisfy that.

    Returns ``None`` for the manual sentinel, signalling "no resolution
    needed".
    """
    if ref == MANUAL_REF:
        return None
    parts = ref.split("::")
    path_str = parts[0]
    test_name = parts[-1] if len(parts) > 1 else None
    return (
        REPO_ROOT / path_str.strip(),
        test_name.strip() if test_name else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any row with status '🚧 in progress' (merge-gate mode).",
    )
    args = parser.parse_args()

    if not FEATURES_MD.exists():
        print(f"[features-check] {FEATURES_MD} does not exist", file=sys.stderr)
        return 2

    rows, warnings = parse_tables(FEATURES_MD.read_text())
    for w in warnings:
        print(f"[features-check] WARN: {w}", file=sys.stderr)
    if not rows:
        print("[features-check] no rows parsed from FEATURES.md", file=sys.stderr)
        return 2

    feature_rows = [
        r for r in rows if "Status" in r and "Test ref" in r
    ]
    if not feature_rows:
        print(
            "[features-check] no rows with Status + Test ref columns found",
            file=sys.stderr,
        )
        return 2

    errors: list[str] = []
    in_progress: list[dict[str, str]] = []
    n_shipping = 0

    for row in feature_rows:
        status = row["Status"]
        ref = row["Test ref"]
        feature = row.get("Feature", "?")

        if status == STATUS_IN_PROGRESS:
            in_progress.append(row)
            continue
        if status != STATUS_SHIPPING:
            continue

        n_shipping += 1
        resolved = resolve_test_ref(ref)
        if resolved is None:
            continue
        path, test_name = resolved
        if not path.exists():
            errors.append(
                f"missing test file: {path.relative_to(REPO_ROOT)} "
                f"(row: {feature})"
            )
            continue
        if test_name is not None:
            content = path.read_text()
            if not re.search(rf"\bdef\s+{re.escape(test_name)}\s*\(", content):
                errors.append(
                    f"test '{test_name}' not found in "
                    f"{path.relative_to(REPO_ROOT)} (row: {feature})"
                )

    if args.strict:
        for row in in_progress:
            errors.append(
                f"row in '🚧 in progress' (rejected by merge gate): "
                f"{row.get('Feature', '?')} — {row.get('Element', '?')}"
            )

    if errors:
        print("[features-check] violations:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print(
        f"[features-check] OK ({len(feature_rows)} feature rows, "
        f"{n_shipping} shipping, {len(in_progress)} in progress)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
