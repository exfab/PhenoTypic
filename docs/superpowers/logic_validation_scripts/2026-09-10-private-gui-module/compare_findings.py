"""Compare mypy / ruff / Sphinx findings between main and refactor/private-gui.

Normalizes away only what the package move legitimately changes -- the package
path (``phenotypic/gui/`` -> ``phenotypic/_gui/``, ``phenotypic.gui`` ->
``phenotypic._gui``), the first line/column number on each line, and each
checkout's absolute prefix -- then compares findings as multisets. Exits 1 when
the branch has any finding that main does not, so a gate cannot pass silently.

Depends only on the standard library and never imports ``phenotypic``.

Usage:
    uv run python compare_findings.py mypy MAIN_FILE BRANCH_FILE
    uv run python compare_findings.py ruff MAIN_FILE BRANCH_FILE
    uv run python compare_findings.py docs MAIN_LOG  BRANCH_LOG
    uv run python compare_findings.py --selftest
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

# Any absolute path up to the repo-relative root it contains (src/, docs/,
# tests/, scripts/), so two checkouts at different locations compare equal.
CHECKOUT_PREFIX = re.compile(r"^/\S*?/(?=(?:src|docs|tests|scripts)/)")
PACKAGE_PATH = re.compile(r"\bphenotypic/gui/")
PACKAGE_DOTTED = re.compile(r"\bphenotypic\.gui\b")
FIRST_LINE_COL = re.compile(r":\d+(:\d+)?:")
RUFF_FINDING = re.compile(r"^\S+:\d+:\d+: ")

KEEP = {
    "mypy": lambda line: ": error:" in line,
    "ruff": lambda line: RUFF_FINDING.match(line) is not None,
    "docs": lambda line: "WARNING" in line or "ERROR" in line,
}


def normalize(line: str) -> str:
    """Remove the differences the move is allowed to cause."""
    line = line.rstrip("\r\n")
    line = CHECKOUT_PREFIX.sub("", line)
    line = PACKAGE_PATH.sub("phenotypic/_gui/", line)
    line = PACKAGE_DOTTED.sub("phenotypic._gui", line)
    return FIRST_LINE_COL.sub(":", line, count=1)


def findings(kind: str, lines: list[str]) -> Counter[str]:
    keep = KEEP[kind]
    return Counter(normalize(line) for line in lines if keep(line.rstrip("\r\n")))


def compare(kind: str, main_lines: list[str], branch_lines: list[str]) -> tuple[Counter[str], Counter[str]]:
    """Return (only in branch, only in main) as multisets."""
    main, branch = findings(kind, main_lines), findings(kind, branch_lines)
    return branch - main, main - branch


def selftest() -> None:
    same = [
        ("mypy", "src/phenotypic/gui/shell/_app.py:12: error: Incompatible types  [assignment]",
                 "src/phenotypic/_gui/shell/_app.py:14: error: Incompatible types  [assignment]"),
        ("mypy", 'src/phenotypic/sdk_/x.py:3: error: Module "phenotypic.gui.shell" has no attribute "y"  [attr-defined]',
                 'src/phenotypic/sdk_/x.py:3: error: Module "phenotypic._gui.shell" has no attribute "y"  [attr-defined]'),
        ("ruff", "src/phenotypic/gui/a.py:3:1: F401 [*] `os` imported but unused",
                 "src/phenotypic/_gui/a.py:9:1: F401 [*] `os` imported but unused\r"),
        ("docs", "/tmp/pht-main/src/phenotypic/gui/builder/_state.py:docstring of phenotypic.gui.builder._state.Edge:1: WARNING: py:class reference target not found: X",
                 "/Users/alex/Projects/PhenoTypic/src/phenotypic/_gui/builder/_state.py:docstring of phenotypic._gui.builder._state.Edge:3: WARNING: py:class reference target not found: X"),
        ("docs", "/private/tmp/pht-main/docs/source/how_to/a.md:4: WARNING: undefined label: x",
                 "/home/ci/work/PhenoTypic/docs/source/how_to/a.md:9: WARNING: undefined label: x"),
    ]
    for kind, main_line, branch_line in same:
        new, gone = compare(kind, [main_line], [branch_line])
        assert not new and not gone, (kind, main_line, branch_line, new, gone)

    # Negative controls: differences the move must NOT be allowed to hide.
    different = [
        ("ruff", "tests/unit/gui/test_a.py:3:1: F401 x", "tests/unit/_gui/test_a.py:3:1: F401 x"),
        ("docs", "README.md:5: WARNING: see phenotypic-gui", "README.md:5: WARNING: see phenotypic-_gui"),
        ("mypy", "src/phenotypic/_cli/a.py:1: error: A  [x]", "src/phenotypic/_cli/a.py:1: error: B  [x]"),
        ("docs", "/tmp/pht-main/docs/source/a.md:1: WARNING: x", "/tmp/pht-main/docs/source/b.md:1: WARNING: x"),
    ]
    for kind, main_line, branch_line in different:
        new, gone = compare(kind, [main_line], [branch_line])
        assert new and gone, (kind, main_line, branch_line)

    # Multiset: a duplicated finding in the branch is new even if the text exists in main.
    new, _ = compare("mypy", ["src/phenotypic/a.py:1: error: A  [x]"],
                     ["src/phenotypic/a.py:1: error: A  [x]", "src/phenotypic/a.py:7: error: A  [x]"])
    assert sum(new.values()) == 1, new

    # Filters: notes and summaries are not findings.
    assert not findings("mypy", ["src/a.py:1: note: See docs", "Found 418 errors in 121 files"])
    assert not findings("ruff", ["Found 65 errors.", "[*] 7 fixable with the `--fix` option."])
    print("selftest: ok")


def main(argv: list[str]) -> int:
    if argv == ["--selftest"]:
        selftest()
        return 0
    if len(argv) != 3 or argv[0] not in KEEP:
        print(__doc__, file=sys.stderr)
        return 2
    kind, main_path, branch_path = argv
    read = lambda p: Path(p).read_text(encoding="utf-8", errors="replace").splitlines()
    main_lines, branch_lines = read(main_path), read(branch_path)
    new, gone = compare(kind, main_lines, branch_lines)
    print(f"{kind}: main {sum(findings(kind, main_lines).values())} findings, "
          f"branch {sum(findings(kind, branch_lines).values())} findings")
    print(f"only in branch (new): {sum(new.values())}")
    for line, count in sorted(new.items()):
        print(f"  +{count}  {line}")
    print(f"only in main (gone): {sum(gone.values())}")
    for line, count in sorted(gone.items()):
        print(f"  -{count}  {line}")
    return 1 if new else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
