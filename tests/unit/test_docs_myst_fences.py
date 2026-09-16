"""Gate: a MyST directive that silently stops being a directive.

``sphinx-build`` exits 0 for both defects below. It emits a warning that scrolls
past among the pre-existing lexing failures, and the page still renders -- as a
**syntax-highlighted Python code block** containing the next section's headings
and prose. Nothing fails, and the only way to notice is to read the generated
HTML for a section you already suspect.

Two causes, both demonstrated in this tree:

1. **A backtick in the info string.** CommonMark forbids it, so
   ```` ```{admonition} Migrating from a legacy `sweep` manifest ```` is not a
   fence at all -- it is paragraph text. The ``` intended as its closer becomes
   an *opener*, and swallows everything up to the next fence.
2. **A same-length fence nested inside it.** A ```` ``` ````-fenced directive
   containing a ```` ```bash ```` block cannot nest: the inner fence terminates
   the outer one, and the same swallowing follows.

Both are fixed the same way -- write the directive with MyST's colon fence
(``:::{admonition}`` / ``:::``), which has neither restriction.

**Why fence-parity counting cannot find either.** In case 1 the backtick counts
balance perfectly; the document has an even number of ```` ``` ```` lines and is
still wrong. In case 2 they balance too. The structure is legal and the meaning
is not, which is why this asserts the *shape of the opener* rather than a tally.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs" / "source"

#: A backtick-fenced MyST directive: ```` ```{name} `` optionally followed by a
#: title. Colon-fenced directives (``:::{name}``) are exempt by construction --
#: they are the fix.
_DIRECTIVE_OPEN = re.compile(r"^(?P<ticks>`{3,})\{(?P<rest>.*)$")
#: Any backtick fence carrying an info string, e.g. ```` ```bash ````.
_INFO_FENCE = re.compile(r"^(?P<ticks>`{3,})(?P<info>\S.*)$")


def _fence_defects(text: str) -> list[str]:
    """Return one message per broken directive fence in *text*."""
    defects: list[str] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        opener = _DIRECTIVE_OPEN.match(line)
        if opener is None:
            continue
        ticks = opener.group("ticks")
        lineno = index + 1
        if "`" in opener.group("rest"):
            defects.append(
                f"line {lineno}: backtick in the directive info string -- "
                f"CommonMark refuses it, so this is not a fence. Use "
                f"':::{{...}}' / ':::'. Got: {line.strip()!r}"
            )
            continue
        # Walk to this directive's terminator. A same-length fence that carries
        # an info string cannot close it and cannot nest inside it.
        for follower in lines[index + 1 :]:
            if not follower.startswith(ticks):
                continue
            nested = _INFO_FENCE.match(follower)
            if nested is not None and len(nested.group("ticks")) == len(ticks):
                defects.append(
                    f"line {lineno}: directive contains a same-length fence "
                    f"({follower.strip()!r}), which terminates it instead of "
                    f"nesting. Use ':::{{...}}' / ':::' for the outer block."
                )
            break
    return defects


def test_no_markdown_directive_silently_stops_being_a_directive():
    """Whole docs tree, not only the pages this change touched.

    Scoped wide deliberately. The instance that motivated this gate
    (``how_to/pages/tuning.md``) predates this change and had been rendering its
    "Python interface" section as a Python code block for as long as the
    admonition had existed -- found only because an identical defect was
    introduced beside it. A gate that covered only the edited files would have
    left it, and left the next reader to conclude the pattern is acceptable.
    """
    failures: list[str] = []
    for path in sorted(DOCS.rglob("*.md")):
        for defect in _fence_defects(path.read_text(encoding="utf-8")):
            failures.append(f"{path.relative_to(REPO)}: {defect}")
    assert not failures, "\n".join(failures)


def test_the_gate_detects_both_causes():
    """The gate itself, against synthetic inputs carrying each defect.

    Without this, a scanner that matched nothing would pass the tree-wide test
    exactly as convincingly as a correct one -- the green would report "no
    defects found" when it meant "no defects detectable".
    """
    backtick_in_title = "```{admonition} Migrating a legacy `sweep` manifest\n:::\n"
    nested_same_length = "```{warning}\nprose\n\n```bash\nls\n```\n```\n"
    clean_colon_fence = ":::{admonition} A `code` title\n```bash\nls\n```\n:::\n"

    assert len(_fence_defects(backtick_in_title)) == 1
    assert len(_fence_defects(nested_same_length)) == 1
    assert _fence_defects(clean_colon_fence) == []
