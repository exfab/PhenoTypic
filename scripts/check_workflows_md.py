"""WORKFLOWS.md validator -- keeps the GUI tutorial registry, the capture
script, the on-disk screenshots, and the tutorial pages in lockstep.

Behaviour:
    * Parses ``src/phenotypic/gui/WORKFLOWS.md`` as a single feature table
      (header columns: ``ID``, ``Title``, ``Description``, ``Capture
      function``, ``Tutorial page``, ``Status``).
    * AST-walks ``scripts/capture_gui_tutorial_screenshots.py`` to
      collect every ``_capture_*`` function definition AND every call to
      such a function inside ``capture_workflow_screenshots`` /
      ``capture_standalone_viewer_screenshots``.
    * Each row's ``Capture function`` cell must list one or more
      backticked ``_capture_<id>`` references; each one MUST be defined
      AND dispatched.
    * For ``Status == ✅ shipping`` rows, also verifies:
        - ``docs/source/_static/gui_images/<ID>/`` contains at least
          one ``.png``;
        - the referenced tutorial page exists under
          ``docs/source/how_to/pages/``.
    * Catches orphans: capture functions that no row references.

Usage::

    python scripts/check_workflows_md.py        # pre-commit + CI gate
    python scripts/check_workflows_md.py -v     # print every row inspected
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS_MD = REPO_ROOT / "src" / "phenotypic" / "gui" / "WORKFLOWS.md"
CAPTURE_SCRIPT = REPO_ROOT / "scripts" / "capture_gui_tutorial_screenshots.py"
SCREENSHOTS_ROOT = REPO_ROOT / "docs" / "source" / "_static" / "gui_images"
TUTORIAL_ROOT = REPO_ROOT / "docs" / "source" / "how_to" / "pages"

STATUS_SHIPPING = "✅ shipping"
STATUS_IN_PROGRESS = "🚧 in progress"
STATUS_PLANNED = "🔭 planned"

# Functions that dispatch _capture_* helpers; calls inside these count as
# "this workflow is wired into the script's main run".
DISPATCH_FUNCTIONS = {
    "capture_workflow_screenshots",
    "capture_standalone_viewer_screenshots",
}

ROW_RE = re.compile(r"^\|(.+)\|\s*$")
SEPARATOR_RE = re.compile(r"^\|[\s:|-]+\|\s*$")
CAPTURE_REF_RE = re.compile(r"`(_capture_[A-Za-z0-9_]+)`")
BACKTICK_INNER_RE = re.compile(r"`([^`]+)`")


def _parse_workflows_table(text: str) -> list[dict[str, str]]:
    """Extract the workflow registry table from WORKFLOWS.md.

    Recognises the table by its header — the first table whose header
    cells include both ``ID`` and ``Capture function``. Every other
    markdown table in the document is ignored (legend, etc.).
    """
    headers: list[str] | None = None
    in_table = False
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if SEPARATOR_RE.match(line):
            in_table = headers is not None
            continue
        m = ROW_RE.match(line)
        if not m:
            headers = None
            in_table = False
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        if headers is None:
            if "ID" in cells and "Capture function" in cells:
                headers = cells
            continue
        if in_table and len(cells) == len(headers):
            rows.append(dict(zip(headers, cells)))
    return rows


def _scan_capture_script(path: Path) -> tuple[set[str], set[str]]:
    """Return ``(defined, dispatched)`` capture-function name sets.

    ``defined`` is every top-level ``def _capture_*`` in the script;
    ``dispatched`` is every ``_capture_*(...)`` call site found inside
    one of :data:`DISPATCH_FUNCTIONS`. ``ast.walk`` recurses into nested
    statements, so calls inside ``try``/``with``/``if`` blocks under the
    dispatcher count. Calls made *transitively* through a separate
    top-level helper do NOT count — workflows must be wired through one
    of the explicit dispatch entry points so a contributor can audit the
    surface by reading those two functions alone.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    defined: set[str] = set()
    dispatched: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith(
            "_capture_"
        ):
            defined.add(node.name)
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in DISPATCH_FUNCTIONS
        ):
            for sub in ast.walk(node):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id.startswith("_capture_")
                ):
                    dispatched.add(sub.func.id)
    return defined, dispatched


def _resolve_tutorial_path(field: str) -> Path | None:
    """Return the absolute Path for the ``Tutorial page`` cell, or None."""
    inner = BACKTICK_INNER_RE.findall(field)
    if not inner:
        return None
    rel = inner[0].strip()
    return TUTORIAL_ROOT / rel


def _check_row(
    row: dict[str, str],
    defined: set[str],
    dispatched: set[str],
) -> tuple[set[str], list[str]]:
    """Return ``(referenced_funcs, errors)`` for a single workflow row."""
    errors: list[str] = []
    wid = row.get("ID", "<missing>")
    fn_field = row.get("Capture function", "")
    funcs = set(CAPTURE_REF_RE.findall(fn_field))
    if not funcs:
        errors.append(
            f"row {wid!r}: ``Capture function`` cell has no "
            f"``_capture_*`` reference (got {fn_field!r})"
        )
    for fn in funcs:
        if fn not in defined:
            errors.append(
                f"row {wid!r}: references {fn!r} but no def is found in "
                f"{CAPTURE_SCRIPT.name}"
            )
        if fn not in dispatched:
            errors.append(
                f"row {wid!r}: {fn!r} is defined but not dispatched from "
                + " or ".join(sorted(DISPATCH_FUNCTIONS))
            )

    if row.get("Status") == STATUS_SHIPPING:
        sdir = SCREENSHOTS_ROOT / wid
        if not sdir.is_dir() or not list(sdir.glob("*.png")):
            errors.append(
                f"row {wid!r} is ✅ shipping but no PNGs under "
                f"{_pretty(sdir)}"
            )
        tpath = _resolve_tutorial_path(row.get("Tutorial page", ""))
        if tpath is None:
            errors.append(
                f"row {wid!r}: ``Tutorial page`` cell has no path"
            )
        elif not tpath.exists():
            errors.append(
                f"row {wid!r}: tutorial page "
                f"{_pretty(tpath)} does not exist"
            )
    return funcs, errors


def _pretty(path: Path) -> str:
    """Render *path* relative to the repo when possible, else absolute.

    ``Path.relative_to`` raises ``ValueError`` when the path is not under
    REPO_ROOT (the case for tmp_path fixtures in unit tests). The error
    messages don't care which form they get — they just need to be
    readable — so we fall back gracefully.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print every row inspected.",
    )
    args = parser.parse_args(argv)

    if not WORKFLOWS_MD.exists():
        print(f"missing {_pretty(WORKFLOWS_MD)}", file=sys.stderr)
        return 1
    if not CAPTURE_SCRIPT.exists():
        print(f"missing {_pretty(CAPTURE_SCRIPT)}", file=sys.stderr)
        return 1

    rows = _parse_workflows_table(WORKFLOWS_MD.read_text(encoding="utf-8"))
    if not rows:
        print(
            "WORKFLOWS.md parsed 0 rows -- expected a table with ``ID`` "
            "and ``Capture function`` headers",
            file=sys.stderr,
        )
        return 1

    defined, dispatched = _scan_capture_script(CAPTURE_SCRIPT)

    errors: list[str] = []
    referenced: set[str] = set()
    for row in rows:
        used, row_errors = _check_row(row, defined, dispatched)
        referenced |= used
        errors.extend(row_errors)
        if args.verbose:
            print(
                f"  - {row.get('ID', '?'):<20} "
                f"{row.get('Status', '?'):<14} -> {sorted(used) or 'NONE'}"
            )

    orphans = sorted(defined - referenced)
    for fn in orphans:
        errors.append(
            f"{fn!r} is defined in {CAPTURE_SCRIPT.name} but no "
            f"WORKFLOWS.md row references it"
        )
    undefined = sorted(dispatched - defined)
    for fn in undefined:
        errors.append(
            f"{fn!r} is dispatched but never defined "
            f"(impossible at runtime — script will NameError)"
        )

    if errors:
        print("WORKFLOWS.md validation failed:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print(
        f"WORKFLOWS.md OK -- {len(rows)} workflows, "
        f"{len(defined)} capture functions, "
        f"{len(dispatched)} dispatched"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
