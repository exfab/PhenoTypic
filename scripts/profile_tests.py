"""Two-pass pytest timing profiler with fixture attribution and optional memory profiling.

Pass 1 maps tests to fixtures via ``--setup-plan`` (no execution).
Pass 2 runs tests with ``--durations=0`` and collects timing data.
Cross-references the two passes to produce sorted tables showing slowest
tests, category breakdowns, and fixture attribution.

Memory profiling is opt-in via two independent flags:

- ``--memory``: uses memray (requires ``pytest-memray``; not available on Windows).
  Binary results are written to ``.benchmark/memray/``. Generate a flamegraph with::

      memray flamegraph .benchmark/memray/<result.bin>

- ``--tracemalloc``: uses stdlib ``tracemalloc`` (built-in pytest flag, no extra deps).
  Output is embedded in the normal pytest terminal output and captured in
  ``.benchmark/tests_full.log``.

Usage::

    uv run python scripts/profile_tests.py
    uv run python scripts/profile_tests.py tests/unit/core -n 10
    uv run python scripts/profile_tests.py --category detect --fixture synth_plate
    uv run python scripts/profile_tests.py --json profile_results.json
    uv run python scripts/profile_tests.py --memory tests/smoke
    uv run python scripts/profile_tests.py --tracemalloc tests/smoke
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = PROJECT_ROOT / ".benchmark"
MEMRAY_DIR = BENCHMARK_DIR / "memray"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCOPE_MAP: dict[str, str] = {
    "S": "session",
    "M": "module",
    "C": "class",
    "F": "function",
    "P": "package",
}

SETUP_RE = re.compile(r"^\s*SETUP\s+([SMCFP])\s+(\w+)(?:\[(.+?)\])?")
TEST_WITH_FIXTURES_RE = re.compile(
    r"^\s+(.+?::.+?)\s+\(fixtures used: (.+)\)\s*$"
)
TEST_PLAIN_RE = re.compile(r"^\s+(\S+::\S+)\s*$")

DURATION_RE = re.compile(r"^\s*(\d+\.\d+)s\s+(setup|call|teardown)\s+(.+)$")
SUMMARY_RE = re.compile(
    r"(\d+)\s+passed|(\d+)\s+failed|(\d+)\s+skipped|(\d+)\s+error"
)

_PYTEST_BASE_ARGS = [
    "-p",
    "no:napari",
    "-p",
    "no:testmon",
    "-p",
    "no:xdist",
    "--override-ini=addopts=",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DurationRecord:
    """A single timing record from pytest ``--durations`` output."""

    seconds: float
    phase: str  # "setup", "call", "teardown"
    nodeid: str


@dataclass
class TestTiming:
    """Aggregated timing for a single test."""

    nodeid: str
    category: str
    setup_time: float
    call_time: float
    teardown_time: float
    total_time: float
    fixtures: list[str]


@dataclass
class CategoryStats:
    """Aggregate statistics for a test category."""

    name: str
    test_count: int
    total_time: float
    mean_time: float
    slowest_test: str
    slowest_time: float


@dataclass
class FixtureStats:
    """Aggregate statistics for a fixture."""

    name: str
    scope: str  # "session", "module", "class", "function"
    test_count: int
    attributed_setup_time: float


@dataclass
class ProfileResult:
    """Full profiling result for JSON serialization."""

    total_tests: int
    total_time: float
    passed: int
    failed: int
    skipped: int
    errors: int
    test_timings: list[TestTiming]
    category_stats: list[CategoryStats]
    fixture_stats: list[FixtureStats]


# ---------------------------------------------------------------------------
# Category extraction
# ---------------------------------------------------------------------------


def extract_category(nodeid: str) -> str:
    """Extract test category from nodeid.

    Args:
        nodeid: Pytest node identifier, e.g.
            ``tests/unit/core/test_image.py::test_foo``.

    Returns:
        The category string derived from the path structure.

    Examples:
        >>> extract_category("tests/unit/core/test_image.py::test_foo")
        'core'
        >>> extract_category("tests/smoke/test_foo.py::test_bar")
        'smoke'
        >>> extract_category("tests/integration/test_foo.py::test_bar")
        'integration'
        >>> extract_category("tests/unit/test_fixtures.py::test_baz")
        'unit'
    """
    path_part = nodeid.split("::")[0]
    parts = Path(path_part).parts

    # Walk past "tests" and known tier directories
    tiers = {"unit", "integration", "smoke"}
    idx = 0
    for i, p in enumerate(parts):
        if p == "tests":
            idx = i + 1
            break

    if idx >= len(parts):
        return "unknown"

    # If the next part is a tier directory (unit/integration/smoke)
    if parts[idx] in tiers:
        tier = parts[idx]
        # If there's a subdirectory after the tier, that's the category
        if idx + 1 < len(parts) and not parts[idx + 1].endswith(".py"):
            return parts[idx + 1]
        # Otherwise the tier itself is the category
        return tier

    # Not under a tier — use the directory name itself
    if not parts[idx].endswith(".py"):
        return parts[idx]

    return "unknown"


# ---------------------------------------------------------------------------
# Time formatting
# ---------------------------------------------------------------------------


def format_duration(seconds: float) -> str:
    """Format seconds as a human-readable duration string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like ``2m 13.4s`` or ``0.56s``.
    """
    if seconds >= 60:
        minutes = int(seconds // 60)
        remaining = seconds - minutes * 60
        return f"{minutes}m {remaining:.1f}s"
    return f"{seconds:.2f}s"


# ---------------------------------------------------------------------------
# Pass 1 — Fixture mapping
# ---------------------------------------------------------------------------


def run_fixture_map(
    targets: list[str],
    extra_args: list[str],
    console: Console,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Run ``--setup-plan`` to map tests to their fixtures.

    Args:
        targets: Test paths to profile.
        extra_args: Additional pytest arguments.
        console: Rich console for status output.

    Returns:
        A tuple of ``(test_fixtures, fixture_scopes)`` where
        *test_fixtures* maps ``{nodeid: [fixture_names]}`` and
        *fixture_scopes* maps ``{fixture_name: scope_letter}``.
    """
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--setup-plan",
        *_PYTEST_BASE_ARGS,
        *extra_args,
        *targets,
    ]

    console.print("[bold cyan]Pass 1:[/] Collecting fixture map ...", highlight=False)
    console.print(f"  [dim]{' '.join(cmd)}[/dim]", highlight=False)

    result = subprocess.run(cmd, capture_output=True, text=True)

    test_fixtures: dict[str, list[str]] = {}
    fixture_scopes: dict[str, str] = {}

    if result.returncode != 0 and not result.stdout:
        console.print(
            "[bold yellow]Warning:[/] Pass 1 failed; fixture data unavailable.",
            highlight=False,
        )
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-5:]:
                console.print(f"  [dim]{line}[/dim]", highlight=False)
        return test_fixtures, fixture_scopes

    for line in result.stdout.splitlines():
        setup_m = SETUP_RE.match(line)
        if setup_m:
            scope_letter, fixture_name, _param = setup_m.groups()
            fixture_scopes[fixture_name] = SCOPE_MAP.get(scope_letter, scope_letter)
            continue

        test_m = TEST_WITH_FIXTURES_RE.match(line)
        if test_m:
            nodeid = test_m.group(1).strip()
            fixtures = [f.strip() for f in test_m.group(2).split(",")]
            test_fixtures[nodeid] = fixtures
            continue

        plain_m = TEST_PLAIN_RE.match(line)
        if plain_m:
            nodeid = plain_m.group(1).strip()
            test_fixtures.setdefault(nodeid, [])

    console.print(
        f"  Mapped [bold]{len(test_fixtures)}[/bold] tests, "
        f"[bold]{len(fixture_scopes)}[/bold] fixtures.",
        highlight=False,
    )
    return test_fixtures, fixture_scopes


# ---------------------------------------------------------------------------
# Pass 2 — Timing
# ---------------------------------------------------------------------------


def run_timing(
    targets: list[str],
    extra_args: list[str],
    console: Console,
    memory_args: list[str] | None = None,
) -> tuple[list[DurationRecord], dict[str, int], str]:
    """Run pytest with ``--durations=0`` and parse timing output.

    Args:
        targets: Test paths to profile.
        extra_args: Additional pytest arguments.
        console: Rich console for status output.
        memory_args: Optional memory-profiling flags appended after ``extra_args``
            (e.g. ``["--memray", "--memray-bin-path", "/path"]`` or
            ``["--tracemalloc"]``).

    Returns:
        A tuple of ``(records, summary, raw_output)`` where *records* is
        a list of :class:`DurationRecord`, *summary* is a dict with keys
        ``passed``, ``failed``, ``skipped``, ``errors``, and
        *raw_output* is the complete pytest stdout+stderr.
    """
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--durations=0",
        "--durations-min=0",
        "-q",
        *_PYTEST_BASE_ARGS,
        *extra_args,
        *(memory_args or []),
        *targets,
    ]

    console.print("[bold cyan]Pass 2:[/] Running tests with timing ...", highlight=False)
    console.print(f"  [dim]{' '.join(cmd)}[/dim]", highlight=False)

    result = subprocess.run(cmd, capture_output=True, text=True)

    records: list[DurationRecord] = []
    summary: dict[str, int] = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}

    output = result.stdout or ""
    raw_output = output + "\n" + (result.stderr or "")
    if result.returncode != 0 and not output:
        console.print(
            "[bold yellow]Warning:[/] Pass 2 failed.",
            highlight=False,
        )
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-5:]:
                console.print(f"  [dim]{line}[/dim]", highlight=False)
        return records, summary, raw_output

    in_durations = False
    for line in output.splitlines():
        if "slowest durations" in line.lower():
            in_durations = True
            continue

        if in_durations:
            dur_m = DURATION_RE.match(line)
            if dur_m:
                records.append(
                    DurationRecord(
                        seconds=float(dur_m.group(1)),
                        phase=dur_m.group(2),
                        nodeid=dur_m.group(3).strip(),
                    )
                )
                continue
            # A separator or different section ends the durations block
            if line.startswith("=") and in_durations and records:
                in_durations = False

        # Parse summary line (e.g. "308 passed, 3 failed, 1 skipped")
        for m in SUMMARY_RE.finditer(line):
            if m.group(1):
                summary["passed"] = int(m.group(1))
            if m.group(2):
                summary["failed"] = int(m.group(2))
            if m.group(3):
                summary["skipped"] = int(m.group(3))
            if m.group(4):
                summary["errors"] = int(m.group(4))

    console.print(
        f"  Collected [bold]{len(records)}[/bold] duration records.",
        highlight=False,
    )
    return records, summary, raw_output


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_test_timings(
    records: list[DurationRecord],
    test_fixtures: dict[str, list[str]],
    has_fixture_map: bool,
) -> list[TestTiming]:
    """Aggregate duration records into per-test timings.

    Args:
        records: Parsed duration records from Pass 2.
        test_fixtures: Mapping from nodeid to fixture names (from Pass 1).
        has_fixture_map: Whether fixture map data is available.

    Returns:
        List of :class:`TestTiming`, one per unique test nodeid.
    """
    # Group by nodeid
    by_test: dict[str, dict[str, float]] = {}
    for rec in records:
        phases = by_test.setdefault(rec.nodeid, {"setup": 0.0, "call": 0.0, "teardown": 0.0})
        phases[rec.phase] += rec.seconds

    timings: list[TestTiming] = []
    for nodeid, phases in by_test.items():
        fixtures = test_fixtures.get(nodeid, ["?"] if has_fixture_map else ["?"])
        timings.append(
            TestTiming(
                nodeid=nodeid,
                category=extract_category(nodeid),
                setup_time=phases["setup"],
                call_time=phases["call"],
                teardown_time=phases["teardown"],
                total_time=phases["setup"] + phases["call"] + phases["teardown"],
                fixtures=fixtures,
            )
        )

    timings.sort(key=lambda t: t.total_time, reverse=True)
    return timings


def aggregate_category_stats(timings: list[TestTiming]) -> list[CategoryStats]:
    """Compute per-category statistics.

    Args:
        timings: Aggregated test timings.

    Returns:
        List of :class:`CategoryStats`, sorted by total time descending.
    """
    cats: dict[str, list[TestTiming]] = {}
    for t in timings:
        cats.setdefault(t.category, []).append(t)

    stats: list[CategoryStats] = []
    for name, tests in cats.items():
        total = sum(t.total_time for t in tests)
        slowest = max(tests, key=lambda t: t.total_time)
        stats.append(
            CategoryStats(
                name=name,
                test_count=len(tests),
                total_time=total,
                mean_time=total / len(tests),
                slowest_test=slowest.nodeid,
                slowest_time=slowest.total_time,
            )
        )

    stats.sort(key=lambda s: s.total_time, reverse=True)
    return stats


def aggregate_fixture_stats(
    timings: list[TestTiming],
    test_fixtures: dict[str, list[str]],
    fixture_scopes: dict[str, str],
) -> list[FixtureStats]:
    """Compute per-fixture impact statistics.

    For each fixture, sums the setup-phase durations of all tests that use
    it, giving an "attributed setup time" estimate.

    Args:
        timings: Aggregated test timings.
        test_fixtures: Mapping from nodeid to fixture names.
        fixture_scopes: Mapping from fixture name to scope string.

    Returns:
        List of :class:`FixtureStats`, sorted by attributed setup time
        descending.
    """
    # Build lookup from nodeid -> setup_time
    setup_by_test: dict[str, float] = {t.nodeid: t.setup_time for t in timings}

    # Invert: fixture -> list of nodeids
    fixture_tests: dict[str, list[str]] = {}
    for nodeid, fixtures in test_fixtures.items():
        for fx in fixtures:
            fixture_tests.setdefault(fx, []).append(nodeid)

    stats: list[FixtureStats] = []
    for fx_name, nodeids in fixture_tests.items():
        attributed = sum(setup_by_test.get(nid, 0.0) for nid in nodeids)
        stats.append(
            FixtureStats(
                name=fx_name,
                scope=fixture_scopes.get(fx_name, "unknown"),
                test_count=len(nodeids),
                attributed_setup_time=attributed,
            )
        )

    stats.sort(key=lambda s: s.attributed_setup_time, reverse=True)
    return stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_session_summary(
    console: Console,
    total_tests: int,
    total_time: float,
    summary: dict[str, int],
) -> None:
    """Print the session summary banner.

    Args:
        console: Rich console.
        total_tests: Number of unique tests timed.
        total_time: Sum of all test total times.
        summary: Dict with passed/failed/skipped/errors counts.
    """
    console.print()
    console.rule("[bold]Session Summary")
    parts = [
        f"[bold]{total_tests}[/bold] tests",
        f"[bold]{format_duration(total_time)}[/bold] total",
    ]
    if summary.get("passed"):
        parts.append(f"[green]{summary['passed']} passed[/green]")
    if summary.get("failed"):
        parts.append(f"[red]{summary['failed']} failed[/red]")
    if summary.get("skipped"):
        parts.append(f"[yellow]{summary['skipped']} skipped[/yellow]")
    if summary.get("errors"):
        parts.append(f"[red]{summary['errors']} errors[/red]")
    console.print("  " + "  |  ".join(parts), highlight=False)
    console.print()


def print_slowest_tests(
    console: Console,
    timings: list[TestTiming],
    top_n: int,
    category_filter: str | None,
    fixture_filter: str | None,
) -> None:
    """Print the slowest-tests table.

    Args:
        console: Rich console.
        timings: All test timings (already sorted by total descending).
        top_n: Number of rows to display.
        category_filter: If set, only show tests in this category.
        fixture_filter: If set, only show tests using this fixture.
    """
    filtered = timings
    if category_filter:
        filtered = [t for t in filtered if t.category == category_filter]
    if fixture_filter:
        filtered = [t for t in filtered if fixture_filter in t.fixtures]

    display = filtered[:top_n]

    title = f"Slowest Tests (top {min(top_n, len(display))})"
    if category_filter:
        title += f" — category: {category_filter}"
    if fixture_filter:
        title += f" — fixture: {fixture_filter}"

    table = Table(title=title, show_lines=False, pad_edge=True)
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Total", justify="right", style="bold")
    table.add_column("Setup", justify="right")
    table.add_column("Call", justify="right")
    table.add_column("Teardown", justify="right")
    table.add_column("Fixtures", max_width=40, overflow="ellipsis")
    table.add_column("Test", overflow="ellipsis")

    for i, t in enumerate(display, 1):
        fx_str = ", ".join(t.fixtures) if t.fixtures else "-"
        table.add_row(
            str(i),
            format_duration(t.total_time),
            format_duration(t.setup_time),
            format_duration(t.call_time),
            format_duration(t.teardown_time),
            fx_str,
            t.nodeid,
        )

    console.print(table)
    console.print()


def print_category_breakdown(
    console: Console,
    stats: list[CategoryStats],
) -> None:
    """Print the category breakdown table.

    Args:
        console: Rich console.
        stats: Category statistics sorted by total time descending.
    """
    table = Table(title="Category Breakdown", show_lines=False, pad_edge=True)
    table.add_column("Category", style="bold")
    table.add_column("Tests", justify="right")
    table.add_column("Total Time", justify="right")
    table.add_column("Mean Time", justify="right")
    table.add_column("Slowest Time", justify="right")

    for s in stats:
        table.add_row(
            s.name,
            str(s.test_count),
            format_duration(s.total_time),
            format_duration(s.mean_time),
            format_duration(s.slowest_time),
        )

    console.print(table)
    console.print()


def print_fixture_impact(
    console: Console,
    stats: list[FixtureStats],
    top_n: int = 15,
) -> None:
    """Print the fixture impact table.

    Args:
        console: Rich console.
        stats: Fixture statistics sorted by attributed setup time descending.
        top_n: Number of rows to display.
    """
    display = stats[:top_n]

    table = Table(title=f"Fixture Impact (top {min(top_n, len(display))})", show_lines=False, pad_edge=True)
    table.add_column("Fixture", style="bold")
    table.add_column("Scope")
    table.add_column("Tests", justify="right")
    table.add_column("Attributed Setup", justify="right")

    for s in display:
        table.add_row(
            s.name,
            s.scope,
            str(s.test_count),
            format_duration(s.attributed_setup_time),
        )

    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# .benchmark/ log output
# ---------------------------------------------------------------------------

_FAILURE_HEADER_RE = re.compile(r"^={3,}\s*(FAILURES|ERRORS)\s*={3,}$")
_FAILURE_SEPARATOR_RE = re.compile(r"^_{3,}\s*(.+?)\s*_{3,}$")
_SHORT_SUMMARY_RE = re.compile(r"^={3,}\s*short test summary info\s*={3,}$")


def _parse_failures(raw_output: str) -> list[tuple[str, str]]:
    """Extract failure/error blocks from raw pytest output.

    Args:
        raw_output: Complete pytest stdout+stderr.

    Returns:
        List of ``(test_nodeid, block_text)`` tuples.
    """
    failures: list[tuple[str, str]] = []
    lines = raw_output.splitlines()
    i = 0
    while i < len(lines):
        # Find FAILURES or ERRORS section
        if _FAILURE_HEADER_RE.match(lines[i]):
            i += 1
            while i < len(lines):
                sep = _FAILURE_SEPARATOR_RE.match(lines[i])
                if sep:
                    nodeid = sep.group(1).strip()
                    block_lines: list[str] = []
                    i += 1
                    while i < len(lines):
                        if _FAILURE_SEPARATOR_RE.match(lines[i]):
                            break
                        if lines[i].startswith("="):
                            break
                        block_lines.append(lines[i])
                        i += 1
                    failures.append((nodeid, "\n".join(block_lines)))
                else:
                    if lines[i].startswith("="):
                        break
                    i += 1
        else:
            i += 1

    # Also grab short summary lines (FAILED / ERROR lines)
    in_summary = False
    for line in lines:
        if _SHORT_SUMMARY_RE.match(line):
            in_summary = True
            continue
        if in_summary:
            if line.startswith("="):
                break
            stripped = line.strip()
            if stripped.startswith(("FAILED", "ERROR")):
                # Only add if we didn't already capture the full block
                nodeid = stripped.split(" ")[1] if " " in stripped else stripped
                if not any(nodeid in f[0] for f in failures):
                    failures.append((nodeid, stripped))

    return failures


def _write_benchmark_logs(
    raw_output: str,
    failures: list[tuple[str, str]],
    console: Console,
) -> None:
    """Write full and deduplicated logs to ``.benchmark/``.

    Creates three files:

    - ``tests_full.log`` — complete pytest output
    - ``tests_errors.log`` — all failure/error blocks
    - ``tests_errors_dedup.log`` — unique error messages with counts

    Args:
        raw_output: Complete pytest stdout+stderr.
        failures: Parsed failure tuples from ``_parse_failures``.
        console: Rich console for status messages.
    """
    BENCHMARK_DIR.mkdir(exist_ok=True)

    # --- Full log ---
    full_log = BENCHMARK_DIR / "tests_full.log"
    full_log.write_text(raw_output, encoding="utf-8")

    # --- All errors ---
    errors_log = BENCHMARK_DIR / "tests_errors.log"
    error_lines: list[str] = []
    for nodeid, block in failures:
        error_lines.append(f"--- {nodeid} ---")
        error_lines.append(block)
        error_lines.append("")
    errors_log.write_text(
        "\n".join(error_lines) + "\n" if error_lines else "",
        encoding="utf-8",
    )

    # --- Deduplicated errors ---
    # Group by the last line of traceback (the actual error message)
    from collections import Counter, defaultdict

    error_msgs: Counter[str] = Counter()
    error_nodeids: dict[str, list[str]] = defaultdict(list)
    for nodeid, block in failures:
        # Use the last non-empty line as the dedup key
        last_line = ""
        for line in reversed(block.splitlines()):
            stripped = line.strip()
            if stripped:
                last_line = stripped
                break
        if not last_line:
            last_line = "(empty)"
        error_msgs[last_line] += 1
        error_nodeids[last_line].append(nodeid)

    dedup_log = BENCHMARK_DIR / "tests_errors_dedup.log"
    dedup_lines: list[str] = []
    for msg, count in error_msgs.most_common():
        dedup_lines.append(f"(x{count}) {msg}")
        for nid in error_nodeids[msg]:
            dedup_lines.append(f"    {nid}")
        dedup_lines.append("")
    dedup_log.write_text(
        "\n".join(dedup_lines) + "\n" if dedup_lines else "",
        encoding="utf-8",
    )

    console.print(
        f"\n[dim]Logs written to {BENCHMARK_DIR}/[/dim]",
        highlight=False,
    )
    console.print(
        f"  [dim]tests_full.log        "
        f"({len(raw_output.splitlines())} lines)[/dim]",
        highlight=False,
    )
    console.print(
        f"  [dim]tests_errors.log      "
        f"({len(failures)} errors)[/dim]",
        highlight=False,
    )
    console.print(
        f"  [dim]tests_errors_dedup.log "
        f"({len(error_msgs)} unique errors)[/dim]",
        highlight=False,
    )


# ---------------------------------------------------------------------------
# JSON export
# ---------------------------------------------------------------------------


def write_json(path: str, profile_result: ProfileResult) -> None:
    """Write profiling results as JSON.

    Args:
        path: Output file path.
        profile_result: Aggregated profiling data.
    """
    data = asdict(profile_result)
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        description="Two-pass pytest timing profiler with fixture attribution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "targets",
        nargs="*",
        default=[],
        help="Test paths to profile (default: uses pytest's configured testpaths).",
    )
    parser.add_argument(
        "-n",
        "--top-n",
        type=int,
        default=20,
        help="Number of slowest tests to show (default: 20).",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Filter results to a specific category (e.g. 'core', 'detect').",
    )
    parser.add_argument(
        "--fixture",
        default=None,
        help="Filter to tests using a specific fixture.",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        default=None,
        help="Write results as JSON to this path.",
    )
    parser.add_argument(
        "--skip-fixture-map",
        action="store_true",
        help="Skip Pass 1 (faster, but no fixture info in report).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output.",
    )
    parser.add_argument(
        "-k",
        dest="k_expression",
        default=None,
        help="Pass-through to pytest -k.",
    )
    parser.add_argument(
        "-x",
        "--exitfirst",
        action="store_true",
        help="Stop on first test failure.",
    )
    memory_group = parser.add_argument_group("memory profiling (opt-in)")
    memory_group.add_argument(
        "--memory",
        action="store_true",
        help=(
            "Profile memory with memray (requires pytest-memray; not available on Windows). "
            "Binary results → .benchmark/memray/. "
            "Flamegraph: memray flamegraph .benchmark/memray/<file>.bin"
        ),
    )
    memory_group.add_argument(
        "--tracemalloc",
        action="store_true",
        help=(
            "Profile memory with stdlib tracemalloc (built-in pytest flag, no extra deps). "
            "Output embedded in pytest terminal → .benchmark/tests_full.log."
        ),
    )
    return parser


def main() -> None:
    """Entry point for the pytest timing profiler."""
    parser = build_parser()
    args = parser.parse_args()

    console = Console(no_color=args.no_color)

    # Build extra pytest args from pass-through flags
    extra_args: list[str] = []
    if args.k_expression:
        extra_args.extend(["-k", args.k_expression])
    if args.exitfirst:
        extra_args.append("-x")

    targets = args.targets

    # --- Pass 1: Fixture mapping ---
    test_fixtures: dict[str, list[str]] = {}
    fixture_scopes: dict[str, str] = {}
    has_fixture_map = False

    if not args.skip_fixture_map:
        test_fixtures, fixture_scopes = run_fixture_map(targets, extra_args, console)
        has_fixture_map = bool(test_fixtures)
    else:
        console.print("[dim]Skipping Pass 1 (fixture map).[/dim]", highlight=False)

    # --- Memory profiling args (Pass 2 only) ---
    memory_args: list[str] = []
    if args.memory:
        MEMRAY_DIR.mkdir(parents=True, exist_ok=True)
        memory_args.extend(["--memray", "--memray-bin-path", str(MEMRAY_DIR)])
        console.print(
            f"[bold cyan]Memory (memray):[/] binary results → {MEMRAY_DIR}/",
            highlight=False,
        )
        console.print(
            "  [dim]Flamegraph after run: "
            f"memray flamegraph {MEMRAY_DIR}/<result.bin>[/dim]",
            highlight=False,
        )
    if args.tracemalloc:
        memory_args.append("--tracemalloc")
        console.print(
            "[bold cyan]Memory (tracemalloc):[/] output → .benchmark/tests_full.log",
            highlight=False,
        )

    # --- Pass 2: Timing ---
    records, summary, raw_output = run_timing(targets, extra_args, console, memory_args)

    if not records:
        console.print("[bold red]No timing data collected. Exiting.[/bold red]", highlight=False)
        sys.exit(1)

    # --- Aggregation ---
    timings = aggregate_test_timings(records, test_fixtures, has_fixture_map)
    total_time = sum(t.total_time for t in timings)
    category_stats = aggregate_category_stats(timings)
    fixture_stats = (
        aggregate_fixture_stats(timings, test_fixtures, fixture_scopes)
        if has_fixture_map
        else []
    )

    # --- Report ---
    print_session_summary(console, len(timings), total_time, summary)
    print_slowest_tests(console, timings, args.top_n, args.category, args.fixture)
    print_category_breakdown(console, category_stats)
    if has_fixture_map:
        print_fixture_impact(console, fixture_stats)

    # --- Benchmark logs ---
    failures = _parse_failures(raw_output)
    _write_benchmark_logs(raw_output, failures, console)

    # --- JSON export ---
    if args.json_path:
        profile_result = ProfileResult(
            total_tests=len(timings),
            total_time=total_time,
            passed=summary.get("passed", 0),
            failed=summary.get("failed", 0),
            skipped=summary.get("skipped", 0),
            errors=summary.get("errors", 0),
            test_timings=timings,
            category_stats=category_stats,
            fixture_stats=fixture_stats,
        )
        write_json(args.json_path, profile_result)
        console.print(
            f"[bold green]Results written to {args.json_path}[/bold green]",
            highlight=False,
        )


if __name__ == "__main__":
    main()
