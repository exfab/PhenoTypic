"""Sphinx build profiler for PhenoTypic documentation.

Runs the full Sphinx build pipeline while timing each step and
categorizing warnings. Uses ``sphinx.ext.duration`` to identify the
slowest documents and ``rich`` for formatted terminal output.

Usage::

    pixi run python scripts/profile_docs.py
    pixi run python scripts/profile_docs.py --top-n 30 --json report.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Generator

from rich.console import Console
from rich.table import Table

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
SOURCE_DIR = DOCS_DIR / "source"
BUILD_DIR = DOCS_DIR / "build"
CONF_PY = SOURCE_DIR / "conf.py"
BENCHMARK_DIR = PROJECT_ROOT / ".benchmark"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SphinxWarning:
    """A single warning emitted by Sphinx."""

    file: str | None
    line: int | None
    category: str
    message: str


@dataclass
class StepTiming:
    """Wall-clock duration for one build step."""

    name: str
    seconds: float
    skipped: bool = False


@dataclass
class DocDuration:
    """Per-document duration reported by ``sphinx.ext.duration``."""

    document: str
    seconds: float
    phase: str  # "reading" or "writing"


@dataclass
class BuildResult:
    """Aggregated result of a profiled Sphinx build."""

    status: str  # "success" or "failure"
    return_code: int
    total_time: float
    steps: list[StepTiming] = field(default_factory=list)
    durations: list[DocDuration] = field(default_factory=list)
    warnings: list[SphinxWarning] = field(default_factory=list)
    error_output: str = ""
    raw_output: str = ""


# ---------------------------------------------------------------------------
# Warning categorisation
# ---------------------------------------------------------------------------

_WARNING_CATEGORIES: list[tuple[str, re.Pattern[str]]] = [
    ("reference", re.compile(r"undefined label|unresolved reference|unknown document", re.I)),
    ("toctree", re.compile(r"not included in any toctree|toctree", re.I)),
    ("autodoc", re.compile(r"failed to import|duplicate object description", re.I)),
    ("intersphinx", re.compile(r"inventory|intersphinx", re.I)),
    ("notebook", re.compile(r"nbsphinx|kernel|execution|ipynb", re.I)),
    ("type-hint", re.compile(r"class reference target not found|py:class|py:obj", re.I)),
    ("deprecation", re.compile(r"deprecated|RemovedIn", re.I)),
]

_WARNING_LINE_RE = re.compile(
    r"^(?:(?P<file>.+?)"          # optional file path
    r"(?::(?P<line>\d+))?"        # optional line number
    r":\s*)?"
    r"WARNING:\s*(?P<message>.+)$"
)


def _categorise_warning(message: str) -> str:
    """Return the warning bucket for *message*."""
    for name, pattern in _WARNING_CATEGORIES:
        if pattern.search(message):
            return name
    return "other"


def _parse_warnings(warning_file: Path) -> list[SphinxWarning]:
    """Parse a Sphinx ``-w`` warning file into categorised warnings."""
    warnings: list[SphinxWarning] = []
    text = warning_file.read_text(encoding="utf-8", errors="replace")
    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        m = _WARNING_LINE_RE.match(raw_line)
        if m is None:
            # Not a recognised warning line — skip silently.
            continue
        file_path = m.group("file")
        line_no_str = m.group("line")
        message = m.group("message")
        category = _categorise_warning(message)
        warnings.append(
            SphinxWarning(
                file=file_path if file_path else None,
                line=int(line_no_str) if line_no_str else None,
                category=category,
                message=message,
            )
        )
    return warnings


# ---------------------------------------------------------------------------
# Duration parsing
# ---------------------------------------------------------------------------

_DURATION_HEADER_RE = re.compile(r"={3,}\s*slowest\s+(reading|writing)\s+durations\s*={3,}", re.I)
_DURATION_LINE_RE = re.compile(r"^\s*(?P<seconds>[\d.]+)\s+(?P<doc>\S.*)$")


def _parse_durations(build_output: str) -> list[DocDuration]:
    """Extract per-document durations from Sphinx build output."""
    durations: list[DocDuration] = []
    current_phase: str | None = None

    for line in build_output.splitlines():
        header_match = _DURATION_HEADER_RE.search(line)
        if header_match:
            current_phase = header_match.group(1).lower()
            continue

        if current_phase is None:
            continue

        dur_match = _DURATION_LINE_RE.match(line)
        if dur_match:
            durations.append(
                DocDuration(
                    document=dur_match.group("doc").strip(),
                    seconds=float(dur_match.group("seconds")),
                    phase=current_phase,
                )
            )
        elif line.strip() == "" or line.startswith("="):
            # Blank line or a new header ends the current section.
            if line.startswith("=") and not header_match:
                current_phase = None

    return durations


# ---------------------------------------------------------------------------
# conf.py patching
# ---------------------------------------------------------------------------

_DURATION_EXT = '"sphinx.ext.duration"'


@contextmanager
def _patch_confpy() -> Generator[None, None, None]:
    """Temporarily inject ``sphinx.ext.duration`` into *conf.py*.

    The original file content is restored in a ``finally`` block so the
    patch is reverted even when the build fails.
    """
    original = CONF_PY.read_text(encoding="utf-8")
    if "sphinx.ext.duration" in original:
        # Already present — nothing to do.
        yield
        return

    # Insert right after the opening bracket of the extensions list.
    patched = original.replace(
        'extensions = [\n',
        f'extensions = [\n    {_DURATION_EXT},\n',
        1,
    )
    CONF_PY.write_text(patched, encoding="utf-8")
    try:
        yield
    finally:
        CONF_PY.write_text(original, encoding="utf-8")


# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------


def _run_step(
    name: str,
    cmd: list[str],
    *,
    cwd: Path = DOCS_DIR,
    capture: bool = False,
) -> tuple[StepTiming, subprocess.CompletedProcess[str]]:
    """Run a subprocess, time it, and return the timing plus the result."""
    t0 = time.perf_counter()
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else subprocess.STDOUT,
    )
    elapsed = time.perf_counter() - t0
    return StepTiming(name=name, seconds=elapsed), result


def _run_apidoc() -> StepTiming:
    """Run ``sphinx-apidoc`` and remove the top-level module stub."""
    timing, _ = _run_step(
        "apidoc",
        [
            "sphinx-apidoc",
            "-o", str(SOURCE_DIR / "api_reference"),
            str(PROJECT_ROOT / "src" / "phenotypic"),
            "--module-first",
            "--separate",
            "--no-toc",
        ],
        cwd=DOCS_DIR,
    )
    stub = SOURCE_DIR / "api_reference" / "phenotypic.rst"
    if stub.exists():
        stub.unlink()
    return timing


def _run_clean() -> StepTiming:
    """Run ``sphinx-build -M clean``."""
    timing, _ = _run_step(
        "clean",
        ["sphinx-build", "-M", "clean", "source", "build"],
        cwd=DOCS_DIR,
    )
    return timing


def _run_html_build(warning_file: Path) -> tuple[StepTiming, subprocess.CompletedProcess[str]]:
    """Run the main HTML build with nitpicky mode and warning capture."""
    return _run_step(
        "html",
        [
            "sphinx-build",
            "-n",
            "-b", "html",
            "source",
            "build/html",
            "-w", str(warning_file),
        ],
        cwd=DOCS_DIR,
        capture=True,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def profile_build(
    *,
    skip_apidoc: bool = False,
    skip_clean: bool = False,
) -> BuildResult:
    """Execute the full Sphinx build pipeline and collect profiling data.

    Args:
        skip_apidoc: Skip the ``sphinx-apidoc`` regeneration step.
        skip_clean: Skip the ``sphinx-build -M clean`` step.

    Returns:
        A ``BuildResult`` with timings, durations, and categorised warnings.
    """
    steps: list[StepTiming] = []
    total_t0 = time.perf_counter()

    # 1. apidoc
    if skip_apidoc:
        steps.append(StepTiming(name="apidoc", seconds=0.0, skipped=True))
    else:
        steps.append(_run_apidoc())

    # 2. clean
    if skip_clean:
        steps.append(StepTiming(name="clean", seconds=0.0, skipped=True))
    else:
        steps.append(_run_clean())

    # 3. HTML build (with duration extension injected)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="sphinx_warnings_"
    ) as wf:
        warning_path = Path(wf.name)

    with _patch_confpy():
        html_timing, html_result = _run_html_build(warning_path)

    steps.append(html_timing)

    total_time = time.perf_counter() - total_t0

    # Combine stdout and stderr for duration parsing.
    build_output = (html_result.stdout or "") + "\n" + (html_result.stderr or "")

    durations = _parse_durations(build_output)
    warnings = _parse_warnings(warning_path)

    # Clean up temp file.
    warning_path.unlink(missing_ok=True)

    status = "success" if html_result.returncode == 0 else "failure"
    error_output = build_output if html_result.returncode != 0 else ""

    return BuildResult(
        status=status,
        return_code=html_result.returncode,
        total_time=total_time,
        steps=steps,
        durations=durations,
        warnings=warnings,
        error_output=error_output,
        raw_output=build_output,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_report(result: BuildResult, *, top_n: int, console: Console) -> None:
    """Render the profiling report to the terminal.

    Args:
        result: The completed build result.
        top_n: Number of slowest documents to display.
        console: Rich console instance.
    """
    # -- Build summary -------------------------------------------------------
    console.rule("[bold]Build Summary[/bold]")

    status_style = "green" if result.status == "success" else "red bold"
    console.print(f"  Status:     [{status_style}]{result.status}[/{status_style}]")
    console.print(f"  Return code: {result.return_code}")
    console.print(f"  Total time:  {result.total_time:.2f}s")
    console.print()

    step_table = Table(title="Step Timings", show_lines=False)
    step_table.add_column("Step", style="cyan")
    step_table.add_column("Time (s)", justify="right")
    step_table.add_column("Status")
    for step in result.steps:
        if step.skipped:
            step_table.add_row(step.name, "-", "[dim]skipped[/dim]")
        else:
            step_table.add_row(step.name, f"{step.seconds:.2f}", "[green]done[/green]")
    console.print(step_table)
    console.print()

    # -- Slowest documents ---------------------------------------------------
    if result.durations:
        console.rule("[bold]Slowest Documents[/bold]")
        sorted_durations = sorted(result.durations, key=lambda d: d.seconds, reverse=True)
        shown = sorted_durations[:top_n]

        dur_table = Table(title=f"Top {min(top_n, len(shown))} Slowest Documents")
        dur_table.add_column("Rank", justify="right", style="dim")
        dur_table.add_column("Time (s)", justify="right", style="bold")
        dur_table.add_column("Phase")
        dur_table.add_column("Document", style="cyan")
        for i, d in enumerate(shown, 1):
            phase_style = "blue" if d.phase == "reading" else "magenta"
            dur_table.add_row(
                str(i),
                f"{d.seconds:.3f}",
                f"[{phase_style}]{d.phase}[/{phase_style}]",
                d.document,
            )
        console.print(dur_table)
        console.print()
    else:
        console.print("[yellow]No document durations captured.[/yellow]")
        console.print()

    # -- Warning summary -----------------------------------------------------
    if result.warnings:
        console.rule("[bold]Warning Summary[/bold]")

        category_counts: dict[str, int] = {}
        for w in result.warnings:
            category_counts[w.category] = category_counts.get(w.category, 0) + 1

        sorted_cats = sorted(category_counts.items(), key=lambda kv: kv[1], reverse=True)
        warn_table = Table(title=f"Warnings by Category ({len(result.warnings)} total)")
        warn_table.add_column("Category", style="yellow")
        warn_table.add_column("Count", justify="right", style="bold")
        for cat, count in sorted_cats:
            warn_table.add_row(cat, str(count))
        console.print(warn_table)
        console.print()

        # -- Warning details -------------------------------------------------
        console.rule("[bold]Warning Details[/bold]")
        grouped: dict[str, list[SphinxWarning]] = {}
        for w in result.warnings:
            grouped.setdefault(w.category, []).append(w)

        for cat, cat_warnings in sorted(grouped.items(), key=lambda kv: -len(kv[1])):
            console.print(f"  [bold yellow]{cat}[/bold yellow] ({len(cat_warnings)})")
            for w in cat_warnings:
                loc = ""
                if w.file:
                    loc = f"{w.file}"
                    if w.line is not None:
                        loc += f":{w.line}"
                    loc += ": "
                console.print(f"    {loc}{w.message}")
            console.print()
    else:
        console.print("[green]No warnings.[/green]")

    # -- Error output (on failure) -------------------------------------------
    if result.error_output:
        console.rule("[bold red]Build Error Output[/bold red]")
        console.print(result.error_output)


def _write_json_report(result: BuildResult, path: Path) -> None:
    """Serialise the build result to a JSON file.

    Args:
        result: The completed build result.
        path: Destination file path.
    """
    data = asdict(result)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# .benchmark/ log output
# ---------------------------------------------------------------------------


def _format_warning_line(w: SphinxWarning) -> str:
    """Format a single warning as a log line."""
    loc = ""
    if w.file:
        loc = w.file
        if w.line is not None:
            loc += f":{w.line}"
        loc += ": "
    return f"[{w.category}] {loc}{w.message}"


def _write_benchmark_logs(
    result: BuildResult,
    build_output: str,
    console: Console,
) -> None:
    """Write full and deduplicated logs to ``.benchmark/``.

    Creates three files:

    - ``docs_full.log`` — complete sphinx-build stdout/stderr
    - ``docs_warnings.log`` — all warnings, one per line
    - ``docs_warnings_dedup.log`` — unique warnings with counts and
      locations

    Args:
        result: The completed build result.
        build_output: Raw sphinx-build stdout+stderr text.
        console: Rich console for status messages.
    """
    BENCHMARK_DIR.mkdir(exist_ok=True)

    # --- Full build log ---
    full_log = BENCHMARK_DIR / "docs_full.log"
    full_log.write_text(build_output, encoding="utf-8")

    # --- All warnings (one per line) ---
    warnings_log = BENCHMARK_DIR / "docs_warnings.log"
    lines = [_format_warning_line(w) for w in result.warnings]
    warnings_log.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

    # --- Deduplicated warnings ---
    dedup_log = BENCHMARK_DIR / "docs_warnings_dedup.log"
    # Group by (category, message) — location varies but the issue is the same
    from collections import Counter, defaultdict

    msg_counts: Counter[tuple[str, str]] = Counter()
    msg_locations: dict[tuple[str, str], list[str]] = defaultdict(list)
    for w in result.warnings:
        key = (w.category, w.message)
        msg_counts[key] += 1
        loc = ""
        if w.file:
            loc = w.file
            if w.line is not None:
                loc += f":{w.line}"
        if loc:
            msg_locations[key].append(loc)

    dedup_lines: list[str] = []
    for (cat, msg), count in msg_counts.most_common():
        dedup_lines.append(f"[{cat}] (x{count}) {msg}")
        locs = msg_locations[(cat, msg)]
        if locs:
            for loc in sorted(set(locs)):
                dedup_lines.append(f"    {loc}")
        dedup_lines.append("")
    dedup_log.write_text(
        "\n".join(dedup_lines) + "\n" if dedup_lines else "",
        encoding="utf-8",
    )

    console.print(
        f"\n[dim]Logs written to {BENCHMARK_DIR}/[/dim]"
    )
    console.print(
        f"  [dim]docs_full.log          "
        f"({len(build_output.splitlines())} lines)[/dim]"
    )
    console.print(
        f"  [dim]docs_warnings.log      "
        f"({len(result.warnings)} warnings)[/dim]"
    )
    console.print(
        f"  [dim]docs_warnings_dedup.log "
        f"({len(msg_counts)} unique warnings)[/dim]"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Profile a full Sphinx documentation build.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of slowest documents to display (default: 20).",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        dest="json_path",
        metavar="PATH",
        help="Write a JSON report to this file.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        default=False,
        help="Disable colored terminal output.",
    )
    parser.add_argument(
        "--skip-apidoc",
        action="store_true",
        default=False,
        help="Skip the sphinx-apidoc regeneration step.",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        default=False,
        help="Skip the sphinx-build clean step.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the Sphinx build profiler."""
    args = _parse_args()
    console = Console(no_color=args.no_color)

    console.print("[bold]Starting Sphinx build profiler...[/bold]\n")

    result = profile_build(
        skip_apidoc=args.skip_apidoc,
        skip_clean=args.skip_clean,
    )

    _print_report(result, top_n=args.top_n, console=console)
    _write_benchmark_logs(result, result.raw_output, console)

    if args.json_path is not None:
        _write_json_report(result, args.json_path)
        console.print(f"\n[dim]JSON report written to {args.json_path}[/dim]")


if __name__ == "__main__":
    main()
