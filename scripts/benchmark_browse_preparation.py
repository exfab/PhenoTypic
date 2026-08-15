"""Reproducible local cold/warm benchmark for Browse preparation."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from PIL import Image as PILImage

from phenotypic.gui.browse._cache import BrowseCache, CacheLocation
from phenotypic.gui.browse._preparation import BrowsePreparationManager
from phenotypic.gui.browse._source_probe import probe_source
from phenotypic.gui.results_viewer._dzi_tiler import DZI_BACKEND_INFO


def benchmark_browse_preparation() -> None:
    """Measure preparation-to-ready latency for cold and persistent-cache paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Cold/warm samples per generated fixture (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; stdout is always populated.",
    )
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    with tempfile.TemporaryDirectory(
        prefix="phenotypic-browse-benchmark-"
    ) as raw:
        root = Path(raw)
        fixtures = _write_fixtures(root / "sources")
        results = [
            _benchmark_fixture(root, fixture, args.iterations)
            for fixture in fixtures
        ]

    report: dict[str, Any] = {
        "schema": 1,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "backend": {
            "name": DZI_BACKEND_INFO.name,
            "version": DZI_BACKEND_INFO.version,
            "fallback_reason": DZI_BACKEND_INFO.fallback_reason,
        },
        "tile_size": 254,
        "overlap": 1,
        "iterations": args.iterations,
        "fixtures": results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")


def _write_fixtures(root: Path) -> list[Path]:
    root.mkdir(parents=True)
    fixtures = []
    for suffix, size in (
        ("png", (640, 480)),
        ("jpg", (1024, 768)),
        ("tiff", (1280, 960)),
    ):
        path = root / f"generated.{suffix}"
        PILImage.new("RGB", size, (48, 96, 144)).save(path)
        fixtures.append(path)
    return fixtures


def _benchmark_fixture(
    root: Path, source: Path, iterations: int
) -> dict[str, Any]:
    cold_ms: list[float] = []
    warm_ms: list[float] = []
    dimensions: tuple[int | None, int | None] = (None, None)
    for index in range(iterations):
        cache = BrowseCache(
            CacheLocation(
                root / f"cache-{source.suffix[1:]}-{index}", "temporary", False
            )
        )
        revision = probe_source(
            source,
            sandbox_root=source.parent,
            relative_path=source.name,
        )
        dimensions = revision.width, revision.height
        manager = BrowsePreparationManager(cache)
        started = time.perf_counter()
        handle = manager.replace_selected("benchmark", index, revision)
        if not handle.complete.wait(300) or handle.snapshot().phase != "ready":
            raise RuntimeError("cold Browse preparation failed")
        cold_ms.append((time.perf_counter() - started) * 1000)
        manager.close()

        warm_manager = BrowsePreparationManager(cache)
        started = time.perf_counter()
        warm = warm_manager.replace_selected("benchmark", index, revision)
        if not warm.complete.wait(30) or warm.snapshot().phase != "ready":
            raise RuntimeError("warm Browse preparation failed")
        warm_ms.append((time.perf_counter() - started) * 1000)
        warm_manager.close()
    return {
        "name": source.suffix.lower().lstrip("."),
        "dimensions": list(dimensions),
        "cold_ms": _summary(cold_ms),
        "warm_ms": _summary(warm_ms),
    }


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "p50": round(statistics.median(ordered), 3),
        "p95": round(
            ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3
        ),
    }


if __name__ == "__main__":
    benchmark_browse_preparation()
