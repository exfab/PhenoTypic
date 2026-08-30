"""Shared store-driven overlay discovery and rendering.

Both ``--mode migrate`` and ``--mode recompile`` need to recreate a missing
PNG overlay from a promoted OME-Zarr store.  This module owns the common
discovery, writer construction, and bounded thread-pool execution.  Callers
retain their different publication policies: migration publishes all image
markers after rendering, while recompile may repair an existing marker in the
same guarded transaction as the overlay.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from phenotypic._cli._cli_directory_scanner import scan_store_outputs
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_utils import resolve_local_worker_count
from phenotypic.sdk_ import load_image_from_store, store_stem


@dataclass(frozen=True)
class OverlayWork:
    """One missing overlay backed by a promoted image store."""

    dataset: str
    store: Path
    overlay: Path

    @property
    def stem(self) -> str:
        """Return the image stem encoded by :attr:`store`."""
        return store_stem(self.store)


@dataclass(frozen=True)
class OverlayRenderReport:
    """Result of one shared overlay rendering pass."""

    rendered: int = 0
    skipped: int = 0
    failures: tuple[tuple[Path, str], ...] = ()


def valid_migration_overlay(
    path: Path, expected_shape: tuple[int, ...]
) -> bool:
    """Return whether *path* is a decoded full-plane PNG overlay.

    Pillow's ``verify`` catches truncated payloads without decoding pixels;
    reopening and loading then proves the payload is actually readable. Image
    arrays use ``(height, width, ...)`` while Pillow exposes ``(width, height)``.
    """
    from PIL import Image as PILImage

    if len(expected_shape) < 2:
        return False
    height, width = int(expected_shape[0]), int(expected_shape[1])
    if height <= 0 or width <= 0:
        return False
    try:
        with PILImage.open(path) as image:
            if image.format != "PNG":
                return False
            image.verify()
        with PILImage.open(path) as image:
            if image.format != "PNG":
                return False
            image.load()
            return (
                image.width > 0
                and image.height > 0
                and image.size == (width, height)
            )
    except Exception:  # noqa: BLE001 - every decode failure means invalid
        return False


def overlay_output_manager(
    output_dir: Path, *, overlay_alpha: float
) -> OutputManager:
    """Build the forward-run overlay writer for an existing output tree."""
    return OutputManager.from_config(
        base_dir=Path(output_dir),
        ext=".png",
        include_dataset_column=False,
        overlay_alpha=overlay_alpha,
        save_overlays=True,
    )


def discover_missing_overlays(
    output_dir: Path, output_manager: OutputManager
) -> tuple[list[OverlayWork], int]:
    """Return missing store-backed overlays and the number already present.

    Args:
        output_dir: Existing run output root.
        output_manager: Writer used to resolve canonical overlay paths.

    Returns:
        ``(work, skipped)`` in deterministic dataset/stem order.

    Raises:
        ValueError: No image stores exist below ``results/``.
    """
    work: list[OverlayWork] = []
    skipped = 0
    for dataset in scan_store_outputs(Path(output_dir)):
        for store in dataset.images:
            overlay = output_manager.get_output_path(
                dataset.name, "overlays", store_stem(store)
            )
            if overlay.is_file():
                skipped += 1
                continue
            work.append(
                OverlayWork(
                    dataset=dataset.name,
                    store=Path(store),
                    overlay=overlay,
                )
            )
    return work, skipped


def render_overlay_work(
    work: Sequence[OverlayWork],
    *,
    output_manager: OutputManager,
    n_jobs: int,
    render_one: Callable[[OverlayWork, OutputManager], None] | None = None,
) -> OverlayRenderReport:
    """Render missing overlays with the shared bounded thread-pool policy.

    Args:
        work: Missing overlays returned by :func:`discover_missing_overlays`.
        output_manager: Writer configured with the requested alpha.
        n_jobs: Requested worker count. ``-1`` uses allocated/host CPUs.
        render_one: Optional caller policy around one render. The default
            loads the store and calls :meth:`OutputManager.save_overlay`.

    Returns:
        Rendered count and per-overlay failures. Exceptions are reported, not
        raised, so callers can include every failed image in their summary.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not work:
        return OverlayRenderReport()

    for item in work:
        item.overlay.parent.mkdir(parents=True, exist_ok=True)

    def _default_render(
        item: OverlayWork, manager: OutputManager
    ) -> None:
        image = load_image_from_store(item.store)
        manager.save_overlay(image, item.dataset, item.stem)

    renderer = render_one or _default_render
    failures: list[tuple[Path, str]] = []
    rendered = 0
    workers = resolve_local_worker_count(n_jobs, len(work))
    if workers == 1:
        for item in work:
            try:
                renderer(item, output_manager)
            except Exception as exc:  # noqa: BLE001 - reported per image
                failures.append(
                    (item.overlay, f"{type(exc).__name__}: {exc}")
                )
            else:
                rendered += 1
        return OverlayRenderReport(
            rendered=rendered, failures=tuple(failures)
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(renderer, item, output_manager): item for item in work
        }
        for future in as_completed(futures):
            item = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - reported per image
                failures.append(
                    (item.overlay, f"{type(exc).__name__}: {exc}")
                )
            else:
                rendered += 1
    return OverlayRenderReport(
        rendered=rendered,
        failures=tuple(sorted(failures, key=lambda item: str(item[0]))),
    )


__all__ = [
    "OverlayRenderReport",
    "OverlayWork",
    "discover_missing_overlays",
    "overlay_output_manager",
    "render_overlay_work",
    "valid_migration_overlay",
]
