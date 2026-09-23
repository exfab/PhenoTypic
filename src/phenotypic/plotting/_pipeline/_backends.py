"""What can render here, and the one Plotly bundle every page shares.

Kept apart from ``_writer`` so the capability probe can be imported by CLI
validation without dragging in publication.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any

from phenotypic.sdk_._file_locking import exclusive_path_lock

logger = logging.getLogger(__name__)

#: Memoised verdict of :func:`chrome_available`. ``None`` means "not yet asked".
_CHROME: bool | None = None

#: Smallest size a complete bundle can plausibly have, used to detect a
#: truncated file rather than trusting mere existence.
#:
#: Note what this does NOT guard: :func:`ensure_plotlyjs_bundle` writes to a
#: temporary sibling and ``os.replace``s it, which is atomic, so *this* function
#: cannot leave a half-written bundle behind. The floor guards what it did not
#: write -- a file truncated by a full filesystem, copied in by hand, produced
#: by a different tool, or left by an older implementation. It is NOT a guard
#: against the zero-byte remnant of an interrupted write, which ``os.replace``
#: makes impossible here.
#:
#: Measured 2026-09-21: ``len(get_plotlyjs())`` is 4,847,452 chars /
#: 4,847,499 utf-8 bytes on plotly 6.6.0 -- 4.85x this floor. The comparison is
#: against ``st_size``, i.e. BYTES, which is the correct side: the bundle is not
#: pure ASCII, so chars and bytes differ by 47.
_MIN_BUNDLE_BYTES = 1_000_000


def chrome_available() -> bool:
    """Return whether Plotly can rasterise here, probing at most once.

    Kaleido shells out to Chrome for PNG export. The probe renders a minimal
    figure because that exercises exactly what publication will do. A check for
    a browser binary answers a weaker question -- present but broken or
    sandboxed Chrome passes it and still cannot produce a PNG.

    Note this repo already has such a binary check:
    ``Chromium.find_browser(skip_local=False)`` in
    ``tests/unit/cli/_kaleido_utils.py``, behind ``requires_kaleido_chrome``.
    That is the right probe for *skipping a test* and the wrong one for
    *deciding what to publish*; the two coexist deliberately.

    Returns:
        ``True`` if a PNG can be produced, ``False`` otherwise.

    Never raises -- but note that is a weaker guarantee than always returning
    promptly. kaleido 1.2.0 bounds the *render* (``Kaleido.__init__`` defaults
    ``timeout=90``; ``calc_fig`` wraps it in ``asyncio.wait_for``), so a cleanly
    failing browser costs at most ~90 s. Everything *before* the render is
    unbounded: ``_get_kaleido_tab`` is a bare queue await and browser launch
    sits outside that wait. A present-but-hung Chrome is therefore the one input
    whose termination is unproven -- and it is exactly the input this probe was
    chosen over a binary check to catch. This runs on the submitting process
    during CLI validation, so a hang blocks submission rather than failing it.
    """
    global _CHROME
    if _CHROME is not None:
        return _CHROME

    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        pio.to_image(go.Figure(), format="png", width=8, height=8)
        _CHROME = True
    except Exception as exc:  # noqa: BLE001 - any failure means "cannot"
        logger.debug("Plotly PNG backend unavailable: %s", exc)
        _CHROME = False
    return _CHROME


def reset_chrome_probe() -> None:
    """Clear the memoised verdict. Tests only."""
    global _CHROME
    _CHROME = None


def ensure_plotlyjs_bundle(plots_base: Path) -> Path:
    """Write ``plotly.min.js`` under *plots_base* once, returning its path.

    Concurrent SLURM workers race to create it, so the write is locked and
    skipped when a complete file is already present. A short or truncated file
    is rewritten -- existence alone is not evidence of a usable bundle.

    Args:
        plots_base: Resolved ``deliverables/plots`` directory.

    Returns:
        Path to the bundle.

    Raises:
        ArtifactLockTimeout: If the bundle lock cannot be acquired. This is
            **not** swallowed: a caller that cannot obtain the bundle cannot
            write a page that references it, so failing here is correct. Note
            this makes the function unsuitable for use inside an ``except``
            block that must not raise.
    """
    from phenotypic.sdk_ import plotlyjs_bundle_path

    bundle = plotlyjs_bundle_path(plots_base)
    if _is_complete_bundle(bundle):
        return bundle

    plots_base.mkdir(parents=True, exist_ok=True)
    with exclusive_path_lock(plots_base / ".plotlyjs.lock"):
        if _is_complete_bundle(bundle):
            return bundle
        from plotly.offline import get_plotlyjs

        temporary = bundle.with_name(f".{bundle.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(get_plotlyjs(), encoding="utf-8")
            os.replace(temporary, bundle)
        finally:
            temporary.unlink(missing_ok=True)
    return bundle


def _is_complete_bundle(bundle: Path) -> bool:
    """Return whether *bundle* exists and is long enough to be usable."""
    return bundle.is_file() and bundle.stat().st_size >= _MIN_BUNDLE_BYTES


def plotlyjs_src_for(page_dir: Path, bundle: Path) -> str:
    """Return the ``src`` a page in *page_dir* uses to reach *bundle*.

    Plotly emits a string ``include_plotlyjs`` value verbatim as the script
    src, so a computed relative path hoists one bundle across every layout
    without hard-coding directory depth.

    Args:
        page_dir: Directory the HTML page will be written into.
        bundle: Path returned by :func:`ensure_plotlyjs_bundle`.

    Returns:
        A relative POSIX path such as ``"../../plotly.min.js"``.

    Note:
        Both arguments must be anchored the same way -- both absolute or both
        relative. ``os.path.relpath`` is purely lexical, so it resolves a
        relative argument against the *process cwd*, and a mixed pair yields a
        path that is both wrong and dependent on where the process was
        launched. Every caller descends from ``PlotCoordinator._plots_base``
        and so is consistently anchored; this note exists because nothing in
        the signature enforces it. Cross-mount pairs are fine -- lexical means
        no filesystem is consulted.
    """
    return Path(os.path.relpath(bundle, page_dir)).as_posix()


class PlotBackendUnavailable(RuntimeError):
    """A declared figure backend cannot be used at all."""


def preflight_plot_backends(pipeline: Any) -> list[str]:
    """Check what the configured plots can render, before any image work.

    Missing Chrome is **not** an error: Plotly publishes HTML regardless, so
    the run is complete either way and the caller is told what it will not get.
    A declared backend whose library will not import IS an error -- that plot
    cannot publish at all.

    Bindings fall into three groups by the backend of the figure their
    ``inspect()`` renders (see :func:`_declared_backends`). A binding whose
    backend is not declared -- most plots that override ``inspect`` directly
    -- has a backend decided only at render time, so it is named
    conditionally in the Chrome warning and never fails the import check.

    :func:`chrome_available` is called only when some binding declares
    ``plotly`` or declares nothing; a pipeline that can only produce
    matplotlib output never launches a browser.

    Args:
        pipeline: Pipeline whose normalized plot bindings are inspected.

    Returns:
        Human-readable warning lines, empty when everything is available.

    Raises:
        PlotBackendUnavailable: If a declared backend's library is missing.
    """
    from phenotypic.abc_.plotting import PlotImage

    plotly_ids: list[str] = []
    mpl_ids: list[str] = []
    undeclared_ids: list[str] = []
    for binding in pipeline.get_plots():
        if isinstance(binding.plot, PlotImage):
            spec = declared_figure_spec(binding.plot)
            # An image plot renders PNG only if it declared it (spec §2);
            # an undeclared one stores its backend default, which for Plotly
            # needs no Chrome.
            if spec is not None and spec.backend == "plotly" and "png" in spec.store:
                plotly_ids.append(binding.id)
            elif spec is not None and spec.backend == "mpl":
                mpl_ids.append(binding.id)
            elif spec is not None:
                _require_importable("plotly", "plotly", [binding.id])
            continue
        backend = _declared_backends(binding.plot)
        if backend == "plotly":
            plotly_ids.append(binding.id)
        elif backend == "mpl":
            mpl_ids.append(binding.id)
        else:
            undeclared_ids.append(binding.id)

    _require_importable("matplotlib", "mpl", mpl_ids)
    _require_importable("plotly", "plotly", plotly_ids)

    if not (plotly_ids or undeclared_ids) or chrome_available():
        return []
    parts = ["Chrome is not available;"]
    if plotly_ids:
        parts.append(
            f"{len(plotly_ids)} Plotly plots will publish HTML only, without "
            f"PNG: {', '.join(plotly_ids)}."
        )
    if undeclared_ids:
        parts.append(
            f"{len(undeclared_ids)} {'more ' if plotly_ids else ''}plots "
            "declare no figure backend and will publish HTML only, without "
            f"PNG, if they return Plotly: {', '.join(undeclared_ids)}."
        )
    parts.append("Install it for raster output with:  plotly_get_chrome")
    return [" ".join(parts)]


def _declared_backends(plot: Any) -> str | None:
    """Return the backend *plot*'s ``inspect()`` declares it renders, if any.

    ``inspect()`` publishes ONE figure, so a binding is classified by that
    figure alone -- not by the union of its ``@figure`` methods, which would
    name a plot with an mpl primary and a Plotly secondary as "HTML only,
    without PNG" when it publishes a PNG. Three steps, in order:

    1. The effective ``inspect`` itself carries ``@figure``: its declared
       backend. ``MeasureSymZones`` and ``MeasureOrientationZones`` decorate
       their override, so the override IS the declaration, enforced by the
       wrapper at render time.
    2. The effective ``inspect`` is :meth:`PhtPlot.inspect`: the primary
       figure's backend, since that is what it renders. A class with no
       selectable primary (none declared, or several and none marked) raises
       from ``inspect`` anyway, and is undeclared here.
    3. Any other override: ``None``. It renders what it chooses at run time,
       whatever its ``@figure`` methods declare.

    A QC-recipe binding holds its ``QcRecipeEntry``: the ``PlotQc`` instance
    exists only once the QC runner has analyzed the check, so the same rule is
    applied to ``entry.cls`` without instantiating it.
    normalize_plot_bindings admits no other non-PhtPlot, and requires ``cls``
    to subclass PlotQc.
    """
    spec = declared_figure_spec(plot)
    return spec.backend if spec is not None else None


def declared_figure_spec(plot: Any) -> Any:
    """Return the ``FigureSpec`` *plot*'s ``inspect()`` renders, if declared.

    The three-step rule documented on :func:`_declared_backends`, returning
    the spec rather than its backend so a caller can also read ``spec.store``.
    """
    from phenotypic.abc_.plotting import PhtPlot

    if isinstance(plot, PhtPlot):
        owner: Any = type(plot)
        primary_spec = plot._primary_spec
    else:
        owner = plot.cls
        primary_spec = owner._class_primary_spec
    effective_inspect = owner.inspect
    declared = getattr(effective_inspect, "__figure_spec__", None)
    if declared is not None:
        return declared
    if effective_inspect is not PhtPlot.inspect:
        return None
    try:
        return primary_spec()
    except RuntimeError:
        return None


def _require_importable(module: str, backend: str, binding_ids: list[str]) -> None:
    """Raise :class:`PlotBackendUnavailable` if *binding_ids* need a missing *module*."""
    if not binding_ids:
        return
    try:
        __import__(module)
    except ImportError as exc:
        raise PlotBackendUnavailable(
            f"{len(binding_ids)} configured plots declare backend={backend!r} "
            f"but {module} is not importable: {', '.join(binding_ids)}"
        ) from exc


__all__ = [
    "PlotBackendUnavailable",
    "chrome_available",
    "ensure_plotlyjs_bundle",
    "plotlyjs_src_for",
    "preflight_plot_backends",
    "reset_chrome_probe",
]
