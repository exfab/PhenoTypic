"""What can render here, and the one Plotly bundle every page shares.

Kept apart from ``_writer`` so the capability probe can be imported by CLI
validation without dragging in publication.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from phenotypic.sdk_._file_locking import exclusive_path_lock

logger = logging.getLogger(__name__)

#: Memoised verdict of :func:`chrome_available`. ``None`` means "not yet asked".
_CHROME: bool | None = None

#: Smallest size a complete bundle can plausibly have, used to detect a
#: truncated or half-written file rather than trusting mere existence. A bare
#: ``is_file()`` check would accept the zero-byte remnant of an interrupted
#: write as a finished bundle, and every page pointed at it would load nothing.
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
        ``True`` if a PNG can be produced, ``False`` otherwise. Never raises.
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
    """
    return Path(os.path.relpath(bundle, page_dir)).as_posix()


__all__ = [
    "chrome_available",
    "ensure_plotlyjs_bundle",
    "plotlyjs_src_for",
    "reset_chrome_probe",
]
