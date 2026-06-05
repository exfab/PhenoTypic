"""Path → ``Capabilities`` classifier.

Pure function, file-stat only, no parsing of file contents (other than a
4 KB peek for pipeline-json detection). The classifier runs on every visible
sidebar node and every Recent Runs row, so it must stay cheap.

Output is consumed by:
    * The sidebar (``img`` / ``cfg`` / ``out`` / ``?`` badges; see ``_sidebar.py``).
    * The home page (sandbox capability summary; see ``_home.py``).
    * The Run console's Recent Runs panel (``has_dashboard`` enables the
      iframe link).

Caching: results are memoised on ``(absolute_path, mtime_ns)`` via an internal
LRU. The "Refresh" button in the sidebar calls ``invalidate_cache()`` to flush;
no filesystem watcher in v1. The cache keys on the *root* dir's mtime, so a CLI
artifact that lands inside ``<root>/deliverables/`` (e.g. ``dashboard.html`` or
``master_measurements.parquet``, both of which now live under that subdir) does
not bump the root mtime and may need a sidebar Refresh to surface — matching the
existing behaviour for files written under ``results/``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from phenotypic.gui._config import (
    DASHBOARD_FILENAME,
    DELIVERABLES_DIRNAME,
    MASTER_MEASUREMENTS_PARQUET,
    RESULTS_DIRNAME,
)
from phenotypic.gui.builder._directory_browser import IMAGE_EXTS

__all__ = ["Capabilities", "classify", "invalidate_cache"]


# Markers that identify a CLI-output directory. The master parquet name is
# centralised in ``gui/_config.py`` so the shell classifier and the results
# viewer's ``_output_root`` stay in lockstep without either importing the
# other (which would re-introduce the historical circular dependency).
#
# Layout note: ``master_measurements.parquet`` and ``dashboard.html`` now live
# under ``<root>/deliverables/`` (not the output root itself), so the dir
# classifier stats them inside that subdir. Only ``results/`` stays at the
# output root.
_MASTER_MEASUREMENTS_FILENAME = MASTER_MEASUREMENTS_PARQUET
_RESULTS_DIRNAME = RESULTS_DIRNAME
_DASHBOARD_FILENAME = DASHBOARD_FILENAME

# Image-count cap. Surfaces "many images" without paying for a full scan when
# a folder holds tens of thousands of plate images.
_IMAGE_COUNT_CAP = 1000

# Pipeline-json peek constants. Cheap and good-enough heuristic.
# Match either ``"pipe_cfgs"`` (what ``ImagePipeline.to_json`` writes today)
# or the legacy ``"operations"`` key so older pipelines exported before the
# rename still light up the ``cfg`` badge.
_PIPELINE_JSON_PEEK_BYTES = 4096
_PIPELINE_JSON_MARKERS: tuple[bytes, ...] = (b'"pipe_cfgs"', b'"operations"')


@dataclass(frozen=True)
class Capabilities:
    """Capability summary for a single path.

    Capability fields are mutually non-exclusive — a directory can be
    ``is_image_dir`` *and* contain a pipeline JSON, in which case the sidebar
    will show both ``img`` and ``cfg`` badges.

    Attributes:
        is_image_dir: Directory contains at least one file whose extension is
            in :data:`IMAGE_EXTS`.
        has_pipeline_json: Path is a JSON file whose first 4 KB contains the
            ``"operations"`` key. The marker is what
            :class:`phenotypic.ImagePipeline.to_json` writes; cheap heuristic.
        is_cli_output: Directory contains both
            ``deliverables/master_measurements.parquet`` and a ``results/``
            subdirectory — the layout produced by ``python -m phenotypic``.
            The master marker lives under ``deliverables/``; ``results/``
            stays at the output root.
        has_dashboard: Directory contains ``deliverables/dashboard.html``.
            Used by the Run console's Recent Runs panel to enable the
            iframe link.
        is_process_only_output: Directory is a ``--process-only`` run — it
            carries a ``.phenotypic/progress/manifest.json`` (machine-state
            under the hidden cache) but lacks the full forward-run
            ``results/`` + ``deliverables/master_measurements.parquet``
            markers. Lets the Run console list a process-only run (D13) even
            though it has no dashboard / results affordance.
        image_count: Image-file count (capped at :data:`_IMAGE_COUNT_CAP`)
            when ``is_image_dir`` is true; ``None`` otherwise.
        bad_perms: ``True`` if listing the directory raised
            :class:`PermissionError`. Surfaces the ``?`` badge.
    """

    is_image_dir: bool
    has_pipeline_json: bool
    is_cli_output: bool
    has_dashboard: bool
    is_process_only_output: bool
    image_count: int | None
    bad_perms: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(path: str | os.PathLike[str]) -> Capabilities:
    """Classify ``path`` into a :class:`Capabilities` summary.

    The result is LRU-cached on ``(absolute_path, mtime_ns)``. A subsequent
    call returns the cached value when the path's mtime is unchanged; an
    mtime bump (e.g. a new file dropped into the directory) bypasses the
    cache transparently.

    Missing or unreadable paths return an "empty" :class:`Capabilities` with
    ``bad_perms`` set when the failure was permission-related. They never
    raise — the classifier is called on UI hot paths and must be infallible.

    Note:
        The cache key assumes file content is a function of ``(path, mtime)``.
        Build systems that preserve mtime across atomic replacement (``rsync
        -t``, ``cp -p``, ``os.utime`` after a swap, etc.) will keep returning
        stale results until :func:`invalidate_cache` is called — which is
        exactly what the sidebar's "Refresh" button does.

    Args:
        path: A path inside the sandbox. Caller is expected to have run it
            through :meth:`SandboxRoot.resolve` first.

    Returns:
        Frozen :class:`Capabilities` summary.
    """
    abs_path = Path(path).absolute()
    try:
        mtime_ns = abs_path.stat().st_mtime_ns
    except FileNotFoundError:
        return _EMPTY
    except PermissionError:
        return _BAD_PERMS
    except OSError:
        # Broken symlinks etc. Return empty rather than raising.
        return _EMPTY
    return _classify_cached(str(abs_path), mtime_ns)


def invalidate_cache() -> None:
    """Flush the classifier's LRU cache.

    Called by the sidebar's "Refresh" button so newly-arrived files are
    surfaced. No-op if the cache is already empty.
    """
    _classify_cached.cache_clear()


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

_EMPTY = Capabilities(
    is_image_dir=False,
    has_pipeline_json=False,
    is_cli_output=False,
    has_dashboard=False,
    is_process_only_output=False,
    image_count=None,
    bad_perms=False,
)
_BAD_PERMS = Capabilities(
    is_image_dir=False,
    has_pipeline_json=False,
    is_cli_output=False,
    has_dashboard=False,
    is_process_only_output=False,
    image_count=None,
    bad_perms=True,
)


@lru_cache(maxsize=2048)
def _classify_cached(path_str: str, mtime_ns: int) -> Capabilities:
    """Cache key includes mtime so writes invalidate without manual flush.

    ``mtime_ns`` is unused in the body — it's a cache key only. The body
    is called fresh whenever the directory's mtime changes (e.g. a new
    file lands in it), which is the cache-invalidation contract we want.
    """
    del mtime_ns  # cache-key-only
    path = Path(path_str)
    if path.is_file():
        return _classify_file(path)
    if path.is_dir():
        return _classify_dir(path)
    # Symlink to a missing target, FIFO, socket, etc. — surface as empty.
    return _EMPTY


def _classify_file(path: Path) -> Capabilities:
    """File-only classifier. Currently only flags pipeline JSONs."""
    if path.suffix.lower() == ".json" and _peek_for_pipeline_marker(path):
        return Capabilities(
            is_image_dir=False,
            has_pipeline_json=True,
            is_cli_output=False,
            has_dashboard=False,
            is_process_only_output=False,
            image_count=None,
            bad_perms=False,
        )
    return _EMPTY


def _classify_dir(path: Path) -> Capabilities:
    """Directory classifier. One ``listdir``; stat-only per child."""
    try:
        children = list(path.iterdir())
    except PermissionError:
        return _BAD_PERMS
    except OSError:
        return _EMPTY

    image_count = 0
    is_cli_output_master = False
    is_cli_output_results = False
    has_dashboard = False
    has_deliverables_dir = False

    for child in children:
        name = child.name
        if name == _RESULTS_DIRNAME and child.is_dir():
            is_cli_output_results = True
            continue
        if name == DELIVERABLES_DIRNAME and child.is_dir():
            has_deliverables_dir = True
            continue
        if image_count < _IMAGE_COUNT_CAP:
            suffix = child.suffix.lower()
            if suffix in IMAGE_EXTS and child.is_file():
                image_count += 1

    # The master parquet and dashboard now live under ``<root>/deliverables/``.
    # Only stat them when that subdir exists, so a non-output directory pays
    # at most the cheap iterdir above (no extra stats per child).
    if has_deliverables_dir:
        deliverables = path / DELIVERABLES_DIRNAME
        is_cli_output_master = (deliverables / _MASTER_MEASUREMENTS_FILENAME).is_file()
        has_dashboard = (deliverables / _DASHBOARD_FILENAME).is_file()

    is_cli_output = is_cli_output_master and is_cli_output_results

    is_process_only_output = False
    if not is_cli_output and not is_cli_output_results:
        # A process-only run writes machine-state (its
        # .phenotypic/progress/manifest.json) but NO results/ and NO
        # deliverables/. Gating on ``not is_cli_output_results`` prevents a
        # forward run that has *started* (results/ + manifest already written)
        # but not yet finalized (no master_measurements.parquet) from being
        # transiently misclassified as process-only. Resolve via the helper so a
        # legacy-root manifest still surfaces.
        from phenotypic.tools_ import resolve_manifest_json_path

        is_process_only_output = resolve_manifest_json_path(path).is_file()

    is_image_dir = image_count > 0
    return Capabilities(
        is_image_dir=is_image_dir,
        has_pipeline_json=False,  # only flagged for JSON files, not dirs
        is_cli_output=is_cli_output,
        has_dashboard=has_dashboard,
        is_process_only_output=is_process_only_output,
        image_count=image_count if is_image_dir else None,
        bad_perms=False,
    )


def _peek_for_pipeline_marker(path: Path) -> bool:
    """Cheap heuristic for "is this a PhenoTypic pipeline JSON?".

    Reads the first 4 KB and checks for either the current
    ``"pipe_cfgs"`` key (what :meth:`ImagePipeline.to_json` writes) or the
    legacy ``"operations"`` key.
    """
    try:
        with path.open("rb") as f:
            head = f.read(_PIPELINE_JSON_PEEK_BYTES)
    except (PermissionError, OSError):
        return False
    return any(marker in head for marker in _PIPELINE_JSON_MARKERS)
