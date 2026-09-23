"""Safe, best-effort publication for one or many plot pages."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from phenotypic.sdk_._file_locking import exclusive_path_lock
from phenotypic.sdk_ import CommitGuard, publication_commit

# ``figure_backend_of`` is imported at MODULE level on purpose: tests patch this
# module's own binding (``_writer.figure_backend_of``), and a function-scope
# import would leave no attribute to patch. ``chrome_available`` is deliberately
# the other way round -- imported inside the functions that use it, because its
# tests patch ``_backends.chrome_available``, the defining module's attribute.
# Each import style is chosen by how its consumer is patched; do not harmonise.
from phenotypic.abc_.plotting import PlotOutput, figure_backend_of

from ._adapter import FigureAdapter
# `_format_error` is private to `_failures`, and reused here on purpose rather
# than reimplemented: the manifest formats caller-supplied exceptions, and
# interpolating one runs user code. `_format_error` degrades to a placeholder
# instead of raising, so a figure whose exception has a broken `__str__` costs
# one unreadable field rather than the whole publication. It is also the single
# definition of how a plot failure is spelled, so `.failures.jsonl` and the
# manifest cannot drift apart.
from ._failures import _format_error, record_plot_failure
from ._output import normalize_plot_output

logger = logging.getLogger(__name__)

_UNSAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


class PlotPublicationBlocked(RuntimeError):
    """Raised when a late publication predicate rejects a plot write."""


@contextmanager
def _enter_commit(commit_guard: CommitGuard | None) -> Iterator[None]:
    """Enter *commit_guard*, reporting a refusal as ``PlotPublicationBlocked``.

    Only ENTERING the guard is translated. A guard that will not admit the
    commit -- the staged worker's lifecycle fence
    (``SlurmGenerationInactiveError``), or its lock timing out
    (``ArtifactLockTimeout``) -- means this process cannot confirm it still
    owns the output, so the write is void rather than a plot failure. The
    original is kept as ``__cause__``, where ``slurm_generation_inactive_cause``
    finds it, and this layer never imports the ``_cli`` exception.

    The body is NOT translated: an ``os.replace`` that fails inside the guard is
    a genuine write error and stays one.

    Note what this costs on the CLI full path: a lock timeout fails the image,
    like any other commit that cannot confirm ownership of the run, rather than
    losing one plot while the image continues.
    """
    with ExitStack() as stack:
        try:
            stack.enter_context(publication_commit(commit_guard))
        except PlotPublicationBlocked:
            raise
        except Exception as exc:  # noqa: BLE001 - any refusal voids the write
            raise PlotPublicationBlocked(
                "Plot publication blocked because the commit guard refused: "
                f"{_format_error(exc)}"
            ) from exc
        yield


@contextmanager
def _guarded_commit(
    publication_guard: Callable[[], bool] | None,
    commit_guard: CommitGuard | None,
) -> Iterator[None]:
    """Enter *commit_guard*, then recheck *publication_guard*, for one mutation.

    Every canonical write, replace or removal here goes through this, so the
    predicate is always rechecked inside the commit guard and immediately
    before the mutation in the body -- never outside it.
    """
    with _enter_commit(commit_guard):
        _require_plot_publication(publication_guard)
        yield


def safe_path_component(value: str) -> str:
    """Return one filesystem-safe path component.

    Args:
        value: Human-readable identifier or label.

    Returns:
        Sanitized non-empty component.

    Raises:
        ValueError: If the value is traversal-like or sanitizes to nothing.
    """
    if not isinstance(value, str):
        raise TypeError(f"path component must be str, got {type(value).__name__}")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"unsafe path component {value!r}")
    component = _UNSAFE_COMPONENT.sub("-", value.strip()).strip("-.")
    if not component or component in {".", ".."}:
        raise ValueError(f"path component {value!r} has no safe filename characters")
    return component


def _atomic_write(
    destination: Path,
    write: Callable[[Path], None],
    *,
    publication_guard: Callable[[], bool] | None,
    commit_guard: CommitGuard | None,
) -> None:
    """Write via a temporary sibling and replace, or leave nothing behind."""
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        write(temporary)
        with _guarded_commit(publication_guard, commit_guard):
            os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _render_page(
    figure: Any,
    directory: Path,
    stem: str,
    *,
    plots_base: Path,
    plot_id: str,
    publication_guard: Callable[[], bool] | None = None,
    commit_guard: CommitGuard | None = None,
) -> tuple[dict[str, str], list[BaseException], str | None]:
    """Render one figure to every format its backend supports.

    This is the single definition of "what files does a page produce". It is
    called from :func:`_publish_plot_output_locked` for multi-page and aggregate
    output, and from ``PlotCoordinator._publish_image_value`` for the flat
    single-page image path -- which does not go through the writer, takes no
    directory lock and writes no manifest, and would otherwise never gain HTML.

    HTML is attempted first: it needs no Chrome, so a page that can be published
    at all is on disk before anything that might fail is tried.

    Args:
        figure: The figure to render.
        directory: Directory the page files are written into.
        stem: Filename stem, without extension.
        plots_base: Resolved ``deliverables/plots`` directory, for the bundle.
        plot_id: Binding id, for diagnostics.
        publication_guard: Optional GUI compare-and-set predicate.
        commit_guard: Optional commit guard.

    Returns:
        ``(files, errors, backend)`` -- a mapping of format to filename for
        everything that published, the **exceptions** raised by everything that
        did not, and the figure's backend (``None`` if unsupported, in which
        case *files* is empty).

        The errors are exceptions rather than pre-formatted strings so that the
        one caller that records them and the one that puts them in the manifest
        format them the same way, once. Returning strings would double-format:
        the recorder formats what it is given, so it would write
        ``"RuntimeError: RuntimeError: raster exploded"`` into the durable
        record.

    Raises:
        PlotPublicationBlocked: If a guard rejects the write. Never swallowed --
            it means the output snapshot changed and this whole publication is
            void.
    """
    from ._backends import chrome_available, ensure_plotlyjs_bundle, plotlyjs_src_for

    backend = figure_backend_of(figure)
    if backend is None:
        return {}, [
            TypeError(
                "unsupported figure type "
                f"{type(figure).__module__}.{type(figure).__qualname__}"
            )
        ], None

    files: dict[str, str] = {}
    errors: list[BaseException] = []

    if backend == "plotly":
        try:
            bundle = ensure_plotlyjs_bundle(plots_base)
            src = plotlyjs_src_for(directory, bundle)
            _atomic_write(
                directory / f"{stem}.html",
                lambda dest: FigureAdapter.save_html(figure, dest, plotlyjs_src=src),
                publication_guard=publication_guard,
                commit_guard=commit_guard,
            )
            files["html"] = f"{stem}.html"
        except PlotPublicationBlocked:
            raise
        except Exception as exc:  # noqa: BLE001 - plots are best-effort
            errors.append(exc)
            logger.warning(
                "Plot %s page %s failed during HTML save: %s", plot_id, stem, exc
            )

    if backend == "mpl" or chrome_available():
        try:
            _atomic_write(
                directory / f"{stem}.png",
                lambda dest: FigureAdapter.save_png(figure, dest),
                publication_guard=publication_guard,
                commit_guard=commit_guard,
            )
            files["png"] = f"{stem}.png"
        except PlotPublicationBlocked:
            # S11: a blocked PNG must not leave a published HTML sibling behind
            # asserting a page that this publication no longer owns.
            if "html" in files:
                (directory / files["html"]).unlink(missing_ok=True)
            raise
        except Exception as exc:  # noqa: BLE001 - plots are best-effort
            errors.append(exc)
            logger.warning(
                "Plot %s page %s failed during PNG save: %s", plot_id, stem, exc
            )

    return files, errors, backend


def _remove_stale_sibling(
    directory: Path,
    stem: str,
    backend: str | None,
    files: dict[str, str],
    *,
    publication_guard: Callable[[], bool] | None,
    commit_guard: CommitGuard | None,
) -> None:
    """Remove the rendering of *stem* that this generation did not write.

    A rerun on a node without Chrome writes HTML only; a plot that switched
    to matplotlib writes PNG only. Either way the other rendering from the
    earlier run would survive beside the new one and read as this run's --
    on the flat path because nothing records which generation a file belongs
    to, and in a manifest directory because the manifest stops naming it
    while it stays on disk.

    Only *stem*'s own sibling is touched. Files a manifest no longer names
    for other reasons -- a page whose key vanished between runs -- are not
    swept: that needs ownership of the whole directory, which this does not
    establish.

    Committed through :func:`_guarded_commit`, so a fenced or refused removal
    raises :class:`PlotPublicationBlocked` like every other commit here.
    """
    stale: list[str] = []
    if backend == "plotly" and "png" not in files:
        stale.append(f"{stem}.png")
    if backend == "mpl":
        stale.append(f"{stem}.html")
    for name in stale:
        path = directory / name
        if not path.exists():
            continue
        with _guarded_commit(publication_guard, commit_guard):
            path.unlink(missing_ok=True)


def unique_page_stems(names: Sequence[tuple[str, str]]) -> list[str]:
    """Return one filesystem stem per page, unique under case folding.

    Args:
        names: ``(page_key, preferred_name)`` per page, in page order. The
            manifest writer prefers the label; the store uses the key.

    Returns:
        Stems in the same order. Sanitization is many-to-one, so a collision
        gets a digest suffix derived from the page key -- stable across reruns.
    """
    used: dict[str, str] = {}
    stems: list[str] = []
    for key, preferred in names:
        try:
            stem = safe_path_component(preferred)
        except Exception:
            stem = "page"
        base_stem = stem
        folded = stem.casefold()
        attempt = 0
        while folded in used and used[folded] != key:
            digest_input = key if attempt == 0 else f"{key}:{attempt}"
            digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:8]
            stem = f"{base_stem}-{digest}"
            folded = stem.casefold()
            attempt += 1
        used[folded] = key
        stems.append(stem)
    return stems


def publish_plot_output(
    value: Any | PlotOutput,
    directory: Path,
    *,
    plot_id: str,
    plot_class: str | None = None,
    plots_base: Path | None = None,
    publication_guard: Callable[[], bool] | None = None,
    commit_guard: CommitGuard | None = None,
) -> dict[str, Any]:
    """Publish successful pages and an authoritative per-plot manifest.

    A page failure is logged and omitted while sibling pages continue. The
    manifest is replaced last and therefore lists only durable page files.

    Args:
        value: Raw supported figure or normalized multi-page output.
        directory: Destination directory for page PNGs and the manifest.
        plot_id: Stable binding ID used in diagnostics.
        plot_class: Producer class name. Defaults to ``plot_id`` for direct
            writer calls.
        plots_base: Resolved ``deliverables/plots`` directory, under which the
            one shared ``plotly.min.js`` and ``.failures.jsonl`` live. Omitting
            it falls back to *directory*, which writes a 4.8 MB bundle per page
            directory -- correct output, but the duplication this design exists
            to avoid. Every production caller passes it.
        publication_guard: Optional GUI compare-and-set predicate rechecked
            immediately before directory creation and every canonical page
            or manifest replacement. CLI callers omit it.

    Returns:
        JSON-native manifest payload.
    """
    _require_plot_publication(publication_guard)
    output = normalize_plot_output(value)
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_path_lock(directory / ".publication.lock"):
        _require_plot_publication(publication_guard)
        return _publish_plot_output_locked(
            output,
            directory,
            plot_id=plot_id,
            plot_class=plot_class,
            plots_base=plots_base,
            publication_guard=publication_guard,
            commit_guard=commit_guard,
        )


def _publish_plot_output_locked(
    output: PlotOutput,
    directory: Path,
    *,
    plot_id: str,
    plot_class: str | None,
    plots_base: Path | None = None,
    publication_guard: Callable[[], bool] | None,
    commit_guard: CommitGuard | None,
) -> dict[str, Any]:
    """Publish one plot generation while its directory lock is held."""
    # Function-scope import on purpose; see the note at this module's imports.
    from ._backends import chrome_available

    pages: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    backends: set[str | None] = set()
    # NOT `plots_base or directory`. Defaulting to the page directory writes a
    # 4.8 MB bundle into EVERY directory -- one per image for a multi-page image
    # plot -- which is the gigabyte trap this design exists to avoid, and it
    # does so while emitting a correct-looking relative src.
    base = plots_base if plots_base is not None else directory

    stems = unique_page_stems(
        [(page.key, page.label or page.key) for page in output.pages]
    )
    for page, stem in zip(output.pages, stems):
        try:
            files, errors, backend = _render_page(
                page.figure, directory, stem,
                plots_base=base,
                plot_id=plot_id,
                publication_guard=publication_guard,
                commit_guard=commit_guard,
            )
        except PlotPublicationBlocked:
            FigureAdapter.close(page.figure)
            raise
        FigureAdapter.close(page.figure)
        backends.add(backend)
        if files:
            # Only for pages the manifest will list: a failed page's files
            # are not described by this generation either way.
            _remove_stale_sibling(
                directory, stem, backend, files,
                publication_guard=publication_guard,
                commit_guard=commit_guard,
            )

        # S3: every swallowed error gets a durable record, not just a log line.
        # The exception is passed through unformatted; `record_plot_failure`
        # spells it, so the record and the manifest below agree exactly.
        for exc in errors:
            record_plot_failure(
                base,
                binding_id=plot_id,
                plot_class=plot_class or plot_id,
                lifecycle="page",
                error=exc,
            )

        if not files:
            failed.append({
                "key": page.key,
                "label": page.label,
                "error": (
                    _format_error(errors[0])
                    if errors
                    else "no renderer produced a file"
                ),
            })
            continue

        entry: dict[str, Any] = {
            "key": page.key,
            "label": page.label,
            "files": files,
            "backend": "matplotlib" if backend == "mpl" else "plotly",
            "metadata": dict(page.metadata),
        }
        if errors:
            # S2: one renderer failed while the other succeeded. The page is
            # published AND the failure is on the record.
            entry["partial"] = [_format_error(exc) for exc in errors]
        pages.append(entry)

    # `renderers` answers a CAPABILITY question -- what this machine could
    # render for this directory -- not an OUTCOME question about what landed on
    # disk. `files` and `failed` carry outcomes. Describing a capability in
    # outcome terms ("the matplotlib pages have a PNG and the Plotly pages do
    # not") is how a reader ends up unable to tell which question a value
    # answers, and "fixes" one branch to match another.
    renderers: dict[str, str] = {}
    has_plotly = "plotly" in backends
    has_mpl = "mpl" in backends
    if has_plotly:
        renderers["html"] = "available"
        # Probed only here, under `has_plotly`: an all-matplotlib publication
        # must not launch a browser for a value it cannot use. _render_page's
        # own `backend == "mpl" or chrome_available()` short-circuits for the
        # same reason.
        if chrome_available():
            renderers["png"] = "available"
        elif has_mpl:
            # Mixed directory, no Chrome: PNG capability exists, but only for
            # the matplotlib backend.
            renderers["png"] = "available: matplotlib only; chrome not found"
        else:
            renderers["png"] = "unavailable: chrome not found"
    elif has_mpl:
        renderers["png"] = "available"

    manifest = {
        "schema_version": 2,
        "plot_id": plot_id,
        "class": plot_class or plot_id,
        "renderers": renderers,
        "pages": pages,
        "failed": failed,
    }
    manifest_path = directory / "manifest.json"
    temporary_manifest = directory / f".manifest.{uuid.uuid4().hex}.tmp"
    try:
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        with _guarded_commit(publication_guard, commit_guard):
            os.replace(temporary_manifest, manifest_path)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return manifest


def _require_plot_publication(
    publication_guard: Callable[[], bool] | None,
) -> None:
    """Fail closed immediately before a canonical plot mutation."""
    if publication_guard is not None and not publication_guard():
        raise PlotPublicationBlocked(
            "Plot publication blocked because its output snapshot changed."
        )


__all__ = [
    "PlotPublicationBlocked",
    "publish_plot_output",
    "safe_path_component",
]
