"""Durable record of plot failures that publication swallowed.

Plot output is best-effort by design: one bad figure must not kill a run that
produced good measurements. This module is what stops "best-effort" from
meaning "silent" -- a green run with missing plots carries the reason on disk
rather than only in a log nobody reads.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from phenotypic.sdk_._file_locking import exclusive_path_lock

logger = logging.getLogger(__name__)


def _format_error(error: BaseException) -> str:
    """Render *error* for the record, degrading rather than raising.

    Interpolating an exception runs **user code**: ``error`` is whatever a
    plot provider raised, and an exception whose ``__str__`` formats a frame
    or reads an attribute that has since gone will fail here. Letting that
    escape would break the never-raises contract; swallowing it one level up
    would drop the whole entry, restoring exactly the silence this module
    exists to remove. So the field degrades and the record survives.
    """
    try:
        name = type(error).__name__
    except Exception:  # noqa: BLE001 - the record matters more than the name
        name = "UnknownError"
    try:
        return f"{name}: {error}"
    except Exception as exc:  # noqa: BLE001 - see above
        return f"{name}: <unprintable: {type(exc).__name__}>"


def record_plot_failure(
    plots_base: Path,
    *,
    binding_id: str,
    plot_class: str,
    lifecycle: str,
    error: BaseException,
    dataset: str | None = None,
    image_stem: str | None = None,
) -> None:
    """Append one failure to ``<plots_base>/.failures.jsonl``.

    **Never raises an** :class:`Exception`. Called from ``except`` blocks whose purpose
    is to keep a plot failure from ending a run; letting the recorder throw
    would turn the soft failure it is describing into a hard one. **Every**
    step is inside the one handler, not only the filesystem ones -- creating
    the directory, taking the lock, opening the file and writing the line are
    each an independent way to fail on an unwritable target, and building the
    entry runs the caller's exception through ``__str__``. That last one is
    the reason the boundary is drawn around the whole body rather than around
    the I/O: "formatting cannot fail" is an assumption, not a guarantee.

    A :class:`BaseException` still propagates. An ``error`` whose ``__str__``
    raises :class:`KeyboardInterrupt` escapes, and that is deliberate --
    widening the catch would swallow Ctrl-C and :class:`SystemExit`, which is
    worse than the hole it closes. Note the signature already accepts
    ``BaseException``, so this gap is known rather than accidental.

    Args:
        plots_base: Resolved ``deliverables/plots`` directory.
        binding_id: Stable plot binding id.
        plot_class: Producer class name.
        lifecycle: Where the failure happened. ``"image"``,
            ``"measurements"``, ``"analysis"`` and ``"qc"`` are the
            coordinator's four emit points -- *when* a plot was being
            refreshed. ``"page"`` is the writer's own level and is orthogonal
            to them: one page of a multi-page output failed to render, inside
            whichever of the four was running. The writer does not know which,
            which is why it is a fifth value rather than a subdivision of the
            other four.
        error: The exception that was swallowed.
        dataset: Dataset name, for the image lifecycle only.
        image_stem: Image stem, for the image lifecycle only.
    """
    try:
        # Inside the handler, not above it. A lazy import can fail -- a circular
        # import, a partially-initialised package during interpreter shutdown --
        # and "every step is inside the one handler" would be false if this sat
        # outside it.
        from phenotypic.sdk_ import plot_failures_jsonl_path

        entry: dict[str, str] = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "binding_id": binding_id,
            "plot_class": plot_class,
            "lifecycle": lifecycle,
            "error": _format_error(error),
        }
        if dataset is not None:
            entry["dataset"] = dataset
        if image_stem is not None:
            entry["image_stem"] = image_stem

        plots_base.mkdir(parents=True, exist_ok=True)
        # default=str: a field that is not JSON-native must degrade, not
        # destroy the entry. These fields are TYPED str but arrive from
        # callers -- a Path, a numpy scalar or bytes makes dumps raise
        # INSIDE this handler, and the whole record vanishes silently.
        # Measured: np.str_ survives (it subclasses str); np.int64, Path
        # and bytes do not. Serialising the entry is one more step that can
        # fail, like the caller's __str__ one layer in.
        line = json.dumps(entry, sort_keys=True, default=str) + "\n"
        record = plot_failures_jsonl_path(plots_base)
        with exclusive_path_lock(plots_base / ".failures.lock"):
            with record.open("a", encoding="utf-8") as handle:
                handle.write(line)
    except Exception:  # noqa: BLE001 - recording must never escalate
        logger.debug(
            "Could not record plot failure for %s", binding_id, exc_info=True
        )


__all__ = ["record_plot_failure"]
