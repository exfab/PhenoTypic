"""Write the curated winner to ``deliverables/best_pipeline.json`` (Task B5).

When a user picks a tuned candidate in the Curate view and clicks "Set as
winner", :func:`write_winner` materializes the candidate pipeline
(``build_pipeline(base, winner.params)``) and writes it **atomically** (temp
file + ``os.replace``) to the run's ``deliverables/best_pipeline.json`` — the
same path the engine writes the auto-selected winner to, so a hand-curated
choice overrides it in place.

The write is atomic so a crash mid-write can never leave a half-written winner
the Launch view would then apply. On an HPCC read-only output directory the
``os.replace`` raises :class:`PermissionError`; the helper re-raises it so the
callback can catch it and surface a toast (OQ7) rather than failing silently.

Importing this module must never drag ``optuna`` into ``sys.modules``:
``build_pipeline`` lives in the optuna-free
:mod:`phenotypic.tune._evaluation._builder`, and ``ImagePipeline`` is optuna-free.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type-only
    from phenotypic import ImagePipeline
    from phenotypic.gui.tune._run_root import TuneRunRoot
    from phenotypic.tune._study_store import Trial


def write_winner(
    root: "TuneRunRoot", base: "ImagePipeline", winner: "Trial"
) -> Path:
    """Write the curated ``winner`` pipeline to ``root.best_pipeline_path``.

    Overlays ``winner.params`` onto ``base`` via
    :func:`~phenotypic.tune._evaluation._builder.build_pipeline` (the same combo
    grammar the engine uses), then serializes the result with
    ``pipeline.to_json()`` and writes it atomically to
    ``deliverables/best_pipeline.json``. Works for both single- and
    multi-objective runs — the curated choice is a single explicit override.

    The write is atomic: the JSON is first written to a temp file in the same
    directory, then ``os.replace``d onto the target so a reader never sees a
    partial file and a crash mid-write can't corrupt an existing winner.

    Args:
        root: The validated tune output handle; ``root.best_pipeline_path`` is
            the write target.
        base: The base :class:`~phenotypic.ImagePipeline` the trial's params
            overlay onto (not mutated — ``build_pipeline`` deep-copies it).
        winner: The curated trial; its ``params`` combo is applied to ``base``.

    Returns:
        The path written (``root.best_pipeline_path``).

    Raises:
        PermissionError: When the output directory is read-only (HPCC) and the
            atomic ``os.replace`` cannot complete. Re-raised so the caller can
            surface it in a toast.

    Examples:
        >>> from pathlib import Path
        >>> import tempfile
        >>> from phenotypic import ImagePipeline
        >>> from phenotypic.detect import OtsuDetector
        >>> from phenotypic.enhance import GaussianBlur
        >>> from phenotypic.gui.tune._run_root import TuneRunRoot
        >>> from phenotypic.sdk_ import best_pipeline_path
        >>> from phenotypic.tune._study_store import Trial
        >>> d = Path(tempfile.mkdtemp())
        >>> root = TuneRunRoot(
        ...     path=d, trials_path=None, storage_url=None, study_name="tune_cost_v1",
        ...     directions=None, images_dir=None,
        ...     best_pipeline_path=best_pipeline_path(d),
        ... )
        >>> base = ImagePipeline(ops=[GaussianBlur(sigma=1.0), OtsuDetector()])
        >>> winner = Trial(number=0, params={"0.sigma": 3.0}, score=0.05,
        ...                terms={}, n_images=2)
        >>> written = write_winner(root, base, winner)
        >>> restored = ImagePipeline.from_json(written.read_text())
        >>> list(restored.get_ops().values())[0].sigma  # the override landed
        3.0
    """
    from phenotypic._cli._cli_output_manager import _atomic_write
    from phenotypic.tune._evaluation._builder import build_pipeline

    # TODO(multi-obj): for a multi-objective (Pareto) run, "the winner" is
    # ambiguous — the curated single override here writes whichever trial the
    # user pinned, which is the right v1 behavior (the human disambiguates the
    # front by pinning). A future pass could persist the chosen Pareto point's
    # objectives alongside the pipeline for traceability.
    pipeline = build_pipeline(base, winner.params)
    # to_json() returns the JSON string when no filepath is given (it returns
    # None only in the write-to-file overload); assert for the type-checker.
    payload = pipeline.to_json()
    assert payload is not None

    target = Path(root.best_pipeline_path)

    def _write(tmp: str) -> None:
        Path(tmp).write_text(payload, encoding="utf-8")

    # Atomic write (shared CLI helper): temp file in the target's directory +
    # ``os.replace``, so a reader never sees a partial file and a crash mid-write
    # can't corrupt an existing winner. A read-only output dir (HPCC) raises
    # PermissionError, which the helper re-raises after cleaning up the temp file.
    _atomic_write(target, _write)
    return target


__all__ = ["write_winner"]
