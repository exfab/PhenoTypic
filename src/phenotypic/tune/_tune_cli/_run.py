"""Run-a-tuning-spec orchestration + the ``deliverables/`` writes."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from phenotypic import GridImage
from phenotypic.tools_ import _io_constants as io

from .._engine import TuningEngine
from .._screening import compute_param_importance
from .._spec import TuningSpec
from .._study_store import StudyStore, Trial

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".h5"}


def _load_images(input_dir: Path) -> list:
    """Load every image file under ``input_dir`` as a ``GridImage``.

    Mirrors the forward CLI's directory scan; tuning targets arrayed plates, so
    images load as ``GridImage`` via ``imread``. Unreadable / non-grid files are
    skipped (warned) rather than aborting the whole run.

    Args:
        input_dir: The directory to scan (non-recursive).

    Returns:
        The loaded ``GridImage`` instances, in sorted filename order.
    """
    paths = sorted(
        p for p in Path(input_dir).iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    images: list = []
    failures: list[tuple[str, str]] = []
    for path in paths:
        try:
            images.append(GridImage.imread(path))
        except Exception as exc:  # skip unreadable / non-grid files, don't abort
            failures.append((path.name, str(exc)))
    if failures:
        logging.getLogger(__name__).warning(
            "skipped %d unreadable image(s): %s",
            len(failures), ", ".join(name for name, _ in failures),
        )
    return images


def run_tuning(
    spec: TuningSpec, images: list, output_dir: Path
) -> Optional[Trial]:
    """Run ``spec`` over ``images`` and write the ``deliverables/`` artifacts.

    Writes ``trials.parquet`` (root), and under ``deliverables/``:
    ``tuning_spec.json`` (resolved spec), ``best_pipeline.json`` (the winner),
    ``param_importance.json``. Resumes if ``trials.parquet`` already exists.

    Args:
        spec: The tuning recipe (embeds the base pipeline + scorer + strategy).
        images: The calibration images.
        output_dir: The run directory.

    Returns:
        The best :class:`Trial`, or ``None`` if none succeeded.
    """
    output_dir = Path(output_dir)
    io.deliverables_dir(output_dir).mkdir(parents=True, exist_ok=True)

    trials_path = io.trials_parquet_path(output_dir)
    store = (
        StudyStore.from_parquet(trials_path)
        if trials_path.exists()
        else StudyStore()
    )

    engine = TuningEngine(spec, store=store)
    best = engine.optimize(images)

    store.to_parquet(trials_path)
    io.tuning_spec_path(output_dir).write_text(spec.model_dump_json(indent=2))
    io.param_importance_path(output_dir).write_text(
        json.dumps(compute_param_importance(store), indent=2)
    )
    winner = engine.best_pipeline()
    if winner is not None:
        io.best_pipeline_path(output_dir).write_text(winner.to_json() or "")
    return best
