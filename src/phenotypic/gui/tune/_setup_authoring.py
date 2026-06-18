"""Setup authoring helpers for GUI-created tuning specs."""
from __future__ import annotations

import re
from pathlib import Path

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.gui._config import tune_presets_dir
from phenotypic.gui.tune._space import space_to_spec
from phenotypic.sdk_ import CONFIG_SUFFIX_TUNING, matches_any_suffix
from phenotypic.tune import QCScorer, TuningSpec

_SAFE_STEM = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_stem(path: Path) -> str:
    """Return a filesystem-safe stem for a GUI-authored spec."""
    stem = _SAFE_STEM.sub("-", path.stem).strip(".-")
    return stem or "tuning-spec"


def _load_pipeline_or_spec(path: Path) -> ImagePipeline | TuningSpec:
    """Load a selected pipeline or existing tuning spec from disk."""
    text = path.read_text(encoding="utf-8")
    if matches_any_suffix(path, (CONFIG_SUFFIX_TUNING,)):
        return TuningSpec.model_validate_json(text)
    return ImagePipeline.from_json(text)


def authored_setup_spec_path(*, sandbox_root: Path, source_path: Path) -> Path:
    """Return the GUI preset path for a spec authored from ``source_path``."""
    return (
        tune_presets_dir(sandbox_root)
        / f"{_safe_stem(source_path)}.setup.json.pht-tune"
    )


def write_authored_setup_spec(
    *,
    sandbox_root: Path,
    pipeline_or_spec_path: Path,
    metadata_path: Path,
    metadata_groupby: list[str] | None = None,
) -> Path:
    """Write a GUI-authored, launchable tuning spec and return its path.

    Args:
        sandbox_root: GUI sandbox root used for the tune preset directory.
        pipeline_or_spec_path: Existing pipeline or tuning spec selected in Setup.
        metadata_path: Layout CSV/Parquet used by the default QC scorer.
        metadata_groupby: Metadata columns used to group expected counts. Defaults
            to ``["Metadata_ImageName"]``.

    Returns:
        The typed ``.json.pht-tune`` file written under the GUI tune presets dir.

    Raises:
        FileNotFoundError: If either selected path does not exist.
        ValueError: If the selected pipeline cannot infer any tunable knobs.
    """
    if not pipeline_or_spec_path.is_file():
        raise FileNotFoundError(pipeline_or_spec_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    groupby = metadata_groupby or ["Metadata_ImageName"]
    pipeline_or_spec = _load_pipeline_or_spec(pipeline_or_spec_path)
    spec = (
        pipeline_or_spec
        if isinstance(pipeline_or_spec, TuningSpec)
        else space_to_spec(pipeline_or_spec, edits={})
    )
    if not spec.search_space.knobs:
        raise ValueError("No active knobs to tune.")
    scorer = QCScorer(
        check=ExpectedVsDetectedCount(metadata=str(metadata_path), groupby=groupby)
    )
    spec = spec.model_copy(update={"scorer": scorer})

    target = authored_setup_spec_path(
        sandbox_root=sandbox_root,
        source_path=pipeline_or_spec_path,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    spec.to_json(target)
    return target


__all__ = [
    "authored_setup_spec_path",
    "write_authored_setup_spec",
]
