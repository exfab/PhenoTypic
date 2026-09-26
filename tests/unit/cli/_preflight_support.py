"""Shared builders for run-preflight tests (spec 2026-09-24-cli-preflight).

Checks read a real ``ExecutionConfig`` and scanned ``Dataset`` values, so tests
build those rather than passing ``None``: a check that dereferences a field the
context lacks would otherwise surface as ``PF-CHECK-CRASHED`` and hide the
behaviour under test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from phenotypic import ImagePipeline
from phenotypic._cli._cli_preflight import PreflightContext
from phenotypic._cli._cli_types import Dataset, ExecutionConfig


def make_config(**overrides: Any) -> ExecutionConfig:
    """An ``ExecutionConfig`` for a local, full-mode GridImage run."""
    values: dict[str, Any] = dict(
        pipeline_json=Path("pipeline.json"),
        input_path=Path("images"),
        output_dir=Path("out"),
        image_type="GridImage",
        nrows=None,
        ncols=None,
        bit_depth=None,
        n_jobs=1,
        slurm_args={},
        force_local=True,
        wait=False,
        ext=".tiff",
        overlay_alpha=0.3,
        include_dataset_column=True,
        dry_run=False,
        sample=None,
        resume=False,
        retry_failures=False,
        skip_validation=False,
    )
    values.update(overrides)
    return ExecutionConfig(**values)


def make_datasets(*images: Path | str, name: str = "plate1") -> tuple[Dataset, ...]:
    """One dataset holding *images* (paths need not exist unless a check reads them)."""
    paths = [Path(image) for image in images]
    return (
        Dataset(
            name=name,
            images=paths,
            input_dir=paths[0].parent if paths else Path("."),
            output_dir=Path("out") / name,
        ),
    )


def make_context(
    pipeline: ImagePipeline,
    mode: str = "full",
    datasets: Sequence[Dataset] = (),
    **config: Any,
) -> PreflightContext:
    """A ``PreflightContext`` whose config matches *mode*."""
    if mode == "measure":
        config.setdefault("measure_only", True)
    if mode == "process":
        config.setdefault("process_only_layer", "gray")
    return PreflightContext(
        config=make_config(**config),
        pipeline=pipeline,
        datasets=tuple(datasets),
        mode=mode,  # type: ignore[arg-type]
    )
