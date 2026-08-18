"""Shared fixtures for CLI unit tests.

Provides lightweight builders used by the process-only strategy / SLURM /
end-to-end tests: a one-level synthetic input tree, a serialized minimal
pipeline with one detector, and ``ExecutionConfig`` / ``OutputManager``
factories with sensible defaults overridable by keyword.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pytest
from PIL import Image as PILImage

from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_types import ExecutionConfig
from phenotypic.data import load_synth_yeast_plate
from phenotypic.prefab import RoundPeaksPipeline


@pytest.fixture
def synth_one_level_input(tmp_path: Path) -> Path:
    """One-level input tree: ``<tmp>/in/day1/plateA.tif`` (one synth plate).

    Returns the input root (``<tmp>/in``) so callers can pass it as
    ``--input`` / ``input_path`` and assert on the mirrored output tree.
    """
    root = tmp_path / "in"
    day = root / "day1"
    day.mkdir(parents=True)
    grid_image = load_synth_yeast_plate()
    pil_img = PILImage.fromarray(grid_image.rgb[:].astype("uint8"))
    pil_img.save(day / "plateA.tif")
    return root


@pytest.fixture
def simple_pipeline_json() -> Path:
    """Write a minimal ``RoundPeaksPipeline`` (one detector) JSON to a temp file."""
    pipeline = RoundPeaksPipeline(
        blur_sigma=3,
        detector_thresh_method="otsu",
        detector_subtract_background=True,
        detector_remove_noise=True,
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as handle:
        handle.write(pipeline.to_json())
        pipeline_path = Path(handle.name)
    try:
        yield pipeline_path
    finally:
        if pipeline_path.exists():
            pipeline_path.unlink()


@pytest.fixture
def make_exec_config() -> Callable[..., ExecutionConfig]:
    """Factory: build an ``ExecutionConfig`` with defaults overridable by kwargs."""

    def _build(
        *,
        pipeline_json: Path,
        input_path: Path,
        output_dir: Optional[Path] = None,
        image_type: str = "GridImage",
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
        bit_depth: Optional[int] = None,
        n_jobs: int = 1,
        slurm_args: Optional[Dict[str, Any]] = None,
        force_local: bool = True,
        wait: bool = False,
        ext: str = ".tiff",
        overlay_alpha: float = 0.3,
        include_dataset_column: bool = True,
        dry_run: bool = False,
        sample: Optional[int] = None,
        resume: bool = False,
        retry_failures: bool = False,
        skip_validation: bool = True,
        detect_mode: str = "gray",
        process_only_layer: Optional[str] = None,
        **overrides: Any,
    ) -> ExecutionConfig:
        return ExecutionConfig(
            pipeline_json=pipeline_json,
            input_path=input_path,
            output_dir=output_dir,
            image_type=image_type,  # type: ignore[arg-type]
            nrows=nrows,
            ncols=ncols,
            bit_depth=bit_depth,
            n_jobs=n_jobs,
            slurm_args=slurm_args if slurm_args is not None else {},
            force_local=force_local,
            wait=wait,
            ext=ext,
            overlay_alpha=overlay_alpha,
            include_dataset_column=include_dataset_column,
            dry_run=dry_run,
            sample=sample,
            resume=resume,
            retry_failures=retry_failures,
            skip_validation=skip_validation,
            detect_mode=detect_mode,
            process_only_layer=process_only_layer,  # type: ignore[arg-type]
            **overrides,
        )

    return _build


@pytest.fixture
def make_output_manager() -> Callable[..., OutputManager]:
    """Factory: build an ``OutputManager`` rooted at a given output dir."""

    def _build(output_dir: Path, **overrides: Any) -> OutputManager:
        return OutputManager.from_config(
            base_dir=output_dir,
            ext=overrides.pop("ext", ".tiff"),
            **overrides,
        )

    return _build


@pytest.fixture
def array_script_kwargs(tmp_path: Path) -> Dict[str, Any]:
    """Minimal valid call for ``build_array_script_spec`` / ``generate_array_job_script``.

    Supplies every argument except ``output_dir``, so a caller can render the
    same chunk twice against the same output directory:

        build_array_script_spec(output_dir=out, **array_script_kwargs)
        generate_array_job_script(output_dir=out, **array_script_kwargs)

    Mirrors ``generate_array_job_script``'s signature at
    ``_cli_slurm_array_scripts.py`` -- ``dataset``, ``array_indices``,
    ``config``, ``chunk_id``, ``checkpoint_interval``, ``is_last_chunk``.
    """
    from phenotypic._cli._cli_types import Dataset

    src = tmp_path / "array_src"
    src.mkdir()
    images = []
    for i in range(4):
        img_path = src / f"image_{i:03d}.tif"
        img_path.touch()
        images.append(img_path)

    pipeline_json = tmp_path / "array_pipeline.json"
    pipeline_json.write_text('{"operations": []}')

    dataset = Dataset(
        name="array_dataset",
        images=images,
        input_dir=src,
        output_dir=tmp_path / "array_out",
    )
    config = ExecutionConfig(
        pipeline_json=pipeline_json,
        input_path=src,
        output_dir=tmp_path / "array_out",
        image_type="GridImage",
        nrows=8,
        ncols=12,
        bit_depth=None,
        n_jobs=-1,
        slurm_args={"slurm_partition": "short", "mem_gb": 16, "time": 60},
        force_local=False,
        wait=False,
        ext=".tiff",
        overlay_alpha=0.3,
        include_dataset_column=False,
        dry_run=False,
        sample=None,
        resume=False,
        retry_failures=False,
        skip_validation=False,
    )
    return {
        "dataset": dataset,
        "array_indices": (0, 4),
        "config": config,
        "chunk_id": 0,
        "checkpoint_interval": 2,
        "is_last_chunk": False,
    }
