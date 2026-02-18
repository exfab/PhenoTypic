"""SLURM array job script generation for the sweep CLI.

Generates bash scripts that process one (image, pipeline) pair per array task,
using 2D indexing over the image x pipeline product space.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Dict, List, Optional

from phenotypic._cli._cli_slurm_config import calculate_optimal_array_chunks
from phenotypic._cli._cli_slurm_scripts import generate_slurm_directives
from phenotypic._cli._cli_utils import get_python_command


def _build_worker_command(
    manifest_path: Path,
    output_dir: Path,
    image_type: str,
    read_kwargs: Dict[str, Any],
    save_layers: Dict[str, bool],
    overlay_mode: str,
    overlay_alpha: float,
) -> str:
    """Build the worker CLI command string (without image/pipeline args).

    Args:
        manifest_path: Path to the sweep manifest JSON.
        output_dir: Base output directory.
        image_type: ``"Image"`` or ``"GridImage"``.
        read_kwargs: Image read kwargs (nrows, ncols, etc.).
        save_layers: Which layers to save.
        overlay_mode: Overlay saving mode.
        overlay_alpha: Overlay transparency.

    Returns:
        Command string with ``${CURRENT_IMAGE}`` and ``${CURRENT_PIPELINE}``
        placeholders.
    """
    event_log = output_dir / "processing_events.log"
    python_cmd, _ = get_python_command()

    cmd_parts = [
        *python_cmd,
        "-m",
        "phenotypic.sweep._sweep_process_image",
        "--manifest",
        shlex.quote(str(manifest_path.absolute())),
        "--image",
        '"${CURRENT_IMAGE}"',
        "--output-dir",
        shlex.quote(str(output_dir.absolute())),
        "--image-type",
        image_type,
    ]

    # Grid params
    nrows = read_kwargs.get("nrows")
    ncols = read_kwargs.get("ncols")
    if image_type == "GridImage":
        cmd_parts.extend(["--nrows", str(nrows or 8)])
        cmd_parts.extend(["--ncols", str(ncols or 12)])

    bit_depth = read_kwargs.get("bit_depth")
    if bit_depth is not None:
        cmd_parts.extend(["--bit-depth", str(bit_depth)])

    detect_mode = read_kwargs.get("detect_mode", "gray")
    if detect_mode != "gray":
        cmd_parts.extend(["--detect-mode", detect_mode])

    # Save layer flags
    if save_layers.get("rgb"):
        cmd_parts.append("--save-rgb")
    if save_layers.get("gray"):
        cmd_parts.append("--save-gray")
    if save_layers.get("detect_mat"):
        cmd_parts.append("--save-detect-mat")
    if save_layers.get("objmask"):
        cmd_parts.append("--save-objmask")
    if save_layers.get("objmap"):
        cmd_parts.append("--save-objmap")
    if save_layers.get("objmap_overlay"):
        cmd_parts.append("--save-objmap-overlay")
    if save_layers.get("detect_mat_overlay"):
        cmd_parts.append("--save-detect-mat-overlay")
    if save_layers.get("objmask_overlay"):
        cmd_parts.append("--save-objmask-overlay")

    cmd_parts.extend(["--overlay-mode", overlay_mode])
    cmd_parts.extend(["--overlay-alpha", str(overlay_alpha)])
    cmd_parts.extend(["--event-log", shlex.quote(str(event_log.absolute()))])
    cmd_parts.extend(["--pipeline-name", '"${CURRENT_PIPELINE}"'])

    return " \\\n    ".join(cmd_parts)


def generate_sweep_array_script(
    image_paths: List[Path],
    pipeline_names: List[str],
    manifest_path: Path,
    output_dir: Path,
    image_type: str,
    read_kwargs: Dict[str, Any],
    slurm_args: Dict[str, Any],
    save_layers: Optional[Dict[str, bool]] = None,
    overlay_mode: str = "image",
    overlay_alpha: float = 0.3,
    global_offset: int = 0,
    num_local_tasks: Optional[int] = None,
    script_name: str = "sweep_array_job.sh",
) -> Path:
    """Generate a SLURM array job script for sweep processing.

    Each array task processes **one (image, pipeline) pair** via 2D indexing
    over the image x pipeline product space.

    Args:
        image_paths: All images to process.
        pipeline_names: All pipeline names from the manifest.
        manifest_path: Path to the sweep manifest JSON.
        output_dir: Base output directory.
        image_type: ``"Image"`` or ``"GridImage"``.
        read_kwargs: Image read kwargs (nrows, ncols, etc.).
        slurm_args: SLURM parameters dict.
        save_layers: Which layers to save.
        overlay_mode: Overlay saving mode.
        overlay_alpha: Overlay transparency.
        global_offset: Offset for 2D indexing (used by chunked scripts).
        num_local_tasks: Number of tasks in this chunk (defaults to total).
        script_name: Filename for the generated script.

    Returns:
        Path to the generated script.
    """
    save_layers = save_layers or {}

    if not image_paths or not pipeline_names:
        raise ValueError(
            "Cannot generate SLURM script with empty image_paths or pipeline_names"
        )

    num_images = len(image_paths)
    num_pipelines = len(pipeline_names)
    total_tasks = num_images * num_pipelines

    # Create script directory
    script_dir = output_dir / "slurm_scripts"
    script_dir.mkdir(parents=True, exist_ok=True)

    # Log paths
    log_dir = output_dir / "logs" / "slurm"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "sweep_%A_%a.log"

    # Generate SBATCH directives
    directives = generate_slurm_directives(
        job_name="pheno-sweep",
        slurm_args=slurm_args,
        output_log=log_path,
        error_log=log_path,
    )

    # Array directive (local indices for this chunk)
    local_tasks = num_local_tasks if num_local_tasks is not None else total_tasks
    array_directive = f"#SBATCH --array=0-{local_tasks - 1}"

    # Build image list for bash array
    image_list_lines = []
    for img in image_paths:
        image_list_lines.append(f"    {shlex.quote(str(img.absolute()))}")
    image_list_content = "\n".join(image_list_lines)

    # Build pipeline names for bash array
    pipeline_list_lines = []
    for pname in pipeline_names:
        pipeline_list_lines.append(f"    {shlex.quote(pname)}")
    pipeline_list_content = "\n".join(pipeline_list_lines)

    # Build worker command
    cmd = _build_worker_command(
        manifest_path=manifest_path,
        output_dir=output_dir,
        image_type=image_type,
        read_kwargs=read_kwargs,
        save_layers=save_layers,
        overlay_mode=overlay_mode,
        overlay_alpha=overlay_alpha,
    )

    # Offset section (only included for chunked scripts)
    offset_section = ""
    if global_offset > 0:
        offset_section = f"""
GLOBAL_OFFSET={global_offset}
GLOBAL_TASK_ID=$((SLURM_ARRAY_TASK_ID + GLOBAL_OFFSET))
"""
    else:
        offset_section = """
GLOBAL_TASK_ID=$SLURM_ARRAY_TASK_ID
"""

    script_content = f"""#!/bin/bash
{directives}
{array_directive}

# Auto-generated by PhenoTypic Sweep CLI
# Manifest: {manifest_path}
# Images: {num_images}, Pipelines: {num_pipelines}, Total tasks: {total_tasks}

set -u

IMAGE_LIST=(
{image_list_content}
)

PIPELINE_NAMES=(
{pipeline_list_content}
)

N_PIPELINES={num_pipelines}

if [ "${{SLURM_ARRAY_TASK_ID:-}}" = "" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set"
    exit 1
fi
{offset_section}
# 2D indexing: image = global_id / N_PIPELINES, pipeline = global_id % N_PIPELINES
IMAGE_IDX=$((GLOBAL_TASK_ID / N_PIPELINES))
PIPE_IDX=$((GLOBAL_TASK_ID % N_PIPELINES))

if [ "$IMAGE_IDX" -ge "${{#IMAGE_LIST[@]}}" ]; then
    echo "ERROR: Image index $IMAGE_IDX exceeds image list size ${{#IMAGE_LIST[@]}}"
    exit 1
fi

if [ "$PIPE_IDX" -ge "${{#PIPELINE_NAMES[@]}}" ]; then
    echo "ERROR: Pipeline index $PIPE_IDX exceeds pipeline list size ${{#PIPELINE_NAMES[@]}}"
    exit 1
fi

CURRENT_IMAGE="${{IMAGE_LIST[$IMAGE_IDX]}}"
CURRENT_PIPELINE="${{PIPELINE_NAMES[$PIPE_IDX]}}"

echo "======================================"
echo "Job ID: ${{SLURM_JOB_ID:-unknown}}"
echo "Array Task ID: ${{SLURM_ARRAY_TASK_ID:-unknown}}"
echo "Global Task: $((GLOBAL_TASK_ID + 1))/{total_tasks}"
echo "Image: $CURRENT_IMAGE"
echo "Pipeline: $CURRENT_PIPELINE"
echo "Node: ${{SLURMD_NODENAME:-$(hostname)}}"
echo "Start Time: $(date)"
echo "======================================"

{cmd}

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
echo "======================================"

exit $EXIT_CODE
"""

    script_path = script_dir / script_name
    script_path.write_text(script_content)
    script_path.chmod(0o755)

    return script_path


def generate_sweep_array_scripts_chunked(
    image_paths: List[Path],
    pipeline_names: List[str],
    manifest_path: Path,
    output_dir: Path,
    image_type: str,
    read_kwargs: Dict[str, Any],
    slurm_args: Dict[str, Any],
    array_limit: int,
    save_layers: Optional[Dict[str, bool]] = None,
    overlay_mode: str = "image",
    overlay_alpha: float = 0.3,
) -> List[Path]:
    """Generate chunked SLURM array scripts for large sweeps.

    When ``N_images * N_pipelines`` exceeds the cluster's MaxArraySize,
    splits the total task space into multiple array job scripts. Each chunk
    covers a contiguous range of the global (image, pipeline) index space.

    Args:
        image_paths: All images to process.
        pipeline_names: All pipeline names from the manifest.
        manifest_path: Path to the sweep manifest JSON.
        output_dir: Base output directory.
        image_type: ``"Image"`` or ``"GridImage"``.
        read_kwargs: Image read kwargs (nrows, ncols, etc.).
        slurm_args: SLURM parameters dict.
        array_limit: Maximum SLURM array size (from ``get_slurm_array_limit``).
        save_layers: Which layers to save.
        overlay_mode: Overlay saving mode.
        overlay_alpha: Overlay transparency.

    Returns:
        List of paths to generated scripts (one per chunk).
    """
    total_tasks = len(image_paths) * len(pipeline_names)
    chunks = calculate_optimal_array_chunks(total_tasks, array_limit)

    if len(chunks) == 1:
        # Single script, no chunking needed
        start, end = chunks[0]
        path = generate_sweep_array_script(
            image_paths=image_paths,
            pipeline_names=pipeline_names,
            manifest_path=manifest_path,
            output_dir=output_dir,
            image_type=image_type,
            read_kwargs=read_kwargs,
            slurm_args=slurm_args,
            save_layers=save_layers,
            overlay_mode=overlay_mode,
            overlay_alpha=overlay_alpha,
            global_offset=0,
            num_local_tasks=end - start,
            script_name="sweep_array_job.sh",
        )
        return [path]

    # Multiple chunks
    script_paths: List[Path] = []
    for chunk_idx, (start, end) in enumerate(chunks):
        script_name = f"sweep_array_job_chunk{chunk_idx}.sh"
        path = generate_sweep_array_script(
            image_paths=image_paths,
            pipeline_names=pipeline_names,
            manifest_path=manifest_path,
            output_dir=output_dir,
            image_type=image_type,
            read_kwargs=read_kwargs,
            slurm_args=slurm_args,
            save_layers=save_layers,
            overlay_mode=overlay_mode,
            overlay_alpha=overlay_alpha,
            global_offset=start,
            num_local_tasks=end - start,
            script_name=script_name,
        )
        script_paths.append(path)

    return script_paths
