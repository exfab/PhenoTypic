"""SLURM array job script generation for the sweep CLI.

Generates bash scripts that process one or more (image, pipeline) pairs per
array task, using image-major 2D indexing over the image x pipeline product
space.
"""

from __future__ import annotations

import math
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
    verbose: bool = False,
    save_intermediates: bool = False,
) -> str:
    """Build the worker CLI command string (without image/pipeline args).

    Args:
        manifest_path: Path to the sweep manifest JSON.
        output_dir: Base output directory.
        image_type: ``"Image"`` or ``"GridImage"``.
        read_kwargs: Image read kwargs (nrows, ncols, etc.).
        verbose: When True, append ``--verbose`` to enable per-operation
            debug logging in the worker process.
        save_intermediates: When True, append ``--save-intermediates``
            to save intermediate image state after each pipeline operation.

    Returns:
        Command string with ``${CURRENT_IMAGE}`` and ``${CURRENT_PIPELINE}``
        placeholders.
    """
    event_log = output_dir / "processing_events.log"
    python_cmd, _ = get_python_command(for_slurm=True)

    cmd_parts = [
        *python_cmd,
        "-m",
        "phenotypic.sweep._sweep_cli._sweep_process_image",
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

    cmd_parts.extend(["--event-log", shlex.quote(str(event_log.absolute()))])
    cmd_parts.extend(["--pipeline-name", '"${CURRENT_PIPELINE}"'])

    if verbose:
        cmd_parts.append("--verbose")

    if save_intermediates:
        cmd_parts.append("--save-intermediates")

    return " \\\n    ".join(cmd_parts)


def generate_sweep_array_script(
    image_paths: List[Path],
    pipeline_names: List[str],
    manifest_path: Path,
    output_dir: Path,
    image_type: str,
    read_kwargs: Dict[str, Any],
    slurm_args: Dict[str, Any],
    global_offset: int = 0,
    num_local_tasks: Optional[int] = None,
    script_name: str = "sweep_array_job.sh",
    verbose: bool = False,
    batch_size: int = 1,
    save_intermediates: bool = False,
) -> Path:
    """Generate a SLURM array job script for sweep processing.

    Each array task processes one or more (image, pipeline) pairs via
    image-major 2D indexing over the image x pipeline product space.

    Args:
        image_paths: All images to process.
        pipeline_names: All pipeline names from the manifest.
        manifest_path: Path to the sweep manifest JSON.
        output_dir: Base output directory.
        image_type: ``"Image"`` or ``"GridImage"``.
        read_kwargs: Image read kwargs (nrows, ncols, etc.).
        slurm_args: SLURM parameters dict.
        global_offset: Offset for 2D indexing (used by chunked scripts).
        num_local_tasks: Number of *array elements* in this chunk
            (defaults to ``ceil(total_tasks / batch_size)``).
        script_name: Filename for the generated script.
        verbose: When True, pass ``--verbose`` to the worker CLI.
        batch_size: Number of (image, pipeline) pairs processed
            sequentially by each array task.  When ``1`` (default), each
            array task processes exactly one pair — identical to the
            original un-batched behavior.
        save_intermediates: When True, pass ``--save-intermediates``
            to the worker CLI to save intermediate image state after
            each pipeline operation.

    Returns:
        Path to the generated script.
    """
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
    effective_tasks = math.ceil(total_tasks / batch_size)
    local_tasks = num_local_tasks if num_local_tasks is not None else effective_tasks
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
        verbose=verbose,
        save_intermediates=save_intermediates,
    )

    # Offset section (only included for chunked scripts)
    if global_offset > 0:
        offset_section = f"""
GLOBAL_OFFSET={global_offset}
BASE_TASK_ID=$((SLURM_ARRAY_TASK_ID + GLOBAL_OFFSET))
"""
    else:
        offset_section = """
BASE_TASK_ID=$SLURM_ARRAY_TASK_ID
"""

    # Build processing section — batched loop or single-pair
    if batch_size > 1:
        processing_section = f"""
# --- Batched processing: {batch_size} (image, pipeline) pairs per array task ---
BATCH_SIZE={batch_size}
TOTAL_TASKS={total_tasks}
BATCH_START=$((BASE_TASK_ID * BATCH_SIZE))
BATCH_END=$((BATCH_START + BATCH_SIZE))
if [ "$BATCH_END" -gt "$TOTAL_TASKS" ]; then
    BATCH_END=$TOTAL_TASKS
fi

BATCH_OK=0
BATCH_FAIL=0

echo "======================================"
echo "Job ID: ${{SLURM_JOB_ID:-unknown}}"
echo "Array Task ID: ${{SLURM_ARRAY_TASK_ID:-unknown}}"
echo "Batch: tasks $BATCH_START..$((BATCH_END - 1)) of $TOTAL_TASKS"
echo "Node: ${{SLURMD_NODENAME:-$(hostname)}}"
echo "Start Time: $(date)"
echo "======================================"

for GLOBAL_TASK_ID in $(seq $BATCH_START $((BATCH_END - 1))); do
    # 2D indexing (image-major): cycles through all images first
    IMAGE_IDX=$((GLOBAL_TASK_ID % N_IMAGES))
    PIPE_IDX=$((GLOBAL_TASK_ID / N_IMAGES))

    if [ "$IMAGE_IDX" -ge "${{#IMAGE_LIST[@]}}" ]; then
        echo "SKIP: Image index $IMAGE_IDX out of range"
        continue
    fi
    if [ "$PIPE_IDX" -ge "${{#PIPELINE_NAMES[@]}}" ]; then
        echo "SKIP: Pipeline index $PIPE_IDX out of range"
        continue
    fi

    CURRENT_IMAGE="${{IMAGE_LIST[$IMAGE_IDX]}}"
    CURRENT_PIPELINE="${{PIPELINE_NAMES[$PIPE_IDX]}}"

    echo ""
    echo "--- Pair $((GLOBAL_TASK_ID + 1))/$TOTAL_TASKS ---"
    echo "Image: $CURRENT_IMAGE"
    echo "Pipeline: $CURRENT_PIPELINE"

    {cmd} && BATCH_OK=$((BATCH_OK + 1)) || BATCH_FAIL=$((BATCH_FAIL + 1))
done

echo ""
echo "======================================"
echo "Batch complete: $BATCH_OK succeeded, $BATCH_FAIL failed"
echo "End Time: $(date)"
echo "======================================"

# Fail the task only if ALL pairs in the batch failed
if [ "$BATCH_OK" -eq 0 ] && [ "$BATCH_FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
"""
    else:
        processing_section = f"""
# --- Single-pair processing ---
GLOBAL_TASK_ID=$BASE_TASK_ID

# 2D indexing (image-major): cycles through all images first
IMAGE_IDX=$((GLOBAL_TASK_ID % N_IMAGES))
PIPE_IDX=$((GLOBAL_TASK_ID / N_IMAGES))

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

    script_content = f"""#!/bin/bash
{directives}
{array_directive}

# Auto-generated by PhenoTypic Sweep CLI
# Manifest: {manifest_path}
# Images: {num_images}, Pipelines: {num_pipelines}
# Total (image, pipeline) pairs: {total_tasks}
# Batch size: {batch_size}

set -e  # Exit on error
set -u  # Exit on undefined variable

IMAGE_LIST=(
{image_list_content}
)

PIPELINE_NAMES=(
{pipeline_list_content}
)

N_IMAGES={num_images}

if [ "${{SLURM_ARRAY_TASK_ID:-}}" = "" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set"
    exit 1
fi
{offset_section}{processing_section}"""

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
    verbose: bool = False,
    batch_size: int = 1,
    save_intermediates: bool = False,
) -> List[Path]:
    """Generate chunked SLURM array scripts for large sweeps.

    When the effective number of array elements (after batching) exceeds
    the cluster's ``MaxArraySize``, splits the task space into multiple
    array job scripts.  Each chunk covers a contiguous range of the
    global (image, pipeline) index space.

    Args:
        image_paths: All images to process.
        pipeline_names: All pipeline names from the manifest.
        manifest_path: Path to the sweep manifest JSON.
        output_dir: Base output directory.
        image_type: ``"Image"`` or ``"GridImage"``.
        read_kwargs: Image read kwargs (nrows, ncols, etc.).
        slurm_args: SLURM parameters dict.
        array_limit: Maximum SLURM array size (from ``get_slurm_array_limit``).
        verbose: When True, pass ``--verbose`` to the worker CLI.
        batch_size: (image, pipeline) pairs per array task.
        save_intermediates: When True, pass ``--save-intermediates``
            to the worker CLI to save intermediate image state after
            each pipeline operation.

    Returns:
        List of paths to generated scripts (one per chunk).
    """
    total_tasks = len(image_paths) * len(pipeline_names)
    effective_tasks = math.ceil(total_tasks / batch_size)
    chunks = calculate_optimal_array_chunks(effective_tasks, array_limit)

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
            global_offset=0,
            num_local_tasks=end - start,
            script_name="sweep_array_job.sh",
            verbose=verbose,
            batch_size=batch_size,
            save_intermediates=save_intermediates,
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
            global_offset=start,
            num_local_tasks=end - start,
            script_name=script_name,
            verbose=verbose,
            batch_size=batch_size,
            save_intermediates=save_intermediates,
        )
        script_paths.append(path)

    return script_paths
