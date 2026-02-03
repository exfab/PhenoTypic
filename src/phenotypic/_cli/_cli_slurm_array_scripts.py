"""
SLURM array job script generation for the PhenoTypic CLI.

This module generates bash scripts for SLURM array jobs, enabling efficient
batch processing of images with minimal queue overhead. Each array job script
processes a chunk of images from a dataset using array task indexing.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Dict, List, Tuple

from ._cli_slurm_scripts import generate_slurm_directives
from ._cli_types import Dataset, ExecutionConfig
from ._cli_utils import get_python_command


def generate_array_job_script(
    dataset: Dataset,
    array_indices: Tuple[int, int],
    config: ExecutionConfig,
    output_dir: Path,
    chunk_id: int = 0,
) -> Path:
    """
    Generate a SLURM array job script for processing a dataset chunk.

    Creates a bash script with SBATCH directives for array job submission.
    The script builds an array of image paths and uses $SLURM_ARRAY_TASK_ID
    to index into the array for parallel processing.

    Args:
        dataset: Dataset containing images to process
        array_indices: (start, end) tuple for this chunk (0-based, end exclusive)
        config: Execution configuration with SLURM parameters
        output_dir: Base output directory
        chunk_id: Chunk number for multi-chunk datasets (default: 0)

    Returns:
        Path to generated array job script

    Examples:
        >>> from pathlib import Path
        >>> dataset = Dataset(
        ...     name="plate1",
        ...     images=[Path(f"image_{i}.tif") for i in range(100)],
        ...     input_dir=Path("."),
        ...     output_dir=Path("./output")
        ... )
        >>> config = ExecutionConfig(...)  # doctest: +SKIP
        >>> script = generate_array_job_script(
        ...     dataset, (0, 100), config, Path("./output")
        ... )  # doctest: +SKIP

    Notes:
        - Array indices are 0-based (Python/bash convention)
        - End index is exclusive (slice notation)
        - Generated script is executable (chmod 0o755)
        - Logs use SLURM %A (job ID) and %a (task ID) placeholders
    """
    # Extract image subset for this chunk
    start_idx, end_idx = array_indices
    chunk_images = dataset.images[start_idx:end_idx]

    if not chunk_images:
        raise ValueError(
            f"Empty chunk for dataset {dataset.name}: indices ({start_idx}, {end_idx})"
        )

    # Create script directory
    script_dir = output_dir / "slurm_scripts" / dataset.name
    script_dir.mkdir(parents=True, exist_ok=True)

    # Generate job name
    if chunk_id == 0 and end_idx == len(dataset.images):
        # Single chunk, simpler name
        job_name = f"pheno-{dataset.name}"
        script_name = "array_job.sh"
    else:
        # Multiple chunks, include chunk ID
        job_name = f"pheno-{dataset.name}-chunk{chunk_id}"
        script_name = f"array_job_chunk{chunk_id}.sh"

    # Generate log paths (using SLURM placeholders)
    log_dir = output_dir / "logs" / "slurm" / dataset.name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{dataset.name}_%A_%a.log"

    # Generate SBATCH directives with array specification
    slurm_args_with_array = config.slurm_args.copy()
    # Don't add --array here, we'll add it manually after other directives
    directives = generate_slurm_directives(
        job_name=job_name,
        slurm_args=slurm_args_with_array,
        output_log=log_path,
        error_log=log_path,  # Combined log
    )

    # Add array directive (0-based indexing)
    num_tasks = len(chunk_images)
    array_directive = f"#SBATCH --array=0-{num_tasks - 1}"

    # Build image list for bash array
    # Use absolute paths for reliability
    image_list_lines = []
    for img_path in chunk_images:
        # Quote paths for safety (spaces, special chars)
        quoted_path = shlex.quote(str(img_path.absolute()))
        image_list_lines.append(f"    {quoted_path}")

    image_list_content = "\n".join(image_list_lines)

    # Build command arguments for single-image processor
    event_log = output_dir / "processing_events.log"

    # Get Python command (uses uv run python if available)
    python_cmd, _ = get_python_command()

    cmd_parts = [
        *python_cmd,
        "-m",
        "phenotypic._cli._cli_process_single",
        "--pipeline",
        shlex.quote(str(config.pipeline_json.absolute())),
        "--image",
        '"${CURRENT_IMAGE}"',  # Will be populated from array
        "--output-dir",
        shlex.quote(str(output_dir.absolute())),
        "--dataset-name",
        shlex.quote(dataset.name),
        "--image-type",
        config.image_type,
    ]

    # Add grid parameters if GridImage
    if config.image_type == "GridImage":
        cmd_parts.extend(["--nrows", str(config.nrows)])
        cmd_parts.extend(["--ncols", str(config.ncols)])

    # Add bit depth if specified
    if config.bit_depth is not None:
        cmd_parts.extend(["--bit-depth", str(config.bit_depth)])

    # Add detect mode if not default
    if config.detect_mode != "gray":
        cmd_parts.extend(["--detect-mode", config.detect_mode])

    # Add save layer flags
    if config.save_rgb:
        cmd_parts.append("--save-rgb")
    if config.save_gray:
        cmd_parts.append("--save-gray")
    if config.save_detect_mat:
        cmd_parts.append("--save-enh-gray")
    if config.save_objmask:
        cmd_parts.append("--save-objmask")
    if config.save_objmap:
        cmd_parts.append("--save-objmap")
    if config.save_objmap_overlay:
        cmd_parts.append("--save-objmap-overlay")
    if config.save_detect_mat_overlay:
        cmd_parts.append("--save-enh-gray-overlay")
    if config.save_objmask_overlay:
        cmd_parts.append("--save-objmask-overlay")

    # Add extensions
    cmd_parts.extend(["--rgb-ext", config.rgb_ext])
    cmd_parts.extend(["--gray-ext", config.gray_ext])
    cmd_parts.extend(["--enh-gray-ext", config.detect_mat_ext])
    cmd_parts.extend(["--objmask-ext", config.objmask_ext])
    cmd_parts.extend(["--objmap-ext", config.objmap_ext])
    cmd_parts.extend(["--objmap-overlay-ext", config.objmap_overlay_ext])

    # Add overlay options
    cmd_parts.extend(["--overlay-mode", config.overlay_mode])
    cmd_parts.extend(["--overlay-alpha", str(config.overlay_alpha)])

    # Add dataset column flag (default is to include, so only add flag to exclude)
    if not config.include_dataset_column:
        cmd_parts.append("--no-dataset-column")

    # Add event log
    cmd_parts.extend(["--event-log", shlex.quote(str(event_log.absolute()))])

    # Join command with line continuations for readability
    cmd = " \\\n    ".join(cmd_parts)

    # Generate complete script
    script_content = f"""#!/bin/bash
{directives}
{array_directive}

# Auto-generated by PhenoTypic CLI v2.0 (SLURM array job mode)
# Dataset: {dataset.name}
# Chunk: {chunk_id} (images {start_idx}-{end_idx-1})
# Pipeline: {config.pipeline_json}

set -e  # Exit on error
set -u  # Exit on undefined variable

# Record start time
echo "======================================"
echo "Job ID: ${{SLURM_JOB_ID:-unknown}}"
echo "Array Task ID: ${{SLURM_ARRAY_TASK_ID:-unknown}}"
echo "Node: ${{SLURMD_NODENAME:-$(hostname)}}"
echo "Start Time: $(date)"
echo "======================================"

# Build image list (0-based indexing)
IMAGE_LIST=(
{image_list_content}
)

# Validate array task ID
if [ "${{SLURM_ARRAY_TASK_ID:-}}" = "" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID not set (not running in array job?)"
    exit 1
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "${{#IMAGE_LIST[@]}}" ]; then
    echo "ERROR: Array task ID $SLURM_ARRAY_TASK_ID exceeds image list size ${{#IMAGE_LIST[@]}}"
    exit 1
fi

# Get current image using array task ID
CURRENT_IMAGE="${{IMAGE_LIST[$SLURM_ARRAY_TASK_ID]}}"

echo "Processing image $((SLURM_ARRAY_TASK_ID + 1))/${{#IMAGE_LIST[@]}}: $CURRENT_IMAGE"
echo ""

# Run processing
{cmd}

EXIT_CODE=$?

echo ""
echo "======================================"
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
echo "======================================"

exit $EXIT_CODE
"""

    # Write script
    script_path = script_dir / script_name
    script_path.write_text(script_content)
    script_path.chmod(0o755)  # Make executable

    return script_path


def generate_all_array_job_scripts(
    datasets: List[Dataset],
    config: ExecutionConfig,
    output_dir: Path,
    array_limit: int,
) -> Dict[str, List[Path]]:
    """
    Generate array job scripts for all datasets with automatic chunking.

    Creates one or more array job scripts per dataset depending on the
    number of images and SLURM array size limits. Large datasets are
    automatically split into multiple chunks.

    Args:
        datasets: List of datasets to process
        config: Execution configuration
        output_dir: Base output directory
        array_limit: Maximum array size from SLURM configuration

    Returns:
        Dictionary mapping dataset names to lists of script paths.
        Each dataset may have multiple scripts if chunked.

    Examples:
        >>> datasets = [...]  # doctest: +SKIP
        >>> config = ExecutionConfig(...)  # doctest: +SKIP
        >>> scripts = generate_all_array_job_scripts(
        ...     datasets, config, Path("./output"), array_limit=1000
        ... )  # doctest: +SKIP
        >>> len(scripts["dataset1"])  # Number of chunks for dataset1  # doctest: +SKIP
        1

    Notes:
        - Datasets with <= array_limit images get single script
        - Large datasets split into multiple chunks
        - Chunk size determined by calculate_optimal_array_chunks()
    """
    from ._cli_slurm_config import calculate_optimal_array_chunks

    all_scripts = {}

    for dataset in datasets:
        num_images = len(dataset.images)

        if num_images == 0:
            # Skip empty datasets
            continue

        # Calculate chunks based on array limit
        chunks = calculate_optimal_array_chunks(num_images, array_limit)

        dataset_scripts = []
        for chunk_id, (start, end) in enumerate(chunks):
            script_path = generate_array_job_script(
                dataset=dataset,
                array_indices=(start, end),
                config=config,
                output_dir=output_dir,
                chunk_id=chunk_id,
            )
            dataset_scripts.append(script_path)

        all_scripts[dataset.name] = dataset_scripts

    return all_scripts
