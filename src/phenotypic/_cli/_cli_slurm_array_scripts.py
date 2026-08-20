"""
SLURM array job script generation for the PhenoTypic CLI.

This module generates bash scripts for SLURM array jobs, enabling efficient
batch processing of images with minimal queue overhead. Each array job script
processes a chunk of images from a dataset using array task indexing.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from ._cli_failure_tracker import file_sha256, work_id_for_image
from ._cli_types import Dataset, ExecutionConfig
from ._cli_update_state import (
    PROCESSING_GENERATION_ENV_VAR,
    SLURM_GENERATION_ENV_VAR,
)
from ._cli_utils import SLURM_THREAD_PIN_BASH, get_python_command
from phenotypic.sdk_ import event_log_path, logs_dir, slurm_scripts_dir
from phenotypic.sdk_.slurm import (
    SlurmArrayScriptSpec,
    write_slurm_array_script,
)

# Sentinel value inserted into the image list to trigger checkpoint aggregation
_CHECKPOINT_SENTINEL = "__PHENOTYPIC_CHECKPOINT__"
# Sentinel value inserted after each checkpoint to trigger manifest rebuild
_MANIFEST_SENTINEL = "__PHENOTYPIC_MANIFEST__"
def _set_worker_mode(cmd_parts: List[str], mode: str) -> None:
    """Set the worker mode token in a command-argument list."""
    mode_value_index = cmd_parts.index("--mode") + 1
    cmd_parts[mode_value_index] = mode


def _build_entry_list(
    chunk_images: List[Path],
    checkpoint_interval: Optional[int],
    is_last_chunk: bool = False,
) -> List[str]:
    """Build the task entry list, optionally interleaving checkpoint sentinels.

    Args:
        chunk_images: Image paths for this chunk.
        checkpoint_interval: Insert a sentinel every N images, or ``None``
            to skip sentinel insertion.
        is_last_chunk: Retained for caller compatibility. Terminal work is
            submitted as a separate dependent lifecycle job.

    Returns:
        List of absolute path strings with optional sentinel markers.
    """
    if checkpoint_interval is not None and checkpoint_interval > 0:
        entries: List[str] = []
        for i, img_path in enumerate(chunk_images):
            entries.append(str(img_path.absolute()))
            if (i + 1) % checkpoint_interval == 0:
                entries.append(_CHECKPOINT_SENTINEL)
                entries.append(_MANIFEST_SENTINEL)
        return entries

    return [str(img_path.absolute()) for img_path in chunk_images]


def _max_images_per_chunk(array_limit: int, checkpoint_interval: int) -> int:
    """Largest per-chunk image count whose entry list fits ``array_limit``.

    The bash entry list interleaves checkpoint/manifest sentinels every
    ``checkpoint_interval`` images. Terminal finalization is a separate job.
    The ``--array=0-N`` directive is sized from ``len(entries)`` and must
    satisfy ``len(entries) <= MaxArraySize``.
    """
    if checkpoint_interval is None or checkpoint_interval <= 0:
        return max(1, array_limit)
    limit = max(1, array_limit)
    interval = checkpoint_interval
    low, high = 1, limit
    while low < high:
        candidate = (low + high + 1) // 2
        entry_count = candidate + 2 * (candidate // interval)
        if entry_count <= limit:
            low = candidate
        else:
            high = candidate - 1
    return low


def _resolve_checkpoint_interval(config: ExecutionConfig) -> int:
    """Resolve checkpoint interval from config or auto-estimate from SLURM capacity.

    Args:
        config: Execution configuration with optional checkpoint_interval
            and SLURM partition details.

    Returns:
        Checkpoint interval clamped to [50, 500].
    """
    if config.checkpoint_interval is not None:
        return config.checkpoint_interval

    from ._cli_slurm_config import estimate_concurrent_capacity

    partition = config.slurm_args.get("slurm_partition", "")
    if partition:
        cpus_per_task = int(config.slurm_args.get("slurm_cpus_per_task", 1))
        mem_gb = float(config.slurm_args.get("mem_gb", 4.0))
        concurrent_capacity = estimate_concurrent_capacity(
            partition=partition,
            cpus_per_task=cpus_per_task,
            mem_gb_per_task=mem_gb,
        )
    else:
        concurrent_capacity = 100

    return max(50, min(3 * concurrent_capacity, 500))


def generate_array_job_script(
    dataset: Dataset,
    array_indices: Tuple[int, int],
    config: ExecutionConfig,
    output_dir: Path,
    chunk_id: int = 0,
    checkpoint_interval: Optional[int] = None,
    is_last_chunk: bool = False,
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
        checkpoint_interval: If set, insert checkpoint sentinel entries
            every N images so SLURM tasks can trigger chunk aggregation
        is_last_chunk: Retained for caller compatibility. Terminal work is
            submitted as a separate dependent lifecycle job.

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

    # Build task entries, interleaving checkpoint sentinels at regular intervals.
    # Process-only chunks contain images only. Their manifest publisher is a
    # separate scheduler-dependent finalizer, so it cannot race sibling array
    # tasks in the last chunk.
    if config.process_only_layer:
        entries = [str(img_path.absolute()) for img_path in chunk_images]
    else:
        entries = _build_entry_list(
            chunk_images, checkpoint_interval, is_last_chunk
        )

    # Create script directory
    script_dir = slurm_scripts_dir(output_dir) / dataset.name
    script_dir.mkdir(parents=True, exist_ok=True)

    # Generate job name
    if chunk_id == 0 and end_idx == len(dataset.images):
        # Single chunk, simpler name
        job_name = f"pht-{dataset.name}"
        script_name = "array_job.sh"
    else:
        # Multiple chunks, include chunk ID
        job_name = f"pht-{dataset.name}-chunk{chunk_id}"
        script_name = f"array_job_chunk{chunk_id}.sh"

    # Generate log paths (using SLURM placeholders)
    log_dir = logs_dir(output_dir) / "slurm" / dataset.name
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{dataset.name}_%A_%a.log"

    # Build command arguments for single-image processor
    event_log = event_log_path(output_dir)

    # Get Python command (uses uv run python if available)
    python_cmd, _ = get_python_command(for_slurm=True)

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
        "--mode",
        "full",
        "--input-root",
        shlex.quote(str(config.input_path.absolute())),
        "--expected-work-id",
        '"${CURRENT_WORK_ID}"',
        "--expected-input-sha256",
        '"${CURRENT_INPUT_SHA256}"',
        "--expected-pipeline-sha256",
        '"${EXPECTED_PIPELINE_SHA256}"',
        "--attempt-id",
        '"${CURRENT_ATTEMPT_ID}"',
    ]

    # Omit when unset so the worker falls back to the pipeline's preset.
    if config.image_type == "GridImage":
        if config.nrows is not None:
            cmd_parts.extend(["--nrows", str(config.nrows)])
        if config.ncols is not None:
            cmd_parts.extend(["--ncols", str(config.ncols)])

    # Add bit depth if specified
    if config.bit_depth is not None:
        cmd_parts.extend(["--bit-depth", str(config.bit_depth)])

    # Add detect mode if not default
    if config.detect_mode != "gray":
        cmd_parts.extend(["--detect-mode", config.detect_mode])

    # Add extension if non-default
    if config.ext != ".tiff":
        cmd_parts.extend(["--ext", config.ext])

    # Add overlay alpha if non-default
    if config.overlay_alpha != 0.3:
        cmd_parts.extend(["--overlay-alpha", str(config.overlay_alpha)])

    # Add dataset column flag (default is to include, so only add flag to exclude)
    if not config.include_dataset_column:
        cmd_parts.append("--no-dataset-column")

    # Durability is tri-state. Emit ONLY an explicit choice: an unset flag has
    # to stay unset so the worker re-detects SLURM on its own node, while
    # --no-durable-writes exists only here and would otherwise be lost, leaving
    # every array task silently fsyncing (spec §3.7).
    if config.durable_writes is not None:
        cmd_parts.append(
            "--durable-writes"
            if config.durable_writes
            else "--no-durable-writes"
        )

    # Process-only (apply-only) mode: export a single layer, mirror the input
    # tree, and skip overlays/measurement entirely. Mutually exclusive with the
    # measure / forward overlay flags.
    if config.process_only_layer:
        _set_worker_mode(cmd_parts, "process")
        cmd_parts.extend(
            [
                "--layer",
                config.process_only_layer,
            ]
        )
    elif config.measure_only:
        # Measure-only mode supersedes overlay generation; mutually exclusive.
        _set_worker_mode(cmd_parts, "measure")
    else:
        cmd_parts.append("--save-overlays")

    # Add event log
    cmd_parts.extend(["--event-log", shlex.quote(str(event_log.absolute()))])

    # Join command with line continuations for readability
    cmd = " \\\n    ".join(cmd_parts)

    # Build checkpoint command
    checkpoint_cmd_parts = [
        *python_cmd,
        "-m",
        "phenotypic._cli._cli_chunk_writer",
        "--output-dir",
        shlex.quote(str(output_dir.absolute())),
    ]
    checkpoint_cmd = " \\\n    ".join(checkpoint_cmd_parts)

    python_str = " ".join(python_cmd)
    q_output_dir = shlex.quote(str(output_dir.absolute()))

    manifest_cmd = (
        f"{python_str} -m phenotypic._cli._cli_checkpoint_handler "
        f"--output-dir {q_output_dir} "
        f"--checkpoint-type manifest"
    )

    # Process-only runs execute image work only. Their manifest finalizer is a
    # separate job submitted with ``afterany`` on the last chunk.
    if config.process_only_layer:
        dispatch_block = f"""# Apply-only image processing
echo "Processing image $((SLURM_ARRAY_TASK_ID + 1))/${{#IMAGE_LIST[@]}}: $CURRENT_IMAGE"
echo ""

{cmd}"""
    else:
        dispatch_block = f"""if [ "$CURRENT_IMAGE" = "{_CHECKPOINT_SENTINEL}" ]; then
    # Checkpoint task: aggregate per-image Parquets into a dashboard chunk
    echo "Running checkpoint aggregation (task $SLURM_ARRAY_TASK_ID)"
    echo ""

    {checkpoint_cmd}
elif [ "$CURRENT_IMAGE" = "{_MANIFEST_SENTINEL}" ]; then
    # Manifest task: rebuild manifest after checkpoint aggregation
    echo "Running manifest rebuild (task $SLURM_ARRAY_TASK_ID)"
    echo ""

    {manifest_cmd}
else
    # Normal image processing
    echo "Processing image $((SLURM_ARRAY_TASK_ID + 1))/${{#IMAGE_LIST[@]}}: $CURRENT_IMAGE"
    echo ""

    {cmd}
fi"""

    identity_assignment = (
        'CURRENT_WORK_ID="${EXPECTED_WORK_IDS[$SLURM_ARRAY_TASK_ID]}"\n'
        'CURRENT_INPUT_SHA256="${EXPECTED_INPUT_SHA256S[$SLURM_ARRAY_TASK_ID]}"\n'
        'CURRENT_ATTEMPT_ID="${ATTEMPT_IDS[$SLURM_ARRAY_TASK_ID]}"'
    )
    dispatch_block = f"{identity_assignment}\n\n{dispatch_block}"

    script_path = script_dir / script_name
    prelude = SLURM_THREAD_PIN_BASH
    identity_rows: list[tuple[str, str, str]] = []
    for entry in entries:
        if entry in {_CHECKPOINT_SENTINEL, _MANIFEST_SENTINEL}:
            identity_rows.append(("", "", ""))
            continue
        image_path = Path(entry)
        work_id, _ = work_id_for_image(config, dataset.name, image_path)
        identity_rows.append((work_id, file_sha256(image_path), uuid4().hex))
    prelude += "\nEXPECTED_WORK_IDS=(\n" + "\n".join(
        f"    {shlex.quote(row[0])}" for row in identity_rows
    ) + "\n)"
    prelude += "\nEXPECTED_INPUT_SHA256S=(\n" + "\n".join(
        f"    {shlex.quote(row[1])}" for row in identity_rows
    ) + "\n)"
    prelude += "\nATTEMPT_IDS=(\n" + "\n".join(
        f"    {shlex.quote(row[2])}" for row in identity_rows
    ) + "\n)"
    prelude += (
        "\nEXPECTED_PIPELINE_SHA256="
        f"{shlex.quote(file_sha256(config.pipeline_json))}"
    )
    if config.processing_generation:
        prelude += (
            "\nexport "
            f"{PROCESSING_GENERATION_ENV_VAR}="
            f"{shlex.quote(config.processing_generation)}"
        )
    if config.slurm_generation:
        prelude += (
            "\nexport "
            f"{SLURM_GENERATION_ENV_VAR}="
            f"{shlex.quote(config.slurm_generation)}"
        )
    write_slurm_array_script(
        script_path,
        SlurmArrayScriptSpec(
            job_name=job_name,
            slurm_args=config.slurm_args,
            log_path=log_path,
            task_indices=entries,
            body=dispatch_block,
            prelude=prelude,
            comments=[
                "# Auto-generated by PhenoTypic CLI v2.0 (SLURM array job mode)",
                f"# Dataset: {dataset.name}",
                f"# Chunk: {chunk_id} (images {start_idx}-{end_idx - 1})",
                f"# Pipeline: {config.pipeline_json}",
                "# Build image list (0-based indexing)",
                (
                    "# Entries may include sentinel markers for checkpoint, "
                    "and manifest tasks"
                ),
            ],
            array_name="IMAGE_LIST",
            current_var="CURRENT_IMAGE",
            missing_task_id_message=(
                "ERROR: SLURM_ARRAY_TASK_ID not set (not running in array job?)"
            ),
            bounds_error_message=(
                "ERROR: Array task ID $SLURM_ARRAY_TASK_ID exceeds image list "
                "size ${#IMAGE_LIST[@]}"
            ),
        ),
    )

    return script_path


def generate_terminal_finalizer_script(
    config: ExecutionConfig,
    output_dir: Path,
) -> Path:
    """Write the dependent terminal publisher for an ordinary SLURM run."""
    python_cmd, _ = get_python_command(for_slurm=True)
    checkpoint_type = (
        "manifest" if config.process_only_layer is not None else "finalize"
    )
    finalizer_name = (
        "process_export_finalizer"
        if config.process_only_layer is not None
        else "ordinary_finalizer"
    )
    command = " ".join(
        [
            *(shlex.quote(part) for part in python_cmd),
            "-m",
            "phenotypic._cli._cli_checkpoint_handler",
            "--output-dir",
            shlex.quote(str(output_dir.absolute())),
            "--checkpoint-type",
            checkpoint_type,
        ]
    )
    script_path = slurm_scripts_dir(output_dir) / f"{finalizer_name}.sh"
    log_path = logs_dir(output_dir) / "slurm" / f"{finalizer_name}_%A_%a.log"
    prelude = SLURM_THREAD_PIN_BASH
    if config.processing_generation:
        prelude += (
            "\nexport "
            f"{PROCESSING_GENERATION_ENV_VAR}="
            f"{shlex.quote(config.processing_generation)}"
        )
    if config.slurm_generation:
        prelude += (
            "\nexport "
            f"{SLURM_GENERATION_ENV_VAR}="
            f"{shlex.quote(config.slurm_generation)}"
        )
    return write_slurm_array_script(
        script_path,
        SlurmArrayScriptSpec(
            job_name="pht-finalizer",
            slurm_args=config.slurm_args,
            log_path=log_path,
            task_indices=[0],
            body=command,
            prelude=prelude,
            comments=[
                "# Ordinary SLURM terminal finalizer",
                "# Submitted after the final image chunk becomes terminal",
            ],
        ),
    )


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

    checkpoint_interval = _resolve_checkpoint_interval(config)
    # Reserve array slots for sentinels so len(entries) stays within MaxArraySize.
    image_chunk_limit = _max_images_per_chunk(array_limit, checkpoint_interval)
    all_scripts: Dict[str, List[Path]] = {}

    # Pre-compute chunk lists per dataset so we can identify the last chunk
    # across all datasets for finalizer sentinel placement.
    active_datasets = [ds for ds in datasets if ds.images]
    dataset_chunks = [
        (ds, calculate_optimal_array_chunks(len(ds.images), image_chunk_limit))
        for ds in active_datasets
    ]

    # Total number of chunks across all datasets
    total_chunks = sum(len(chunks) for _, chunks in dataset_chunks)
    chunk_counter = 0

    for dataset, chunks in dataset_chunks:
        scripts: List[Path] = []
        for chunk_id, (start, end) in enumerate(chunks):
            chunk_counter += 1
            last_chunk = chunk_counter == total_chunks
            scripts.append(
                generate_array_job_script(
                    dataset=dataset,
                    array_indices=(start, end),
                    config=config,
                    output_dir=output_dir,
                    chunk_id=chunk_id,
                    checkpoint_interval=checkpoint_interval,
                    is_last_chunk=last_chunk,
                )
            )
        all_scripts[dataset.name] = scripts

    return all_scripts
