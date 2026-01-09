"""
SLURM bash script generation for the PhenoTypic CLI.

This module generates standalone bash scripts for autonomous SLURM execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any
import shlex

from ._cli_types import Dataset, ExecutionConfig


def generate_slurm_directives(
    job_name: str,
    slurm_kwds: Dict[str, Any],
    output_log: Path,
    error_log: Path
) -> str:
    """
    Generate SBATCH directive lines for SLURM script.
    
    Args:
        job_name: Job name
        slurm_kwds: SLURM parameters dict
        output_log: Path for stdout log
        error_log: Path for stderr log
        
    Returns:
        String with all #SBATCH directives
    """
    directives = [f"#SBATCH --job-name={job_name}"]
    
    # Add output/error logs
    directives.append(f"#SBATCH --output={output_log}")
    directives.append(f"#SBATCH --error={error_log}")
    
    # Add user-provided SLURM parameters
    for key, value in slurm_kwds.items():
        # Convert submitit-style keys to SBATCH directive names
        directive_name = key.replace("slurm_", "").replace("_", "-")
        
        # Handle special cases
        if key == "time_min" or key == "slurm_time":
            # Convert minutes to HH:MM:SS format if needed
            if isinstance(value, int):
                hours = value // 60
                minutes = value % 60
                value = f"{hours:02d}:{minutes:02d}:00"
            directive_name = "time"
        elif key == "mem_gb":
            value = f"{value}G"
            directive_name = "mem"
        elif key == "slurm_mem":
            directive_name = "mem"
        elif key == "slurm_mem_per_cpu":
            directive_name = "mem-per-cpu"
        elif key == "slurm_cpus_per_task":
            directive_name = "cpus-per-task"
        elif key == "slurm_gpus_per_node":
            directive_name = "gpus-per-node"
        
        directives.append(f"#SBATCH --{directive_name}={value}")
    
    return "\n".join(directives)


def generate_image_processing_script(
    image_path: Path,
    dataset: Dataset,
    config: ExecutionConfig,
    output_dir: Path,
    event_log: Path,
    script_dir: Path
) -> Path:
    """
    Generate standalone bash script for processing a single image.
    
    Args:
        image_path: Path to image file
        dataset: Dataset containing this image
        config: Execution configuration
        output_dir: Base output directory
        event_log: Path to event log file
        script_dir: Directory to save script in
        
    Returns:
        Path to generated script file
    """
    # Create script directory
    script_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate job name
    image_stem = image_path.stem
    job_name = f"pt_{dataset.name}_{image_stem}"
    
    # Generate log paths
    log_dir = output_dir / "logs" / "slurm" / dataset.name
    log_dir.mkdir(parents=True, exist_ok=True)
    output_log = log_dir / f"{image_stem}_%j.out"
    error_log = log_dir / f"{image_stem}_%j.err"
    
    # Generate SBATCH directives
    directives = generate_slurm_directives(
        job_name=job_name,
        slurm_kwds=config.slurm_kwds,
        output_log=output_log,
        error_log=error_log
    )
    
    # Build command arguments
    cmd_parts = [
        "python", "-m", "phenotypic._core._cli_process_single",
        "--pipeline", shlex.quote(str(config.pipeline_json.absolute())),
        "--image", shlex.quote(str(image_path.absolute())),
        "--output-dir", shlex.quote(str(output_dir.absolute())),
        "--dataset-name", shlex.quote(dataset.name),
        "--image-type", config.image_type,
    ]
    
    # Add grid parameters if GridImage
    if config.image_type == "GridImage":
        cmd_parts.extend(["--nrows", str(config.nrows)])
        cmd_parts.extend(["--ncols", str(config.ncols)])
    
    # Add bit depth if specified
    if config.bit_depth is not None:
        cmd_parts.extend(["--bit-depth", str(config.bit_depth)])
    
    # Add save layer flags
    if config.save_rgb:
        cmd_parts.append("--save-rgb")
    if config.save_gray:
        cmd_parts.append("--save-gray")
    if config.save_enh_gray:
        cmd_parts.append("--save-enh-gray")
    if config.save_objmask:
        cmd_parts.append("--save-objmask")
    if config.save_objmap:
        cmd_parts.append("--save-objmap")
    if config.save_objmap_rgb:
        cmd_parts.append("--save-objmap-rgb")
    
    # Add extensions
    cmd_parts.extend(["--rgb-ext", config.rgb_ext])
    cmd_parts.extend(["--gray-ext", config.gray_ext])
    cmd_parts.extend(["--enh-gray-ext", config.enh_gray_ext])
    cmd_parts.extend(["--objmask-ext", config.objmask_ext])
    cmd_parts.extend(["--objmap-ext", config.objmap_ext])
    cmd_parts.extend(["--objmap-rgb-ext", config.objmap_rgb_ext])
    
    # Add dataset column flag
    if config.include_dataset_column:
        cmd_parts.append("--include-dataset-column")
    
    # Add event log
    cmd_parts.extend(["--event-log", shlex.quote(str(event_log.absolute()))])
    
    # Join command
    cmd = " \\\n    ".join(cmd_parts)
    
    # Generate complete script
    script_content = f"""#!/bin/bash
{directives}

# Auto-generated by PhenoTypic CLI v2.0
# Image: {dataset.name}/{image_path.name}
# Pipeline: {config.pipeline_json}

set -e  # Exit on error
set -u  # Exit on undefined variable

# Record start time
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: ${{SLURM_JOB_ID:-local}}"

# Run processing
{cmd}

echo "Job completed: $(date)"
exit 0
"""
    
    # Write script
    script_path = script_dir / f"{dataset.name}_{image_stem}.sh"
    script_path.write_text(script_content)
    script_path.chmod(0o755)  # Make executable
    
    return script_path


def generate_all_image_scripts(
    datasets: List[Dataset],
    config: ExecutionConfig,
    output_dir: Path
) -> Dict[str, List[Path]]:
    """
    Generate bash scripts for all images across all datasets.
    
    Args:
        datasets: List of datasets to process
        config: Execution configuration
        output_dir: Base output directory
        
    Returns:
        Dictionary mapping dataset names to lists of script paths
    """
    script_dir = output_dir / "slurm_scripts"
    event_log = output_dir / "processing_events.log"
    
    all_scripts = {}
    
    for dataset in datasets:
        dataset_scripts = []
        
        for image_path in dataset.images:
            script_path = generate_image_processing_script(
                image_path=image_path,
                dataset=dataset,
                config=config,
                output_dir=output_dir,
                event_log=event_log,
                script_dir=script_dir / dataset.name
            )
            dataset_scripts.append(script_path)
        
        all_scripts[dataset.name] = dataset_scripts
    
    return all_scripts
