"""
Directory scanning and dataset organization for the PhenoTypic CLI.

This module handles recursive directory scanning (1 level deep), image
file collection, and organization into datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from datetime import datetime

from phenotypic.tools_.constants_ import IO
from ._cli_types import Dataset


def generate_timestamped_output_dir() -> Path:
    """
    Generate timestamped output directory name.
    
    Returns:
        Path like ./phenotypic_results_20260108_143022/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"./phenotypic_results_{timestamp}")


def scan_directory_structure(input_path: Path) -> Dict[str, List[Path]]:
    """
    Scan directory structure and organize images by dataset.
    
    Supports:
    - Single file: returns {"_root": [file]}
    - Flat directory: returns {"_root": [img1, img2, ...]}
    - Recursive (1 level): returns {"dataset1": [...], "dataset2": [...]}
    - Mixed: returns {"_root": [...], "dataset1": [...], "dataset2": [...]}
    
    Args:
        input_path: Path to image file or directory
        
    Returns:
        Dictionary mapping dataset names to lists of image paths
        Use "_root" as dataset name for images in the root directory
        
    Raises:
        FileNotFoundError: If input_path doesn't exist
        ValueError: If no valid images found
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Get valid image extensions
    valid_exts = set(IO.ACCEPTED_FILE_EXTENSIONS + IO.RAW_FILE_EXTENSIONS)
    valid_exts = {ext.lower() for ext in valid_exts}

    datasets = {}

    # Case 1: Single file
    if input_path.is_file():
        if input_path.suffix.lower() in valid_exts:
            datasets["_root"] = [input_path]
            return datasets
        else:
            raise ValueError(
                    f"File {input_path.name} is not a supported image format. "
                    f"Supported: {', '.join(sorted(valid_exts))}"
            )

    # Case 2 & 3: Directory (flat or recursive)
    if not input_path.is_dir():
        raise ValueError(f"Input path is neither file nor directory: {input_path}")

    # Collect images directly in root directory
    root_images = [
        p for p in input_path.iterdir()
        if p.is_file() and p.suffix.lower() in valid_exts
    ]
    if root_images:
        datasets["_root"] = sorted(root_images)

    # Scan one level of subdirectories
    for subdir in input_path.iterdir():
        if not subdir.is_dir():
            continue

        # Collect images in this subdirectory
        sub_images = [
            p for p in subdir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
        ]

        if sub_images:
            datasets[subdir.name] = sorted(sub_images)

    # Validate we found at least some images
    if not datasets:
        raise ValueError(f"No valid images found in {input_path}")

    return datasets


def organize_by_dataset(
        image_paths_by_dataset: Dict[str, List[Path]],
        output_dir: Path
) -> List[Dataset]:
    """
    Convert dictionary of image paths into Dataset objects.
    
    Args:
        image_paths_by_dataset: Dict from scan_directory_structure()
        output_dir: Base output directory for all datasets
        
    Returns:
        List of Dataset objects with proper input/output paths
    """
    datasets = []

    for dataset_name, image_paths in image_paths_by_dataset.items():
        if not image_paths:
            continue

        # Determine input directory (parent of first image)
        input_dir = image_paths[0].parent

        # Create Dataset object
        dataset = Dataset(
                name=dataset_name,
                images=image_paths,
                input_dir=input_dir,
                output_dir=output_dir
        )
        datasets.append(dataset)

    return datasets


def collect_image_paths(input_path: Path) -> List[Path]:
    """
    Simple flat collection of all image paths (for backward compatibility).
    
    This is a simpler version that returns a flat list without dataset
    organization. Useful for code that doesn't need dataset structure.
    
    Args:
        input_path: Path to image file or directory
        
    Returns:
        Flat list of all image paths found
    """
    datasets = scan_directory_structure(input_path)
    all_images = []
    for images in datasets.values():
        all_images.extend(images)
    return all_images


def get_input_structure_summary(input_path: Path) -> Dict[str, any]:
    """
    Get summary information about input structure without collecting files.
    
    Useful for dry-run mode to quickly understand the input.
    
    Args:
        input_path: Path to image file or directory
        
    Returns:
        Dictionary with structure information:
        {
            "type": "single_file" | "flat_directory" | "recursive",
            "total_images": int,
            "datasets": {"dataset1": count1, "dataset2": count2, ...}
        }
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    valid_exts = set(IO.ACCEPTED_FILE_EXTENSIONS + IO.RAW_FILE_EXTENSIONS)
    valid_exts = {ext.lower() for ext in valid_exts}

    # Single file case
    if input_path.is_file():
        if input_path.suffix.lower() not in valid_exts:
            raise ValueError(f"File {input_path.name} is not a supported image format")
        return {
            "type"        : "single_file",
            "total_images": 1,
            "datasets"    : {"_root": 1}
        }

    # Directory case - count without loading
    dataset_counts = {}

    # Count root images
    root_count = sum(
            1 for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in valid_exts
    )
    if root_count > 0:
        dataset_counts["_root"] = root_count

    # Count subdirectory images
    for subdir in input_path.iterdir():
        if not subdir.is_dir():
            continue

        sub_count = sum(
                1 for p in subdir.iterdir()
                if p.is_file() and p.suffix.lower() in valid_exts
        )

        if sub_count > 0:
            dataset_counts[subdir.name] = sub_count

    if not dataset_counts:
        raise ValueError(f"No valid images found in {input_path}")

    total_images = sum(dataset_counts.values())

    # Determine structure type
    has_root = "_root" in dataset_counts
    has_subdirs = len(dataset_counts) > 1 or not has_root

    if total_images == 1:
        structure_type = "single_file"
    elif has_subdirs:
        structure_type = "recursive"
    else:
        structure_type = "flat_directory"

    return {
        "type"        : structure_type,
        "total_images": total_images,
        "datasets"    : dataset_counts
    }
