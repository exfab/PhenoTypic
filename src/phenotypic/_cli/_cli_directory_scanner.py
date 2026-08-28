"""
Directory scanning and dataset organization for the PhenoTypic CLI.

This module handles recursive directory scanning (1 level deep), image
file collection, and organization into datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from phenotypic.sdk_.constants_ import IO
from phenotypic.sdk_ import (
    DIR_RESULTS,
    DIR_ZARR,
    STORE_SUFFIX,
    default_output_dir_name,
)
from ._cli_types import Dataset


def _is_store_dir(path: Path) -> bool:
    """True if ``path`` is an OME-Zarr store directory.

    Tested by **name**, not by opening the store: the scanner runs over every
    entry of every candidate directory, and reading a root ``zarr.json`` per
    entry would cost an open per file at 10k-image scale. A directory named
    ``*.ome.zarr`` that is not a store fails later, loudly, in ``imread``.

    This is the same shape as :func:`scan_store_outputs`'s glob, which matches
    directories non-recursively for the same reason its docstring gives: a
    store is a directory full of files, so anything recursive costs roughly
    forty stat calls per store.
    """
    return (
        path.is_dir()
        and not path.name.startswith(".")
        and path.name.endswith(STORE_SUFFIX)
    )


def _is_image_input(path: Path, valid_exts: set[str]) -> bool:
    """True if ``path`` is a real input image -- a flat file or a store.

    Dotfiles are excluded, which an extension-only test does not do. macOS
    writes an AppleDouble ``._<name>`` sidecar beside every file on exFAT/FAT
    volumes — the usual format for an external drive — and
    ``Path("._x.tif").suffix`` is ``".tif"``, so each image would be counted
    twice. Observed on a real run: ``manifest.json`` reported
    ``total_images: 60`` for 30 images and ``is_complete: false`` on a run that
    had finished, which anything gating on completion reads as still running.

    The same dot test also excludes ``promote_store``'s in-flight
    ``.<stem>.ome.zarr.<uuid>.part`` and ``.trash`` siblings, which is exactly
    right: neither is a readable store.

    A ``*.ome.zarr`` store counts as one input even though it is a directory.
    Its contents are never enumerated: see :func:`_is_store_dir`.
    """
    if _is_store_dir(path):
        return True
    return (
        path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in valid_exts
    )


def generate_timestamped_output_dir() -> Path:
    """
    Generate timestamped output directory name.

    Returns:
        Path like ./phenotypic_results_20260108_143022/
    """
    return Path(default_output_dir_name())


def scan_directory_structure(input_path: Path) -> Dict[str, List[Path]]:
    """
    Scan directory structure and organize images by dataset.

    Supports:
    - Single file: returns {"single_image": [file]}
    - Flat directory: returns {"<dir_name>": [img1, img2, ...]}
    - Recursive (1 level): returns {"dataset1": [...], "dataset2": [...]}

    Mixed directories (root images + subdirectories) are NOT allowed.

    Args:
        input_path: Path to image file or directory

    Returns:
        Dictionary mapping dataset names to lists of image paths

    Raises:
        FileNotFoundError: If input_path doesn't exist
        ValueError: If no valid images found or mixed directory structure
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    # Get valid image extensions
    valid_exts = set(IO.ACCEPTED_FILE_EXTENSIONS + IO.RAW_FILE_EXTENSIONS)
    valid_exts = {ext.lower() for ext in valid_exts}

    datasets = {}

    # Case 1: Single file, or a single store
    if input_path.is_file() or _is_store_dir(input_path):
        if _is_image_input(input_path, valid_exts):
            datasets["single_image"] = [input_path]
            return datasets
        else:
            raise ValueError(
                    f"File {input_path.name} is not a supported image format. "
                    f"Supported: {', '.join(sorted(valid_exts))}, "
                    f"or an *.ome.zarr store"
            )

    # Case 2 & 3: Directory (flat or recursive)
    if not input_path.is_dir():
        raise ValueError(f"Input path is neither file nor directory: {input_path}")

    # Collect images directly in root directory
    root_images = [
        p for p in input_path.iterdir()
        if _is_image_input(p, valid_exts)
    ]

    # Scan one level of subdirectories
    subdatasets = {}
    for subdir in input_path.iterdir():
        # A store IS a directory. Without this it is enumerated as a dataset,
        # finds no image files inside itself, and silently contributes
        # nothing -- so the images disappear from the run with no error.
        if not subdir.is_dir() or _is_store_dir(subdir):
            continue

        # Collect images in this subdirectory
        sub_images = [
            p for p in subdir.iterdir()
            if _is_image_input(p, valid_exts)
        ]

        if sub_images:
            subdatasets[subdir.name] = sorted(sub_images)

    # Validate: reject mixed directories (root images + subdirectories with images)
    if root_images and subdatasets:
        raise ValueError(
            f"Mixed input structure not allowed: found {len(root_images)} image(s) "
            f"in root directory AND {len(subdatasets)} subdirectory dataset(s). "
            f"Use either a flat directory of images OR a directory containing "
            f"dataset subdirectories, not both."
        )

    # Assign datasets based on structure
    if root_images:
        # Flat directory: use the directory name as dataset name
        datasets[input_path.name] = sorted(root_images)
    else:
        # Recursive structure: use subdirectory names
        datasets.update(subdatasets)

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


def scan_store_outputs(output_dir: Path) -> List[Dataset]:
    """
    Discover datasets already written as OME-Zarr stores by a previous run.

    Walks ``<output_dir>/results/``; for each subdirectory containing a
    non-empty ``zarr/`` directory, constructs a :class:`Dataset` whose
    ``images`` are the sorted ``*.ome.zarr`` **store directories** in that
    ``zarr/``, ``input_dir`` is the ``zarr/`` directory itself, ``output_dir``
    is the supplied root output directory, and ``name`` is the subdirectory
    name.

    Intended for measure mode and overlay recompile, which skip detection and
    reload the stores emitted by a prior forward run.

    The glob is deliberately **non-recursive** and matches **directories**. A
    store *is* a directory full of files, so ``rglob`` would descend into every
    one of them — roughly forty stat calls per store, 400k at 10k images — and
    an ``is_file()`` filter would find nothing at all.

    Args:
        output_dir: Root output directory from a previous forward run.
            Expected to contain ``<output_dir>/results/<dataset>/zarr/``.

    Returns:
        List of :class:`Dataset` objects, one per subdirectory with a
        non-empty ``zarr/``.  Empty ``results/`` and missing ``results/``
        both raise ``ValueError``; per-dataset empty ``zarr/`` folders are
        skipped silently.

    Raises:
        ValueError: If no stores are found under ``<output_dir>/results``.
    """
    output_dir = Path(output_dir)
    results_dir = output_dir / DIR_RESULTS

    datasets: List[Dataset] = []

    if results_dir.is_dir():
        for subdir in sorted(results_dir.iterdir()):
            if not subdir.is_dir():
                continue

            zarr_dir = subdir / DIR_ZARR
            if not zarr_dir.is_dir():
                continue

            # Skip dotted entries. This is the AppleDouble guard the HDF scan
            # carried (macOS leaves a `._<name>` beside every entry on
            # exFAT/FAT), and it now also excludes the in-flight
            # `.<stem>.ome.zarr.<uuid>.part` / `.trash` siblings that
            # `promote_store` creates — which is exactly right, since neither
            # is a readable store.
            stores = sorted(
                p
                for p in zarr_dir.glob(f"*{STORE_SUFFIX}")
                if p.is_dir() and not p.name.startswith(".")
            )
            if not stores:
                continue

            datasets.append(
                Dataset(
                    name=subdir.name,
                    images=stores,
                    input_dir=zarr_dir,
                    output_dir=output_dir,
                )
            )

    if not datasets:
        raise ValueError(
            f"No OME-Zarr outputs found under {results_dir}. "
            f"--mode measure expects a previous forward run to have written "
            f"image stores under "
            f"<output-dir>/results/<dataset>/zarr/*{STORE_SUFFIX}."
        )

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


def get_input_structure_summary(input_path: Path) -> Dict[str, Any]:
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

    Raises:
        FileNotFoundError: If input_path doesn't exist
        ValueError: If no valid images found or mixed directory structure
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    valid_exts = set(IO.ACCEPTED_FILE_EXTENSIONS + IO.RAW_FILE_EXTENSIONS)
    valid_exts = {ext.lower() for ext in valid_exts}

    # Single file case, or a single store
    if input_path.is_file() or _is_store_dir(input_path):
        if not _is_image_input(input_path, valid_exts):
            raise ValueError(f"File {input_path.name} is not a supported image format")
        return {
            "type"        : "single_file",
            "total_images": 1,
            "datasets"    : {"single_image": 1}
        }

    # Directory case - count without loading
    dataset_counts = {}

    # Count root images
    root_count = sum(
            1 for p in input_path.iterdir()
            if _is_image_input(p, valid_exts)
    )

    # Count subdirectory images
    subdir_counts = {}
    for subdir in input_path.iterdir():
        # The same store skip as scan_directory_structure: a store is a
        # directory, and enumerating it as a dataset drops its image silently.
        if not subdir.is_dir() or _is_store_dir(subdir):
            continue

        sub_count = sum(
                1 for p in subdir.iterdir()
                if _is_image_input(p, valid_exts)
        )

        if sub_count > 0:
            subdir_counts[subdir.name] = sub_count

    # Validate: reject mixed directories
    if root_count > 0 and subdir_counts:
        raise ValueError(
            f"Mixed input structure not allowed: found {root_count} image(s) "
            f"in root directory AND {len(subdir_counts)} subdirectory dataset(s). "
            f"Use either a flat directory of images OR a directory containing "
            f"dataset subdirectories, not both."
        )

    # Assign dataset counts based on structure
    if root_count > 0:
        # Flat directory: use the directory name
        dataset_counts[input_path.name] = root_count
        structure_type = "flat_directory"
    elif subdir_counts:
        # Recursive structure: use subdirectory names
        dataset_counts.update(subdir_counts)
        structure_type = "recursive"
    else:
        raise ValueError(f"No valid images found in {input_path}")

    total_images = sum(dataset_counts.values())

    # Special case: single file in flat directory
    if total_images == 1 and structure_type == "flat_directory":
        structure_type = "single_file"

    return {
        "type"        : structure_type,
        "total_images": total_images,
        "datasets"    : dataset_counts
    }
