"""
Directory scanning and dataset organization for the PhenoTypic CLI.

This module handles recursive directory scanning (1 level deep), image
file collection, and organization into datasets.

It also owns the **image manifest** (``--image-manifest``): a plain list of
image paths naming an approved subset of ``--input``. See
:func:`read_image_manifest` for the file format and
:func:`apply_image_manifest` for how it narrows a scan.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from phenotypic.sdk_.constants_ import IO
from phenotypic.sdk_ import default_output_dir_name, DIR_RESULTS, DIR_HDF
from phenotypic.sdk_._io_constants import file_fingerprint
from ._cli_types import Dataset


def _is_image_file(path: Path, valid_exts: set[str]) -> bool:
    """True if ``path`` is a real input image.

    Dotfiles are excluded, which an extension-only test does not do. macOS
    writes an AppleDouble ``._<name>`` sidecar beside every file on exFAT/FAT
    volumes — the usual format for an external drive — and
    ``Path("._x.tif").suffix`` is ``".tif"``, so each image would be counted
    twice. Observed on a real run: ``manifest.json`` reported
    ``total_images: 60`` for 30 images and ``is_complete: false`` on a run that
    had finished, which anything gating on completion reads as still running.
    """
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

    # Case 1: Single file
    if input_path.is_file():
        if input_path.suffix.lower() in valid_exts:
            datasets["single_image"] = [input_path]
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
        if _is_image_file(p, valid_exts)
    ]

    # Scan one level of subdirectories
    subdatasets = {}
    for subdir in input_path.iterdir():
        if not subdir.is_dir():
            continue

        # Collect images in this subdirectory
        sub_images = [
            p for p in subdir.iterdir()
            if _is_image_file(p, valid_exts)
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


# ---------------------------------------------------------------------------
# Image manifest (``--image-manifest``)
# ---------------------------------------------------------------------------


class ImageManifestError(ValueError):
    """Raised when an image manifest is unreadable or does not match --input."""


def image_manifest_digest(manifest_path: Path) -> str:
    """Return the SHA-256 content digest of an image manifest.

    This is the **file's bytes**, not a digest of the resolved image set, so
    it is the same number the MCP server binds into a plan token
    (``image_manifest_digest``, spec ``05-deploy-and-slurm.md`` §5.4). One
    definition on both sides means the server can compare its bound value
    against what a run recorded, and any edit to the artifact a human approved
    — including one that happens to resolve to the same images — invalidates
    the approval rather than being quietly tolerated.

    **Digest-format warning.** The returned value carries the ``"sha256:"``
    prefix, matching :func:`~phenotypic.sdk_._io_constants.directory_digest`,
    :func:`~phenotypic._services.staging.subset_digest`, and the
    ``*_fingerprint`` family. That is not cosmetic here: this value and
    ``subset_digest`` sit side by side as fields of one plan-token record and
    are string-compared across the tool boundary at ``deploy_start``, so a
    record spelling two of its digests two ways is a mismatch waiting for the
    first caller who copies the wrong one. ``pipeline_content_digest``
    (``_cli/_cli_staged_resume.py``) is the remaining **bare**-hex digest; it
    keeps that spelling because it is persisted in resume state written by
    runs already on disk, and it never appears in the token. The two forms are
    the same hash over different inputs and **never string-compare equal**.

    Args:
        manifest_path: Path to the ``.images`` manifest.

    Returns:
        ``"sha256:<lowercase hex>"``.

    Raises:
        OSError: If the manifest cannot be read.
    """
    # Delegated, not re-implemented: this value and ``subset_digest`` are
    # string-compared as fields of one plan-token record, so the two must not
    # be able to drift into different framings of the same bytes.
    return file_fingerprint(Path(manifest_path))


def read_image_manifest(manifest_path: Path) -> List[str]:
    """Parse an image manifest into its ordered raw entries.

    Format — deliberately a plain list rather than the JSON envelope
    ``load_staged_manifest`` reads, because this artifact is written by the
    MCP server at plan time, bound by a content digest, and read by a human
    deciding whether to spend cluster hours. It is:

    * UTF-8 text, one image path per line; a leading byte-order mark is
      tolerated (``utf-8-sig``) so an editor-written manifest does not fail as
      a phantom unknown path — the digest is over the raw bytes either way, so
      accepting it moves nothing the server bound;
    * blank lines are ignored;
    * a line whose first non-whitespace character is ``#`` is a comment;
    * surrounding whitespace on a path line is stripped;
    * each path is either absolute or relative to ``--input``;
    * no Unicode normalization is applied, on either side. An NFD entry
      against an NFC filename is refused as an unknown path rather than
      guessed at — the fail-closed direction, and the only one that keeps the
      approved count honest.

    Nothing is deduplicated and nothing is sorted: order and multiplicity are
    reported as written, and :func:`apply_image_manifest` is what rejects a
    repeat. A manifest that silently collapsed duplicates would process fewer
    images than the count a human approved.

    Args:
        manifest_path: Path to the ``.images`` manifest.

    Returns:
        The manifest's path entries, in file order.

    Raises:
        ImageManifestError: If the file cannot be read, is not valid UTF-8, or
            contains no entries.
    """
    manifest_path = Path(manifest_path)
    try:
        text = manifest_path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ImageManifestError(
            f"Cannot read image manifest {manifest_path}: {exc}"
        ) from exc
    except UnicodeDecodeError as exc:
        raise ImageManifestError(
            f"Image manifest {manifest_path} is not valid UTF-8 text: {exc}"
        ) from exc

    entries = [
        stripped
        for line in text.splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    ]
    if not entries:
        raise ImageManifestError(
            f"Image manifest {manifest_path} lists no images. An empty "
            "manifest is refused rather than treated as 'process everything'."
        )
    return entries


def apply_image_manifest(
    image_paths_by_dataset: Dict[str, List[Path]],
    manifest_path: Path,
    input_path: Path,
) -> Dict[str, List[Path]]:
    """Narrow a scan of ``input_path`` to the images an approved manifest names.

    The manifest selects *within* the scan; it never adds to it. That is what
    keeps every image's identity unchanged: ``work_id_for_image`` derives the
    relative path from ``config.input_path``, so a manifest run and a
    parent-directory run over the same image produce the same work ID, and
    resume, retry, and SLURM continuation all still line up.

    Every entry must therefore resolve to an image the scan already found.
    An entry that does not is an error naming it, rather than a silent
    omission — the caller approved a specific count.

    Selection carries each entry's *scan identity* — the ``(dataset, path)``
    pair the entry matched — rather than re-testing membership by resolved
    path afterwards. Resolved-path membership follows symlinks, so in a tree
    where two scan entries alias one real file (a symlinked image, or a
    symlinked dataset directory) a manifest naming one of them would select
    both, turning an approved line into two units of compute. On top of that,
    the selected total is checked against the entry count: one line in, one
    image out, or the run is refused. That is what makes "the count approved
    is the count that runs" a checked property rather than an emergent one.

    Args:
        image_paths_by_dataset: Output of :func:`scan_directory_structure`.
        manifest_path: Path to the ``.images`` manifest.
        input_path: The ``--input`` root the manifest's relative paths and the
            scan are both expressed against.

    Returns:
        The same mapping shape, keeping only manifest-named images and
        dropping datasets left empty. Within a dataset, scan order is
        preserved rather than manifest order, so the work list does not depend
        on how the manifest happened to be written.

    Raises:
        ImageManifestError: If the manifest is unreadable or empty, names a
            path the scan did not find, names the same image twice, or
            selects a number of images other than the number it names.
    """
    entries = read_image_manifest(manifest_path)
    input_path = Path(input_path)

    # Two lookups, because a scan can contain aliases of one real file (a
    # symlinked image, or a symlinked dataset directory — ordinary staging
    # practice on this cluster). ``by_spelling`` answers "which scan entry did
    # this manifest line name", so an aliased entry selects the image it was
    # written as; ``scanned`` is the resolved-path fallback that still finds a
    # differently-spelled but equivalent path (``..`` segments, or an absolute
    # entry against a relatively-spelled scan).
    scanned: Dict[Path, tuple[str, Path]] = {}
    by_spelling: Dict[Path, tuple[str, Path]] = {}
    for dataset_name, image_paths in image_paths_by_dataset.items():
        for image_path in image_paths:
            scanned.setdefault(
                _resolved(image_path), (dataset_name, image_path)
            )
            by_spelling.setdefault(Path(image_path), (dataset_name, image_path))

    selected: set[Path] = set()
    chosen: Dict[str, set[Path]] = {}
    for entry in entries:
        candidate = Path(entry)
        if not candidate.is_absolute():
            candidate = input_path / candidate
        resolved = _resolved(candidate)
        match = by_spelling.get(candidate) or scanned.get(resolved)
        if match is None:
            raise ImageManifestError(
                f"Image manifest {manifest_path} names {entry!r}, which is "
                f"not one of the images found under --input {input_path}. "
                "Manifest entries must be images of the input tree; the "
                "manifest selects a subset, it cannot introduce new inputs."
            )
        if resolved in selected:
            raise ImageManifestError(
                f"Image manifest {manifest_path} names {entry!r} more than "
                "once. Duplicates are refused rather than deduplicated so "
                "the image count stays the one that was approved."
            )
        selected.add(resolved)
        dataset_name, image_path = match
        chosen.setdefault(dataset_name, set()).add(image_path)

    # Rebuild from the identities the entry loop chose, never from a second
    # membership test over resolved paths: resolved-path membership selects
    # *every* alias of a named image, so one approved line becomes two units
    # of compute.
    filtered: Dict[str, List[Path]] = {}
    for dataset_name, image_paths in image_paths_by_dataset.items():
        wanted = chosen.get(dataset_name)
        if not wanted:
            continue
        filtered[dataset_name] = [p for p in image_paths if p in wanted]

    kept_total = sum(len(paths) for paths in filtered.values())
    if kept_total != len(entries):
        raise ImageManifestError(
            f"Image manifest {manifest_path} names {len(entries)} image(s) "
            f"but selecting them under --input {input_path} yielded "
            f"{kept_total}. The count that runs must be the count that was "
            "approved, so an aliased or ambiguous scan is refused rather "
            "than run."
        )
    return filtered


def _resolved(path: Path) -> Path:
    """Normalize a path for manifest/scan comparison without requiring it exist."""
    return Path(path).resolve(strict=False)


def scan_hdf_outputs(output_dir: Path) -> List[Dataset]:
    """
    Discover datasets already written as HDF by a previous forward run.

    Walks ``<output_dir>/results/``; for each subdirectory containing a
    non-empty ``hdf/`` directory, constructs a :class:`Dataset` whose
    ``images`` are the sorted ``*.h5`` files in that ``hdf/``,
    ``input_dir`` is the ``hdf/`` directory itself, ``output_dir`` is the
    supplied root output directory, and ``name`` is the subdirectory name.

    Intended for measure mode, which skips detection and reloads
    HDFs emitted by a prior forward run.

    Args:
        output_dir: Root output directory from a previous forward run.
            Expected to contain ``<output_dir>/results/<dataset>/hdf/``.

    Returns:
        List of :class:`Dataset` objects, one per subdirectory with a
        non-empty ``hdf/``.  Empty ``results/`` and missing ``results/``
        both raise ``ValueError``; per-dataset empty ``hdf/`` folders are
        skipped silently.

    Raises:
        ValueError: If no HDFs are found under ``<output_dir>/results``.
    """
    output_dir = Path(output_dir)
    results_dir = output_dir / DIR_RESULTS

    datasets: List[Dataset] = []

    if results_dir.is_dir():
        for subdir in sorted(results_dir.iterdir()):
            if not subdir.is_dir():
                continue

            hdf_dir = subdir / DIR_HDF
            if not hdf_dir.is_dir():
                continue

            # Skip dotfiles: on an exFAT/FAT volume macOS leaves an
            # AppleDouble `._<name>.h5` beside every HDF, and it is binary
            # junk, not an HDF — `--mode measure` on such a tree would try to
            # load it.
            hdf_files = sorted(
                p for p in hdf_dir.glob("*.h5") if not p.name.startswith(".")
            )
            if not hdf_files:
                continue

            datasets.append(
                Dataset(
                    name=subdir.name,
                    images=hdf_files,
                    input_dir=hdf_dir,
                    output_dir=output_dir,
                )
            )

    if not datasets:
        raise ValueError(
            f"No HDF outputs found under {results_dir}. "
            f"--mode measure expects a previous forward run to have written "
            f"HDF files under <output-dir>/results/<dataset>/hdf/*.h5."
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

    # Single file case
    if input_path.is_file():
        if input_path.suffix.lower() not in valid_exts:
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
            if _is_image_file(p, valid_exts)
    )

    # Count subdirectory images
    subdir_counts = {}
    for subdir in input_path.iterdir():
        if not subdir.is_dir():
            continue

        sub_count = sum(
                1 for p in subdir.iterdir()
                if _is_image_file(p, valid_exts)
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
