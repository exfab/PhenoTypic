"""Resolve each Browse source image to a (row, time) matrix record.

The per-axis source picker (spec §5.2): row ∈ {folder | {plate} pattern |
CSV column}; time ∈ {EXIF | folder | {time} pattern | CSV column}. The
``folder`` time source supports folder-per-timepoint layouts (each dataset
folder is a timepoint, so repeated filenames order by folder not name). CSV
joins are folder-scoped by image **stem** (no path column, spec §15.4);
pattern rows are folder-scoped (spec §15.5). Pure — Dash wiring lives in the
callbacks.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from phenotypic.gui.browse._plate_pattern import parse_plate_identity

__all__ = ["BrowseAxisConfig", "build_browse_records"]


@dataclass(frozen=True)
class BrowseAxisConfig:
    """Which source feeds each Timeline axis."""

    row_source: str          # "folder" | "pattern" | "csv"
    time_source: str         # "exif" | "folder" | "pattern" | "csv"
    pattern: str = ""
    advanced_pattern: bool = False
    csv_image_col: str | None = None
    row_csv_col: str | None = None
    time_csv_col: str | None = None


_UNMATCHED = "unmatched"


def _sandbox_rel(src_root_rel: str, folder: str, filename: str) -> str:
    parts = [p for p in (src_root_rel, folder) if p and p != "."]
    return PurePosixPath(*parts, filename).as_posix() if parts else filename


def build_browse_records(
    datasets: Mapping[str, Sequence[str]],
    src_root_rel: str,
    config: BrowseAxisConfig,
    *,
    csv_rows: Sequence[Mapping[str, object]] | None = None,
    capture_time_of: Callable[[str], str | None],
) -> tuple[list[dict[str, object]], list[str]]:
    """Build matrix records + warnings for the Browse Timeline.

    Args:
        datasets: ``{dataset_folder: [filename, ...]}`` (from ``list_datasets``).
        src_root_rel: Source root relative to the sandbox (POSIX).
        config: The per-axis source selection.
        csv_rows: Parsed metadata-CSV rows (dicts), or ``None``. Required when
            either axis source is ``"csv"``.
        capture_time_of: ``sandbox_rel_path -> capture-time str | None`` used
            for the EXIF time source.

    Returns:
        ``(records, warnings)``. Each record is
        ``{"row_value": str, "time_value": str, "cell_ref": sandbox_rel}``.
        Warnings are human-readable strings for the UI (e.g. CSV stem
        collisions).
    """
    warnings: list[str] = []
    uses_csv = "csv" in (config.row_source, config.time_source)

    # CSV lookup, keyed by image STEM (matches the existing stem-join convention).
    csv_by_stem: dict[str, Mapping[str, object]] = {}
    if uses_csv and csv_rows and config.csv_image_col:
        for row in csv_rows:
            raw = row.get(config.csv_image_col)
            if raw is None:
                continue
            csv_by_stem[Path(str(raw)).stem] = row

    # Pattern matches, computed per folder so rows stay folder-scoped.
    pattern_by_folder: dict[str, dict[str, tuple[str | None, str | None]]] = {}
    uses_pattern = "pattern" in (config.row_source, config.time_source)
    if uses_pattern:
        for folder, files in datasets.items():
            stems = [Path(f).stem for f in files]
            matches = parse_plate_identity(
                stems, config.pattern, advanced=config.advanced_pattern
            )
            pattern_by_folder[folder] = {
                pm.stem: (pm.plate, pm.time) for pm in matches
            }

    # Cross-folder stem collision check (only meaningful when CSV drives an axis).
    if uses_csv:
        seen: dict[str, set[str]] = {}
        for folder, files in datasets.items():
            for filename in files:
                seen.setdefault(Path(filename).stem, set()).add(folder)
        collided = sorted(s for s, folders in seen.items() if len(folders) > 1)
        if collided:
            warnings.append(
                "CSV axis: stem(s) appear in multiple folders and cannot be "
                f"disambiguated per folder: {', '.join(collided)}"
            )

    records: list[dict[str, object]] = []
    for folder, files in datasets.items():
        for filename in files:
            stem = Path(filename).stem
            rel = _sandbox_rel(src_root_rel, folder, filename)
            plate, ptime = pattern_by_folder.get(folder, {}).get(stem, (None, None))
            csv_row = csv_by_stem.get(stem)

            row_value = _resolve_row(config, folder, plate, csv_row)
            time_value = _resolve_time(
                config, ptime, csv_row, rel, capture_time_of, filename, folder
            )
            records.append(
                {"row_value": row_value, "time_value": time_value, "cell_ref": rel}
            )
    return records, warnings


def _resolve_row(
    config: BrowseAxisConfig,
    folder: str,
    plate: str | None,
    csv_row: Mapping[str, object] | None,
) -> str:
    if config.row_source == "folder":
        return folder
    if config.row_source == "pattern":
        if plate is None:
            return _UNMATCHED
        return plate if folder == "." else f"{folder}/{plate}"
    # csv
    if csv_row is None or config.row_csv_col is None:
        return _UNMATCHED
    return str(csv_row.get(config.row_csv_col, _UNMATCHED))


def _resolve_time(
    config: BrowseAxisConfig,
    ptime: str | None,
    csv_row: Mapping[str, object] | None,
    rel: str,
    capture_time_of: Callable[[str], str | None],
    filename: str,
    folder: str,
) -> str:
    if config.time_source == "exif":
        return capture_time_of(rel) or filename
    if config.time_source == "folder":
        # Folder-per-timepoint layouts: the dataset folder name IS the time
        # axis, so repeated filenames across folders order by folder, not name.
        return folder
    if config.time_source == "pattern":
        return ptime if ptime is not None else (capture_time_of(rel) or filename)
    # csv
    if csv_row is None or config.time_csv_col is None:
        return ""
    return str(csv_row.get(config.time_csv_col, ""))
