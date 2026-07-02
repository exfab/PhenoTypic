"""Pure builder for the REMBI run manifest (deliverables/rembi.yaml).

Folds the per-colony measurements mirror up to each REMBI module's scope:
distinct-collapse (scalar-or-list) for biosample/specimen-prep/acquisition,
a per-image file list for image-data, a feature catalog for analyzed-data, and
a one-value-per-field study section (CSV constants overridable by a study file).
No I/O; see _write below for serialization.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from phenotypic.schema import REMBI_MODULE, header_to_module

from ._metadata_helpers import is_metadata_header

# module -> manifest section key
_SECTION = {
    REMBI_MODULE.STUDY: "study",
    REMBI_MODULE.BIOSAMPLE: "biosample",
    REMBI_MODULE.SPECIMEN_PREP: "specimen_preparation",
    REMBI_MODULE.IMAGE_ACQUISITION: "image_acquisition",
    REMBI_MODULE.UNCATEGORIZED: "uncategorized",
}


def _distinct(series: pd.Series) -> Any:
    vals = sorted({v for v in series.dropna().tolist()}, key=str)
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else vals


def _label_of(header: str) -> str:
    # strip the "<Category>_" prefix -> bare label
    return header.split("_", 1)[1] if "_" in header else header


def build_rembi_manifest(
    measurements: pd.DataFrame,
    image_metadata: list[dict],
    study_config: dict | None = None,
) -> dict:
    idx = header_to_module()
    manifest: dict[str, dict] = {}

    # --- distinct-collapse sections (study/biosample/specimen/acquisition/uncat)
    for col in measurements.columns:
        if not is_metadata_header(str(col)):
            continue  # measurement/locator columns handled in analyzed_data
        module = idx.get(col, REMBI_MODULE.UNCATEGORIZED)
        section = _SECTION.get(module)
        if section is None:
            continue
        value = _distinct(measurements[col])
        if value is None:
            continue
        manifest.setdefault(section, {})[_label_of(col)] = value

    # --- study file overrides csv constants
    if study_config:
        study = manifest.setdefault("study", {})
        study.update({k: v for k, v in study_config.items() if v is not None})

    # --- image_data: per-image files + rollups (ALWAYS present)
    files = [
        {
            "name": im.get("ImageName"),
            "uuid": im.get("UUID"),
            "bit_depth": im.get("BitDepth"),
            "image_type": im.get("ImageType"),
        }
        for im in image_metadata
    ]
    manifest["image_data"] = {
        "n_images": len(files),
        "bit_depth": sorted({f["bit_depth"] for f in files if f["bit_depth"] is not None}),
        "files": files,
    }

    # --- analyzed_data: feature catalog grouped by category prefix
    features: dict[str, list[str]] = {}
    for col in measurements.columns:
        col = str(col)
        if is_metadata_header(col) or "_" not in col:
            continue
        cat, label = col.split("_", 1)
        features.setdefault(cat, []).append(label)
    if features:
        manifest["analyzed_data"] = {"features": {k: sorted(v) for k, v in features.items()}}

    return manifest


def write_rembi_manifest(
    output_dir,
    measurements: pd.DataFrame,
    image_metadata: list[dict],
    study_config: dict | None = None,
):
    """Best-effort: build + write the manifest to deliverables/rembi.yaml.

    Never raises — logs and returns None on any failure, so finalize is never
    blocked (same contract as the best-effort metadata.csv copy). No-ops
    (returns ``None``) when the deliverables dir does not exist.
    """
    import logging
    from pathlib import Path

    import yaml

    from ._io_constants import rembi_manifest_path

    log = logging.getLogger(__name__)
    try:
        manifest = build_rembi_manifest(measurements, image_metadata, study_config)
        path = rembi_manifest_path(Path(output_dir))
        if not path.parent.exists():
            return None
        path.write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))
        return path
    except Exception:  # noqa: BLE001 - best-effort, never block finalize
        log.warning("REMBI manifest write failed", exc_info=True)
        return None
