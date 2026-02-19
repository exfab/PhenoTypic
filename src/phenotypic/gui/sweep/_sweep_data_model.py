"""Data model for sweep output browsing.

Scans a sweep output directory, parses the manifest, and indexes all result
files into lookup structures for the napari viewer widgets.  This module has
**no** Qt or napari dependencies.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = {".png", ".tiff", ".tif", ".jpg", ".jpeg"}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SweepImageFile:
    """A single image file inside the sweep results tree."""

    path: Path
    image_stem: str
    component: str
    pipeline_name: str


@dataclass
class PipelineConfig:
    """Parsed configuration for one pipeline from the sweep manifest."""

    name: str
    config_group: str
    operations: List[Dict]
    measurements: List[Dict]
    raw_json: dict


@dataclass
class SweepOutputData:
    """Fully indexed sweep output ready for the viewer widgets."""

    root_dir: Path
    manifest_raw: dict
    pipeline_configs: Dict[str, PipelineConfig]
    image_files: List[SweepImageFile]
    pipeline_names: List[str]
    image_stems: List[str]
    components: List[str]
    # [pipeline][stem][component] -> SweepImageFile
    by_pipeline: Dict[str, Dict[str, Dict[str, SweepImageFile]]] = field(
        default_factory=dict,
    )
    # [stem][component][pipeline] -> SweepImageFile
    by_image: Dict[str, Dict[str, Dict[str, SweepImageFile]]] = field(
        default_factory=dict,
    )


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------


class SweepOutputScanner:
    """Scan and index a sweep output directory."""

    @staticmethod
    def scan(sweep_dir: Path) -> SweepOutputData:
        """Scan a sweep output directory and build lookup indexes.

        Args:
            sweep_dir: Root of the sweep output (contains
                ``sweep_manifest.json`` and ``results/``).

        Returns:
            Fully populated :class:`SweepOutputData`.

        Raises:
            FileNotFoundError: If *sweep_dir* or its manifest is missing.
        """
        sweep_dir = Path(sweep_dir).resolve()
        manifest_path = sweep_dir / "sweep_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No sweep_manifest.json found in {sweep_dir}"
            )

        manifest_raw, pipeline_configs = SweepOutputScanner._parse_manifest(
            manifest_path,
        )

        results_dir = sweep_dir / "results"
        if results_dir.is_dir():
            image_files = SweepOutputScanner._scan_results(results_dir)
        else:
            logger.warning(
                "Results directory not found: %s", results_dir,
            )
            image_files = []

        # Derive sorted unique lists
        pipeline_names = sorted({f.pipeline_name for f in image_files})
        image_stems = sorted({f.image_stem for f in image_files})
        components = sorted({f.component for f in image_files})

        # Build lookup indexes
        by_pipeline: Dict[str, Dict[str, Dict[str, SweepImageFile]]] = {}
        by_image: Dict[str, Dict[str, Dict[str, SweepImageFile]]] = {}

        for f in image_files:
            by_pipeline.setdefault(f.pipeline_name, {}).setdefault(
                f.image_stem, {},
            )[f.component] = f

            by_image.setdefault(f.image_stem, {}).setdefault(
                f.component, {},
            )[f.pipeline_name] = f

        logger.debug(
            "Scan complete: %d files, %d pipelines, %d stems, "
            "%d components",
            len(image_files), len(pipeline_names),
            len(image_stems), len(components),
        )

        return SweepOutputData(
            root_dir=sweep_dir,
            manifest_raw=manifest_raw,
            pipeline_configs=pipeline_configs,
            image_files=image_files,
            pipeline_names=pipeline_names,
            image_stems=image_stems,
            components=components,
            by_pipeline=by_pipeline,
            by_image=by_image,
        )

    @staticmethod
    def detect_sweep_dir(path: Optional[Path] = None) -> Path:
        """Find the sweep output directory.

        Args:
            path: Explicit directory, or ``None`` to use the current working
                directory.

        Returns:
            Resolved path that contains ``sweep_manifest.json``.

        Raises:
            FileNotFoundError: If no manifest is found in *path*.
        """
        target = Path(path).resolve() if path else Path.cwd().resolve()
        manifest = target / "sweep_manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"No sweep_manifest.json found in {target}. "
                "Pass the path to a sweep output directory."
            )
        return target

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_manifest(
        manifest_path: Path,
    ) -> Tuple[dict, Dict[str, PipelineConfig]]:
        """Parse the sweep manifest JSON into :class:`PipelineConfig` objects.

        Args:
            manifest_path: Path to ``sweep_manifest.json``.

        Returns:
            Tuple of (raw manifest dict, pipeline configs dict keyed by
            pipeline name).
        """
        manifest_raw = json.loads(manifest_path.read_text())
        configs: Dict[str, PipelineConfig] = {}

        for cfg_name, cfg_data in manifest_raw.get("configs", {}).items():
            for pipe_name, pipe_dict in cfg_data.get("pipelines", {}).items():
                # Extract operations from pipe_cfgs
                operations: List[Dict] = []
                pipe_cfgs = pipe_dict.get("pipe_cfgs", {})
                for op_key in sorted(pipe_cfgs.keys()):
                    op_data = pipe_cfgs[op_key]
                    operations.append(
                        {
                            "name": op_key,
                            "class": op_data.get("class", "Unknown"),
                            "params": op_data.get("params", {}),
                        }
                    )

                # Extract measurement operations
                measurements: List[Dict] = []
                meas_cfgs = pipe_dict.get("meas_cfgs", {})
                for meas_key in sorted(meas_cfgs.keys()):
                    meas_data = meas_cfgs[meas_key]
                    measurements.append(
                        {
                            "name": meas_key,
                            "class": meas_data.get("class", "Unknown"),
                            "params": meas_data.get("params", {}),
                        }
                    )

                configs[pipe_name] = PipelineConfig(
                    name=pipe_name,
                    config_group=cfg_name,
                    operations=operations,
                    measurements=measurements,
                    raw_json=pipe_dict,
                )

        logger.debug(
            "Parsed %d pipeline configs from manifest",
            len(configs),
        )
        return manifest_raw, configs

    @staticmethod
    def _scan_results(results_dir: Path) -> List[SweepImageFile]:
        """Walk ``results/<pipeline>/<component>/<image>`` for image files.

        Skips the ``measurements/`` subdirectory (CSV data, not images).

        Args:
            results_dir: The ``results/`` directory inside the sweep output.

        Returns:
            List of :class:`SweepImageFile` entries.
        """
        files: List[SweepImageFile] = []

        for pipeline_dir in sorted(results_dir.iterdir()):
            if not pipeline_dir.is_dir():
                continue
            pipeline_name = pipeline_dir.name
            pipe_file_count = 0
            pipe_comp_count = 0

            for component_dir in sorted(pipeline_dir.iterdir()):
                if not component_dir.is_dir():
                    continue
                component = component_dir.name
                if component == "measurements":
                    continue  # CSV data, not images

                comp_file_count = 0
                for img_path in sorted(component_dir.iterdir()):
                    if (
                        img_path.is_file()
                        and img_path.suffix.lower() in _IMAGE_EXTENSIONS
                    ):
                        files.append(
                            SweepImageFile(
                                path=img_path,
                                image_stem=img_path.stem,
                                component=component,
                                pipeline_name=pipeline_name,
                            )
                        )
                        comp_file_count += 1

                if comp_file_count:
                    pipe_comp_count += 1
                    pipe_file_count += comp_file_count

            logger.debug(
                "Scanned pipeline %r: %d files across"
                " %d components",
                pipeline_name, pipe_file_count,
                pipe_comp_count,
            )

        return files
