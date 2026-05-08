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
from typing import Dict, List, Optional, Tuple, get_args

from phenotypic.gui._config import ChannelName, DIR_MEASUREMENTS, DIR_RESULTS

logger = logging.getLogger(__name__)

_HDF5_EXTENSIONS = {".h5", ".hdf5"}

_CHANNELS = get_args(ChannelName)  # ("rgb", "gray", "detect_mat", "objmap")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SweepHDF5File:
    """A single HDF5 result file in the sweep output."""

    path: Path
    image_stem: str
    pipeline_name: str


@dataclass(frozen=True)
class IntermediateStep:
    """An intermediate HDF5 file saved after a single pipeline operation."""

    index: int              # 0, 1, 2, ...
    operation_name: str     # "GaussianBlur"
    h5_path: Path           # absolute path to HDF5
    layers: tuple[str, ...] = ()    # which datasets this file contains
    is_base: bool = False           # whether this is a base snapshot


@dataclass(frozen=True)
class ResolvedLayerSources:
    """For a given step, maps each layer name to its source HDF5 path."""

    rgb: Path | None = None
    gray: Path | None = None
    detect_mat: Path | None = None
    objmap: Path | None = None


def build_layer_resolution_index(
    steps: list[IntermediateStep],
) -> dict[int, ResolvedLayerSources]:
    """Build a mapping from step index to resolved layer sources.

    For each step, determines which HDF5 file contains the most recent
    version of each layer by scanning forward through the step list.

    Args:
        steps: Sorted list of intermediate steps.

    Returns:
        Dict mapping step index to :class:`ResolvedLayerSources`.
    """
    if not steps:
        return {}

    # Running "latest source" for each layer
    latest: dict[str, Path | None] = {name: None for name in _CHANNELS}
    index: dict[int, ResolvedLayerSources] = {}

    for step in steps:
        for layer_name in step.layers:
            if layer_name in latest:
                latest[layer_name] = step.h5_path
        index[step.index] = ResolvedLayerSources(**latest)

    return index


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
    hdf5_files: List[SweepHDF5File]
    pipeline_names: List[str]
    image_stems: List[str]
    # [pipeline][stem] -> SweepHDF5File
    by_pipeline: Dict[str, Dict[str, SweepHDF5File]] = field(
        default_factory=dict,
    )
    # [stem][pipeline] -> SweepHDF5File
    by_image: Dict[str, Dict[str, SweepHDF5File]] = field(
        default_factory=dict,
    )
    # [stem][pipeline] -> sorted list of IntermediateStep
    intermediates: Dict[str, Dict[str, List[IntermediateStep]]] = field(
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

        manifest_raw, pipeline_configs = (
            SweepOutputScanner._parse_manifest(manifest_path)
        )

        results_dir = sweep_dir / DIR_RESULTS
        if results_dir.is_dir():
            hdf5_files = SweepOutputScanner._scan_results(results_dir)
        else:
            logger.warning(
                "Results directory not found: %s", results_dir,
            )
            hdf5_files = []

        # Derive sorted unique lists
        pipeline_names = sorted(
            {f.pipeline_name for f in hdf5_files},
        )
        image_stems = sorted({f.image_stem for f in hdf5_files})

        # Build lookup indexes
        by_pipeline: Dict[str, Dict[str, SweepHDF5File]] = {}
        by_image: Dict[str, Dict[str, SweepHDF5File]] = {}

        for f in hdf5_files:
            by_pipeline.setdefault(
                f.pipeline_name, {},
            )[f.image_stem] = f

            by_image.setdefault(
                f.image_stem, {},
            )[f.pipeline_name] = f

        # Scan for intermediate HDF5 files
        if results_dir.is_dir():
            intermediates = SweepOutputScanner._scan_intermediates(
                results_dir,
            )
        else:
            intermediates = {}

        logger.debug(
            "Scan complete: %d HDF5 files, %d pipelines, %d stems",
            len(hdf5_files), len(pipeline_names),
            len(image_stems),
        )

        return SweepOutputData(
            root_dir=sweep_dir,
            manifest_raw=manifest_raw,
            pipeline_configs=pipeline_configs,
            hdf5_files=hdf5_files,
            pipeline_names=pipeline_names,
            image_stems=image_stems,
            by_pipeline=by_pipeline,
            by_image=by_image,
            intermediates=intermediates,
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
        target = (
            Path(path).resolve() if path else Path.cwd().resolve()
        )
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
        """Parse the sweep manifest JSON into PipelineConfig objects.

        Args:
            manifest_path: Path to ``sweep_manifest.json``.

        Returns:
            Tuple of (raw manifest dict, pipeline configs dict keyed by
            pipeline name).
        """
        manifest_raw = json.loads(manifest_path.read_text())
        configs: Dict[str, PipelineConfig] = {}

        for cfg_name, cfg_data in manifest_raw.get(
            "configs", {},
        ).items():
            for pipe_name, pipe_dict in cfg_data.get(
                "pipelines", {},
            ).items():
                # Extract operations from pipe_cfgs
                operations: List[Dict] = []
                pipe_cfgs = pipe_dict.get("pipe_cfgs", {})
                for op_key in pipe_cfgs:
                    op_data = pipe_cfgs[op_key]
                    operations.append(
                        {
                            "name": op_key,
                            "class": op_data.get(
                                "class", "Unknown",
                            ),
                            "params": op_data.get("params", {}),
                        }
                    )

                # Extract measurement operations
                measurements: List[Dict] = []
                meas_cfgs = pipe_dict.get("meas_cfgs", {})
                for meas_key in meas_cfgs:
                    meas_data = meas_cfgs[meas_key]
                    measurements.append(
                        {
                            "name": meas_key,
                            "class": meas_data.get(
                                "class", "Unknown",
                            ),
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
    def _scan_results(
        results_dir: Path,
    ) -> List[SweepHDF5File]:
        """Walk ``results/<image_stem>/<pipeline>/*.h5`` for HDF5 files.

        Args:
            results_dir: The ``results/`` directory inside the sweep
                output.

        Returns:
            List of :class:`SweepHDF5File` entries.
        """
        files: List[SweepHDF5File] = []

        for stem_dir in sorted(results_dir.iterdir()):
            if not stem_dir.is_dir():
                continue
            image_stem = stem_dir.name
            stem_file_count = 0

            for pipe_dir in sorted(stem_dir.iterdir()):
                if not pipe_dir.is_dir():
                    continue
                pipeline_name = pipe_dir.name

                for h5_path in sorted(pipe_dir.iterdir()):
                    if (
                        h5_path.is_file()
                        and h5_path.suffix.lower()
                        in _HDF5_EXTENSIONS
                    ):
                        files.append(
                            SweepHDF5File(
                                path=h5_path,
                                image_stem=image_stem,
                                pipeline_name=pipeline_name,
                            )
                        )
                        stem_file_count += 1

            if stem_file_count:
                logger.debug(
                    "Scanned stem %r: %d HDF5 files",
                    image_stem, stem_file_count,
                )

        return files

    @staticmethod
    def _scan_intermediates(
        results_dir: Path,
    ) -> Dict[str, Dict[str, List[IntermediateStep]]]:
        """Scan for intermediate HDF5 files in ``intermediates/`` subdirectories.

        Expected structure::

            results/<image_stem>/<pipeline>/intermediates/00_OpName.h5
            results/<image_stem>/<pipeline>/intermediates/base_00.h5

        Args:
            results_dir: The ``results/`` directory inside the sweep output.

        Returns:
            Nested dict: ``intermediates[image_stem][pipeline_name]`` -> sorted
            list of :class:`IntermediateStep`.
        """
        import h5py

        _LAYER_KEYS = set(get_args(ChannelName))
        intermediates: Dict[str, Dict[str, List[IntermediateStep]]] = {}

        for stem_dir in sorted(results_dir.iterdir()):
            if not stem_dir.is_dir():
                continue
            image_stem = stem_dir.name

            for pipe_dir in sorted(stem_dir.iterdir()):
                if not pipe_dir.is_dir():
                    continue
                pipeline_name = pipe_dir.name

                inter_dir = pipe_dir / "intermediates"
                if not inter_dir.is_dir():
                    continue

                steps: List[IntermediateStep] = []
                for h5_path in sorted(inter_dir.iterdir()):
                    if not h5_path.is_file():
                        continue
                    if h5_path.suffix.lower() not in _HDF5_EXTENSIONS:
                        continue

                    name_no_ext = h5_path.stem
                    is_base = False

                    # Parse base files: "base_00.h5"
                    if name_no_ext.startswith("base_"):
                        try:
                            idx = int(name_no_ext[5:])
                        except ValueError:
                            logger.warning(
                                "Skipping unparseable base intermediate: %s",
                                h5_path.name,
                            )
                            continue
                        op_name = "base"
                        is_base = True
                    else:
                        # Parse delta files: "00_GaussianBlur.h5"
                        parts = name_no_ext.split("_", 1)
                        if len(parts) != 2:
                            logger.warning(
                                "Skipping unparseable intermediate: %s",
                                h5_path.name,
                            )
                            continue
                        try:
                            idx = int(parts[0])
                        except ValueError:
                            logger.warning(
                                "Skipping intermediate with non-integer"
                                " index: %s",
                                h5_path.name,
                            )
                            continue
                        op_name = parts[1]

                    # Probe HDF5 for layer datasets
                    try:
                        with h5py.File(h5_path, "r") as f:
                            layers = tuple(
                                k for k in _LAYER_KEYS if k in f
                            )
                    except Exception:
                        logger.warning(
                            "Could not probe HDF5 file: %s",
                            h5_path.name,
                        )
                        layers = ()

                    steps.append(IntermediateStep(
                        index=idx,
                        operation_name=op_name,
                        h5_path=h5_path,
                        layers=layers,
                        is_base=is_base,
                    ))

                if steps:
                    steps.sort(key=lambda s: s.index)
                    intermediates.setdefault(
                        image_stem, {},
                    )[pipeline_name] = steps
                    logger.debug(
                        "Found %d intermediates for %s/%s",
                        len(steps), image_stem, pipeline_name,
                    )

        return intermediates
