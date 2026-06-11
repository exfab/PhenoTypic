"""
README generation for PhenoTypic CLI output directories.

This module generates a comprehensive README.md file explaining the output
structure and documenting all measurements produced by the pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List

from phenotypic.tools_ import readme_md_path

if TYPE_CHECKING:
    from phenotypic._core._image_pipeline import ImagePipeline
    from phenotypic._cli._cli_types import ExecutionConfig, Dataset

logger = logging.getLogger(__name__)


class READMEGenerator:
    """Generates README.md documentation for CLI output directories.

    Creates a markdown file explaining output structure and documenting
    all measurements from the pipeline's MeasureFeatures operations.
    """

    def __init__(self, config: ExecutionConfig, pipeline: ImagePipeline):
        """Initialize README generator.

        Args:
            config: CLI execution configuration.
            pipeline: The ImagePipeline used for processing.
        """
        self.config = config
        self.pipeline = pipeline

    def generate(self, output_dir: Path, datasets: List[Dataset]) -> Path:
        """Generate README.md file in the output directory.

        Args:
            output_dir: Base output directory.
            datasets: List of processed datasets.

        Returns:
            Path to generated README.md file.
        """
        readme_path = readme_md_path(output_dir)

        sections = [
            self._generate_header(),
            self._generate_output_structure(datasets),
            self._generate_layers_section(),
            self._generate_measurements_section(),
            self._generate_footer(),
        ]

        content = "\n\n".join(filter(None, sections))
        # The README writer does not go through the atomic writer, so the
        # deliverables/ directory may not exist yet — create it explicitly.
        readme_path.parent.mkdir(parents=True, exist_ok=True)
        readme_path.write_text(content, encoding="utf-8")

        return readme_path

    def _generate_header(self) -> str:
        """Generate README header section."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        grid_info = ""
        if self.config.image_type == "GridImage":
            grid_info = f"\nGrid Dimensions: {self.config.nrows} rows x {self.config.ncols} columns"

        return f"""# PhenoTypic Processing Results

Generated: {timestamp}

Pipeline: `{self.config.pipeline_json.name}`

Image Type: {self.config.image_type}{grid_info}"""

    def _generate_output_structure(self, datasets: List[Dataset]) -> str:
        """Generate output structure documentation."""
        dataset_list = "\n".join(f"|   +-- {d.name}/" for d in datasets[:5])
        if len(datasets) > 5:
            dataset_list += f"\n|   +-- ... ({len(datasets) - 5} more datasets)"

        return f"""## Output Structure

```
output_folder/
+-- deliverables/                     # User-facing run outputs
|   +-- master_measurements.csv       # Aggregated measurements (all datasets)
|   +-- measurements.csv              # Editable copy used by the GUI results viewer (refreshed on every run)
|   +-- measurements.parquet          # Parquet companion of measurements.csv
|   +-- pipeline.json                 # Reproducibility spec (operations + filters + model)
|   +-- analysis.csv                  # Model-fit output (only when pipeline has a `model` configured)
|   +-- analysis.parquet              # Parquet companion of analysis.csv
|   +-- dashboard.html                # Live processing dashboard
|   +-- analysis.html                 # Analysis & visualization
|   +-- processing_report.html        # HTML summary report
|   +-- README.md                     # This file
+-- results/                          # All dataset results
{dataset_list}
|       +-- hdf/                      # Processed images as single .h5 per input (layers + metadata + grid state)
|       +-- measurements/             # Per-image Parquet measurement files
|       +-- overlays/                 # Detection overlay PNGs (one per input image)
+-- processing_state.json             # Resume/state tracking
+-- logs/                             # Execution logs
|   +-- slurm/                        # SLURM job logs (if applicable)
```"""

    def _generate_layers_section(self) -> str:
        """Generate documentation for saved layers."""
        return """## Saved Layers

Each dataset directory contains the following folders:

| Folder | Format | Description |
|--------|--------|-------------|
| `hdf/` | HDF5 | Processed image (layers + metadata + grid state) saved as a single `.h5` per input image, reloadable via `Image.load_hdf5` / `GridImage.load_hdf5`. |
| `measurements/` | Parquet | Per-object measurements. |
| `overlays/` | PNG | Detection overlay visualizations (always written for forward runs). |"""

    def _generate_measurements_section(self) -> str:
        """Generate measurement documentation from pipeline's MeasureFeatures.

        Iterates through pipeline._meas to find MeasureFeatures instances
        and generates tables for each using MeasurementInfo metadata.
        """
        from phenotypic.abc_ import MeasureFeatures

        if not self.pipeline._meas:
            return """## Measurements

No measurements configured in this pipeline."""

        sections = [
            "## Measurements",
            "",
            "The following measurements are extracted for each detected object:",
        ]

        from phenotypic.schema import OBJECT
        from phenotypic.schema import BBOX

        sections.append(
            "\n### Object\n\n"
            "| Column | Description |\n"
            "|--------|-------------|\n"
            f"| `{OBJECT.LABEL}` "
            "| Unique integer label identifying each detected object. |"
        )
        bbox_table = self._generate_measurement_table(BBOX)
        if bbox_table:
            sections.append(bbox_table)

        if self.config.image_type == "GridImage":
            from phenotypic.schema import GRID

            grid_table = self._generate_measurement_table(GRID)
            if grid_table:
                sections.append(grid_table)

        for meas_name, measurer in self.pipeline._meas.items():
            if not isinstance(measurer, MeasureFeatures):
                continue

            # Get measurement info class if defined on the measurer
            measurement_infos = self._get_measurement_infoclasses(measurer)

            if not measurement_infos:
                sections.append(f"\n### {meas_name}\n\n*No measurement documentation available.*")
                continue

            for info_cls in measurement_infos:
                table = self._generate_measurement_table(info_cls)
                if table:
                    sections.append(table)

        return "\n".join(sections)

    def _get_measurement_infoclasses(self, measurer) -> list:
        """Extract MeasurementInfo classes associated with a MeasureFeatures instance.

        Looks for class attributes that are MeasurementInfo subclasses or
        references to measurement info in the measurer's implementation.
        """
        from phenotypic.schema import (
            SHAPE,
            INTENSITY,
            TEXTURE,
            ColorLab,
            ColorHSV,
            ColorComposition,
            SIZE,
            BBOX,
            GRID_SPREAD,
            GRID_LINREG_STATS,
            GRID_SPATIAL,
        )

        # Map measurer class names to their MeasurementInfo classes
        measurer_to_info = {
            "MeasureShape": [SHAPE],
            "MeasureIntensity": [INTENSITY],
            "MeasureTexture": [TEXTURE],
            "MeasureColor": [ColorLab, ColorHSV],
            "MeasureColorComposition": [ColorComposition],
            "MeasureSize": [SIZE],
            "MeasureBounds": [BBOX],
            "MeasureGridLinRegStats": [GRID_LINREG_STATS],
            "MeasureGridSpread": [GRID_SPREAD],
            "MeasureGridSpatial": [GRID_SPATIAL],
        }

        class_name = measurer.__class__.__name__
        return measurer_to_info.get(class_name, [])

    def _generate_measurement_table(self, info_cls) -> str:
        """Generate markdown table for a MeasurementInfo class."""
        try:
            category = info_cls.category()
            members = list(info_cls)

            if not members:
                return ""

            table = f"\n### {category}\n\n"
            table += "| Column | Description |\n"
            table += "|--------|-------------|\n"

            for member in members:
                # Use the full header name (category_label)
                col_name = str(member)
                desc = member.desc if hasattr(member, "desc") else ""
                # Escape pipe characters in descriptions
                desc = desc.replace("|", "\\|").replace("\n", " ")
                # Truncate very long descriptions
                if len(desc) > 200:
                    desc = desc[:197] + "..."
                table += f"| `{col_name}` | {desc} |\n"

            return table
        except Exception as e:
            logger.warning(f"Could not generate table for {info_cls}: {e}")
            return ""

    def _generate_footer(self) -> str:
        """Generate README footer."""
        import phenotypic

        return f"""---

## About

Generated by PhenoTypic v{phenotypic.__version__}

For more information, visit: https://github.com/exfab/PhenoTypic"""
