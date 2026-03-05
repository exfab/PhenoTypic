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
        readme_path = output_dir / "README.md"

        sections = [
            self._generate_header(),
            self._generate_output_structure(datasets),
            self._generate_layers_section(),
            self._generate_measurements_section(),
            self._generate_footer(),
        ]

        content = "\n\n".join(filter(None, sections))
        readme_path.write_text(content)

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
+-- results/                      # All dataset results
{dataset_list}
|       +-- measurements/         # Per-image CSV measurement files
|       +-- overlays/             # Detection overlay visualizations
+-- dashboard.html                # Live processing dashboard
+-- master_measurements.csv       # Aggregated measurements (all datasets)
+-- processing_state.json         # Resume/state tracking
+-- processing_report.html        # HTML summary report
+-- logs/                         # Execution logs
|   +-- slurm/                    # SLURM job logs (if applicable)
+-- README.md                     # This file
```"""

    def _generate_layers_section(self) -> str:
        """Generate documentation for saved layers."""
        layer_descriptions = {
            "rgb": "Original RGB images",
            "gray": "Grayscale images",
            "detect_mat": "Detection matrix (after preprocessing)",
            "objmask": "Binary object masks",
            "objmap": "Labeled object maps (integer labels per object)",
            "objmap_overlay": "Colorized object map overlays",
            "detect_mat_overlay": "Detection matrix with detection overlay",
            "objmask_overlay": "Object mask with detection overlay",
        }

        layers = []
        if self.config.save_rgb:
            layers.append(("rgb/", layer_descriptions["rgb"]))
        if self.config.save_gray:
            layers.append(("gray/", layer_descriptions["gray"]))
        if self.config.save_detect_mat:
            layers.append(("detect_mat/", layer_descriptions["detect_mat"]))
        if self.config.save_objmask:
            layers.append(("objmask/", layer_descriptions["objmask"]))
        if self.config.save_objmap:
            layers.append(("objmap/", layer_descriptions["objmap"]))
        if self.config.save_objmap_overlay:
            layers.append(("objmap_overlay/", layer_descriptions["objmap_overlay"]))
        if self.config.save_detect_mat_overlay:
            layers.append(("detect_mat_overlay/", layer_descriptions["detect_mat_overlay"]))
        if self.config.save_objmask_overlay:
            layers.append(("objmask_overlay/", layer_descriptions["objmask_overlay"]))

        if not layers:
            return """## Saved Layers

Only standard outputs (measurements, overlays) were saved.
Use `--save-*` CLI flags to save additional image layers."""

        layer_table = "| Layer | Description |\n|-------|-------------|\n"
        for layer_dir, desc in layers:
            layer_table += f"| `{layer_dir}` | {desc} |\n"

        return f"""## Saved Layers

The following optional layers were saved:

{layer_table}"""

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

        for meas_name, measurer in self.pipeline._meas.items():
            if not isinstance(measurer, MeasureFeatures):
                continue

            # Get measurement info class if defined on the measurer
            measurement_infos = self._get_measurement_info_classes(measurer)

            if not measurement_infos:
                sections.append(f"\n### {meas_name}\n\n*No measurement documentation available.*")
                continue

            for info_cls in measurement_infos:
                table = self._generate_measurement_table(info_cls)
                if table:
                    sections.append(table)

        return "\n".join(sections)

    def _get_measurement_info_classes(self, measurer) -> list:
        """Extract MeasurementInfo classes associated with a MeasureFeatures instance.

        Looks for class attributes that are MeasurementInfo subclasses or
        references to measurement info in the measurer's implementation.
        """
        from phenotypic.tools_.measurement_info_ import (
            SHAPE,
            INTENSITY,
            TEXTURE,
            ColorXYZ,
            ColorLab,
            ColorHSV,
            Colorxy,
            ColorComposition,
            SIZE,
            BBOX,
            GRID_SPREAD,
            GRID_LINREG_STATS,
        )

        # Map measurer class names to their MeasurementInfo classes
        measurer_to_info = {
            "MeasureShape": [SHAPE],
            "MeasureIntensity": [INTENSITY],
            "MeasureTexture": [TEXTURE],
            "MeasureColor": [ColorXYZ, ColorLab, ColorHSV, Colorxy],
            "MeasureColorComposition": [ColorComposition],
            "MeasureSize": [SIZE],
            "MeasureBounds": [BBOX],
            "MeasureGridLinRegStats": [GRID_LINREG_STATS],
            "MeasureGridSpread": [GRID_SPREAD],
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
