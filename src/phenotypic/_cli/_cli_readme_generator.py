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

from phenotypic.sdk_ import readme_md_path

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
            self._generate_model_section(),
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
            dataset_list += (
                f"\n|   +-- ... ({len(datasets) - 5} more datasets)"
            )

        return f"""## Output Structure

```
output_folder/
+-- deliverables/                     # User-facing run outputs
|   +-- master_measurements.csv       # Aggregated measurements (all datasets)
|   +-- measurements.csv              # Editable copy used by the GUI results viewer (refreshed on every run)
|   +-- measurements.parquet          # Parquet companion of measurements.csv
|   +-- pipeline.json                 # Reproducibility spec (operations + filters + model)
|   +-- <AnalysisClass>.csv           # Class-named model-fit output
|   +-- <AnalysisClass>.parquet       # Parquet companion
|   +-- analysis_manifest.json        # Named-analysis artifact index and checksums
|   +-- plots/                         # Configured plot outputs and page manifests
|   +-- dashboard.html                # Live processing dashboard
|   +-- processing_report.html        # HTML summary report
|   +-- overlays/                     # Detection overlay PNGs (per-dataset subfolders, one per input image)
|   +-- README.md                     # This file
+-- results/                          # All dataset results
{dataset_list}
|       +-- zarr/                     # Processed images as one OME-Zarr store per input (<stem>.ome.zarr: layers + objmap label image + metadata + grid state)
|       +-- measurements/             # Per-image Parquet measurement files
+-- .phenotypic/                      # Hidden machine-state cache
|   +-- processing_state.json         # Continuation/state tracking
|   +-- processing_events.log         # Append-only event log
|   +-- logs/                         # Execution logs
|   |   +-- slurm/                    # SLURM job logs (if applicable)
|   +-- slurm_scripts/                # Generated SLURM scripts (if applicable)
```"""

    def _generate_layers_section(self) -> str:
        """Generate documentation for saved layers."""
        return """## Saved Layers

Each dataset directory contains the following folders:

| Folder | Format | Description |
|--------|--------|-------------|
| `zarr/` | OME-Zarr (NGFF 0.5) | Processed image (layers + `objmap` label image + metadata + grid state) saved as a single `<stem>.ome.zarr` store directory per input image, reloadable via `Image.load_zarr` / `GridImage.load_zarr`. |
| `measurements/` | Parquet | Per-object measurements. |

Each `.ome.zarr` store is a standard OME-NGFF image: **napari, QuPath, and
Vizarr open it directly, with no PhenoTypic install** — drag the store
directory in, or point the viewer at its path. It carries a resolution
pyramid, so a viewer reads only the levels and tiles it needs instead of
decoding the whole plate.

Detection overlay PNGs are written per input image under
`deliverables/overlays/<dataset>/` (always written for forward runs).

Use the Results Viewer and its `/analysis/` workspace for interactive result
exploration."""

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
                sections.append(
                    f"\n### {meas_name}\n\n*No measurement documentation available.*"
                )
                continue

            for info_cls in measurement_infos:
                table = self._generate_measurement_table(info_cls)
                if table:
                    sections.append(table)

        return "\n".join(sections)

    def _get_measurement_infoclasses(self, measurer) -> list[type]:
        """Extract MeasurementInfo classes associated with a MeasureFeatures instance.

        Uses the operation-level schema contract so built-in and custom
        measurers are documented without a second class-name registry.
        """
        return list(measurer.get_measurement_infoclasses())

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

    def _generate_model_section(self) -> str:
        """Document the configured analysis model's metric-qualified columns.

        Renders only when the pipeline has a ``model`` configured. Column
        headers embed the fitted metric (``model.on``) so the README matches
        the actual class-named analysis CSV produced by this run.
        """
        from phenotypic.schema import MODEL_METRICS, qualified_header
        from phenotypic.util._measurement_outputs import metric_token

        model = self.pipeline.get_model()
        if model is None:
            return ""
        info_cls = getattr(model, "_measurement_infoclass", None)
        if info_cls is None:
            return ""

        token = metric_token(str(model.on))
        model_name = type(model).__name__

        lines = [
            "## Models & Analysis",
            "",
            f"Model `{model_name}` fit on metric `{model.on}` "
            f"(output written to `deliverables/{model_name}.csv`).",
            "",
            "Output columns follow `<Model>_<metric>_<parameter>`; for this "
            f"run `<metric>` = `{token}`.",
            "",
            "| Column | Description |",
            "|--------|-------------|",
        ]
        for member in list(info_cls) + list(MODEL_METRICS):
            header = qualified_header(member, token)
            desc = (member.desc or "").replace("|", "\\|").replace("\n", " ")
            if len(desc) > 200:
                desc = desc[:197] + "..."
            lines.append(f"| `{header}` | {desc} |")
        return "\n".join(lines)

    def _generate_footer(self) -> str:
        """Generate README footer."""
        import phenotypic

        return f"""---

## About

Generated by PhenoTypic v{phenotypic.__version__}

For more information, visit: https://github.com/exfab/PhenoTypic"""
