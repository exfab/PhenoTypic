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

from phenotypic.sdk_ import DIR_MEASUREMENTS_BY_CATEGORY, readme_md_path

if TYPE_CHECKING:
    from phenotypic._core._image_pipeline import ImagePipeline
    from phenotypic._cli._cli_types import ExecutionConfig, Dataset
    from phenotypic.schema import MeasurementInfo

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
            self._generate_categories_section(),
            self._generate_embeddings_section(),
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
        if getattr(self.config, "measure_only", False):
            # --mode measure takes the grid from each image store, not from
            # flags or the pipeline.
            grid_info = "\nGrid Dimensions: as stored in each image store"
        elif self.config.image_type == "GridImage":
            from phenotypic._cli._cli_utils import resolve_grid_shape

            nrows, ncols = resolve_grid_shape(
                getattr(self.config, "nrows", None),
                getattr(self.config, "ncols", None),
                getattr(self.pipeline, "nrows", None),
                getattr(self.pipeline, "ncols", None),
            )
            grid_info = f"\nGrid Dimensions: {nrows} rows x {ncols} columns"

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
|   +-- master_measurements.parquet   # Clean archive: every measured object, un-joined (Parquet only)
|   +-- measurements.tsv              # Metadata-joined mirror, tab-separated (refreshed on every run)
|   +-- measurements.csv              # Same mirror, comma-separated: the file the GUI results viewer reads and curates (kept until #283)
|   +-- measurements.parquet          # Parquet companion of the mirror
|   +-- measurements_by_feature/      # One TSV + Parquet per measurer: shared context columns + that measurer's columns
|   +-- measurements_by_category/     # One TSV + Parquet per measurement category (see Measurement Categories)
|   +-- metrics.json                  # Run summary: image/object/phantom counts, PhenoTypic version, pipeline sha256
|   +-- schema.json                   # Every column of each published table: dtype, unit, measurer, schema member, description
|   +-- pipeline.pht-pipe.json        # Reproducibility spec (operations + filters + model); seeded from --pipeline at start
|   +-- <AnalysisClass>.tsv           # Class-named model-fit output
|   +-- <AnalysisClass>.parquet       # Parquet companion
|   +-- analysis_manifest.json        # Named-analysis artifact index and checksums
|   +-- plots/                        # Configured plot outputs and page manifests
|   +-- dashboard.html                # Live processing dashboard
|   +-- overlays/                     # Detection overlay PNGs (per-dataset subfolders, one per input image)
|   +-- README.md                     # This file
+-- results/                          # All dataset results
{dataset_list}
|       +-- zarr/                     # Processed images as one OME-Zarr store per input (<stem>.ome.zarr: layers + objmap label image + grid state, plus the authoritative object measurements at tables/<n>-<sha12>/measurements/table.parquet, one folder per measuring pipeline)
|       +-- measurements/             # Legacy per-image Parquets — only in a tree written by an older release, never created by this run
+-- .phenotypic/                      # Hidden machine-state cache
|   +-- processing_state.json         # Continuation/state tracking
|   +-- processing_events.log         # Append-only event log
|   +-- logs/                         # Execution logs
|   |   +-- slurm/                    # SLURM job logs (if applicable)
|   +-- slurm_scripts/                # Generated SLURM scripts (if applicable)
```

**Generated tables are tab-separated (`.tsv`).** Releases before this change
wrote `.csv` for the per-feature and per-category splits and for
`<AnalysisClass>`; re-running finalize into such a folder replaces those with
`.tsv` and deletes the old `.csv` files it replaces. Two CSVs stay: `metadata.csv`
(or `metadata.tsv`; the byte-for-byte copy of the table you passed to
`--metadata`) and `error_analysis.csv` (written by the results viewer).
`measurements.csv` is kept because the results viewer's curation writes to it
and to `measurements.parquet`, **not** to `measurements.tsv`: until #283 lands,
curated edits reach `measurements.csv` and `measurements.parquet` but not
`measurements.tsv`. `--metadata` must now be a `.csv` or `.tsv` table (the
delimiter follows the suffix); any other suffix, such as a comma-separated
`.txt` that earlier releases read, is refused, so rename it.

**`metrics.json` and `schema.json` are for automation** (for example
`dvc metrics show`). `metrics.json` counts the images the run accepted,
succeeded on and failed terminally, the objects measured and the
metadata-only phantom rows, and names the PhenoTypic version and the sha256 of
the pipeline every aggregated table came from (`null` when they differ or it
is not recorded). `schema.json` lists, for every published table (keyed by
its path under `deliverables/` without the suffix), each column's Parquet
dtype, unit, producing measurer, schema member and description. Both are
rewritten each time a full, measure or recompile run publishes its tables
(`--mode migrate` does not write them), and they hold no timestamps, so
identical runs give byte-identical files."""

    def _generate_layers_section(self) -> str:
        """Generate documentation for saved layers."""
        return """## Saved Layers

Each dataset directory contains the following:

| Path | Format | Description |
|------|--------|-------------|
| `zarr/` | OME-Zarr (NGFF 0.5) | Processed image (layers + `objmap` label image + metadata + grid state) saved as a single `<stem>.ome.zarr` store directory per input image, reloadable via `Image.load_zarr` / `GridImage.load_zarr`. |
| `zarr/<stem>.ome.zarr/tables/<n>-<sha12>/measurements/table.parquet` | Parquet | Authoritative per-object measurements for that image, embedded in its store: one `<n>-<sha12>/` folder per pipeline that measured the store, numbered in the order the pipelines ran and named by the first 12 hex of the pipeline file's sha256 (`<n>-unknown` when no digest is known). Each folder is described by its entry in the store's root `attributes.phenotypic.tables.runs`. Re-running a pipeline replaces its own folder; another pipeline adds a folder beside it. This run writes no separate per-image `measurements/` folder. |

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
        from phenotypic.abc_ import MeasureEmbedding, MeasureFeatures

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
            # An embedder publishes no mirror column; its tables are
            # documented in their own section (#281 spec E5).
            if not isinstance(measurer, MeasureFeatures) or isinstance(
                measurer, MeasureEmbedding
            ):
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

    def _generate_categories_section(self) -> str:
        """Document each category the configured measurers emit columns for.

        Only columns this pipeline's measurers can produce are listed, so a
        category whose members all come from an unconfigured measurer is
        omitted. Returns ``""`` when no category applies.
        """
        from phenotypic.abc_ import MeasureEmbedding, MeasureFeatures
        from phenotypic.schema import CATEGORIES

        infos = [
            info
            for measurer in (self.pipeline._meas or {}).values()
            if isinstance(measurer, MeasureFeatures)
            and not isinstance(measurer, MeasureEmbedding)
            for info in self._get_measurement_infoclasses(measurer)
        ]
        # One malformed info must cost only its own columns, not the README,
        # mirroring the per-info guard in _generate_measurement_table.
        tagged: list[tuple[str, frozenset[CATEGORIES]]] = []
        for info in infos:
            try:
                tagged.extend(
                    [
                        (str(member), frozenset(getattr(member, "categories", ())))
                        for member in info
                    ]
                )
            except Exception as e:
                logger.warning(f"Could not read categories for {info}: {e}")
        blocks: list[str] = []
        for category in CATEGORIES:
            columns = [
                column for column, categories in tagged if category in categories
            ]
            if not columns:
                continue
            listed = "\n".join(f"- `{column}`" for column in dict.fromkeys(columns))
            blocks.append(
                f"### {category.display_name}\n\n{category.desc}\n\n"
                f"Written to `deliverables/{DIR_MEASUREMENTS_BY_CATEGORY}/{category.label}.tsv` "
                f"(and `.parquet`), alongside every context column.\n\n{listed}"
            )
        if not blocks:
            return ""
        return "## Measurement Categories\n\n" + "\n\n".join(blocks)

    def _generate_embeddings_section(self) -> str:
        """Document the embedding deliverables, one table per embedder (#281 spec E5).

        An embedder writes no column of the measurement tables, so the
        measurements section leaves it out; its vectors are published as
        ``deliverables/embeddings/<model>/<layer>/<region>.parquet``. Returns
        ``""`` when the pipeline has no embedder. Reads only the embedder's
        fields, never its weights.
        """
        from phenotypic.abc_ import MeasureEmbedding
        from phenotypic.sdk_ import DIR_DELIVERABLES, DIR_EMBEDDINGS, EMBEDDINGS_MANIFEST_JSON
        from phenotypic.util._measurement_outputs import embedding_vector_member

        embedders = [
            (name, measurer)
            for name, measurer in (self.pipeline._meas or {}).items()
            if isinstance(measurer, MeasureEmbedding)
        ]
        if not embedders:
            return ""
        root = f"{DIR_DELIVERABLES}/{DIR_EMBEDDINGS}"
        lines = [
            "## Embeddings",
            "",
            "Each embedder writes one Parquet file of feature vectors, never "
            "columns of the measurement tables. A row carries "
            "`Metadata_Dataset`, `Metadata_ImageName`, the table's key "
            "(`Object_Label` for colonies, the `Grid_*` cell keys for a "
            "grid) and `embedding`, a fixed-length float16 vector. A file is "
            "`<model>/<layer>/<region>.parquet`: the layer is `rgb` or `gray`, "
            "the pixels embedded, and the region is `objects` or `grid`, or "
            "`objects-<name>` / `grid-<name>` for an embedder given a `name`, "
            "so two grids of one model (the wells and the whole agar, say), or "
            "one grid on rgb and on gray, are two files. Join a "
            "colony row to `master_measurements.parquet` on `Metadata_Dataset`, "
            "`Metadata_ImageName` and `Object_Label`. These files are never "
            "joined to metadata. "
            f"`{root}/{EMBEDDINGS_MANIFEST_JSON}` records each file's path, "
            "row count, sha256, store count and model descriptors.",
            "",
            "Each row also carries `descriptor_index` (int16), its 0-based "
            "position in that file's `descriptors` list in the manifest: the "
            "model id and revision, vector length, pooling and parameters "
            "that produced the vector. It is 0 on every row when all stores "
            "share one descriptor. Stores can differ when `revision` is left "
            "unpinned (each process loads the hub's latest weights), or when "
            "part of the run was re-measured with other settings; "
            "finalization then logs a warning. Compare vectors only within "
            "one `descriptor_index`. The device is not part of the "
            "descriptor: a CPU and a GPU embedding differ by about 1e-6 in "
            "cosine distance, so a run that used both is one descriptor; each "
            "image's store records the device (and GPU name) that ran it.",
            "",
            "| Embedder | File |",
            "|----------|------|",
        ]
        descs: dict[str, str] = {}
        for name, embedder in embedders:
            key = embedder.payload_key()
            lines.append(f"| `{name}` ({type(embedder).__name__}) | `{root}/{key}.parquet` |")
            member = embedding_vector_member(type(embedder).__name__)
            if member is not None and member.desc:
                descs.setdefault(type(embedder).__name__, member.desc)
        for producer, desc in descs.items():
            text = desc.replace("|", "\\|").replace("\n", " ")
            lines += ["", f"`embedding` ({producer}): {text}"]
        return "\n".join(lines)

    def _get_measurement_infoclasses(self, measurer) -> list[type[MeasurementInfo]]:
        """Extract MeasurementInfo classes associated with a MeasureFeatures instance.

        Uses the operation-level schema contract so built-in and custom
        measurers are documented without a second class-name registry.
        """
        return list(measurer.get_measurement_infoclasses())

    def _generate_measurement_table(self, info_cls) -> str:
        """Generate markdown table for a MeasurementInfo class.

        A Categories column is added only when some member of *info_cls*
        carries a category.
        """
        from phenotypic.schema import CATEGORIES

        try:
            family = info_cls.metric_family()
            members = list(info_cls)

            if not members:
                return ""

            has_categories = any(member.categories for member in members)
            table = f"\n### {family}\n\n"
            if has_categories:
                table += "| Column | Description | Categories |\n"
                table += "|--------|-------------|------------|\n"
            else:
                table += "| Column | Description |\n"
                table += "|--------|-------------|\n"

            for member in members:
                # Use the full header name (family_label)
                col_name = str(member)
                desc = member.desc if hasattr(member, "desc") else ""
                # Escape pipe characters in descriptions
                desc = desc.replace("|", "\\|").replace("\n", " ")
                # Truncate very long descriptions
                if len(desc) > 200:
                    desc = desc[:197] + "..."
                row = f"| `{col_name}` | {desc} |"
                if has_categories:
                    names = ", ".join(
                        category.display_name
                        for category in CATEGORIES.in_order(member.categories)
                    )
                    row += f" {names} |"
                table += row + "\n"

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
            f"(output written to `deliverables/{model_name}.tsv`).",
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

Documentation: https://exfab.github.io/PhenoTypic/ (source:
https://github.com/exfab/PhenoTypic)

This table was written by PhenoTypic v{phenotypic.__version__}. In the docs' version menu, choose that version, or the nearest earlier one listed."""
