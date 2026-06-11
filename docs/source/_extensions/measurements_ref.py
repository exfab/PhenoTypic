"""Sphinx extension that generates the Measurements Reference page tree.

At ``builder-inited`` time, this extension writes ``docs/source/measurements_ref``
as a deterministic build artifact. The generated section has a compact landing
page, separate Measurements and Metadata indexes, and one child page per public
``phenotypic.schema.MeasurementInfo`` subclass so every schema enum participates
in Sphinx navigation.

The pages are regenerated on every build, so renames or new measurements added
to ``phenotypic.schema`` surface immediately. Do not edit generated
``measurements_ref/*.rst`` files by hand; edit the source enums or this
extension instead.
"""

from __future__ import annotations

import importlib
import inspect
import os
import re
import shutil
from pathlib import Path
from typing import Any

from sphinx.util import logging as sphinx_logging

logger = sphinx_logging.getLogger(__name__)


# (MeasureFeature qualified path, [MeasurementInfo qualified paths])
# Per-object measurements only. Order here drives the operator-oriented
# section order on the rendered Measurements page.
_REGISTRY: list[tuple[str, list[str]]] = [
    ("phenotypic.measure._measure_size.MeasureSize",
        ["phenotypic.schema.SIZE"]),
    ("phenotypic.measure._measure_shape.MeasureShape",
        ["phenotypic.schema.SHAPE"]),
    ("phenotypic.measure._measure_intensity.MeasureIntensity",
        ["phenotypic.schema.INTENSITY"]),
    ("phenotypic.measure._measure_bounds.MeasureBounds",
        ["phenotypic.schema.BBOX"]),
    ("phenotypic.measure._measure_texture.MeasureTexture",
        ["phenotypic.schema.TEXTURE"]),
    ("phenotypic.measure._measure_color.MeasureColor",
        [
            "phenotypic.schema.ColorLab",
            "phenotypic.schema.ColorHSV",
        ]),
    # MeasureColorComposition is commented out of ``phenotypic.measure.__all__``
    # pending completion (see the TODO in ``measure/__init__.py``). Its enum is
    # still documented on the generated schema page.
    ("phenotypic.measure._measure_grid_spatial.MeasureGridSpatial",
        ["phenotypic.schema.GRID_SPATIAL"]),
    ("phenotypic.measure._measure_grid_linreg_stats.MeasureGridLinRegStats",
        ["phenotypic.schema.GRID_LINREG_STATS"]),
    ("phenotypic.measure._measure_grid_spread.MeasureGridSpread",
        ["phenotypic.schema.GRID_SPREAD"]),
    ("phenotypic.measure._measure_symmetric_zones.MeasureSymmetricZones",
        ["phenotypic.schema.SYMMETRIC_ZONES"]),
]

_METADATA_INFO_NAMES: set[str] = {
    "METADATA",
    "ACQUISITION_METADATA",
    "CONDITION_METADATA",
    "EXPERIMENT_METADATA",
    "GENETIC_METADATA",
    "INCUBATION_METADATA",
    "PLATE_METADATA",
    "SAMPLE_METADATA",
}

_EXPERIMENTAL_TAG_NAMES: tuple[str, ...] = (
    "ACQUISITION_METADATA",
    "CONDITION_METADATA",
    "EXPERIMENT_METADATA",
    "GENETIC_METADATA",
    "INCUBATION_METADATA",
    "PLATE_METADATA",
    "SAMPLE_METADATA",
)

_METADATA_OVERVIEWS: dict[str, tuple[str, str]] = {
    "METADATA": (
        "Framework-populated image bookkeeping, including image names, UUIDs, "
        "file formats, image types, bit depth, and file suffixes.",
        "Use when reading provenance emitted by PhenoTypic itself; these are "
        "not the biological metadata columns users normally supply.",
    ),
    "SAMPLE_METADATA": (
        "Sample identity and provenance, including sample IDs, replicates, "
        "clones, source plate/well, library IDs, barcodes, and controls.",
        "Use for sample-level biological identity and for linking colonies "
        "back to source materials.",
    ),
    "PLATE_METADATA": (
        "Assay plate and physical layout, including plate IDs, batches, array "
        "density, and incubator position.",
        "Use when grouping measurements by plate, batch, or spatial assay "
        "layout.",
    ),
    "CONDITION_METADATA": (
        "Media, nutrients, supplements, treatments, compounds, doses, and "
        "stress conditions applied to colonies.",
        "Use when comparing phenotypes across growth environments or "
        "perturbations.",
    ),
    "INCUBATION_METADATA": (
        "Temperature, elapsed time, time units, timepoints, day indices, "
        "generation, humidity, and atmosphere.",
        "Use for time-course analyses and incubation-condition grouping.",
    ),
    "ACQUISITION_METADATA": (
        "Image acquisition details, including imaging date, instrument, "
        "experimenter, resolution, and exposure time.",
        "Use when tracking imaging batches or diagnosing acquisition effects.",
    ),
    "GENETIC_METADATA": (
        "Organism and genetic identity, including species, strain, genotype, "
        "background, alleles, plasmids, markers, mating type, and ploidy.",
        "Use when grouping or filtering colonies by genetic background.",
    ),
    "EXPERIMENT_METADATA": (
        "Experiment-level bookkeeping, including experiment IDs, projects, "
        "datasets, protocols, and notes.",
        "Use when organizing outputs across projects, protocols, or datasets.",
    ),
}

_RST_ROLE_RE = re.compile(r":[a-zA-Z0-9_.:]+:`([^`]+)`")

_ROOT_INTRO = """\
Measurements
============

PhenoTypic uses ``MeasurementInfo`` enums to define stable column names for
measurement outputs and metadata joins. Use this section to look up the columns
that appear in exported DataFrames and to reuse the same names in downstream
analysis code.

.. toctree::
   :maxdepth: 2
   :hidden:

   measurements/index
   metadata/index

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Measurements

      Per-object measurements, analysis outputs, model metrics, quality-control
      labels, and other non-metadata columns.

      +++

      .. button-ref:: measurements/index
         :ref-type: doc
         :click-parent:
         :color: secondary
         :expand:

         Browse measurements

   .. grid-item-card:: Metadata

      Framework metadata and recommended experimental ``Metadata_*`` tags for
      sample, plate, condition, incubation, acquisition, genetic, and experiment
      annotations.

      +++

      .. button-ref:: metadata/index
         :ref-type: doc
         :click-parent:
         :color: secondary
         :expand:

         Browse metadata

"""

_MEASUREMENTS_INTRO = """\
Measurements
============

Every non-metadata ``MeasurementInfo`` enum exported by ``phenotypic.schema``.
The generated pages below document each enum's full DataFrame column labels and
descriptions.

.. toctree::
   :maxdepth: 1
   :caption: Measurement Categories
   :hidden:

{toctree_entries}

Operator-Oriented Overview
--------------------------

The sections below retain the original operator grouping for per-object
measurement operators. Each operator description is followed by the schema
tables it emits.

"""

_METADATA_INTRO = """\
Metadata
========

Use the ``Metadata_*`` column labels documented here when preparing external
metadata tables for PhenoTypic. These labels streamline downstream processing
because the package offers processing helpers based on these assumptions. If
your input tables use different column names, provide a mapping before feeding
them into PhenoTypic workflows.

.. toctree::
   :maxdepth: 1
   :caption: MetadataInfo
   :hidden:

{toctree_entries}

Metadata Tag Overview
---------------------

.. list-table::
   :header-rows: 1

   * - Tag class
     - Includes
     - Use for
{metadata_overview_rows}

Framework Metadata
------------------

``METADATA`` covers framework-populated image bookkeeping columns.

Experimental Tags
-----------------

The experimental-tag enums live under ``phenotypic.schema._experimental_tags``
and provide a recommended vocabulary for biological and experimental
annotations. They are recommended labels, not validators: arbitrary metadata
columns are still accepted, but using the labels below keeps downstream
processing simpler.

{experimental_tag_list}

"""


def _import(path: str) -> Any:
    """Import ``module.attr`` and return the attribute."""
    module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _strip_appended_table(doc: str) -> str:
    """Drop the auto-appended ``MeasurementInfo`` table from a docstring."""
    if not doc:
        return ""
    marker = "\n\n.. list-table::"
    idx = doc.find(marker)
    trimmed = doc if idx == -1 else doc[:idx]
    return inspect.cleandoc(trimmed)


def _lead_paragraphs(doc: str, max_paragraphs: int = 3) -> str:
    """Return the first few paragraphs of a docstring."""
    if not doc:
        return ""
    sections = (
        "Args:", "Arguments:", "Returns:", "Yields:", "Raises:",
        "Example:", "Examples:", "Note:", "Notes:", "Attributes:",
        "Warning:", "See Also:", "References:", "Best For:",
        "Consider Also:",
    )
    paragraphs: list[str] = []
    current: list[str] = []
    for line in doc.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(s) for s in sections):
            break
        if not stripped:
            if current:
                paragraphs.append("\n".join(current))
                current = []
                if len(paragraphs) >= max_paragraphs:
                    break
        else:
            current.append(line)
    if current and len(paragraphs) < max_paragraphs:
        paragraphs.append("\n".join(current))
    return "\n\n".join(paragraphs).strip()


def _doc_stem(info_name: str) -> str:
    """Return the generated filename stem for a ``MeasurementInfo`` class."""
    return info_name.lower()


def _heading(title: str, underline: str) -> list[str]:
    """Return an RST heading block."""
    return [title, underline * len(title), ""]


def _toctree_entries(
        info_names: list[str],
        public_infos: dict[str, type[Any]] | None = None,
        *,
        label_by_category: bool = False,
) -> str:
    """Format hidden toctree entries for generated enum pages."""
    entries: list[str] = []
    for name in info_names:
        if label_by_category and public_infos is not None:
            label = public_infos[name].category()
            entries.append(f"   {label} <{_doc_stem(name)}>")
        else:
            entries.append(f"   {_doc_stem(name)}")
    return "\n".join(entries)


def _experimental_tag_list() -> str:
    """Return a bullet list of experimental-tag enum doc links."""
    return "\n".join(
        f"- :doc:`{name} <{_doc_stem(name)}>`"
        for name in _EXPERIMENTAL_TAG_NAMES
    )


def _metadata_overview_rows(info_names: list[str]) -> str:
    """Return overview table rows for the metadata index."""
    rows: list[str] = []
    for name in info_names:
        includes, use_for = _METADATA_OVERVIEWS[name]
        rows.extend([
            f"   * - :doc:`{name} <{_doc_stem(name)}>`",
            f"     - {includes}",
            f"     - {use_for}",
        ])
    return "\n".join(rows)


def _public_measurement_info_classes() -> dict[str, type[Any]]:
    """Return public ``MeasurementInfo`` subclasses exported by schema."""
    import phenotypic.schema as schema

    info_base = schema.MeasurementInfo
    infos: dict[str, type[Any]] = {}
    for name in schema.__all__:
        if name == "MeasurementInfo":
            continue
        value = getattr(schema, name, None)
        if isinstance(value, type) and issubclass(value, info_base):
            infos[name] = value
    return infos


def _full_column_table(info_cls: type[Any]) -> str:
    """Render a table of full DataFrame column labels for an info enum."""
    lines = [
        f".. list-table:: Category: **{info_cls.category()}**",
        "   :header-rows: 1",
        "",
        "   * - Column label",
        "     - Description",
    ]
    for member in info_cls:
        lines += [
            f"   * - ``{member.value}``",
            f"     - {_rst_cell_text(member.desc)}",
        ]
    return "\n".join(lines)


def _rst_cell_text(text: str) -> str:
    """Escape text that would otherwise be parsed as RST markup."""
    normalized = _RST_ROLE_RE.sub(lambda match: f"``{match.group(1)}``", text)
    return normalized.replace("|", r"\|")


def _enum_page(info_cls: type[Any]) -> str:
    """Build a standalone page for one ``MeasurementInfo`` enum."""
    title = info_cls.__name__
    out: list[str] = _heading(title, "=")
    out.append(f"Python object: ``{info_cls.__module__}.{title}``")
    out.append("")

    description = _lead_paragraphs(_strip_appended_table(info_cls.__doc__ or ""))
    if description:
        out.append(_rst_cell_text(description))
        out.append("")

    out.append(_full_column_table(info_cls))
    out.append("")
    return "\n".join(out)


def _append_object_identifier_section(out: list[str]) -> None:
    """Append the shared object-label explanation to the measurements page."""
    try:
        object_info = _import("phenotypic.schema.OBJECT")
    except (ImportError, AttributeError) as err:
        logger.warning(
            "measurements_ref: could not import phenotypic.schema.OBJECT: %s", err
        )
        return

    out.extend(_heading("Object Identifier", "^"))
    out.append(
        "``Object_Label`` is the shared per-object key used by every "
        "per-object measurement operator. Each detected colony is assigned a "
        "unique integer label so its measurements line up across operators "
        "when joined on this column."
    )
    out.append("")
    out.append(object_info.rst_table())
    out.append("")
    out.append("")


def _append_operator_sections(out: list[str]) -> None:
    """Append the existing operator-oriented measurement overview."""
    _append_object_identifier_section(out)

    for measure_path, info_paths in _REGISTRY:
        try:
            measure_cls = _import(measure_path)
        except (ImportError, AttributeError) as err:
            logger.warning(
                "measurements_ref: could not import %s: %s", measure_path, err
            )
            continue

        info_classes = []
        for info_path in info_paths:
            try:
                info_classes.append(_import(info_path))
            except (ImportError, AttributeError) as err:
                logger.warning(
                    "measurements_ref: could not import %s: %s", info_path, err
                )
        if not info_classes:
            continue

        heading = measure_cls.__name__
        out.extend(_heading(heading, "^"))

        description = _lead_paragraphs(_strip_appended_table(measure_cls.__doc__ or ""))
        if description:
            out.append(description)
            out.append("")

        for info_cls in info_classes:
            if len(info_classes) > 1:
                sub = info_cls.category()
                out.extend(_heading(sub, '"'))
            out.append(info_cls.rst_table())
            out.append("")
        out.append("")


def _append_metadata_sections(
        out: list[str],
        info_names: list[str],
        public_infos: dict[str, type[Any]],
) -> None:
    """Append inline metadata class documentation to the metadata index."""
    for name in info_names:
        info_cls = public_infos[name]
        out.extend(_heading(name, "-"))

        description = _lead_paragraphs(_strip_appended_table(info_cls.__doc__ or ""))
        if description:
            out.append(_rst_cell_text(description))
            out.append("")

        out.append(info_cls.rst_table())
        out.append("")
        out.append("")


def _write(path: Path, contents: str) -> None:
    """Write a generated page, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _build_measurements_index(
        info_names: list[str],
        public_infos: dict[str, type[Any]],
) -> str:
    """Build the non-metadata measurements index page."""
    out = [
        _MEASUREMENTS_INTRO.format(
            toctree_entries=_toctree_entries(
                info_names,
                public_infos,
                label_by_category=True,
            )
        )
    ]
    _append_operator_sections(out)
    return "\n".join(out)


def _build_metadata_index(
        info_names: list[str],
        public_infos: dict[str, type[Any]],
) -> str:
    """Build the metadata index page."""
    out = [
        _METADATA_INTRO.format(
            toctree_entries=_toctree_entries(info_names),
            metadata_overview_rows=_metadata_overview_rows(info_names),
            experimental_tag_list=_experimental_tag_list(),
        )
    ]
    _append_metadata_sections(out, info_names, public_infos)
    return "\n".join(out)


def _build_pages(srcdir: str) -> None:
    """Generate the complete measurements reference page tree under ``srcdir``."""
    output_dir = Path(srcdir) / "measurements_ref"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    public_infos = _public_measurement_info_classes()
    metadata_names = [
        name for name in public_infos
        if name in _METADATA_INFO_NAMES
    ]
    measurement_names = [
        name for name in public_infos
        if name not in _METADATA_INFO_NAMES
    ]

    _write(output_dir / "index.rst", _ROOT_INTRO)
    _write(
        output_dir / "measurements" / "index.rst",
        _build_measurements_index(measurement_names, public_infos),
    )
    _write(
        output_dir / "metadata" / "index.rst",
        _build_metadata_index(metadata_names, public_infos),
    )

    for name in measurement_names:
        _write(
            output_dir / "measurements" / f"{_doc_stem(name)}.rst",
            _enum_page(public_infos[name]),
        )
    for name in metadata_names:
        _write(
            output_dir / "metadata" / f"{_doc_stem(name)}.rst",
            _enum_page(public_infos[name]),
        )


def _generate(app):
    _build_pages(app.srcdir)
    print(f"Generated {os.path.join(app.srcdir, 'measurements_ref')}")


def setup(app):
    app.connect("builder-inited", _generate)
    return {
        "version": "0.3",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
