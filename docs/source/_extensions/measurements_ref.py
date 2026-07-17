"""Sphinx extension that generates the Measurements Reference page tree.

At ``builder-inited`` time, this extension writes ``docs/source/measurements_ref``
as a deterministic build artifact. The generated section has a compact landing
page with one card per captioned group, a per-group index, and one child page
per public ``phenotypic.schema.MeasurementInfo`` subclass so every schema enum
participates in Sphinx navigation. It also copies packaged measurement images
into the docs static tree so ``/_static/measurements/...`` references resolve.

The pages are regenerated on every build, so renames or new measurements added
to ``phenotypic.schema`` surface immediately. Do not edit generated
``measurements_ref/*.rst`` files by hand; edit the source enums or this
extension instead.
"""

from __future__ import annotations

import inspect
import os
import shutil
from pathlib import Path
from typing import Any

from phenotypic.schema._measurement_info import _rst_cell_text

#: Toctree groups (caption -> ordered enum names). Single source of truth; a test
#: asserts every public schema enum lands in exactly one group.
_GROUPS: dict[str, tuple[str, ...]] = {
    "Measurements": (
        "SIZE",
        "SHAPE",
        "BBOX",
        "INTENSITY",
        "TEXTURE",
        "ColorLab",
        "ColorHSV",
        "Colorxy",
        "ColorXYZ",
        "ColorComposition",
        "OBJECT",
        "GRID",
        "NEIGHBOR_DIST",
        "GRID_LINREG_STATS",
        "GRID_SPREAD",
        "SYMMETRIC_ZONES",
        "ORIENTATION_ZONE_PRIMARY",
        "RADIAL_EXPANSION",
    ),
    "Models & Analysis": (
        "LOG_GROWTH_MODEL",
        "LINEAR_LAG_MODEL",
        "LINEAR_CAP_AND_LAG_MODEL",
        "EDGE_CORRECTION",
        "MODEL_METRICS",
    ),
    "Quality Control": (
        "QUALITY_CHECK",
        "QUALITY_COUNT",
        "QUALITY_OCCUPANCY",
        "QUALITY_ICC",
        "QUALITY_MAD",
        "QUALITY_SE",
        "QUALITY_TUKEY",
        "QUALITY_ZMAX",
        "METADATA_MATCH",
        "ORIENTATION_ZONE_DIAGNOSTIC",
    ),
    "Curation & Errors": ("CURATION", "ErrorCategory"),
    "Compatibility": ("ORIENTATION_ZONES",),
    "Metadata": (
        "METADATA",
        "ACQUISITION_METADATA",
        "CONDITION_METADATA",
        "CULTURE_METADATA",
        "EXPERIMENT_METADATA",
        "GENETIC_METADATA",
        "PLATE_METADATA",
        "SAMPLE_METADATA",
        "STUDY_METADATA",
    ),
}

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
    "CULTURE_METADATA": (
        "Temperature, elapsed time, time units, timepoints, day indices, "
        "generation, humidity, and atmosphere.",
        "Use for time-course analyses and culture-condition grouping.",
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
    "STUDY_METADATA": (
        "Study-level descriptors for one run: title, description, keywords, "
        "authors, license, funding, publications, links, and acknowledgements.",
        "Use when recording REMBI Study-component provenance for a dataset.",
    ),
}


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
        "Args:",
        "Arguments:",
        "Returns:",
        "Yields:",
        "Raises:",
        "Example:",
        "Examples:",
        "Note:",
        "Notes:",
        "Attributes:",
        "Warning:",
        "See Also:",
        "References:",
        "Best For:",
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


def _metadata_overview_rows(info_names: list[str]) -> str:
    """Return overview table rows for the metadata index."""
    rows: list[str] = []
    for name in info_names:
        includes, use_for = _METADATA_OVERVIEWS[name]
        rows.extend(
            [
                f"   * - :doc:`{name} <{_doc_stem(name)}>`",
                f"     - {includes}",
                f"     - {use_for}",
            ]
        )
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


def _enum_page(info_cls: type[Any]) -> str:
    """Build a standalone page for one ``MeasurementInfo`` enum."""
    title = info_cls.__name__
    out: list[str] = _heading(title, "=")
    out.append(f"Python object: ``{info_cls.__module__}.{title}``")
    out.append("")

    description = _lead_paragraphs(
        _strip_appended_table(info_cls.__doc__ or "")
    )
    if description:
        out.append(_rst_cell_text(description))
        out.append("")

    out.append(
        info_cls.rst_table(
            header=("Column label", "Description"), use_headers=True
        )
    )
    out.append("")
    return "\n".join(out)


def _write(path: Path, contents: str) -> None:
    """Write a generated page, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _group_slug(caption: str) -> str:
    return caption.lower().replace(" & ", "-and-").replace(" ", "-")


def _build_root_index(groups, public_infos) -> str:
    toc, cards = [], []
    for caption, names in groups.items():
        if not any(n in public_infos for n in names):
            continue
        slug = _group_slug(caption)
        toc.append(f"   {slug}/index")
        cards += [
            f"   .. grid-item-card:: {caption}",
            "",
            "      +++",
            "",
            f"      .. button-ref:: {slug}/index",
            "         :ref-type: doc",
            "         :click-parent:",
            "         :color: secondary",
            "         :expand:",
            "",
            f"         Browse {caption}",
            "",
        ]
    out = [
        "Measurements",
        "============",
        "",
        "PhenoTypic uses ``MeasurementInfo`` enums to define stable column names",
        "for measurement outputs and metadata joins. Browse by group:",
        "",
        ".. toctree::",
        "   :maxdepth: 2",
        "   :hidden:",
        "",
        *toc,
        "",
        ".. grid:: 1 2 2 2",
        "   :gutter: 3",
        "",
        *cards,
    ]
    return "\n".join(out)


def _metadata_overview_block(names) -> str:
    present = [n for n in names if n in _METADATA_OVERVIEWS]
    if not present:
        return ""
    return "\n".join(
        [
            "Metadata Tag Overview",
            "---------------------",
            "",
            ".. list-table::",
            "   :header-rows: 1",
            "",
            "   * - Tag class",
            "     - Includes",
            "     - Use for",
            _metadata_overview_rows(present),
            "",
        ]
    )


def _build_group_index(caption, names, public_infos) -> str:
    out = [
        caption,
        "=" * len(caption),
        "",
        f"Schema enums in the **{caption}** group. Each page documents an enum's",
        "DataFrame column labels and descriptions.",
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        f"   :caption: {caption}",
        "",
    ]
    # Label toctree entries by category when those are unique within the group
    # (e.g. Measurements: "Size", "Shape"); fall back to the enum name when a
    # group shares one category across members (Metadata and Quality Control
    # both collapse to a single category()), so the entries stay distinct.
    categories = [public_infos[name].category() for name in names]
    unique_categories = len(set(categories)) == len(categories)
    for name in names:
        label = public_infos[name].category() if unique_categories else name
        out.append(f"   {label} <{_doc_stem(name)}>")
    out.append("")
    block = _metadata_overview_block(names) if caption == "Metadata" else ""
    if block:
        out += ["", block]
    return "\n".join(out)


def _copy_measurement_assets(srcdir: str) -> None:
    """Copy packaged measurement images into the docs static tree so that
    ``/_static/measurements/...`` references resolve in the built HTML."""
    import phenotypic

    src = (
        Path(phenotypic.__file__).resolve().parent / "_assets" / "measurements"
    )
    if not src.is_dir():
        return
    dest = Path(srcdir) / "_static" / "measurements"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def _build_pages(srcdir: str) -> None:
    output_dir = Path(srcdir) / "measurements_ref"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    public_infos = _public_measurement_info_classes()
    _write(output_dir / "index.rst", _build_root_index(_GROUPS, public_infos))
    for caption, names in _GROUPS.items():
        present = [n for n in names if n in public_infos]
        if not present:
            continue
        slug = _group_slug(caption)
        _write(
            output_dir / slug / "index.rst",
            _build_group_index(caption, present, public_infos),
        )
        for name in present:
            _write(
                output_dir / slug / f"{_doc_stem(name)}.rst",
                _enum_page(public_infos[name]),
            )


def _generate(app):
    _copy_measurement_assets(app.srcdir)
    _build_pages(app.srcdir)
    print(f"Generated {os.path.join(app.srcdir, 'measurements_ref')}")


def setup(app):
    app.connect("builder-inited", _generate)
    return {
        "version": "0.3",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
