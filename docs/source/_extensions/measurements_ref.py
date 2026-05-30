"""Sphinx extension that generates the Measurements Reference page.

At ``builder-inited`` time, walks a registry of per-object ``MeasureFeature``
classes paired with their ``MeasurementInfo`` enum(s) and writes
``docs/source/measurements_ref/index.rst`` with one section per operator —
the operator's lead docstring followed by a table of every label/description
written to ``measurements.parquet``.

This uses the same generate-at-build-time pattern as ``generate_downloadables_rst``
in ``conf.py``; emitting a plain ``.rst`` (rather than a custom directive)
guarantees Sphinx treats the section titles as real headings so the right-rail
TOC, cross-references, and the parent toctree all work without special handling.

The page is regenerated on every build, so renames or new measurements added to
a ``MeasurementInfo`` subclass surface immediately — no manual maintenance.
"""

from __future__ import annotations

import importlib
import inspect
import os

from sphinx.util import logging as sphinx_logging

logger = sphinx_logging.getLogger(__name__)


# (MeasureFeature qualified path, [MeasurementInfo qualified paths])
# Per-object measurements only — QC, model metrics, and edge correction are
# documented elsewhere. Order here drives section order on the rendered page.
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
            "phenotypic.schema.ColorXYZ",
            "phenotypic.schema.Colorxy",
            "phenotypic.schema.ColorLab",
            "phenotypic.schema.ColorHSV",
        ]),
    # MeasureColorComposition is commented out of ``phenotypic.measure.__all__``
    # pending completion (see the TODO in ``measure/__init__.py``). Re-add it
    # to the registry when it ships.
    ("phenotypic.measure._measure_grid_spatial.MeasureGridSpatial",
        ["phenotypic.schema.GRID_SPATIAL"]),
    ("phenotypic.measure._measure_grid_linreg_stats.MeasureGridLinRegStats",
        ["phenotypic.schema.GRID_LINREG_STATS"]),
    ("phenotypic.measure._measure_grid_spread.MeasureGridSpread",
        ["phenotypic.schema.GRID_SPREAD"]),
    ("phenotypic.measure._measure_symmetric_zones.MeasureSymmetricZones",
        ["phenotypic.schema.SYMMETRIC_ZONES"]),
]


_PAGE_INTRO = """\
Measurements Reference
======================

Every column produced by PhenoTypic's per-object measurement operators,
grouped by operator. If you've received a ``measurements.parquet`` from
someone else and need to know what a column means, this is the page for
you.

Each section below shows the operator's lead description followed by a
table of every column it emits, with the short name (what appears in
the parquet, prefixed with the category — e.g. ``Size_Area``) and a
one-line description.

This page is generated from the ``MeasurementInfo`` enums in
``phenotypic.schema`` and stays in sync with the code
automatically — do not edit ``measurements_ref/index.rst`` by hand; edit
the docstrings or the registry in ``docs/source/_extensions/measurements_ref.py``.

"""


def _import(path: str):
    """Import ``module.attr`` and return the attribute."""
    module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _strip_appended_table(doc: str) -> str:
    """Drop the auto-appended ``MeasurementInfo`` table from a docstring.

    Each ``MeasureFeature`` module ends with
    ``MeasureX.__doc__ = INFO.append_rst_to_doc(MeasureX)``, which glues
    ``\\n\\n.. list-table::`` onto the bottom of the docstring at indent 0.
    Strip the table *before* running ``inspect.cleandoc`` — otherwise the
    flush-left table pins cleandoc's common margin to 0 and disables its
    dedent on the still-indented continuation paragraphs above.
    """
    if not doc:
        return ""
    marker = "\n\n.. list-table::"
    idx = doc.find(marker)
    trimmed = doc if idx == -1 else doc[:idx]
    return inspect.cleandoc(trimmed)


def _lead_paragraphs(doc: str, max_paragraphs: int = 3) -> str:
    """Return the first few paragraphs of a docstring.

    Stops at the first Google-style section header (``Args:``, ``Returns:``,
    ``Raises:``, ``Example:``, ``Notes:``, ``Attributes:``, etc.). This keeps
    the reference page focused on the *what* of each operator rather than
    its parameter table — that detail lives on the API page.
    """
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


def _build_page(output_path: str) -> None:
    """Assemble the full RST page and write it to ``output_path``."""
    out: list[str] = [_PAGE_INTRO]

    # ``Object_Label`` is shared across every operator rather than owned by one
    # ``MeasureFeature``, so it has no registry entry — emit it as a leading
    # section above the per-operator tables.
    try:
        object_info = _import("phenotypic.schema.OBJECT")
    except (ImportError, AttributeError) as err:
        logger.warning(
            "measurements_ref: could not import phenotypic.schema.OBJECT: %s", err
        )
    else:
        object_heading = "Object identifier"
        out.append(object_heading)
        out.append("-" * len(object_heading))
        out.append("")
        out.append(
            "``Object_Label`` is the shared per-object key — the first column of "
            "every operator's table below. Each detected colony is assigned a "
            "unique integer label so all of its measurements line up across "
            "operators when joined on this column."
        )
        out.append("")
        out.append(object_info.rst_table())
        out.append("")
        out.append("")

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
        out.append(heading)
        out.append("-" * len(heading))
        out.append("")

        description = _lead_paragraphs(_strip_appended_table(measure_cls.__doc__ or ""))
        if description:
            out.append(description)
            out.append("")

        for info_cls in info_classes:
            # Sub-heading per MeasurementInfo when an operator emits several
            # (e.g. MeasureColor → XYZ / xy / Lab / HSV).
            if len(info_classes) > 1:
                sub = info_cls.category()
                out.append(sub)
                out.append("^" * len(sub))
                out.append("")
            out.append(info_cls.rst_table())
            out.append("")
        out.append("")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))


def _generate(app):
    output_file = os.path.join(app.srcdir, "measurements_ref", "index.rst")
    _build_page(output_file)
    print(f"Generated {output_file}")


def setup(app):
    app.connect("builder-inited", _generate)
    return {
        "version": "0.2",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
