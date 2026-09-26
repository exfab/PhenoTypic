"""Generate the Measurements reference pages from the public schema.

At ``config-inited`` time, before Sphinx discovers source files, this extension
discovers every public
``phenotypic.schema.MeasurementInfo`` class and writes three deterministic pages:
Measurements, Metadata, and Categories (a subpage of Measurements, listing each
``CATEGORIES`` member's ``desc`` and columns). Each canonical class contributes
one linked section heading and its measurement table. It also copies packaged
measurement images
into the docs static tree so ``/_static/measurements/...`` references resolve.

The pages are regenerated on every build, so new public schema classes surface
automatically. Do not edit generated ``measurements_ref/**/index.rst`` files by hand;
edit the source enums or this extension instead.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

def _heading(title: str, underline: str) -> list[str]:
    """Return an RST heading block."""
    return [title, underline * len(title), ""]


def _public_measurement_info_classes() -> tuple[type[Any], ...]:
    """Return canonical public ``MeasurementInfo`` classes in export order."""
    import phenotypic.schema as schema

    info_base = schema.MeasurementInfo
    infos: list[type[Any]] = []
    seen: set[type[Any]] = set()
    for name in schema.__all__:
        if name == "MeasurementInfo":
            continue
        value = getattr(schema, name, None)
        if (
            isinstance(value, type)
            and issubclass(value, info_base)
            and bool(getattr(value, "__members__", None))
            and value not in seen
        ):
            infos.append(value)
            seen.add(value)
    return tuple(infos)


def _section_label(info_cls: type[Any]) -> str:
    """Return the stable cross-reference label for one class section."""
    slug = info_cls.__name__.lower().replace("_", "-")
    return f"measurement-info-{slug}"


def _class_section(info_cls: type[Any]) -> str:
    """Render one linked class heading, its change note if any, and its table."""
    class_name = info_cls.__name__
    api_doc = f"/api_reference/api/phenotypic.schema.{class_name}"
    linked_heading = f":doc:`{class_name} <{api_doc}>`"
    note = info_cls.change_note()
    out = [
        f".. _{_section_label(info_cls)}:",
        "",
        *_heading(linked_heading, "-"),
        *([note, ""] if note else []),
        info_cls.rst_table(
            header=("Column label", "Description"), use_headers=True
        ),
        "",
    ]
    return "\n".join(out)


def _write(path: Path, contents: str) -> None:
    """Write a generated page, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


def _build_reference_page(
    title: str,
    info_classes: tuple[type[Any], ...],
    *,
    child_pages: tuple[str, ...] = (),
) -> str:
    """Build one table-only reference page."""
    out = _heading(title, "=")
    if child_pages:
        out.extend([".. toctree::", "   :hidden:", ""])
        out.extend(f"   ../{child}/index" for child in child_pages)
        out.append("")
    for info_cls in info_classes:
        out.append(_class_section(info_cls))
    return "\n".join(out)


_CATEGORIES_INTRO = (
    "Categories are curated groupings of measurement columns drawn from "
    "several metric families; one column can belong to several categories. "
    "Unlike the Type badge, a category makes no claim about how far a single "
    "value can be trusted (see :ref:`measurement-categories`). Every run that "
    "measures objects writes one spreadsheet per category that has at least "
    "one column in the run, under "
    "``deliverables/measurements_by_category/``, holding the shared context "
    "columns (metadata, object label, grid) plus that category's columns."
)


def _category_section(category: Any) -> str:
    """Render one category: anchor, heading, verbatim desc, output file, column table."""
    out = [
        f".. _{category.anchor}:",
        "",
        *_heading(category.display_name, "-"),
        category.desc,
        "",
        f"Written to ``deliverables/measurements_by_category/{category.label}.csv`` "
        "(and ``.parquet``).",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - Column",
        "     - Metric family",
        "     - Type",
    ]
    members = category.members()
    if not members:
        # A header-only list-table is a docutils ERROR; say so in prose instead.
        return "\n".join([*out[:out.index(".. list-table::")], "No columns carry this category yet.", ""])
    for member in members:
        info_cls = type(member)
        out.extend(
            [
                f"   * - ``{member.value}``",
                f"     - :ref:`{info_cls.metric_family()} <{_section_label(info_cls)}>`",
                f"     - {member.use_badge}",
            ]
        )
    out.append("")
    return "\n".join(out)


def _build_categories_page() -> str:
    """Build the generated Categories page, one section per CATEGORIES member."""
    from phenotypic.schema import CATEGORIES

    out = [*_heading("Categories", "="), _CATEGORIES_INTRO, ""]
    out.extend(_category_section(category) for category in CATEGORIES)
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
    import phenotypic.schema as schema

    output_dir = Path(srcdir) / "measurements_ref"
    if output_dir.exists():
        shutil.rmtree(output_dir)

    public_infos = _public_measurement_info_classes()
    metadata_infos = tuple(
        info_cls
        for info_cls in public_infos
        if issubclass(info_cls, schema.MetadataInfo)
    )
    measurement_infos = tuple(
        info_cls for info_cls in public_infos if info_cls not in metadata_infos
    )
    _write(
        output_dir / "measurements" / "index.rst",
        _build_reference_page(
            "Measurements",
            measurement_infos,
            child_pages=("metadata", "categories"),
        ),
    )
    _write(
        output_dir / "metadata" / "index.rst",
        _build_reference_page("Metadata", metadata_infos),
    )
    _write(output_dir / "categories" / "index.rst", _build_categories_page())


def _generate(app, _config):
    _copy_measurement_assets(app.srcdir)
    _build_pages(app.srcdir)
    print(f"Generated {os.path.join(app.srcdir, 'measurements_ref')}")


def setup(app):
    app.connect("config-inited", _generate)
    return {
        "version": "0.5",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
