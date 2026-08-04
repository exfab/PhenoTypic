"""Generate the two-page Measurements reference from the public schema.

At ``builder-inited`` time, this extension discovers every public
``phenotypic.schema.MeasurementInfo`` class and writes two deterministic pages:
Measurements and Metadata. Each canonical class contributes one linked section
heading and its measurement table. It also copies packaged measurement images
into the docs static tree so ``/_static/measurements/...`` references resolve.

The pages are regenerated on every build, so new public schema classes surface
automatically. Do not edit generated ``measurements_ref/*.rst`` files by hand;
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
    """Render one linked class heading followed by its measurement table."""
    class_name = info_cls.__name__
    api_doc = f"/api_reference/api/phenotypic.schema.{class_name}"
    linked_heading = f":doc:`{class_name} <{api_doc}>`"
    out = [
        f".. _{_section_label(info_cls)}:",
        "",
        *_heading(linked_heading, "-"),
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
    metadata_child: bool = False,
) -> str:
    """Build one table-only reference page."""
    out = _heading(title, "=")
    if metadata_child:
        out.extend(
            [
                ".. toctree::",
                "   :hidden:",
                "",
                "   metadata/index",
                "",
            ]
        )
    for info_cls in info_classes:
        out.append(_class_section(info_cls))
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
    metadata_infos = tuple(
        info_cls
        for info_cls in public_infos
        if info_cls.category().startswith("Metadata")
    )
    measurement_infos = tuple(
        info_cls for info_cls in public_infos if info_cls not in metadata_infos
    )
    _write(
        output_dir / "index.rst",
        _build_reference_page(
            "Measurements", measurement_infos, metadata_child=True
        ),
    )
    _write(
        output_dir / "metadata" / "index.rst",
        _build_reference_page("Metadata", metadata_infos),
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
