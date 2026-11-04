"""Generate the Measurements reference from the schema and its table producers.

At ``config-inited`` time, before Sphinx discovers source files, this extension
writes the Measurements section of the docs::

    measurements_ref/
      index.rst                Overview; the section root
      tier_system.md           Tier System (hand-written and tracked; never touched)
      categories/index.rst     one section per ``CATEGORIES`` member
      shared/index.rst         columns no single operation owns
      metadata/index.rst       ``IMAGE`` and the experimental tags
      <group>/<Operation>.rst  one page per operation that writes a table

The section sidebar lists Overview, Tier System, Categories, Shared columns and
Metadata, then one captioned group of operation pages per stage: object
measurements, quality control, growth models, plate correction. The operations
come from :func:`phenotypic.util.measurement_producers`, the registry that also
drives ``deliverables/measurements_by_feature/``, so a new public measurer or
analyzer gets a page without any change here.

The reference documents only columns that appear in output tables, under the
header actually written. Each column name comes from the producer's
``output_header``, the method its ``measure``/``analyze`` uses (``QC_ICC_Metric``,
not the ``QUALITY_CHECK`` value ``QC_Metric``). Schemas and members that no
operation or CLI stage writes are listed in :data:`_NOT_IN_OUTPUT_TABLES` and
:data:`_OMITTED_MEMBERS`, with the reason.

Pages are regenerated on every build, so they never drift from the code. Do not
edit generated pages by hand; edit the schema enums, the producers, or this
extension. The one hand-written page, ``tier_system.md``, is kept across builds
(see :data:`_HAND_WRITTEN_PAGES`) and re-included in ``.gitignore``.
"""


import inspect
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

#: Files under ``measurements_ref/`` that are written by hand and tracked in git.
#: The generator deletes everything else in the directory before each build.
_HAND_WRITTEN_PAGES = frozenset({"tier_system.md"})

#: Schemas whose columns no operation or CLI stage writes, with the reason. They
#: stay public for code that reads them, but the reference omits them. Source:
#: docs/superpowers/reports/2026-09-30-measurements-reference-layout/output-column-audit.md
_NOT_IN_OUTPUT_TABLES: dict[str, str] = {
    "RADIAL_EXPANSION": "No operation or CLI stage writes these columns.",
    "ErrorCategory": (
        "Its labels are values of the ``Curation_Category`` column and names of the "
        "files under ``deliverables/errors/``, not column headers."
    ),
    "ColorComposition": (
        "``MeasureColorComposition`` is not exported from ``phenotypic.measure``, "
        "so no pipeline can run it."
    ),
}

#: Members of a documented schema that are never written to a table, by member name.
_OMITTED_MEMBERS: dict[str, frozenset[str]] = {
    # GridFinder writes the row/column numbers and indices, never the intervals.
    "GRID": frozenset(
        {"ROW_INTERVAL_START", "ROW_INTERVAL_END", "COL_INTERVAL_START", "COL_INTERVAL_END"}
    ),
    # The UUID is private metadata; the parent and format fields are never set.
    "IMAGE": frozenset({"UUID", "PARENT_IMAGE_NAME", "PARENT_UUID", "IMFORMAT"}),
}

#: Schemas that no producer declares but that appear on output rows, as
#: ``(schema, section key, note)``. Each is documented once, on the Shared
#: columns page, in the section of :data:`_SHARED_SECTIONS` its key names. Notes
#: are ``str.format`` templates over :func:`_path_names`.
_SHARED_ONLY: tuple[tuple[str, str, str], ...] = (
    ("OBJECT", "rows", ""),
    ("GRID", "rows", "Written on rows from a ``GridImage`` only."),
    (
        "METADATA_MATCH",
        "metadata",
        "Written only when a run is given ``--metadata``. True on a row that "
        "comes from the metadata file alone, with no detected colony.",
    ),
    (
        "CURATION",
        "curation",
        "Written once you curate colonies in the results viewer, to "
        "``{d}/{qc}/{curation_labels}`` and to one ``{d}/{errors}/<category>.parquet`` "
        "per category.",
    ),
)

#: Sections of the Shared columns page, in order, as ``(key, heading, intro)``.
#: ``stages`` expands to one section per stage whose producers share a schema
#: (every quality check writes ``QUALITY_CHECK``); ``several`` holds schemas two
#: or more operations declare as their own. Intros are templates like the notes.
_SHARED_SECTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "rows",
        "On every measurement row",
        "Every row of ``{d}/{measurements}`` says which colony it describes and "
        "where that colony is. The pipeline adds the ten bounding-box columns to "
        "every row itself, whether or not :doc:`../measure/MeasureBounds` is in "
        "your pipeline; they are documented on that page "
        "(:ref:`measurement-info-bbox`). The image bookkeeping columns are on "
        ":doc:`../metadata/index`.",
    ),
    ("metadata", "With a metadata file", ""),
    ("stages", "", ""),
    ("several", "Written by several operations", ""),
    ("curation", "After curation", ""),
)

#: Stages that group the operation pages, in sidebar order. ``base`` is resolved
#: lazily so the extension imports nothing heavy at module load.
_GROUP_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("measure", "Object measurements", "measurement operation", "phenotypic.abc_:MeasureFeatures"),
    ("qc", "Quality control", "quality check", "phenotypic.analysis.abc_:QualityCheck"),
    ("models", "Growth models", "growth model", "phenotypic.analysis.abc_:ModelFitter"),
    ("correction", "Plate correction", "plate correction", "phenotypic.analysis.abc_:EdgeCorrection"),
)


@dataclass(frozen=True)
class _Group:
    """One captioned group of operation pages."""

    slug: str
    caption: str
    noun: str
    base: type


@dataclass(frozen=True)
class _OperationPage:
    """One operation that writes a table, and where its page lives."""

    producer: Any  # phenotypic.util.MeasurementProducer
    group: _Group

    @property
    def name(self) -> str:
        return self.producer.output_key

    @property
    def docname(self) -> str:
        return f"{self.group.slug}/{self.name}"


# --------------------------------------------------------------------------- #
# RST helpers
# --------------------------------------------------------------------------- #


def _heading(title: str, underline: str) -> list[str]:
    """Return an RST heading block."""
    return [title, underline * len(title), ""]


def _section_label(info_cls: type[Any]) -> str:
    """Return the stable cross-reference label for one schema section."""
    slug = info_cls.__name__.lower().replace("_", "-")
    return f"measurement-info-{slug}"


def _operation_label(name: str) -> str:
    """Return the stable cross-reference label for one operation page."""
    return f"measurement-op-{name.lower()}"


def _linked_class_heading(info_cls: type[Any]) -> str:
    """A schema heading that links to the class's API page."""
    class_name = info_cls.__name__
    return f":doc:`{class_name} </api_reference/api/phenotypic.schema.{class_name}>`"


def _emitted_members(info_cls: type[Any]) -> list[Any]:
    """The members of *info_cls* that appear in output tables, in declaration order."""
    omitted = _OMITTED_MEMBERS.get(info_cls.__name__, frozenset())
    return [member for member in info_cls if member.name not in omitted]


def _class_section(
    info_cls: type[Any],
    *,
    header_for: Callable[[Any], str] | None = None,
    label: bool = True,
    note: str | None = None,
    underline: str = "-",
) -> str:
    """Render one schema: optional anchor, linked heading, change note, note, table.

    Args:
        info_cls: The schema to render.
        header_for: The producer's ``output_header``; ``None`` shows each
            member's value, which is right for schemas written as declared.
        label: Whether this is the schema's one canonical section, which
            carries the ``measurement-info-*`` anchor.
        note: A sentence placed above the table (a switch, a placeholder).
        underline: Heading underline character, for nesting.
    """
    out = [f".. _{_section_label(info_cls)}:", ""] if label else []
    out.extend(_heading(_linked_class_heading(info_cls), underline))
    change_note = info_cls.change_note()
    if change_note:
        out.extend([change_note, ""])
    if note:
        out.extend([note, ""])
    out.extend(
        [
            info_cls.rst_table(
                header=("Column", "Description"),
                use_headers=True,
                members=_emitted_members(info_cls),
                header_for=header_for,
            ),
            "",
        ]
    )
    return "\n".join(out)


def _write(path: Path, contents: str) -> None:
    """Write a generated page, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


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


def _resolve(dotted: str) -> type:
    """Import ``module:attr``."""
    import importlib

    module_name, attr = dotted.split(":")
    return getattr(importlib.import_module(module_name), attr)


def _groups() -> tuple[_Group, ...]:
    return tuple(
        _Group(slug, caption, noun, _resolve(base)) for slug, caption, noun, base in _GROUP_SPECS
    )


def _group_for(producer_cls: type, groups: tuple[_Group, ...]) -> _Group:
    """The first group whose base *producer_cls* subclasses."""
    # Analyzer bases are checked before MeasureFeatures only by declaration
    # order; the two hierarchies are disjoint, so the order never decides.
    for group in groups:
        if issubclass(producer_cls, group.base):
            return group
    raise ValueError(
        f"{producer_cls.__name__} writes a table but belongs to no reference group; "
        "add its base class to _GROUP_SPECS in docs/source/_extensions/measurements_ref.py."
    )


def _operation_pages() -> tuple[_OperationPage, ...]:
    """Every table-producing operation, grouped, in group then registry order."""
    from phenotypic.util import measurement_producers

    groups = _groups()
    pages = [_OperationPage(p, _group_for(p.producer, groups)) for p in measurement_producers()]
    order = {group.slug: index for index, group in enumerate(groups)}
    return tuple(sorted(pages, key=lambda page: order[page.group.slug]))


@dataclass(frozen=True)
class _SharedEntry:
    """One schema on the Shared columns page and how to name its columns."""

    info: type
    header_for: Callable[[Any], str] | None
    note: str | None


@dataclass(frozen=True)
class _SharedSection:
    heading: str
    intro: str
    entries: tuple[_SharedEntry, ...]


def _path_names() -> dict[str, str]:
    """Output path names the Shared page's templates refer to."""
    from phenotypic.sdk_ import (
        CURATION_LABELS_PARQUET,
        DIR_DELIVERABLES,
        DIR_ERRORS,
        DIR_QC,
        MEASUREMENTS_CSV,
    )

    return {
        "d": DIR_DELIVERABLES,
        "qc": DIR_QC,
        "errors": DIR_ERRORS,
        "curation_labels": CURATION_LABELS_PARQUET,
        "measurements": MEASUREMENTS_CSV,
    }


def _owners(pages: tuple[_OperationPage, ...]) -> dict[type, list[_OperationPage]]:
    """Operation pages that declare each schema as their own."""
    owners: dict[type, list[_OperationPage]] = {}
    for page in pages:
        for info in page.producer.primary_infos:
            owners.setdefault(info, []).append(page)
    return owners


def _shared_layout(pages: tuple[_OperationPage, ...]) -> tuple[_SharedSection, ...]:
    """Every section of the Shared columns page, derived from the producers.

    The single description of that page: :func:`_canonical_docnames` anchors
    each schema listed here on it, :func:`_build_shared_page` renders it, and
    :func:`_written_headers` names its columns, so the three cannot disagree.
    """
    import phenotypic.schema as schema

    paths = _path_names()
    by_key: dict[str, list[_SharedEntry]] = {}
    for name, key, note in _SHARED_ONLY:
        by_key.setdefault(key, []).append(
            _SharedEntry(getattr(schema, name), None, note.format(**paths) or None)
        )

    sections: list[_SharedSection] = []
    for key, heading, intro in _SHARED_SECTIONS:
        if key == "stages":
            sections.extend(_stage_sections(pages))
            continue
        if key == "several":
            entries = tuple(
                _SharedEntry(
                    info,
                    owners[0].producer.output_header,
                    "Written by "
                    + ", ".join(f":doc:`../{page.docname}`" for page in owners)
                    + ".",
                )
                for info, owners in _owners(pages).items()
                if len(owners) > 1
            )
        else:
            entries = tuple(by_key.get(key, ()))
        if entries:
            sections.append(_SharedSection(heading, intro.format(**paths), entries))
    return tuple(sections)


def _stage_sections(pages: tuple[_OperationPage, ...]) -> list[_SharedSection]:
    """One section per stage whose producers write a shared schema."""
    sections = []
    for group in _groups():
        infos = list(
            dict.fromkeys(
                info
                for page in pages
                if page.group == group
                for info in page.producer.shared_infos
            )
        )
        if not infos:
            continue
        note = _placeholder_sentence(group.base) or None
        entries = tuple(_SharedEntry(info, group.base.output_header, note) for info in infos)
        sections.append(_SharedSection(f"On every {group.noun} table", "", entries))
    return sections


def _canonical_docnames(pages: tuple[_OperationPage, ...]) -> dict[type, str]:
    """Map each schema to the one page that carries its anchor.

    A schema that exactly one operation declares as its own is anchored on
    that operation's page. Every schema in :func:`_shared_layout` (declared by
    no operation, shared by a stage, or declared by several operations) is
    anchored on the Shared columns page, and metadata schemas on the Metadata
    page.
    """
    import phenotypic.schema as schema

    canonical = {
        info: owners[0].docname
        for info, owners in _owners(pages).items()
        if len(owners) == 1
    }
    for section in _shared_layout(pages):
        for entry in section.entries:
            canonical[entry.info] = "shared/index"
    for info in _public_measurement_info_classes():
        if issubclass(info, schema.MetadataInfo):
            canonical[info] = "metadata/index"
    return canonical


def _written_headers(pages: tuple[_OperationPage, ...]) -> dict[Any, str]:
    """The header each documented member is written under, as its page shows it."""
    headers: dict[Any, str] = {}
    for page in pages:
        for info in page.producer.primary_infos:
            for member in info:
                headers.setdefault(member, page.producer.output_header(member))
    for section in _shared_layout(pages):
        for entry in section.entries:
            for member in entry.info:
                if entry.header_for is not None:
                    headers[member] = entry.header_for(member)
    return headers


def _optional_switches(producer_cls: type) -> dict[type, str]:
    """Schemas a boolean parameter switches on, mapped to that parameter.

    Found by construction rather than declared: build the operation with its
    defaults, then with each ``False``-default boolean set to ``True``, and
    compare the schemas it reports. Only ``MeasureFeatures`` operations report
    per-instance schemas.
    """
    getter = getattr(producer_cls, "get_measurement_infoclasses", None)
    if getter is None:
        return {}
    try:
        default = set(producer_cls().get_measurement_infoclasses())
    except Exception:  # noqa: BLE001 - an op that needs arguments has no switches to find
        return {}
    switches: dict[type, str] = {}
    for field_name, field in producer_cls.model_fields.items():
        if field.annotation is not bool or field.default is not False:
            continue
        try:
            enabled = set(producer_cls(**{field_name: True}).get_measurement_infoclasses())
        except Exception:  # noqa: BLE001
            continue
        for info in enabled - default:
            switches[info] = field_name
    return switches


def _public_path(producer_cls: type) -> str:
    """The public dotted path of an operation class."""
    import phenotypic.analysis
    import phenotypic.measure

    for module in (phenotypic.measure, phenotypic.analysis):
        if getattr(module, producer_cls.__name__, None) is producer_cls:
            return f"{module.__name__}.{producer_cls.__name__}"
    return f"{producer_cls.__module__}.{producer_cls.__qualname__}"


def _summary(producer_cls: type) -> str:
    """The first paragraph of the operation's docstring."""
    doc = inspect.getdoc(producer_cls) or ""
    paragraph = doc.strip().split("\n\n", 1)[0]
    return " ".join(line.strip() for line in paragraph.splitlines())


# --------------------------------------------------------------------------- #
# Operation pages
# --------------------------------------------------------------------------- #


def _group_facts(page: _OperationPage) -> list[tuple[str, str]]:
    """The facts every page of a group shares: rows, destination, header form."""
    from phenotypic.sdk_ import (
        DIR_DELIVERABLES,
        DIR_MEASUREMENTS_BY_FEATURE,
        DIR_QC,
        MEASUREMENTS_CSV,
        QC_DUCKDB,
    )

    name = page.name
    slug = page.group.slug
    if slug == "measure":
        return [
            ("One row per", "detected colony"),
            (
                "Written to",
                f"``{DIR_DELIVERABLES}/{MEASUREMENTS_CSV}`` (every operation, with "
                f"your metadata joined) and ``{DIR_DELIVERABLES}/"
                f"{DIR_MEASUREMENTS_BY_FEATURE}/{name}.csv``, each with a "
                "``.parquet`` beside it",
            ),
        ]
    if slug == "qc":
        return [
            (
                "Written to",
                f"the frame ``analyze()`` returns; a CLI run stores it as one table "
                f"in ``{DIR_DELIVERABLES}/{DIR_QC}/{QC_DUCKDB}``",
            ),
        ]
    if slug == "models":
        return [
            ("One row per", "fitted group (the ``groupby`` columns)"),
            (
                "Written to",
                f"the frame ``analyze()`` returns; when it is the pipeline's model, "
                f"a CLI run writes ``{DIR_DELIVERABLES}/{name}.csv`` and ``.parquet``",
            ),
        ]
    return [
        (
            "Written to",
            "the frame ``analyze()`` returns only; a CLI run does not write it to a file",
        ),
    ]


def _placeholder_block(placeholders: dict[str, str]) -> list[str]:
    """Define each placeholder a page's column patterns use.

    The definitions come from the producer's ``output_header_placeholders``,
    written next to the code that fills the placeholders in.
    """
    if not placeholders:
        return []
    out = [
        ".. rubric:: Placeholders",
        "",
        "Some column names depend on how you run the operation. They are shown "
        "as patterns, with these placeholders:",
        "",
    ]
    for placeholder, definition in placeholders.items():
        out.extend([f"``{placeholder}``", f"   {definition[0].upper()}{definition[1:]}", ""])
    return out


def _placeholder_sentence(producer_cls: type) -> str:
    """One sentence per placeholder, for a note above a shared table."""
    return " ".join(
        f"``{placeholder}`` is {definition}"
        for placeholder, definition in producer_cls.output_header_placeholders().items()
    )


def _field_list(facts: list[tuple[str, str]]) -> list[str]:
    # A bullet list, not an RST field list: pydata-sphinx-theme sets field
    # names in a narrow column that breaks "One row per" one letter per line.
    return [*(f"- **{key}:** {value}" for key, value in facts), ""]


def _operation_page(page: _OperationPage, canonical: dict[type, str]) -> str:
    """One operation: summary, output facts, then each schema it writes."""
    producer = page.producer
    cls = producer.producer
    columns = sum(
        len(_emitted_members(info))
        for info in (*producer.primary_infos, *producer.shared_infos)
    )
    placeholders = cls.output_header_placeholders()
    count = f"{columns}, each a pattern (see Placeholders)" if placeholders else str(columns)
    facts = [*_group_facts(page), ("Columns", count)]

    out = [
        f".. _{_operation_label(page.name)}:",
        "",
        *_heading(page.name, "="),
        _summary(cls),
        "",
        f"Parameters and usage: :py:class:`{page.name} <{_public_path(cls)}>`.",
        "",
        *_field_list(facts),
        *_placeholder_block(placeholders),
    ]
    switches = _optional_switches(cls)
    # Always-written schemas first, then the ones a parameter switches on.
    for info in sorted(producer.primary_infos, key=lambda info: info in switches):
        switch = switches.get(info)
        out.append(
            _class_section(
                info,
                header_for=producer.output_header,
                label=canonical.get(info) == page.docname,
                note=f"Written only when ``{switch}=True``." if switch else None,
            )
        )
    for info in producer.shared_infos:
        out.append(
            _class_section(
                info,
                header_for=producer.output_header,
                label=False,
                note=(
                    f"Every {page.group.noun} writes these columns; "
                    "see :ref:`measurement-shared-columns`."
                ),
            )
        )
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Overview, Shared columns, Metadata, Categories
# --------------------------------------------------------------------------- #

_OVERVIEW_INTRO = """\
PhenoTypic writes its results as tables. This reference documents every column
that appears in them, grouped by the operation that writes it: one page per
measurement operation and per analyzer, listed in the sidebar by stage. Each
column carries a **Type** badge that says how far a single value can be
trusted; the :doc:`tier_system` page explains the badges."""


def _overview_toctrees(pages: tuple[_OperationPage, ...]) -> list[str]:
    out = [
        ".. toctree::",
        "   :hidden:",
        "",
        "   Overview <self>",
        "   Tier System <tier_system>",
        "   Categories <categories/index>",
        "   Shared columns <shared/index>",
        "   Metadata <metadata/index>",
        "",
    ]
    for group in _groups():
        members = [page for page in pages if page.group == group]
        if not members:
            continue
        out.extend([".. toctree::", "   :hidden:", f"   :caption: {group.caption}", ""])
        out.extend(f"   {page.docname}" for page in members)
        out.append("")
    return out


def _build_overview_page(pages: tuple[_OperationPage, ...]) -> str:
    from phenotypic.sdk_ import (
        CURATION_LABELS_PARQUET,
        DIR_DELIVERABLES,
        DIR_MEASUREMENTS_BY_CATEGORY,
        DIR_MEASUREMENTS_BY_FEATURE,
        DIR_QC,
        MEASUREMENTS_CSV,
        QC_DUCKDB,
    )

    d = DIR_DELIVERABLES
    out = [
        ".. _measurements-overview:",
        "",
        *_heading("Measurements", "="),
        *_overview_toctrees(pages),
        _OVERVIEW_INTRO,
        "",
        *_heading("Reading a measurements table", "-"),
        f"Each row of ``{d}/{MEASUREMENTS_CSV}`` describes one detected colony. Its "
        "columns come in four blocks, always in this order:",
        "",
        "#. **Your experimental metadata**, from the ``--metadata`` file: strain, "
        "condition, replicate and the like (``Metadata_Strain``). See :doc:`metadata/index`.",
        "#. **Measurements**, one block per measurement operation in pipeline order. "
        "Every column name starts with its metric family: ``Size_Area`` is in the "
        "*Size* family and is written by :doc:`measure/MeasureSize`.",
        "#. **Image bookkeeping**: the image each colony came from "
        "(``Metadata_ImageName``, ``Metadata_BitDepth``).",
        "#. **Where the colony is**: ``Object_Label``, its bounding box (``Bbox_*``) "
        "and, on gridded plates, its well (``Grid_*``). See :doc:`shared/index`.",
        "",
        *_heading("Where tables are written", "-"),
        "A CLI run writes these files under its output directory:",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - File",
        "     - One row per",
        "     - Holds",
        f"   * - ``{d}/{MEASUREMENTS_CSV}`` (and ``.parquet``)",
        "     - detected colony",
        "     - every measurement column, with your metadata joined",
        f"   * - ``{d}/{DIR_MEASUREMENTS_BY_FEATURE}/<Operation>.csv``",
        "     - detected colony",
        "     - one measurement operation's columns, plus the metadata and position columns",
        f"   * - ``{d}/{DIR_MEASUREMENTS_BY_CATEGORY}/<Category>.csv``",
        "     - detected colony",
        "     - one :doc:`category <categories/index>`'s columns, plus the metadata and "
        "position columns",
        f"   * - ``{d}/{DIR_QC}/{QC_DUCKDB}``",
        "     - one table per quality check",
        "     - each check's columns; see *Quality control* in the sidebar",
        f"   * - ``{d}/<Model>.csv`` (and ``.parquet``)",
        "     - fitted group",
        "     - the pipeline's growth-model parameters and fit metrics",
        f"   * - ``{d}/{DIR_QC}/{CURATION_LABELS_PARQUET}``",
        "     - curated colony",
        "     - the category you assigned in the results viewer",
        "",
        "Analyzers you run yourself return the same columns from ``analyze()``.",
        "",
        *_heading("Reading an operation page", "-"),
        "Every operation page has the same parts:",
        "",
        "#. A one-line summary of what the operation measures, and a link to its "
        "parameters.",
        "#. The table's facts: what one row describes, where the table is written, and "
        "how many columns the operation adds.",
        "#. One column table per schema, listing each column exactly as it is written. "
        "Where a header embeds the analyzed column, the page shows a placeholder "
        "(``<metric>``) and an example.",
        "",
        *_heading("Operations", "-"),
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 30 20 50",
        "",
        "   * - Operation",
        "     - Stage",
        "     - What it writes",
    ]
    for page in pages:
        out.extend(
            [
                f"   * - :doc:`{page.name} <{page.docname}>`",
                f"     - {page.group.caption}",
                f"     - {_summary(page.producer.producer)}",
            ]
        )
    out.append("")
    return "\n".join(out)


def _build_shared_page(pages: tuple[_OperationPage, ...]) -> str:
    """Columns no single operation owns, each documented once."""
    out = [
        ".. _measurement-shared-columns:",
        "",
        *_heading("Shared columns", "="),
        "Columns that no single operation owns: they sit on every row, or on every "
        "table of one kind. Each is documented once, here.",
        "",
    ]
    for section in _shared_layout(pages):
        out.extend(_heading(section.heading, "-"))
        if section.intro:
            out.extend([section.intro, ""])
        out.extend(
            _class_section(
                entry.info, header_for=entry.header_for, note=entry.note, underline="~"
            )
            for entry in section.entries
        )
    return "\n".join(out)


_METADATA_INTRO = """\
Metadata columns all start with ``Metadata_``. The image bookkeeping columns are
set by PhenoTypic for every image. The experimental tags are the recommended
names for the columns of your ``--metadata`` file: a column named for a tag's
label is recognized as that tag, and any other column is kept as
``Metadata_<Label>`` (see :doc:`/explanation/metadata_namespace`)."""


def _build_metadata_page() -> str:
    import phenotypic.schema as schema

    infos = [
        info
        for info in _public_measurement_info_classes()
        if issubclass(info, schema.MetadataInfo)
    ]
    out = [*_heading("Metadata", "="), _METADATA_INTRO, ""]
    out.extend(_class_section(info) for info in infos)
    return "\n".join(out)


_CATEGORIES_INTRO = """\
.. _measurement-categories:

Every measurement column begins with its metric family: ``Size_Area`` belongs to
the **Size** family, which says only which schema the column comes from. A
**category** is a separate, curated grouping that cuts across families: one
category can gather size, intensity and colour columns that three different
schemas produce, and one column can belong to several categories.

A category tells you where to look first, not how far to trust a single value;
that is the job of the Type badge (see :doc:`../tier_system`).

Every run that measures objects writes one spreadsheet per category that has at
least one column in the run, under ``{deliverables}/{split_dir}/``, holding the
metadata and position columns plus that category's columns."""


def _category_section(
    category: Any,
    documented: set[Any] | None = None,
    headers: dict[Any, str] | None = None,
) -> str:
    """Render one category: anchor, heading, verbatim desc, output file, column table.

    Args:
        category: A ``CATEGORIES`` member.
        documented: Members the reference documents; others are left out so
            every row links to a real section. ``None`` keeps every member.
        headers: The header each member is written under (``QC_ICC_Metric``,
            not the ``QC_Metric`` value); a member missing from it shows its
            value.
    """
    from phenotypic.sdk_ import DIR_MEASUREMENTS_BY_CATEGORY

    out = [
        f".. _{category.anchor}:",
        "",
        *_heading(category.display_name, "-"),
        category.desc,
        "",
        f"Written to ``deliverables/{DIR_MEASUREMENTS_BY_CATEGORY}/{category.label}.csv`` "
        "(and ``.parquet``).",
        "",
    ]
    members = [
        member
        for member in category.members()
        if documented is None or member in documented
    ]
    if not members:
        # A header-only list-table is a docutils ERROR; say so in prose instead.
        out.extend(["No columns carry this category yet.", ""])
        return "\n".join(out)
    out.extend(
        [
            ".. list-table::",
            "   :header-rows: 1",
            "",
            "   * - Column",
            "     - Metric family",
            "     - Type",
        ]
    )
    for member in members:
        info_cls = type(member)
        out.extend(
            [
                f"   * - ``{(headers or {}).get(member, member.value)}``",
                f"     - :ref:`{info_cls.metric_family()} <{_section_label(info_cls)}>`",
                f"     - {member.use_badge}",
            ]
        )
    out.append("")
    return "\n".join(out)


def _build_categories_page(documented: set[Any], headers: dict[Any, str]) -> str:
    """Build the generated Categories page, one section per CATEGORIES member."""
    from phenotypic.schema import CATEGORIES
    from phenotypic.sdk_ import DIR_DELIVERABLES, DIR_MEASUREMENTS_BY_CATEGORY

    intro = _CATEGORIES_INTRO.format(
        deliverables=DIR_DELIVERABLES, split_dir=DIR_MEASUREMENTS_BY_CATEGORY
    )
    out = [*_heading("Categories", "="), intro, ""]
    out.extend(_category_section(category, documented, headers) for category in CATEGORIES)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def _documented_members(canonical: dict[type, str]) -> set[Any]:
    """Every member the reference documents somewhere."""
    return {member for info in canonical for member in _emitted_members(info)}


def _check_coverage(canonical: dict[type, str]) -> None:
    """Fail the build when a public schema has no place in the reference.

    Every public schema must be documented (declared by an operation, listed in
    :data:`_SHARED_ONLY`, or metadata) or excluded in
    :data:`_NOT_IN_OUTPUT_TABLES`, never both; every name in
    :data:`_OMITTED_MEMBERS` must still be a member. Without this, a new schema
    would drop out of the reference silently.

    Raises:
        RuntimeError: Naming each schema or member out of place.
    """
    import phenotypic.schema as schema

    documented = {info.__name__ for info in canonical}
    problems = [
        f"{info.__name__} is public but has no place in the reference: set "
        "_measurement_infoclass on the operation that writes it, or add it to "
        "_SHARED_ONLY or _NOT_IN_OUTPUT_TABLES in docs/source/_extensions/measurements_ref.py"
        for info in _public_measurement_info_classes()
        if info.__name__ not in documented and info.__name__ not in _NOT_IN_OUTPUT_TABLES
    ]
    problems.extend(
        f"{name} is listed in _NOT_IN_OUTPUT_TABLES but an operation declares it"
        for name in _NOT_IN_OUTPUT_TABLES
        if name in documented
    )
    known_sections = {key for key, _heading_text, _intro in _SHARED_SECTIONS}
    problems.extend(
        f"_SHARED_ONLY puts {name} in section {key!r}, which _SHARED_SECTIONS lacks"
        for name, key, _note in _SHARED_ONLY
        if key not in known_sections or key in {"stages", "several"}
    )
    for name, members in _OMITTED_MEMBERS.items():
        known = set(getattr(schema, name).__members__)
        problems.extend(
            f"_OMITTED_MEMBERS names {name}.{member}, which is not a member"
            for member in sorted(members - known)
        )
    if problems:
        raise RuntimeError("Measurements reference coverage:\n  " + "\n  ".join(problems))


#: Top-level entries under ``measurements_ref/`` the generator writes, besides
#: one folder per stage in :data:`_GROUP_SPECS`.
_GENERATED_ENTRIES = frozenset({"index.rst", "categories", "shared", "metadata"})

#: Entries earlier layouts wrote, removed so stale pages never reach Sphinx.
#: ``measurements`` held the single all-schemas page before 2026-09.
_RETIRED_ENTRIES = frozenset({"measurements"})


def _clean_generated(output_dir: Path) -> None:
    """Remove last build's generated pages, and nothing else.

    Only entries the generator owns are deleted. Anything else that is not a
    registered hand-written page is kept, with a warning: ``.gitignore`` hides
    ``measurements_ref/`` from git except the pages it re-includes, so such a
    file would otherwise sit untracked without anyone noticing.
    """
    import warnings

    if not output_dir.exists():
        return
    owned = _GENERATED_ENTRIES | _RETIRED_ENTRIES | {spec[0] for spec in _GROUP_SPECS}
    for path in output_dir.iterdir():
        if path.name in owned:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        elif path.name not in _HAND_WRITTEN_PAGES:
            warnings.warn(
                f"measurements_ref/{path.name} is not generated and not a registered "
                "hand-written page, so git ignores it. Add it to _HAND_WRITTEN_PAGES "
                "in docs/source/_extensions/measurements_ref.py and re-include it in "
                ".gitignore.",
                stacklevel=2,
            )


def _copy_measurement_assets(srcdir: str) -> None:
    """Copy packaged measurement images into the docs static tree so that
    ``/_static/measurements/...`` references resolve in the built HTML."""
    import phenotypic

    src = Path(phenotypic.__file__).resolve().parent / "_assets" / "measurements"
    if not src.is_dir():
        return
    dest = Path(srcdir) / "_static" / "measurements"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def _build_pages(srcdir: str) -> None:
    output_dir = Path(srcdir) / "measurements_ref"
    _clean_generated(output_dir)

    pages = _operation_pages()
    canonical = _canonical_docnames(pages)
    _check_coverage(canonical)
    _write(output_dir / "index.rst", _build_overview_page(pages))
    for page in pages:
        _write(output_dir / f"{page.docname}.rst", _operation_page(page, canonical))
    _write(output_dir / "shared" / "index.rst", _build_shared_page(pages))
    _write(output_dir / "metadata" / "index.rst", _build_metadata_page())
    _write(
        output_dir / "categories" / "index.rst",
        _build_categories_page(_documented_members(canonical), _written_headers(pages)),
    )


def _generate(app, _config):
    _copy_measurement_assets(app.srcdir)
    _build_pages(app.srcdir)
    print(f"Generated {os.path.join(app.srcdir, 'measurements_ref')}")


def setup(app):
    app.connect("config-inited", _generate)
    return {
        "version": "0.6",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
