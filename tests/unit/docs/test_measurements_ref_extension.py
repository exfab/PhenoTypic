"""Unit tests for the generated Measurements reference docs."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

import pytest
from pytest import MonkeyPatch

import phenotypic.schema as schema
from phenotypic.schema import Entry, MeasurementInfo, MetadataInfo
from phenotypic.util import measurement_producers


_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXTENSION_PATH = (
    _REPO_ROOT / "docs" / "source" / "_extensions" / "measurements_ref.py"
)
_API_INDEX_PATH = (
    _REPO_ROOT / "docs" / "source" / "api_reference" / "index.rst"
)
_TIER_SYSTEM_PATH = (
    _REPO_ROOT / "docs" / "source" / "measurements_ref" / "tier_system.md"
)
_GROUP_ORDER = ("measure", "qc", "models", "correction")


def _load_extension(monkeypatch: MonkeyPatch) -> Any:
    # The extension imports nothing from Sphinx at module load, so these unit
    # tests do not require a Sphinx application.
    spec = importlib.util.spec_from_file_location(
        "measurements_ref_extension_under_test",
        _EXTENSION_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_public_classes() -> tuple[type[MeasurementInfo], ...]:
    classes: list[type[MeasurementInfo]] = []
    seen: set[type[MeasurementInfo]] = set()
    for name in schema.__all__:
        value = getattr(schema, name, None)
        if (
            name != "MeasurementInfo"
            and isinstance(value, type)
            and issubclass(value, MeasurementInfo)
            and bool(getattr(value, "__members__", None))
            and value not in seen
        ):
            classes.append(value)
            seen.add(value)
    return tuple(classes)


def _build_reference_tree(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    extension = _load_extension(monkeypatch)
    extension._build_pages(str(tmp_path))
    return tmp_path / "measurements_ref"


def _all_pages(docs_root: Path) -> dict[str, str]:
    return {
        path.relative_to(docs_root).with_suffix("").as_posix(): path.read_text(
            encoding="utf-8"
        )
        for path in sorted(docs_root.rglob("*.rst"))
    }


def _class_heading(class_name: str) -> str:
    return (
        f":doc:`{class_name} "
        f"</api_reference/api/phenotypic.schema.{class_name}>`"
    )


def _anchor(info_cls: type) -> str:
    return f".. _measurement-info-{info_cls.__name__.lower().replace('_', '-')}:"


@pytest.fixture(scope="module")
def reference(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """The generated reference, built once for the read-only tests."""
    root = tmp_path_factory.mktemp("docs")
    extension = _load_extension(MonkeyPatch())
    extension._build_pages(str(root))
    return _all_pages(root / "measurements_ref")


# --------------------------------------------------------------------------- #
# Structure and sidebar
# --------------------------------------------------------------------------- #


def test_one_page_per_table_producer_plus_the_section_pages(
    reference: dict[str, str],
) -> None:
    operation_pages = {
        name for name in reference if name.split("/")[0] in _GROUP_ORDER
    }
    assert {name.split("/")[1] for name in operation_pages} == {
        producer.output_key for producer in measurement_producers()
    }
    assert set(reference) - operation_pages == {
        "index",
        "categories/index",
        "shared/index",
        "metadata/index",
    }


def test_outlier_removers_have_no_page(reference: dict[str, str]) -> None:
    # They drop rows and add no columns, so there is nothing to document.
    names = {name.split("/")[-1] for name in reference}
    assert "MADOutlierRemover" not in names
    assert "TukeyOutlierRemover" not in names


def test_sidebar_is_overview_block_then_one_captioned_group_per_stage(
    reference: dict[str, str],
) -> None:
    overview = reference["index"]
    assert (
        ".. toctree::\n   :hidden:\n\n"
        "   Overview <self>\n"
        "   Tier System <tier_system>\n"
        "   Categories <categories/index>\n"
        "   Shared columns <shared/index>\n"
        "   Metadata <metadata/index>\n"
    ) in overview
    captions = re.findall(r":caption: (.+)", overview)
    assert captions == [
        "Object measurements",
        "Quality control",
        "Growth models",
        "Plate correction",
    ]
    for slug, caption in zip(_GROUP_ORDER, captions, strict=True):
        block = overview.split(f":caption: {caption}\n\n", 1)[1].split("\n\n", 1)[0]
        entries = [line.strip() for line in block.splitlines()]
        assert entries, caption
        assert all(entry.startswith(f"{slug}/") for entry in entries)


def test_only_the_overview_carries_toctrees(reference: dict[str, str]) -> None:
    for name, text in reference.items():
        if name != "index":
            assert ".. toctree::" not in text, name


def test_setup_generates_pages_before_sphinx_source_discovery(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    extension = _load_extension(monkeypatch)
    callbacks: dict[str, Any] = {}

    class FakeApp:
        srcdir = str(tmp_path)

        def connect(self, event: str, callback: Any) -> None:
            callbacks[event] = callback

    extension.setup(FakeApp())

    assert set(callbacks) == {"config-inited"}
    callbacks["config-inited"](FakeApp(), object())
    docs_root = tmp_path / "measurements_ref"
    for page in ("index", "shared/index", "metadata/index", "categories/index"):
        assert (docs_root / f"{page}.rst").is_file()


def test_rebuild_keeps_the_hand_written_tier_page_and_drops_stale_pages(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    docs_root = tmp_path / "measurements_ref"
    (docs_root / "measurements").mkdir(parents=True)
    (docs_root / "measurements" / "index.rst").write_text("stale")
    (docs_root / "tier_system.md").write_text("hand-written")

    _build_reference_tree(tmp_path, monkeypatch)

    assert (docs_root / "tier_system.md").read_text() == "hand-written"
    assert not (docs_root / "measurements").exists()


def test_tier_page_is_tracked_and_carries_the_badge_anchors() -> None:
    text = _TIER_SYSTEM_PATH.read_text(encoding="utf-8")
    assert text.startswith("(measurement-classification)=\n\n# Tier System")
    assert "(measurement-tiers)=" in text
    gitignore = (_REPO_ROOT / ".gitignore").read_text()
    assert "!docs/source/measurements_ref/tier_system.md" in gitignore


# --------------------------------------------------------------------------- #
# Every schema documented once, and only what appears in output tables
# --------------------------------------------------------------------------- #


def test_every_documented_schema_has_exactly_one_anchor(
    reference: dict[str, str], monkeypatch: MonkeyPatch
) -> None:
    extension = _load_extension(monkeypatch)
    combined = "\n".join(reference.values())
    for info_cls in _canonical_public_classes():
        count = combined.count(_anchor(info_cls) + "\n")
        if info_cls.__name__ in extension._NOT_IN_OUTPUT_TABLES:
            assert count == 0, info_cls.__name__
            assert _class_heading(info_cls.__name__) not in combined
        else:
            assert count == 1, info_cls.__name__


def test_schema_anchors_sit_on_the_owning_page(reference: dict[str, str]) -> None:
    assert _anchor(schema.SIZE) in reference["measure/MeasureSize"]
    assert _anchor(schema.BBOX) in reference["measure/MeasureBounds"]
    assert _anchor(schema.QUALITY_ICC) in reference["qc/ICC"]
    for shared in (schema.OBJECT, schema.GRID, schema.QUALITY_CHECK, schema.MODEL_METRICS):
        assert _anchor(shared) in reference["shared/index"], shared.__name__
    for info_cls in _canonical_public_classes():
        if issubclass(info_cls, MetadataInfo):
            assert _anchor(info_cls) in reference["metadata/index"]


def test_shared_schemas_repeat_on_each_producer_page_without_an_anchor(
    reference: dict[str, str],
) -> None:
    for producer in measurement_producers():
        for info_cls in producer.shared_infos:
            text = next(
                body for name, body in reference.items() if name.endswith(f"/{producer.output_key}")
            )
            assert _class_heading(info_cls.__name__) in text
            assert _anchor(info_cls) not in text


def test_members_never_written_are_left_out(reference: dict[str, str]) -> None:
    combined = "\n".join(reference.values())
    for member in (
        schema.GRID.ROW_INTERVAL_START,
        schema.GRID.COL_INTERVAL_END,
        schema.IMAGE.UUID,
        schema.IMAGE.PARENT_UUID,
    ):
        assert f"``{member.value}``" not in combined
    assert "``Grid_RowNum``" in combined
    assert "``Metadata_ImageName``" in combined


def test_compatibility_alias_is_deduplicated_to_canonical_class(
    reference: dict[str, str],
) -> None:
    combined = "\n".join(reference.values())
    assert combined.count(_class_heading("ORIENTATION_ZONE_DIAGNOSTIC")) == 1
    assert _class_heading("ORIENTATION_ZONES") not in combined


def test_an_undeclared_public_schema_fails_the_build(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    class FUTURE_MEASUREMENT(MeasurementInfo):
        @classmethod
        def metric_family(cls) -> str:
            return "FutureMeasurement"

        VALUE = Entry("Value", "A future measurement value.")

    monkeypatch.setattr(schema, "FUTURE_MEASUREMENT", FUTURE_MEASUREMENT, raising=False)
    monkeypatch.setattr(schema, "__all__", [*schema.__all__, "FUTURE_MEASUREMENT"])

    with pytest.raises(RuntimeError, match="FUTURE_MEASUREMENT is public but has no place in the reference"):
        _build_reference_tree(tmp_path, monkeypatch)


def test_a_future_metadata_owner_is_documented_automatically(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    class FUTURE_METADATA(MetadataInfo):
        VALUE = Entry("Value", "A future metadata value.")

    monkeypatch.setattr(schema, "FUTURE_METADATA", FUTURE_METADATA, raising=False)
    monkeypatch.setattr(schema, "__all__", [*schema.__all__, "FUTURE_METADATA"])

    pages = _all_pages(_build_reference_tree(tmp_path, monkeypatch))
    assert _class_heading("FUTURE_METADATA") in pages["metadata/index"]


def test_omitted_member_names_must_exist(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    extension = _load_extension(monkeypatch)
    monkeypatch.setitem(
        extension._OMITTED_MEMBERS, "GRID", frozenset({"RENAMED_AWAY"})
    )
    with pytest.raises(RuntimeError, match="GRID.RENAMED_AWAY"):
        extension._build_pages(str(tmp_path))


# --------------------------------------------------------------------------- #
# Column names are the headers actually written
# --------------------------------------------------------------------------- #


def _name_cells(text: str) -> list[str]:
    return re.findall(r"^   \* - ``([^`]+)``$", text, flags=re.MULTILINE)


def test_operation_pages_show_each_producers_emitted_header(
    reference: dict[str, str], monkeypatch: MonkeyPatch
) -> None:
    extension = _load_extension(monkeypatch)
    for producer in measurement_producers():
        text = next(
            body for name, body in reference.items() if name.endswith(f"/{producer.output_key}")
        )
        cells = set(_name_cells(text))
        for info_cls in (*producer.primary_infos, *producer.shared_infos):
            for member in extension._emitted_members(info_cls):
                assert producer.output_header(member) in cells, (producer.output_key, member)


def test_analyzer_pages_never_show_enum_values_that_are_not_written(
    reference: dict[str, str],
) -> None:
    icc = reference["qc/ICC"]
    assert "``QC_ICC_Metric``" in icc
    assert "``QC_Metric``" not in icc
    model = reference["models/LogGrowthModel"]
    assert "``LogGrowthModel_<metric>_r``" in model
    assert "``LogGrowthModel_r``" not in model
    edge = reference["correction/EdgeCorrector"]
    assert "``EdgeCorrection_NewVal-<column>``" in edge


def test_every_placeholder_on_a_page_is_defined_there(reference: dict[str, str]) -> None:
    for name, text in reference.items():
        if name.split("/")[0] not in _GROUP_ORDER:
            continue
        used = {token for cell in _name_cells(text) for token in re.findall(r"<[a-z]+>", cell)}
        defined = set(re.findall(r"^``(<[a-z]+>)``$", text, flags=re.MULTILINE))
        assert used == defined, name


def test_texture_scale_placeholder_is_defined_with_its_default(
    reference: dict[str, str],
) -> None:
    text = reference["measure/MeasureTexture"]
    assert "``Texture_Contrast-<direction>-scale<x>``" in text
    definition = text.split("``<x>``\n", 1)[1].split("\n\n", 1)[0]
    assert "``scale``" in definition
    assert "``scale05``" in definition
    assert "``avg``" in text.split("``<direction>``\n", 1)[1]


def test_parameter_switched_schemas_say_which_parameter(reference: dict[str, str]) -> None:
    color = reference["measure/MeasureColor"]
    assert "Written only when ``include_XYZ=True``." in color
    assert "Written only when ``include_xy=True``." in color
    assert color.index(_class_heading("ColorLab")) < color.index(_class_heading("ColorXYZ"))
    zones = reference["measure/MeasureOrientationZones"]
    assert "Written only when ``include_diagnostics=True``." in zones


def test_shared_page_defines_its_placeholders(reference: dict[str, str]) -> None:
    shared = reference["shared/index"]
    assert "``QC_<name>_Metric``" in shared
    assert "``<name>`` is the check's ``name``: ``ICC`` writes ``QC_ICC_Metric``" in shared
    assert "``ModelMetrics_<metric>_R2``" in shared
    assert "``<metric>`` is the fitted column" in shared


# --------------------------------------------------------------------------- #
# Tables, change notes, escaping
# --------------------------------------------------------------------------- #


def test_class_section_renders_the_change_note_above_the_table(monkeypatch: MonkeyPatch):
    """Mutation: drop the change_note() line from _class_section -> fails."""
    extension = _load_extension(monkeypatch)
    section = extension._class_section(schema.SIZE)
    marker = ".. versionchanged:: 0.20.0"
    assert marker in section
    assert section.index(marker) < section.index(".. list-table::")
    # TEXTURE carries the single-channel note too; BBOX carries none.
    assert marker not in extension._class_section(schema.BBOX)


def test_generated_tables_escape_rst_markup(reference: dict[str, str]) -> None:
    combined = "\n".join(reference.values())
    # Roles, not the list-table ``:class:`` option the tables carry.
    assert ":mod:`" not in reference["metadata/index"]
    assert ":class:`" not in reference["metadata/index"]
    assert ":meth:`" not in combined
    assert "   :class: phenotypic-measurement-table" in combined
    assert r"\|mean\|" in combined


def test_schema_is_included_in_api_reference_autosummary() -> None:
    api_index = _API_INDEX_PATH.read_text()

    assert "   phenotypic.schema\n" in api_index


# --------------------------------------------------------------------------- #
# Categories
# --------------------------------------------------------------------------- #


def _category_sections(rst: str) -> dict[str, str]:
    """Split the Categories page into one text block per category anchor.

    The parameter is not called ``page``: the CI shard guard
    (``tests/unit/ci/test_pytest_shard_manifest.py``) reads a function
    parameter of that name as Playwright's fixture and would demand a browser
    shard for this module.
    """
    from phenotypic.schema import CATEGORIES

    starts = sorted((rst.index(f".. _{c.anchor}:"), c.anchor) for c in CATEGORIES)
    ends = [start for start, _anchor in starts[1:]] + [len(rst)]
    return {
        anchor: rst[start:end]
        for (start, anchor), end in zip(starts, ends, strict=True)
    }


def test_categories_page_has_one_section_per_category_with_verbatim_desc(
    reference: dict[str, str],
) -> None:
    from phenotypic.schema import CATEGORIES

    text = reference["categories/index"]
    assert text.count(".. _measurement-categories:") == 1
    for category in CATEGORIES:
        assert text.count(f".. _{category.anchor}:") == 1
        assert category.display_name in text
        assert category.desc in text
        assert f"measurements_by_category/{category.label}.csv" in text
    assert "``Size_Area``" in text
    assert ":ref:`Size <measurement-info-size>`" in text


def test_categories_page_lists_every_documented_member_once_with_its_type_badge(
    reference: dict[str, str],
) -> None:
    from phenotypic.schema import CATEGORIES

    sections = _category_sections(reference["categories/index"])
    for category in CATEGORIES:
        section = sections[category.anchor]
        members = category.members()
        assert members, f"{category.label} has no members"
        assert section.count("   * - ``") == len(members)
        for member in members:
            row = f"   * - ``{member.value}``\n"
            assert section.count(row) == 1
            row_block = section[section.index(row) :].split("   * - ", 2)[1]
            assert member.use_badge in row_block


def test_every_category_row_links_to_a_real_schema_anchor(
    reference: dict[str, str],
) -> None:
    combined = "\n".join(reference.values())
    for target in re.findall(r"<(measurement-info-[a-z0-9-]+)>", reference["categories/index"]):
        assert f".. _{target}:" in combined


def test_every_badge_anchor_resolves_on_the_categories_page(
    reference: dict[str, str],
) -> None:
    anchors = {
        match
        for info_cls in _canonical_public_classes()
        for match in re.findall(r"<(measurement-category-[a-z0-9]+)>", info_cls.rst_table())
    }
    assert anchors, "no category badge rendered anywhere"
    for anchor in anchors:
        assert f".. _{anchor}:" in reference["categories/index"]


def test_category_without_members_renders_a_sentence_not_an_empty_table(
    monkeypatch: MonkeyPatch,
) -> None:
    # A header-only list-table is a docutils ERROR that drops the section's
    # table while the build still exits 0; an empty category must not emit one.
    extension = _load_extension(monkeypatch)

    class _EmptyCategory:
        anchor = "measurement-category-empty"
        display_name = "Empty"
        desc = "A category nothing is tagged with yet."
        label = "Empty"

        def members(self) -> tuple[()]:
            return ()

    section = extension._category_section(_EmptyCategory())
    assert ".. _measurement-category-empty:" in section
    assert "A category nothing is tagged with yet." in section
    assert "No columns carry this category yet." in section
    assert ".. list-table::" not in section


# --------------------------------------------------------------------------- #
# Navbar and sidebar resizing
# --------------------------------------------------------------------------- #


def test_navbar_measurements_is_a_plain_link_to_the_overview() -> None:
    navbar = (_REPO_ROOT / "docs" / "source" / "_templates" / "navbar-nav.html").read_text()
    item = navbar[navbar.index("{# --- Measurements") : navbar.index("{# --- Remaining")]
    assert "pathto('measurements_ref/index')" in item
    assert "_pn.startswith('measurements_ref/')" in item
    assert "dropdown" not in item


def test_sidebar_resize_assets_are_registered_and_match_theme_breakpoints() -> None:
    conf = (_REPO_ROOT / "docs" / "source" / "conf.py").read_text()
    assert '"sidebar-resize.css"' in conf
    assert '"sidebar-resize.js"' in conf
    css = (_REPO_ROOT / "docs" / "source" / "_static" / "sidebar-resize.css").read_text()
    # pydata-sphinx-theme turns the sidebars into drawers below lg / xl.
    assert "@media (min-width: 960px)" in css
    assert "@media (min-width: 1200px)" in css


# --------------------------------------------------------------------------- #
# Shared page, categories and cleanup follow the data, not a fixed list
# --------------------------------------------------------------------------- #


def test_a_schema_several_operations_declare_is_anchored_once_on_the_shared_page(
    monkeypatch: MonkeyPatch,
) -> None:
    import dataclasses

    extension = _load_extension(monkeypatch)
    pages = extension._operation_pages()
    size = next(page for page in pages if page.name == "MeasureSize")
    twin = dataclasses.replace(
        size, producer=dataclasses.replace(size.producer, output_key="MeasureSizeTwin")
    )
    pages = (*pages, twin)

    assert extension._canonical_docnames(pages)[schema.SIZE] == "shared/index"
    shared = extension._build_shared_page(pages)
    assert shared.count(_anchor(schema.SIZE) + "\n") == 1
    assert "Written by several operations" in shared
    assert ":doc:`../measure/MeasureSize`, :doc:`../measure/MeasureSizeTwin`" in shared
    extension._check_coverage(extension._canonical_docnames(pages))
    assert _anchor(schema.SIZE) not in extension._operation_page(
        size, extension._canonical_docnames(pages)
    )


def test_every_shared_page_section_comes_from_the_layout(monkeypatch: MonkeyPatch) -> None:
    extension = _load_extension(monkeypatch)
    pages = extension._operation_pages()
    shared = extension._build_shared_page(pages)
    for section in extension._shared_layout(pages):
        assert f"{section.heading}\n{'-' * len(section.heading)}" in shared
        for entry in section.entries:
            assert shared.count(_anchor(entry.info) + "\n") == 1


def test_a_shared_only_entry_in_an_unknown_section_fails_the_build(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    extension = _load_extension(monkeypatch)
    monkeypatch.setattr(
        extension, "_SHARED_ONLY", (*extension._SHARED_ONLY, ("OBJECT", "nowhere", ""))
    )
    with pytest.raises(RuntimeError, match="section 'nowhere'"):
        extension._build_pages(str(tmp_path))


def test_category_rows_show_the_written_header_not_the_enum_value(
    monkeypatch: MonkeyPatch,
) -> None:
    extension = _load_extension(monkeypatch)
    headers = extension._written_headers(extension._operation_pages())

    class _QcCategory:
        anchor = "measurement-category-qc"
        display_name = "Qc"
        desc = "A category holding the shared QC metric."
        label = "Qc"

        def members(self) -> tuple[Any, ...]:
            return (schema.QUALITY_CHECK.METRIC, schema.LOG_GROWTH_MODEL.GROWTH_RATE)

    rate = schema.LOG_GROWTH_MODEL.GROWTH_RATE

    section = extension._category_section(_QcCategory(), headers=headers)
    assert "``QC_<name>_Metric``" in section
    assert f"``LogGrowthModel_<metric>_{rate.label}``" in section
    assert "``QC_Metric``" not in section
    assert f"``{rate.value}``" not in section


def test_rebuild_keeps_and_warns_about_an_unregistered_hand_written_page(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    docs_root = tmp_path / "measurements_ref"
    docs_root.mkdir()
    (docs_root / "faq.md").write_text("hand-written, not registered")

    with pytest.warns(UserWarning, match="measurements_ref/faq.md is not generated"):
        _build_reference_tree(tmp_path, monkeypatch)

    assert (docs_root / "faq.md").read_text() == "hand-written, not registered"
