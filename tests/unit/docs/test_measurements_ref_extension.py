"""Unit tests for the generated Measurements reference docs."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import phenotypic.schema as schema
from phenotypic.schema import Entry, MeasurementInfo
from pytest import MonkeyPatch


_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXTENSION_PATH = (
    _REPO_ROOT / "docs" / "source" / "_extensions" / "measurements_ref.py"
)
_API_INDEX_PATH = (
    _REPO_ROOT / "docs" / "source" / "api_reference" / "index.rst"
)


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
            and value not in seen
        ):
            classes.append(value)
            seen.add(value)
    return tuple(classes)


def _build_reference_tree(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    extension = _load_extension(monkeypatch)
    extension._build_pages(str(tmp_path))
    return tmp_path / "measurements_ref"


def _class_heading(class_name: str) -> str:
    return (
        f":doc:`{class_name} "
        f"</api_reference/api/phenotypic.schema.{class_name}>`"
    )


def test_build_pages_creates_exactly_two_reference_pages(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    docs_root = _build_reference_tree(tmp_path, monkeypatch)

    pages = sorted(
        path.relative_to(docs_root).as_posix()
        for path in docs_root.rglob("*.rst")
    )
    assert pages == ["measurements/index.rst", "metadata/index.rst"]


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
    assert (docs_root / "measurements" / "index.rst").is_file()
    assert (docs_root / "metadata" / "index.rst").is_file()


def test_measurements_page_has_metadata_as_its_only_toctree_child(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    docs_root = _build_reference_tree(tmp_path, monkeypatch)
    measurements_page = (docs_root / "measurements" / "index.rst").read_text()
    metadata_page = (docs_root / "metadata" / "index.rst").read_text()

    assert ".. toctree::\n   :hidden:\n\n   ../metadata/index" in measurements_page
    assert ".. toctree::" not in metadata_page


def test_every_canonical_public_class_appears_once_on_the_correct_page(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    docs_root = _build_reference_tree(tmp_path, monkeypatch)
    measurements_page = (docs_root / "measurements" / "index.rst").read_text()
    metadata_page = (docs_root / "metadata" / "index.rst").read_text()
    combined = measurements_page + metadata_page

    public_classes = _canonical_public_classes()
    assert combined.count(".. list-table:: Category:") == len(public_classes)
    for info_cls in public_classes:
        heading = _class_heading(info_cls.__name__)
        expected_page = (
            metadata_page
            if info_cls.category().startswith("Metadata")
            else measurements_page
        )
        other_page = (
            measurements_page
            if info_cls.category().startswith("Metadata")
            else metadata_page
        )
        assert expected_page.count(heading) == 1
        assert heading not in other_page


def test_class_order_follows_schema_export_order(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    docs_root = _build_reference_tree(tmp_path, monkeypatch)
    measurements_page = (docs_root / "measurements" / "index.rst").read_text()
    metadata_page = (docs_root / "metadata" / "index.rst").read_text()

    public_classes = _canonical_public_classes()
    expected_groups = (
        (
            measurements_page,
            [
                info_cls
                for info_cls in public_classes
                if not info_cls.category().startswith("Metadata")
            ],
        ),
        (
            metadata_page,
            [
                info_cls
                for info_cls in public_classes
                if info_cls.category().startswith("Metadata")
            ],
        ),
    )
    for page, expected_classes in expected_groups:
        positions = [page.index(_class_heading(info_cls.__name__)) for info_cls in expected_classes]
        assert positions == sorted(positions)


def test_future_classes_are_discovered_and_partitioned_automatically(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    class FUTURE_MEASUREMENT(MeasurementInfo):
        @classmethod
        def category(cls) -> str:
            return "FutureMeasurement"

        VALUE = Entry("Value", "A future measurement value.")

    class FUTURE_METADATA(MeasurementInfo):
        @classmethod
        def category(cls) -> str:
            return "MetadataFuture"

        VALUE = Entry("Value", "A future metadata value.")

    monkeypatch.setattr(
        schema, "FUTURE_MEASUREMENT", FUTURE_MEASUREMENT, raising=False
    )
    monkeypatch.setattr(schema, "FUTURE_METADATA", FUTURE_METADATA, raising=False)
    monkeypatch.setattr(
        schema,
        "__all__",
        [*schema.__all__, "FUTURE_MEASUREMENT", "FUTURE_METADATA"],
    )

    docs_root = _build_reference_tree(tmp_path, monkeypatch)
    measurements_page = (docs_root / "measurements" / "index.rst").read_text()
    metadata_page = (docs_root / "metadata" / "index.rst").read_text()

    assert _class_heading("FUTURE_MEASUREMENT") in measurements_page
    assert _class_heading("FUTURE_MEASUREMENT") not in metadata_page
    assert _class_heading("FUTURE_METADATA") in metadata_page
    assert _class_heading("FUTURE_METADATA") not in measurements_page


def test_compatibility_alias_is_deduplicated_to_canonical_class(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    docs_root = _build_reference_tree(tmp_path, monkeypatch)
    measurements_page = (docs_root / "measurements" / "index.rst").read_text()

    assert measurements_page.count(
        _class_heading("ORIENTATION_ZONE_DIAGNOSTIC")
    ) == 1
    assert _class_heading("ORIENTATION_ZONES") not in measurements_page


def test_pages_only_add_linked_class_sections_and_tables(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    docs_root = _build_reference_tree(tmp_path, monkeypatch)
    combined = "\n".join(
        path.read_text() for path in sorted(docs_root.rglob("*.rst"))
    )

    assert ".. _measurement-info-shape:" in combined
    assert _class_heading("SHAPE") in combined
    assert "Python export:" not in combined
    assert "Compatibility alias for" not in combined
    assert "Metadata Tag Overview" not in combined
    assert ".. grid::" not in combined
    assert "Browse " not in combined


def test_generated_tables_escape_rst_markup(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    docs_root = _build_reference_tree(tmp_path, monkeypatch)
    measurements_page = (docs_root / "measurements" / "index.rst").read_text()
    metadata_page = (docs_root / "metadata" / "index.rst").read_text()

    assert ":mod:" not in metadata_page
    assert ":class:" not in metadata_page
    assert ":meth:" not in measurements_page
    assert r"\|mean\|" in measurements_page


def test_schema_is_included_in_api_reference_autosummary() -> None:
    api_index = _API_INDEX_PATH.read_text()

    assert "   phenotypic.schema\n" in api_index
