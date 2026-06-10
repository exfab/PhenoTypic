"""Unit tests for the generated Measurements Reference docs."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import phenotypic.schema as schema
from phenotypic.schema import MeasurementInfo
from pytest import MonkeyPatch


_EXTENSION_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "source"
    / "_extensions"
    / "measurements_ref.py"
)

_METADATA_ENUM_NAMES = {
    "METADATA",
    "ACQUISITION_METADATA",
    "CONDITION_METADATA",
    "EXPERIMENT_METADATA",
    "GENETIC_METADATA",
    "INCUBATION_METADATA",
    "PLATE_METADATA",
    "SAMPLE_METADATA",
}

_EXPERIMENTAL_TAG_NAMES = _METADATA_ENUM_NAMES - {"METADATA"}


def _load_extension(monkeypatch: MonkeyPatch):
    logging_module = types.ModuleType("sphinx.util.logging")
    logging_module.getLogger = lambda _name: types.SimpleNamespace(warning=lambda *a: None)
    sphinx_module = types.ModuleType("sphinx")
    sphinx_util_module = types.ModuleType("sphinx.util")
    sphinx_util_module.logging = logging_module
    monkeypatch.setitem(sys.modules, "sphinx", sphinx_module)
    monkeypatch.setitem(sys.modules, "sphinx.util", sphinx_util_module)
    monkeypatch.setitem(sys.modules, "sphinx.util.logging", logging_module)

    spec = importlib.util.spec_from_file_location(
        "measurements_ref_extension_under_test",
        _EXTENSION_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _public_measurement_info_classes() -> dict[str, type[MeasurementInfo]]:
    return {
        name: value
        for name in schema.__all__
        if name != "MeasurementInfo"
        and isinstance((value := getattr(schema, name, None)), type)
        and issubclass(value, MeasurementInfo)
    }


def _build_reference_tree(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    extension = _load_extension(monkeypatch)
    extension._build_pages(str(tmp_path))
    return tmp_path / "measurements_ref"


def test_build_pages_creates_grouped_reference_indexes(
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
) -> None:
    docs_root = _build_reference_tree(tmp_path, monkeypatch)

    assert (docs_root / "index.rst").is_file()
    assert (docs_root / "measurements" / "index.rst").is_file()
    assert (docs_root / "metadata" / "index.rst").is_file()


def test_every_public_measurement_info_has_one_generated_page(
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
) -> None:
    docs_root = _build_reference_tree(tmp_path, monkeypatch)
    page_paths = [
        path.relative_to(docs_root).as_posix()
        for path in docs_root.rglob("*.rst")
        if path.name != "index.rst"
    ]

    public_infos = _public_measurement_info_classes()
    assert len(page_paths) == len(public_infos)
    for name in public_infos:
        expected_stem = name.lower()
        matches = [path for path in page_paths if Path(path).stem == expected_stem]
        assert matches == [
            f"metadata/{expected_stem}.rst"
            if name in _METADATA_ENUM_NAMES
            else f"measurements/{expected_stem}.rst"
        ]


def test_experimental_tags_are_listed_under_metadata_only(
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
) -> None:
    docs_root = _build_reference_tree(tmp_path, monkeypatch)
    metadata_index = (docs_root / "metadata" / "index.rst").read_text()
    measurements_index = (docs_root / "measurements" / "index.rst").read_text()

    for name in _EXPERIMENTAL_TAG_NAMES:
        assert name in metadata_index
        assert name not in measurements_index


def test_metadata_index_encourages_standard_labels_and_mapping(
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
) -> None:
    docs_root = _build_reference_tree(tmp_path, monkeypatch)
    metadata_index = (docs_root / "metadata" / "index.rst").read_text()

    assert "``Metadata_*`` column labels" in metadata_index
    assert "downstream processing" in metadata_index
    assert "provide a mapping" in metadata_index


def test_measurement_toctree_uses_category_labels(
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
) -> None:
    docs_root = _build_reference_tree(tmp_path, monkeypatch)
    measurements_index = (docs_root / "measurements" / "index.rst").read_text()

    assert "   Size <size>" in measurements_index
    assert "   Shape <shape>" in measurements_index
    assert "   SIZE" not in measurements_index
    assert "   SHAPE" not in measurements_index


def test_metadata_index_embeds_overview_and_class_tables(
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
) -> None:
    docs_root = _build_reference_tree(tmp_path, monkeypatch)
    metadata_index = (docs_root / "metadata" / "index.rst").read_text()

    assert "Metadata Tag Overview" in metadata_index
    assert "Framework-populated image bookkeeping" in metadata_index
    assert "Use for sample-level biological identity" in metadata_index
    assert "METADATA\n--------" in metadata_index
    assert "SAMPLE_METADATA\n---------------" in metadata_index
    assert ".. list-table:: Category: **Metadata**" in metadata_index


def test_generated_enum_pages_escape_rst_markup(
        tmp_path: Path,
        monkeypatch: MonkeyPatch,
) -> None:
    docs_root = _build_reference_tree(tmp_path, monkeypatch)
    metadata_page = (docs_root / "metadata" / "metadata.rst").read_text()
    quality_check_page = (
        docs_root / "measurements" / "quality_check.rst"
    ).read_text()
    quality_se_page = (docs_root / "measurements" / "quality_se.rst").read_text()

    assert ":mod:" not in metadata_page
    assert ":class:" not in metadata_page
    assert ":meth:" not in quality_check_page
    assert r"\|mean\|" in quality_se_page
