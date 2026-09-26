"""README: split folders in the tree, a Categories column, a Categories section."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from phenotypic import ImagePipeline
from phenotypic._cli._cli_readme_generator import READMEGenerator
from phenotypic.measure import MeasureShape, MeasureSize
from phenotypic.schema import CATEGORIES, SHAPE, SIZE


def _generator(*measurers) -> READMEGenerator:
    # generate() reads config.image_type and config.pipeline_json.name.
    return READMEGenerator(
        config=SimpleNamespace(image_type="Image", pipeline_json=Path("pipeline.json")),
        pipeline=ImagePipeline(meas=list(measurers)),
    )


def test_output_tree_lists_both_split_folders() -> None:
    tree = _generator(MeasureSize())._generate_output_structure([])
    assert "measurements_by_feature/" in tree
    assert "measurements_by_category/" in tree


def test_categorized_table_gains_a_categories_column() -> None:
    table = _generator()._generate_measurement_table(SIZE)
    assert "| Column | Description | Categories |" in table
    assert f"| `{SIZE.AREA}` |" in table
    assert "Starting Metrics |" in table


def test_uncategorized_table_has_no_categories_column() -> None:
    table = _generator()._generate_measurement_table(SHAPE)
    assert "Categories" not in table


def test_categories_section_lists_present_columns_only() -> None:
    section = _generator(MeasureSize())._generate_categories_section()
    assert "## Measurement Categories" in section
    assert "### Starting Metrics" in section
    assert CATEGORIES.STARTING_METRICS.desc in section
    assert "measurements_by_category/StartingMetrics.csv" in section
    assert f"`{SIZE.AREA}`" in section
    assert "ColorLab_L*Medoid" not in section  # MeasureColor not configured


def test_categories_section_is_empty_without_categorized_measurers() -> None:
    assert _generator(MeasureShape())._generate_categories_section() == ""


def test_generate_includes_the_categories_section(tmp_path) -> None:
    path = _generator(MeasureSize()).generate(tmp_path, [])
    assert "## Measurement Categories" in path.read_text(encoding="utf-8")
