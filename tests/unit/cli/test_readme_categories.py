"""README: split folders in the tree, a Categories column, a Categories section."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from phenotypic import ImagePipeline
from phenotypic._cli._cli_readme_generator import READMEGenerator
from phenotypic.measure import MeasureShape, MeasureSize
from phenotypic.schema import CATEGORIES, SHAPE, SIZE
from phenotypic.sdk_ import DIR_MEASUREMENTS_BY_CATEGORY, DIR_MEASUREMENTS_BY_FEATURE


def _generator(*measurers) -> READMEGenerator:
    # generate() reads config.image_type and config.pipeline_json.name.
    return READMEGenerator(
        config=SimpleNamespace(image_type="Image", pipeline_json=Path("pipeline.json")),
        pipeline=ImagePipeline(meas=list(measurers)),
    )


def test_output_tree_lists_both_split_folders() -> None:
    tree = _generator(MeasureSize())._generate_output_structure([])
    assert f"{DIR_MEASUREMENTS_BY_FEATURE}/" in tree
    assert f"{DIR_MEASUREMENTS_BY_CATEGORY}/" in tree


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


def test_categories_section_lists_a_repeated_measurer_column_once() -> None:
    section = _generator(MeasureSize(), MeasureSize())._generate_categories_section()
    assert section.count(f"`{SIZE.AREA}`") == 1


class _UnreadableInfo:
    """Stands in for a third-party info class whose members cannot be read."""

    def __iter__(self):
        raise RuntimeError("malformed info")


def test_malformed_info_does_not_break_the_readme(tmp_path, caplog) -> None:
    generator = _generator(MeasureSize())
    real = generator._get_measurement_infoclasses
    # A member without a `categories` attribute is uncategorized, not an error.
    generator._get_measurement_infoclasses = lambda measurer: [  # type: ignore[method-assign]
        _UnreadableInfo(),
        ["Custom_Untagged"],
        *real(measurer),
    ]

    with caplog.at_level("WARNING", logger="phenotypic._cli._cli_readme_generator"):
        section = generator._generate_categories_section()
        text = generator.generate(tmp_path, []).read_text(encoding="utf-8")

    assert f"`{SIZE.AREA}`" in section
    assert "Custom_Untagged" not in section
    assert "## Measurement Categories" in text
    assert f"`{SIZE.AREA}`" in text
    assert "malformed info" in caplog.text
