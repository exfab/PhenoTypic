"""Path-helper tests for the curation-labels store + error deliverables."""

from pathlib import Path

from phenotypic.sdk_ import (
    curation_labels_parquet_path,
    custom_categories_json_path,
    deliverables_dir,
    error_analysis_csv_path,
    error_analysis_html_path,
    error_analysis_parquet_path,
    error_category_parquet_path,
    errors_dir,
    qc_dir,
)


def test_errors_dir_under_deliverables():
    out = Path("/tmp/run")
    assert errors_dir(out) == deliverables_dir(out) / "errors"


def test_error_category_parquet_path_uses_bare_token():
    out = Path("/tmp/run")
    assert (
        error_category_parquet_path(out, "background_noise")
        == errors_dir(out) / "background_noise.parquet"
    )


def test_error_analysis_paths_under_deliverables():
    out = Path("/tmp/run")
    assert error_analysis_parquet_path(out) == deliverables_dir(out) / "error_analysis.parquet"
    assert error_analysis_csv_path(out) == deliverables_dir(out) / "error_analysis.csv"
    assert error_analysis_html_path(out) == deliverables_dir(out) / "error_analysis.html"


def test_curation_store_paths_under_qc():
    out = Path("/tmp/run")
    assert curation_labels_parquet_path(out) == qc_dir(out) / "curation_labels.parquet"
    assert custom_categories_json_path(out) == qc_dir(out) / "custom_categories.json"
