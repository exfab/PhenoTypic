"""Unit tests for the headless error-deliverables re-emit at CLI finalize.

``reemit_error_deliverables`` re-keys the durable ``qc/curation_labels.parquet``
onto the fresh clean master (the SAME frame the GUI's ``CurationLabels`` loads),
re-writes ``deliverables/errors/*.parquet`` + the labels parquet (no mirror),
and computes ``deliverables/error_analysis.{parquet,csv,html}`` across every
labeled category. It is a no-op without a durable labels store and never writes
``verified.parquet`` or rewrites ``measurements.parquet``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import phenotypic.sdk_ as tools_
from phenotypic._cli._cli_error_outputs import reemit_error_deliverables
from phenotypic.schema import IMAGE

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="CLI error outputs use POSIX atomic writes",
)


def _master_df(n_good: int = 30, n_err: int = 10) -> pl.DataFrame:
    """A clean master: ``n_good`` small-area + ``n_err`` clearly-larger objects.

    The error objects (the LAST ``n_err`` labels) have a well-separated
    ``Size_Area`` so ``ErrorCutoffFinder`` ranks it top for the error class.
    """
    rng = np.random.default_rng(0)
    n = n_good + n_err
    labels = list(range(1, n + 1))
    good_area = rng.normal(100.0, 5.0, n_good)
    err_area = rng.normal(500.0, 5.0, n_err)
    area = np.concatenate([good_area, err_area])
    return pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): ["plateA"] * n,
            "Metadata_Dataset": ["ds1"] * n,
            "Object_Label": labels,
            "Bbox_CenterRR": [10.0 * i for i in labels],
            "Bbox_CenterCC": [20.0 * i for i in labels],
            "Size_Area": area.tolist(),
            "Shape_Circularity": rng.normal(0.8, 0.05, n).tolist(),
        }
    )


def _write_labels_parquet(
    output_dir: Path, master_df: pl.DataFrame, error_labels: list[int], category: str
) -> None:
    """Stage a durable ``qc/curation_labels.parquet`` labeling ``error_labels``."""
    rows = master_df.filter(pl.col("Object_Label").is_in(error_labels))
    labels = pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): rows.get_column(str(IMAGE.IMAGE_NAME)).to_list(),
            "Object_Label": rows.get_column("Object_Label").to_list(),
            "Curation_Category": [category] * rows.height,
            "Bbox_CenterRR": rows.get_column("Bbox_CenterRR").to_list(),
            "Bbox_CenterCC": rows.get_column("Bbox_CenterCC").to_list(),
        },
        schema={
            str(IMAGE.IMAGE_NAME): pl.String,
            "Object_Label": pl.Int64,
            "Curation_Category": pl.String,
            "Bbox_CenterRR": pl.Float64,
            "Bbox_CenterCC": pl.Float64,
        },
    )
    path = tools_.curation_labels_parquet_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.write_parquet(path)


def test_happy_path_writes_error_deliverables(tmp_path: Path) -> None:
    master = _master_df()
    err_labels = list(range(31, 41))  # the 10 large-area objects
    _write_labels_parquet(tmp_path, master, err_labels, "debris")

    reemit_error_deliverables(tmp_path, master)

    # Per-category error parquet exists + carries Curation_Category.
    debris = pl.read_parquet(tools_.error_category_parquet_path(tmp_path, "debris"))
    assert debris.height == 10
    assert debris.get_column("Curation_Category").unique().to_list() == ["debris"]

    # error_analysis.parquet exists, first column is ``category``, Size_Area ranks
    # top for debris.
    ea = pl.read_parquet(tools_.error_analysis_parquet_path(tmp_path))
    assert ea.columns[0] == "category"
    debris_rows = ea.filter(pl.col("category") == "debris")
    assert debris_rows.height > 0
    assert debris_rows.row(0, named=True)["measurement"] == "Size_Area"

    # csv + html exist.
    assert tools_.error_analysis_csv_path(tmp_path).exists()
    assert tools_.error_analysis_html_path(tmp_path).exists()
    html = tools_.error_analysis_html_path(tmp_path).read_text(encoding="utf-8")
    assert "debris" in html and "Size_Area" in html

    # Labels store preserved; verified.parquet never written.
    assert tools_.curation_labels_parquet_path(tmp_path).exists()
    assert not tools_.verified_parquet_path(tmp_path).exists()


def test_guard_no_op_without_labels_store(tmp_path: Path) -> None:
    master = _master_df()
    # No curation_labels.parquet staged.
    reemit_error_deliverables(tmp_path, master)

    assert not tools_.errors_dir(tmp_path).exists()
    assert not tools_.error_analysis_parquet_path(tmp_path).exists()
    assert not tools_.measurements_parquet_path(tmp_path).exists()


def test_no_op_when_labels_store_empty(tmp_path: Path) -> None:
    """A present-but-empty labels store re-keys to no labels → no analysis."""
    master = _master_df()
    empty = pl.DataFrame(
        {
            str(IMAGE.IMAGE_NAME): [],
            "Object_Label": [],
            "Curation_Category": [],
            "Bbox_CenterRR": [],
            "Bbox_CenterCC": [],
        },
        schema={
            str(IMAGE.IMAGE_NAME): pl.String,
            "Object_Label": pl.Int64,
            "Curation_Category": pl.String,
            "Bbox_CenterRR": pl.Float64,
            "Bbox_CenterCC": pl.Float64,
        },
    )
    path = tools_.curation_labels_parquet_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    empty.write_parquet(path)

    reemit_error_deliverables(tmp_path, master)
    assert not tools_.error_analysis_parquet_path(tmp_path).exists()


def test_idempotent_and_prunes_stale_category(tmp_path: Path) -> None:
    master = _master_df()
    err_labels = list(range(31, 41))
    _write_labels_parquet(tmp_path, master, err_labels, "debris")

    reemit_error_deliverables(tmp_path, master)
    reemit_error_deliverables(tmp_path, master)  # twice → same files
    assert pl.read_parquet(
        tools_.error_category_parquet_path(tmp_path, "debris")
    ).height == 10

    # Re-label the SAME objects under a new category; the old debris parquet must
    # be pruned on the next re-emit.
    _write_labels_parquet(tmp_path, master, err_labels, "background_noise")
    reemit_error_deliverables(tmp_path, master)

    assert not tools_.error_category_parquet_path(tmp_path, "debris").exists()
    assert tools_.error_category_parquet_path(tmp_path, "background_noise").exists()
    ea = pl.read_parquet(tools_.error_analysis_parquet_path(tmp_path))
    assert set(ea.get_column("category").to_list()) == {"background_noise"}


def test_headless_error_services_do_not_import_dash() -> None:
    """CLI finalization imports shared Error services without GUI extras."""
    script = textwrap.dedent(
        """
        import importlib.abc
        import importlib.util
        import sys

        real_find_spec = importlib.util.find_spec
        def find_spec_without_dash(name, *args, **kwargs):
            if name == "dash" or name.startswith("dash."):
                return None
            return real_find_spec(name, *args, **kwargs)
        importlib.util.find_spec = find_spec_without_dash

        class BlockDash(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "dash" or fullname.startswith("dash."):
                    raise ImportError("Dash intentionally unavailable")
                return None

        sys.meta_path.insert(0, BlockDash())
        import phenotypic._cli._cli_error_outputs
        from phenotypic.gui.results_viewer._curation_labels import CurationLabels
        from phenotypic.gui.results_viewer._error_tab._publication import (
            compute_all_category_analysis,
        )

        assert CurationLabels is not None
        assert compute_all_category_analysis is not None
        assert not any(
            name == "dash" or name.startswith("dash.")
            for name in sys.modules
        )
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
