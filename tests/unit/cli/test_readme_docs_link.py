"""README footer links the live docs home and names the writing version."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import phenotypic
from phenotypic import ImagePipeline
from phenotypic._cli._cli_readme_generator import READMEGenerator
from phenotypic.measure import MeasureSize

_SRC = Path(phenotypic.__file__).resolve().parent


def _readme(tmp_path: Path, **config) -> str:
    """The README a ``MeasureSize`` pipeline writes, with *config* over a plain-Image run."""
    # generate() reads config.image_type and config.pipeline_json.name.
    values = {"image_type": "Image", "pipeline_json": Path("pipeline.json"), **config}
    generator = READMEGenerator(
        config=SimpleNamespace(**values),
        pipeline=ImagePipeline(meas=[MeasureSize()]),
    )
    return generator.generate(tmp_path, []).read_text(encoding="utf-8")


def test_readme_links_the_docs_home_and_names_the_version(tmp_path) -> None:
    text = _readme(tmp_path)
    assert "https://exfab.github.io/PhenoTypic/" in text
    last_line = text.rstrip().splitlines()[-1]
    assert f"v{phenotypic.__version__}" in last_line
    assert "nearest earlier one listed" in last_line


def test_no_source_file_links_the_dead_docs_host() -> None:
    hits = [
        p
        for p in _SRC.rglob("*.py")
        if "wheeldon-lab.github.io" in p.read_text(encoding="utf-8")
    ]
    assert hits == []


def test_grid_dimensions_use_the_shape_the_run_used(tmp_path) -> None:
    text = _readme(tmp_path, image_type="GridImage", nrows=None, ncols=None)
    line = next(ln for ln in text.splitlines() if ln.startswith("Grid Dimensions"))
    assert "None" not in line
    assert "8" in line and "12" in line


def test_measure_mode_readme_says_the_grid_is_stored_per_image(tmp_path) -> None:
    text = _readme(
        tmp_path, image_type="GridImage", nrows=None, ncols=None, measure_only=True
    )
    assert "Grid Dimensions: as stored in each image store" in text
    assert "rows x" not in text
