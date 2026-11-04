"""README footer links the live docs home and names the writing version."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import phenotypic
from phenotypic import ImagePipeline
from phenotypic._cli._cli_readme_generator import READMEGenerator
from phenotypic.measure import MeasureSize

_SRC = Path(phenotypic.__file__).resolve().parent


def _generator(*measurers) -> READMEGenerator:
    # generate() reads config.image_type and config.pipeline_json.name.
    return READMEGenerator(
        config=SimpleNamespace(image_type="Image", pipeline_json=Path("pipeline.json")),
        pipeline=ImagePipeline(meas=list(measurers)),
    )


def test_readme_links_the_docs_home_and_names_the_version(tmp_path) -> None:
    text = _generator(MeasureSize()).generate(tmp_path, []).read_text(encoding="utf-8")
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
    gen = READMEGenerator(
        config=SimpleNamespace(
            image_type="GridImage",
            pipeline_json=Path("pipeline.json"),
            nrows=None,
            ncols=None,
        ),
        pipeline=ImagePipeline(meas=[MeasureSize()]),
    )
    text = gen.generate(tmp_path, []).read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if ln.startswith("Grid Dimensions"))
    assert "None" not in line
    assert "8" in line and "12" in line
