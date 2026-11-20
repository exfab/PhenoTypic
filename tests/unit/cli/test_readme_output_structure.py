"""The README's output-structure section (#281 P2 review M11).

The tree is a fixed-width block a user reads in a terminal or a Markdown
viewer, so every annotated line puts its ``#`` in one column. The paragraph
under it says which runs rewrite ``metrics.json`` and ``schema.json``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from phenotypic import ImagePipeline
from phenotypic._cli._cli_readme_generator import READMEGenerator
from phenotypic._cli._cli_types import Dataset


def _structure() -> str:
    config = SimpleNamespace()
    generator = READMEGenerator(config=config, pipeline=ImagePipeline())  # type: ignore[arg-type]
    dataset = Dataset(
        name="ds", images=[], input_dir=Path("in"), output_dir=Path("out")
    )
    return generator._generate_output_structure([dataset])


def test_every_tree_comment_starts_in_one_column() -> None:
    block = _structure().split("```")[1]
    columns = {
        line.split("+-- ", 1)[1].split()[0]: line.index("# ")
        for line in block.splitlines()
        if "+-- " in line and "# " in line
    }
    assert "pipeline.pht-pipe.json" in columns, sorted(columns)
    assert len(set(columns.values())) == 1, columns


def test_the_run_summaries_paragraph_names_the_runs_that_rewrite_them() -> (
    None
):
    """Only ``finalize_run`` writes them; ``--mode migrate``'s re-publication does not."""
    text = " ".join(_structure().split())
    assert "rewritten by every finalization" not in text
    assert "--mode migrate` does not write them" in text
