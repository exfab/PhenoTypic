"""``deploy_plan`` previews an sbatch script; a preview that writes is not a preview.

Phase 2C renders the script a run *would* submit before anything is approved.
``generate_array_job_script`` cannot serve that: it creates
``<output_dir>/.phenotypic/slurm_scripts/`` and ``<output_dir>/logs/`` and writes
an executable script, which would then trip ``deploy_start``'s own
``output_not_empty`` guard on the directory the preview only claimed to read.
``build_array_script_spec`` is the pure half.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict


def _tree_digest(root: Path) -> str:
    """Digest every path under ``root`` plus the bytes of every file."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        h.update(str(p.relative_to(root)).encode())
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def test_build_array_script_spec_writes_nothing(
    tmp_path: Path, array_script_kwargs: Dict[str, Any]
) -> None:
    """Building the spec must leave the output tree byte-identical."""
    from phenotypic._cli._cli_slurm_array_scripts import build_array_script_spec

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    before = _tree_digest(output_dir)

    spec = build_array_script_spec(output_dir=output_dir, **array_script_kwargs)

    assert _tree_digest(output_dir) == before, "the builder touched the output dir"
    assert spec.render(), "the spec must still render a script"


def test_generator_and_builder_agree(
    tmp_path: Path, array_script_kwargs: Dict[str, Any]
) -> None:
    """The real generator must consume the extracted builder, not duplicate it.

    Both calls use the SAME ``output_dir``. The spec embeds ``output_dir``-derived
    absolute paths -- ``log_dir = logs_dir(output_dir)/"slurm"/dataset.name`` and
    ``log_path = log_dir/f"{dataset.name}_%A_%a.log"`` -- so rendering from two
    different directories produces two different ``#SBATCH --output`` lines and
    the comparison could never pass. Take the builder's render FIRST, while the
    directory is still untouched, then let the generator write into it.
    """
    from phenotypic._cli._cli_slurm_array_scripts import (
        build_array_script_spec,
        generate_array_job_script,
    )

    output_dir = tmp_path / "run"
    output_dir.mkdir()

    previewed = build_array_script_spec(
        output_dir=output_dir, **array_script_kwargs
    ).render()
    written = Path(
        generate_array_job_script(output_dir=output_dir, **array_script_kwargs)
    )

    assert written.read_text() == previewed


def test_generator_still_writes_the_script(
    tmp_path: Path, array_script_kwargs: Dict[str, Any]
) -> None:
    """Guards the other direction: the split must not have made the writer pure."""
    from phenotypic._cli._cli_slurm_array_scripts import generate_array_job_script

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    before = _tree_digest(output_dir)

    written = Path(
        generate_array_job_script(output_dir=output_dir, **array_script_kwargs)
    )

    assert written.is_file()
    assert _tree_digest(output_dir) != before
