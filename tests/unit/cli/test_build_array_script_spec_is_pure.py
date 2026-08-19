"""``deploy_plan`` previews an sbatch script; a preview that writes is not a preview.

Phase 2C renders the script a run *would* submit before anything is approved.
``generate_array_job_script`` cannot serve that: it creates
``<output_dir>/.phenotypic/slurm_scripts/`` and ``<output_dir>/logs/`` and writes
an executable script, which would then trip ``deploy_start``'s own
``output_not_empty`` guard on the directory the preview only claimed to read.
``build_array_script_spec`` is the half that creates nothing.

Two properties of the identity mechanism the generated script carries -- the
per-task work ids, input digests, and attempt ids -- shape the tests below and
are pinned by their own cases:

* every ``ATTEMPT_IDS`` entry is a fresh ``uuid4()``, so no two renders of the
  same chunk are byte-identical (:func:`test_attempt_ids_are_the_only_drift`);
* building a spec **reads** every image in the chunk and the pipeline JSON to
  hash them (:func:`test_building_a_spec_reads_every_input_image`).

Neither touches ``output_dir``, so neither weakens the preview guarantee -- but
a preview caller pays a full read of the chunk and cannot expect the preview to
match the eventual submission byte for byte.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from pathlib import Path
from typing import Any, Dict

import pytest

_ATTEMPT_IDS_BLOCK = re.compile(r"ATTEMPT_IDS=\(\n.*?\n\)", re.DOTALL)


def _tree_digest(root: Path) -> str:
    """Digest every path under ``root`` plus the bytes of every file."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        h.update(str(p.relative_to(root)).encode())
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()


def _mask_attempt_ids(script: str) -> str:
    """Blank the one array whose contents are freshly generated per call."""
    masked, n = _ATTEMPT_IDS_BLOCK.subn("ATTEMPT_IDS=(<masked>)", script)
    assert n == 1, f"expected exactly one ATTEMPT_IDS block, found {n}"
    return masked


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

    ``ATTEMPT_IDS`` is masked because it is regenerated per call
    (:func:`test_attempt_ids_are_the_only_drift` proves that mask is not hiding
    anything else). Every other byte -- SBATCH directives, work ids, input
    digests, dispatch block, prelude -- must match exactly.
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

    assert _mask_attempt_ids(written.read_text()) == _mask_attempt_ids(previewed)


def test_generator_consumes_the_builder(
    tmp_path: Path,
    array_script_kwargs: Dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structural proof of consumption, immune to the attempt-id drift.

    Replace the builder the generator resolves with one that stamps a job name
    nothing else in the module produces. If the generator still built its own
    spec, the stamp would be absent from the script it wrote.
    """
    from phenotypic._cli import _cli_slurm_array_scripts as mod

    real_builder = mod.build_array_script_spec
    captured: list[Any] = []

    def _stamped_builder(*args: Any, **kwargs: Any) -> Any:
        spec = dataclasses.replace(
            real_builder(*args, **kwargs), job_name="pht-STAMPED-BY-THE-TEST"
        )
        captured.append(spec)
        return spec

    monkeypatch.setattr(mod, "build_array_script_spec", _stamped_builder)

    output_dir = tmp_path / "run"
    output_dir.mkdir()
    written = Path(
        mod.generate_array_job_script(output_dir=output_dir, **array_script_kwargs)
    )

    assert len(captured) == 1, "the generator called the builder exactly once"
    assert written.read_text() == captured[0].render()
    assert "#SBATCH --job-name=pht-STAMPED-BY-THE-TEST\n" in written.read_text()


def test_attempt_ids_are_the_only_drift(
    tmp_path: Path, array_script_kwargs: Dict[str, Any]
) -> None:
    """Pin the nondeterminism: fresh ``uuid4()`` attempt ids, nothing else.

    Guards the mask in :func:`test_generator_and_builder_agree`. If a second
    field ever became per-call random, the masked comparison there would go on
    passing while this case fails.
    """
    from phenotypic._cli._cli_slurm_array_scripts import build_array_script_spec

    output_dir = tmp_path / "run"
    output_dir.mkdir()

    first = build_array_script_spec(output_dir=output_dir, **array_script_kwargs).render()
    second = build_array_script_spec(output_dir=output_dir, **array_script_kwargs).render()

    assert first != second, "attempt ids are expected to be regenerated per call"
    assert _mask_attempt_ids(first) == _mask_attempt_ids(second)


def test_building_a_spec_reads_the_inputs_it_hashes(
    tmp_path: Path, array_script_kwargs: Dict[str, Any]
) -> None:
    """Pin the read cost a preview caller inherits from the identity mechanism.

    ``work_id_for_image`` hashes the image and the pipeline JSON, and the
    identity-row loop hashes the image again, so every chunk image is opened.
    Recorded here so a preview that must stay cheap has a place to fail rather
    than a surprise on a big dataset.
    """
    from phenotypic._cli import _cli_failure_tracker
    from phenotypic._cli._cli_slurm_array_scripts import build_array_script_spec

    hashed: list[Path] = []
    real_sha = _cli_failure_tracker.file_sha256

    def _recording_sha(path: Path) -> str:
        hashed.append(Path(path))
        return real_sha(path)

    import phenotypic._cli._cli_slurm_array_scripts as mod

    mod_sha = mod.file_sha256
    assert mod_sha is real_sha

    output_dir = tmp_path / "run"
    output_dir.mkdir()

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(mod, "file_sha256", _recording_sha)
        monkey.setattr(_cli_failure_tracker, "file_sha256", _recording_sha)
        build_array_script_spec(output_dir=output_dir, **array_script_kwargs)
    finally:
        monkey.undo()

    chunk_images = set(array_script_kwargs["dataset"].images)
    assert chunk_images <= set(hashed), "every chunk image is hashed"
    assert array_script_kwargs["config"].pipeline_json in set(hashed)


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
