"""Every store-writing site must actually honour the run's durability mode.

Spec §3.7 / Phase 3 Task 3.7. ``grep -n 'save_image_store' src`` finds exactly
**three** write sites:

* ``_cli_process_single.py`` -- the single-pass CPU/SLURM per-image worker,
* ``_cli_staged_workers.py`` Stage 1 -- publishes the staged store,
* ``_cli_staged_workers.py`` Stage 3 -- **re-promotes** it after post-ops.

The characteristic failure of a flag task is a flag that is threaded but inert:
it reaches three of four writers, or reaches all of them and changes nothing.
So these tests do not assert that an argument was passed. They drive each site
for real and capture ``fsync`` at the bottom of the chain --
``ngff_.promote_store``, the single function that decides whether the store is
flushed before the rename. Nothing between the OutputManager and that call can
swallow the value without one of these turning red.

Each site is exercised in three states, which between them catch a hard-coded
value in **either** direction at that site:

======================  ==========  ==========  ====================================
manager.durable_writes  SLURM_JOB_  fsync       what a mutation would have to survive
======================  ==========  ==========  ====================================
``True``                unset       ``True``    pinning the site to ``False``
``False``               set         ``False``   pinning the site to ``True``
``None``                set         ``True``    pinning the site to ``False``
======================  ==========  ==========  ====================================

The middle row is the one that matters most: it is the only row where an
inert site would be rescued by the environment rather than exposed by it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from phenotypic import ImagePipeline
from phenotypic._cli._cli_output_manager import OutputManager
from phenotypic._cli._cli_types import Dataset
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSize
from phenotypic.sdk_ import CommitGuard

#: ``(durable_writes, on_slurm, expected_fsync)``. See the module docstring.
DURABILITY_CASES = [
    pytest.param(True, False, True, id="override-on-beats-a-local-run"),
    pytest.param(False, True, False, id="override-off-beats-slurm"),
    pytest.param(None, True, True, id="unset-still-auto-detects-slurm"),
]


@pytest.fixture
def fsync_calls(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Record the ``fsync`` argument of every ``promote_store`` call."""
    from phenotypic.sdk_ import ngff_

    seen: list[bool] = []
    real = ngff_.promote_store

    def _spy(
        part: Path,
        final: Path,
        *,
        fsync: bool = False,
        commit_guard: CommitGuard | None = None,
    ) -> Path:
        seen.append(fsync)
        return real(part, final, fsync=fsync, commit_guard=commit_guard)

    monkeypatch.setattr(ngff_, "promote_store", _spy)
    return seen


def _set_slurm(monkeypatch: pytest.MonkeyPatch, on_slurm: bool) -> None:
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    if on_slurm:
        monkeypatch.setenv("SLURM_JOB_ID", "12345")
    else:
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)


def _manager(output_dir: Path, durable_writes: bool | None) -> OutputManager:
    manager = OutputManager.from_config(
        output_dir,
        ".tiff",
        save_overlays=False,
        durable_writes=durable_writes,
    )
    manager.create_structure(
        [Dataset("ds", [], output_dir, output_dir)]
    )
    return manager


# ---------------------------------------------------------------------------
# Site 1: the single-pass per-image worker (_cli_process_single.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def single_pass_run(tmp_path: Path) -> dict[str, Any]:
    image_path = tmp_path / "img.tiff"
    load_synth_yeast_plate().rgb.imsave(filepath=image_path)
    pipeline_path = tmp_path / "pipe.json"
    pipeline_path.write_text(
        ImagePipeline(ops=[OtsuDetector()], meas=[MeasureSize()]).to_json()
    )
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    return {
        "image_path": image_path,
        "pipeline_path": pipeline_path,
        "output_dir": output_dir,
    }


@pytest.mark.parametrize(
    ("durable_writes", "on_slurm", "expected"), DURABILITY_CASES
)
def test_process_single_image_core_honours_the_mode(
    monkeypatch: pytest.MonkeyPatch,
    fsync_calls: list[bool],
    single_pass_run: dict[str, Any],
    durable_writes: bool | None,
    on_slurm: bool,
    expected: bool,
) -> None:
    from phenotypic._cli._cli_process_single import process_single_image_core

    _set_slurm(monkeypatch, on_slurm)
    manager = _manager(single_pass_run["output_dir"], durable_writes)

    process_single_image_core(
        pipeline_path=single_pass_run["pipeline_path"],
        image_path=single_pass_run["image_path"],
        output_dir=single_pass_run["output_dir"],
        dataset_name="ds",
        image_type="Image",
        read_kwargs={},
        output_manager=manager,
    )

    # Full-forward publishes a decoded-source checkpoint before processing and
    # then atomically promotes the completed processed store.
    assert fsync_calls == [expected, expected]


# ---------------------------------------------------------------------------
# Sites 2 and 3: the staged workers (_cli_staged_workers.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("durable_writes", "on_slurm", "expected"), DURABILITY_CASES
)
def test_stage1_preprocess_core_honours_the_mode(
    monkeypatch: pytest.MonkeyPatch,
    fsync_calls: list[bool],
    staged_run,
    durable_writes: bool | None,
    on_slurm: bool,
    expected: bool,
) -> None:
    _set_slurm(monkeypatch, on_slurm)
    staged_run.output_manager.durable_writes = durable_writes

    staged_run.run_stage1()

    # Stage 1 publishes the decoded-source checkpoint and then the staged
    # pre-operation result.
    assert fsync_calls == [expected, expected]


@pytest.mark.parametrize(
    ("durable_writes", "on_slurm", "expected"), DURABILITY_CASES
)
def test_stage3_re_promote_honours_the_mode(
    monkeypatch: pytest.MonkeyPatch,
    fsync_calls: list[bool],
    staged_run,
    durable_writes: bool | None,
    on_slurm: bool,
    expected: bool,
) -> None:
    """Stage 3 is a separate write site from Stage 1 and fails separately.

    Stage 1 runs under the *default* mode here, so the assertion is about the
    LAST promote only -- pin Stage 3 to a constant and this goes red while the
    Stage-1 test above stays green, which is exactly the "reaches three of
    four writers" failure this task exists to rule out.
    """
    staged_run.run_stage1()
    staged_run.run_stage2()
    assert len(fsync_calls) == 2, "Stage 1 publishes checkpoint then staged store"

    _set_slurm(monkeypatch, on_slurm)
    staged_run.output_manager.durable_writes = durable_writes
    staged_run.run_stage3()

    assert fsync_calls[2:] == [expected]
