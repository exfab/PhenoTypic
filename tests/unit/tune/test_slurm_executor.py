"""SlurmExecutor + the tune worker — script generation, NO live SLURM (H1).

Mirrors the forward-CLI script-gen tests: we generate the array worker script
and the drip-feed chain and assert on their contents, mock the sbatch submission,
and exercise the dead-worker re-enqueue. The worker binds to the shared study by
name + a plain Optuna storage URL (SQLite-WAL or a password-less
``postgresql+psycopg://…`` — libpq resolves the secret from ``~/.pgpass``).
"""
from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

pytestmark = [
    pytest.mark.skipif(
        sys.platform == "win32", reason="SLURM not available on Windows"
    ),
]

from phenotypic._execution import Executor  # noqa: E402
from phenotypic._execution._slurm import SlurmExecutor  # noqa: E402
from phenotypic.sdk_ import logs_dir, slurm_scripts_dir  # noqa: E402


def _executor(tmp_path, *, n_workers=4, storage_url=None):
    return SlurmExecutor(
        output_dir=tmp_path,
        spec_path=tmp_path / "tuning_spec.json",
        images_dir=tmp_path / "images",
        split_path=tmp_path / ".pht-tune-cache" / "splits" / "split.json",
        study_name="study0",
        storage_url=storage_url or f"sqlite:///{tmp_path / 'study.db'}",
        n_workers=n_workers,
        slurm_args={"slurm_partition": "short", "mem_gb": 8, "time": 120},
    )


# --- array worker script generation -------------------------------------------


def test_worker_array_script_sizes_array_to_worker_count(tmp_path):
    ex = _executor(tmp_path, n_workers=5)
    script = ex.generate_worker_array_script()
    content = script.read_text()
    assert script.parent == slurm_scripts_dir(tmp_path)
    # One array task = one worker; sized 0..n-1 (NOT image-chunk sized).
    assert "#SBATCH --array=0-4" in content


def test_worker_array_script_invokes_the_tune_worker_module(tmp_path):
    ex = _executor(tmp_path)
    content = ex.generate_worker_array_script().read_text()
    # The body launches the tune worker — never the forward single-image worker.
    assert "phenotypic.tune._tune_cli._worker" in content
    assert "phenotypic._cli._cli_process_single" not in content


def test_worker_array_script_has_no_image_chunk_sentinels(tmp_path):
    ex = _executor(tmp_path)
    content = ex.generate_worker_array_script().read_text()
    # Fresh tune body — none of the forward CLI's checkpoint/manifest/finalizer
    # sentinels appear.
    assert "CHECKPOINT" not in content
    assert "MANIFEST" not in content
    assert "FINALIZER" not in content


def test_worker_array_script_binds_to_shared_study_name_and_url(tmp_path):
    url = f"sqlite:///{tmp_path / 'study.db'}"
    ex = _executor(tmp_path, storage_url=url)
    content = ex.generate_worker_array_script().read_text()
    assert "--study-name" in content
    assert "study0" in content
    assert "--storage-url" in content
    assert url in content
    assert "--split" in content
    assert "split.json" in content


def test_worker_array_script_carries_sbatch_directives(tmp_path):
    ex = _executor(tmp_path)
    content = ex.generate_worker_array_script().read_text()
    assert content.startswith("#!/bin/bash")
    assert "--partition=short" in content
    assert "--mem=8G" in content
    assert str(logs_dir(tmp_path) / "slurm" / "tune_worker_%A_%a.log") in content


# --- Executor protocol --------------------------------------------------------


def test_satisfies_executor_protocol_by_calling(tmp_path):
    ex = _executor(tmp_path, n_workers=3)
    assert isinstance(ex, Executor)
    # run() submits the worker fleet; mock the drip-feed start (no live SLURM).
    with patch(
        "phenotypic._execution._slurm.submit_drip_feed_start",
        return_value=(["1001"], None),
    ) as submit:
        job_ids = ex.run(lambda w: w, list(range(3)))
    submit.assert_called_once()
    assert job_ids == ["1001"]


# --- dead-worker re-enqueue ---------------------------------------------------


def test_dead_worker_re_enqueued_once(tmp_path):
    ex = _executor(tmp_path, n_workers=4)
    with patch(
        "phenotypic._execution._slurm.submit_script", return_value="2002"
    ) as submit:
        first = ex.reenqueue_dead_worker(worker_index=2)
        assert first == "2002"
        submit.assert_called_once()
        assert submit.call_args.kwargs["array_index"] == 2
        # A second re-enqueue of the same worker is a no-op (re-enqueue once).
        second = ex.reenqueue_dead_worker(worker_index=2)
    assert second is None


def test_dead_worker_reenqueue_rejects_out_of_range_index(tmp_path):
    ex = _executor(tmp_path, n_workers=4)
    with pytest.raises(ValueError, match="worker_index"):
        ex.reenqueue_dead_worker(worker_index=4)


# --- worker entry binds to the shared study -----------------------------------


def test_worker_builds_optuna_store_bound_to_name_and_url(tmp_path):
    pytest.importorskip("optuna")
    from phenotypic.tune._tune_cli._worker import build_worker_store

    url = f"sqlite:///{tmp_path / 'study.db'}"
    store = build_worker_store(storage_url=url, study_name="bound")
    # The worker store is the resumable Optuna-backed store on the shared DB.
    assert store.is_resumable_in_place() is True
    from phenotypic.tune._study._optuna_store import OptunaStudyStore

    assert isinstance(store, OptunaStudyStore)
