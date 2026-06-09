import pytest

from phenotypic.gui.tune._run_argv import tune_run_argv


def test_minimal_local_argv():
    argv = tune_run_argv(
        spec_path="/sbx/spec.json.pht-tune",
        images_dir="/data/imgs",
        output_dir="/out/run1",
        strategy="tpe",
        n_trials=50,
        storage_url="sqlite:///out/run1/.pht-tune-cache/study.db",
        n_workers=8,
        slurm_partition=None,
        slurm_mem=None,
        slurm_time=None,
        held_out_fraction=0.2,
        cv_group="plate_id",
        slurm=False,
        screen=False,
        python="python",
    )
    assert argv[:4] == ["python", "-m", "phenotypic.tune", "run"]
    assert "/sbx/spec.json.pht-tune" in argv
    assert argv[argv.index("-i") + 1] == "/data/imgs"
    assert argv[argv.index("-o") + 1] == "/out/run1"
    assert argv[argv.index("--strategy") + 1] == "tpe"
    assert argv[argv.index("--n-trials") + 1] == "50"
    assert argv[argv.index("--n-workers") + 1] == "8"
    assert argv[argv.index("--held-out-fraction") + 1] == "0.2"
    assert argv[argv.index("--cv-group") + 1] == "plate_id"
    assert "--slurm" not in argv


def test_grid_omits_n_trials_and_slurm_flag_present():
    argv = tune_run_argv(
        spec_path="s",
        images_dir="i",
        output_dir="o",
        strategy="grid",
        n_trials=50,
        storage_url=None,
        n_workers=None,
        slurm_partition="batch",
        slurm_mem="8G",
        slurm_time="04:00:00",
        held_out_fraction=None,
        cv_group=None,
        slurm=True,
        screen=True,
        python="python",
    )
    assert "--n-trials" not in argv
    assert "--slurm" in argv
    assert "--screen" in argv
    assert argv[argv.index("--slurm-partition") + 1] == "batch"
    assert argv[argv.index("--slurm-mem") + 1] == "8G"
    assert argv[argv.index("--slurm-time") + 1] == "04:00:00"
    assert "--storage-url" not in argv


def test_missing_required_slot_raises():
    with pytest.raises(ValueError, match="spec_path"):
        tune_run_argv(
            spec_path="",
            images_dir="i",
            output_dir="o",
            strategy="tpe",
            n_trials=None,
            storage_url=None,
            n_workers=None,
            slurm_partition=None,
            slurm_mem=None,
            slurm_time=None,
            held_out_fraction=None,
            cv_group=None,
            slurm=False,
            screen=False,
            python="python",
        )
