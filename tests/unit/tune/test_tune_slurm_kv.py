"""``python -m phenotypic.tune run --slurm key=value`` reaches the fleet.

The tune CLI used to accept only four discrete SLURM flags — ``--slurm-partition``,
``--slurm-mem``, ``--slurm-time``, ``--slurm-constraint`` — so ``account``, ``qos``,
``cpus_per_task`` and ``gpus_per_node`` could not reach a tune fleet at all. On UCR
HPCC that is not cosmetic: ``--account`` is *mandatory* for the ``exfab`` and
``preempt`` partitions, so no tune fleet could reach the GPU node or the preempt
pool, and everywhere else the work was silently billed to the default account.

Both engines already funnel into ``format_sbatch_directives``, which handles
arbitrary keys; the narrowing was purely in this CLI's argument surface.

This CLI is **argparse**, not Click — there is no ``cli`` object and no
``CliRunner``. Everything below drives ``main([...])`` or
``_build_parser().parse_args(...)``.
"""
from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from phenotypic import ImagePipeline
from phenotypic.analysis import ExpectedVsDetectedCount
from phenotypic.detect import OtsuDetector
from phenotypic.tune import Budget, Categorical, Evaluator, Knob, SearchSpace
from phenotypic.tune.score import QCScorer
from phenotypic.tune.strategy import GridConfig
from phenotypic.tune._spec import TuningSpec
from phenotypic.tune.__main__ import _build_parser, main
from phenotypic.tune._tune_cli import _run as run_mod
from phenotypic.tune._tune_cli._run import merge_slurm_args

_OPTUNA = importlib.util.find_spec("optuna") is not None


def _spec(tmp_path) -> TuningSpec:
    """A registry-resolvable spec: the CLI reloads it from JSON, so every
    component has to round-trip through the class registry."""
    layout = tmp_path / "layout.csv"
    pd.DataFrame(
        {"Metadata_ImageName": ["Synthetic96PlateWithObjects"], "Object_Label": [1]}
    ).to_csv(layout, index=False)
    return TuningSpec(
        pipeline=ImagePipeline(ops=[OtsuDetector()]),
        search_space=SearchSpace(knobs=(
            Knob(key="0.ignore_zeros", domain=Categorical(choices=(True, False))),
        )),
        scorer=QCScorer(check=ExpectedVsDetectedCount(
            metadata=str(layout), groupby=["Metadata_ImageName"],
        )),
        evaluator=Evaluator(),
        strategy=GridConfig(),
        budget=Budget(),
    )


@pytest.fixture
def cli_sandbox(tmp_path, monkeypatch):
    """A spec on disk, a stubbed image scan, and a stubbed ``_submit_slurm_fleet``.

    ``_load_images`` is replaced so the CLI's ``no images found`` guard does not
    fire without writing plates to disk (the on-disk scan is covered elsewhere),
    and the submission itself is replaced, so nothing here reaches ``sbatch``.
    Returns the kwargs ``_submit_slurm_fleet`` was called with.
    """
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(_spec(tmp_path).model_dump_json())
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()

    from phenotypic.data import load_synth_yeast_plate

    monkeypatch.setattr(
        run_mod, "_load_images", lambda *a, **kw: [load_synth_yeast_plate()]
    )
    monkeypatch.setattr(
        "phenotypic.tune.__main__._load_images",
        lambda *a, **kw: [load_synth_yeast_plate()],
    )

    captured: dict = {}

    def _fake_submit(spec, output_dir, **kwargs):
        captured.update(kwargs)
        captured["output_dir"] = output_dir
        return None

    monkeypatch.setattr(run_mod, "_submit_slurm_fleet", _fake_submit)

    return spec_path, images_dir, tmp_path / "out", captured


def _argv(spec_path, images_dir, out, *extra: str) -> list[str]:
    return ["run", str(spec_path), "-i", str(images_dir), "-o", str(out), *extra]


# --- the CLI surface ---------------------------------------------------------


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_account_reaches_the_fleet(cli_sandbox):
    """The whole point: exfab and preempt are unreachable without --account."""
    spec_path, images_dir, out, captured = cli_sandbox

    main(_argv(
        spec_path, images_dir, out,
        "--strategy", "tpe", "--n-trials", "4",
        "--slurm", "slurm_account=exfab",
        "--slurm", "slurm_partition=exfab",
        "--slurm", "slurm_cpus_per_task=8",
    ))

    assert captured["slurm_args"]["slurm_account"] == "exfab"
    assert captured["slurm_args"]["slurm_partition"] == "exfab"
    # ast.literal_eval in parse_slurm_args gives a real int, not "8".
    assert captured["slurm_args"]["slurm_cpus_per_task"] == 8


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_legacy_flags_still_work(cli_sandbox):
    """The four sugar flags keep their shipped behaviour."""
    spec_path, images_dir, out, captured = cli_sandbox

    main(_argv(
        spec_path, images_dir, out,
        "--strategy", "tpe", "--n-trials", "4", "--slurm",
        "--slurm-partition", "batch", "--slurm-mem", "16G",
        "--slurm-time", "08:00:00", "--slurm-constraint", "avx2",
    ))

    assert captured["slurm_args"] == {
        "slurm_partition": "batch",
        "slurm_mem": "16G",
        "slurm_time": "08:00:00",
        "slurm_constraint": "avx2",
    }


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_a_bare_slurm_still_means_submit(cli_sandbox):
    """It shipped as a boolean; scripts pass it with no value and must keep working."""
    spec_path, images_dir, out, captured = cli_sandbox

    main(_argv(spec_path, images_dir, out, "--strategy", "tpe", "--n-trials", "4",
               "--slurm"))

    assert captured["output_dir"] == out, "the fleet path was not taken"
    assert captured["slurm_args"] == {}


def test_no_slurm_flag_runs_locally(cli_sandbox, monkeypatch):
    """Absence of --slurm must not be read as an empty passthrough profile."""
    spec_path, images_dir, out, captured = cli_sandbox

    class _FakeEngine:
        def __init__(self, spec, store):
            pass

        def optimize(self, images):
            return None

    monkeypatch.setattr(run_mod, "TuningEngine", _FakeEngine)

    main(_argv(spec_path, images_dir, out))

    assert captured == {}, "a local run must not submit a fleet"


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_explicit_kv_wins_over_the_sugar_flag(cli_sandbox):
    """Both spellings present: the specific one wins, and only one survives."""
    spec_path, images_dir, out, captured = cli_sandbox

    main(_argv(
        spec_path, images_dir, out,
        "--strategy", "tpe", "--n-trials", "4",
        "--slurm-partition", "batch",
        "--slurm", "slurm_partition=epyc",
    ))

    assert captured["slurm_args"] == {"slurm_partition": "epyc"}


@pytest.mark.skipif(not _OPTUNA, reason="optuna extra not installed")
def test_the_unprefixed_spelling_also_wins_over_the_sugar_flag(cli_sandbox):
    """``partition=`` and ``slurm_partition=`` render the SAME #SBATCH directive.

    ``format_sbatch_directives`` strips a leading ``slurm_``, so leaving the two
    spellings as separate dict keys would emit ``#SBATCH --partition`` twice with
    contradictory values. ``--slurm partition=`` is also exactly what the forward
    CLI's own emitter (``slurm_argv_extension``) produces, so this is the spelling
    a copied command line arrives in.
    """
    spec_path, images_dir, out, captured = cli_sandbox

    main(_argv(
        spec_path, images_dir, out,
        "--strategy", "tpe", "--n-trials", "4",
        "--slurm-partition", "batch",
        "--slurm", "partition=epyc",
    ))

    assert captured["slurm_args"] == {"slurm_partition": "epyc"}


def test_a_malformed_pair_is_a_usage_error_not_a_traceback(cli_sandbox, capsys):
    """``parse_slurm_args`` raises click.BadParameter; argparse must own the exit."""
    spec_path, images_dir, out, _ = cli_sandbox

    with pytest.raises(SystemExit) as excinfo:
        main(_argv(spec_path, images_dir, out, "--slurm", "no-equals-sign"))

    assert excinfo.value.code == 2
    assert "KEY=VALUE" in capsys.readouterr().err


def test_slurm_parses_as_a_repeatable_optional_value():
    """The parser shape itself: append + nargs='?' is what allows both spellings."""
    parser = _build_parser()

    bare = parser.parse_args(["run", "s.json", "-i", "in", "--slurm"])
    assert bare.slurm == [None]

    paired = parser.parse_args(
        ["run", "s.json", "-i", "in", "--slurm", "a=1", "--slurm", "b=2"]
    )
    assert paired.slurm == ["a=1", "b=2"]

    absent = parser.parse_args(["run", "s.json", "-i", "in"])
    assert absent.slurm is None


# --- the merge itself --------------------------------------------------------


def test_merge_omits_unset_sugar_flags():
    """An unset flag emits NO key, so no directive and the cluster default holds."""
    assert merge_slurm_args(
        None, partition=None, mem=None, time=None, constraint=None
    ) == {}


def test_merge_keeps_keys_the_sugar_flags_cannot_express():
    merged = merge_slurm_args(
        {"slurm_account": "exfab", "slurm_qos": "normal", "slurm_gpus_per_node": 1},
        partition="exfab", mem=None, time=None, constraint=None,
    )
    assert merged == {
        "slurm_partition": "exfab",
        "slurm_account": "exfab",
        "slurm_qos": "normal",
        "slurm_gpus_per_node": 1,
    }


def test_merge_leaves_non_sugar_keys_unprefixed():
    """``mem_gb`` is a distinct special case in format_sbatch_directives.

    Folding it to ``slurm_mem_gb`` would render ``#SBATCH --mem-gb=8`` instead of
    ``--mem=8G``, so canonicalization is deliberately limited to the four names
    that have a sugar flag.
    """
    merged = merge_slurm_args(
        {"mem_gb": 8}, partition=None, mem=None, time=None, constraint=None
    )
    assert merged == {"mem_gb": 8}
