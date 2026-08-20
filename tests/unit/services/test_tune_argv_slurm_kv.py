"""``tune_run_tail`` can render the free-form ``--slurm key=value`` passthrough.

The tune CLI grew a repeatable ``--slurm key=value`` so ``account`` / ``qos`` /
``cpus_per_task`` can reach a tune fleet (``exfab`` and ``preempt`` are
unreachable without ``--account``). The service-tier argv builder has to be able
to render it, or the MCP server and the GUI can express a profile the CLI accepts
but they cannot spell.

The subtle part is the **coupling to submission**: on this CLI the presence of
``--slurm`` is what requests a fleet, so a rendered pair is not decoration — it
flips a local run into a cluster job. The builder therefore emits pairs only when
``slurm`` is true, and a non-empty profile replaces the bare flag rather than
joining it (two ``--slurm`` occurrences would still parse, but the rendered
command line is what a human copies).
"""

from __future__ import annotations

from phenotypic._services.argv import tune_run_tail

_BASE = dict(
    spec_path="/sbx/spec.json",
    images_dir="/data/imgs",
    output_dir="/out/run1",
    strategy="tpe",
    n_trials=50,
    storage_url=None,
    n_workers=None,
    slurm_partition=None,
    slurm_mem=None,
    slurm_time=None,
    held_out_fraction=None,
    cv_group=None,
    screen=False,
)


def _pairs(tail: list[str]) -> list[str]:
    return [tail[i + 1] for i, token in enumerate(tail) if token == "--slurm"]


def test_account_and_qos_are_renderable() -> None:
    tail = tune_run_tail(
        **_BASE,
        slurm=True,
        slurm_args={"slurm_account": "exfab", "slurm_qos": "normal"},
    )
    assert _pairs(tail) == ["slurm_account=exfab", "slurm_qos=normal"]


def test_a_rendered_profile_replaces_the_bare_flag() -> None:
    """One submission request, not two: no bare ``--slurm`` alongside the pairs."""
    tail = tune_run_tail(**_BASE, slurm=True, slurm_args={"slurm_account": "exfab"})
    assert tail.count("--slurm") == 1
    assert _pairs(tail) == ["slurm_account=exfab"]


def test_no_profile_still_emits_the_bare_flag() -> None:
    """The shipped boolean spelling: ``--slurm`` alone still means submit."""
    tail = tune_run_tail(**_BASE, slurm=True, slurm_args=None)
    assert tail.count("--slurm") == 1
    assert tail[-1] == "--slurm"


def test_a_profile_never_turns_a_local_run_into_a_cluster_job() -> None:
    """``slurm=False`` with a stale profile attached must stay local.

    A caller that keeps a SLURM profile in its form state while toggling the
    execution target back to local is the realistic way this happens; emitting
    the pairs anyway would submit a fleet nobody asked for.
    """
    tail = tune_run_tail(
        **_BASE, slurm=False, slurm_args={"slurm_account": "exfab"}
    )
    assert "--slurm" not in tail


def test_empty_keys_and_values_are_skipped() -> None:
    tail = tune_run_tail(
        **_BASE,
        slurm=True,
        slurm_args={"": "x", "slurm_account": "", "slurm_qos": None,
                    "slurm_partition": "epyc"},
    )
    assert _pairs(tail) == ["slurm_partition=epyc"]


def test_the_sugar_flags_and_the_profile_both_render() -> None:
    """The builder does not merge; precedence is resolved by ``run_tuning``.

    Rendering both is correct here — the CLI parses them and
    ``merge_slurm_args`` gives the explicit pair the win. A builder that
    silently dropped one would make the rendered command line disagree with
    what the CLI actually does with it.
    """
    tail = tune_run_tail(
        **{**_BASE, "slurm_partition": "batch"},
        slurm=True,
        slurm_args={"slurm_partition": "epyc"},
    )
    assert tail[tail.index("--slurm-partition") + 1] == "batch"
    assert _pairs(tail) == ["slurm_partition=epyc"]
