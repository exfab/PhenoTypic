"""No standalone parallel job beside an active ordinary array.

Project ``CLAUDE.md`` and ``_cli/CLAUDE.md``: allocation and submission bounds
are already consumed by the array cohort, so ancillary work routes through
reserved trigger entries **inside** an array task list rather than through a
sidecar ``sbatch``. A terminal ``afterany`` finalizer is explicitly **not** a
parallel sidecar and is allowed (``_cli/CLAUDE.md:104-107``).

**Which array, and it is the whole point of this file.** P5's aggregation
fan-out reserves ``TASK_FINALIZE`` inside the *dependent finalizer* array --
what was a one-task ``pht-finalizer`` becomes ``--array=0-K``. It is **not**
added to the image-processing array's sentinel entry list beside
``__PHENOTYPIC_CHECKPOINT__`` and ``__PHENOTYPIC_MANIFEST__``, for two
independent reasons:

* ``_cli/CLAUDE.md:104-107`` forbids converting a terminal finalizer into an
  array entry in as many words; and
* array indices run concurrently, so aggregation of image *i*'s table can
  never be ordered after image *i* within one array. There is no arrangement
  of that which aggregates a complete run.

``_cli_slurm_array_scripts.py:28`` says which list those two sentinels belong
to in its own comment -- *"inserted into the image list"*.

**The guards here are deliberately thin, and that is the strength rather than
an oversight.** The fan-out changes ``task_indices`` on a script the drip-feed
dispatcher already submits (``_cli_execution_strategies.py:1063,1067`` ->
``submit_slurm_script_chain``), so the dispatcher keeps submitting one thing
where it submitted one thing before. *No standalone parallel job is submitted*
therefore holds **by construction**: there is no new submission site to guard,
and the tests below assert that no such site was introduced rather than
policing the behaviour of one.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

#: Names that submit work to the scheduler. A fan-out module that reaches any
#: of them has stopped being a task body and become a submitter.
#:
#: Deliberately specific. Bare verbs like ``run`` or ``check_output`` would
#: also catch ``subprocess.run``, but they collide with ordinary method names
#: and would fail this test for reasons that have nothing to do with the
#: contract. ``subprocess`` and ``Popen`` catch the shell-out route on their
#: own, because reaching ``subprocess.run`` necessarily names ``subprocess``.
_SUBMISSION_PRIMITIVES = frozenset(
    {
        "sbatch",
        "submit_script",
        "submit_drip_feed_start",
        "submit_slurm_script_chain",
        "generate_dispatcher_chain",
        "subprocess",
        "Popen",
    }
)


def _module_path(dotted: str) -> Path:
    import importlib.util

    spec = importlib.util.find_spec(dotted)
    assert spec is not None and spec.origin is not None, dotted
    return Path(spec.origin)


def test_the_fanout_module_contains_no_submission_site_at_all() -> None:
    """The 'by construction' half, asserted rather than asserted-about.

    ``_cli_finalize_fanout`` is a task **body**: it is what an array index
    runs, never something that submits an array. If it ever grows an ``sbatch``
    or a ``submit_*`` call it has become a sidecar submitter, and the contract
    it would be breaking names that outcome specifically.

    Checked over the module's own source rather than by patching, because the
    property is *absence of a call site*, and no amount of patching can prove
    a call site does not exist on a path the test did not take.
    """
    source = _module_path("phenotypic._cli._cli_finalize_fanout").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)

    reached: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            reached.add(node.id)
        elif isinstance(node, ast.Attribute):
            reached.add(node.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                reached.add((alias.asname or alias.name).split(".")[0])

    offenders = sorted(reached & _SUBMISSION_PRIMITIVES)
    assert not offenders, (
        "the aggregation fan-out reached a scheduler submission primitive "
        f"{offenders}; it is an array task body, and a body that submits is "
        "the sidecar the array-auxiliary contract forbids"
    )


def test_the_image_array_entry_list_carries_no_finalize_token() -> None:
    """The File Structure row this phase does NOT follow, pinned as a guard.

    ``phase-5-fanout.md``'s file table said to add the finalize trigger beside
    ``_CHECKPOINT_SENTINEL`` and ``_MANIFEST_SENTINEL`` at
    ``_cli_slurm_array_scripts.py:30``. That is the **image** array's entry
    list, and putting aggregation there is both contract-violating and
    mechanically impossible (see this module's docstring).

    Without this test the rejected reading is only a decision recorded in
    prose, and prose is what this change exists to stop trusting.
    """
    from phenotypic._cli import _cli_slurm_array_scripts as scripts

    images = [Path("/tmp/a.tiff"), Path("/tmp/b.tiff")]
    entries = scripts._build_entry_list(images, checkpoint_interval=1)

    assert scripts._CHECKPOINT_SENTINEL in entries, (
        "the fixture produced no sentinels at all, so the assertion below "
        "would hold against an entry list this test never exercised"
    )
    finalize_tokens = [
        entry
        for entry in entries
        if "FINALIZE" in entry.upper() or "AGGREGAT" in entry.upper()
    ]
    assert not finalize_tokens, (
        f"a finalize trigger reached the IMAGE array's entry list: "
        f"{finalize_tokens}. It belongs to the dependent finalizer array."
    )


def test_only_two_sentinels_are_defined_for_the_image_array() -> None:
    """A third ``__PHENOTYPIC_*__`` token in that module means someone took
    the File Structure row's reading after all.

    Named separately from the entry-list test because the two fail for
    different reasons: a token can be *defined* without being *inserted*, and
    the definition is the earlier and cheaper signal.
    """
    from phenotypic._cli import _cli_slurm_array_scripts as scripts

    tokens = {
        value
        for name, value in vars(scripts).items()
        if isinstance(value, str)
        and value.startswith("__PHENOTYPIC_")
        and value.endswith("__")
    }
    assert tokens == {
        scripts._CHECKPOINT_SENTINEL,
        scripts._MANIFEST_SENTINEL,
    }, f"unexpected image-array trigger tokens: {sorted(tokens)}"


@pytest.mark.parametrize("shards", [1, 2, 8])
def test_the_finalize_entry_is_the_last_index_of_the_dependent_array(
    tmp_path: Path,
    make_exec_config,
    simple_pipeline_json: Path,
    shards: int,
) -> None:
    """``TASK_FINALIZE`` is a reserved entry INSIDE the array, at index K.

    The array is ``0-K``: K aggregation shards at indices ``0..K-1`` and the
    finalizer at ``K``. ``MaxArraySize`` caps the index, so the top index must
    be K itself and the rendered directive must say so.
    """
    from phenotypic._cli._cli_slurm_array_scripts import (
        generate_terminal_finalizer_script,
    )

    output_dir = tmp_path / "out"
    config = make_exec_config(
        pipeline_json=simple_pipeline_json,
        input_path=tmp_path / "in",
        output_dir=output_dir,
        slurm_args={"slurm_partition": "short"},
        force_local=False,
    )

    script = generate_terminal_finalizer_script(
        config, output_dir, shard_count=shards
    )
    text = script.read_text(encoding="utf-8")

    assert f"#SBATCH --array=0-{shards}" in text, (
        f"expected K={shards} shards plus one reserved finalizer index; "
        f"script says:\n{text}"
    )


def test_finalization_submits_no_job_beside_the_array() -> None:
    """The ordinary SLURM path has exactly ONE submission call, and it is the
    mandated chokepoint.

    Asserted over ``AutonomousSLURMStrategy.execute``'s own AST, and the
    instrument is chosen deliberately.

    **The rejected alternative, recorded because it looks more thorough and is
    weaker.** The plan drafted a ``fake_sbatch`` fixture capturing raw
    ``sbatch`` argv. That would go **green** on code that bypassed the
    drip-feed dispatcher and shelled out itself -- precisely the failure
    ``_cli/CLAUDE.md:120-124`` records as having already happened: *"eager
    submission is what caused the AssocMaxSubmitJobLimit failures."*

    **And the second rejected alternative, which is worse and is why this test
    is not behavioural.** Patching ``submit_slurm_script_chain`` and then
    calling it proves only that the test called it. Patching it and driving
    the strategy would be real, but a mock that records one call still cannot
    show that no *other* submission exists on a branch the fixture did not
    take. The claim under test is *no second submission site was introduced* --
    absence of a call site is a structural property, and only a structural
    instrument decides it.
    """
    import inspect
    import textwrap

    from phenotypic._cli._cli_execution_strategies import (
        AutonomousSLURMStrategy,
    )

    # `inspect.getsource` on a METHOD returns it at class-body indentation,
    # which `ast.parse` rejects outright. Dedent first.
    tree = ast.parse(
        textwrap.dedent(inspect.getsource(AutonomousSLURMStrategy.execute))
    )
    called: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else ""
        )
        if name in _SUBMISSION_PRIMITIVES:
            called.append(name)

    assert called == ["submit_slurm_script_chain"], (
        "the ordinary SLURM path must submit exactly once, through the "
        "drip-feed dispatcher chain that `_cli/CLAUDE.md:120-124` mandates; "
        f"it reached {called}"
    )


@pytest.mark.parametrize("shards", [1, 2, 8])
def test_growing_the_array_produces_no_second_script(
    tmp_path: Path,
    make_exec_config,
    simple_pipeline_json: Path,
    shards: int,
) -> None:
    """K shards plus a finalizer is ONE script, whatever K is.

    The behavioural half of the "by construction" claim. If the fan-out ever
    grew a second script the dispatcher would have to submit it separately,
    and a sidecar is exactly what that would be -- so a single returned path
    across a K sweep is the thing worth pinning.
    """
    from phenotypic._cli._cli_slurm_array_scripts import (
        generate_terminal_finalizer_script,
    )

    output_dir = tmp_path / "out"
    config = make_exec_config(
        pipeline_json=simple_pipeline_json,
        input_path=tmp_path / "in",
        output_dir=output_dir,
        slurm_args={"slurm_partition": "short"},
        force_local=False,
    )

    script = generate_terminal_finalizer_script(
        config, output_dir, shard_count=shards
    )

    assert script.is_file()
    siblings = sorted(script.parent.glob("*finalizer*.sh"))
    assert siblings == [script], (
        f"K={shards} produced more than one finalizer script: {siblings}"
    )
