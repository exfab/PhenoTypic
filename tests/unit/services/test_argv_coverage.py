"""Every top-level CLI option is either emittable from ``_services.argv``
or explicitly denied with a reason.

A hand-written list of "flags the service tier can render" has already leaked
four times: three flags a reviewer found, ``--gpu-workers-per-gpu`` that nobody
listed, and ``--image-manifest`` itself. OME-Zarr's ``--durable-writes`` will be
the next. So the coverage is derived, not written down — enumerated from
``phenotypic_cli.params`` on one side and parsed out of the argv builders on the
other, with a deny-list that must justify each omission and must not name a flag
that no longer exists.

Same shape as the tune annotation-coverage gate and the ``FEATURES.md`` ledger
gate: the gate fails on the day a flag lands, not the day someone remembers.
"""

from __future__ import annotations

import ast
import inspect
import re

import click
import pytest

from phenotypic._services import argv as argv_module
from phenotypic.phenotypicCLI import phenotypic_cli

# The functions that render ``python -m phenotypic`` argv. Restricting the walk
# to these three is what keeps ``phenotypic.tune``'s flags (``--strategy``,
# ``--n-trials``, …), which live in the same module, out of the comparison.
_EMITTING_FUNCTIONS = ("to_argv", "slurm_argv_extension", "to_subprocess_argv")

_FLAG = re.compile(r"^--[a-z0-9][a-z0-9-]*$")

# Top-level options the service tier deliberately cannot emit. Each entry is a
# reason, not a placeholder: an unexplained line here is how a flag that should
# be reachable gets parked instead of implemented.
_DENIED: dict[str, str] = {
    "--bit-depth": "Source property of the images, not a run request; the MCP server has no way to know it and the pipeline reads it from the file.",
    "--detect-mode": "Belongs to the pipeline JSON, which already pins the detection matrix; a second spelling on the command line could contradict it.",
    "--gpu-workers-per-gpu": "Reserved Stage-2 replica count; recorded but not acted on by the current worker, so emitting it would imply a capability that does not exist.",
    "--force-local": "Routing is chosen by the caller through slurm_args (empty means local); a second, contradictory switch is exactly the ambiguity RunConsoleState removes.",
    "--wait": "Never emitted by design: an MCP call must not block for hours, and the GUI polls. Spec 05 §5.4 states this.",
    "--ext": "Legacy overlay-PNG extension. Forward runs write one HDF per image; exposing it would resurrect a switch that no longer selects anything.",
    "--overlay-alpha": "Rendering cosmetic with a sane default, and it participates in the resume-compatibility digest, so an incidental change would break continuation.",
    "--no-dataset-column": "Drops the Dataset column from measurements, which every downstream deliverable and the GUI mirror rely on.",
    "--random-seed": "Only meaningful with --sample's random draw; the subset mechanism (spec §10) is the supported way to thin an input set.",
    "--overwrite": "Deletes the output tree. --restart clears machine state and keeps artifacts; a serialized state must not be able to ask for the destructive one.",
    "--study": "REMBI study.yaml is workspace provenance a human authors, not a per-run argument.",
    "--checkpoint-interval": "Auto-estimated from the array size; a wrong manual value silently degrades SLURM continuation.",
    "--skip-validation": "Disables pipeline validation. Nothing that submits to a cluster unattended should be able to turn the checks off.",
    "--no-qc": "QC presence is a property of the pipeline's 'qc' section; suppressing it from the launcher would contradict the pipeline.",
    "--layer": "Only valid with --mode process, and v1 deploy is always the full pipeline (spec 05 §5.3 cut mode, layer and sample).",
}


def _cli_option_flags() -> set[str]:
    """Every long-form top-level option of ``python -m phenotypic``.

    Read from the Click command rather than from the source: a decorator added
    anywhere in the ~240-line option block is picked up with no edit here.
    """
    flags: set[str] = set()
    for param in phenotypic_cli.params:
        if not isinstance(param, click.Option):
            continue
        flags.update(opt for opt in param.opts if opt.startswith("--"))
    return flags


def _emitted_flags() -> set[str]:
    """Flag literals the argv builders can put on a command line.

    An AST walk over the three emitting functions, not a substring scan of the
    module: prose in a docstring mentioning ``--wait`` must not count as an
    emitter, and a flag assembled from a variable must not count either — if
    one ever is, this gate should notice the flag went missing.
    """
    tree = ast.parse(inspect.getsource(argv_module))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in _EMITTING_FUNCTIONS:
            continue
        body = ast.Module(body=node.body, type_ignores=[])
        for child in ast.walk(body):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                if _FLAG.match(child.value):
                    flags.add(child.value)
    return flags


def test_the_emitting_functions_all_exist() -> None:
    """A rename must break this file loudly, not silently empty the walk."""
    tree = ast.parse(inspect.getsource(argv_module))
    defined = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }
    missing = sorted(set(_EMITTING_FUNCTIONS) - defined)
    assert not missing, (
        f"_services.argv no longer defines {missing}; this gate would walk "
        "nothing and pass vacuously. Update _EMITTING_FUNCTIONS."
    )


def test_every_cli_option_is_emitted_or_explicitly_denied() -> None:
    uncovered = sorted(_cli_option_flags() - _emitted_flags() - set(_DENIED))
    assert not uncovered, (
        "These top-level CLI options can be neither emitted from "
        f"_services/argv.py nor explained by the deny-list: {uncovered}. "
        "Either give RunConsoleState a field and to_argv a branch, or add a "
        "one-line reason to _DENIED in this file."
    )


def test_the_deny_list_names_only_real_options() -> None:
    """A stale entry silently excuses a flag that no longer exists."""
    stale = sorted(set(_DENIED) - _cli_option_flags())
    assert not stale, f"_DENIED names options the CLI no longer has: {stale}"


def test_nothing_is_both_emitted_and_denied() -> None:
    both = sorted(_emitted_flags() & set(_DENIED))
    assert not both, (
        f"_DENIED claims these are unreachable, but _services/argv.py emits "
        f"them: {both}"
    )


def test_every_emitted_flag_is_a_real_cli_option() -> None:
    """Catches a typo or a rename on the emitting side.

    ``--image-manifest`` is exactly the shape of mistake this finds: the run
    manifest that ``deploy_status`` polls is also called "the manifest", and an
    emitter spelling it ``--manifest`` would collide with the staged worker's
    internal flag while passing every other test in this file.
    """
    unknown = sorted(_emitted_flags() - _cli_option_flags())
    assert not unknown, (
        f"_services/argv.py emits flags python -m phenotypic does not accept: "
        f"{unknown}"
    )


def test_every_deny_list_entry_carries_a_reason() -> None:
    empty = sorted(flag for flag, reason in _DENIED.items() if len(reason) < 20)
    assert not empty, f"_DENIED entries with no real reason: {empty}"


@pytest.mark.parametrize(
    "flag",
    [
        "--pipeline",
        "--input",
        "--image-manifest",
        "--output",
        "--mode",
        "--metadata",
        "--dry-run",
        "--retry-failures",
        "--restart",
        "--sample",
        "--nrows",
        "--ncols",
        "--image-type",
        "--njobs",
        "--slurm",
        "--gpu-slurm",
        "--gpu-shards",
    ],
)
def test_named_flags_stay_emittable(flag: str) -> None:
    """The seventeen the service tier is expected to reach, named explicitly.

    The derived gate above passes if a flag moves from "emitted" to "denied";
    this one says which ones may not make that move without a deliberate edit.
    """
    assert flag in _emitted_flags()


def test_the_counts_are_what_the_plan_recorded() -> None:
    """32 options, 17 emitted, 15 denied — a drift tripwire, not a target."""
    options = _cli_option_flags()
    emitted = _emitted_flags() & options
    assert (len(options), len(emitted), len(_DENIED)) == (32, 17, 15), (
        f"option/emit/deny counts moved to "
        f"{(len(options), len(emitted), len(_DENIED))}. That is fine if it was "
        "intended — update this number in the same commit as the flag."
    )
