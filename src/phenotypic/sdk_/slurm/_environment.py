"""Environment handoff helpers for SLURM submissions and batch scripts."""

from __future__ import annotations

import os
from collections.abc import Mapping

SLURM_PYTHONPATH_ENV_VAR = "PHENOTYPIC_SLURM_PYTHONPATH"

SLURM_PYTHONPATH_BOOTSTRAP_BASH = f"""\
# Restore the caller's Python import path when site policy filters PYTHONPATH.
if [ -n "${{{SLURM_PYTHONPATH_ENV_VAR}:-}}" ]; then
    if [ -z "${{PYTHONPATH:-}}" ]; then
        export PYTHONPATH="${{{SLURM_PYTHONPATH_ENV_VAR}}}"
    elif [ "$PYTHONPATH" != "${{{SLURM_PYTHONPATH_ENV_VAR}}}" ]; then
        export PYTHONPATH="${{{SLURM_PYTHONPATH_ENV_VAR}}}:$PYTHONPATH"
    fi
fi"""


def sbatch_submission_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Snapshot the caller's Python path under a SLURM-safe variable.

    Some clusters explicitly filter ``PYTHONPATH`` even when ``sbatch`` uses
    ``--export=ALL``. PhenoTypic batch scripts restore this namespaced copy
    before invoking Python.

    Args:
        environment: Environment to copy. Defaults to ``os.environ``.

    Returns:
        A detached environment mapping suitable for ``subprocess.run``.
    """
    submitted = dict(os.environ if environment is None else environment)
    python_path = submitted.get("PYTHONPATH", "")
    if python_path:
        submitted[SLURM_PYTHONPATH_ENV_VAR] = python_path
    else:
        submitted.pop(SLURM_PYTHONPATH_ENV_VAR, None)
    return submitted
