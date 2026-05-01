"""Thin shell-out wrapper over ``phenotypic._cli._cli_slurm_submission`` (Phase 6).

Phase 0 placeholder — implementation lands in Phase 6. See ``GUI_SPEC_V1.md``
section 5.

Per plan (CRITICAL): invoke via subprocess (``python -m phenotypic ...
--slurm k=v ...``) — do NOT import CLI internals. This guarantees that
GUI-submitted SLURM runs are indistinguishable from hand-typed CLI
submissions. Job-id resolution reads ``<output_dir>/progress/job_metadata.json``
after the submitter exits — DO NOT parse Rich-formatted stdout (locale /
terminal-width fragile). The CLI already writes structured ``chunk_job_ids``
to that file.
"""
from __future__ import annotations

# TODO(Phase 6): submit(form_values, sandbox) -> SlurmRunHandle;
# read_job_metadata(output_dir) -> dict.

__all__: list[str] = []
