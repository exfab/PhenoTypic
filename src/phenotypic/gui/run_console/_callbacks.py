"""Run console Dash callbacks (Phase 6).

Phase 0 placeholder — implementation lands in Phase 6. See ``GUI_SPEC_V1.md``
section 5.

Wired callbacks (per plan):
    * Run (Local) → ``LocalRunner.start(...)`` → poll for ``dashboard.html``
      (max 10s) → set iframe ``src`` via clientside callback; stream log tail
      via ``dcc.Interval``.
    * Run (SLURM) → ``_slurm.submit(...)`` → register ``SlurmRunHandle`` → set
      iframe ``src`` immediately (SLURM submitter writes ``dashboard.html``
      up-front).
    * Validate (dry-run) → spawn with ``--dry-run``; log only; no iframe.
    * Cancel (running) → ``LocalRunner.stop(run_id)``.
    * Save preset → write to ``<root>/.phenotypic-gui/presets/<name>.json``.
    * Recent Runs row click → re-point iframe at ``/runs/<rel>/dashboard.html``.
"""
from __future__ import annotations

# TODO(Phase 6): register_callbacks(app, sandbox, runs_registry).

__all__: list[str] = []
