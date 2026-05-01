"""Run console form — pipeline/input/output pickers + mode + advanced + slurm (Phase 6).

Phase 0 placeholder — implementation lands in Phase 6. See ``GUI_SPEC_V1.md``
section 5.

Per plan: pipeline/input/output pickers reuse builder's ``_modal_browser.py``
patterns. Mode toggle (Local / SLURM). Inline ``Dry-run`` + ``Resume``
checkboxes. ``Advanced`` collapse (sample, nrows/ncols, image-type, workers,
log-level). ``SLURM config`` collapse (typed common fields + free-form ``k=v``
rows).
"""
from __future__ import annotations

# TODO(Phase 6): build_form(sandbox) + collect_form_values(state).

__all__: list[str] = []
