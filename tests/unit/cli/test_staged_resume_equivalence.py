"""The staged resume decisions must be identical after the marker collapse.

``classify_staged_image`` reads four independent filesystem probes today: a valid
stage-1 store, a stage-2 token, a retained stage-2 raw array, and a stage-3
completion marker. Spec §6.1 collapses the stage-3 marker into the per-image
record. The record format is not the risk -- the resume DECISIONS are, because a
wrong one either reprocesses 6,000 images or silently skips one.

This is a table test over every reachable combination, captured from the CURRENT
behaviour and re-run after the change.

**What is frozen, and what deliberately is not.** The frozen thing is the
*decision function over semantic states* -- "this image's stage 3 was reported",
not "this file exists at this path". So ``_plant`` writes the stage-3 fact
through :func:`write_stage3_completion_marker`, the same writer the classifier's
reader is paired with, and both move together across the collapse. Hand-joining
``progress/stage3_complete/<ds>/<stem>.json`` instead would make every
``s3_done=True`` cell flip the moment the writer moved, which is a tautological
failure detector rather than an equivalence gate.

Asked the other way -- what would a green run here have looked like if the
collapse HAD broken a decision? The table is keyed by the full axis tuple and
:func:`test_the_expected_table_covers_exactly_the_enumerated_combinations`
refuses a table that is missing or over-supplied for the enumeration, so a
truncated capture cannot pass by not being asked. And
:func:`test_the_axes_reach_every_outcome` refuses a table that only ever says
``"stage1"``.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image
from phenotypic._cli._cli_process_only import process_only_output_path
from phenotypic._cli._cli_stage2_token import (
    write_stage2_raw,
    write_stage2_token,
)
from phenotypic._cli._cli_staged_resume import (
    classify_staged_image,
    write_stage3_completion_marker,
)
from phenotypic.sdk_ import (
    MEASUREMENT_TABLE_RELATIVE_PATH,
    zarr_store_path,
)

_DATASET = "plate"
_STEM = "img"
_IMAGE_NAME = f"{_STEM}.tif"

#: The identity the caller expects when ``expect_work_id`` is True.
_WORK_ID = "w-expected"

#: What a store bound to somebody else's work carries.
_OTHER_WORK_ID = "w-other"


# ---------------------------------------------------------------------------
# The axes, and the evidence for each
# ---------------------------------------------------------------------------
#
# CAN-16 replaced the original `product([False, True], repeat=4)` -- store,
# s2_token, s2_raw, s3_done -- because it covered a small fraction of the
# reachable space and collapsed four distinct store predicates into one boolean.
# The plan's replacement has seven axes. This enumeration keeps all seven and
# repairs three gaps found by reading the branch conditions:
#
# 1. `_STORE_STATES` gains a FIFTH state. `classify_staged_image` reads three
#    independent store predicates -- `valid_staged_store` (V), `valid_stage1_store`
#    (S) and a work-id match (W), with S implying V -- and selects between them at
#    `_cli_staged_resume.py:229-234`. The reachable triples are (F,F,F), (T,F,F),
#    (T,F,T), (T,T,F) and (T,T,T), and each produces a distinct
#    `(store_valid, stage2_store_valid)` pair across the two `expected_work_id`
#    arms. The missing one was (T,F,T) -- a structurally valid store whose journal
#    is `in_progress` but which carries the expected work id -- which is precisely
#    the state the comment at `:235-238` exists for.
# 2. A MEASUREMENT-TABLE axis. The branch at `:248-257` is
#    `layer is None and work_id is not None and s3_done and table.is_file()`, so
#    without this axis it can never fire in the True direction -- and it is one of
#    only two branches that consult the stage-3 fact this task rewrites. The
#    plan's own CAN-16 note lists "the in-store measurement table (:250-256, :258)"
#    as a branch axis; `_COMBOS` had no such axis.
# 3. The `"objmap"` layer SPLITS on the terminal output. `:220-224` consults
#    `process_only_output_path(..., "objmap", fmt="tiff")` only under
#    `process_only_layer == "objmap"`, so one `"objmap"` state leaves that branch
#    half-frozen.
#
# Gaps 1-3 are one defect class, and THIS REPO HAS ALREADY PAID FOR IT ONCE.
# `test_staged_resume_parity.py:26-32`, on the sibling harness over this same
# function:
#
#     "The FIFTH axis is load-bearing. classify_staged_image's first branch
#      consults valid_image_success, which reads the per-image completion
#      marker. Without this axis that branch is never exercised --
#      valid_image_success returns False in both worlds -- and the parity test
#      passes while production breaks."
#
# Someone hit exactly that on the parity harness, fixed it, and wrote down why.
# An axis a table does not carry is not a case the table permits; it is a branch
# the table cannot see, and the table goes green either way. Every gap repaired
# above is the same shape, and the measurement table is the one that matters
# most, because `:248-257` is one of only two branches this task rewrites.
#
# NOT an axis, by evidence: `valid_image_success`. Its call at `:209-218` is an
# unconditional early return that sits ABOVE every line this task edits, and it is
# being rewritten concurrently onto the record by P3 Task 2 -- so parameterizing it
# would freeze another cluster's in-flight behaviour into this table. It is covered
# instead by `test_the_image_success_early_return_still_short_circuits`, which
# patches the function rather than any marker shape.

#: ``(V, S, W)``: ``valid_staged_store``, ``valid_stage1_store``, work-id match.
_STORE_STATES: tuple[str, ...] = (
    "absent",  # (F, F, F)
    "in_progress",  # (T, F, F) -- decoded stage-1 checkpoint, no work id
    "in_progress_matching_work_id",  # (T, F, T)
    "mismatched_work_id",  # (T, T, F)
    "matching_work_id",  # (T, T, T)
)

#: ``process_only_layer``, with the objmap arm split on its terminal output.
#: ``"rgb"`` stands for any non-``None`` non-objmap layer: `:243`, `:249` and
#: `:261` only ever test ``is None``, and `:220` only ever tests ``== "objmap"``.
_LAYER_STATES: tuple[str, ...] = (
    "none",
    "objmap",
    "objmap_terminal",
    "rgb",
)

_LAYER_VALUES: dict[str, str | None] = {
    "none": None,
    "objmap": "objmap",
    "objmap_terminal": "objmap",
    "rgb": "rgb",
}

#: The measurement table lives INSIDE the store, so "no store, but a table" is
#: not a state of this tree -- creating it would materialise a store directory
#: and make ``store="absent"`` a lie. The axis is therefore coupled rather than
#: crossed: nine store/table combinations, not ten.
_STORE_TABLE_COMBOS: tuple[tuple[str, bool], ...] = tuple(
    (store, table)
    for store in _STORE_STATES
    for table in ((False,) if store == "absent" else (False, True))
)

_COMBOS = [
    (store, table, s2_token, s2_raw, s3_done, layer, markers, expect_work_id)
    for store, table in _STORE_TABLE_COMBOS
    for s2_token in (False, True)
    for s2_raw in (False, True)
    for s3_done in (False, True)
    for layer in _LAYER_STATES
    for markers in (False, True)
    for expect_work_id in (False, True)
]

_Key = tuple[str, bool, bool, bool, bool, str, bool, bool]

#: Captured from the PRE-CHANGE behaviour, as a literal table. Do NOT derive
#: these by reasoning about what the classifier should do -- the point is to
#: freeze what it DOES, so the collapse is provably behaviour-preserving. If one
#: of them looks wrong, record it in a comment and leave it: fixing a resume bug
#: inside a refactor makes both unreviewable.
#:
#: The key is the full eight-axis tuple, matching ``_COMBOS``.
#:
#: The two sentinel comments below delimit the generated region. The Step-2
#: capture splices between them, so a re-capture is a mechanical replacement of
#: a marked block rather than a hand-edit of 1,152 lines -- and a reviewer can
#: see exactly which lines were machine-written.
#:
#: **The generated entries run to ~104 columns, past this repo's
#: ``line-length = 79``, and that is deliberate.** ``[tool.ruff]`` sets no
#: ``select``, so ruff lints on its default ``E4,E7,E9,F`` -- ``E501`` is not
#: enabled and these lines are not flagged. Wrapping each key across four lines
#: to fit would quadruple the block to roughly 4,600 lines and make a table
#: whose only purpose is to be diffed unreadable in a diff.
# --- BEGIN CAPTURED TABLE (generated; do not hand-edit) ---
_EXPECTED: dict[_Key, str] = {
    ('absent', False, False, False, False, 'none', False, False): 'stage1',
    ('absent', False, False, False, False, 'none', False, True): 'stage1',
    ('absent', False, False, False, False, 'none', True, False): 'stage1',
    ('absent', False, False, False, False, 'none', True, True): 'stage1',
    ('absent', False, False, False, False, 'objmap', False, False): 'stage1',
    ('absent', False, False, False, False, 'objmap', False, True): 'stage1',
    ('absent', False, False, False, False, 'objmap', True, False): 'stage1',
    ('absent', False, False, False, False, 'objmap', True, True): 'stage1',
    ('absent', False, False, False, False, 'objmap_terminal', False, False): 'complete',
    ('absent', False, False, False, False, 'objmap_terminal', False, True): 'stage1',
    ('absent', False, False, False, False, 'objmap_terminal', True, False): 'complete',
    ('absent', False, False, False, False, 'objmap_terminal', True, True): 'stage1',
    ('absent', False, False, False, False, 'rgb', False, False): 'stage1',
    ('absent', False, False, False, False, 'rgb', False, True): 'stage1',
    ('absent', False, False, False, False, 'rgb', True, False): 'stage1',
    ('absent', False, False, False, False, 'rgb', True, True): 'stage1',
    ('absent', False, False, False, True, 'none', False, False): 'stage1',
    ('absent', False, False, False, True, 'none', False, True): 'stage1',
    ('absent', False, False, False, True, 'none', True, False): 'stage1',
    ('absent', False, False, False, True, 'none', True, True): 'stage1',
    ('absent', False, False, False, True, 'objmap', False, False): 'stage1',
    ('absent', False, False, False, True, 'objmap', False, True): 'stage1',
    ('absent', False, False, False, True, 'objmap', True, False): 'stage1',
    ('absent', False, False, False, True, 'objmap', True, True): 'stage1',
    ('absent', False, False, False, True, 'objmap_terminal', False, False): 'complete',
    ('absent', False, False, False, True, 'objmap_terminal', False, True): 'stage1',
    ('absent', False, False, False, True, 'objmap_terminal', True, False): 'complete',
    ('absent', False, False, False, True, 'objmap_terminal', True, True): 'stage1',
    ('absent', False, False, False, True, 'rgb', False, False): 'stage1',
    ('absent', False, False, False, True, 'rgb', False, True): 'stage1',
    ('absent', False, False, False, True, 'rgb', True, False): 'stage1',
    ('absent', False, False, False, True, 'rgb', True, True): 'stage1',
    ('absent', False, False, True, False, 'none', False, False): 'stage1',
    ('absent', False, False, True, False, 'none', False, True): 'stage1',
    ('absent', False, False, True, False, 'none', True, False): 'stage1',
    ('absent', False, False, True, False, 'none', True, True): 'stage1',
    ('absent', False, False, True, False, 'objmap', False, False): 'stage1',
    ('absent', False, False, True, False, 'objmap', False, True): 'stage1',
    ('absent', False, False, True, False, 'objmap', True, False): 'stage1',
    ('absent', False, False, True, False, 'objmap', True, True): 'stage1',
    ('absent', False, False, True, False, 'objmap_terminal', False, False): 'complete',
    ('absent', False, False, True, False, 'objmap_terminal', False, True): 'stage1',
    ('absent', False, False, True, False, 'objmap_terminal', True, False): 'complete',
    ('absent', False, False, True, False, 'objmap_terminal', True, True): 'stage1',
    ('absent', False, False, True, False, 'rgb', False, False): 'stage1',
    ('absent', False, False, True, False, 'rgb', False, True): 'stage1',
    ('absent', False, False, True, False, 'rgb', True, False): 'stage1',
    ('absent', False, False, True, False, 'rgb', True, True): 'stage1',
    ('absent', False, False, True, True, 'none', False, False): 'stage1',
    ('absent', False, False, True, True, 'none', False, True): 'stage1',
    ('absent', False, False, True, True, 'none', True, False): 'stage1',
    ('absent', False, False, True, True, 'none', True, True): 'stage1',
    ('absent', False, False, True, True, 'objmap', False, False): 'stage1',
    ('absent', False, False, True, True, 'objmap', False, True): 'stage1',
    ('absent', False, False, True, True, 'objmap', True, False): 'stage1',
    ('absent', False, False, True, True, 'objmap', True, True): 'stage1',
    ('absent', False, False, True, True, 'objmap_terminal', False, False): 'complete',
    ('absent', False, False, True, True, 'objmap_terminal', False, True): 'stage1',
    ('absent', False, False, True, True, 'objmap_terminal', True, False): 'complete',
    ('absent', False, False, True, True, 'objmap_terminal', True, True): 'stage1',
    ('absent', False, False, True, True, 'rgb', False, False): 'stage1',
    ('absent', False, False, True, True, 'rgb', False, True): 'stage1',
    ('absent', False, False, True, True, 'rgb', True, False): 'stage1',
    ('absent', False, False, True, True, 'rgb', True, True): 'stage1',
    ('absent', False, True, False, False, 'none', False, False): 'stage1',
    ('absent', False, True, False, False, 'none', False, True): 'stage1',
    ('absent', False, True, False, False, 'none', True, False): 'stage1',
    ('absent', False, True, False, False, 'none', True, True): 'stage1',
    ('absent', False, True, False, False, 'objmap', False, False): 'stage1',
    ('absent', False, True, False, False, 'objmap', False, True): 'stage1',
    ('absent', False, True, False, False, 'objmap', True, False): 'stage1',
    ('absent', False, True, False, False, 'objmap', True, True): 'stage1',
    ('absent', False, True, False, False, 'objmap_terminal', False, False): 'complete',
    ('absent', False, True, False, False, 'objmap_terminal', False, True): 'stage1',
    ('absent', False, True, False, False, 'objmap_terminal', True, False): 'complete',
    ('absent', False, True, False, False, 'objmap_terminal', True, True): 'stage1',
    ('absent', False, True, False, False, 'rgb', False, False): 'stage1',
    ('absent', False, True, False, False, 'rgb', False, True): 'stage1',
    ('absent', False, True, False, False, 'rgb', True, False): 'stage1',
    ('absent', False, True, False, False, 'rgb', True, True): 'stage1',
    ('absent', False, True, False, True, 'none', False, False): 'stage1',
    ('absent', False, True, False, True, 'none', False, True): 'stage1',
    ('absent', False, True, False, True, 'none', True, False): 'stage1',
    ('absent', False, True, False, True, 'none', True, True): 'stage1',
    ('absent', False, True, False, True, 'objmap', False, False): 'stage1',
    ('absent', False, True, False, True, 'objmap', False, True): 'stage1',
    ('absent', False, True, False, True, 'objmap', True, False): 'stage1',
    ('absent', False, True, False, True, 'objmap', True, True): 'stage1',
    ('absent', False, True, False, True, 'objmap_terminal', False, False): 'complete',
    ('absent', False, True, False, True, 'objmap_terminal', False, True): 'stage1',
    ('absent', False, True, False, True, 'objmap_terminal', True, False): 'complete',
    ('absent', False, True, False, True, 'objmap_terminal', True, True): 'stage1',
    ('absent', False, True, False, True, 'rgb', False, False): 'stage1',
    ('absent', False, True, False, True, 'rgb', False, True): 'stage1',
    ('absent', False, True, False, True, 'rgb', True, False): 'stage1',
    ('absent', False, True, False, True, 'rgb', True, True): 'stage1',
    ('absent', False, True, True, False, 'none', False, False): 'stage1',
    ('absent', False, True, True, False, 'none', False, True): 'stage1',
    ('absent', False, True, True, False, 'none', True, False): 'stage1',
    ('absent', False, True, True, False, 'none', True, True): 'stage1',
    ('absent', False, True, True, False, 'objmap', False, False): 'stage1',
    ('absent', False, True, True, False, 'objmap', False, True): 'stage1',
    ('absent', False, True, True, False, 'objmap', True, False): 'stage1',
    ('absent', False, True, True, False, 'objmap', True, True): 'stage1',
    ('absent', False, True, True, False, 'objmap_terminal', False, False): 'complete',
    ('absent', False, True, True, False, 'objmap_terminal', False, True): 'stage1',
    ('absent', False, True, True, False, 'objmap_terminal', True, False): 'complete',
    ('absent', False, True, True, False, 'objmap_terminal', True, True): 'stage1',
    ('absent', False, True, True, False, 'rgb', False, False): 'stage1',
    ('absent', False, True, True, False, 'rgb', False, True): 'stage1',
    ('absent', False, True, True, False, 'rgb', True, False): 'stage1',
    ('absent', False, True, True, False, 'rgb', True, True): 'stage1',
    ('absent', False, True, True, True, 'none', False, False): 'stage1',
    ('absent', False, True, True, True, 'none', False, True): 'stage1',
    ('absent', False, True, True, True, 'none', True, False): 'stage1',
    ('absent', False, True, True, True, 'none', True, True): 'stage1',
    ('absent', False, True, True, True, 'objmap', False, False): 'stage1',
    ('absent', False, True, True, True, 'objmap', False, True): 'stage1',
    ('absent', False, True, True, True, 'objmap', True, False): 'stage1',
    ('absent', False, True, True, True, 'objmap', True, True): 'stage1',
    ('absent', False, True, True, True, 'objmap_terminal', False, False): 'complete',
    ('absent', False, True, True, True, 'objmap_terminal', False, True): 'stage1',
    ('absent', False, True, True, True, 'objmap_terminal', True, False): 'complete',
    ('absent', False, True, True, True, 'objmap_terminal', True, True): 'stage1',
    ('absent', False, True, True, True, 'rgb', False, False): 'stage1',
    ('absent', False, True, True, True, 'rgb', False, True): 'stage1',
    ('absent', False, True, True, True, 'rgb', True, False): 'stage1',
    ('absent', False, True, True, True, 'rgb', True, True): 'stage1',
    ('in_progress', False, False, False, False, 'none', False, False): 'stage1',
    ('in_progress', False, False, False, False, 'none', False, True): 'stage1',
    ('in_progress', False, False, False, False, 'none', True, False): 'stage1',
    ('in_progress', False, False, False, False, 'none', True, True): 'stage1',
    ('in_progress', False, False, False, False, 'objmap', False, False): 'stage1',
    ('in_progress', False, False, False, False, 'objmap', False, True): 'stage1',
    ('in_progress', False, False, False, False, 'objmap', True, False): 'stage1',
    ('in_progress', False, False, False, False, 'objmap', True, True): 'stage1',
    ('in_progress', False, False, False, False, 'objmap_terminal', False, False): 'complete',
    ('in_progress', False, False, False, False, 'objmap_terminal', False, True): 'stage1',
    ('in_progress', False, False, False, False, 'objmap_terminal', True, False): 'complete',
    ('in_progress', False, False, False, False, 'objmap_terminal', True, True): 'stage1',
    ('in_progress', False, False, False, False, 'rgb', False, False): 'stage1',
    ('in_progress', False, False, False, False, 'rgb', False, True): 'stage1',
    ('in_progress', False, False, False, False, 'rgb', True, False): 'stage1',
    ('in_progress', False, False, False, False, 'rgb', True, True): 'stage1',
    ('in_progress', False, False, False, True, 'none', False, False): 'stage1',
    ('in_progress', False, False, False, True, 'none', False, True): 'stage1',
    ('in_progress', False, False, False, True, 'none', True, False): 'stage1',
    ('in_progress', False, False, False, True, 'none', True, True): 'stage1',
    ('in_progress', False, False, False, True, 'objmap', False, False): 'stage1',
    ('in_progress', False, False, False, True, 'objmap', False, True): 'stage1',
    ('in_progress', False, False, False, True, 'objmap', True, False): 'stage1',
    ('in_progress', False, False, False, True, 'objmap', True, True): 'stage1',
    ('in_progress', False, False, False, True, 'objmap_terminal', False, False): 'complete',
    ('in_progress', False, False, False, True, 'objmap_terminal', False, True): 'stage1',
    ('in_progress', False, False, False, True, 'objmap_terminal', True, False): 'complete',
    ('in_progress', False, False, False, True, 'objmap_terminal', True, True): 'stage1',
    ('in_progress', False, False, False, True, 'rgb', False, False): 'stage1',
    ('in_progress', False, False, False, True, 'rgb', False, True): 'stage1',
    ('in_progress', False, False, False, True, 'rgb', True, False): 'stage1',
    ('in_progress', False, False, False, True, 'rgb', True, True): 'stage1',
    ('in_progress', False, False, True, False, 'none', False, False): 'stage1',
    ('in_progress', False, False, True, False, 'none', False, True): 'stage1',
    ('in_progress', False, False, True, False, 'none', True, False): 'stage1',
    ('in_progress', False, False, True, False, 'none', True, True): 'stage1',
    ('in_progress', False, False, True, False, 'objmap', False, False): 'stage1',
    ('in_progress', False, False, True, False, 'objmap', False, True): 'stage1',
    ('in_progress', False, False, True, False, 'objmap', True, False): 'stage1',
    ('in_progress', False, False, True, False, 'objmap', True, True): 'stage1',
    ('in_progress', False, False, True, False, 'objmap_terminal', False, False): 'complete',
    ('in_progress', False, False, True, False, 'objmap_terminal', False, True): 'stage1',
    ('in_progress', False, False, True, False, 'objmap_terminal', True, False): 'complete',
    ('in_progress', False, False, True, False, 'objmap_terminal', True, True): 'stage1',
    ('in_progress', False, False, True, False, 'rgb', False, False): 'stage1',
    ('in_progress', False, False, True, False, 'rgb', False, True): 'stage1',
    ('in_progress', False, False, True, False, 'rgb', True, False): 'stage1',
    ('in_progress', False, False, True, False, 'rgb', True, True): 'stage1',
    ('in_progress', False, False, True, True, 'none', False, False): 'stage1',
    ('in_progress', False, False, True, True, 'none', False, True): 'stage1',
    ('in_progress', False, False, True, True, 'none', True, False): 'stage1',
    ('in_progress', False, False, True, True, 'none', True, True): 'stage1',
    ('in_progress', False, False, True, True, 'objmap', False, False): 'stage1',
    ('in_progress', False, False, True, True, 'objmap', False, True): 'stage1',
    ('in_progress', False, False, True, True, 'objmap', True, False): 'stage1',
    ('in_progress', False, False, True, True, 'objmap', True, True): 'stage1',
    ('in_progress', False, False, True, True, 'objmap_terminal', False, False): 'complete',
    ('in_progress', False, False, True, True, 'objmap_terminal', False, True): 'stage1',
    ('in_progress', False, False, True, True, 'objmap_terminal', True, False): 'complete',
    ('in_progress', False, False, True, True, 'objmap_terminal', True, True): 'stage1',
    ('in_progress', False, False, True, True, 'rgb', False, False): 'stage1',
    ('in_progress', False, False, True, True, 'rgb', False, True): 'stage1',
    ('in_progress', False, False, True, True, 'rgb', True, False): 'stage1',
    ('in_progress', False, False, True, True, 'rgb', True, True): 'stage1',
    ('in_progress', False, True, False, False, 'none', False, False): 'stage2',
    ('in_progress', False, True, False, False, 'none', False, True): 'stage1',
    ('in_progress', False, True, False, False, 'none', True, False): 'stage2',
    ('in_progress', False, True, False, False, 'none', True, True): 'stage1',
    ('in_progress', False, True, False, False, 'objmap', False, False): 'stage2',
    ('in_progress', False, True, False, False, 'objmap', False, True): 'stage1',
    ('in_progress', False, True, False, False, 'objmap', True, False): 'stage2',
    ('in_progress', False, True, False, False, 'objmap', True, True): 'stage1',
    ('in_progress', False, True, False, False, 'objmap_terminal', False, False): 'complete',
    ('in_progress', False, True, False, False, 'objmap_terminal', False, True): 'stage1',
    ('in_progress', False, True, False, False, 'objmap_terminal', True, False): 'complete',
    ('in_progress', False, True, False, False, 'objmap_terminal', True, True): 'stage1',
    ('in_progress', False, True, False, False, 'rgb', False, False): 'stage2',
    ('in_progress', False, True, False, False, 'rgb', False, True): 'stage1',
    ('in_progress', False, True, False, False, 'rgb', True, False): 'stage2',
    ('in_progress', False, True, False, False, 'rgb', True, True): 'stage1',
    ('in_progress', False, True, False, True, 'none', False, False): 'complete',
    ('in_progress', False, True, False, True, 'none', False, True): 'stage1',
    ('in_progress', False, True, False, True, 'none', True, False): 'complete',
    ('in_progress', False, True, False, True, 'none', True, True): 'stage1',
    ('in_progress', False, True, False, True, 'objmap', False, False): 'stage2',
    ('in_progress', False, True, False, True, 'objmap', False, True): 'stage1',
    ('in_progress', False, True, False, True, 'objmap', True, False): 'stage2',
    ('in_progress', False, True, False, True, 'objmap', True, True): 'stage1',
    ('in_progress', False, True, False, True, 'objmap_terminal', False, False): 'complete',
    ('in_progress', False, True, False, True, 'objmap_terminal', False, True): 'stage1',
    ('in_progress', False, True, False, True, 'objmap_terminal', True, False): 'complete',
    ('in_progress', False, True, False, True, 'objmap_terminal', True, True): 'stage1',
    ('in_progress', False, True, False, True, 'rgb', False, False): 'stage2',
    ('in_progress', False, True, False, True, 'rgb', False, True): 'stage1',
    ('in_progress', False, True, False, True, 'rgb', True, False): 'stage2',
    ('in_progress', False, True, False, True, 'rgb', True, True): 'stage1',
    ('in_progress', False, True, True, False, 'none', False, False): 'stage3',
    ('in_progress', False, True, True, False, 'none', False, True): 'stage1',
    ('in_progress', False, True, True, False, 'none', True, False): 'stage3',
    ('in_progress', False, True, True, False, 'none', True, True): 'stage1',
    ('in_progress', False, True, True, False, 'objmap', False, False): 'stage3',
    ('in_progress', False, True, True, False, 'objmap', False, True): 'stage1',
    ('in_progress', False, True, True, False, 'objmap', True, False): 'stage3',
    ('in_progress', False, True, True, False, 'objmap', True, True): 'stage1',
    ('in_progress', False, True, True, False, 'objmap_terminal', False, False): 'complete',
    ('in_progress', False, True, True, False, 'objmap_terminal', False, True): 'stage1',
    ('in_progress', False, True, True, False, 'objmap_terminal', True, False): 'complete',
    ('in_progress', False, True, True, False, 'objmap_terminal', True, True): 'stage1',
    ('in_progress', False, True, True, False, 'rgb', False, False): 'stage3',
    ('in_progress', False, True, True, False, 'rgb', False, True): 'stage1',
    ('in_progress', False, True, True, False, 'rgb', True, False): 'stage3',
    ('in_progress', False, True, True, False, 'rgb', True, True): 'stage1',
    ('in_progress', False, True, True, True, 'none', False, False): 'complete',
    ('in_progress', False, True, True, True, 'none', False, True): 'stage1',
    ('in_progress', False, True, True, True, 'none', True, False): 'complete',
    ('in_progress', False, True, True, True, 'none', True, True): 'stage1',
    ('in_progress', False, True, True, True, 'objmap', False, False): 'stage3',
    ('in_progress', False, True, True, True, 'objmap', False, True): 'stage1',
    ('in_progress', False, True, True, True, 'objmap', True, False): 'stage3',
    ('in_progress', False, True, True, True, 'objmap', True, True): 'stage1',
    ('in_progress', False, True, True, True, 'objmap_terminal', False, False): 'complete',
    ('in_progress', False, True, True, True, 'objmap_terminal', False, True): 'stage1',
    ('in_progress', False, True, True, True, 'objmap_terminal', True, False): 'complete',
    ('in_progress', False, True, True, True, 'objmap_terminal', True, True): 'stage1',
    ('in_progress', False, True, True, True, 'rgb', False, False): 'stage3',
    ('in_progress', False, True, True, True, 'rgb', False, True): 'stage1',
    ('in_progress', False, True, True, True, 'rgb', True, False): 'stage3',
    ('in_progress', False, True, True, True, 'rgb', True, True): 'stage1',
    ('in_progress', True, False, False, False, 'none', False, False): 'stage1',
    ('in_progress', True, False, False, False, 'none', False, True): 'stage1',
    ('in_progress', True, False, False, False, 'none', True, False): 'stage1',
    ('in_progress', True, False, False, False, 'none', True, True): 'stage1',
    ('in_progress', True, False, False, False, 'objmap', False, False): 'stage1',
    ('in_progress', True, False, False, False, 'objmap', False, True): 'stage1',
    ('in_progress', True, False, False, False, 'objmap', True, False): 'stage1',
    ('in_progress', True, False, False, False, 'objmap', True, True): 'stage1',
    ('in_progress', True, False, False, False, 'objmap_terminal', False, False): 'complete',
    ('in_progress', True, False, False, False, 'objmap_terminal', False, True): 'stage1',
    ('in_progress', True, False, False, False, 'objmap_terminal', True, False): 'complete',
    ('in_progress', True, False, False, False, 'objmap_terminal', True, True): 'stage1',
    ('in_progress', True, False, False, False, 'rgb', False, False): 'stage1',
    ('in_progress', True, False, False, False, 'rgb', False, True): 'stage1',
    ('in_progress', True, False, False, False, 'rgb', True, False): 'stage1',
    ('in_progress', True, False, False, False, 'rgb', True, True): 'stage1',
    ('in_progress', True, False, False, True, 'none', False, False): 'stage1',
    ('in_progress', True, False, False, True, 'none', False, True): 'stage1',
    ('in_progress', True, False, False, True, 'none', True, False): 'stage1',
    ('in_progress', True, False, False, True, 'none', True, True): 'stage1',
    ('in_progress', True, False, False, True, 'objmap', False, False): 'stage1',
    ('in_progress', True, False, False, True, 'objmap', False, True): 'stage1',
    ('in_progress', True, False, False, True, 'objmap', True, False): 'stage1',
    ('in_progress', True, False, False, True, 'objmap', True, True): 'stage1',
    ('in_progress', True, False, False, True, 'objmap_terminal', False, False): 'complete',
    ('in_progress', True, False, False, True, 'objmap_terminal', False, True): 'stage1',
    ('in_progress', True, False, False, True, 'objmap_terminal', True, False): 'complete',
    ('in_progress', True, False, False, True, 'objmap_terminal', True, True): 'stage1',
    ('in_progress', True, False, False, True, 'rgb', False, False): 'stage1',
    ('in_progress', True, False, False, True, 'rgb', False, True): 'stage1',
    ('in_progress', True, False, False, True, 'rgb', True, False): 'stage1',
    ('in_progress', True, False, False, True, 'rgb', True, True): 'stage1',
    ('in_progress', True, False, True, False, 'none', False, False): 'stage1',
    ('in_progress', True, False, True, False, 'none', False, True): 'stage1',
    ('in_progress', True, False, True, False, 'none', True, False): 'stage1',
    ('in_progress', True, False, True, False, 'none', True, True): 'stage1',
    ('in_progress', True, False, True, False, 'objmap', False, False): 'stage1',
    ('in_progress', True, False, True, False, 'objmap', False, True): 'stage1',
    ('in_progress', True, False, True, False, 'objmap', True, False): 'stage1',
    ('in_progress', True, False, True, False, 'objmap', True, True): 'stage1',
    ('in_progress', True, False, True, False, 'objmap_terminal', False, False): 'complete',
    ('in_progress', True, False, True, False, 'objmap_terminal', False, True): 'stage1',
    ('in_progress', True, False, True, False, 'objmap_terminal', True, False): 'complete',
    ('in_progress', True, False, True, False, 'objmap_terminal', True, True): 'stage1',
    ('in_progress', True, False, True, False, 'rgb', False, False): 'stage1',
    ('in_progress', True, False, True, False, 'rgb', False, True): 'stage1',
    ('in_progress', True, False, True, False, 'rgb', True, False): 'stage1',
    ('in_progress', True, False, True, False, 'rgb', True, True): 'stage1',
    ('in_progress', True, False, True, True, 'none', False, False): 'stage1',
    ('in_progress', True, False, True, True, 'none', False, True): 'stage1',
    ('in_progress', True, False, True, True, 'none', True, False): 'stage1',
    ('in_progress', True, False, True, True, 'none', True, True): 'stage1',
    ('in_progress', True, False, True, True, 'objmap', False, False): 'stage1',
    ('in_progress', True, False, True, True, 'objmap', False, True): 'stage1',
    ('in_progress', True, False, True, True, 'objmap', True, False): 'stage1',
    ('in_progress', True, False, True, True, 'objmap', True, True): 'stage1',
    ('in_progress', True, False, True, True, 'objmap_terminal', False, False): 'complete',
    ('in_progress', True, False, True, True, 'objmap_terminal', False, True): 'stage1',
    ('in_progress', True, False, True, True, 'objmap_terminal', True, False): 'complete',
    ('in_progress', True, False, True, True, 'objmap_terminal', True, True): 'stage1',
    ('in_progress', True, False, True, True, 'rgb', False, False): 'stage1',
    ('in_progress', True, False, True, True, 'rgb', False, True): 'stage1',
    ('in_progress', True, False, True, True, 'rgb', True, False): 'stage1',
    ('in_progress', True, False, True, True, 'rgb', True, True): 'stage1',
    ('in_progress', True, True, False, False, 'none', False, False): 'stage2',
    ('in_progress', True, True, False, False, 'none', False, True): 'stage1',
    ('in_progress', True, True, False, False, 'none', True, False): 'stage2',
    ('in_progress', True, True, False, False, 'none', True, True): 'stage1',
    ('in_progress', True, True, False, False, 'objmap', False, False): 'stage2',
    ('in_progress', True, True, False, False, 'objmap', False, True): 'stage1',
    ('in_progress', True, True, False, False, 'objmap', True, False): 'stage2',
    ('in_progress', True, True, False, False, 'objmap', True, True): 'stage1',
    ('in_progress', True, True, False, False, 'objmap_terminal', False, False): 'complete',
    ('in_progress', True, True, False, False, 'objmap_terminal', False, True): 'stage1',
    ('in_progress', True, True, False, False, 'objmap_terminal', True, False): 'complete',
    ('in_progress', True, True, False, False, 'objmap_terminal', True, True): 'stage1',
    ('in_progress', True, True, False, False, 'rgb', False, False): 'stage2',
    ('in_progress', True, True, False, False, 'rgb', False, True): 'stage1',
    ('in_progress', True, True, False, False, 'rgb', True, False): 'stage2',
    ('in_progress', True, True, False, False, 'rgb', True, True): 'stage1',
    ('in_progress', True, True, False, True, 'none', False, False): 'complete',
    ('in_progress', True, True, False, True, 'none', False, True): 'stage1',
    ('in_progress', True, True, False, True, 'none', True, False): 'complete',
    ('in_progress', True, True, False, True, 'none', True, True): 'stage1',
    ('in_progress', True, True, False, True, 'objmap', False, False): 'stage2',
    ('in_progress', True, True, False, True, 'objmap', False, True): 'stage1',
    ('in_progress', True, True, False, True, 'objmap', True, False): 'stage2',
    ('in_progress', True, True, False, True, 'objmap', True, True): 'stage1',
    ('in_progress', True, True, False, True, 'objmap_terminal', False, False): 'complete',
    ('in_progress', True, True, False, True, 'objmap_terminal', False, True): 'stage1',
    ('in_progress', True, True, False, True, 'objmap_terminal', True, False): 'complete',
    ('in_progress', True, True, False, True, 'objmap_terminal', True, True): 'stage1',
    ('in_progress', True, True, False, True, 'rgb', False, False): 'stage2',
    ('in_progress', True, True, False, True, 'rgb', False, True): 'stage1',
    ('in_progress', True, True, False, True, 'rgb', True, False): 'stage2',
    ('in_progress', True, True, False, True, 'rgb', True, True): 'stage1',
    ('in_progress', True, True, True, False, 'none', False, False): 'stage3',
    ('in_progress', True, True, True, False, 'none', False, True): 'stage1',
    ('in_progress', True, True, True, False, 'none', True, False): 'stage3',
    ('in_progress', True, True, True, False, 'none', True, True): 'stage1',
    ('in_progress', True, True, True, False, 'objmap', False, False): 'stage3',
    ('in_progress', True, True, True, False, 'objmap', False, True): 'stage1',
    ('in_progress', True, True, True, False, 'objmap', True, False): 'stage3',
    ('in_progress', True, True, True, False, 'objmap', True, True): 'stage1',
    ('in_progress', True, True, True, False, 'objmap_terminal', False, False): 'complete',
    ('in_progress', True, True, True, False, 'objmap_terminal', False, True): 'stage1',
    ('in_progress', True, True, True, False, 'objmap_terminal', True, False): 'complete',
    ('in_progress', True, True, True, False, 'objmap_terminal', True, True): 'stage1',
    ('in_progress', True, True, True, False, 'rgb', False, False): 'stage3',
    ('in_progress', True, True, True, False, 'rgb', False, True): 'stage1',
    ('in_progress', True, True, True, False, 'rgb', True, False): 'stage3',
    ('in_progress', True, True, True, False, 'rgb', True, True): 'stage1',
    ('in_progress', True, True, True, True, 'none', False, False): 'complete',
    ('in_progress', True, True, True, True, 'none', False, True): 'stage1',
    ('in_progress', True, True, True, True, 'none', True, False): 'complete',
    ('in_progress', True, True, True, True, 'none', True, True): 'stage1',
    ('in_progress', True, True, True, True, 'objmap', False, False): 'stage3',
    ('in_progress', True, True, True, True, 'objmap', False, True): 'stage1',
    ('in_progress', True, True, True, True, 'objmap', True, False): 'stage3',
    ('in_progress', True, True, True, True, 'objmap', True, True): 'stage1',
    ('in_progress', True, True, True, True, 'objmap_terminal', False, False): 'complete',
    ('in_progress', True, True, True, True, 'objmap_terminal', False, True): 'stage1',
    ('in_progress', True, True, True, True, 'objmap_terminal', True, False): 'complete',
    ('in_progress', True, True, True, True, 'objmap_terminal', True, True): 'stage1',
    ('in_progress', True, True, True, True, 'rgb', False, False): 'stage3',
    ('in_progress', True, True, True, True, 'rgb', False, True): 'stage1',
    ('in_progress', True, True, True, True, 'rgb', True, False): 'stage3',
    ('in_progress', True, True, True, True, 'rgb', True, True): 'stage1',
    ('in_progress_matching_work_id', False, False, False, False, 'none', False, False): 'stage1',
    ('in_progress_matching_work_id', False, False, False, False, 'none', False, True): 'stage1',
    ('in_progress_matching_work_id', False, False, False, False, 'none', True, False): 'stage1',
    ('in_progress_matching_work_id', False, False, False, False, 'none', True, True): 'stage1',
    ('in_progress_matching_work_id', False, False, False, False, 'objmap', False, False): 'stage1',
    ('in_progress_matching_work_id', False, False, False, False, 'objmap', False, True): 'stage1',
    ('in_progress_matching_work_id', False, False, False, False, 'objmap', True, False): 'stage1',
    ('in_progress_matching_work_id', False, False, False, False, 'objmap', True, True): 'stage1',
    ('in_progress_matching_work_id', False, False, False, False, 'objmap_terminal', False, False): 'complete',
    ('in_progress_matching_work_id', False, False, False, False, 'objmap_terminal', False, True): 'stage1',
    ('in_progress_matching_work_id', False, False, False, False, 'objmap_terminal', True, False): 'complete',
    ('in_progress_matching_work_id', False, False, False, False, 'objmap_terminal', True, True): 'stage1',
    ('in_progress_matching_work_id', False, False, False, False, 'rgb', False, False): 'stage1',
    ('in_progress_matching_work_id', False, False, False, False, 'rgb', False, True): 'stage1',
    ('in_progress_matching_work_id', False, False, False, False, 'rgb', True, False): 'stage1',
    ('in_progress_matching_work_id', False, False, False, False, 'rgb', True, True): 'stage1',
    ('in_progress_matching_work_id', False, False, False, True, 'none', False, False): 'stage1',
    ('in_progress_matching_work_id', False, False, False, True, 'none', False, True): 'stage1',
    ('in_progress_matching_work_id', False, False, False, True, 'none', True, False): 'stage1',
    ('in_progress_matching_work_id', False, False, False, True, 'none', True, True): 'stage1',
    ('in_progress_matching_work_id', False, False, False, True, 'objmap', False, False): 'stage1',
    ('in_progress_matching_work_id', False, False, False, True, 'objmap', False, True): 'stage1',
    ('in_progress_matching_work_id', False, False, False, True, 'objmap', True, False): 'stage1',
    ('in_progress_matching_work_id', False, False, False, True, 'objmap', True, True): 'stage1',
    ('in_progress_matching_work_id', False, False, False, True, 'objmap_terminal', False, False): 'complete',
    ('in_progress_matching_work_id', False, False, False, True, 'objmap_terminal', False, True): 'stage1',
    ('in_progress_matching_work_id', False, False, False, True, 'objmap_terminal', True, False): 'complete',
    ('in_progress_matching_work_id', False, False, False, True, 'objmap_terminal', True, True): 'stage1',
    ('in_progress_matching_work_id', False, False, False, True, 'rgb', False, False): 'stage1',
    ('in_progress_matching_work_id', False, False, False, True, 'rgb', False, True): 'stage1',
    ('in_progress_matching_work_id', False, False, False, True, 'rgb', True, False): 'stage1',
    ('in_progress_matching_work_id', False, False, False, True, 'rgb', True, True): 'stage1',
    ('in_progress_matching_work_id', False, False, True, False, 'none', False, False): 'stage1',
    ('in_progress_matching_work_id', False, False, True, False, 'none', False, True): 'stage1',
    ('in_progress_matching_work_id', False, False, True, False, 'none', True, False): 'stage1',
    ('in_progress_matching_work_id', False, False, True, False, 'none', True, True): 'stage1',
    ('in_progress_matching_work_id', False, False, True, False, 'objmap', False, False): 'stage1',
    ('in_progress_matching_work_id', False, False, True, False, 'objmap', False, True): 'stage1',
    ('in_progress_matching_work_id', False, False, True, False, 'objmap', True, False): 'stage1',
    ('in_progress_matching_work_id', False, False, True, False, 'objmap', True, True): 'stage1',
    ('in_progress_matching_work_id', False, False, True, False, 'objmap_terminal', False, False): 'complete',
    ('in_progress_matching_work_id', False, False, True, False, 'objmap_terminal', False, True): 'stage1',
    ('in_progress_matching_work_id', False, False, True, False, 'objmap_terminal', True, False): 'complete',
    ('in_progress_matching_work_id', False, False, True, False, 'objmap_terminal', True, True): 'stage1',
    ('in_progress_matching_work_id', False, False, True, False, 'rgb', False, False): 'stage1',
    ('in_progress_matching_work_id', False, False, True, False, 'rgb', False, True): 'stage1',
    ('in_progress_matching_work_id', False, False, True, False, 'rgb', True, False): 'stage1',
    ('in_progress_matching_work_id', False, False, True, False, 'rgb', True, True): 'stage1',
    ('in_progress_matching_work_id', False, False, True, True, 'none', False, False): 'stage1',
    ('in_progress_matching_work_id', False, False, True, True, 'none', False, True): 'stage1',
    ('in_progress_matching_work_id', False, False, True, True, 'none', True, False): 'stage1',
    ('in_progress_matching_work_id', False, False, True, True, 'none', True, True): 'stage1',
    ('in_progress_matching_work_id', False, False, True, True, 'objmap', False, False): 'stage1',
    ('in_progress_matching_work_id', False, False, True, True, 'objmap', False, True): 'stage1',
    ('in_progress_matching_work_id', False, False, True, True, 'objmap', True, False): 'stage1',
    ('in_progress_matching_work_id', False, False, True, True, 'objmap', True, True): 'stage1',
    ('in_progress_matching_work_id', False, False, True, True, 'objmap_terminal', False, False): 'complete',
    ('in_progress_matching_work_id', False, False, True, True, 'objmap_terminal', False, True): 'stage1',
    ('in_progress_matching_work_id', False, False, True, True, 'objmap_terminal', True, False): 'complete',
    ('in_progress_matching_work_id', False, False, True, True, 'objmap_terminal', True, True): 'stage1',
    ('in_progress_matching_work_id', False, False, True, True, 'rgb', False, False): 'stage1',
    ('in_progress_matching_work_id', False, False, True, True, 'rgb', False, True): 'stage1',
    ('in_progress_matching_work_id', False, False, True, True, 'rgb', True, False): 'stage1',
    ('in_progress_matching_work_id', False, False, True, True, 'rgb', True, True): 'stage1',
    ('in_progress_matching_work_id', False, True, False, False, 'none', False, False): 'stage2',
    ('in_progress_matching_work_id', False, True, False, False, 'none', False, True): 'stage2',
    ('in_progress_matching_work_id', False, True, False, False, 'none', True, False): 'stage2',
    ('in_progress_matching_work_id', False, True, False, False, 'none', True, True): 'stage2',
    ('in_progress_matching_work_id', False, True, False, False, 'objmap', False, False): 'stage2',
    ('in_progress_matching_work_id', False, True, False, False, 'objmap', False, True): 'stage2',
    ('in_progress_matching_work_id', False, True, False, False, 'objmap', True, False): 'stage2',
    ('in_progress_matching_work_id', False, True, False, False, 'objmap', True, True): 'stage2',
    ('in_progress_matching_work_id', False, True, False, False, 'objmap_terminal', False, False): 'complete',
    ('in_progress_matching_work_id', False, True, False, False, 'objmap_terminal', False, True): 'stage2',
    ('in_progress_matching_work_id', False, True, False, False, 'objmap_terminal', True, False): 'complete',
    ('in_progress_matching_work_id', False, True, False, False, 'objmap_terminal', True, True): 'stage2',
    ('in_progress_matching_work_id', False, True, False, False, 'rgb', False, False): 'stage2',
    ('in_progress_matching_work_id', False, True, False, False, 'rgb', False, True): 'stage2',
    ('in_progress_matching_work_id', False, True, False, False, 'rgb', True, False): 'stage2',
    ('in_progress_matching_work_id', False, True, False, False, 'rgb', True, True): 'stage2',
    ('in_progress_matching_work_id', False, True, False, True, 'none', False, False): 'complete',
    ('in_progress_matching_work_id', False, True, False, True, 'none', False, True): 'stage2',
    ('in_progress_matching_work_id', False, True, False, True, 'none', True, False): 'complete',
    ('in_progress_matching_work_id', False, True, False, True, 'none', True, True): 'stage2',
    ('in_progress_matching_work_id', False, True, False, True, 'objmap', False, False): 'stage2',
    ('in_progress_matching_work_id', False, True, False, True, 'objmap', False, True): 'stage2',
    ('in_progress_matching_work_id', False, True, False, True, 'objmap', True, False): 'stage2',
    ('in_progress_matching_work_id', False, True, False, True, 'objmap', True, True): 'stage2',
    ('in_progress_matching_work_id', False, True, False, True, 'objmap_terminal', False, False): 'complete',
    ('in_progress_matching_work_id', False, True, False, True, 'objmap_terminal', False, True): 'stage2',
    ('in_progress_matching_work_id', False, True, False, True, 'objmap_terminal', True, False): 'complete',
    ('in_progress_matching_work_id', False, True, False, True, 'objmap_terminal', True, True): 'stage2',
    ('in_progress_matching_work_id', False, True, False, True, 'rgb', False, False): 'stage2',
    ('in_progress_matching_work_id', False, True, False, True, 'rgb', False, True): 'stage2',
    ('in_progress_matching_work_id', False, True, False, True, 'rgb', True, False): 'stage2',
    ('in_progress_matching_work_id', False, True, False, True, 'rgb', True, True): 'stage2',
    ('in_progress_matching_work_id', False, True, True, False, 'none', False, False): 'stage3',
    ('in_progress_matching_work_id', False, True, True, False, 'none', False, True): 'stage3',
    ('in_progress_matching_work_id', False, True, True, False, 'none', True, False): 'stage3',
    ('in_progress_matching_work_id', False, True, True, False, 'none', True, True): 'stage3',
    ('in_progress_matching_work_id', False, True, True, False, 'objmap', False, False): 'stage3',
    ('in_progress_matching_work_id', False, True, True, False, 'objmap', False, True): 'stage3',
    ('in_progress_matching_work_id', False, True, True, False, 'objmap', True, False): 'stage3',
    ('in_progress_matching_work_id', False, True, True, False, 'objmap', True, True): 'stage3',
    ('in_progress_matching_work_id', False, True, True, False, 'objmap_terminal', False, False): 'complete',
    ('in_progress_matching_work_id', False, True, True, False, 'objmap_terminal', False, True): 'stage3',
    ('in_progress_matching_work_id', False, True, True, False, 'objmap_terminal', True, False): 'complete',
    ('in_progress_matching_work_id', False, True, True, False, 'objmap_terminal', True, True): 'stage3',
    ('in_progress_matching_work_id', False, True, True, False, 'rgb', False, False): 'stage3',
    ('in_progress_matching_work_id', False, True, True, False, 'rgb', False, True): 'stage3',
    ('in_progress_matching_work_id', False, True, True, False, 'rgb', True, False): 'stage3',
    ('in_progress_matching_work_id', False, True, True, False, 'rgb', True, True): 'stage3',
    ('in_progress_matching_work_id', False, True, True, True, 'none', False, False): 'complete',
    ('in_progress_matching_work_id', False, True, True, True, 'none', False, True): 'stage3',
    ('in_progress_matching_work_id', False, True, True, True, 'none', True, False): 'complete',
    ('in_progress_matching_work_id', False, True, True, True, 'none', True, True): 'stage3',
    ('in_progress_matching_work_id', False, True, True, True, 'objmap', False, False): 'stage3',
    ('in_progress_matching_work_id', False, True, True, True, 'objmap', False, True): 'stage3',
    ('in_progress_matching_work_id', False, True, True, True, 'objmap', True, False): 'stage3',
    ('in_progress_matching_work_id', False, True, True, True, 'objmap', True, True): 'stage3',
    ('in_progress_matching_work_id', False, True, True, True, 'objmap_terminal', False, False): 'complete',
    ('in_progress_matching_work_id', False, True, True, True, 'objmap_terminal', False, True): 'stage3',
    ('in_progress_matching_work_id', False, True, True, True, 'objmap_terminal', True, False): 'complete',
    ('in_progress_matching_work_id', False, True, True, True, 'objmap_terminal', True, True): 'stage3',
    ('in_progress_matching_work_id', False, True, True, True, 'rgb', False, False): 'stage3',
    ('in_progress_matching_work_id', False, True, True, True, 'rgb', False, True): 'stage3',
    ('in_progress_matching_work_id', False, True, True, True, 'rgb', True, False): 'stage3',
    ('in_progress_matching_work_id', False, True, True, True, 'rgb', True, True): 'stage3',
    ('in_progress_matching_work_id', True, False, False, False, 'none', False, False): 'stage1',
    ('in_progress_matching_work_id', True, False, False, False, 'none', False, True): 'stage1',
    ('in_progress_matching_work_id', True, False, False, False, 'none', True, False): 'stage1',
    ('in_progress_matching_work_id', True, False, False, False, 'none', True, True): 'stage1',
    ('in_progress_matching_work_id', True, False, False, False, 'objmap', False, False): 'stage1',
    ('in_progress_matching_work_id', True, False, False, False, 'objmap', False, True): 'stage1',
    ('in_progress_matching_work_id', True, False, False, False, 'objmap', True, False): 'stage1',
    ('in_progress_matching_work_id', True, False, False, False, 'objmap', True, True): 'stage1',
    ('in_progress_matching_work_id', True, False, False, False, 'objmap_terminal', False, False): 'complete',
    ('in_progress_matching_work_id', True, False, False, False, 'objmap_terminal', False, True): 'stage1',
    ('in_progress_matching_work_id', True, False, False, False, 'objmap_terminal', True, False): 'complete',
    ('in_progress_matching_work_id', True, False, False, False, 'objmap_terminal', True, True): 'stage1',
    ('in_progress_matching_work_id', True, False, False, False, 'rgb', False, False): 'stage1',
    ('in_progress_matching_work_id', True, False, False, False, 'rgb', False, True): 'stage1',
    ('in_progress_matching_work_id', True, False, False, False, 'rgb', True, False): 'stage1',
    ('in_progress_matching_work_id', True, False, False, False, 'rgb', True, True): 'stage1',
    ('in_progress_matching_work_id', True, False, False, True, 'none', False, False): 'stage1',
    ('in_progress_matching_work_id', True, False, False, True, 'none', False, True): 'stage1',
    ('in_progress_matching_work_id', True, False, False, True, 'none', True, False): 'stage1',
    ('in_progress_matching_work_id', True, False, False, True, 'none', True, True): 'stage1',
    ('in_progress_matching_work_id', True, False, False, True, 'objmap', False, False): 'stage1',
    ('in_progress_matching_work_id', True, False, False, True, 'objmap', False, True): 'stage1',
    ('in_progress_matching_work_id', True, False, False, True, 'objmap', True, False): 'stage1',
    ('in_progress_matching_work_id', True, False, False, True, 'objmap', True, True): 'stage1',
    ('in_progress_matching_work_id', True, False, False, True, 'objmap_terminal', False, False): 'complete',
    ('in_progress_matching_work_id', True, False, False, True, 'objmap_terminal', False, True): 'stage1',
    ('in_progress_matching_work_id', True, False, False, True, 'objmap_terminal', True, False): 'complete',
    ('in_progress_matching_work_id', True, False, False, True, 'objmap_terminal', True, True): 'stage1',
    ('in_progress_matching_work_id', True, False, False, True, 'rgb', False, False): 'stage1',
    ('in_progress_matching_work_id', True, False, False, True, 'rgb', False, True): 'stage1',
    ('in_progress_matching_work_id', True, False, False, True, 'rgb', True, False): 'stage1',
    ('in_progress_matching_work_id', True, False, False, True, 'rgb', True, True): 'stage1',
    ('in_progress_matching_work_id', True, False, True, False, 'none', False, False): 'stage1',
    ('in_progress_matching_work_id', True, False, True, False, 'none', False, True): 'stage1',
    ('in_progress_matching_work_id', True, False, True, False, 'none', True, False): 'stage1',
    ('in_progress_matching_work_id', True, False, True, False, 'none', True, True): 'stage1',
    ('in_progress_matching_work_id', True, False, True, False, 'objmap', False, False): 'stage1',
    ('in_progress_matching_work_id', True, False, True, False, 'objmap', False, True): 'stage1',
    ('in_progress_matching_work_id', True, False, True, False, 'objmap', True, False): 'stage1',
    ('in_progress_matching_work_id', True, False, True, False, 'objmap', True, True): 'stage1',
    ('in_progress_matching_work_id', True, False, True, False, 'objmap_terminal', False, False): 'complete',
    ('in_progress_matching_work_id', True, False, True, False, 'objmap_terminal', False, True): 'stage1',
    ('in_progress_matching_work_id', True, False, True, False, 'objmap_terminal', True, False): 'complete',
    ('in_progress_matching_work_id', True, False, True, False, 'objmap_terminal', True, True): 'stage1',
    ('in_progress_matching_work_id', True, False, True, False, 'rgb', False, False): 'stage1',
    ('in_progress_matching_work_id', True, False, True, False, 'rgb', False, True): 'stage1',
    ('in_progress_matching_work_id', True, False, True, False, 'rgb', True, False): 'stage1',
    ('in_progress_matching_work_id', True, False, True, False, 'rgb', True, True): 'stage1',
    ('in_progress_matching_work_id', True, False, True, True, 'none', False, False): 'stage1',
    ('in_progress_matching_work_id', True, False, True, True, 'none', False, True): 'stage1',
    ('in_progress_matching_work_id', True, False, True, True, 'none', True, False): 'stage1',
    ('in_progress_matching_work_id', True, False, True, True, 'none', True, True): 'stage1',
    ('in_progress_matching_work_id', True, False, True, True, 'objmap', False, False): 'stage1',
    ('in_progress_matching_work_id', True, False, True, True, 'objmap', False, True): 'stage1',
    ('in_progress_matching_work_id', True, False, True, True, 'objmap', True, False): 'stage1',
    ('in_progress_matching_work_id', True, False, True, True, 'objmap', True, True): 'stage1',
    ('in_progress_matching_work_id', True, False, True, True, 'objmap_terminal', False, False): 'complete',
    ('in_progress_matching_work_id', True, False, True, True, 'objmap_terminal', False, True): 'stage1',
    ('in_progress_matching_work_id', True, False, True, True, 'objmap_terminal', True, False): 'complete',
    ('in_progress_matching_work_id', True, False, True, True, 'objmap_terminal', True, True): 'stage1',
    ('in_progress_matching_work_id', True, False, True, True, 'rgb', False, False): 'stage1',
    ('in_progress_matching_work_id', True, False, True, True, 'rgb', False, True): 'stage1',
    ('in_progress_matching_work_id', True, False, True, True, 'rgb', True, False): 'stage1',
    ('in_progress_matching_work_id', True, False, True, True, 'rgb', True, True): 'stage1',
    ('in_progress_matching_work_id', True, True, False, False, 'none', False, False): 'stage2',
    ('in_progress_matching_work_id', True, True, False, False, 'none', False, True): 'stage2',
    ('in_progress_matching_work_id', True, True, False, False, 'none', True, False): 'stage2',
    ('in_progress_matching_work_id', True, True, False, False, 'none', True, True): 'stage2',
    ('in_progress_matching_work_id', True, True, False, False, 'objmap', False, False): 'stage2',
    ('in_progress_matching_work_id', True, True, False, False, 'objmap', False, True): 'stage2',
    ('in_progress_matching_work_id', True, True, False, False, 'objmap', True, False): 'stage2',
    ('in_progress_matching_work_id', True, True, False, False, 'objmap', True, True): 'stage2',
    ('in_progress_matching_work_id', True, True, False, False, 'objmap_terminal', False, False): 'complete',
    ('in_progress_matching_work_id', True, True, False, False, 'objmap_terminal', False, True): 'stage2',
    ('in_progress_matching_work_id', True, True, False, False, 'objmap_terminal', True, False): 'complete',
    ('in_progress_matching_work_id', True, True, False, False, 'objmap_terminal', True, True): 'stage2',
    ('in_progress_matching_work_id', True, True, False, False, 'rgb', False, False): 'stage2',
    ('in_progress_matching_work_id', True, True, False, False, 'rgb', False, True): 'stage2',
    ('in_progress_matching_work_id', True, True, False, False, 'rgb', True, False): 'stage2',
    ('in_progress_matching_work_id', True, True, False, False, 'rgb', True, True): 'stage2',
    ('in_progress_matching_work_id', True, True, False, True, 'none', False, False): 'complete',
    ('in_progress_matching_work_id', True, True, False, True, 'none', False, True): 'stage3',
    ('in_progress_matching_work_id', True, True, False, True, 'none', True, False): 'complete',
    ('in_progress_matching_work_id', True, True, False, True, 'none', True, True): 'stage3',
    ('in_progress_matching_work_id', True, True, False, True, 'objmap', False, False): 'stage2',
    ('in_progress_matching_work_id', True, True, False, True, 'objmap', False, True): 'stage2',
    ('in_progress_matching_work_id', True, True, False, True, 'objmap', True, False): 'stage2',
    ('in_progress_matching_work_id', True, True, False, True, 'objmap', True, True): 'stage2',
    ('in_progress_matching_work_id', True, True, False, True, 'objmap_terminal', False, False): 'complete',
    ('in_progress_matching_work_id', True, True, False, True, 'objmap_terminal', False, True): 'stage2',
    ('in_progress_matching_work_id', True, True, False, True, 'objmap_terminal', True, False): 'complete',
    ('in_progress_matching_work_id', True, True, False, True, 'objmap_terminal', True, True): 'stage2',
    ('in_progress_matching_work_id', True, True, False, True, 'rgb', False, False): 'stage2',
    ('in_progress_matching_work_id', True, True, False, True, 'rgb', False, True): 'stage2',
    ('in_progress_matching_work_id', True, True, False, True, 'rgb', True, False): 'stage2',
    ('in_progress_matching_work_id', True, True, False, True, 'rgb', True, True): 'stage2',
    ('in_progress_matching_work_id', True, True, True, False, 'none', False, False): 'stage3',
    ('in_progress_matching_work_id', True, True, True, False, 'none', False, True): 'stage3',
    ('in_progress_matching_work_id', True, True, True, False, 'none', True, False): 'stage3',
    ('in_progress_matching_work_id', True, True, True, False, 'none', True, True): 'stage3',
    ('in_progress_matching_work_id', True, True, True, False, 'objmap', False, False): 'stage3',
    ('in_progress_matching_work_id', True, True, True, False, 'objmap', False, True): 'stage3',
    ('in_progress_matching_work_id', True, True, True, False, 'objmap', True, False): 'stage3',
    ('in_progress_matching_work_id', True, True, True, False, 'objmap', True, True): 'stage3',
    ('in_progress_matching_work_id', True, True, True, False, 'objmap_terminal', False, False): 'complete',
    ('in_progress_matching_work_id', True, True, True, False, 'objmap_terminal', False, True): 'stage3',
    ('in_progress_matching_work_id', True, True, True, False, 'objmap_terminal', True, False): 'complete',
    ('in_progress_matching_work_id', True, True, True, False, 'objmap_terminal', True, True): 'stage3',
    ('in_progress_matching_work_id', True, True, True, False, 'rgb', False, False): 'stage3',
    ('in_progress_matching_work_id', True, True, True, False, 'rgb', False, True): 'stage3',
    ('in_progress_matching_work_id', True, True, True, False, 'rgb', True, False): 'stage3',
    ('in_progress_matching_work_id', True, True, True, False, 'rgb', True, True): 'stage3',
    ('in_progress_matching_work_id', True, True, True, True, 'none', False, False): 'complete',
    ('in_progress_matching_work_id', True, True, True, True, 'none', False, True): 'stage3',
    ('in_progress_matching_work_id', True, True, True, True, 'none', True, False): 'complete',
    ('in_progress_matching_work_id', True, True, True, True, 'none', True, True): 'stage3',
    ('in_progress_matching_work_id', True, True, True, True, 'objmap', False, False): 'stage3',
    ('in_progress_matching_work_id', True, True, True, True, 'objmap', False, True): 'stage3',
    ('in_progress_matching_work_id', True, True, True, True, 'objmap', True, False): 'stage3',
    ('in_progress_matching_work_id', True, True, True, True, 'objmap', True, True): 'stage3',
    ('in_progress_matching_work_id', True, True, True, True, 'objmap_terminal', False, False): 'complete',
    ('in_progress_matching_work_id', True, True, True, True, 'objmap_terminal', False, True): 'stage3',
    ('in_progress_matching_work_id', True, True, True, True, 'objmap_terminal', True, False): 'complete',
    ('in_progress_matching_work_id', True, True, True, True, 'objmap_terminal', True, True): 'stage3',
    ('in_progress_matching_work_id', True, True, True, True, 'rgb', False, False): 'stage3',
    ('in_progress_matching_work_id', True, True, True, True, 'rgb', False, True): 'stage3',
    ('in_progress_matching_work_id', True, True, True, True, 'rgb', True, False): 'stage3',
    ('in_progress_matching_work_id', True, True, True, True, 'rgb', True, True): 'stage3',
    ('mismatched_work_id', False, False, False, False, 'none', False, False): 'stage2',
    ('mismatched_work_id', False, False, False, False, 'none', False, True): 'stage1',
    ('mismatched_work_id', False, False, False, False, 'none', True, False): 'stage2',
    ('mismatched_work_id', False, False, False, False, 'none', True, True): 'stage1',
    ('mismatched_work_id', False, False, False, False, 'objmap', False, False): 'stage2',
    ('mismatched_work_id', False, False, False, False, 'objmap', False, True): 'stage1',
    ('mismatched_work_id', False, False, False, False, 'objmap', True, False): 'stage2',
    ('mismatched_work_id', False, False, False, False, 'objmap', True, True): 'stage1',
    ('mismatched_work_id', False, False, False, False, 'objmap_terminal', False, False): 'complete',
    ('mismatched_work_id', False, False, False, False, 'objmap_terminal', False, True): 'stage1',
    ('mismatched_work_id', False, False, False, False, 'objmap_terminal', True, False): 'complete',
    ('mismatched_work_id', False, False, False, False, 'objmap_terminal', True, True): 'stage1',
    ('mismatched_work_id', False, False, False, False, 'rgb', False, False): 'stage2',
    ('mismatched_work_id', False, False, False, False, 'rgb', False, True): 'stage1',
    ('mismatched_work_id', False, False, False, False, 'rgb', True, False): 'stage2',
    ('mismatched_work_id', False, False, False, False, 'rgb', True, True): 'stage1',
    ('mismatched_work_id', False, False, False, True, 'none', False, False): 'complete',
    ('mismatched_work_id', False, False, False, True, 'none', False, True): 'stage1',
    ('mismatched_work_id', False, False, False, True, 'none', True, False): 'complete',
    ('mismatched_work_id', False, False, False, True, 'none', True, True): 'stage1',
    ('mismatched_work_id', False, False, False, True, 'objmap', False, False): 'stage2',
    ('mismatched_work_id', False, False, False, True, 'objmap', False, True): 'stage1',
    ('mismatched_work_id', False, False, False, True, 'objmap', True, False): 'stage2',
    ('mismatched_work_id', False, False, False, True, 'objmap', True, True): 'stage1',
    ('mismatched_work_id', False, False, False, True, 'objmap_terminal', False, False): 'complete',
    ('mismatched_work_id', False, False, False, True, 'objmap_terminal', False, True): 'stage1',
    ('mismatched_work_id', False, False, False, True, 'objmap_terminal', True, False): 'complete',
    ('mismatched_work_id', False, False, False, True, 'objmap_terminal', True, True): 'stage1',
    ('mismatched_work_id', False, False, False, True, 'rgb', False, False): 'stage2',
    ('mismatched_work_id', False, False, False, True, 'rgb', False, True): 'stage1',
    ('mismatched_work_id', False, False, False, True, 'rgb', True, False): 'stage2',
    ('mismatched_work_id', False, False, False, True, 'rgb', True, True): 'stage1',
    ('mismatched_work_id', False, False, True, False, 'none', False, False): 'stage2',
    ('mismatched_work_id', False, False, True, False, 'none', False, True): 'stage1',
    ('mismatched_work_id', False, False, True, False, 'none', True, False): 'stage2',
    ('mismatched_work_id', False, False, True, False, 'none', True, True): 'stage1',
    ('mismatched_work_id', False, False, True, False, 'objmap', False, False): 'stage2',
    ('mismatched_work_id', False, False, True, False, 'objmap', False, True): 'stage1',
    ('mismatched_work_id', False, False, True, False, 'objmap', True, False): 'stage2',
    ('mismatched_work_id', False, False, True, False, 'objmap', True, True): 'stage1',
    ('mismatched_work_id', False, False, True, False, 'objmap_terminal', False, False): 'complete',
    ('mismatched_work_id', False, False, True, False, 'objmap_terminal', False, True): 'stage1',
    ('mismatched_work_id', False, False, True, False, 'objmap_terminal', True, False): 'complete',
    ('mismatched_work_id', False, False, True, False, 'objmap_terminal', True, True): 'stage1',
    ('mismatched_work_id', False, False, True, False, 'rgb', False, False): 'stage2',
    ('mismatched_work_id', False, False, True, False, 'rgb', False, True): 'stage1',
    ('mismatched_work_id', False, False, True, False, 'rgb', True, False): 'stage2',
    ('mismatched_work_id', False, False, True, False, 'rgb', True, True): 'stage1',
    ('mismatched_work_id', False, False, True, True, 'none', False, False): 'complete',
    ('mismatched_work_id', False, False, True, True, 'none', False, True): 'stage1',
    ('mismatched_work_id', False, False, True, True, 'none', True, False): 'complete',
    ('mismatched_work_id', False, False, True, True, 'none', True, True): 'stage1',
    ('mismatched_work_id', False, False, True, True, 'objmap', False, False): 'stage2',
    ('mismatched_work_id', False, False, True, True, 'objmap', False, True): 'stage1',
    ('mismatched_work_id', False, False, True, True, 'objmap', True, False): 'stage2',
    ('mismatched_work_id', False, False, True, True, 'objmap', True, True): 'stage1',
    ('mismatched_work_id', False, False, True, True, 'objmap_terminal', False, False): 'complete',
    ('mismatched_work_id', False, False, True, True, 'objmap_terminal', False, True): 'stage1',
    ('mismatched_work_id', False, False, True, True, 'objmap_terminal', True, False): 'complete',
    ('mismatched_work_id', False, False, True, True, 'objmap_terminal', True, True): 'stage1',
    ('mismatched_work_id', False, False, True, True, 'rgb', False, False): 'stage2',
    ('mismatched_work_id', False, False, True, True, 'rgb', False, True): 'stage1',
    ('mismatched_work_id', False, False, True, True, 'rgb', True, False): 'stage2',
    ('mismatched_work_id', False, False, True, True, 'rgb', True, True): 'stage1',
    ('mismatched_work_id', False, True, False, False, 'none', False, False): 'stage2',
    ('mismatched_work_id', False, True, False, False, 'none', False, True): 'stage1',
    ('mismatched_work_id', False, True, False, False, 'none', True, False): 'stage2',
    ('mismatched_work_id', False, True, False, False, 'none', True, True): 'stage1',
    ('mismatched_work_id', False, True, False, False, 'objmap', False, False): 'stage2',
    ('mismatched_work_id', False, True, False, False, 'objmap', False, True): 'stage1',
    ('mismatched_work_id', False, True, False, False, 'objmap', True, False): 'stage2',
    ('mismatched_work_id', False, True, False, False, 'objmap', True, True): 'stage1',
    ('mismatched_work_id', False, True, False, False, 'objmap_terminal', False, False): 'complete',
    ('mismatched_work_id', False, True, False, False, 'objmap_terminal', False, True): 'stage1',
    ('mismatched_work_id', False, True, False, False, 'objmap_terminal', True, False): 'complete',
    ('mismatched_work_id', False, True, False, False, 'objmap_terminal', True, True): 'stage1',
    ('mismatched_work_id', False, True, False, False, 'rgb', False, False): 'stage2',
    ('mismatched_work_id', False, True, False, False, 'rgb', False, True): 'stage1',
    ('mismatched_work_id', False, True, False, False, 'rgb', True, False): 'stage2',
    ('mismatched_work_id', False, True, False, False, 'rgb', True, True): 'stage1',
    ('mismatched_work_id', False, True, False, True, 'none', False, False): 'complete',
    ('mismatched_work_id', False, True, False, True, 'none', False, True): 'stage1',
    ('mismatched_work_id', False, True, False, True, 'none', True, False): 'complete',
    ('mismatched_work_id', False, True, False, True, 'none', True, True): 'stage1',
    ('mismatched_work_id', False, True, False, True, 'objmap', False, False): 'stage2',
    ('mismatched_work_id', False, True, False, True, 'objmap', False, True): 'stage1',
    ('mismatched_work_id', False, True, False, True, 'objmap', True, False): 'stage2',
    ('mismatched_work_id', False, True, False, True, 'objmap', True, True): 'stage1',
    ('mismatched_work_id', False, True, False, True, 'objmap_terminal', False, False): 'complete',
    ('mismatched_work_id', False, True, False, True, 'objmap_terminal', False, True): 'stage1',
    ('mismatched_work_id', False, True, False, True, 'objmap_terminal', True, False): 'complete',
    ('mismatched_work_id', False, True, False, True, 'objmap_terminal', True, True): 'stage1',
    ('mismatched_work_id', False, True, False, True, 'rgb', False, False): 'stage2',
    ('mismatched_work_id', False, True, False, True, 'rgb', False, True): 'stage1',
    ('mismatched_work_id', False, True, False, True, 'rgb', True, False): 'stage2',
    ('mismatched_work_id', False, True, False, True, 'rgb', True, True): 'stage1',
    ('mismatched_work_id', False, True, True, False, 'none', False, False): 'stage3',
    ('mismatched_work_id', False, True, True, False, 'none', False, True): 'stage1',
    ('mismatched_work_id', False, True, True, False, 'none', True, False): 'stage3',
    ('mismatched_work_id', False, True, True, False, 'none', True, True): 'stage1',
    ('mismatched_work_id', False, True, True, False, 'objmap', False, False): 'stage3',
    ('mismatched_work_id', False, True, True, False, 'objmap', False, True): 'stage1',
    ('mismatched_work_id', False, True, True, False, 'objmap', True, False): 'stage3',
    ('mismatched_work_id', False, True, True, False, 'objmap', True, True): 'stage1',
    ('mismatched_work_id', False, True, True, False, 'objmap_terminal', False, False): 'complete',
    ('mismatched_work_id', False, True, True, False, 'objmap_terminal', False, True): 'stage1',
    ('mismatched_work_id', False, True, True, False, 'objmap_terminal', True, False): 'complete',
    ('mismatched_work_id', False, True, True, False, 'objmap_terminal', True, True): 'stage1',
    ('mismatched_work_id', False, True, True, False, 'rgb', False, False): 'stage3',
    ('mismatched_work_id', False, True, True, False, 'rgb', False, True): 'stage1',
    ('mismatched_work_id', False, True, True, False, 'rgb', True, False): 'stage3',
    ('mismatched_work_id', False, True, True, False, 'rgb', True, True): 'stage1',
    ('mismatched_work_id', False, True, True, True, 'none', False, False): 'complete',
    ('mismatched_work_id', False, True, True, True, 'none', False, True): 'stage1',
    ('mismatched_work_id', False, True, True, True, 'none', True, False): 'complete',
    ('mismatched_work_id', False, True, True, True, 'none', True, True): 'stage1',
    ('mismatched_work_id', False, True, True, True, 'objmap', False, False): 'stage3',
    ('mismatched_work_id', False, True, True, True, 'objmap', False, True): 'stage1',
    ('mismatched_work_id', False, True, True, True, 'objmap', True, False): 'stage3',
    ('mismatched_work_id', False, True, True, True, 'objmap', True, True): 'stage1',
    ('mismatched_work_id', False, True, True, True, 'objmap_terminal', False, False): 'complete',
    ('mismatched_work_id', False, True, True, True, 'objmap_terminal', False, True): 'stage1',
    ('mismatched_work_id', False, True, True, True, 'objmap_terminal', True, False): 'complete',
    ('mismatched_work_id', False, True, True, True, 'objmap_terminal', True, True): 'stage1',
    ('mismatched_work_id', False, True, True, True, 'rgb', False, False): 'stage3',
    ('mismatched_work_id', False, True, True, True, 'rgb', False, True): 'stage1',
    ('mismatched_work_id', False, True, True, True, 'rgb', True, False): 'stage3',
    ('mismatched_work_id', False, True, True, True, 'rgb', True, True): 'stage1',
    ('mismatched_work_id', True, False, False, False, 'none', False, False): 'complete',
    ('mismatched_work_id', True, False, False, False, 'none', False, True): 'stage1',
    ('mismatched_work_id', True, False, False, False, 'none', True, False): 'stage2',
    ('mismatched_work_id', True, False, False, False, 'none', True, True): 'stage1',
    ('mismatched_work_id', True, False, False, False, 'objmap', False, False): 'stage2',
    ('mismatched_work_id', True, False, False, False, 'objmap', False, True): 'stage1',
    ('mismatched_work_id', True, False, False, False, 'objmap', True, False): 'stage2',
    ('mismatched_work_id', True, False, False, False, 'objmap', True, True): 'stage1',
    ('mismatched_work_id', True, False, False, False, 'objmap_terminal', False, False): 'complete',
    ('mismatched_work_id', True, False, False, False, 'objmap_terminal', False, True): 'stage1',
    ('mismatched_work_id', True, False, False, False, 'objmap_terminal', True, False): 'complete',
    ('mismatched_work_id', True, False, False, False, 'objmap_terminal', True, True): 'stage1',
    ('mismatched_work_id', True, False, False, False, 'rgb', False, False): 'stage2',
    ('mismatched_work_id', True, False, False, False, 'rgb', False, True): 'stage1',
    ('mismatched_work_id', True, False, False, False, 'rgb', True, False): 'stage2',
    ('mismatched_work_id', True, False, False, False, 'rgb', True, True): 'stage1',
    ('mismatched_work_id', True, False, False, True, 'none', False, False): 'complete',
    ('mismatched_work_id', True, False, False, True, 'none', False, True): 'stage1',
    ('mismatched_work_id', True, False, False, True, 'none', True, False): 'complete',
    ('mismatched_work_id', True, False, False, True, 'none', True, True): 'stage1',
    ('mismatched_work_id', True, False, False, True, 'objmap', False, False): 'stage2',
    ('mismatched_work_id', True, False, False, True, 'objmap', False, True): 'stage1',
    ('mismatched_work_id', True, False, False, True, 'objmap', True, False): 'stage2',
    ('mismatched_work_id', True, False, False, True, 'objmap', True, True): 'stage1',
    ('mismatched_work_id', True, False, False, True, 'objmap_terminal', False, False): 'complete',
    ('mismatched_work_id', True, False, False, True, 'objmap_terminal', False, True): 'stage1',
    ('mismatched_work_id', True, False, False, True, 'objmap_terminal', True, False): 'complete',
    ('mismatched_work_id', True, False, False, True, 'objmap_terminal', True, True): 'stage1',
    ('mismatched_work_id', True, False, False, True, 'rgb', False, False): 'stage2',
    ('mismatched_work_id', True, False, False, True, 'rgb', False, True): 'stage1',
    ('mismatched_work_id', True, False, False, True, 'rgb', True, False): 'stage2',
    ('mismatched_work_id', True, False, False, True, 'rgb', True, True): 'stage1',
    ('mismatched_work_id', True, False, True, False, 'none', False, False): 'complete',
    ('mismatched_work_id', True, False, True, False, 'none', False, True): 'stage1',
    ('mismatched_work_id', True, False, True, False, 'none', True, False): 'stage2',
    ('mismatched_work_id', True, False, True, False, 'none', True, True): 'stage1',
    ('mismatched_work_id', True, False, True, False, 'objmap', False, False): 'stage2',
    ('mismatched_work_id', True, False, True, False, 'objmap', False, True): 'stage1',
    ('mismatched_work_id', True, False, True, False, 'objmap', True, False): 'stage2',
    ('mismatched_work_id', True, False, True, False, 'objmap', True, True): 'stage1',
    ('mismatched_work_id', True, False, True, False, 'objmap_terminal', False, False): 'complete',
    ('mismatched_work_id', True, False, True, False, 'objmap_terminal', False, True): 'stage1',
    ('mismatched_work_id', True, False, True, False, 'objmap_terminal', True, False): 'complete',
    ('mismatched_work_id', True, False, True, False, 'objmap_terminal', True, True): 'stage1',
    ('mismatched_work_id', True, False, True, False, 'rgb', False, False): 'stage2',
    ('mismatched_work_id', True, False, True, False, 'rgb', False, True): 'stage1',
    ('mismatched_work_id', True, False, True, False, 'rgb', True, False): 'stage2',
    ('mismatched_work_id', True, False, True, False, 'rgb', True, True): 'stage1',
    ('mismatched_work_id', True, False, True, True, 'none', False, False): 'complete',
    ('mismatched_work_id', True, False, True, True, 'none', False, True): 'stage1',
    ('mismatched_work_id', True, False, True, True, 'none', True, False): 'complete',
    ('mismatched_work_id', True, False, True, True, 'none', True, True): 'stage1',
    ('mismatched_work_id', True, False, True, True, 'objmap', False, False): 'stage2',
    ('mismatched_work_id', True, False, True, True, 'objmap', False, True): 'stage1',
    ('mismatched_work_id', True, False, True, True, 'objmap', True, False): 'stage2',
    ('mismatched_work_id', True, False, True, True, 'objmap', True, True): 'stage1',
    ('mismatched_work_id', True, False, True, True, 'objmap_terminal', False, False): 'complete',
    ('mismatched_work_id', True, False, True, True, 'objmap_terminal', False, True): 'stage1',
    ('mismatched_work_id', True, False, True, True, 'objmap_terminal', True, False): 'complete',
    ('mismatched_work_id', True, False, True, True, 'objmap_terminal', True, True): 'stage1',
    ('mismatched_work_id', True, False, True, True, 'rgb', False, False): 'stage2',
    ('mismatched_work_id', True, False, True, True, 'rgb', False, True): 'stage1',
    ('mismatched_work_id', True, False, True, True, 'rgb', True, False): 'stage2',
    ('mismatched_work_id', True, False, True, True, 'rgb', True, True): 'stage1',
    ('mismatched_work_id', True, True, False, False, 'none', False, False): 'stage2',
    ('mismatched_work_id', True, True, False, False, 'none', False, True): 'stage1',
    ('mismatched_work_id', True, True, False, False, 'none', True, False): 'stage2',
    ('mismatched_work_id', True, True, False, False, 'none', True, True): 'stage1',
    ('mismatched_work_id', True, True, False, False, 'objmap', False, False): 'stage2',
    ('mismatched_work_id', True, True, False, False, 'objmap', False, True): 'stage1',
    ('mismatched_work_id', True, True, False, False, 'objmap', True, False): 'stage2',
    ('mismatched_work_id', True, True, False, False, 'objmap', True, True): 'stage1',
    ('mismatched_work_id', True, True, False, False, 'objmap_terminal', False, False): 'complete',
    ('mismatched_work_id', True, True, False, False, 'objmap_terminal', False, True): 'stage1',
    ('mismatched_work_id', True, True, False, False, 'objmap_terminal', True, False): 'complete',
    ('mismatched_work_id', True, True, False, False, 'objmap_terminal', True, True): 'stage1',
    ('mismatched_work_id', True, True, False, False, 'rgb', False, False): 'stage2',
    ('mismatched_work_id', True, True, False, False, 'rgb', False, True): 'stage1',
    ('mismatched_work_id', True, True, False, False, 'rgb', True, False): 'stage2',
    ('mismatched_work_id', True, True, False, False, 'rgb', True, True): 'stage1',
    ('mismatched_work_id', True, True, False, True, 'none', False, False): 'complete',
    ('mismatched_work_id', True, True, False, True, 'none', False, True): 'stage1',
    ('mismatched_work_id', True, True, False, True, 'none', True, False): 'complete',
    ('mismatched_work_id', True, True, False, True, 'none', True, True): 'stage1',
    ('mismatched_work_id', True, True, False, True, 'objmap', False, False): 'stage2',
    ('mismatched_work_id', True, True, False, True, 'objmap', False, True): 'stage1',
    ('mismatched_work_id', True, True, False, True, 'objmap', True, False): 'stage2',
    ('mismatched_work_id', True, True, False, True, 'objmap', True, True): 'stage1',
    ('mismatched_work_id', True, True, False, True, 'objmap_terminal', False, False): 'complete',
    ('mismatched_work_id', True, True, False, True, 'objmap_terminal', False, True): 'stage1',
    ('mismatched_work_id', True, True, False, True, 'objmap_terminal', True, False): 'complete',
    ('mismatched_work_id', True, True, False, True, 'objmap_terminal', True, True): 'stage1',
    ('mismatched_work_id', True, True, False, True, 'rgb', False, False): 'stage2',
    ('mismatched_work_id', True, True, False, True, 'rgb', False, True): 'stage1',
    ('mismatched_work_id', True, True, False, True, 'rgb', True, False): 'stage2',
    ('mismatched_work_id', True, True, False, True, 'rgb', True, True): 'stage1',
    ('mismatched_work_id', True, True, True, False, 'none', False, False): 'stage3',
    ('mismatched_work_id', True, True, True, False, 'none', False, True): 'stage1',
    ('mismatched_work_id', True, True, True, False, 'none', True, False): 'stage3',
    ('mismatched_work_id', True, True, True, False, 'none', True, True): 'stage1',
    ('mismatched_work_id', True, True, True, False, 'objmap', False, False): 'stage3',
    ('mismatched_work_id', True, True, True, False, 'objmap', False, True): 'stage1',
    ('mismatched_work_id', True, True, True, False, 'objmap', True, False): 'stage3',
    ('mismatched_work_id', True, True, True, False, 'objmap', True, True): 'stage1',
    ('mismatched_work_id', True, True, True, False, 'objmap_terminal', False, False): 'complete',
    ('mismatched_work_id', True, True, True, False, 'objmap_terminal', False, True): 'stage1',
    ('mismatched_work_id', True, True, True, False, 'objmap_terminal', True, False): 'complete',
    ('mismatched_work_id', True, True, True, False, 'objmap_terminal', True, True): 'stage1',
    ('mismatched_work_id', True, True, True, False, 'rgb', False, False): 'stage3',
    ('mismatched_work_id', True, True, True, False, 'rgb', False, True): 'stage1',
    ('mismatched_work_id', True, True, True, False, 'rgb', True, False): 'stage3',
    ('mismatched_work_id', True, True, True, False, 'rgb', True, True): 'stage1',
    ('mismatched_work_id', True, True, True, True, 'none', False, False): 'complete',
    ('mismatched_work_id', True, True, True, True, 'none', False, True): 'stage1',
    ('mismatched_work_id', True, True, True, True, 'none', True, False): 'complete',
    ('mismatched_work_id', True, True, True, True, 'none', True, True): 'stage1',
    ('mismatched_work_id', True, True, True, True, 'objmap', False, False): 'stage3',
    ('mismatched_work_id', True, True, True, True, 'objmap', False, True): 'stage1',
    ('mismatched_work_id', True, True, True, True, 'objmap', True, False): 'stage3',
    ('mismatched_work_id', True, True, True, True, 'objmap', True, True): 'stage1',
    ('mismatched_work_id', True, True, True, True, 'objmap_terminal', False, False): 'complete',
    ('mismatched_work_id', True, True, True, True, 'objmap_terminal', False, True): 'stage1',
    ('mismatched_work_id', True, True, True, True, 'objmap_terminal', True, False): 'complete',
    ('mismatched_work_id', True, True, True, True, 'objmap_terminal', True, True): 'stage1',
    ('mismatched_work_id', True, True, True, True, 'rgb', False, False): 'stage3',
    ('mismatched_work_id', True, True, True, True, 'rgb', False, True): 'stage1',
    ('mismatched_work_id', True, True, True, True, 'rgb', True, False): 'stage3',
    ('mismatched_work_id', True, True, True, True, 'rgb', True, True): 'stage1',
    ('matching_work_id', False, False, False, False, 'none', False, False): 'stage2',
    ('matching_work_id', False, False, False, False, 'none', False, True): 'stage2',
    ('matching_work_id', False, False, False, False, 'none', True, False): 'stage2',
    ('matching_work_id', False, False, False, False, 'none', True, True): 'stage2',
    ('matching_work_id', False, False, False, False, 'objmap', False, False): 'stage2',
    ('matching_work_id', False, False, False, False, 'objmap', False, True): 'stage2',
    ('matching_work_id', False, False, False, False, 'objmap', True, False): 'stage2',
    ('matching_work_id', False, False, False, False, 'objmap', True, True): 'stage2',
    ('matching_work_id', False, False, False, False, 'objmap_terminal', False, False): 'complete',
    ('matching_work_id', False, False, False, False, 'objmap_terminal', False, True): 'stage2',
    ('matching_work_id', False, False, False, False, 'objmap_terminal', True, False): 'complete',
    ('matching_work_id', False, False, False, False, 'objmap_terminal', True, True): 'stage2',
    ('matching_work_id', False, False, False, False, 'rgb', False, False): 'stage2',
    ('matching_work_id', False, False, False, False, 'rgb', False, True): 'stage2',
    ('matching_work_id', False, False, False, False, 'rgb', True, False): 'stage2',
    ('matching_work_id', False, False, False, False, 'rgb', True, True): 'stage2',
    ('matching_work_id', False, False, False, True, 'none', False, False): 'complete',
    ('matching_work_id', False, False, False, True, 'none', False, True): 'stage2',
    ('matching_work_id', False, False, False, True, 'none', True, False): 'complete',
    ('matching_work_id', False, False, False, True, 'none', True, True): 'stage2',
    ('matching_work_id', False, False, False, True, 'objmap', False, False): 'stage2',
    ('matching_work_id', False, False, False, True, 'objmap', False, True): 'stage2',
    ('matching_work_id', False, False, False, True, 'objmap', True, False): 'stage2',
    ('matching_work_id', False, False, False, True, 'objmap', True, True): 'stage2',
    ('matching_work_id', False, False, False, True, 'objmap_terminal', False, False): 'complete',
    ('matching_work_id', False, False, False, True, 'objmap_terminal', False, True): 'stage2',
    ('matching_work_id', False, False, False, True, 'objmap_terminal', True, False): 'complete',
    ('matching_work_id', False, False, False, True, 'objmap_terminal', True, True): 'stage2',
    ('matching_work_id', False, False, False, True, 'rgb', False, False): 'stage2',
    ('matching_work_id', False, False, False, True, 'rgb', False, True): 'stage2',
    ('matching_work_id', False, False, False, True, 'rgb', True, False): 'stage2',
    ('matching_work_id', False, False, False, True, 'rgb', True, True): 'stage2',
    ('matching_work_id', False, False, True, False, 'none', False, False): 'stage2',
    ('matching_work_id', False, False, True, False, 'none', False, True): 'stage2',
    ('matching_work_id', False, False, True, False, 'none', True, False): 'stage2',
    ('matching_work_id', False, False, True, False, 'none', True, True): 'stage2',
    ('matching_work_id', False, False, True, False, 'objmap', False, False): 'stage2',
    ('matching_work_id', False, False, True, False, 'objmap', False, True): 'stage2',
    ('matching_work_id', False, False, True, False, 'objmap', True, False): 'stage2',
    ('matching_work_id', False, False, True, False, 'objmap', True, True): 'stage2',
    ('matching_work_id', False, False, True, False, 'objmap_terminal', False, False): 'complete',
    ('matching_work_id', False, False, True, False, 'objmap_terminal', False, True): 'stage2',
    ('matching_work_id', False, False, True, False, 'objmap_terminal', True, False): 'complete',
    ('matching_work_id', False, False, True, False, 'objmap_terminal', True, True): 'stage2',
    ('matching_work_id', False, False, True, False, 'rgb', False, False): 'stage2',
    ('matching_work_id', False, False, True, False, 'rgb', False, True): 'stage2',
    ('matching_work_id', False, False, True, False, 'rgb', True, False): 'stage2',
    ('matching_work_id', False, False, True, False, 'rgb', True, True): 'stage2',
    ('matching_work_id', False, False, True, True, 'none', False, False): 'complete',
    ('matching_work_id', False, False, True, True, 'none', False, True): 'stage2',
    ('matching_work_id', False, False, True, True, 'none', True, False): 'complete',
    ('matching_work_id', False, False, True, True, 'none', True, True): 'stage2',
    ('matching_work_id', False, False, True, True, 'objmap', False, False): 'stage2',
    ('matching_work_id', False, False, True, True, 'objmap', False, True): 'stage2',
    ('matching_work_id', False, False, True, True, 'objmap', True, False): 'stage2',
    ('matching_work_id', False, False, True, True, 'objmap', True, True): 'stage2',
    ('matching_work_id', False, False, True, True, 'objmap_terminal', False, False): 'complete',
    ('matching_work_id', False, False, True, True, 'objmap_terminal', False, True): 'stage2',
    ('matching_work_id', False, False, True, True, 'objmap_terminal', True, False): 'complete',
    ('matching_work_id', False, False, True, True, 'objmap_terminal', True, True): 'stage2',
    ('matching_work_id', False, False, True, True, 'rgb', False, False): 'stage2',
    ('matching_work_id', False, False, True, True, 'rgb', False, True): 'stage2',
    ('matching_work_id', False, False, True, True, 'rgb', True, False): 'stage2',
    ('matching_work_id', False, False, True, True, 'rgb', True, True): 'stage2',
    ('matching_work_id', False, True, False, False, 'none', False, False): 'stage2',
    ('matching_work_id', False, True, False, False, 'none', False, True): 'stage2',
    ('matching_work_id', False, True, False, False, 'none', True, False): 'stage2',
    ('matching_work_id', False, True, False, False, 'none', True, True): 'stage2',
    ('matching_work_id', False, True, False, False, 'objmap', False, False): 'stage2',
    ('matching_work_id', False, True, False, False, 'objmap', False, True): 'stage2',
    ('matching_work_id', False, True, False, False, 'objmap', True, False): 'stage2',
    ('matching_work_id', False, True, False, False, 'objmap', True, True): 'stage2',
    ('matching_work_id', False, True, False, False, 'objmap_terminal', False, False): 'complete',
    ('matching_work_id', False, True, False, False, 'objmap_terminal', False, True): 'stage2',
    ('matching_work_id', False, True, False, False, 'objmap_terminal', True, False): 'complete',
    ('matching_work_id', False, True, False, False, 'objmap_terminal', True, True): 'stage2',
    ('matching_work_id', False, True, False, False, 'rgb', False, False): 'stage2',
    ('matching_work_id', False, True, False, False, 'rgb', False, True): 'stage2',
    ('matching_work_id', False, True, False, False, 'rgb', True, False): 'stage2',
    ('matching_work_id', False, True, False, False, 'rgb', True, True): 'stage2',
    ('matching_work_id', False, True, False, True, 'none', False, False): 'complete',
    ('matching_work_id', False, True, False, True, 'none', False, True): 'stage2',
    ('matching_work_id', False, True, False, True, 'none', True, False): 'complete',
    ('matching_work_id', False, True, False, True, 'none', True, True): 'stage2',
    ('matching_work_id', False, True, False, True, 'objmap', False, False): 'stage2',
    ('matching_work_id', False, True, False, True, 'objmap', False, True): 'stage2',
    ('matching_work_id', False, True, False, True, 'objmap', True, False): 'stage2',
    ('matching_work_id', False, True, False, True, 'objmap', True, True): 'stage2',
    ('matching_work_id', False, True, False, True, 'objmap_terminal', False, False): 'complete',
    ('matching_work_id', False, True, False, True, 'objmap_terminal', False, True): 'stage2',
    ('matching_work_id', False, True, False, True, 'objmap_terminal', True, False): 'complete',
    ('matching_work_id', False, True, False, True, 'objmap_terminal', True, True): 'stage2',
    ('matching_work_id', False, True, False, True, 'rgb', False, False): 'stage2',
    ('matching_work_id', False, True, False, True, 'rgb', False, True): 'stage2',
    ('matching_work_id', False, True, False, True, 'rgb', True, False): 'stage2',
    ('matching_work_id', False, True, False, True, 'rgb', True, True): 'stage2',
    ('matching_work_id', False, True, True, False, 'none', False, False): 'stage3',
    ('matching_work_id', False, True, True, False, 'none', False, True): 'stage3',
    ('matching_work_id', False, True, True, False, 'none', True, False): 'stage3',
    ('matching_work_id', False, True, True, False, 'none', True, True): 'stage3',
    ('matching_work_id', False, True, True, False, 'objmap', False, False): 'stage3',
    ('matching_work_id', False, True, True, False, 'objmap', False, True): 'stage3',
    ('matching_work_id', False, True, True, False, 'objmap', True, False): 'stage3',
    ('matching_work_id', False, True, True, False, 'objmap', True, True): 'stage3',
    ('matching_work_id', False, True, True, False, 'objmap_terminal', False, False): 'complete',
    ('matching_work_id', False, True, True, False, 'objmap_terminal', False, True): 'stage3',
    ('matching_work_id', False, True, True, False, 'objmap_terminal', True, False): 'complete',
    ('matching_work_id', False, True, True, False, 'objmap_terminal', True, True): 'stage3',
    ('matching_work_id', False, True, True, False, 'rgb', False, False): 'stage3',
    ('matching_work_id', False, True, True, False, 'rgb', False, True): 'stage3',
    ('matching_work_id', False, True, True, False, 'rgb', True, False): 'stage3',
    ('matching_work_id', False, True, True, False, 'rgb', True, True): 'stage3',
    ('matching_work_id', False, True, True, True, 'none', False, False): 'complete',
    ('matching_work_id', False, True, True, True, 'none', False, True): 'stage3',
    ('matching_work_id', False, True, True, True, 'none', True, False): 'complete',
    ('matching_work_id', False, True, True, True, 'none', True, True): 'stage3',
    ('matching_work_id', False, True, True, True, 'objmap', False, False): 'stage3',
    ('matching_work_id', False, True, True, True, 'objmap', False, True): 'stage3',
    ('matching_work_id', False, True, True, True, 'objmap', True, False): 'stage3',
    ('matching_work_id', False, True, True, True, 'objmap', True, True): 'stage3',
    ('matching_work_id', False, True, True, True, 'objmap_terminal', False, False): 'complete',
    ('matching_work_id', False, True, True, True, 'objmap_terminal', False, True): 'stage3',
    ('matching_work_id', False, True, True, True, 'objmap_terminal', True, False): 'complete',
    ('matching_work_id', False, True, True, True, 'objmap_terminal', True, True): 'stage3',
    ('matching_work_id', False, True, True, True, 'rgb', False, False): 'stage3',
    ('matching_work_id', False, True, True, True, 'rgb', False, True): 'stage3',
    ('matching_work_id', False, True, True, True, 'rgb', True, False): 'stage3',
    ('matching_work_id', False, True, True, True, 'rgb', True, True): 'stage3',
    ('matching_work_id', True, False, False, False, 'none', False, False): 'complete',
    ('matching_work_id', True, False, False, False, 'none', False, True): 'complete',
    ('matching_work_id', True, False, False, False, 'none', True, False): 'stage2',
    ('matching_work_id', True, False, False, False, 'none', True, True): 'stage2',
    ('matching_work_id', True, False, False, False, 'objmap', False, False): 'stage2',
    ('matching_work_id', True, False, False, False, 'objmap', False, True): 'stage2',
    ('matching_work_id', True, False, False, False, 'objmap', True, False): 'stage2',
    ('matching_work_id', True, False, False, False, 'objmap', True, True): 'stage2',
    ('matching_work_id', True, False, False, False, 'objmap_terminal', False, False): 'complete',
    ('matching_work_id', True, False, False, False, 'objmap_terminal', False, True): 'stage2',
    ('matching_work_id', True, False, False, False, 'objmap_terminal', True, False): 'complete',
    ('matching_work_id', True, False, False, False, 'objmap_terminal', True, True): 'stage2',
    ('matching_work_id', True, False, False, False, 'rgb', False, False): 'stage2',
    ('matching_work_id', True, False, False, False, 'rgb', False, True): 'stage2',
    ('matching_work_id', True, False, False, False, 'rgb', True, False): 'stage2',
    ('matching_work_id', True, False, False, False, 'rgb', True, True): 'stage2',
    ('matching_work_id', True, False, False, True, 'none', False, False): 'complete',
    ('matching_work_id', True, False, False, True, 'none', False, True): 'stage3',
    ('matching_work_id', True, False, False, True, 'none', True, False): 'complete',
    ('matching_work_id', True, False, False, True, 'none', True, True): 'stage3',
    ('matching_work_id', True, False, False, True, 'objmap', False, False): 'stage2',
    ('matching_work_id', True, False, False, True, 'objmap', False, True): 'stage2',
    ('matching_work_id', True, False, False, True, 'objmap', True, False): 'stage2',
    ('matching_work_id', True, False, False, True, 'objmap', True, True): 'stage2',
    ('matching_work_id', True, False, False, True, 'objmap_terminal', False, False): 'complete',
    ('matching_work_id', True, False, False, True, 'objmap_terminal', False, True): 'stage2',
    ('matching_work_id', True, False, False, True, 'objmap_terminal', True, False): 'complete',
    ('matching_work_id', True, False, False, True, 'objmap_terminal', True, True): 'stage2',
    ('matching_work_id', True, False, False, True, 'rgb', False, False): 'stage2',
    ('matching_work_id', True, False, False, True, 'rgb', False, True): 'stage2',
    ('matching_work_id', True, False, False, True, 'rgb', True, False): 'stage2',
    ('matching_work_id', True, False, False, True, 'rgb', True, True): 'stage2',
    ('matching_work_id', True, False, True, False, 'none', False, False): 'complete',
    ('matching_work_id', True, False, True, False, 'none', False, True): 'complete',
    ('matching_work_id', True, False, True, False, 'none', True, False): 'stage2',
    ('matching_work_id', True, False, True, False, 'none', True, True): 'stage2',
    ('matching_work_id', True, False, True, False, 'objmap', False, False): 'stage2',
    ('matching_work_id', True, False, True, False, 'objmap', False, True): 'stage2',
    ('matching_work_id', True, False, True, False, 'objmap', True, False): 'stage2',
    ('matching_work_id', True, False, True, False, 'objmap', True, True): 'stage2',
    ('matching_work_id', True, False, True, False, 'objmap_terminal', False, False): 'complete',
    ('matching_work_id', True, False, True, False, 'objmap_terminal', False, True): 'stage2',
    ('matching_work_id', True, False, True, False, 'objmap_terminal', True, False): 'complete',
    ('matching_work_id', True, False, True, False, 'objmap_terminal', True, True): 'stage2',
    ('matching_work_id', True, False, True, False, 'rgb', False, False): 'stage2',
    ('matching_work_id', True, False, True, False, 'rgb', False, True): 'stage2',
    ('matching_work_id', True, False, True, False, 'rgb', True, False): 'stage2',
    ('matching_work_id', True, False, True, False, 'rgb', True, True): 'stage2',
    ('matching_work_id', True, False, True, True, 'none', False, False): 'complete',
    ('matching_work_id', True, False, True, True, 'none', False, True): 'stage3',
    ('matching_work_id', True, False, True, True, 'none', True, False): 'complete',
    ('matching_work_id', True, False, True, True, 'none', True, True): 'stage3',
    ('matching_work_id', True, False, True, True, 'objmap', False, False): 'stage2',
    ('matching_work_id', True, False, True, True, 'objmap', False, True): 'stage2',
    ('matching_work_id', True, False, True, True, 'objmap', True, False): 'stage2',
    ('matching_work_id', True, False, True, True, 'objmap', True, True): 'stage2',
    ('matching_work_id', True, False, True, True, 'objmap_terminal', False, False): 'complete',
    ('matching_work_id', True, False, True, True, 'objmap_terminal', False, True): 'stage2',
    ('matching_work_id', True, False, True, True, 'objmap_terminal', True, False): 'complete',
    ('matching_work_id', True, False, True, True, 'objmap_terminal', True, True): 'stage2',
    ('matching_work_id', True, False, True, True, 'rgb', False, False): 'stage2',
    ('matching_work_id', True, False, True, True, 'rgb', False, True): 'stage2',
    ('matching_work_id', True, False, True, True, 'rgb', True, False): 'stage2',
    ('matching_work_id', True, False, True, True, 'rgb', True, True): 'stage2',
    ('matching_work_id', True, True, False, False, 'none', False, False): 'stage2',
    ('matching_work_id', True, True, False, False, 'none', False, True): 'stage2',
    ('matching_work_id', True, True, False, False, 'none', True, False): 'stage2',
    ('matching_work_id', True, True, False, False, 'none', True, True): 'stage2',
    ('matching_work_id', True, True, False, False, 'objmap', False, False): 'stage2',
    ('matching_work_id', True, True, False, False, 'objmap', False, True): 'stage2',
    ('matching_work_id', True, True, False, False, 'objmap', True, False): 'stage2',
    ('matching_work_id', True, True, False, False, 'objmap', True, True): 'stage2',
    ('matching_work_id', True, True, False, False, 'objmap_terminal', False, False): 'complete',
    ('matching_work_id', True, True, False, False, 'objmap_terminal', False, True): 'stage2',
    ('matching_work_id', True, True, False, False, 'objmap_terminal', True, False): 'complete',
    ('matching_work_id', True, True, False, False, 'objmap_terminal', True, True): 'stage2',
    ('matching_work_id', True, True, False, False, 'rgb', False, False): 'stage2',
    ('matching_work_id', True, True, False, False, 'rgb', False, True): 'stage2',
    ('matching_work_id', True, True, False, False, 'rgb', True, False): 'stage2',
    ('matching_work_id', True, True, False, False, 'rgb', True, True): 'stage2',
    ('matching_work_id', True, True, False, True, 'none', False, False): 'complete',
    ('matching_work_id', True, True, False, True, 'none', False, True): 'stage3',
    ('matching_work_id', True, True, False, True, 'none', True, False): 'complete',
    ('matching_work_id', True, True, False, True, 'none', True, True): 'stage3',
    ('matching_work_id', True, True, False, True, 'objmap', False, False): 'stage2',
    ('matching_work_id', True, True, False, True, 'objmap', False, True): 'stage2',
    ('matching_work_id', True, True, False, True, 'objmap', True, False): 'stage2',
    ('matching_work_id', True, True, False, True, 'objmap', True, True): 'stage2',
    ('matching_work_id', True, True, False, True, 'objmap_terminal', False, False): 'complete',
    ('matching_work_id', True, True, False, True, 'objmap_terminal', False, True): 'stage2',
    ('matching_work_id', True, True, False, True, 'objmap_terminal', True, False): 'complete',
    ('matching_work_id', True, True, False, True, 'objmap_terminal', True, True): 'stage2',
    ('matching_work_id', True, True, False, True, 'rgb', False, False): 'stage2',
    ('matching_work_id', True, True, False, True, 'rgb', False, True): 'stage2',
    ('matching_work_id', True, True, False, True, 'rgb', True, False): 'stage2',
    ('matching_work_id', True, True, False, True, 'rgb', True, True): 'stage2',
    ('matching_work_id', True, True, True, False, 'none', False, False): 'stage3',
    ('matching_work_id', True, True, True, False, 'none', False, True): 'stage3',
    ('matching_work_id', True, True, True, False, 'none', True, False): 'stage3',
    ('matching_work_id', True, True, True, False, 'none', True, True): 'stage3',
    ('matching_work_id', True, True, True, False, 'objmap', False, False): 'stage3',
    ('matching_work_id', True, True, True, False, 'objmap', False, True): 'stage3',
    ('matching_work_id', True, True, True, False, 'objmap', True, False): 'stage3',
    ('matching_work_id', True, True, True, False, 'objmap', True, True): 'stage3',
    ('matching_work_id', True, True, True, False, 'objmap_terminal', False, False): 'complete',
    ('matching_work_id', True, True, True, False, 'objmap_terminal', False, True): 'stage3',
    ('matching_work_id', True, True, True, False, 'objmap_terminal', True, False): 'complete',
    ('matching_work_id', True, True, True, False, 'objmap_terminal', True, True): 'stage3',
    ('matching_work_id', True, True, True, False, 'rgb', False, False): 'stage3',
    ('matching_work_id', True, True, True, False, 'rgb', False, True): 'stage3',
    ('matching_work_id', True, True, True, False, 'rgb', True, False): 'stage3',
    ('matching_work_id', True, True, True, False, 'rgb', True, True): 'stage3',
    ('matching_work_id', True, True, True, True, 'none', False, False): 'complete',
    ('matching_work_id', True, True, True, True, 'none', False, True): 'stage3',
    ('matching_work_id', True, True, True, True, 'none', True, False): 'complete',
    ('matching_work_id', True, True, True, True, 'none', True, True): 'stage3',
    ('matching_work_id', True, True, True, True, 'objmap', False, False): 'stage3',
    ('matching_work_id', True, True, True, True, 'objmap', False, True): 'stage3',
    ('matching_work_id', True, True, True, True, 'objmap', True, False): 'stage3',
    ('matching_work_id', True, True, True, True, 'objmap', True, True): 'stage3',
    ('matching_work_id', True, True, True, True, 'objmap_terminal', False, False): 'complete',
    ('matching_work_id', True, True, True, True, 'objmap_terminal', False, True): 'stage3',
    ('matching_work_id', True, True, True, True, 'objmap_terminal', True, False): 'complete',
    ('matching_work_id', True, True, True, True, 'objmap_terminal', True, True): 'stage3',
    ('matching_work_id', True, True, True, True, 'rgb', False, False): 'stage3',
    ('matching_work_id', True, True, True, True, 'rgb', False, True): 'stage3',
    ('matching_work_id', True, True, True, True, 'rgb', True, False): 'stage3',
    ('matching_work_id', True, True, True, True, 'rgb', True, True): 'stage3',
}
# --- END CAPTURED TABLE ---


# ---------------------------------------------------------------------------
# Planting one cell's artifacts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Case:
    """The arguments one planted cell needs for its classifier call."""

    output_dir: Path
    dataset: str
    image: Path
    input_root: Path
    work_id: str


def build_store_templates(base: Path) -> dict[str, Path]:
    """Build each store state once, for the cells to copy.

    A store per cell is the dominant cost of this enumeration, and every cell
    sharing a store state wants a byte-identical one, so they are built here and
    ``copytree``d rather than rebuilt.

    A module-level function rather than only a fixture body, so the Step-2
    capture that produced :data:`_EXPECTED` planted its cells through *this*
    code and not through a second copy of it. A capture harness that builds its
    own worlds is freezing its own behaviour, not the classifier's.
    """
    templates: dict[str, Path] = {}
    for state in _STORE_STATES:
        if state == "absent":
            continue
        path = base / f"{state}.ome.zarr"
        work_id = {
            "in_progress": None,
            "in_progress_matching_work_id": _WORK_ID,
            "mismatched_work_id": _OTHER_WORK_ID,
            "matching_work_id": _WORK_ID,
        }[state]
        Image(np.zeros((4, 4, 3), dtype=np.uint8)).save2zarr(
            path, work_id=work_id
        )
        if state.startswith("in_progress"):
            _mark_journal_in_progress(path)
        templates[state] = path
    return templates


@pytest.fixture(scope="module")
def store_templates(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The store templates, built once per module."""
    return build_store_templates(tmp_path_factory.mktemp("store_templates"))


def _mark_journal_in_progress(store: Path) -> None:
    """Give a store a provenance journal ``valid_stage1_store`` rejects.

    ``save2zarr`` writes no journal, and an absent journal is *accepted* by
    ``valid_stage1_store`` through its compatibility branch
    (``_cli_staged_resume.py:99-100``). The only way to reach the
    ``valid_staged_store``-but-not-``valid_stage1_store`` states is therefore to
    write one whose ``status`` is outside ``{"staged", "complete"}``.
    """
    root = store / "zarr.json"
    payload = json.loads(root.read_text(encoding="utf-8"))
    block = payload.setdefault("attributes", {}).setdefault("phenotypic", {})
    block["provenance"] = {"status": "in_progress", "applications": []}
    root.write_text(json.dumps(payload), encoding="utf-8")


def _plant(
    root: Path,
    templates: dict[str, Path],
    *,
    store: str,
    table: bool,
    s2_token: bool,
    s2_raw: bool,
    s3_done: bool,
    layer: str,
) -> _Case:
    """Create exactly the durable artifacts one cell names, and nothing else."""
    output_dir = root / "out"
    input_root = root / "in"
    input_root.mkdir(parents=True, exist_ok=True)
    image = input_root / _IMAGE_NAME
    image.write_bytes(b"image")

    store_path = zarr_store_path(output_dir, _DATASET, _STEM)
    if store != "absent":
        store_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(templates[store], store_path)
        if table:
            embedded = store_path / MEASUREMENT_TABLE_RELATIVE_PATH
            embedded.parent.mkdir(parents=True, exist_ok=True)
            embedded.write_bytes(b"measurements")

    if s2_raw:
        write_stage2_raw(
            output_dir, _DATASET, _STEM, np.zeros((4, 4), dtype=np.uint16)
        )
    if s2_token:
        write_stage2_token(output_dir, _DATASET, _STEM, objmap_shape=(4, 4))
    if s3_done:
        # Through the WRITER, not a hand-joined path: see the module docstring.
        write_stage3_completion_marker(
            output_dir, _DATASET, _IMAGE_NAME, _STEM
        )
    if layer == "objmap_terminal":
        terminal = process_only_output_path(
            output_dir, image, input_root, "objmap", fmt="tiff"
        )
        terminal.parent.mkdir(parents=True, exist_ok=True)
        terminal.write_bytes(b"objmap")

    return _Case(
        output_dir=output_dir,
        dataset=_DATASET,
        image=image,
        input_root=input_root,
        work_id=_WORK_ID,
    )


def _classify(case: _Case, *, layer: str, markers: bool, expect: bool) -> str:
    return classify_staged_image(
        output_dir=case.output_dir,
        dataset=case.dataset,
        image=case.image,
        input_root=case.input_root,
        process_only_layer=_LAYER_VALUES[layer],
        markers_required=markers,
        expected_work_id=case.work_id if expect else None,
    )


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "store,table,s2_token,s2_raw,s3_done,layer,markers,expect_work_id",
    _COMBOS,
)
def test_classification_is_unchanged_by_the_collapse(
    tmp_path: Path,
    store_templates: dict[str, Path],
    store: str,
    table: bool,
    s2_token: bool,
    s2_raw: bool,
    s3_done: bool,
    layer: str,
    markers: bool,
    expect_work_id: bool,
) -> None:
    case = _plant(
        tmp_path,
        store_templates,
        store=store,
        table=table,
        s2_token=s2_token,
        s2_raw=s2_raw,
        s3_done=s3_done,
        layer=layer,
    )
    actual = _classify(
        case, layer=layer, markers=markers, expect=expect_work_id
    )
    key = (
        store,
        table,
        s2_token,
        s2_raw,
        s3_done,
        layer,
        markers,
        expect_work_id,
    )
    assert actual == _EXPECTED[key], (
        f"the collapse changed a resume decision for {key}: "
        f"{_EXPECTED[key]!r} before, {actual!r} now"
    )


def test_the_expected_table_covers_exactly_the_enumerated_combinations() -> None:
    """A silently truncated capture must not pass by not being asked.

    ``_EXPECTED`` is a literal table typed out from a capture run. Missing keys
    would surface only as ``KeyError`` in whichever cells happened to run, and
    stale keys would survive an axis change with nothing to notice them.
    """
    assert set(_EXPECTED) == set(_COMBOS)


#: Axis names, positionally matching a ``_COMBOS`` tuple.
_AXIS_NAMES = (
    "store",
    "table",
    "s2_token",
    "s2_raw",
    "s3_done",
    "layer",
    "markers_required",
    "expect_work_id",
)


def _axis_is_live(position: int) -> bool:
    """Return whether varying one axis alone ever changes the verdict.

    Groups the table by every axis *except* ``position``; the axis is live if
    any group holds more than one distinct outcome.
    """
    groups: dict[tuple, set[str]] = {}
    for key, verdict in _EXPECTED.items():
        rest = key[:position] + key[position + 1 :]
        groups.setdefault(rest, set()).add(verdict)
    return any(len(outcomes) > 1 for outcomes in groups.values())


def test_every_axis_changes_at_least_one_outcome() -> None:
    """The guard against a harness that agrees with itself.

    **This is the failure the two-capture protocol cannot see.** The table is
    captured before the change and re-checked after, using the *same*
    ``_plant``. So an axis ``_plant`` fails to actually plant -- a typo in a
    path, a directory created where a file belongs -- produces the identical
    wrong table both times, the comparison reports "identical", and the gate
    passes having tested nothing on that axis. Agreement between two runs of a
    broken harness is not evidence.

    Proving each axis moves at least one verdict is a check that *can* fail,
    and it fails loudly with the axis named. Every one of the eight is
    load-bearing by inspection of `_cli_staged_resume.py`: `store` at
    `:239-240`, `table` at `:250-256` and `:263`, `s2_token` at `:285`,
    `s2_raw` at the FLOW-40 branch `:279-283`, `s3_done` at `:242-247`,
    `layer` at `:220` and `:243`, `markers_required` at `:262`,
    `expect_work_id` at `:229-234`. An inert axis is therefore a harness bug
    or a real finding, and both are worth stopping for.
    """
    inert = [
        name
        for position, name in enumerate(_AXIS_NAMES)
        if not _axis_is_live(position)
    ]
    assert not inert, (
        f"these axes never change a verdict: {inert}. Either _plant is not "
        "creating what they name, or the branch they were chosen for is "
        "unreachable -- do not populate the table until this is resolved"
    )


def test_the_store_templates_have_the_predicates_they_claim(
    tmp_path: Path, store_templates: dict[str, Path]
) -> None:
    """Execute the ``(V, S, W)`` table that ``_STORE_STATES`` only comments.

    The five store states exist to span the three store predicates, and two of
    them are reachable *only* through :func:`_mark_journal_in_progress`. If
    that patch does not actually take -- a renamed attribute block, a status
    value that turns out to be accepted -- both ``in_progress`` states
    silently collapse onto their valid counterparts, a third of the store
    coverage evaporates, and nothing else in this file notices: the store axis
    stays "live" on the strength of ``absent`` alone.

    So the comment table is asserted rather than trusted.
    """
    from phenotypic._cli._cli_staged_resume import (
        _staged_store_has_work_id,
        valid_stage1_store,
        valid_staged_store,
    )

    expected = {
        "absent": (False, False, False),
        "in_progress": (True, False, False),
        "in_progress_matching_work_id": (True, False, True),
        "mismatched_work_id": (True, True, False),
        "matching_work_id": (True, True, True),
    }
    assert set(expected) == set(_STORE_STATES)

    for state, triple in expected.items():
        path = tmp_path / f"{state}.ome.zarr"
        if state != "absent":
            shutil.copytree(store_templates[state], path)
        actual = (
            valid_staged_store(path),
            valid_stage1_store(path),
            _staged_store_has_work_id(path, _WORK_ID),
        )
        assert actual == triple, (
            f"store template {state!r} is "
            f"(valid_staged_store, valid_stage1_store, has_work_id)={actual}, "
            f"not the {triple} it is enumerated for"
        )


def test_the_axes_reach_every_outcome() -> None:
    """A gate that only ever freezes ``"stage1"`` freezes nothing.

    Pins that the enumeration exercises all four return values, so a future
    narrowing of the axes cannot quietly hollow the table out.
    """
    assert set(_EXPECTED.values()) == {
        "stage1",
        "stage2",
        "stage3",
        "complete",
    }


def test_the_stage_names_come_from_one_shared_constant() -> None:
    """CAN-27, the half that could only land once these modules imported them.

    Written in Task 1 and deferred to here by
    ``test_the_stage_names_have_exactly_one_home``'s docstring, because
    ``_cli_stage2_token`` and ``_cli_staged_resume`` do not import the stage
    names until this task collapses the stage-3 tree into the record.

    **``is``, not ``==``.** A shared-*object* check is the entire content of
    CAN-27: ``==`` passes for two modules that happen to spell ``"stage2"``
    identically, which is precisely the state one shared constant exists to
    make unrepresentable.
    """
    from phenotypic._cli import _cli_stage2_token, _cli_staged_resume
    from phenotypic.sdk_._image_record import STAGE_STAGE2, STAGE_STAGE3

    assert _cli_stage2_token.STAGE_STAGE2 is STAGE_STAGE2
    assert _cli_staged_resume.STAGE_STAGE3 is STAGE_STAGE3


def test_the_image_success_early_return_still_short_circuits(
    tmp_path: Path, store_templates: dict[str, Path], monkeypatch
) -> None:
    """``valid_image_success`` returning True must still win outright.

    Kept out of the parametrized axes on purpose. The call at
    ``_cli_staged_resume.py:209-218`` is an unconditional early return above
    every line this task edits, and P3 Task 2 is rewriting the function itself
    onto the record -- so freezing its behaviour in the table would freeze
    another cluster's in-flight work. Patching the name instead pins the
    control flow that matters here and stays independent of whatever marker or
    record shape that task lands.

    The import at ``:210`` is function-local, so the patch is read at call time.
    """
    from phenotypic._cli import _cli_completion

    case = _plant(
        tmp_path,
        store_templates,
        store="absent",
        table=False,
        s2_token=False,
        s2_raw=False,
        s3_done=False,
        layer="none",
    )
    assert _classify(case, layer="none", markers=True, expect=True) == "stage1"

    monkeypatch.setattr(
        _cli_completion, "valid_image_success", lambda *a, **k: True
    )

    assert _classify(case, layer="none", markers=True, expect=True) == (
        "complete"
    )
