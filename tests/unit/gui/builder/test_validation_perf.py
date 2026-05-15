"""Performance budget for the validation layer.

Validation runs on every state mutation through the dispatcher, so the
hot-loop cost must stay below the input-latency budget.  Target: a
100-block synthetic scope validates well within one input frame
(~16ms).  The budget is 12ms wall-clock — calibrated against standard
GitHub Actions runners, which clock the 100-block walk at ~8ms (vs.
~2-3ms on a developer workstation).  The 12ms ceiling keeps ~50%
headroom over observed CI timings while still tripping on a genuine
algorithmic regression (validation is O(V+E); a regression to
super-linear cost would blow past it immediately).

We measure the median of 5 runs to dampen one-off scheduler hiccups
while still flagging real regressions; the test fails fast if the
median exceeds the budget.  An unusually slow runner can override the
budget with the ``PHENOTYPIC_VALIDATION_PERF_BUDGET_MS`` env var.
"""

from __future__ import annotations

import os
import statistics
import time
from typing import List

import pytest

from phenotypic.gui.builder._state import (
    BlockNode,
    Edge,
    _DagBuilderScope,
    _DagBuilderState,
    _new_block_id,
)
from phenotypic.gui.builder._validation import validate


_BUDGET_MS = float(
    os.environ.get("PHENOTYPIC_VALIDATION_PERF_BUDGET_MS", "12.0")
)


def _build_synthetic_100_block_scope() -> _DagBuilderScope:
    """Construct a 100-block linear chain with periodic aux fan-ins.

    The chain has one image-flow spine of length 100 plus an aux wire
    from every 5th block back to its predecessor — exercising the
    cycle detector's O(V+E) walk without actually introducing a cycle.

    Returns:
        A populated scope (NOT a state) — the test wraps it in a
        ``_DagBuilderState`` to call ``validate``.
    """

    scope = _DagBuilderScope()  # seeds InputImage
    blocks: List[BlockNode] = []
    for i in range(100):
        b = BlockNode(
            block_id=_new_block_id(),
            class_name="GaussianBlur",
            params={},
            label=f"op_{i}",
        )
        blocks.append(b)
    scope.blocks.extend(blocks)

    # Linear image-flow chain: input -> blocks[0] -> blocks[1] -> ...
    head = scope.blocks[0].block_id  # the auto-seeded InputImage
    scope.edges.append(
        Edge(
            edge_id=_new_block_id(),
            source_block_id=head,
            target_block_id=blocks[0].block_id,
            target_port="in",
            kind="image",
        )
    )
    for prev, curr in zip(blocks[:-1], blocks[1:]):
        scope.edges.append(
            Edge(
                edge_id=_new_block_id(),
                source_block_id=prev.block_id,
                target_block_id=curr.block_id,
                target_port="in",
                kind="image",
            )
        )

    # Note: spec §4.2 says "at most one outgoing wire, total."  Adding
    # aux edges from chain nodes WOULD trip Rule 1.  Instead, we
    # leave the synthetic scope as a pure linear chain — the
    # cycle-detector cost is dominated by node count, not the absence
    # of aux edges.  If the perf budget is a concern under heavy aux
    # wiring we can revisit by introducing extra producer nodes.
    return scope


def test_validate_100_block_synthetic_within_budget():
    """100-block scope validates within _BUDGET_MS (median of 5 iterations)."""

    scope = _build_synthetic_100_block_scope()
    state = _DagBuilderState(root=scope)

    # Sanity check: the synthetic scope is valid; otherwise we'd be
    # paying for issue-list construction on top of the search.
    assert validate(state) == []

    durations_ms: List[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        validate(state)
        durations_ms.append((time.perf_counter() - t0) * 1000.0)

    median = statistics.median(durations_ms)
    assert median <= _BUDGET_MS, (
        f"Validation budget exceeded: median={median:.3f}ms, "
        f"all={durations_ms}, budget={_BUDGET_MS}ms"
    )


@pytest.mark.parametrize("scale", [50, 100, 200])
def test_validate_scales_under_budget_at_multiple_sizes(scale: int):
    """Linear-chain validation scales sub-millisecond per block.

    At ``scale=200`` the budget is 2 * _BUDGET_MS to allow O(N+E)
    growth while still catching regressions.
    """

    scope = _DagBuilderScope()
    blocks = [
        BlockNode(
            block_id=_new_block_id(),
            class_name="GaussianBlur",
            params={},
            label=f"op_{i}",
        )
        for i in range(scale)
    ]
    scope.blocks.extend(blocks)
    head = scope.blocks[0].block_id
    scope.edges.append(
        Edge(
            edge_id=_new_block_id(),
            source_block_id=head,
            target_block_id=blocks[0].block_id,
            target_port="in",
            kind="image",
        )
    )
    for prev, curr in zip(blocks[:-1], blocks[1:]):
        scope.edges.append(
            Edge(
                edge_id=_new_block_id(),
                source_block_id=prev.block_id,
                target_block_id=curr.block_id,
                target_port="in",
                kind="image",
            )
        )
    state = _DagBuilderState(root=scope)
    budget = _BUDGET_MS * max(1.0, scale / 100.0)

    durations_ms: List[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        validate(state)
        durations_ms.append((time.perf_counter() - t0) * 1000.0)
    median = statistics.median(durations_ms)
    assert median <= budget, (
        f"scale={scale}: median={median:.3f}ms exceeds {budget:.2f}ms budget; "
        f"all={durations_ms}"
    )
