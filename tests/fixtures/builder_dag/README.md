# Builder DAG fixtures

Hand-authored JSON snapshots of `_DagBuilderScope` payloads used by the
DAG-redesign test layers (unit tests in
`tests/unit/gui/builder/` and Playwright E2E in
`tests/e2e/gui/builder/`).

Each `<name>.json` file in this directory is a **bare scope payload**
(not a wrapped `_DagBuilderState`) — it matches the on-disk shape
emitted by `_dag_scope_to_dict` in `phenotypic.gui.builder._state`,
i.e. a top-level object with `"blocks"`, `"edges"`, `"name"`,
`"desc"`, `"nrows"`, `"ncols"`.  The loader
`state_from_json` recognises this shape via the
`_looks_like_dag_payload` heuristic when no `_schema` discriminator
is present, wraps it in a default `_DagBuilderState`, and runs
`_heal_dag_scope_tree` so every reachable scope satisfies Rule 6
(exactly one `InputImage` block) on load.

## Block ID convention

Fixtures use stable 32-character lowercase hex strings of the form
`"00000000000000000000000000000001"`, `"...0002"`, etc.  These match
the `_new_block_id()` output shape (`uuid.uuid4().hex` — 32 chars)
without being randomly generated, so the tests can refer to specific
block_ids in their assertions.

## Provenance

All fixtures are **hand-authored**, never produced by exporting a
running builder.  Several fixtures intentionally encode states the
dispatcher would reject if produced via UI gestures (e.g. forks,
cycles, duplicate `InputImage`) so the validation rules can be
exercised in isolation.

## `expected_issues.json` schema

For invalid fixtures (and any fixture where the validator should
produce a non-empty list of issues), a sibling
`<name>.expected_issues.json` documents the expected output of
`phenotypic.gui.builder._validation.validate(state)` as a list of
issues.  The validator's exact `Issue` dataclass is owned by Agent
1C (see `_validation.py`); fixtures only encode the user-visible
fields:

```json
{
  "issues": [
    {
      "kind": "fork" | "stub" | "required_aux" | "cycle"
            | "container_mode" | "missing_input" | "duplicate_input"
            | "stage_order_hint" | "unknown_class",
      "block_label": "<the BlockNode.label string or null when scope-level>",
      "severity": "error" | "advisory",
      "scope_path": []
    }
  ]
}
```

* `kind` matches the `Issue.kind` Literal in `_validation.py`.
* `block_label` matches the offender's `BlockNode.label` (NOT
  `block_id`, which is opaque — labels are human-readable and
  stable across renames).  For scope-level issues (e.g.
  `missing_input` when no `InputImage` exists, before
  auto-recovery) `block_label` is `null`.
* `severity` is `"error"` for blocking issues (Rules 1–6) and
  `"advisory"` for non-blocking hints (Rule 7 +
  `unknown_class`).
* `scope_path` is the empty list `[]` for issues at the root scope,
  or a list of container `block_id` values (in outer-to-inner
  order) for issues that surface inside a nested container.

Unit tests assert
`set(normalise(actual_issues)) == set(normalise(expected_issues))`
after normalisation; ordering is not significant.

If a fixture is supposed to validate cleanly (e.g. `empty.json`,
`linear_chain.json`), its `expected_issues.json` may be omitted —
absence is treated as "expect zero issues".

## Fixtures owned by Agent 1A (Phase 1 state-shape only)

These fixtures depend only on the schema definitions in
`_state.py`; conversion (Agent 1B) and validation (Agent 1C) tests
treat them as inputs.

* `empty.json` — root scope with only the auto-seeded `InputImage`
  block.  No edges.  Expected to validate cleanly.
* `linear_chain.json` — `InputImage → GaussianBlur → OtsuDetector →
  MeasureSize`.  Three image-flow edges; no aux.  Expected to
  validate cleanly.
* `duplicate_input_image.json` — root scope with TWO `InputImage`
  blocks (invalid state).  Sibling
  `duplicate_input_image.expected_issues.json` lists one
  `duplicate_input` issue at `severity=error`.

Agents 1B and 1C add their own fixtures (round-trip targets,
validation-rule fixtures) alongside these.

## How to load a fixture

In a test:

```python
import json
from pathlib import Path

from phenotypic.gui.builder._state import state_from_json

FIXTURE_DIR = Path(__file__).parents[3] / "fixtures" / "builder_dag"
state = state_from_json(json.loads((FIXTURE_DIR / "empty.json").read_text()))
assert state.root.blocks[0].class_name == "InputImage"
```
