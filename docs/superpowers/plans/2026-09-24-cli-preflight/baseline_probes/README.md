# Baseline probes

These are the scripts behind
`docs/superpowers/reports/2026-09-24-cli-preflight/claim-verification.md`, committed
exactly as they ran against `81d19ec` so the report's outputs can be reproduced. They
import `phenotypic` and drive shipped code, which is why they live beside the plan rather
than under `logic_validation_scripts/` (see `CLAUDE.md`, Agentic AI File Rules).

They are evidence, not maintained code: they are left unlinted so that the committed bytes
match what produced the report. Each plan task that fixes a probed defect turns the probe
into a proper regression test under `tests/`.

Run one with `uv run python <probe>.py` from a scratch directory. `p1/` needs `mods/` on
`PYTHONPATH`; the CLI probes for report §2 were shell invocations and are reproduced in
the report text.
