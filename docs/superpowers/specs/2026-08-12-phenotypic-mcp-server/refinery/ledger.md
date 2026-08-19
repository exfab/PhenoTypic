# Concern ledger — PhenoTypic MCP server

Append-only. Statuses: `open` · `resolved (round N: what changed)` ·
`settled-by-user (round N: ruling)` · `conflict (vs ID)` · `advisory`.

## Pre-loop rulings (permanent — no reviewer may re-raise absent new evidence)

### USER-1 [settled-by-user (round 0)]
Performance/NFR anchor. **Ruling:** interactive responsiveness for `W0`/`W1`;
`W0` < 1 s and never blocking the event loop; `W1` bounded by
`probe_max_images` + `probe_timeout_s`; `W2`/`W3` submit-and-poll with no
latency requirement. Written into the spec as §1.6.1.

### USER-2 [settled-by-user (round 0)]
Rotating specialist = **concurrency**. Parallelism is the spec's central
mechanism and three open hazards are concurrency-shaped.

### USER-3 [settled-by-user (earlier)]
MCPB packaging **rejected**; local stdio confirmed. Bundling would produce two
divergent copies of `phenotypic`.

### USER-4 [settled-by-user (earlier)]
SDK = PyPI **`fastmcp` 3.x** (D1a), superseding the official SDK's bundled
FastMCP 1.0.

### USER-5 [settled-by-user (earlier)]
**Elicitation adopted** (D6) for `campaign_approve` and the §10.5 promotion
gate — shaped now, implemented in Phase 2C, gated on a live test of whether it
surfaces from a subagent's call.

### USER-6 [settled-by-user (earlier)]
No server-side `plate.nrows`/`ncols` backstop (D3) — grid sections are not
always filled, so the product is a poor proxy for expected count.

### USER-7 [settled-by-user (earlier)]
Reviewers run at **cluster boundaries**, not per task (I8).

## Round 1 concerns

_(populated after the panel reports)_
