# Round 2 — reviewer addendum

Read `brief.md` first (round-0 context). This file carries only what changed.

## Scope: THE DIFF, NOT THE SPEC

Review `snapshots/round-1-spec.diff` (1004 lines, 57 KB). Unchanged sections
were reviewed in round 1 and are out of scope. Do not re-read the whole spec —
open a spec file only to check a specific claim the diff raises.

The plan is unchanged since round 1 except `refinery/ledger.md` and the round-1
resolutions recorded in `plans/2026-08-14-phenotypic-mcp-server/`.

## The severity ratchet

This is round 2. New **Minor** concerns are allowed this round but become
`advisory` from round 3 on. Prefer depth on Critical/Major over breadth.

## What round 1 settled — do NOT re-raise

`ledger.md` holds 63 round-1 concerns and **16 user rulings (USER-1..16)**.
User rulings are permanent and may not be re-litigated absent new evidence.
Read the ledger before writing anything. In particular these are settled:

- The workspace root is mandatory and must contain the image data (USER-5)
- `fastmcp` 3.x from PyPI is the transport (D1a)
- Scope was cut to 26 tools in eight groups; the cut table is in §3.0
- Grouping is by multiple metadata columns, agent-supplied (USER-13)
- One pipeline is tried across the experiment first, descending per-group only
  if needed (USER-13)
- A locally-routed batch job suspends interactive probing (USER-11)

## USER-16 — the deferral criterion (new this round)

A concern may be dispositioned **`deferred-to-2A`** — validated against the
running server rather than decided now — **only when its resolution depends on
observing behaviour that does not exist yet.**

**The test:** if the concern would still need a decision *after* the experiment
returned either result, it is design work and does not qualify.

If you believe one of your concerns is deferrable, say so explicitly and state
**the pass condition** — what observation would settle it. If it does not pass
the test, do not defer it; give me the decision you would make.

## The concurrency specialist has already reported (CONC-19..28)

Its round-2 report is merged into `ledger.md` under "Round 2 — concurrency
specialist". One Critical (CONC-22). **Do not duplicate its findings** — if you
independently reach the same conclusion, cite its ID as an alias and add only
what it did not say. Three of its findings are defects in MY OWN round-1 edits;
I would rather you find more of those than confirm the ones already found.

## Where I most want an independent look

Not a checklist — the places I am least confident the round-1 edits are sound:

1. **§9.3.0.2 multi-group experiments** — the largest new surface. Added at the
   user's request, reviewed by nobody yet except the concurrency specialist.
2. **§10.5's promotion fold** — I collapsed a separate promotion tool into
   `deploy_plan {scope:"full"}`. CONC-22 says the result is incoherent. Is the
   *fold* wrong, or only its placement of the human gate?
3. **§1.6.1 the new NFR table** — I wrote it. Does the rest of the spec actually
   satisfy it, or did I write requirements the design violates?
4. **§3.0's annotations paragraph** — CONC-27 says it is a list, not a rule.
5. **The 26-tool cut** — did anything cut leave a caller dangling?

## Output — WRITE IT TO DISK

Write your report to `refinery/reports/round-2-<your-role>.md` **as you go**,
not at the end. A restart lost all three of you last round; a file on disk
survives. Then return a short summary as your final message.

End with severity-tagged concerns bearing stable IDs (`GEN-*`, `FLOW-*`,
`SIMP-*`, continuing round-1 numbering) and `VERDICT: APPROVE` or
`VERDICT: REVISE`. Tag anything touching the spec `spec-change`, and anything
needing a human ruling `needs-user-input`.
