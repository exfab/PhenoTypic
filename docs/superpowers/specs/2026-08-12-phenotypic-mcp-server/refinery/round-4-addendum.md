# Round 4 — reviewer addendum. THIS IS THE ROUND CAP.

Read `ledger.md` (**USER-1..27, all permanent**) and
`defining-sections-map.md` before anything else. Then your scope:
`snapshots/round-3-spec.diff` + `round-3-plan.diff`.

## This is the last round. What that changes

There is no round 5. Anything you raise that is not fixed here goes to the user
as an open item with my recommendation attached. So:

- **Raise Critical and Major only.** The ratchet has been binding since round 3.
- **A Major you cannot state a concrete fix for is an Advisory.** One line, no
  argument.
- **APPROVE is the expected outcome if the diff is sound.** Three of you said in
  round 3 that the *decisions* were right and only the propagation was missing.
  That propagation is what this diff is. Do not manufacture a REVISE.

## What round 3 did

**The round-3 finding was structural, and it was the same one all four of you
reached independently:** decisions were written into the sections that *explain*
them and never into the sections that *define* the artifacts and signatures.

So a sweep built `refinery/defining-sections-map.md` — one row per ruling,
mapping explaining sections against defining sections, with a Y/N/partial column.
**It found 12 of 23 live rulings had a defining-section gap.** Two were total
misses (USER-21, USER-24) — both being rulings whose entire content was a new
primitive, which is the pattern: *a ruling that deletes a design and keeps one
primitive leaves the primitive undeclared, because it is written into the section
doing the deleting.*

Also settled this round, as USER-25..27:
- **USER-25** — §1.6.1's `W0` row gains one exemption (a human-gate elicitation
  outstanding). This settled a Critical: USER-18 had been applied to only one of
  the two human gates, and `campaign_approve` was still `W0` with an explicit
  minutes-long wait. USER-18's outcome is unchanged; only the `W0`-violation half
  of its basis retires.
- **USER-26** — `parent ∩ group_filter` resolves **in place to a manifest** at
  plan time, bound by the token's digest. **Verified cost:** the public CLI's
  `--input` is a single `click.Path` (`phenotypicCLI.py:924-929`), so this needs
  a new top-level flag. `_cli_staged_slurm_worker.py:422` already takes
  `--manifest` internally. Recorded as a **new §7 prerequisite and plan task**.
- **USER-27** — ~161 lines of retired-alternative narration removed. §2.6's two
  shipped-code subsections **moved to §7 P2** as prerequisite rows rather than
  deleted. Verified: 423 rule identifiers before, 423 after, **none lost**.

## Where I am least confident

1. **The spec is now 6237 lines, up from 5753.** The trim removed 161 and the
   propagation added more than that. I believe the additions are defining-section
   content that had to exist — but I have not audited that belief. If the growth
   is narration returning by another route, say so.
2. **The propagation sweep and the trim were done by subagents.** I verified the
   trim lost no rule tokens and that the §7 move landed. I have **not**
   independently verified that each of the sweep's ~30 defining-section edits is
   correct in substance rather than merely present.
3. **USER-26's manifest is specified but its prerequisite is new.** §7 and the
   plan gained it late. Check it is real work with a real owner, not a line.
4. **The per-kind `decision` derivation table** (§3.2) is new and was written to
   fix a finding that the derivation was correct for one edit kind and inverted
   for another. Verify it against the actual edit kinds.

## Standing items neither fixed nor forgotten

Open from earlier rounds, untouched by any diff: GEN-4, 5, 6, 8, 9, 10, 11, 12;
FLOW-1, 2, 5; CONC-18; GEN-18 (`--restart`, `--slurm k=v`, `--gpu-slurm` have no
`_services` emitter). **CONC-8 is a confirmed defect in shipped code**, now a §7
P2 row. These go to the user as open items; do not spend the round re-arguing
them, but say if any has become *more* dangerous under round 3's changes.

## Output

`refinery/reports/round-4-<your-role>.md`, written as you go. End with
severity-tagged concerns and `VERDICT: APPROVE` or `VERDICT: REVISE`.
