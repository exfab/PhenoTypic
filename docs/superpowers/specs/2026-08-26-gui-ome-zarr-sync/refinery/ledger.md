# Concern ledger — GUI simplification, Viv rebuild, builder preview

Append-only entries; statuses updated in place. A `resolved` entry naming what changed IS
the provenance lock for that change.

**Prefixes:** `GEN` general-reviewer · `FLOW` data-flow-reviewer · `SIMP`
simplicity-reviewer · `SEC` security specialist · `ALGO` algorithm-fidelity ·
`USER` orchestrator-raised user ruling.

**Statuses:** `open` · `resolved (round N: …)` · `settled-by-user (round N: …)` ·
`conflict (vs <ID>)` · `advisory`.

---

## Round 0 — user rulings

### USER-1 [Critical] [settled-by-user (round 0: NFR lines added to all three specs)]
- Raised: round 0, orchestrator (spec-anchor check)
- Concern: None of the three specs carried a performance/NFR line, which the refinery
  requires as an anchor for precedence tier 8. Without one, every simplicity-vs-performance
  dispute in the Viv rebuild would resolve against performance by default — and that
  rebuild is *entirely motivated* by performance.
- Resolution: User ruled "Interactive over ssh would be nice but correctness is most
  important", and "no performance requirements" for the removals spec. Added as
  `gui-simplification-removals` §9.1, `viewer-viv-rebuild` §9.1, and
  `builder-preview-viv` "Non-functional requirements". **Binding: correctness. Target,
  non-binding: interactive over an SSH tunnel.** The target ranks above tier 8 and below
  correctness, data integrity, and reference faithfulness.
- **Permanent.** No reviewer may re-raise absent new evidence.

### USER-2 [n/a] [settled-by-user (round 0: Summary accepted as Objective)]
- Raised: round 0, orchestrator (spec-anchor check)
- Concern: Specs 1 and 2 have no heading literally named "Objective & Non-goals".
- Resolution: Orchestrator ruling — each spec's `## Summary` states what the change
  achieves and both carry an explicit `## 9. Non-goals`. Accepted as satisfying the anchor;
  no spec edit made. Spec 3 has both headings literally.

---

## Round 1 — pending dispatch

*(entries appended as reports arrive)*
