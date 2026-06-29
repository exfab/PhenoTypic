# Filamentous-fungi (hyphae) detection — research report bundle

**Date:** 2026-06-26
**Question:** Best methods to produce a binary pixel-level mask of filamentous fungi (hyphae/mycelia)
on macroscopic RGB agar-plate images (PhenoTypic context). Curvilinear/ridge detection in clutter;
low-SNR + dense-overlap (junction/gap) regimes; chromatic-aberration + agar-grain confounders.
Baseline to beat: phase congruency + background subtraction + hysteresis + Dijkstra reconnection.

## Files
- **recommendations.md** — assembled long-form report: 3 ranked recommendations vs the baseline
  (classical-CV / small-data-learned / cross-field transfer). `(V)`=verified, `(u)`=unverified inline.
- **breadth-survey.md** — breadth-first survey of 83 methods across 16 families, each with
  Math / Used-for / How-it-works + a master comparison table. No adversarial verification.
- **claims-verified-66.md** — 66 claims that survived 3-vote adversarial verification (quotes + sources).
- **claims-all-2289.md** — all 2,289 extracted claims (sourced; mostly unverified), grouped by topic.

## Provenance / caveats
- Source workflow `deep-research-litmax` (16 topic angles x web + scite/NCBI/bioRxiv -> fetch ->
  3-vote verify). Fetch (435 sources) + verify (381 verdicts) completed; the run's own synthesis did
  not, so reports were assembled in follow-up `assemble` / `breadth-survey` workflows from recovered claims.
- Nothing here was benchmarked on real agar-plate fungal images — cross-domain results are
  transfer-by-analogy and must be validated on plate data before being trusted.
