---
name: phenotypic-literature-search
description: Use when asked to find, compare or justify algorithms, methods or citations for a PhenoTypic experiment or colony-imaging problem, e.g. "what's the best way to segment touching colonies", "find papers on measuring fungal spread", "is there a better method than X for Y".
---

# Finding algorithms and evidence for a PhenoTypic experiment

1. **Place the problem.** Restate it in imaging terms (signal, noise, failure
   mode) and name the pipeline stage it belongs to. Ask about the organism,
   plate format, imaging setup and timepoints if they're unclear.
2. **Check what we already have before searching.** Look through the relevant
   module, `docs/source/explanation/`, and prior design work in
   `docs/superpowers/specs/` and `plans/`. Find out what has been tried and why
   it fell short. Where you can, reproduce the failure in your own workspace;
   `phenotypic.data.load_synth_yeast_plate()` gives a quick test image.
3. **Search the literature.** Use Scite, bioRxiv and web search. Look beyond
   colony-imaging papers to adjacent fields such as cell microscopy, remote
   sensing or materials imaging. For the strongest leads, follow citations
   both backward and forward.
4. **Check every source.** Cite only papers you actually retrieved, and give a
   DOI for each. Never cite from memory. Check for retractions and corrections,
   and note whether a claim comes from the abstract or the full text.
5. **Present 2–4 candidate approaches.** For each one, give:
   - the core idea in 2–3 sentences, plus the key equation if there is one
   - the primary citation and any reference implementation (repo, language,
     license)
   - the assumptions it makes about the images, and whether our plates meet them
   - its cost in accuracy, compute/memory and implementation effort
   - how it would fit into the repo: the module, the closest existing operation
     to model it on, and which parameters would become fields
   - how it is likely to fail on real plates (condensation, glare, uneven
     lighting, touching colonies, pigmented or translucent colonies)

   End with a recommendation and your reasoning.
6. **Design the validation.** Say what ground truth or controls would show the
   method works (synthetic plates, manual annotations, dilution series,
   replicate plates), which metric to use, and which existing operation is the
   baseline to beat.
7. **If an external algorithm would be ported,** say which source is
   authoritative (the paper or the reference code) and where the two disagree.
   Point to the exact upstream files and versions, and cite `file:line` for
   every claim about what the reference does. Past examples: `FocusEdgePhase`
   and `FocusEdgeMonogenicPhase`, which port Kovesi's
   `phasecong3`/`phasecongmono`.

End every response that uses sources with a References section.
