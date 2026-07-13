# Tensor-voting integration handoff

The A02 core intentionally does not modify shared files. The serialized integrator must complete
the following after independent G3 approval:

1. Re-export `tensor_vote` from `phenotypic.sdk_.reconnect` without importing any wrapper.
2. Preserve the Trevor Linton notice from `source/license.txt` in the repository's root `NOTICE`
   and `licenses/` inventory. Also preserve and disclose the conflicting LGPL-3.0 claim in
   `source/README`; do not label the archive unambiguously BSD or LGPL.
3. Keep the standalone enhancer deferred. No selected source defines an image-to-orientation
   estimator for this helper contract.
4. Treat detector token selection, theta conversion, saliency normalization, ball diagnostics,
   and any cost/evidence equation as separately reviewed PhenoTypic adaptations.
5. When a detector seam is proposed, pass a sparse positive token field and Cartesian axial
   tangent angles. Add tests for the detector's own angle conversion and threshold; the core must
   remain unnormalized.
6. Return the integrated seam to the same A02 reviewer. Any semantic edit to `_tensor_voting.py`
   invalidates the reviewed SHA and requires a fresh source-fixture and mutation gate.
