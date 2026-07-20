# A11 scope decision

## Decision

Select Path 1: `phenotypic.analysis.cubical_persistence` reporting only. Do not implement
`persistence_denoise`, `FocusEdgePersistenceDenoise`, feature cancellation, or scalar-image
reconstruction.

## Evidence

The selected executable source defines cubical construction, persistence computation, interval
queries, Betti queries, and persistence-pair cofaces
(`upstream/src/python/gudhi/cubical_complex.py:29-236`). The extension binds those operations at
`upstream/src/python/gudhi/_cubical_complex.cc:87-123`. A targeted search for `reconstruct`,
`reconstruction`, and `cancel` in the selected cubical and persistent-cohomology corpus finds no
scalar-field reconstruction surface. GUDHI's pair-coface output connects intervals back to input
cells for diagnostic/differentiation purposes; it does not edit the input image
(`upstream/src/python/gudhi/cubical_complex.py:173-204`).

Accordingly, the original design statement that the cancellation/Morse step is a small port is
unsupported. Computing intervals and then applying an ad hoc intensity edit would not match an
original algorithm.

The primary Wagner-Chen-Vuçini paper defines cubical cells and boundary reduction over
`F_2` (`wagner-chen-vucini-2011.txt:119-159,206-209`), but its image convention assigns pixels to
vertices and explicitly assumes 4-connectivity (`wagner-chen-vucini-2011.txt:178-191`). The A11
contract instead selects GUDHI's distinct top-dimensional-cell constructor so it can use the
matching coface-pair API. That selected construction has shared-corner 8-connectivity and is
validated independently; the paper is theory context, not evidence that these two input
conventions are interchangeable.

## Epistemic status

Established for this pinned corpus: GUDHI 3.13.0's selected API does not expose scalar
reconstruction. No source-faithful implementation was found in the targeted reconstruction
search. This must not be restated as evidence that no persistence-based image simplification
algorithm exists anywhere.

## Release consequence

The integrator must correct design sections 1, 2, 4.11, 5, 7, and 8 so A11 appears only under the
standalone analysis API and so the topology extra describes GUDHI as an analysis dependency. The
enhancer, reconnect helper, serialization, taxonomy, tuning, and GUI-discovery entries for A11 are
deleted or explicitly deferred.
