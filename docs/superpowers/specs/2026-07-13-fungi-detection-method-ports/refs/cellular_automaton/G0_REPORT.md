# A06 Cellular Automaton Reference Gate

## Verdict

**G0 is blocked for the full planned image-space contract. No production implementation is
authorized from this corpus.**

TrickTrack 1.0.9 is an authoritative, executable, Apache-2.0-licensed reference for a
cellular-automaton track-seeding core on a caller-supplied detector-layer graph. It is not an
executable reference for PhenoTypic's proposed fungal graph builder or its colony-labelled,
canonical ragged result. The planned split between `build_ca_segment_graph` and `evolve_ca` is
therefore necessary, but only the latter's basic synchronous state transition has a selected
source authority. Treating the complete A06 contract as a port would overstate source fidelity.

This is an established source-level finding. It follows from the pinned files cited below, not
from biological performance evidence. The 2018/2019 TrickTrack paper is more than five years old
in a fast-moving software field; the executable commit, not the paper's general description, is
the behavior authority for this gate.

## Selected corpus

- Primary paper: Valentin Volkl, Felice Pantaleo, and Benedikt Hegner, *TrickTrack: An
  experiment-independent, cellular-automaton based track seeding library*, Connecting the Dots
  2018 proceedings. Local file: `tricktrack-paper.pdf`. Official event record:
  <https://indico.cern.ch/event/658267/contributions/2813731/>.
- Executable source: HSF/TrickTrack 1.0.9, immutable commit
  `b164fad1361505ff8dbf328107b645753ce331ac`.
- Complete source archive: `TrickTrack-b164fad.tar.gz`.
- License: Apache License 2.0, stated by the project and included in the pinned corpus
  (`source/README.md:49-55`, `source/LICENSE:1-201`).
- Upstream origin: the repository identifies the CMSSW files from which TrickTrack was derived
  (`source/README.md:1-5`, `source/doc/CMSSW_sources.md:2-13`).
- Hashes and exact execution commands: `CHECKSUMS.sha256` and `RUN.md`.

The bundled upstream integration oracle compiles and passes on this host with Apple Clang 21.0.0.
Its assertion is limited to finding eight three-hit tracklets
(`source/tests/integration/test_integration.cpp:43-61,77-85`); it does not establish the planned
PhenoTypic fields or ordering.

## What TrickTrack establishes

| Topic | Source-established behavior |
|---|---|
| Cell definition | A cell refers to a hit doublet, and `HitDoublets` is explicitly a pair of seeding hits in neighboring detector layers (`source/include/tricktrack/HitDoublets.h:13-23,32-38`; `source/include/tricktrack/CMCell.h:53-72`). |
| Directed topology | The caller provides named directed layer paths. `createGraph` converts consecutive names into layer pairs and designates the first name of every path as a root (`source/src/tricktrack/CMGraphUtils.cpp:9-25,41-65,96-109`). |
| Friendship | Cells sharing a hit are considered in source container order, then a caller-supplied `TripletFilter` decides whether to add the directed outer-neighbor relation (`source/include/tricktrack/HitChainMaker.ipp:51-65`; `source/include/tricktrack/CMCell.h:144-176,192-195`). |
| Source geometric filter | The supplied default callback is detector-specific. It uses 3D/radial coordinates, transverse-momentum and beam-region parameters, R-Z alignment, and X-Y curvature (`source/include/tricktrack/TripletFilter.h:24-33,35-95,98-120`). |
| State and dtype | Each state and same-state flag is `unsigned char`, initialized to zero. A state increments by one if at least one outer neighbor had the same snapshot state (`source/include/tricktrack/CMCell.h:28-49,125-142`). |
| Update ordering | For ordinary iterations, every cell first computes its same-state flag, then every state is updated in a second pass, so those iterations are synchronous (`source/include/tricktrack/HitChainMaker.ipp:83-100`). |
| Iteration bound | Evolution is not run to convergence. The count is derived from `minHitsPerNtuplet`; the general loop uses `minHitsPerNtuplet - 2`, followed by a special last pass over root layer pairs only (`source/include/tricktrack/HitChainMaker.ipp:83-117`). |
| Root rule | A root is retained if its state is at least `minHitsPerNtuplet - 2` after the special last pass (`source/include/tricktrack/HitChainMaker.ipp:102-115`; `source/include/tricktrack/CMCell.h:42-44`). |
| Fork behavior | Extraction recursively visits every stored outer neighbor in vector insertion order and emits paths at one exact requested length. It does not choose a best fork or canonically sort tied forks (`source/include/tricktrack/CMCell.h:200-224`; `source/include/tricktrack/HitChainMaker.ipp:119-130`). |
| Public output | The integration/Python conversion returns ordered hit identifiers reconstructed from cell paths, not segment states, predecessor arrays, convergence counts, or colony ownership (`source/tests/integration/test_integration.cpp:59-85`; complete source archive `python/pyTT.cpp:152-169`). |

## Contract reconciliation

| Planned A06 item | Source status | Gate consequence |
|---|---|---|
| `(row, col)` float64 points and axial tangent angles in radians | Not present. TrickTrack consumes 3D detector hits and radial/azimuthal geometry. | Must be specified and validated as PhenoTypic-owned logic. It cannot be described as transcribed TrickTrack behavior. |
| Fungal `layers`, endpoint roles, component IDs, `max_link_distance`, `max_layer_skip`, and inclusive angle/distance thresholds | Not present. Detector layer paths and hit doublets are supplied externally. | Full `build_ca_segment_graph` lacks an executable authority. |
| Acyclicity and cycle rejection | Not validated by `createGraph`; it accepts lists of paths and does not run a cycle check (`source/src/tricktrack/CMGraphUtils.cpp:9-109`). | Planned rejection behavior is a new safety contract. |
| Float/int input validation, finite checks, sorted layers, duplicate rejection, and read-only C-contiguous arrays | Not present in the C++ API. | New Python API behavior. |
| `int64` states and convergence to a fixed point | Contradicts the selected source's `unsigned char` state and fixed hit-count-derived evolution. | Requires an intentional deviation plus an independent mathematical specification. Calling it source-identical is incorrect. |
| Canonical lexicographic segments, compatibility, predecessors, and track ties | Contradicts insertion-order cell construction and branch enumeration. | Requires a drift row and permutation-invariance oracle if adopted. |
| `CATrackResult` with all source-visible intermediates | Not present. | New result schema must be designed and independently tested. |
| Colony-labelled ragged paths and ownership conflict rejection | Not present. | New domain integration semantics. |
| Empty graph offsets `[0]` and zero iterations | Not specified by source tests. | New Python contract. |

## Frozen source-faithful sub-contract available for approval

A narrower core can be reviewed independently if the design explicitly adopts it:

1. Inputs are an already validated directed cell graph and a requested minimum hit count of at
   least three.
2. States begin at zero.
3. During each ordinary iteration, each cell observes the prior snapshot and marks whether any
   outer neighbor has equal state; all marked cells then increment by exactly one.
4. The iteration schedule matches `HitChainMaker::evolve`, including its root-only final pass,
   unless that special pass is explicitly listed as a deviation.
5. Root qualification uses state `>= min_hits - 2`.
6. Extraction enumerates all exact-length outer-neighbor branches in input insertion order.

This narrower sub-contract still requires a source harness that serializes cells, all states,
root IDs, neighbor lists, and every extracted path. The bundled integration test is not a
sufficient golden fixture because it checks only the final count.

## Decisions required before G0 can pass

Choose one of these explicit paths:

1. **Source-faithful evolution plus a separately named PhenoTypic graph capability.** Preserve
   TrickTrack's fixed iteration/root/extraction semantics in one core. Specify fungal graph
   construction as new project logic, with its own mathematical oracle and no port-fidelity claim.
2. **PhenoTypic DAG longest-path algorithm.** Keep the current snapshot-to-DP validation design,
   canonical ordering, cycle rejection, and colony outputs, but rename and document it as a new
   image-space algorithm inspired by cellular automata rather than as a TrickTrack port.
3. **Locate a matching executable source.** It must define image-space segment construction,
   axial orientation, boundary semantics, deterministic forks, and the requested result fields.

Until one path is approved and its full contract is frozen, no red tests, fixtures, production
module, logic-validation script, or drift register can honestly claim to validate the planned A06
implementation against an original algorithm.

## G0 checklist

| Requirement | Status |
|---|---|
| Complete local source corpus | Pass |
| Immutable revision and hashes | Pass |
| Compatible source license | Pass, Apache-2.0 |
| Primary paper pinned | Pass, with age caveat |
| Executable source oracle | Pass for upstream integration test |
| Full planned input/output contract supported by source | **Fail** |
| Axes, orientation, boundaries, invalid inputs, ties, and dtype frozen | **Fail for fungal adaptation** |
| Exact source-visible golden fixture | Pending after contract decision |
| Independent G0 reviewer | Required |
