# A06 TrickTrack CA source reconciliation

## Authority and scope

The executable authority is HSF TrickTrack 1.0.9 at immutable commit
`b164fad1361505ff8dbf328107b645753ce331ac`, licensed under Apache-2.0. This
reconciliation covers only cellular-automaton evolution and exact-length path extraction over a
caller-supplied ordered outer-neighbor graph. The source does not establish fungal graph
construction, orientation or distance compatibility, colony ownership, rasterization, or detector
integration. No biological-benefit claim is made.

The 2018/2019 paper is more than five years old in this fast-moving software field. The pinned
executable source is therefore the behavior authority. The paper remains contextual evidence.

## Line-by-line mapping

| Production lines | Executable source lines | Reconciliation |
|---|---|---|
| `_cellular_automaton.py:10-26` | `CMCell.h:28-49`; `HitChainMaker.ipp:83-130` | The Python result exposes final source states, roots, paths, and the fixed ordinary-round count. CSR offsets are a Python representation drift recorded in `DRIFT.md`. |
| `_cellular_automaton.py:29-79` | No source equivalent | Python-boundary validation. Every deviation is recorded in D01-D04. It does not alter valid-source behavior. |
| `_cellular_automaton.py:82-100` | `CMCell.h:125-142` | Scan each stored outer-neighbor vector in insertion order, compare against the cell's current snapshot state, set the flag for the first equal state, and stop. The private returned match index instruments the source's load-bearing `break`; production behavior uses only whether a match exists. |
| `_cellular_automaton.py:103-115` | `HitChainMaker.ipp:88-100`; `CMCell.h:37-40,125-142` | Compute all ordinary flags before any update, then increment each `uint8` state by its flag in a separate pass. |
| `_cellular_automaton.py:118-152` | `HitChainMaker.ipp:102-117`; `CMCell.h:37-44,125-142` | Visit roots in supplied order. For each root, calculate its flag against the current states, update it immediately, and retain it with the inclusive `>= min_hits - 2` rule before visiting the next root. |
| `_cellular_automaton.py:155-183` | `CMCell.h:200-224`; `HitChainMaker.ipp:119-130` | For each retained root in order, recursively traverse every stored outer neighbor in vector order and emit at exactly `min_hits - 1` cells. The Python ragged representation is D05. No state-descending check, best-fork choice, sorting, or cycle rejection is added. |
| `_cellular_automaton.py:186-228` | `HitChainMaker.ipp:83-100`; `CMCell.h:47-48` | Initialize all states to source `unsigned char` zero and run exactly `min_hits - 3` ordinary rounds. The upper API bound ensures the source state needed for a valid track is at most 255. |
| `_cellular_automaton.py:230-246` | `HitChainMaker.ipp:102-130` | Apply the special root pass, then exact-length extraction, and return every public result field without normalization or reordering. |

## Source-generated golden fixture

`source_harness.cpp` includes and calls the archived `CMCell` implementation directly. It constructs
only the already-frozen ordered cell graph, then serializes:

- stored CSR neighbor and root order;
- first equal neighbor, source flag, and source state after every ordinary round;
- first equal neighbor, source flag, and state after each immediate root update;
- retained roots, final states, path offsets, and every emitted cell index.

The fixture covers the lower and upper `min_hits_per_track` bounds, a synchronous-versus-immediate
counterexample, unsorted forks, a fixed-depth cycle, and an isolated root. `generate_fixture.py`
checks the source archive hash, compiles the C++14 harness against the extracted pinned archive, and
writes a byte-hashed manifest. `tests/unit/sdk_/reconnect/test_cellular_automaton.py:49-104` compares
every trace and result field exactly. Integer results require no numerical tolerance.

## Independent oracle

`docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/cellular_automaton.py`
never imports `phenotypic`. It uses immutable Python tuples plus NumPy `uint8` two-buffer addition
for ordinary rounds, direct ordered root updates, and an iterative reversed-stack DFS. This structure
is independent of production's shared scan helper and recursive DFS. It checks every fixture field
exactly and asserts the two scheduling counterexamples and the state-255 upper bound.

## Fidelity limits

This evidence establishes exact transcription for the frozen numerical core only. It is not evidence
that a fungal cell graph can be built correctly, nor that this method improves fungal detection.
Those are deferred contracts requiring separate sources and ground-truth benchmarks.
