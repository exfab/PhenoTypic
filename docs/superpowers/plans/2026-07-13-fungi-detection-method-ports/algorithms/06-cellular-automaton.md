# A06: cellular-automaton track finder

**Implementer:** one dedicated 5.6-sol/high-effort algorithm turn
**Reviewer:** independent 5.6-sol/high-effort turn
**Shape:** Keystone TrickTrack CA core only; fungal detector seam deferred
**Blocked by:** C7 and S00

## Corrected contract

The selected executable authority is HSF TrickTrack v1.0.9 at commit
`b164fad1361505ff8dbf328107b645753ce331ac`. It defines CA evolution and exact-length
fork extraction over a caller-supplied, ordered friendship graph. It does not define fungal
point/layer construction, row/column or axial-angle semantics, component ownership,
rasterization, or colony-labelled output. Those seams are deferred rather than represented as
source-faithful behavior.

The source-faithful helper contract is:

```python
@dataclass(frozen=True)
class TrickTrackCAResult:
    states: np.ndarray              # uint8, (M,)
    retained_root_indices: np.ndarray  # int64, supplied root order
    path_offsets: np.ndarray        # int64, (R + 1,)
    path_cell_indices: np.ndarray   # int64, exact-length DFS paths
    ordinary_rounds: int

def tricktrack_ca(
    outer_neighbor_offsets: np.ndarray,
    outer_neighbor_indices: np.ndarray,
    root_cell_indices: np.ndarray,
    *,
    min_hits_per_track: int,
) -> TrickTrackCAResult: ...
```

CSR cell and neighbor order, and supplied root order, are load-bearing. The helper never sorts or
canonicalizes. States and same-state flags are `uint8` and start at zero. Validate
`3 <= min_hits_per_track <= 257`, valid int64 CSR indices, and optional DAG safety as explicit
Python-boundary drift. Run exactly `min_hits_per_track - 3` ordinary globally synchronous rounds.
Each cell scans stored outer neighbors in supplied order, stops at the first equal-state neighbor,
and is incremented in the separate update pass when flagged. Then run one special final pass over
roots in supplied order: compute and apply each root update immediately, and retain roots whose
state is at least `min_hits_per_track - 2`. For every retained root, depth-first traverse stored
outer-neighbor vectors in insertion order and emit every cell-index path of exactly
`min_hits_per_track - 1` cells. Extraction does not test descending states, choose a best fork,
sort paths, or continue beyond exact length.

The executable source is authoritative where its use of stored outer neighbors differs from the
paper's inner-neighbor wording.

## Owned files and tasks

```text
src/phenotypic/sdk_/reconnect/_cellular_automaton.py
tests/unit/sdk_/reconnect/test_cellular_automaton.py
tests/fixtures/reconnect/cellular_automaton/
docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/cellular_automaton.py
refs/cellular_automaton corpus and reconciliation
```

1. Pin the selected CATS/TrickTrack/related implementation, license, update rule, and extraction.
2. Instrument the archived executable source to capture ordered neighbors, every ordinary-round
   flag and state, immediate root updates, retained roots, and all DFS paths.
3. Capture straight, equal/unequal fork, root-order, neighbor-order, exact-length truncation,
   isolated, lower/upper `min_hits_per_track`, and invalid CSR cases with every result field.
4. Write an independent two-buffer ordinary-round oracle plus a separate immediate-root-pass DFS
   oracle, then red tests.
5. Implement the source-faithful core without graph construction or detector adaptation.
6. Reviewer reruns the source instrumentation, oracle, fixtures, mutants, and focused gates.

## Logic-validation script

Re-derive the fixed ordinary-round schedule with separate flag/state buffers, then independently
apply the immediate root pass and insertion-order DFS. Check every fixture field exactly, including
the counterexample that distinguishes globally synchronous ordinary rounds from immediate root
updates. No convergence or longest-path claim is made.

## Required core mutants

- in-place rather than synchronous ordinary update;
- scan inner instead of stored outer neighbors;
- do not stop at the first equal-state neighbor;
- run to convergence or use the wrong fixed round count;
- make the final root pass globally synchronous;
- change the inclusive retained-root threshold;
- sort neighbors, roots, or emitted paths;
- enforce descending states during DFS, keep one fork, or continue beyond exact length;
- promote states to int64 and thereby hide the source uint8 bound;

## Deferred seam

Fungal graph construction, angle and distance compatibility, colony ownership, rasterization,
cost sampling, and detector integration require a separately sourced or explicitly novel design.
They are not release gates for this core-only A06 port.

## Focused gate

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-07-13-fungi-detection-method-ports/cellular_automaton.py
uv run pytest tests/unit/sdk_/reconnect/test_cellular_automaton.py -q
uv run mypy src/phenotypic/sdk_/reconnect/_cellular_automaton.py
uv run ruff check src/phenotypic/sdk_/reconnect/_cellular_automaton.py tests/unit/sdk_/reconnect/test_cellular_automaton.py
```

There is no A06 detector seam in this release. Any later seam invalidates this core-only scope and
requires a new contract and returning reviewer.
