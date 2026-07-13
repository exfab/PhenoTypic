# Orchestration and execution DAG

## Model and agent policy

The user requested one “5.6 sol” implementation agent per algorithm and an independent
reviewer for each. The current Codex collaboration API does not expose a model selector with
that identifier. At execution time, use that model if the runner exposes it; otherwise use the
strongest session-available frontier model at high reasoning effort. Reviewers must use the
same or stronger tier than implementers.

Each algorithm is one Keystone cluster because the user explicitly requested algorithm-level
isolation. This overrides the clustering skill's normal option to merge adjacent small helpers.
Mechanical shared-file changes are centralized under a Seam owner so algorithm worktrees do not
edit the same registries concurrently.

| Cluster | Implementer | Independent reviewer | Shape | Parallel eligibility |
|---|---|---|---|---|
| A01 GWDT | one dedicated algorithm turn | fresh reviewer, not its author | Keystone | core only |
| A02 Tensor voting | one dedicated algorithm turn | fresh reviewer, not its author | Keystone | core; wrapper conditional on C3 |
| A03 Jerman | one dedicated algorithm turn | fresh reviewer, not its author | Keystone | core plus new wrapper |
| A04 Bowler-hat | one dedicated algorithm turn | fresh reviewer, not its author | Keystone | core plus new wrapper |
| A05 Kalman coast | one dedicated algorithm turn | fresh reviewer, not its author | Keystone | core only |
| A06 Cellular automaton | one dedicated algorithm turn | fresh reviewer, not its author | Keystone | core only |
| A07 NFA | one dedicated algorithm turn | fresh reviewer, not its author | Keystone | statistical core plus detector adapter |
| A08 RORPO | one dedicated algorithm turn | fresh reviewer, not its author | Keystone | core plus new wrapper |
| A09 Rolling Hough | one dedicated algorithm turn | fresh reviewer, not its author | Keystone | core plus new wrapper |
| A10 FilFinder | one dedicated algorithm turn | fresh reviewer, not its author | Keystone | new wrapper only |
| A11 Persistence | one dedicated algorithm turn | fresh reviewer, not its author | Keystone | analysis core, conditional wrapper, or defer |
| S00 Scaffold | integrator | independent phase reviewer | Seam | no |
| S01 Detector strategy integration | integrator | independent phase reviewer | Seam | no |
| S02 Public exports and dependencies | integrator | independent phase reviewer | Seam | no |
| S03 Final simplify/regression | fresh quality agent | independent phase reviewer | Sweep | no |

An “algorithm turn” may be executed by a reused agent process only if its context is reset to the
single assigned method and it did not author that method's review. The ownership unit is the
algorithm, not the process identity.

## Shared Seam ownership

Algorithm agents own only their new implementation modules, focused tests, fixtures, logic
script, and proposed drift rows. They submit a small integration manifest for shared changes.
Only the integrator edits these files. Aliases required by parallel algorithm modules are created
in S00 after contract correction, before the algorithm fan-out; later additions wait for S02:

```text
src/phenotypic/sdk_/reconnect/__init__.py
src/phenotypic/sdk_/typing_.py
src/phenotypic/enhance/__init__.py
src/phenotypic/refine/__init__.py
src/phenotypic/detect/__init__.py
src/phenotypic/analysis/__init__.py
src/phenotypic/detect/_filamentous_fungi_detector.py
src/phenotypic/prefab/_filamentous_fungi_pipeline.py
tests/unit/abc_/test_enhancer_taxonomy.py
tests/unit/sdk_/test_typing_aliases.py
tests/unit/tune/test_annotation_coverage.py
tests/unit/tune/test_enhance_annotations.py
tests/unit/tune/test_detect_annotations.py
docs/superpowers/specs/2026-07-13-fungi-detection-method-ports/drift-register.md
pyproject.toml
uv.lock
```

This matters because GWDT, Kalman, and CA all converge on
`FilamentousFungiDetector._process_tile` (`src/phenotypic/detect/_filamentous_fungi_detector.py:782-880`),
while every enhancer converges on public exports and the taxonomy roster
(`tests/unit/abc_/test_enhancer_taxonomy.py:28-65`).

## Dependency DAG

```text
D0 review corrections C1-C14
  -> D1 pin per-method references and licenses
      -> S00 reconnect scaffold and shared result dataclasses
          -> A01 GWDT core ------------------------------+
          -> A02 Tensor core/wrapper --------------------|
          -> A03 Jerman core/wrapper --------------------|
          -> A04 Bowler-hat core/wrapper ----------------|
          -> A05 Kalman core ----------------------------|
          -> A06 CA core --------------------------------|--> S02 exports/typing/taxonomy
          -> A07 NFA core/wrapper ------------------------|         |
          -> A08 RORPO core/wrapper ----------------------|         +--> S03 final gate
          -> A09 RHT core/wrapper ------------------------|         |
          -> A10 FilFinder wrapper -----------------------|--> S01 detector strategy seam
          -> A11 Persistence analysis/conditional wrapper +

A01 -> S01
A05 -> S01
A06 -> S01
A07 -> S01
A09 -> A10 only for optional oracle comparison, never runtime import
A10 -> S02 topology extra
A11 -> S02 topology extra only if included in release scope
```

The unique algorithm files have zero overlap and are parallel-worktree candidates after S00.
The shared seams are serialized in S01 then S02. Tier labels do not override dependencies.

## Per-algorithm gate

For each `Axx` cluster:

1. Reference gate: corpus, provenance, license, line reconciliation, and explicit source choice.
2. Contract gate: public signature, shapes, coordinate conventions, dtype, invalid-input behavior,
   and all drift decisions are frozen.
3. Red gate: oracle tests, behavioral controls, wrapper forwarding tests, and mutation targets fail
   for the intended reason.
4. Green gate: helper/wrapper implementation passes focused tests and standalone logic script.
5. Mutation gate: each single-change mutant is killed; exact failing node IDs are recorded.
6. Core review gate: an independent reviewer reruns the source oracle, fixture, core mutations,
   owned-module tests, doctest, targeted mypy, and targeted ruff.
7. After core approval, the integrator applies the shared-file manifest in S01/S02. Tests and
   mutants that touch detector adapters, exports, taxonomy, tune coverage, serialization, GUI
   registries, or optional-dependency metadata run only after that seam exists.
8. Final algorithm review gate: the same independent algorithm reviewer returns after S01/S02 to
   sign off the algorithm's seam tests and seam mutants. A separate phase reviewer signs off the
   combined seam. No cluster is complete after core approval alone.

The focused-gate blocks in algorithm files list the eventual full command set. Commands that touch
integrator-owned files are post-S01/S02 commands, even when shown beside the core commands. Each
affected algorithm file separates those commands explicitly where ownership would otherwise be
ambiguous.

## Phase gates

### Gate P0: corrected contracts

No algorithm implementation starts until C1-C14 applicable to it are settled. Corrections may
change helper signatures, so scaffolding before this gate must contain no algorithm stubs.

### Gate P1: Tier A numerical cores

A01-A06 are independently green and reviewed. Run their logic scripts in one process-isolated
loop, then the SDK, enhancer, detector, serialization, tune, mypy, and ruff suites.

### Gate P2: dependency-free Tier B

A07-A09 are independently green and reviewed. Confirm `import phenotypic` and
`import phenotypic.sdk_.reconnect` do not import FilFinder or GUDHI.

### Gate P3: topology-extra Tier B

A10 and any in-scope A11 path are independently green and reviewed. A11 may instead be explicitly
deferred. Run the base environment without the extra, then a fresh environment with
`uv sync --extra topology --group dev`. A missing optional package must raise a targeted call-time
`ImportError`, not fail package import.

### Gate P4: final combined review

A fresh reviewer checks the combined diff for cross-algorithm contract drift. Then a fresh
quality agent performs behavior-preserving simplification, followed by the full regression suite.

## Git/worktree discipline

- One isolated worktree per simultaneously active algorithm cluster.
- Algorithm agents commit only owned new files. They do not stage shared seams.
- The integrator merges reviewed clusters one at a time and is the only writer of seam files.
- Never use `git add -A`; stage scoped paths.
- Preserve the user's current unrelated changes and untracked design artifacts.
