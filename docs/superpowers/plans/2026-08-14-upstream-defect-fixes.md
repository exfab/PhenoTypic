# Upstream Defect Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two confirmed CLI defects — macOS AppleDouble sidecar files counted
as input images, and SLURM chunked runs silently dropping trailing images from
`master_measurements`.

**Architecture:** Both are discovery bugs with the same shape: a filter that
admits something it should not, and an aggregation that reads a stale derived
artifact instead of the source. Each fix is a small guard plus a regression test
that reproduces the loss first.

**Tech Stack:** Python 3.12, polars, pytest, `uv` as the sole runner.

**Spec:** None — these originate as defect reports from
`docs/superpowers/specs/2026-08-13-streamlit-run-monitor-design.md` §3.3 and
OQ-10, where both were found while validating that design against a real run at
`/Volumes/T9/exfab/UCR-033-E-D_LinzerGanoderma/Results/frame00_discriminability`.
Evidence for each is restated in its task.

## Global Constraints

- `uv` is the sole package manager and runner. Never bare `python` or `pip`.
- `uv run ruff check --fix <paths you changed>` — **always pass explicit paths**.
- Tests must be able to fail: reproduce each defect before fixing it, and record
  the observed failure output in the commit.
- Vendored reference sources under `docs/superpowers/specs/*/refs/` are read-only.
- Do not modify the validation run at `/Volumes/T9/...` — it is read-only evidence.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/phenotypic/_cli/_cli_directory_scanner.py` | Input image discovery and counting | Modify — exclude dotfiles at 4 scan sites |
| `tests/unit/cli/test_cli_directory_scanner_dotfiles.py` | Regression for defect 1 | Create |
| `src/phenotypic/_cli/_cli_checkpoint_handler.py` | SLURM checkpoint/finalize entry points | Modify — flush unchunked parquets before final aggregation |
| `tests/unit/cli/test_slurm_finalize_flushes_trailing.py` | Regression for defect 2 | Create |

---

## Task 1: Exclude AppleDouble and dotfiles from image discovery

**Evidence.** `Path("._d000436_300_001.tif").suffix == ".tif"`, so the scanner's
`p.suffix.lower() in valid_exts` filter admits macOS AppleDouble sidecars. On an
exFAT volume macOS writes one beside every file. Observed on the validation run:
`.phenotypic/progress/manifest.json` reports `total_images: 60` for 30 images,
`completed: 30`, `pending: 30`, `is_complete: false` — on a run that finished
successfully. Anything gating on `is_complete` misreads every such run as
unfinished, including the GUI runs registry (`gui/shell/_runs_registry.py:631`)
and the SLURM observer (`gui/run_console/_slurm_observer.py:1245`).

**Files:**
- Modify: `src/phenotypic/_cli/_cli_directory_scanner.py:78`, `:90`, `:283`, `:294`
- Test: `tests/unit/cli/test_cli_directory_scanner_dotfiles.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `_is_image_file(p: Path, valid_exts: set[str]) -> bool` — the single
  predicate all four sites call. Task 2 does not depend on it.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/cli/test_cli_directory_scanner_dotfiles.py`:

```python
"""AppleDouble sidecars must never be counted as input images.

macOS writes a `._<name>` sidecar beside every file on exFAT/FAT volumes.
`Path("._x.tif").suffix` is `".tif"`, so an extension-only filter admits them
and every count doubles.
"""
from pathlib import Path

from phenotypic._cli._cli_directory_scanner import (
    scan_input_path,
    count_images_by_dataset,
)


def _make_tree(root: Path) -> None:
    """Two real images per dataset, each with an AppleDouble sidecar."""
    for ds in ("plate_a", "plate_b"):
        d = root / ds
        d.mkdir(parents=True)
        for stem in ("img_001", "img_002"):
            (d / f"{stem}.tif").write_bytes(b"real")
            (d / f"._{stem}.tif").write_bytes(b"\x00\x05\x16\x07")  # AppleDouble


def test_scan_excludes_appledouble_sidecars(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    datasets = scan_input_path(tmp_path)
    for name, images in datasets.items():
        names = [p.name for p in images]
        assert not any(n.startswith("._") for n in names), (
            f"dataset {name} admitted AppleDouble sidecars: {names}"
        )
        assert len(images) == 2, f"dataset {name} has {len(images)} images, want 2"


def test_count_excludes_appledouble_sidecars(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    counts = count_images_by_dataset(tmp_path)
    assert sum(counts.values()) == 4, f"counted {counts}, want 2 per dataset"


def test_dotfiles_generally_excluded(tmp_path: Path) -> None:
    """.DS_Store and any other dotfile with an image extension are not inputs."""
    d = tmp_path / "plate_a"
    d.mkdir(parents=True)
    (d / "img_001.tif").write_bytes(b"real")
    (d / ".hidden.tif").write_bytes(b"nope")
    datasets = scan_input_path(tmp_path)
    images = next(iter(datasets.values()))
    assert [p.name for p in images] == ["img_001.tif"]
```

- [ ] **Step 2: Run the test and confirm it fails for the right reason**

```bash
uv run pytest tests/unit/cli/test_cli_directory_scanner_dotfiles.py -v
```

Expected: `test_scan_excludes_appledouble_sidecars` and
`test_count_excludes_appledouble_sidecars` FAIL with 4 images / count 8 rather
than 2 / 4. If the import names are wrong, correct them from the module's
actual public functions before proceeding — do not weaken the assertions.

- [ ] **Step 3: Add the shared predicate**

In `src/phenotypic/_cli/_cli_directory_scanner.py`, above the first use:

```python
def _is_image_file(path: Path, valid_exts: set[str]) -> bool:
    """True for a real input image.

    Excludes dotfiles: macOS writes an AppleDouble `._<name>` sidecar beside
    every file on exFAT/FAT volumes, and `Path("._x.tif").suffix` is `".tif"`,
    so an extension-only filter counts each image twice.
    """
    return (
        path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in valid_exts
    )
```

- [ ] **Step 4: Route all four scan sites through it**

Replace at `:78`, `:90`, `:283`, `:294` — each currently reads
`p.is_file() and p.suffix.lower() in valid_exts`:

```python
        if _is_image_file(p, valid_exts)
```

Leave `:62` (the single-file case) unchanged: passing a dotfile explicitly is a
deliberate user act, not directory discovery, and rejecting it there would be a
separate behaviour change.

- [ ] **Step 5: Run the tests**

```bash
uv run pytest tests/unit/cli/test_cli_directory_scanner_dotfiles.py -v
uv run pytest tests/unit/cli -q
```

Expected: new tests PASS, existing CLI suite unchanged (505 passed, 1 skipped at
baseline).

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/_cli/_cli_directory_scanner.py tests/unit/cli/test_cli_directory_scanner_dotfiles.py
git add src/phenotypic/_cli/_cli_directory_scanner.py tests/unit/cli/test_cli_directory_scanner_dotfiles.py
git commit -m "fix(cli): exclude dotfiles from input image discovery

macOS writes an AppleDouble ._<name> sidecar beside every file on
exFAT/FAT volumes, and Path('._x.tif').suffix is '.tif', so the
extension-only filter counted every image twice. Observed on a real run:
manifest.json reported total_images 60 for 30 images, with
is_complete false on a run that finished."
```

---

## Task 2: Flush unchunked parquets before final SLURM aggregation

**Evidence.** Three links, each verified in source:

1. `_build_entry_list` (`_cli_slurm_array_scripts.py:51-60`) appends a checkpoint
   sentinel only after every `checkpoint_interval` images, and emits **no
   terminal sentinel** — its own docstring says "Terminal work is submitted as a
   separate dependent lifecycle job."
2. `_resolve_checkpoint_interval` (`:86-110`) clamps the interval to `[50, 500]`.
3. `_run_finalize` (`_cli_checkpoint_handler.py:238-272`) calls
   `aggregate_measurements` directly, with no chunk-write first. That resolves
   sources via `discover_measurement_sources`
   (`_measurement_sources.py:104-120`), which **prefers
   `_dataset_aggregated.parquet` and skips the individual per-image parquets**.

So on a SLURM run whose image count is not a multiple of the interval, the
trailing `n_images % interval` images (up to 499) are written to disk as
per-image parquets, never chunked into the aggregate, and then excluded from the
master because aggregation reads the aggregate alone. The data is on disk; the
published result silently omits it.

**Fix:** flush before aggregating. `_scan_unchunked_parquets` already tracks
which per-image parquets have been consumed, so the flush is idempotent and a
no-op when the image count divides evenly.

**Files:**
- Modify: `src/phenotypic/_cli/_cli_checkpoint_handler.py` (in `_run_finalize`,
  immediately before the `aggregate_measurements` call at `:267`)
- Test: `tests/unit/cli/test_slurm_finalize_flushes_trailing.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: no new public symbol. Behaviour change only.

- [ ] **Step 1: Read the two functions you will call**

```bash
uv run python -c "
import inspect
from phenotypic._cli import _cli_chunk_writer as m
print(inspect.signature(m.aggregate_chunks))
print(inspect.signature(m._scan_unchunked_parquets))
"
```

Record the exact signatures — Step 3's call must match them. If
`aggregate_chunks` is not the right entry point, use whichever function the
checkpoint sentinel path invokes (trace from
`_cli_checkpoint_handler.py`'s `--checkpoint-type checkpoint` branch).

- [ ] **Step 2: Write the failing test**

Create `tests/unit/cli/test_slurm_finalize_flushes_trailing.py`:

```python
"""Trailing images must reach the master when the count is not a multiple
of the SLURM checkpoint interval.

Sentinels fire only every `checkpoint_interval` images and there is no
terminal sentinel, so the last partial group is never chunked. Final
aggregation prefers `_dataset_aggregated.parquet` over the individual
per-image parquets, so those rows vanish from the published master.
"""
from pathlib import Path

import polars as pl

from phenotypic.sdk_ import dataset_measurements_dir
from phenotypic._cli._cli_chunk_writer import aggregate_chunks


DATASET = "plate_a"


def _write_per_image(output_dir: Path, stems: list[str]) -> None:
    d = dataset_measurements_dir(output_dir, DATASET)
    d.mkdir(parents=True, exist_ok=True)
    for i, stem in enumerate(stems):
        pl.DataFrame(
            {
                "MetadataImage_ImageName": [stem],
                "Object_Label": [1],
                "Shape_Area": [float(i)],
            }
        ).write_parquet(d / f"{stem}.parquet")


def test_trailing_images_reach_the_aggregate(tmp_path: Path) -> None:
    """Chunk only the first 2 of 3 images, then aggregate: all 3 must appear."""
    stems = ["img_001", "img_002", "img_003"]
    _write_per_image(tmp_path, stems)

    # Simulate a checkpoint that consumed only the first two images —
    # the state a run leaves when 3 % interval != 0.
    aggregate_chunks(tmp_path, [DATASET])  # consumes what exists so far

    # A third image lands after the last checkpoint sentinel.
    _write_per_image(tmp_path, ["img_004"])

    from phenotypic._cli._cli_output_manager import aggregate_measurements

    aggregate_measurements(output_dir=tmp_path, dataset_names=[DATASET])

    from phenotypic.sdk_ import master_measurements_parquet_path

    master = pl.read_parquet(master_measurements_parquet_path(tmp_path))
    got = set(master["MetadataImage_ImageName"].to_list())
    assert got == {"img_001", "img_002", "img_003", "img_004"}, (
        f"master is missing trailing images: {sorted(got)}"
    )
```

- [ ] **Step 3: Run it and record the failure**

```bash
uv run pytest tests/unit/cli/test_slurm_finalize_flushes_trailing.py -v
```

Expected: FAIL — `img_004` absent from the master. **Paste the actual assertion
output into the commit message in Step 6.** If it passes, the defect does not
reproduce this way: stop, and report what the aggregation actually did rather
than adjusting the test until it fails.

- [ ] **Step 4: Flush before aggregating**

In `src/phenotypic/_cli/_cli_checkpoint_handler.py`, in `_run_finalize`,
immediately before the `aggregate_measurements(` call:

```python
    # Flush any per-image parquets written after the last checkpoint sentinel.
    # Sentinels fire every `checkpoint_interval` images and there is no terminal
    # sentinel (_cli_slurm_array_scripts.py:51-60), so a run whose image count is
    # not a multiple of the interval leaves up to interval-1 images unchunked.
    # aggregate_measurements prefers _dataset_aggregated.parquet over the
    # individual parquets, so without this flush those rows never reach the
    # master. Idempotent: _scan_unchunked_parquets skips already-consumed files.
    from ._cli_chunk_writer import aggregate_chunks

    aggregate_chunks(output_dir, list(datasets_totals.keys()))
    _check_epoch()
```

- [ ] **Step 5: Run the tests**

```bash
uv run pytest tests/unit/cli/test_slurm_finalize_flushes_trailing.py -v
uv run pytest tests/unit/cli -q
```

Expected: new test PASSES, existing CLI suite unchanged.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check --fix src/phenotypic/_cli/_cli_checkpoint_handler.py tests/unit/cli/test_slurm_finalize_flushes_trailing.py
git add src/phenotypic/_cli/_cli_checkpoint_handler.py tests/unit/cli/test_slurm_finalize_flushes_trailing.py
git commit -m "fix(cli): flush unchunked parquets before final SLURM aggregation

Checkpoint sentinels fire every checkpoint_interval images (clamped to
[50,500]) and there is no terminal sentinel, so a run whose image count
is not a multiple of the interval leaves up to interval-1 images
unchunked. Final aggregation resolves sources via
discover_measurement_sources, which prefers _dataset_aggregated.parquet
and skips the individual per-image parquets -- so those images were on
disk but absent from the published master.

Observed failure before the fix:
<paste the Step 3 assertion output here>"
```

---

## Task 3: Correct the monitor spec's now-stale typo claims

**Why this is here.** The `MetadatasCondition` schema typo — recorded in the
monitor spec as a live production defect and tracked as a third chip — was fixed
upstream by the `origin/main` merge at `86d341b2f`. Verified in this worktree:
`CONDITION_METADATA.category()` now returns `MetadataCondition`,
`is_metadata_header("MetadataCondition_Media")` is `True`, and
`ensure_metadata_prefix` no longer double-prefixes. The spec now asserts three
things that are false, in a document whose whole value is that its claims were
checked.

**But the data outlives the fix.** The validated run's mirror still contains
`Metadata_MetadataCondition_CarbonSource` and its four siblings, written before
the fix. The monitor must still tolerate that shape, so the spec's test for it
survives — with a different justification.

**Files:**
- Modify: `docs/superpowers/specs/2026-08-13-streamlit-run-monitor-design.md`

- [ ] **Step 1: Find every affected passage**

```bash
grep -n "MetadatasCondition\|MetadataCondition" docs/superpowers/specs/2026-08-13-streamlit-run-monitor-design.md
```

- [ ] **Step 2: Correct the claims**

Three changes, each keeping the evidence and dropping the falsehood:

1. §3.2's warning that a hand-typed `MetadataCondition_Media` "matches nothing"
   and gets double-prefixed — **now false**. Replace with a note that the typo
   was fixed at `86d341b2f`, and that the discovery rule (never assume a column
   name) is what made the design immune either way.
2. §3.3.0.1's "the typo is present in production data" — **still true of the
   data, no longer true of the code**. Reword to: data written before the fix
   carries the double-prefixed columns, which is why the monitor reads column
   names from the frame rather than constructing them.
3. §1's "Out of scope: fixing the `MetadatasCondition` typo" and §15's
   "Explicitly not changed" — remove; there is nothing left to fix.

Keep test 24 (double-prefixed columns are usable) and restate its justification:
it guards against old runs, not against a live bug.

- [ ] **Step 3: Verify no stale claim survives**

```bash
uv run python -c "
import re
t = open('docs/superpowers/specs/2026-08-13-streamlit-run-monitor-design.md').read()
n = re.sub(r'\s+', ' ', t)
for phrase in ['stray s', 'double-prefix', 'MetadatasCondition']:
    print(f'{phrase!r}:', n.count(phrase))
"
```

Every surviving mention must be historical (describing pre-fix data), not a
claim about current code.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-08-13-streamlit-run-monitor-design.md
git commit -m "docs(monitor): the MetadatasCondition typo was fixed upstream

Fixed by the origin/main merge at 86d341b2f. The spec asserted three
things that are now false. Data written before the fix still carries the
double-prefixed columns, so the monitor must still tolerate that shape --
test 24 survives, with staleness rather than a live bug as its reason."
```

---

## Execution: dependency DAG and clustering

Derived from each task's `Files`/`Interfaces` block (see
`orchestration-clustering`). Recorded here because it is a version-controlled
view of the plan, not separate state.

```
T1 scanner filter   ──┐
T2 finalize flush   ──┼──  no edges between tasks; no shared files
T3 spec correction  ──┘
```

| Task | Files | Shape | Executor | Model / effort |
|---|---|---|---|---|
| T2 | `_cli_checkpoint_handler.py` + its test | **Seam** — one risky wiring point between chunk-writing and aggregation, unverifiable without SLURM | subagent | frontier, high |
| T1 | `_cli_directory_scanner.py` + its test | **Leaf** — one file, four mechanical sites, one trivial predicate | subagent | mid-tier, medium |
| T3 | the monitor spec | **Leaf** — docs only | orchestrator | — |

**Why T3 is not dispatched.** It is three surgical corrections to a 2200-line
document whose context the orchestrator already holds. Dispatching would reload
that context into a fresh agent to save nothing — the "repeated context loads"
failure the clustering skill names.

**Why sequential, not parallel.** T1 and T2 have zero file overlap and would
otherwise be parallel-worktree candidates. They share one git index in this
worktree, and two agents running `git add`/`git commit` concurrently can
interleave. The wall-clock saving on two small fixes does not justify that risk
on a defect branch.

**Order:** T2 → gate → T1 → gate → T3 → deep review over the combined diff →
simplify pass → regression run over `tests/unit/cli`.

**Gates.** After each cluster: read the diff, run its test plus
`uv run pytest tests/unit/cli -q`, confirm against the 505-passed/1-skipped
baseline. After all three: a fresh code-review agent over the combined diff at
frontier tier — never weaker than the implementer.

---

## Self-Review

**Spec coverage.** No spec — the source is two defect reports plus one
now-obsolete chip. Task 1 covers the AppleDouble miscount with its four scan
sites. Task 2 covers the trailing-image loss at the finalize path. Task 3 closes
the documentation debt the upstream fix created. The third chip needs no code.

**Placeholders.** One deliberate: Step 6 of Task 2 says "paste the Step 3
assertion output here". That is a real instruction with a real artifact, not a
TBD — the commit must carry the observed failure, per the repo's test-integrity
rule. Task 2 Step 1 asks the implementer to confirm two signatures before
calling them, because the flush entry point is the one thing this plan could not
verify without running a chunked SLURM job.

**Type consistency.** `_is_image_file(path, valid_exts)` is defined in Task 1
Step 3 and used in Step 4 with the same argument order. Task 2 introduces no new
symbol. `aggregate_chunks(output_dir, dataset_names)` is used identically in the
test and the fix, and Step 1 requires confirming that signature first.

**Known risk.** Task 2's test simulates the checkpoint boundary by calling
`aggregate_chunks` mid-stream rather than running SLURM. If the real loss
mechanism differs from this simulation, Step 3 will pass instead of failing —
the plan says to stop and report rather than adjust the test, because a test
that cannot fail is worse than no test.
