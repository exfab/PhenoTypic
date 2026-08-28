# Phase 4 — Scanner and validation

The last half of the loop, then the executable check that keeps the spec honest.
**Task 10 depends on Task 6**; **Task 11 depends on Tasks 7 and 10.**

Read [`README.md`](README.md)'s **Global Constraints** first.

---

### Task 10: the input scanner learns stores

Two distinct traps here, and both produce a *plausible* wrong result rather than
an error.

**Trap A — a store is a directory, so it looks like a dataset.**
`scan_directory_structure` scans one level of subdirectories and treats each as
a dataset (`_cli_directory_scanner.py:104-118`). A `<stem>.ome.zarr` directory
would be enumerated as a dataset name, and its contents (`zarr.json`, `OME/`,
`rgb/`) contain no files matching an image extension — so it silently
contributes nothing and the images vanish from the run.

**Trap B — recursion.** The 2026-08-18 spec records this against a sibling site
(§4.4): *"a naive port recurses INTO every store: 400k stat calls at 10k
images."* An `rglob`-based port produces the **same file list** and differs only
in cost, so a correctness test cannot catch it. Assert on stat count.

**Files:**
- Modify: `src/phenotypic/_cli/_cli_directory_scanner.py:23-39` (`_is_image_file`),
  `:88-118` (root and subdirectory collection)
- Test: `tests/unit/cli/test_scanner_stores.py` (create)

**Interfaces:**
- Consumes: `ngff_.STORE_SUFFIX`.
- Produces: `_is_image_input(path, valid_exts) -> bool` replaces
  `_is_image_file`; `_is_store_dir(path) -> bool` is a new module-level helper.
  `scan_directory_structure` and `collect_image_paths` keep their signatures.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/cli/test_scanner_stores.py`:

```python
"""A tree of .ome.zarr stores is valid input, and scanning it stays cheap."""

from __future__ import annotations

from pathlib import Path

import pytest

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_ import ngff_
from phenotypic._cli._cli_directory_scanner import (
    collect_image_paths,
    scan_directory_structure,
)


def _store(parent: Path, stem: str) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    img = Image(load_synth_yeast_plate())
    return img._save_store(
        parent / f"{stem}{ngff_.STORE_SUFFIX}",
        series=("gray",),
        write_objmap=False,
        levels=ngff_.pyramid_level_count(*img.gray[:].shape[:2]),
        work_id=None,
        durable=False,
        write_image_class=False,
    )


def test_a_flat_tree_of_stores_scans_as_one_dataset(tmp_path: Path) -> None:
    root = tmp_path / "corrected"
    for stem in ("p01", "p02", "p03"):
        _store(root, stem)
    datasets = scan_directory_structure(root)
    assert list(datasets) == ["corrected"]
    assert [p.name for p in datasets["corrected"]] == [
        f"p0{i}{ngff_.STORE_SUFFIX}" for i in (1, 2, 3)
    ]


def test_a_store_is_never_mistaken_for_a_dataset(tmp_path: Path) -> None:
    """Trap A. A store is a directory; it must not become a dataset name."""
    root = tmp_path / "corrected"
    _store(root, "p01")
    datasets = scan_directory_structure(root)
    assert f"p01{ngff_.STORE_SUFFIX}" not in datasets


def test_nested_datasets_of_stores_scan_per_subdirectory(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _store(root / "plateA", "p01")
    _store(root / "plateB", "p02")
    datasets = scan_directory_structure(root)
    assert sorted(datasets) == ["plateA", "plateB"]


def test_a_mixed_tree_of_files_and_stores_is_the_union(tmp_path: Path) -> None:
    root = tmp_path / "mixed"
    _store(root, "p01")
    Image(load_synth_yeast_plate()).rgb.imsave(filepath=root / "p02.tiff")
    names = {p.name for p in collect_image_paths(root)}
    assert names == {f"p01{ngff_.STORE_SUFFIX}", "p02.tiff"}


def test_a_single_store_path_scans_as_a_single_image(tmp_path: Path) -> None:
    store = _store(tmp_path / "corrected", "p01")
    assert scan_directory_structure(store) == {"single_image": [store]}


def test_scanning_does_not_descend_into_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trap B. An rglob port yields the SAME list and only differs in cost.

    A store holds ~10 files; touching them turns a 3-image scan into ~30
    extra stats, and 10k images into 400k. Count iterdir calls instead of
    comparing output.
    """
    root = tmp_path / "corrected"
    stores = [_store(root, f"p0{i}") for i in (1, 2, 3)]

    visited: list[Path] = []
    real_iterdir = Path.iterdir

    def _counting_iterdir(self: Path):
        visited.append(self)
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", _counting_iterdir)
    scan_directory_structure(root)

    for store in stores:
        assert not any(
            store == seen or store in seen.parents for seen in visited
        ), f"scanner descended into {store}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/cli/test_scanner_stores.py -v
```

Expected: `ValueError: No valid images found in …` — a store is neither a file
with a known suffix nor a subdirectory containing one.

- [ ] **Step 3: Add the store predicates**

In `src/phenotypic/_cli/_cli_directory_scanner.py`, replace `_is_image_file`:

```python
def _is_store_dir(path: Path) -> bool:
    """True if ``path`` is an OME-Zarr store directory.

    Tested by name, not by opening the store: the scanner runs over every entry
    of every candidate directory, and reading a root ``zarr.json`` per entry
    would cost an open per file at 10k-image scale. A directory named
    ``*.ome.zarr`` that is not a store fails later, loudly, in ``imread``.
    """
    from phenotypic.sdk_.ngff_ import STORE_SUFFIX

    return (
        path.is_dir()
        and not path.name.startswith(".")
        and path.name.endswith(STORE_SUFFIX)
    )


def _is_image_input(path: Path, valid_exts: set[str]) -> bool:
    """True if ``path`` is a real input image -- a flat file or a store.

    Dotfiles are excluded, which an extension-only test does not do. macOS
    writes an AppleDouble ``._<name>`` sidecar beside every file on exFAT/FAT
    volumes -- the usual format for an external drive -- and
    ``Path("._x.tif").suffix`` is ``".tif"``, so each image would be counted
    twice. Observed on a real run: ``manifest.json`` reported
    ``total_images: 60`` for 30 images and ``is_complete: false`` on a run that
    had finished, which anything gating on completion reads as still running.

    A ``*.ome.zarr`` store counts as one input even though it is a directory.
    Its contents are never enumerated: see :func:`_is_store_dir`.
    """
    if _is_store_dir(path):
        return True
    return (
        path.is_file()
        and not path.name.startswith(".")
        and path.suffix.lower() in valid_exts
    )
```

Keep `_is_image_file` as a thin alias if anything else imports it — run
`grep -rn "_is_image_file" src/ tests/` and only remove it if nothing does.

- [ ] **Step 4: Teach the directory walk about stores**

Three edits in `scan_directory_structure`:

*(a)* Single-path case (`:83-90`) — accept a store path:

```python
    if input_path.is_file() or _is_store_dir(input_path):
        if _is_image_input(input_path, valid_exts):
            datasets["single_image"] = [input_path]
            return datasets
        raise ValueError(
            f"File {input_path.name} is not a supported image format. "
            f"Supported: {', '.join(sorted(valid_exts))}, "
            f"or an *.ome.zarr store"
        )
```

*(b)* Root collection (`:99-102`) — swap the predicate:

```python
    root_images = [
        p for p in input_path.iterdir()
        if _is_image_input(p, valid_exts)
    ]
```

*(c)* Subdirectory loop (`:105-118`) — **skip stores**, or every store becomes
a dataset name and its images vanish:

```python
    subdatasets = {}
    for subdir in input_path.iterdir():
        # A store IS a directory. Without this it is enumerated as a dataset,
        # finds no image files inside itself, and silently contributes nothing.
        if not subdir.is_dir() or _is_store_dir(subdir):
            continue

        sub_images = [
            p for p in subdir.iterdir()
            if _is_image_input(p, valid_exts)
        ]
        if sub_images:
            subdatasets[subdir.name] = sorted(sub_images)
```

Apply the same predicate swap in `get_input_structure_summary` (`:304`,
`:309`, `:329`) so the dry-run summary and the real scan cannot disagree.

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/unit/cli/test_scanner_stores.py -v
```

Expected: PASS (6 tests).

- [ ] **Step 6: Run the full CLI + scanner suite**

```bash
uv run pytest tests/unit/cli/ -q
uv run mypy src/phenotypic/_cli/_cli_directory_scanner.py
uv run ruff check --fix src/phenotypic/_cli/_cli_directory_scanner.py \
    tests/unit/cli/test_scanner_stores.py
```

Expected: PASS. The mixed-structure guard (root images AND subdirectory
datasets) must still fire — a tree of stores at the root is *flat*, not mixed.

- [ ] **Step 7: Confirm work-ID derivation handles a store**

Work IDs come from the input path relative to `--input` (spec 7.3). A store's
relative path is `<stem>.ome.zarr`, and `Path.stem` on that is `<stem>.ome` --
"a plausible-looking wrong name rather than an error", in `store_stem`'s own
words, which propagates into parquet filenames and completion markers so every
image reprocesses forever.

Find the derivation:

```bash
grep -rn "_worker_work_identity\|def .*work_id" src/phenotypic/_cli/ | head
```

Read it. If it reaches for `Path.stem` or `.suffix` on the input path, add a
store branch using `sdk_.store_stem`. If it uses the full relative path
verbatim, no change is needed -- `p01.ome.zarr` is a perfectly good stable key.
Add a test either way:

```python
def test_a_store_and_a_tiff_get_different_work_ids(tmp_path: Path) -> None:
    """They are two distinct inputs, exactly as p01.tiff and p01.png are."""
    # Build both under one input root, derive both work ids, assert !=.
    # Use the real derivation function found by the grep above.
```

Then re-run `uv run pytest tests/unit/cli/ -q`.

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/_cli/_cli_directory_scanner.py \
        tests/unit/cli/test_scanner_stores.py
git commit -m "feat(cli): the input scanner accepts .ome.zarr stores

_is_image_input replaces _is_image_file and counts a *.ome.zarr directory as
one input; the subdirectory walk skips stores, which would otherwise each be
enumerated as a dataset, find no image files inside themselves, and silently
drop every image. Store contents are never enumerated -- the 2026-08-18 spec
records the cost of getting that wrong at a sibling site: 400k stat calls at
10k images. A test asserts non-recursion by counting iterdir calls, because an
rglob port yields the same file list and differs only in cost."
```

---

### Task 11: logic-validation script and documentation

`CLAUDE.md` requires an executable check for any design resting on a numeric
invariant a reader would otherwise take on faith. The load-bearing numbers here
are the 12-file store, the pyramid geometry, and the sharding multiple.

**Files:**
- Create: `docs/superpowers/logic_validation_scripts/2026-08-27-process-mode-ome-zarr/store_geometry.py`
- Modify: `CLAUDE.md` (the `--mode process` bullet under **CLI**)
- Modify: `src/phenotypic/_cli/_cli_readme_generator.py`

**Interfaces:** none consumed or produced; this task is documentation and a
standalone check.

- [ ] **Step 1: Write the validation script**

Create the file. It must import **no** `phenotypic` code — it re-derives the
claims independently, which is the whole point.

```python
#!/usr/bin/env python3
"""Re-derive the store-geometry claims of the process-mode OME-Zarr design.

Spec: docs/superpowers/specs/2026-08-27-process-mode-ome-zarr/design.md

Imports nothing from ``phenotypic``: every number is derived from the NGFF 0.5
and Zarr v3 rules directly, so the script fails if the spec and the format
disagree -- not merely if the spec and the implementation disagree.

Exits non-zero on failure.
"""

from __future__ import annotations

import math
import sys

PYRAMID_STOP_PX = 512
CHUNK_EDGE = 1024
SHARD_EDGE = 4096
FAILURES: list[str] = []


def check(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        FAILURES.append(f"{label}: expected {expected!r}, got {actual!r}")


def level_count(height: int, width: int, stop_px: int = PYRAMID_STOP_PX) -> int:
    """Halve until the longest edge is <= stop_px. Ceil, not floor."""
    longest = max(height, width)
    if longest <= stop_px:
        return 1
    return int(math.ceil(math.log2(longest / stop_px))) + 1


def level_shapes(height: int, width: int) -> list[tuple[int, int]]:
    shapes = []
    h, w = height, width
    for _ in range(level_count(height, width)):
        shapes.append((h, w))
        h, w = max(1, h // 2), max(1, w // 2)
    return shapes


def shards_per_level(h: int, w: int, channels: int) -> int:
    """One shard file per shard-sized block. A shard spans the whole c axis."""
    chunk_h = min(CHUNK_EDGE, h)
    chunk_w = min(CHUNK_EDGE, w)
    shard_h = min(SHARD_EDGE, h) if h > chunk_h else chunk_h
    shard_w = min(SHARD_EDGE, w) if w > chunk_w else chunk_w
    # A shard must be an exact multiple of the chunk in every dimension.
    if shard_h % chunk_h or shard_w % chunk_w or shard_h < chunk_h:
        FAILURES.append(
            f"shard {(channels, shard_h, shard_w)} is not a multiple of "
            f"chunk {(1, chunk_h, chunk_w)}"
        )
    return math.ceil(h / shard_h) * math.ceil(w / shard_w)


def single_series_file_count(height: int, width: int, channels: int) -> int:
    """Files in a single-series store: 4 fixed + 2 per pyramid level."""
    shapes = level_shapes(height, width)
    data = sum(shards_per_level(h, w, channels) for h, w in shapes)
    metadata = (
        1                # root zarr.json
        + 1              # OME/zarr.json
        + 1              # OME/METADATA.ome.xml
        + 1              # <series>/zarr.json
        + len(shapes)    # <series>/<level>/zarr.json
    )
    return data + metadata


def main() -> int:
    # Spec 1.1: the measured geometry of a 4000x3000 rgb store.
    check("levels(4000, 3000)", level_count(4000, 3000), 4)
    check(
        "level shapes(4000, 3000)",
        level_shapes(4000, 3000),
        [(4000, 3000), (2000, 1500), (1000, 750), (500, 375)],
    )
    check(
        "single-series rgb file count at 4000x3000",
        single_series_file_count(4000, 3000, channels=3),
        12,
    )

    # Spec 1.4 (inherited): the sharding codec requires an exact multiple in
    # EVERY dimension, the channel axis included.
    check("shard spans the channel axis: 3 % 1", 3 % 1, 0)
    check("shard/chunk edge ratio", SHARD_EDGE % CHUNK_EDGE, 0)

    # A level at or below the stop threshold is the last one.
    check("levels(512, 512)", level_count(512, 512), 1)
    check("levels(513, 400)", level_count(513, 400), 2)

    # Ceil, not floor: a floor formula stops a level early and leaves
    # 4000x3000's smallest level at 1000x750 rather than 500x375.
    floor_levels = int(math.floor(math.log2(4000 / PYRAMID_STOP_PX))) + 1
    check("floor formula is refuted", floor_levels == 4, False)

    for failure in FAILURES:
        print(f"FAIL: {failure}", file=sys.stderr)
    if FAILURES:
        return 1
    print("All store-geometry claims verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the script**

```bash
uv run python docs/superpowers/logic_validation_scripts/2026-08-27-process-mode-ome-zarr/store_geometry.py
echo "exit=$?"
```

Expected: `All store-geometry claims verified.` and `exit=0`.

If the file count comes out other than 12, do **not** edit the expected value.
Compare against the real store instead:

```bash
uv run python -c "
from pathlib import Path
import tempfile, numpy as np
from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_ import ngff_
img = Image(load_synth_yeast_plate())
tmp = Path(tempfile.mkdtemp())/'p.ome.zarr'
img._save_store(tmp, series=('rgb',), write_objmap=False,
                levels=ngff_.pyramid_level_count(*img.rgb[:].shape[:2]),
                work_id=None, durable=False, write_image_class=False)
print(sorted(p.relative_to(tmp).as_posix() for p in tmp.rglob('*') if p.is_file()))
"
```

The synthetic plate is smaller than 4000x3000, so its level count differs;
reconcile the *formula*, not the constant.

- [ ] **Step 3: Update `CLAUDE.md`**

Replace the `--mode process` bullet under **CLI** with:

```markdown
- `uv run python -m phenotypic --mode process --layer {rgb|gray|detect_mat|objmap}`
  — apply-only export: runs `pipeline.apply()` and writes ONE image layer per
  input, mirroring the input tree. **Output is a single-series OME-Zarr store**
  (`<stem>.ome.zarr/`) for `rgb`/`gray`/`detect_mat`, and a 16-bit raw-label PNG
  for `objmap`; `--process-format {tiff,zarr}` overrides, and
  `--layer objmap --process-format zarr` is refused (NGFF has no standalone
  label-image form). The store carries the pipeline that produced it in
  `attributes.phenotypic.provenance` and omits `image_class`, so
  `Image.load_zarr` refuses it and points at `Image.imread`, which reads any
  OME-Zarr — PhenoTypic's or a third party's — as plain pixels. A tree of stores
  is valid `--input`. Skips measurement/deliverables/QC/dashboard; machine state
  lives under `.phenotypic/`. Full local + SLURM continuation reuse. Run the
  same command again after an interruption or when new compatible inputs appear;
  there is no `--resume` flag.
```

Add to **Gotchas**:

```markdown
- **`imread` vs `load_zarr` on an OME-Zarr store:** the verb decides, never the
  file. `Image.imread(store)` always reads plain pixels — PhenoTypic's own
  output, or a napari/QuPath/`bioformats2raw` export — and refuses rather than
  guessing when a store cannot be projected onto a 2-D image (a real `t` or `z`
  axis, a channel count that is neither 1 nor 3, an HCS plate); pass
  `t=`/`z=`/`c=`/`series=` to choose explicitly. `Image.load_zarr(store)`
  always restores run state and raises on a store with no
  `phenotypic.image_class`. NGFF has no RGB type: `rgb` is a 3-length `channel`
  axis ordered **before** the space axes, so stores are planar `(3,H,W)` and
  `imread` transposes to `(H,W,3)`.
```

- [ ] **Step 4: Update the run README generator**

In `src/phenotypic/_cli/_cli_readme_generator.py`, find where the process-mode
output layout is documented and update it to describe `<stem>.ome.zarr/`
alongside the flat-file form. Read the surrounding generator text first and
match its voice; do not append a new section if an existing one covers output
layout.

- [ ] **Step 5: Run the docs gates**

```bash
uv run pytest tests/unit/test_docs_staged_cli.py -q
uv run pytest tests/unit/cli/ tests/unit/sdk_/ -q
```

Expected: PASS. `test_docs_staged_cli.py` checks documented CLI flags against
the real click options, so a `--process-format` help-text mismatch fails here.

- [ ] **Step 6: Full regression run**

```bash
uv run pytest tests/unit -q -n auto
```

Expected: PASS. On the HPCC, `-n auto` oversubscribes (node cores exceed the
Slurm allocation); if the run thrashes, use
`-n "$(python -c 'import os; print(len(os.sched_getaffinity(0)))')"`.

- [ ] **Step 7: Commit**

```bash
git add docs/superpowers/logic_validation_scripts/2026-08-27-process-mode-ome-zarr/store_geometry.py \
        CLAUDE.md src/phenotypic/_cli/_cli_readme_generator.py
git commit -m "docs: validate store geometry and document process-mode zarr

store_geometry.py re-derives the 12-file single-series count, the pyramid
level count and shapes, and the shard/chunk multiple from the NGFF and Zarr
rules directly, importing no phenotypic code -- so it fails if the spec and the
format disagree, not merely if the spec and the implementation do. It also
asserts the floor-based level formula is refuted, which is the mistake the
ceil is there to prevent.

CLAUDE.md documents the new default output, --process-format, and the
imread-vs-load_zarr contract including NGFF's lack of an RGB type."
```

---

## Phase 4 exit criteria

- [ ] `uv run pytest tests/unit -q` passes.
- [ ] `uv run python docs/superpowers/logic_validation_scripts/2026-08-27-process-mode-ome-zarr/store_geometry.py` exits 0.
- [ ] `uv run mypy src/phenotypic` reports no new errors.
- [ ] A manual end-to-end check of the loop:

```bash
uv run python -m phenotypic --mode process --layer rgb \
    --input <tree of tiffs> --output /tmp/acr-out --pipeline <pipeline.json>
uv run python -m phenotypic --mode process --layer gray \
    --input /tmp/acr-out --output /tmp/acr-out2 --pipeline <pipeline.json>
```

The second command consuming the first's output *is* the deliverable. Confirm
the first run's stores open in napari (`napari /tmp/acr-out/<stem>.ome.zarr`) if
it is available.
