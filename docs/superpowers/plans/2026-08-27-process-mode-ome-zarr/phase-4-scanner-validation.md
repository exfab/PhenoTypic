# Phase 4 — Scanner and validation

The last half of the loop, then the executable check that keeps the spec honest.
**Task 10a is a precondition for 10b**; **Task 10b depends on Task 6**;
**Task 11 depends on Tasks 7, 8b, and 10b.**

Read [`README.md`](README.md)'s **Global Constraints** first.

---

### Task 10a: store-aware input digest and work identity

**Without this, every scanner-fed run dies before it starts.** `file_sha256`
(`_cli_failure_tracker.py:92-98`) opens its argument:

```python
    with path.open("rb") as handle:
```

and it is called with the **input image path** from three places:

| Call site | When |
|---|---|
| `_cli_failure_tracker.py:205` (`work_id_for_image`) | every image, local and SLURM |
| `_cli_slurm_array_scripts.py:381` | once per image at submit time, into the identity ledger |
| `_cli_process_single.py:141` (`_worker_work_identity`) | every worker |
| `_cli_process_single.py:624` | the immutable-identity re-check |

A `*.ome.zarr` input is a **directory**, so all four raise `IsADirectoryError`.
That breaks the whole of spec §7 and this design's own end-to-end criterion, and
it does so before Task 10b's scanner change has any chance to help — which is
why this lands first.

**The store branch digests the root `zarr.json`, not the tree.** That file is
already the store's completeness fingerprint: the promote protocol writes it
last (`_image_io_handler.py:1168-1183`), so it exists only on a fully written
store, and it changes whenever any published content does — the series map, the
pyramid level count, the metadata sections, the provenance journal. Digesting
the whole tree would be correct too, but it costs a directory walk per image at
submit time for no additional guarantee.

**Files:**
- Modify: `src/phenotypic/_cli/_cli_failure_tracker.py:92-98` (`file_sha256`)
  and `:182-211` (`work_id_for_image`)
- Modify: `src/phenotypic/_cli/_cli_process_only.py`
  (`process_only_output_path`'s degenerate-relative-path fallback)
- Test: `tests/unit/cli/test_store_work_identity.py` (create)

**Interfaces:**
- Consumes: `sdk_.STORE_SUFFIX`, `sdk_.STORE_ROOT_JSON` (both re-exported from
  `phenotypic.sdk_`; `sdk_/__init__.py:287,357`).
- Produces: `file_sha256` accepts a store directory. Signature unchanged, so no
  call site needs editing.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/cli/test_store_work_identity.py`:

```python
"""A store is a directory. Work-ID derivation has to survive that."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_ import ngff_
from phenotypic._cli._cli_failure_tracker import file_sha256
from phenotypic._cli._cli_process_only import process_only_output_path


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


def test_a_store_digests_without_raising(tmp_path: Path) -> None:
    """Today: IsADirectoryError, from `with path.open("rb")`."""
    store = _store(tmp_path / "in", "p01")
    assert len(file_sha256(store)) == 64


def test_the_digest_is_the_root_zarr_json(tmp_path: Path) -> None:
    """Named explicitly, so a future change to the tree walk is deliberate."""
    store = _store(tmp_path / "in", "p01")
    expected = hashlib.sha256(
        (store / "zarr.json").read_bytes()
    ).hexdigest()
    assert file_sha256(store) == expected


def test_the_digest_changes_when_the_store_content_does(tmp_path: Path) -> None:
    """The root records the series map, pyramid, metadata, and provenance."""
    a = _store(tmp_path / "a", "p01")
    root = a / "zarr.json"
    before = file_sha256(a)
    root.write_text(
        root.read_text(encoding="utf-8").replace('"gray"', '"grey"'),
        encoding="utf-8",
    )
    assert file_sha256(a) != before


def test_a_flat_file_digest_is_untouched(tmp_path: Path) -> None:
    """The whole-file streaming read is the path 99% of inputs still take."""
    target = tmp_path / "p01.tiff"
    target.write_bytes(b"not really a tiff, but it is a file")
    assert file_sha256(target) == hashlib.sha256(target.read_bytes()).hexdigest()


def test_a_plain_directory_is_still_refused(tmp_path: Path) -> None:
    """A directory that is not a store has no fingerprint. Say so."""
    plain = tmp_path / "just_a_folder"
    plain.mkdir()
    with pytest.raises(IsADirectoryError):
        file_sha256(plain)


def test_a_store_named_input_mirrors_to_a_named_output(tmp_path: Path) -> None:
    """`--input <one store>` must not write `<out>/.ome.zarr`.

    Pre-existing: `image_path.relative_to(input_root)` is `Path(".")` when the
    two are the same path, and `Path(".").stem` is `""`. Verified today on the
    flat-file equivalent -- `--input <one tiff>` writes `<out>/.tiff` -- so
    this is not a regression the store introduces, but the store case is the
    one this design makes routine.
    """
    store = _store(tmp_path / "in", "p01")
    assert process_only_output_path(
        tmp_path / "out", store, store, "rgb", fmt="zarr",
    ).name == f"p01{ngff_.STORE_SUFFIX}"
    single_file = tmp_path / "in" / "p02.tiff"
    Image(load_synth_yeast_plate()).rgb.imsave(filepath=single_file)
    assert process_only_output_path(
        tmp_path / "out", single_file, single_file, "rgb", fmt="tiff",
    ).name == "p02.tiff"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/cli/test_store_work_identity.py -v
```

Expected: `IsADirectoryError` on the first three; the flat-file test PASSES
already; `test_a_plain_directory_is_still_refused` also passes already, by
accident — `open("rb")` on a directory raises exactly that — and stays as the
regression pin that the new branch does not swallow it.

- [ ] **Step 3: Add the store branch to `file_sha256`**

```python
def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of *path* without retaining file contents.

    A ``*.ome.zarr`` input is a **directory**, so the streaming read raises
    ``IsADirectoryError``. A store is digested by its root ``zarr.json``
    instead, which is already its completeness fingerprint: ``promote_store``
    writes that document last, so it exists only on a fully written store, and
    it changes whenever any published content does -- the series map, the
    pyramid level count, the metadata sections, the provenance journal.

    Digesting the whole tree would be correct too, and is not done: it costs a
    directory walk per image at SLURM submit time
    (``_cli_slurm_array_scripts.py:381`` runs this once per image while
    building the identity ledger) for no additional guarantee.

    A directory that is not a store still raises ``IsADirectoryError``. It has
    no meaningful content fingerprint, and inventing one would let a
    mis-specified ``--input`` produce a stable work ID for something that is
    not an image.

    Args:
        path: An input image file, or a ``*.ome.zarr`` store directory.

    Returns:
        The hex digest.

    Raises:
        IsADirectoryError: If *path* is a directory that is not an OME-Zarr
            store.
    """
    from phenotypic.sdk_ import STORE_ROOT_JSON, STORE_SUFFIX

    target = Path(path)
    if target.is_dir():
        if not target.name.endswith(STORE_SUFFIX):
            raise IsADirectoryError(
                f"{target} is a directory but not an OME-Zarr store; "
                f"it has no content fingerprint"
            )
        target = target / STORE_ROOT_JSON

    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

The import is function-local: `_cli_failure_tracker.py` already imports four
names from `phenotypic.sdk_` at module scope (`:22-27`), and adding two more
there is equally fine — pick whichever the surrounding code makes more natural,
but do not import `ngff_` (this needs two constants, not the toolkit).

No call site changes. That is the point of putting the branch here rather than
at each of the four call sites: a fifth call site added later gets the
behaviour for free, where a per-site branch would silently miss it.

- [ ] **Step 4: Fix the degenerate relative path**

In `process_only_output_path` (Task 7 rewrote this function), the
`relative_to` block:

```python
    try:
        rel = image_path.relative_to(input_root)
    except ValueError:
        rel = Path(image_path.name)
    if rel == Path("."):
        # `--input` names the image itself, so `relative_to` yields `.` and
        # `Path(".").stem` is `""` -- the run would write `<out>/.ome.zarr`.
        # Pre-existing on the flat-file path (`--input <one tiff>` writes
        # `<out>/.tiff` today, verified); fixed here because a single store
        # input is exactly what spec 7 makes routine.
        rel = Path(image_path.name)
```

placed immediately before the `store_stem` line, so the store-suffix test runs
against the recovered name.

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/unit/cli/test_store_work_identity.py -v
uv run pytest tests/unit/cli/ -q
uv run mypy src/phenotypic/_cli/_cli_failure_tracker.py \
            src/phenotypic/_cli/_cli_process_only.py
uv run ruff check --fix src/phenotypic/_cli/_cli_failure_tracker.py \
    src/phenotypic/_cli/_cli_process_only.py \
    tests/unit/cli/test_store_work_identity.py
```

Expected: PASS (6 tests in the new file).

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_cli/_cli_failure_tracker.py \
        src/phenotypic/_cli/_cli_process_only.py \
        tests/unit/cli/test_store_work_identity.py
git commit -m "fix(cli): derive a work identity from a store input

file_sha256 opens its argument, and a *.ome.zarr input is a directory, so all
four call sites -- work_id_for_image, the SLURM identity ledger, and the
worker's two -- raised IsADirectoryError. A store is digested by its root
zarr.json, which is already its completeness fingerprint: promote_store writes
it last, so it exists only on a fully written store, and it changes whenever any
published content does. Digesting the whole tree costs a walk per image at
submit time for no additional guarantee. A directory that is not a store still
raises, because it has no fingerprint to invent.

The branch lives in file_sha256 rather than at each call site, so a fifth
caller gets the behaviour instead of silently missing it.

process_only_output_path also recovers from relative_to yielding Path('.'),
which happens when --input names the image itself. That is pre-existing --
--input <one tiff> --mode process writes <out>/.tiff today -- but a single
store input is what this design makes routine."
```

---

### Task 10b: the input scanner learns stores

Two distinct traps here, and both produce a *plausible* wrong result rather than
an error.

**Trap A — a store is a directory, so it looks like a dataset.**
`scan_directory_structure` scans one level of subdirectories and treats each as
a dataset (`_cli_directory_scanner.py:104-117`). A `<stem>.ome.zarr` directory
would be enumerated as a dataset name, and its contents (`zarr.json`, `OME/`,
`rgb/`) contain no files matching an image extension — so it silently
contributes nothing and the images vanish from the run.

**Trap B — recursion.** The 2026-08-18 spec records this against a sibling site
(§4.4), and so does the sibling itself: `scan_store_outputs`'s docstring
(`_cli_directory_scanner.py:192-195`) says *"a store IS a directory full of
files, so `rglob` would descend into every one of them — roughly forty stat
calls per store, 400k at 10k images."* That function is the precedent to copy:
its glob is non-recursive and matches directories (`:230-234`). An `rglob`-based
port produces the **same file list** and differs only in cost, so a correctness
test cannot catch it. Assert on stat count.

**`STORE_SUFFIX` is already imported** at `_cli_directory_scanner.py:14-19`,
from `phenotypic.sdk_`. Do not add a function-local re-import.

**Files:**
- Modify: `src/phenotypic/_cli/_cli_directory_scanner.py:23-38` (`_is_image_file`),
  `:84-92` (single-path case), `:99-102` (root collection), `:104-117`
  (subdirectory loop), `:308-310` (`get_input_structure_summary`'s single-path
  guard), `:321-324` (its root count), `:328-335` (its subdirectory loop)
- Test: `tests/unit/cli/test_scanner_stores.py` (create)

**Interfaces:**
- Consumes: `STORE_SUFFIX` (already imported).
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
    get_input_structure_summary,
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
    """Spec 7.4: no ordering or precedence rule, just the union."""
    root = tmp_path / "mixed"
    _store(root, "p01")
    Image(load_synth_yeast_plate()).rgb.imsave(filepath=root / "p02.tiff")
    names = {p.name for p in collect_image_paths(root)}
    assert names == {f"p01{ngff_.STORE_SUFFIX}", "p02.tiff"}


def test_a_single_store_path_scans_as_a_single_image(tmp_path: Path) -> None:
    store = _store(tmp_path / "corrected", "p01")
    assert scan_directory_structure(store) == {"single_image": [store]}


def test_the_dry_run_summary_agrees_with_the_real_scan(tmp_path: Path) -> None:
    """`get_input_structure_summary` is the --dry-run path a user runs FIRST.

    It has its own copy of every predicate (`:308-310`, `:321-324`,
    `:328-335`). Leaving those unpatched gives a dry run that reports "no valid
    images found" for a tree the real run processes -- which reads as a broken
    input, not as a broken summary.
    """
    root = tmp_path / "corrected"
    for stem in ("p01", "p02"):
        _store(root, stem)
    summary = get_input_structure_summary(root)
    assert summary["total_images"] == 2
    assert summary["datasets"] == {"corrected": 2}
    assert f"p01{ngff_.STORE_SUFFIX}" not in summary["datasets"]


def test_the_dry_run_summary_accepts_a_single_store(tmp_path: Path) -> None:
    """Its single-path guard is a bare suffix check that rejects a store."""
    store = _store(tmp_path / "corrected", "p01")
    assert get_input_structure_summary(store)["total_images"] == 1


def test_scanning_does_not_descend_into_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trap B. An rglob port yields the SAME list and only differs in cost.

    A store holds ~8 files; touching them turns a 3-image scan into ~24
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

In `src/phenotypic/_cli/_cli_directory_scanner.py`, replace `_is_image_file`
(`:23-38`):

```python
def _is_store_dir(path: Path) -> bool:
    """True if ``path`` is an OME-Zarr store directory.

    Tested by **name**, not by opening the store: the scanner runs over every
    entry of every candidate directory, and reading a root ``zarr.json`` per
    entry would cost an open per file at 10k-image scale. A directory named
    ``*.ome.zarr`` that is not a store fails later, loudly, in ``imread``.

    This is the same shape as :func:`scan_store_outputs`'s glob (`:230-234`),
    which matches directories non-recursively for the same reason its docstring
    gives: a store is a directory full of files, so anything recursive costs
    roughly forty stat calls per store.
    """
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

    The same dot test also excludes ``promote_store``'s in-flight
    ``.<stem>.ome.zarr.<uuid>.part`` and ``.trash`` siblings, which is exactly
    right: neither is a readable store.

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

`STORE_SUFFIX` is already imported at `:14-19`. Do **not** add a function-local
`from phenotypic.sdk_.ngff_ import STORE_SUFFIX`.

Keep `_is_image_file` as a thin alias only if something else imports it — run
`grep -rn "_is_image_file" src/ tests/` and delete it if nothing does.

- [ ] **Step 4: Teach `scan_directory_structure` about stores**

Three edits.

*(a)* Single-path case (`:84-92`) — accept a store path:

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

*(c)* Subdirectory loop (`:104-117`) — **skip stores**, or every store becomes
a dataset name and its images vanish:

```python
    # Scan one level of subdirectories
    subdatasets = {}
    for subdir in input_path.iterdir():
        # A store IS a directory. Without this it is enumerated as a dataset,
        # finds no image files inside itself, and silently contributes
        # nothing -- so the images disappear from the run with no error.
        if not subdir.is_dir() or _is_store_dir(subdir):
            continue

        sub_images = [
            p for p in subdir.iterdir()
            if _is_image_input(p, valid_exts)
        ]

        if sub_images:
            subdatasets[subdir.name] = sorted(sub_images)
```

The mixed-structure guard at `:120-126` is unchanged and must still fire: a tree
of stores at the root is *flat*, not mixed, because (c) keeps stores out of
`subdatasets`.

- [ ] **Step 5: Teach `get_input_structure_summary` the same three things**

This is the `--dry-run` path, and a user runs it first. It has its **own copy**
of every predicate, so leaving it unpatched gives a dry run that reports "no
valid images found" for a tree the real run processes — read as a broken input,
not a broken summary.

*(a)* Single-path guard (`:308-310`) — a bare suffix check that rejects a store
outright:

```python
    # Single file, or a single store
    if input_path.is_file() or _is_store_dir(input_path):
        if not _is_image_input(input_path, valid_exts):
            raise ValueError(f"File {input_path.name} is not a supported image format")
        return {
            "type"        : "single_file",
            "total_images": 1,
            "datasets"    : {"single_image": 1}
        }
```

*(b)* Root count (`:321-324`):

```python
    root_count = sum(
            1 for p in input_path.iterdir()
            if _is_image_input(p, valid_exts)
    )
```

*(c)* Subdirectory loop (`:328-335`) — the same store skip as
`scan_directory_structure`:

```python
    subdir_counts = {}
    for subdir in input_path.iterdir():
        if not subdir.is_dir() or _is_store_dir(subdir):
            continue

        sub_count = sum(
                1 for p in subdir.iterdir()
                if _is_image_input(p, valid_exts)
        )

        if sub_count > 0:
            subdir_counts[subdir.name] = sub_count
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/unit/cli/test_scanner_stores.py -v
```

Expected: PASS (8 tests).

- [ ] **Step 7: Run the full CLI + scanner suite**

```bash
uv run pytest tests/unit/cli/ -q
uv run mypy src/phenotypic/_cli/_cli_directory_scanner.py
uv run ruff check --fix src/phenotypic/_cli/_cli_directory_scanner.py \
    tests/unit/cli/test_scanner_stores.py
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/_cli/_cli_directory_scanner.py \
        tests/unit/cli/test_scanner_stores.py
git commit -m "feat(cli): the input scanner accepts .ome.zarr stores

_is_image_input replaces _is_image_file and counts a *.ome.zarr directory as
one input; both directory walks skip stores, which would otherwise each be
enumerated as a dataset, find no image files inside themselves, and silently
drop every image. get_input_structure_summary gets the same three edits: it has
its own copy of every predicate and is the --dry-run path a user runs first, so
leaving it behind would report 'no valid images found' for a tree the real run
processes.

Store contents are never enumerated. scan_store_outputs already records the
cost of getting that wrong at a sibling site -- roughly forty stat calls per
store, 400k at 10k images -- and its non-recursive directory glob is the shape
copied here. A test asserts non-recursion by counting iterdir calls, because an
rglob port yields the same file list and differs only in cost."
```

---

### Task 11: logic-validation script and documentation

`CLAUDE.md` requires an executable check for any design resting on a numeric
invariant a reader would otherwise take on faith. The load-bearing numbers here
are the 12-file store, the pyramid geometry, and the sharding multiple.

**Files:**
- Create: `docs/superpowers/logic_validation_scripts/2026-08-27-process-mode-ome-zarr/store_geometry.py`
- Modify: `CLAUDE.md` (the `--mode process` bullet under **CLI**, and **Gotchas**)

`src/phenotypic/_cli/_cli_readme_generator.py` was listed here in an earlier
draft and is **not** affected. It documents nothing about process mode
(`grep -n process` returns only `"processing"`/`"processed"` prose and the
`processing_state.json` / `processing_events.log` layout lines), and
`phenotypicCLI.py:2324` skips `output_manager.create_structure` for process
runs, so the generator is never reached on this path.

**Interfaces:** none consumed or produced; this task is documentation and a
standalone check.

- [ ] **Step 1: Write the validation script**

Create the file. It must import **no** `phenotypic` code — it re-derives the
claims independently, which is the whole point. This exact text has been run and
exits 0; if you change a formula, re-run before committing.

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
    """Ceil-halve both spatial axes, matching NGFF's stored extents.

    ``(h + 1) // 2``, never ``h // 2``: an odd 1025-pixel axis becomes 513
    pixels, and a floor formula would silently disagree with the writer on
    every odd level.
    """
    shapes = []
    h, w = height, width
    for _ in range(level_count(height, width)):
        shapes.append((h, w))
        h, w = max(1, (h + 1) // 2), max(1, (w + 1) // 2)
    return shapes


def shards_per_level(h: int, w: int, channels: int) -> int:
    """One shard file per shard-sized block. A shard spans the whole c axis.

    The spatial shard edge is the FIXED ``SHARD_EDGE``, never clamped to the
    level extent: the Zarr v3 sharding codec constrains shard-vs-chunk
    divisibility only, never shard-vs-array, and partial edge shards are
    normal. Clamping would turn a 4000-pixel axis under a 4096 shard into four
    shard files instead of one. A level below one chunk collapses to
    ``chunk == shard == extent``, which keeps divisibility trivially true.
    """
    chunk_h = min(CHUNK_EDGE, h)
    chunk_w = min(CHUNK_EDGE, w)
    shard_h = chunk_h if h < CHUNK_EDGE else SHARD_EDGE
    shard_w = chunk_w if w < CHUNK_EDGE else SHARD_EDGE
    # A shard must be an exact multiple of the chunk in every dimension.
    if shard_h % chunk_h or shard_w % chunk_w or shard_h < chunk_h:
        FAILURES.append(
            f"shard {(channels, shard_h, shard_w)} is not a multiple of "
            f"chunk {(1, chunk_h, chunk_w)}"
        )
    return math.ceil(h / shard_h) * math.ceil(w / shard_w)


def single_series_file_count(height: int, width: int, channels: int) -> int:
    """Files in a single-series store: 4 fixed + 2 per pyramid level.

    The ``4 + 2 * levels`` shorthand holds only while every level fits inside
    ONE shard -- true up to a 4096-pixel level-0 edge, which covers every
    camera this design targets. Above that a level contributes more than one
    shard file and the shorthand understates the count; ``shards_per_level``
    is the general form and is what this function actually sums.
    """
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

    # Ceil-halving, not floor: an odd axis keeps the extra row. A floor
    # formula agrees on 4000x3000 (every level is even) and diverges the
    # moment an odd edge appears, which is why the check uses an odd one.
    check(
        "odd axes ceil-halve",
        level_shapes(4001, 3000)[:2],
        [(4001, 3000), (2001, 1500)],
    )

    # Ceil, not floor, in the LEVEL COUNT too: a floor formula stops a level
    # early and leaves 4000x3000's smallest level at 1000x750.
    floor_levels = int(math.floor(math.log2(4000 / PYRAMID_STOP_PX))) + 1
    check("floor level formula is refuted", floor_levels == 4, False)

    # The 4 + 2*levels shorthand, and the extent at which it stops holding.
    check(
        "shorthand holds at 4000x3000",
        single_series_file_count(4000, 3000, channels=3),
        4 + 2 * level_count(4000, 3000),
    )
    check("a 4096-edge level is one shard", shards_per_level(4096, 4096, 3), 1)
    check("a 4097-edge level is four shards", shards_per_level(4097, 4097, 3), 4)

    for failure in FAILURES:
        print(f"FAIL: {failure}", file=sys.stderr)
    if FAILURES:
        return 1
    print("All store-geometry claims verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Two formulas here were wrong in an earlier draft and are the reason this step
carries the full text rather than a sketch:

- `shards_per_level` clamped the shard to the level extent. `ngff_.shard_shape_for`
  does the **opposite**, and its docstring says so outright
  (`ngff_.py:354-373`): the spatial shard edge is the fixed `SHARD_YX` and is
  *not* clamped, because the sharding codec constrains shard-vs-chunk
  divisibility only. Under the clamped version a 4000-pixel axis gets a
  4000-wide shard, `4000 % 1024 != 0`, and the script reports a divisibility
  failure and exits 1.
- `level_shapes` floor-halved (`h // 2`) while `pyramid_level_shapes` ceil-halves
  (`ngff_.py:190`: `max(1, (h + 1) // 2)`). They agree at 4000x3000 and diverge
  on any odd edge, so the script would have silently stopped re-deriving the
  function it claims to re-derive.

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
import tempfile
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

The synthetic plate is 600x800, so `pyramid_level_count` is 2 and the real store
has 8 files, not 12 — verified by execution:
`['OME/METADATA.ome.xml', 'OME/zarr.json', 'rgb/0/c.0.0.0', 'rgb/0/zarr.json',
'rgb/1/c.0.0.0', 'rgb/1/zarr.json', 'rgb/zarr.json', 'zarr.json']`. Reconcile
the *formula*, not the constant: `single_series_file_count(600, 800, 3)` must
equal 8.

- [ ] **Step 3: Update `CLAUDE.md`**

Replace the `--mode process` bullet under **CLI** with:

```markdown
- `uv run python -m phenotypic --mode process --layer {rgb|gray|detect_mat|objmap}`
  — apply-only export: runs `pipeline.apply()` and writes ONE image layer per
  input, mirroring the input tree. **Output is a single-series OME-Zarr store**
  (`<stem>.ome.zarr/`) for `rgb`/`gray`, a float TIFF for `detect_mat`, and a
  16-bit raw-label PNG for `objmap`. `--process-format {tiff,zarr}` overrides;
  `--layer objmap --process-format zarr` and `--layer detect_mat
  --process-format zarr` are both refused, for different reasons — NGFF has no
  standalone label-image form, and PhenoTypic's store writer requires a primary
  series (`rgb` or `gray`). The store carries the pipeline that produced it in
  `attributes.phenotypic.provenance` and omits `image_class`, so
  `Image.load_zarr` refuses it and points at `Image.imread`, which reads any
  OME-Zarr — PhenoTypic's or a third party's — as plain pixels. A tree of stores
  is valid `--input`. Skips measurement/deliverables/QC/dashboard; machine state
  lives under `.phenotypic/`. Full local + SLURM continuation reuse; switching
  `--process-format` invalidates continuation rather than reusing outputs of the
  other kind. Run the same command again after an interruption or when new
  compatible inputs appear; there is no `--resume` flag.
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

**Two edits this must not make.** `tests/unit/test_docs_staged_cli.py` reads
this file, and its assertions are easy to break by accident:

- The staged-GPU bullet must keep mentioning `GpuDetector`, `stage`,
  `stage2_raw`, and `token` (`test_claude_md_documents_local_staged_gpu`), and
  the three `--gpu-*` flags must stay (`test_claude_md_documents_gpu_flags`).
  Leave that bullet alone.
- **Every paragraph containing the word "sidecar" must also contain
  "scheduler"** (`test_the_staged_docs_do_not_still_describe_an_objmap_sidecar`,
  `:44-46`), and the strings `"objmap sidecar"` and `"results/<dataset>/objmap/"`
  must not appear at all. Neither replacement above uses the word; do not
  introduce it.

- [ ] **Step 4: Run the docs gates**

```bash
uv run pytest tests/unit/test_docs_staged_cli.py -q
uv run pytest tests/unit/cli/ tests/unit/sdk_/ -q
```

Expected: PASS.

`test_docs_staged_cli.py` does **not** check documented CLI flags against click
options — an earlier draft said it did. It asserts that `CLAUDE.md` and
`docs/source/how_to/pages/gpu_detection_setup.md` mention the staged-GPU Stage-2
signal (`stage2_raw`, `token`, `stage`, `GpuDetector`), that the three `--gpu-*`
flags appear, that `--gpu-batch-size` does not, and the paragraph-level
"sidecar" rule above. Nothing here validates `--process-format`, so its help
text is pinned by Task 8b's tests instead.

- [ ] **Step 5: Full regression run**

```bash
uv run pytest tests/unit -q \
    -n "$(python -c 'import os; print(len(os.sched_getaffinity(0)))')"
```

Expected: PASS. Do **not** use `-n auto` on the HPCC: it reads the node's core
count rather than the Slurm allocation's and manufactures timeout failures. The
suite is ~65 minutes, so run it as a Slurm job — there is a committed batch
script at
`docs/superpowers/plans/2026-08-18-ome-zarr-image-store/run_unit_suite.sbatch`.

- [ ] **Step 6: Commit**

```bash
git add docs/superpowers/logic_validation_scripts/2026-08-27-process-mode-ome-zarr/store_geometry.py \
        CLAUDE.md
git commit -m "docs: validate store geometry and document process-mode zarr

store_geometry.py re-derives the 12-file single-series count, the pyramid level
count and shapes, and the shard/chunk multiple from the NGFF and Zarr rules
directly, importing no phenotypic code -- so it fails if the spec and the format
disagree, not merely if the spec and the implementation do. It also asserts the
floor-based level formula is refuted and that the 4 + 2*levels shorthand stops
holding above a 4096-pixel edge, which is the boundary the shorthand hides.

CLAUDE.md documents the new default output, --process-format and its two
distinct refusals, and the imread-vs-load_zarr contract including NGFF's lack of
an RGB type. _cli_readme_generator.py is deliberately untouched: it documents
nothing about process mode, and phenotypicCLI.py:2324 skips the generator on
that path."
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

The second command consuming the first's output *is* the deliverable. It
exercises, in one command, every task in Phase 4: the scanner finding stores
(10b), `file_sha256` digesting one (10a), `process_only_output_path` naming
`p01.ome.zarr` rather than `p01.ome.ome.zarr` (7), and `imread` reading it (6).
Confirm the first run's stores open in napari
(`napari /tmp/acr-out/<stem>.ome.zarr`) if it is available.

- [ ] `--dry-run` on the same store tree reports the same image count as the
  real run. That is Task 10b's `get_input_structure_summary` half, and it is the
  command a user types first.
