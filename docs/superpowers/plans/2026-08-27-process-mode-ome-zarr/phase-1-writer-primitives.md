# Phase 1 — Writer primitives

Four independent changes to the writer and the provenance journal. Everything in
later phases depends on Tasks 1 and 4's signatures, so this phase lands first.

Read [`README.md`](README.md)'s **Global Constraints** before starting. They
apply to every task here.

---

### Task 1: `write_image_class` and `consolidate` threading

Two writer parameters, landing together because both are keyword-only additions
to the same three-layer call chain and both exist solely for the `--mode
process` caller.

**`write_image_class`.** Process-mode stores must omit
`attributes.phenotypic.image_class`. Today it is written unconditionally:
`_build_store_attributes` hardcodes `image_class=type(self).__name__`
(`_image_io_handler.py:850`) and `build_phenotypic_attributes` declares it a
required keyword (`ngff_.py:489`, written at `:545`).

**`consolidate`.** Consolidated metadata must be written **inside the `.part`,
immediately before `ngff_.promote_store`** (`_image_io_handler.py:1178`) — never
on the returned final path. Consolidating after the promote rewrites the root
`zarr.json` *at the final path*, which reintroduces the exact failure the design
claims to make unreachable (spec §"Why change" #2: *"a store either has its root
`zarr.json` or does not exist"*), and it lands after `promote_store`'s optional
`fsync`, leaving the consolidated root non-durable under SLURM. So it cannot be
a wrapper around `_save_store`'s return value; it has to be a parameter. Task 9
switches it on for the process writer and tests the resulting store; nothing
else ever passes `True`.

**Files:**
- Modify: `src/phenotypic/sdk_/ngff_.py:487-501` (signature) and the block
  assembly at `:540-546`
- Modify: `src/phenotypic/_core/_image_parts/_image_io_handler.py:829-869`
  (`_build_store_attributes`), `:931-994` (`_save_store`), `:996-1183`
  (`_write_store_part`, whose promote is at `:1178`)
- Test: `tests/unit/sdk_/test_ngff_attributes.py`,
  `tests/unit/sdk_/test_write_image_class.py` (create)

**Interfaces:**
- Produces:
  - `ngff_.build_phenotypic_attributes(*, image_class: str | None, …) -> dict`
    — omits the `image_class` key entirely when `image_class is None`.
  - `Image._save_store(path, *, series, write_objmap, levels, work_id, durable,
    commit_guard=None, measurement_table=None, write_image_class: bool = True,
    consolidate: bool = False)`
  - `Image._write_store_part(part, final, *, …, write_image_class: bool = True,
    consolidate: bool = False)`
  - `Image._build_store_attributes(*, series_names, levels, sections, work_id,
    has_labels=True, write_image_class: bool = True)` — no `consolidate`; it
    builds a dict and never touches the filesystem.
- Consumes: nothing.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/sdk_/test_ngff_attributes.py`:

```python
def test_image_class_is_omitted_entirely_when_none() -> None:
    """A process-mode store carries no image_class key at all.

    Not an empty string and not a null: the KEY is absent, because
    `load_zarr`'s guard tests membership, not truthiness.
    """
    block = ngff_.build_phenotypic_attributes(
        image_class=None,
        series_names=["rgb"],
        pyramid_levels=4,
        metadata_sections=_sections(),
        detect_mode="gray",
        illuminant="D65",
        gamma="SRGB",
        has_labels=False,
    )
    assert PhenotypicAttr.IMAGE_CLASS not in block


def test_image_class_is_written_when_given() -> None:
    block = ngff_.build_phenotypic_attributes(
        image_class="GridImage",
        series_names=["rgb"],
        pyramid_levels=4,
        metadata_sections=_sections(),
        detect_mode="gray",
        illuminant="D65",
        gamma="SRGB",
    )
    assert block[PhenotypicAttr.IMAGE_CLASS] == "GridImage"
```

Create `tests/unit/sdk_/test_write_image_class.py`:

```python
"""write_image_class=False is the only thing that omits image_class."""

from __future__ import annotations

import json
from pathlib import Path

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_ import ngff_
from phenotypic.sdk_.ngff_ import PhenotypicAttr


def _block(store: Path) -> dict:
    payload = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    return payload["attributes"][PhenotypicAttr.ROOT]


def test_save2zarr_still_writes_image_class(tmp_path: Path) -> None:
    """The default is unchanged; only the process-mode caller opts out."""
    img = Image(load_synth_yeast_plate())
    store = img.save2zarr(tmp_path / "bundle.ome.zarr")
    assert _block(store)[PhenotypicAttr.IMAGE_CLASS] == "Image"


def test_save_store_can_omit_image_class(tmp_path: Path) -> None:
    img = Image(load_synth_yeast_plate())
    store = img._save_store(
        tmp_path / "processed.ome.zarr",
        series=("gray",),
        write_objmap=False,
        levels=ngff_.pyramid_level_count(*img.gray[:].shape[:2]),
        work_id=None,
        durable=False,
        write_image_class=False,
    )
    block = _block(store)
    assert PhenotypicAttr.IMAGE_CLASS not in block
    # Everything else the store needs is still there.
    assert block[PhenotypicAttr.STORE_SCHEMA_VERSION] == ngff_.STORE_SCHEMA_VERSION
    assert block[PhenotypicAttr.SERIES] == {"gray": "gray"}
    assert PhenotypicAttr.LABELS not in block


def test_consolidation_is_off_by_default(tmp_path: Path) -> None:
    """Every existing caller keeps today's unconsolidated store."""
    img = Image(load_synth_yeast_plate())
    store = img.save2zarr(tmp_path / "bundle.ome.zarr")
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    assert "consolidated_metadata" not in root


def test_consolidate_writes_the_block_at_the_final_path(tmp_path: Path) -> None:
    """It is produced inside the .part, so the promoted root already has it.

    The promote is a directory rename, so a root ``zarr.json`` that exists at
    the final path is by construction the complete one -- which is the property
    consolidating *after* the promote would destroy.
    """
    img = Image(load_synth_yeast_plate())
    store = img._save_store(
        tmp_path / "consolidated.ome.zarr",
        series=("rgb",),
        write_objmap=False,
        levels=ngff_.pyramid_level_count(*img.rgb[:].shape[:2]),
        work_id=None,
        durable=False,
        consolidate=True,
    )
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    assert root["consolidated_metadata"]["must_understand"] is False
    assert sorted(root["consolidated_metadata"]["metadata"]) == [
        "OME", "rgb", "rgb/0", "rgb/1",
    ]
    # A sibling of `attributes`, so the phenotypic block is untouched.
    assert "consolidated_metadata" not in root["attributes"]
    assert ngff_.read_root_attributes(store)[PhenotypicAttr.ROOT][
        PhenotypicAttr.SERIES
    ] == {"rgb": "rgb"}


def test_consolidation_leaves_no_stray_part(tmp_path: Path) -> None:
    """The consolidated root is the promoted one, not a second write."""
    img = Image(load_synth_yeast_plate())
    target = tmp_path / "clean.ome.zarr"
    img._save_store(
        target,
        series=("rgb",),
        write_objmap=False,
        levels=2,
        work_id=None,
        durable=False,
        consolidate=True,
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clean.ome.zarr"]
```

The `["OME", "rgb", "rgb/0", "rgb/1"]` node list assumes
`pyramid_level_count(*img.rgb[:].shape[:2]) == 2`. `load_synth_yeast_plate()` is
600x800, so `max(600, 800) = 800 > 512` gives `ceil(log2(800/512)) + 1 = 2` —
verified by execution. If the fixture ever changes size, derive the expected
list from `levels` rather than pinning the literal.

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/sdk_/test_write_image_class.py \
              tests/unit/sdk_/test_ngff_attributes.py -v
```

Expected: FAIL. `build_phenotypic_attributes` raising
`TypeError: … got an unexpected keyword` is *not* what you should see — it takes
`image_class` already. You should see the `None` case write `"image_class": None`
(so `not in` fails), and `_save_store` raise
`TypeError: _save_store() got an unexpected keyword argument 'write_image_class'`.
`test_consolidation_is_off_by_default` PASSES already — it is the regression
half, pinning that nothing else starts consolidating.

- [ ] **Step 3: Widen `build_phenotypic_attributes`**

In `src/phenotypic/sdk_/ngff_.py`, change the signature and the block assembly:

```python
def build_phenotypic_attributes(
    *,
    image_class: str | None,
    series_names: Sequence[str],
    ...
) -> dict:
    """...

    Args:
        image_class: ``"Image"`` or ``"GridImage"`` -- drives loader dispatch.
            ``None`` omits the key entirely, which is what marks a store as
            **not** a run bundle: ``Image.load_zarr`` refuses a store without
            it. Only the ``--mode process`` writer passes ``None``.
        ...
    """
```

The block is assembled as a dict literal at `ngff_.py:540-570`, with
`PhenotypicAttr.IMAGE_CLASS: image_class` inline at `:545`. Drop that one line
from the literal and add the conditional immediately after the literal closes,
beside the existing `if provenance is not None:` at `:571`:

```python
    block: dict = {
        PhenotypicAttr.STORE_SCHEMA_VERSION: STORE_SCHEMA_VERSION,
        PhenotypicAttr.PHENOTYPIC_VERSION: (
            phenotypic_version or phenotypic.__version__
        ),
        PhenotypicAttr.SERIES: {name: name for name in series_names},
        ...
    }
    if image_class is not None:
        block[PhenotypicAttr.IMAGE_CLASS] = image_class
    if provenance is not None:
        block[PhenotypicAttr.PROVENANCE] = provenance
```

Key insertion order moves `image_class` after `metadata` rather than between
`phenotypic_version` and `series`. Nothing reads the block positionally — it is
JSON — and the alternative (a literal with a sentinel, popped afterwards) is
worse. Do not "fix" it back.

Leave `primary = primary_series(series_names)` (`ngff_.py:539`) exactly where it
is. It runs before the block is built and raises for a series tuple with no
`rgb`/`gray`, which is one of the two guards that make
`--layer detect_mat --process-format zarr` refusable (Task 8a).

- [ ] **Step 4: Thread the flag through the three `Image` methods**

`_build_store_attributes`:

```python
    def _build_store_attributes(
        self, *, series_names, levels, sections, work_id,
        has_labels=True, write_image_class=True,
    ) -> dict:
        """...

        Args:
            ...
            write_image_class: Write ``image_class``. ``False`` omits it, which
                is what makes ``load_zarr`` refuse the store. Only the
                ``--mode process`` writer passes ``False``.
        """
        return ngff_.build_phenotypic_attributes(
            image_class=type(self).__name__ if write_image_class else None,
            ...
        )
```

`_save_store` (`:931-942`) and `_write_store_part` (`:996-1008`) each gain
`write_image_class: bool = True` as a keyword-only parameter; `_save_store`
forwards it to `_write_store_part` in the call at `:981-991`, and
`_write_store_part` forwards it to `_build_store_attributes` in the call at
`:1157-1163`. Document it in both docstrings with the same wording.

Do **not** touch `save2zarr` or `save_intermediate_zarr` — they keep the default
and their behaviour is unchanged.

- [ ] **Step 5: Thread `consolidate` and apply it before the promote**

`_save_store` and `_write_store_part` each gain `consolidate: bool = False`,
forwarded the same way. In `_write_store_part`, the only new logic is
immediately before the existing `return ngff_.promote_store(...)` at `:1178`:

```python
        if consolidate:
            _consolidate_store_part(part)
        return ngff_.promote_store(
            part,
            final,
            fsync=ngff_.durable_writes_enabled(durable),
            commit_guard=commit_guard,
        )
```

Add the helper as a module-level function in `_image_io_handler.py`:

```python
def _consolidate_store_part(part: Path) -> None:
    """Collapse a written part's per-node metadata into its root ``zarr.json``.

    Opening a store costs one GET per metadata file -- 8 of a single-series
    store's 12 -- which is the latency that matters once the destination is
    object storage and a viewer enumerates many stores. Consolidation collapses
    that to one and **adds no files**: every per-node ``zarr.json`` remains, so
    a reader that ignores the key still walks the tree.

    Called on the ``.part``, never on the promoted path. Consolidating a
    promoted store rewrites its root ``zarr.json`` in place, which is the one
    write this whole design exists to avoid -- a store must either have its
    root or not exist -- and it would land after ``promote_store``'s ``fsync``,
    leaving the consolidated root non-durable under SLURM.

    Legal under the Zarr v3 extension mechanism rather than in spite of it: the
    core specification requires a reader to fail on an unrecognised metadata
    field *unless* it carries ``"must_understand": false``, and zarr-python
    writes exactly that (verified against zarr 3.1.5).

    Two ``ZarrUserWarning``s are suppressed here rather than by a global
    filter: the consolidated-metadata notice, and ``"Object at
    METADATA.ome.xml is not recognized as a component of a Zarr hierarchy"``,
    which fires once per image. Both are instances of the same class -- a
    ``UserWarning`` subclass -- so filtering on the class covers both, where a
    ``message=`` regex naming only consolidation would let the second through.
    """
    import zarr
    from zarr.errors import ZarrUserWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ZarrUserWarning)
        zarr.consolidate_metadata(ngff_.long_path(part))
```

`warnings` and `Path` are already imported at module scope; `ngff_` is imported
lazily inside each method, so import it at the top of the helper the same way
(`from phenotypic.sdk_ import ngff_`) rather than at module scope.

Verified by execution against zarr 3.1.5 that consolidating a part and then
renaming it works: the consolidated block records node paths relative to the
root (`["OME", "rgb", "rgb/0", "rgb/1"]`), so the rename does not invalidate it,
and `zarr.open_group` on the promoted store resolves `rgb/0` through the
consolidated view.

- [ ] **Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/unit/sdk_/test_write_image_class.py \
              tests/unit/sdk_/test_ngff_attributes.py -v
```

Expected: PASS.

- [ ] **Step 7: Run the store regression suite**

```bash
uv run pytest tests/unit/sdk_/ tests/unit/test_ome_zarr_invariants.py -q
uv run mypy src/phenotypic/sdk_/ngff_.py \
            src/phenotypic/_core/_image_parts/_image_io_handler.py
uv run ruff check --fix src/phenotypic/sdk_/ngff_.py \
    src/phenotypic/_core/_image_parts/_image_io_handler.py \
    tests/unit/sdk_/test_write_image_class.py
```

Expected: PASS, no new mypy errors. Widening `image_class` to `str | None` is
low-risk: `build_phenotypic_attributes` has exactly one production caller,
`_build_store_attributes` (`_image_io_handler.py:849`), which passes
`type(self).__name__` — a `str`. If mypy does flag something, make the intent
explicit at that site rather than casting.

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/sdk_/ngff_.py \
        src/phenotypic/_core/_image_parts/_image_io_handler.py \
        tests/unit/sdk_/test_write_image_class.py \
        tests/unit/sdk_/test_ngff_attributes.py
git commit -m "feat(ngff): let a store omit image_class and consolidate its part

build_phenotypic_attributes takes image_class: str | None and omits the key
when None; _save_store / _write_store_part / _build_store_attributes thread
write_image_class. Only the --mode process writer will pass False. save2zarr
and save_intermediate_zarr are unchanged. Omitting the key is what makes
load_zarr able to refuse a store that is not a run bundle (spec 3.3).

_save_store / _write_store_part also take consolidate, applied to the .part
immediately before promote_store. It has to be a parameter rather than a
wrapper around the returned path: consolidating a promoted store rewrites its
root zarr.json in place, reintroducing the truncated-artifact failure the
rename-commit exists to prevent, and lands after promote_store's fsync. Off by
default; only the process writer turns it on (Task 9)."
```

---

### Task 2: `load_zarr` guard

`load_zarr` currently does `block.get(PhenotypicAttr.IMAGE_CLASS)`
(`_image_io_handler.py:1673`). A missing key yields `None`, which is not
`"GridImage"`, so control falls through to `_load_from_store` — which reads
every field with a defaulting `.get()` and happily returns an `Image` with empty
metadata and no objmap. That is a plausible-looking wrong result, not an error.

**The guard goes before `require_readable_store`, not after it.** That call
(`:1672`) delegates to `read_phenotypic_attributes`, whose last line is
`return attributes[PhenotypicAttr.ROOT]` (`ngff_.py:623`) — a bare subscript. On
a third-party store, which carries no `phenotypic` block at all, it raises
`KeyError` before any guard placed after it could run. So the guard reads the
root attributes directly and raises **one** `ValueError` for both shapes of "not
a run bundle": no `phenotypic` block, and a block with no `image_class`. One
error, one remedy, one message — a caller does not care which of the two
applies, and splitting them leaves the more common case (a third-party store)
with the worse error.

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/_image_io_handler.py:1636-1681`
- Test: `tests/unit/sdk_/test_load_zarr_guard.py` (create)

**Interfaces:**
- Consumes: Task 1's `write_image_class=False`, to construct the fixture.
- Produces: `Image.load_zarr` raises `ValueError` when `image_class` is absent.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/sdk_/test_load_zarr_guard.py`:

```python
"""load_zarr refuses a store that is not a PhenoTypic run bundle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_ import ngff_
from phenotypic.sdk_.ngff_ import PhenotypicAttr


def _processed_store(tmp_path: Path) -> Path:
    img = Image(load_synth_yeast_plate())
    return img._save_store(
        tmp_path / "processed.ome.zarr",
        series=("gray",),
        write_objmap=False,
        levels=ngff_.pyramid_level_count(*img.gray[:].shape[:2]),
        work_id=None,
        durable=False,
        write_image_class=False,
    )


def test_load_zarr_refuses_a_processed_store(tmp_path: Path) -> None:
    store = _processed_store(tmp_path)
    with pytest.raises(ValueError, match="image_class"):
        Image.load_zarr(store)


def test_the_error_names_imread_as_the_remedy(tmp_path: Path) -> None:
    """A user who hits this must be told what to call instead."""
    store = _processed_store(tmp_path)
    with pytest.raises(ValueError, match="imread"):
        Image.load_zarr(store)


def test_load_zarr_does_not_return_a_degraded_image(tmp_path: Path) -> None:
    """Regression: today this returns an Image with no objmap, silently.

    The guard exists because a plausible wrong object is worse than an error.
    """
    store = _processed_store(tmp_path)
    try:
        result = Image.load_zarr(store)
    except ValueError:
        return
    pytest.fail(
        f"load_zarr returned {type(result).__name__} with "
        f"num_objects={result.num_objects} instead of raising"
    )


def test_a_third_party_store_raises_the_same_guard(tmp_path: Path) -> None:
    """No phenotypic block at all: a clear ValueError, not a bare KeyError.

    ValueError specifically, and not the wider `(ValueError, KeyError)` an
    earlier draft allowed: KeyError is exactly what this test exists to rule
    out, so accepting it would let the guard be placed after
    `require_readable_store`, where it never fires.
    """
    store = tmp_path / "foreign.ome.zarr"
    store.mkdir()
    (store / "zarr.json").write_text(
        json.dumps({
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {"ome": {"version": "0.5"}},
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="imread"):
        Image.load_zarr(store)


def test_a_missing_store_still_raises_file_not_found(tmp_path: Path) -> None:
    """The guard must not turn an absent store into a 'not a bundle' error.

    `read_root_attributes` raises FileNotFoundError on a store with no root
    zarr.json, which is the normal 'interrupted write' signal every staged
    caller depends on. Reading the root before the guard must not swallow it.
    """
    with pytest.raises(FileNotFoundError):
        Image.load_zarr(tmp_path / "absent.ome.zarr")


def test_a_bundle_store_still_loads(tmp_path: Path) -> None:
    """The guard must not fire on the path it is not for."""
    img = Image(load_synth_yeast_plate())
    store = img.save2zarr(tmp_path / "bundle.ome.zarr")
    assert Image.load_zarr(store).gray[:].shape == img.gray[:].shape
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/sdk_/test_load_zarr_guard.py -v
```

Expected: the three process-store guard tests FAIL (no exception raised);
`test_a_third_party_store_raises_the_same_guard` FAILS with `KeyError` rather
than `ValueError`; `test_a_bundle_store_still_loads` and
`test_a_missing_store_still_raises_file_not_found` PASS already.

- [ ] **Step 3: Add the guard, before the version gate**

Replace `load_zarr`'s two-line preamble (`_image_io_handler.py:1672-1673`):

```python
        from phenotypic.sdk_ import ngff_

        # BEFORE require_readable_store, deliberately. That call ends in
        # `attributes[PhenotypicAttr.ROOT]` (ngff_.py:623), a bare subscript,
        # so a third-party store raises KeyError there and a guard placed
        # after it would never fire on the case it most needs to serve.
        # FileNotFoundError from read_root_attributes still propagates: an
        # interrupted write reads as absent, not as "not a bundle".
        attributes = ngff_.read_root_attributes(path)
        phenotypic_block = attributes.get(ngff_.PhenotypicAttr.ROOT, {})
        if ngff_.PhenotypicAttr.IMAGE_CLASS not in phenotypic_block:
            raise ValueError(
                f"{path} carries no phenotypic.image_class and is not a "
                f"PhenoTypic run bundle. It was written by --mode process or "
                f"by another tool. Use Image.imread() to read its pixels."
            )

        block = ngff_.require_readable_store(path)
        saved_class = block.get(ngff_.PhenotypicAttr.IMAGE_CLASS)
```

Three things are deliberate:

- `not in`, not a falsy test. The contract is key **absence**; an `image_class`
  of `""` is a corrupt bundle, not a processed store.
- `.get(ROOT, {})` on the root attributes, so a store with no `phenotypic` block
  falls into the same branch and gets the same message.
- `require_readable_store` still runs afterwards and is still the version gate.
  The guard answers "is this a bundle at all"; the gate answers "can this build
  decode it". Ordering them this way means a *future* PhenoTypic bundle still
  gets the version error, not the not-a-bundle one.

Add to the `Raises:` block of the docstring (`:1650-1655`):

```
            ValueError: If the store carries no ``phenotypic.image_class`` --
                it was written by ``--mode process`` or by another tool and is
                not a run bundle; use :meth:`imread` to read its pixels. Also
                if ``store_schema_version`` is not this build's.
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/sdk_/test_load_zarr_guard.py -v
```

Expected: PASS (6 tests).

- [ ] **Step 5: Run the regression suite**

```bash
uv run pytest tests/unit/sdk_/ tests/unit/test_ome_zarr_invariants.py -q
uv run ruff check --fix \
    src/phenotypic/_core/_image_parts/_image_io_handler.py \
    tests/unit/sdk_/test_load_zarr_guard.py
```

Expected: PASS. If a GUI preview test fails, `save_intermediate_zarr` is
reaching `load_zarr` somewhere — it writes `image_class` (Task 1 left its
default alone), so investigate rather than weakening the guard.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_core/_image_parts/_image_io_handler.py \
        tests/unit/sdk_/test_load_zarr_guard.py
git commit -m "fix(io): load_zarr refuses a store with no image_class

It previously fell through to _load_from_store, which reads every field with a
defaulting .get() and returned an Image with empty metadata and no objmap -- a
plausible-looking wrong result rather than an error. The guard tests key
absence and names imread as the remedy.

Placed BEFORE require_readable_store: that call ends in a bare
attributes[PhenotypicAttr.ROOT] subscript, so a third-party store raises
KeyError there and a guard after it would never fire on the case it most needs
to serve. Reading the root attributes directly gives both shapes of 'not a run
bundle' -- no phenotypic block, and a block with no image_class -- one error,
one remedy, one message."
```

---

### Task 3: `omero.rdefs.model` on integer series

NGFF 0.5 §2.5 documents `rdefs.model` as taking exactly `"color"` or
`"greyscale"`. It is the only field in the format that states the rendering
model outright, and OMERO and Vizarr read it. Verified absent today:
`grep -rn rdefs src/ tests/` returns nothing.

**Files:**
- Modify: `src/phenotypic/sdk_/ngff_.py:737-795` (`build_omero`); the float
  guard is `:776-777` and the block assembly `:792-795`
- Test: `tests/unit/sdk_/test_ngff_projection.py`

**Interfaces:**
- Produces: `build_omero` output gains `omero.rdefs.model`. The float guard and
  the whole-or-nothing rule are unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/sdk_/test_ngff_projection.py`:

```python
import numpy as np

from phenotypic.sdk_ import ngff_


def test_rgb_declares_the_color_rendering_model() -> None:
    block = ngff_.build_omero(
        series="rgb", dtype=np.dtype("uint16"), bit_depth=16, name="p01"
    )
    assert block["omero"]["rdefs"] == {"model": "color"}


def test_single_channel_integer_declares_greyscale() -> None:
    block = ngff_.build_omero(
        series="gray", dtype=np.dtype("uint8"), bit_depth=8, name="p01"
    )
    assert block["omero"]["rdefs"] == {"model": "greyscale"}


def test_float_series_still_emit_nothing_at_all() -> None:
    """The 2026-08-19 float ruling is untouched: no block means no rdefs."""
    assert ngff_.build_omero(
        series="gray", dtype=np.dtype("float32"), bit_depth=8
    ) == {}
    assert ngff_.build_omero(
        series="detect_mat", dtype=np.dtype("float64"), bit_depth=16
    ) == {}


def test_rdefs_does_not_disturb_the_channel_contract() -> None:
    """NGFF makes omero conditionally strict; rdefs must not weaken it."""
    channels = ngff_.build_omero(
        series="rgb", dtype=np.dtype("uint8"), bit_depth=8
    )["omero"]["channels"]
    assert [c["color"] for c in channels] == ["FF0000", "00FF00", "0000FF"]
    for channel in channels:
        assert set(channel["window"]) == {"min", "max", "start", "end"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/sdk_/test_ngff_projection.py -v -k "rdefs or greyscale or float_series"
```

Expected: `KeyError: 'rdefs'` on the first two;
`test_float_series_still_emit_nothing_at_all` and the channel-contract test
PASS already.

- [ ] **Step 3: Emit `rdefs`**

In `build_omero`, replace the block literal at `:792`:

```python
    block: dict = {
        "channels": channels,
        "rdefs": {"model": "color" if series == "rgb" else "greyscale"},
    }
    if name is not None:
        block["name"] = name
    return {"omero": block}
```

The `name` and `return` lines (`:793-795`) are unchanged; they are shown only so
the insertion point is unambiguous. The float guard at `:776-777` returns `{}`
before this line is reached, which is why no `rdefs` reaches a float series.

Add to the docstring, under the existing explanation of the float guard:

```
    ``rdefs.model`` is the only field in NGFF that states the rendering model
    outright (§2.5: exactly ``"color"`` or ``"greyscale"``), and OMERO and
    Vizarr read it. It is emitted only where ``omero`` itself is emitted, so
    the whole-or-nothing rule per group is unaffected.
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/sdk_/test_ngff_projection.py -v
```

Expected: PASS.

- [ ] **Step 5: Re-run the NGFF conformance gate**

```bash
uv run pytest tests/unit/sdk_/test_ngff_validity.py \
              tests/unit/test_ome_zarr_invariants.py -q
```

Expected: PASS. `rdefs` is a documented `omero` field, so the published schema
accepts it. If validation fails, read the reported error before changing
anything — the schema pins `model` to those two literals and a typo is the
likely cause.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/sdk_/ngff_.py tests/unit/sdk_/test_ngff_projection.py
git commit -m "feat(ngff): declare the omero rendering model

build_omero emits rdefs.model -- \"color\" for rgb, \"greyscale\" for a
single-channel integer series. It is the only NGFF field that states the
rendering model outright and is what OMERO and Vizarr read. Float series
still emit no omero block at all, so the 2026-08-19 ruling is untouched."
```

---

### Task 4: pipeline basename in the provenance journal

`pipeline_source_identity` (`_core/_provenance.py:276-282`) stores
`Path(path).resolve()`. For a store published to a NAS and then to object
storage that leaks cluster filesystem layout, the username, and project
directory names. `sha256` already pins the pipeline's identity exactly, so the
path is convenience, not identity.

**Files:**
- Modify: `src/phenotypic/_core/_provenance.py:276-282`
  (`pipeline_source_identity`) and `:285-306` (`initialize_cli_provenance`)
- Test: `tests/unit/test_provenance_source_identity.py` (create)

**Interfaces:**
- Produces:
  - `pipeline_source_identity(path, *, basename_only: bool = False) -> dict[str, str]`
  - `initialize_cli_provenance(image, pipeline_path, *, pipeline_identity=None,
    status="in_progress", retry_base_length=0, basename_only: bool = False)`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_provenance_source_identity.py`:

```python
"""A published store must not carry the cluster path of its pipeline file."""

from __future__ import annotations

from pathlib import Path

from phenotypic._core._provenance import (
    initialize_cli_provenance,
    pipeline_source_identity,
)
from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate


def _pipeline_file(tmp_path: Path) -> Path:
    nested = tmp_path / "config" / "deep"
    nested.mkdir(parents=True)
    path = nested / "preprocess_pipeline.json.pht-pipe"
    path.write_text('{"name": "acr_preprocess"}', encoding="utf-8")
    return path


def test_default_records_the_resolved_absolute_path(tmp_path: Path) -> None:
    """Bundle stores keep the absolute path; they never leave the run dir."""
    path = _pipeline_file(tmp_path)
    identity = pipeline_source_identity(path)
    assert identity["source_path"] == str(path.resolve())


def test_basename_only_records_just_the_filename(tmp_path: Path) -> None:
    path = _pipeline_file(tmp_path)
    identity = pipeline_source_identity(path, basename_only=True)
    assert identity["source_path"] == "preprocess_pipeline.json.pht-pipe"
    assert "/" not in identity["source_path"]


def test_the_digest_is_identical_either_way(tmp_path: Path) -> None:
    """sha256 is the identity; basename_only must not weaken it."""
    path = _pipeline_file(tmp_path)
    assert (
        pipeline_source_identity(path)["sha256"]
        == pipeline_source_identity(path, basename_only=True)["sha256"]
    )


def test_initialize_cli_provenance_threads_the_flag(tmp_path: Path) -> None:
    path = _pipeline_file(tmp_path)
    img = Image(load_synth_yeast_plate())
    initialize_cli_provenance(img, path, basename_only=True)
    journal = img._metadata.provenance_journal
    assert journal["pipeline"]["source_path"] == "preprocess_pipeline.json.pht-pipe"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_provenance_source_identity.py -v
```

Expected: `TypeError: pipeline_source_identity() got an unexpected keyword
argument 'basename_only'` on three of the four;
`test_default_records_the_resolved_absolute_path` PASSES already.

- [ ] **Step 3: Add the parameter**

```python
def pipeline_source_identity(
    path: str | Path, *, basename_only: bool = False
) -> dict[str, str]:
    """Return the pipeline's recorded source and SHA-256 content identity.

    Args:
        path: The pipeline file.
        basename_only: Record only the file's name rather than its resolved
            absolute path. ``True`` for artifacts that leave the run directory
            -- a ``--mode process`` store is published to a NAS and then to
            object storage, and an absolute path there would carry cluster
            filesystem layout, the username, and project directory names.
            ``sha256`` is unchanged, so identity is not weakened.

    Returns:
        ``{"source_path": …, "sha256": …}``.
    """
    source = Path(path).resolve()
    return {
        "source_path": source.name if basename_only else str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
```

Note the digest still reads from the **resolved** path — only what is *recorded*
changes.

- [ ] **Step 4: Thread it through `initialize_cli_provenance`**

```python
def initialize_cli_provenance(
    image: "Image",
    pipeline_path: str | Path,
    *,
    pipeline_identity: Mapping[str, str] | None = None,
    status: str = "in_progress",
    retry_base_length: int = 0,
    basename_only: bool = False,
) -> None:
    """Reset a decoded image to a fresh CLI journal for this pipeline attempt.

    Args:
        ...
        basename_only: Forwarded to :func:`pipeline_source_identity`. Ignored
            when *pipeline_identity* is supplied, since the caller has then
            already decided what to record.
    """
    journal = new_provenance_journal()
    journal.update(
        {
            "status": status,
            "pipeline": (
                pipeline_source_identity(
                    pipeline_path, basename_only=basename_only
                )
                if pipeline_identity is None
                else deepcopy(dict(pipeline_identity))
            ),
            "retry_base_length": int(retry_base_length),
        }
    )
    image._metadata.provenance_journal = journal
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_provenance_source_identity.py -v
```

Expected: PASS (4 tests).

- [ ] **Step 6: Run the provenance regression suite**

```bash
uv run pytest tests/unit -q -k "provenance"
uv run mypy src/phenotypic/_core/_provenance.py
uv run ruff check --fix src/phenotypic/_core/_provenance.py \
    tests/unit/test_provenance_source_identity.py
```

Expected: PASS. Every existing caller omits the new keyword and keeps today's
behaviour.

- [ ] **Step 7: Commit**

```bash
git add src/phenotypic/_core/_provenance.py \
        tests/unit/test_provenance_source_identity.py
git commit -m "feat(provenance): record the pipeline basename for published stores

pipeline_source_identity and initialize_cli_provenance take basename_only.
A --mode process store is published to a NAS and then to object storage; an
absolute resolved path there carries cluster filesystem layout, the username,
and project directory names. sha256 is computed from the resolved path either
way, so identity is unchanged. Default behaviour is untouched."
```
