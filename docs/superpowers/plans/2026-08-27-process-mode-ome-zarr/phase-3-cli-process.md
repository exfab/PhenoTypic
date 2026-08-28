# Phase 3 — CLI process mode

The writer that produces the artifact, the option that selects it, and the
option's journey out to the command a user actually runs.
**Depends on Tasks 1 and 4** (`write_image_class`, `consolidate`,
`basename_only`).

Read [`README.md`](README.md)'s **Global Constraints** first. One binds hardest
here: **only `rgb` and `gray` can be a store's sole series.** `detect_mat` and
`objmap` are refused, for two different reasons, and the error text has to say
which.

---

### Task 7: process-only zarr writer + provenance init

Two things land together because neither is testable without the other: the
zarr branch of the writer, and the `initialize_cli_provenance` call that gives
the store its `pipeline` identity.

**What provenance init actually buys, and what it does not.** An earlier draft
claimed a process-mode store "would carry the empty journal". That is wrong:
`wrap_image_operation_apply` (`_provenance.py:181-266`) calls
`append_operation_provenance` **unconditionally** at every operation's success
edge (`:227`), so `operations[]` is populated whether or not anything
initialised the journal. What is missing without the call is the `pipeline` key
— `new_provenance_journal()` sets it to `None` (`_provenance.py:77`) — which is
the pipeline's `source_path` and `sha256`, and the whole reason
`ColorCorrector`'s profile can be traced back to a config file. The action is
unchanged; only the reason is.

**`initialize_cli_provenance` RESETS the journal.** Its first statement is
`journal = new_provenance_journal()` (`_provenance.py:294`), which discards
`operations[]`. So the call must come **before** `pipeline.apply()`, not after —
calling it after would silently erase every operation record it exists to
contextualise. This is load-bearing, not stylistic.

**A consequence, recorded deliberately.** When the input is itself a
process-mode store, `_imread_store` (Task 6) restores that store's journal and
this call then discards it. The published store therefore records *this* run's
operations and *this* run's pipeline, not the chain back to the original TIFF.
That is the correct meaning of a per-attempt journal and it matches what the
full-run path does (`_cli_process_single.py:261`), but it means chaining process
runs does not accumulate provenance. If accumulation is ever wanted it is a
change to the journal model, not to this call site.

**Files:**
- Modify: `src/phenotypic/_cli/_cli_process_only.py` (whole file, 120 lines:
  `process_only_output_path` `:23-36`, `write_process_only_layer` `:39-70`,
  `process_single_apply_only_core` `:73-120`)
- Modify: `src/phenotypic/sdk_/typing_.py:94-97` (`ProcessOnlyLayer`'s docstring
  and the new `ProcessFormat`)
- Test: `tests/unit/cli/test_process_only_zarr.py` (create)

**Interfaces:**
- Consumes: `Image._save_store(…, write_image_class=False)` (Task 1),
  `initialize_cli_provenance(…, basename_only=True)` (Task 4),
  `ngff_.pyramid_level_count`, `ngff_.STORE_SUFFIX`, `sdk_.store_stem`.
- Produces:
  - `ProcessFormat = Literal["tiff", "zarr"]` in `sdk_/typing_.py`
  - `_ZARR_CAPABLE_LAYERS: frozenset[str]` in `_cli_process_only.py`
  - `process_only_output_path(output_dir, image_path, input_root, layer, fmt="tiff")`
  - `write_process_only_layer(image, layer, out_path, *, fmt="tiff", commit_guard=None)`
  - `process_single_apply_only_core(…, process_format: ProcessFormat = "tiff")`

  The two parameter names are deliberate and not a slip: the leaf helpers take
  `fmt`, the core takes `process_format` to mirror the CLI option it is fed
  from. Do not "unify" them.

- [ ] **Step 1: Add the format type and correct the layer type's docstring**

In `src/phenotypic/sdk_/typing_.py`, the `ProcessOnlyLayer` comment at `:94-96`
currently claims *"``rgb``/``gray``/``detect_mat`` save as TIFF"*, which stops
being true the moment this task lands. Replace it and add the new alias:

```python
#: Image layer a process-mode CLI run exports. A closed subset of the layers
#: exposed as Image accessors. The output FORMAT is a separate axis -- see
#: :data:`ProcessFormat` -- and its default depends on the layer: ``rgb`` and
#: ``gray`` default to an OME-Zarr store, ``detect_mat`` to a float TIFF, and
#: ``objmap`` to a 16-bit raw-label PNG.
ProcessOnlyLayer = Literal["rgb", "gray", "detect_mat", "objmap"]

#: Output format for ``--mode process``. ``zarr`` writes a single-series
#: OME-Zarr store; ``tiff`` writes the flat image file (a 16-bit PNG for
#: ``objmap``). Only ``rgb`` and ``gray`` have a zarr form -- see
#: ``_cli_process_only.resolve_process_format`` for why, and for the two
#: distinct refusals.
ProcessFormat = Literal["tiff", "zarr"]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/cli/test_process_only_zarr.py`:

```python
"""--mode process writes a single-series store carrying its own provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic import Image, ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_ import ngff_
from phenotypic.sdk_.ngff_ import PhenotypicAttr
from phenotypic._cli._cli_process_only import (
    process_only_output_path,
    process_single_apply_only_core,
    write_process_only_layer,
)


@pytest.fixture
def source_image(tmp_path: Path) -> Path:
    root = tmp_path / "in"
    root.mkdir()
    path = root / "IMG_4471.tiff"
    Image(load_synth_yeast_plate()).rgb.imsave(filepath=path)
    return path


@pytest.fixture
def pipeline_file(tmp_path: Path) -> Path:
    """Named `.pht-pipe` because `to_json` RENAMES anything else.

    `ImagePipeline().to_json(tmp / "preprocess.json")` writes
    `preprocess.json.pht-pipe` and returns None -- verified by execution -- so
    a fixture that returns the path it passed in returns a path that does not
    exist, and every test using it dies in `from_json`.
    """
    nested = tmp_path / "config" / "deep"
    nested.mkdir(parents=True)
    path = nested / "preprocess.json.pht-pipe"
    ImagePipeline().to_json(path)
    assert path.is_file()
    return path


def _block(store: Path) -> dict:
    payload = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    return payload["attributes"][PhenotypicAttr.ROOT]


def test_output_path_is_a_store_for_zarr(tmp_path: Path) -> None:
    out = process_only_output_path(
        tmp_path / "out", tmp_path / "in" / "a" / "p01.tiff",
        tmp_path / "in", "rgb", fmt="zarr",
    )
    assert out == tmp_path / "out" / "a" / f"p01{ngff_.STORE_SUFFIX}"


def test_output_path_is_unchanged_for_tiff(tmp_path: Path) -> None:
    assert process_only_output_path(
        tmp_path / "out", tmp_path / "in" / "p01.tiff", tmp_path / "in", "rgb",
    ).name == "p01.tiff"
    assert process_only_output_path(
        tmp_path / "out", tmp_path / "in" / "p01.tiff", tmp_path / "in", "objmap",
    ).name == "p01.png"


def test_a_store_input_does_not_double_its_suffix(tmp_path: Path) -> None:
    """`Path("p01.ome.zarr").stem` is `"p01.ome"` -> `p01.ome.ome.zarr`.

    Spec 7.3. A tree of stores is valid input, so this is the ordinary case
    for the second run of the loop, not an exotic one -- and the wrong name is
    a plausible-looking one that nothing raises on.
    """
    store_in = tmp_path / "in" / f"p01{ngff_.STORE_SUFFIX}"
    assert process_only_output_path(
        tmp_path / "out", store_in, tmp_path / "in", "rgb", fmt="zarr",
    ).name == f"p01{ngff_.STORE_SUFFIX}"
    assert process_only_output_path(
        tmp_path / "out", store_in, tmp_path / "in", "detect_mat", fmt="tiff",
    ).name == "p01.tiff"


def test_writer_emits_only_the_requested_series(tmp_path: Path) -> None:
    img = Image(load_synth_yeast_plate())
    out = tmp_path / "p01.ome.zarr"
    write_process_only_layer(img, "rgb", out, fmt="zarr")
    assert (out / "rgb").is_dir()
    assert not (out / "gray").exists()
    assert not (out / "detect_mat").exists()
    assert not (out / "rgb" / "labels").exists()
    assert _block(out)[PhenotypicAttr.SERIES] == {"rgb": "rgb"}


def test_writer_omits_image_class(tmp_path: Path) -> None:
    img = Image(load_synth_yeast_plate())
    out = tmp_path / "p01.ome.zarr"
    write_process_only_layer(img, "rgb", out, fmt="zarr")
    assert PhenotypicAttr.IMAGE_CLASS not in _block(out)


@pytest.mark.parametrize("layer", ["detect_mat", "objmap"])
def test_the_writer_refuses_a_layer_with_no_store_form(
    tmp_path: Path, layer: str
) -> None:
    """Belt to the CLI's braces. `_save_store` would raise anyway for
    detect_mat -- `primary_series` accepts only rgb/gray -- but with a message
    about internal series naming rather than about what the user asked for.
    """
    img = Image(load_synth_yeast_plate())
    with pytest.raises(ValueError, match=layer):
        write_process_only_layer(img, layer, tmp_path / "x.ome.zarr", fmt="zarr")


def test_a_single_series_rgb_store_is_twelve_files(tmp_path: Path) -> None:
    """Spec 1.1. Guards against an accidental extra series or level.

    The `4 + 2 * levels` shorthand holds only while every pyramid level fits
    inside ONE shard -- true up to a 4096-pixel level-0 edge, and true for the
    600x800 synthetic plate. Above that a level contributes more than one shard
    file; the committed validation script (Task 11) carries the general form.
    """
    img = Image(load_synth_yeast_plate())
    out = tmp_path / "p01.ome.zarr"
    write_process_only_layer(img, "rgb", out, fmt="zarr")
    levels = ngff_.pyramid_level_count(*img.rgb[:].shape[:2])
    files = [p for p in out.rglob("*") if p.is_file()]
    # root + OME/zarr.json + OME xml + series zarr.json + 2 per level
    assert len(files) == 4 + 2 * levels


def test_core_records_the_pipeline_basename_not_its_path(
    tmp_path: Path, source_image: Path, pipeline_file: Path
) -> None:
    """A published store must not carry cluster filesystem layout."""
    out = tmp_path / "out"
    process_single_apply_only_core(
        pipeline_path=pipeline_file,
        image_path=source_image,
        input_root=source_image.parent,
        output_dir=out,
        image_type="Image",
        layer="rgb",
        read_kwargs={},
        process_format="zarr",
    )
    store = out / f"IMG_4471{ngff_.STORE_SUFFIX}"
    journal = _block(store)[PhenotypicAttr.PROVENANCE]
    assert journal["pipeline"]["source_path"] == "preprocess.json.pht-pipe"
    assert "/" not in journal["pipeline"]["source_path"]
    assert len(journal["pipeline"]["sha256"]) == 64


def test_provenance_init_runs_before_apply_not_after(
    tmp_path: Path, source_image: Path, pipeline_file: Path
) -> None:
    """`initialize_cli_provenance` resets the journal (_provenance.py:294).

    Called after `pipeline.apply()` it would discard `operations[]` -- and the
    store would still have a `pipeline` key, so the store looks fine and the
    operations are simply gone. The pipeline here has no operations, so what
    this pins is that BOTH keys survive: an empty list, not a missing one.
    """
    out = tmp_path / "out"
    process_single_apply_only_core(
        pipeline_path=pipeline_file,
        image_path=source_image,
        input_root=source_image.parent,
        output_dir=out,
        image_type="Image",
        layer="rgb",
        read_kwargs={},
        process_format="zarr",
    )
    journal = _block(out / f"IMG_4471{ngff_.STORE_SUFFIX}")[
        PhenotypicAttr.PROVENANCE
    ]
    assert journal["pipeline"] is not None
    assert journal["operations"] == []


def test_the_store_round_trips_through_imread(
    tmp_path: Path, source_image: Path, pipeline_file: Path
) -> None:
    """The loop closes: what process mode writes, imread reads."""
    out = tmp_path / "out"
    process_single_apply_only_core(
        pipeline_path=pipeline_file,
        image_path=source_image,
        input_root=source_image.parent,
        output_dir=out,
        image_type="Image",
        layer="rgb",
        read_kwargs={},
        process_format="zarr",
    )
    store = out / f"IMG_4471{ngff_.STORE_SUFFIX}"
    assert Image.imread(store).name == "IMG_4471"


def test_tiff_output_is_unchanged(
    tmp_path: Path, source_image: Path, pipeline_file: Path
) -> None:
    """The default path, and the AutoConvertRaw contract, must not move."""
    out = tmp_path / "out"
    process_single_apply_only_core(
        pipeline_path=pipeline_file,
        image_path=source_image,
        input_root=source_image.parent,
        output_dir=out,
        image_type="Image",
        layer="rgb",
        read_kwargs={},
        process_format="tiff",
    )
    assert (out / "IMG_4471.tiff").is_file()
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/unit/cli/test_process_only_zarr.py -v
```

Expected: `TypeError: process_only_output_path() got an unexpected keyword
argument 'fmt'`.

- [ ] **Step 4: Make the output path format-aware and store-stem-aware**

In `src/phenotypic/_cli/_cli_process_only.py`:

```python
def process_only_output_path(
    output_dir: Path,
    image_path: Path,
    input_root: Path,
    layer: ProcessOnlyLayer,
    fmt: ProcessFormat = "tiff",
) -> Path:
    """Mirror ``image_path`` (relative to ``input_root``) under ``output_dir``.

    Args:
        output_dir: Run output root.
        image_path: The input image -- a flat file, or a ``*.ome.zarr`` store
            directory once a tree of process-mode output is used as input.
        input_root: The ``--input`` root the mirror is relative to.
        layer: The layer being exported.
        fmt: ``"zarr"`` names a ``<stem>.ome.zarr`` store directory;
            ``"tiff"`` names ``<stem>.png`` for ``objmap`` and
            ``<stem>.tiff`` otherwise.

    Returns:
        The output path. Bounded by the 1-level dataset scanner (D12).
    """
    from phenotypic.sdk_ import STORE_SUFFIX, store_stem

    if fmt == "zarr":
        ext = STORE_SUFFIX
    else:
        ext = ".png" if layer == "objmap" else ".tiff"
    try:
        rel = image_path.relative_to(input_root)
    except ValueError:
        rel = Path(image_path.name)
    # `store_stem`, never `Path.stem`, on a store input: `.ome.zarr` is a
    # double suffix, so `.stem` yields `p01.ome` and a zarr run would write
    # `p01.ome.ome.zarr`. `store_stem` RAISES on a non-store path
    # (_io_constants.py:1554), so the suffix test is required, not defensive.
    stem = store_stem(rel) if rel.name.endswith(STORE_SUFFIX) else rel.stem
    return output_dir / rel.parent / f"{stem}{ext}"
```

`STORE_SUFFIX` and `store_stem` are both public re-exports of `phenotypic.sdk_`
(`sdk_/__init__.py`), which is how `_cli_directory_scanner.py:14-19` already
imports `STORE_SUFFIX`. Do not reach into `phenotypic.sdk_._io_constants`.

- [ ] **Step 5: Add the zarr branch to the writer**

```python
#: Layers that have a single-series OME-Zarr form. `objmap` and `detect_mat`
#: are absent for two different reasons, both spelled out in
#: :func:`resolve_process_format`'s refusals.
_ZARR_CAPABLE_LAYERS: frozenset[str] = frozenset({"rgb", "gray"})


def write_process_only_layer(
    image: Any,
    layer: ProcessOnlyLayer,
    out_path: Path,
    *,
    fmt: ProcessFormat = "tiff",
    commit_guard: CommitGuard | None = None,
) -> None:
    """Write one image layer, as a flat file or as a single-series store.

    The ``tiff`` branch delegates to the accessor's ``imsave`` through
    ``atomic_write_with_writer``, unchanged. The ``zarr`` branch delegates to
    ``Image._save_store``, whose ``.part``-then-rename promote is atomic by
    construction -- a store either has its root ``zarr.json`` or does not
    exist, so a kill mid-write cannot leave a truncated artifact at the final
    path.

    The store carries **only** *layer*: no objmap, no other series, and no
    ``image_class`` (which is what makes ``Image.load_zarr`` refuse it and
    point at ``Image.imread``).

    For ``objmap`` with no detected objects, emits the D9 warning and still
    writes the (all-zero) map; the run does not fail.
    """
    if fmt == "zarr":
        from phenotypic.sdk_ import ngff_

        if layer not in _ZARR_CAPABLE_LAYERS:
            # The CLI refuses this earlier and with a better message
            # (`resolve_process_format`). This guard exists because
            # `write_process_only_layer` is importable and called directly by
            # the staged strategy, and because `_save_store` would otherwise
            # fail for `detect_mat` with `no primary series among
            # ['detect_mat']` -- true, but about internal series naming rather
            # than about what the caller asked for.
            raise ValueError(
                f"layer {layer!r} has no single-series OME-Zarr form; write "
                f"it with fmt='tiff'"
            )
        height, width = image.gray[:].shape[:2]
        image._save_store(
            out_path,
            series=(layer,),
            write_objmap=False,
            levels=ngff_.pyramid_level_count(height, width),
            work_id=None,
            durable=None,
            commit_guard=commit_guard,
            write_image_class=False,
        )
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    accessor = getattr(image, layer)
    if layer == "objmap" and image.num_objects == 0:
        warnings.warn(
            f"pipeline produced no objects; writing empty object map to {out_path}"
        )
    atomic_write_with_writer(
        out_path,
        lambda temporary: accessor.imsave(filepath=Path(temporary)),
        commit_guard=commit_guard,
        temp_suffix=f".tmp{out_path.suffix}",
    )
```

`durable=None` lets `ngff_` auto-detect SLURM, matching the bundle writer.
`height, width` come from `image.gray[:]` rather than from the layer being
written because `gray` is present for every image, including one built from a
2-D array where `image.rgb[:]` raises `NoArrayError` — the same reasoning
`save_intermediate_zarr` records at `_image_io_handler.py:1234-1237`.

- [ ] **Step 6: Initialise provenance in the core**

In `process_single_apply_only_core`, add the parameter:

```python
def process_single_apply_only_core(
    pipeline_path: Path,
    image_path: Path,
    input_root: Path,
    output_dir: Path,
    image_type: ImageTypeName,
    layer: ProcessOnlyLayer,
    read_kwargs: Dict[str, Any],
    cli_nrows: Optional[int] = None,
    cli_ncols: Optional[int] = None,
    commit_guard: CommitGuard | None = None,
    process_format: ProcessFormat = "tiff",
) -> bool:
```

Inside the `try:`, between the `set_detect_mode` call (`:109-110`) and
`pipeline.apply(...)` (`:112`):

```python
        # BEFORE apply, and that ordering is load-bearing:
        # `initialize_cli_provenance` opens with `new_provenance_journal()`
        # (_provenance.py:294), which discards `operations[]`. Called after
        # apply it would erase the very records it exists to contextualise,
        # and the store would still carry a `pipeline` key -- so it would
        # look right.
        #
        # What it adds is the `pipeline` identity. The operations themselves
        # are recorded either way: `wrap_image_operation_apply` appends one
        # per operation unconditionally (_provenance.py:227).
        #
        # `basename_only` keeps the publishing artifact free of cluster
        # filesystem layout: a process-mode store goes to a NAS and then to
        # object storage, where an absolute path would carry the username and
        # project directory names. sha256 still pins the pipeline exactly.
        initialize_cli_provenance(image, pipeline_path, basename_only=True)
        pipeline.apply(image, inplace=True)
```

Import it at the top of the module with the absolute form, matching
`_cli_process_single.py:25-30`:

```python
from phenotypic._core._provenance import initialize_cli_provenance
```

Then thread the format into the two calls at the end (`:118-119`):

```python
    out_path = process_only_output_path(
        output_dir, image_path, input_root, layer, fmt=process_format
    )
    write_process_only_layer(
        image, layer, out_path, fmt=process_format, commit_guard=commit_guard
    )
    return True
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
uv run pytest tests/unit/cli/test_process_only_zarr.py -v
```

Expected: PASS (12 tests — 11 functions, one of them parametrised over two
layers). If the 12-file assertion fails, print the actual file list before
changing the assertion — an unexpected extra file means the writer emitted a
series or label it should not have, which is the bug the test is for. A likely
culprit is `_original`: `_write_store_part` appends an `"original"` series when
`self._original is not None` (`_image_io_handler.py:1014-1016`), and the
`SERIES` assertion in `test_writer_emits_only_the_requested_series` is what
catches it.

- [ ] **Step 8: Lint, type-check, commit**

```bash
uv run pytest tests/unit/cli/ -q
uv run mypy src/phenotypic/_cli/_cli_process_only.py
uv run ruff check --fix src/phenotypic/_cli/_cli_process_only.py \
    src/phenotypic/sdk_/typing_.py tests/unit/cli/test_process_only_zarr.py
git add src/phenotypic/_cli/_cli_process_only.py src/phenotypic/sdk_/typing_.py \
        tests/unit/cli/test_process_only_zarr.py
git commit -m "feat(cli): --mode process can write a single-series OME-Zarr store

write_process_only_layer gains a zarr branch delegating to _save_store with
series=(layer,), write_objmap=False, write_image_class=False -- so the store
carries only the requested layer and load_zarr refuses it. The .part-then-rename
promote is atomic by construction, so a kill mid-write cannot leave a truncated
artifact at the final path the way an in-place TIFF write can. Only rgb and gray
have a store form; the other two layers are refused here and, with better
messages, at the CLI.

process_only_output_path derives a store input's stem with store_stem rather
than Path.stem, which would yield p01.ome and write p01.ome.ome.zarr.

process_single_apply_only_core now calls initialize_cli_provenance BEFORE
pipeline.apply(), with basename_only=True. The ordering is load-bearing: the
call resets the journal. Operations were always recorded (the apply wrapper
appends unconditionally); what was missing on this path was the pipeline
identity -- source_path and sha256 -- which is what makes a fitted profile
traceable to a config file."
```

---

### Task 8a: `resolve_process_format` and the worker's `--process-format`

`_cli_process_single.py` is the **per-image SLURM worker** (`@click.command()`
at `:420`, function `main` at `:545`), not the command a user runs. It still
needs the option, because the SLURM array builds its command line
(`_cli_slurm_array_scripts.py:297-304`) and would otherwise run every worker at
the default. Task 8b wires the user-facing side.

**Files:**
- Modify: `src/phenotypic/_cli/_cli_process_only.py` (append
  `resolve_process_format` beside `_ZARR_CAPABLE_LAYERS`)
- Modify: `src/phenotypic/_cli/_cli_process_single.py` — option block beside
  `--layer` (`:510-516`), the `main` signature (`:545-571`), the validation
  block (`:588-596`), and the core call (`:681-692`)
- Test: `tests/unit/cli/test_process_format_option.py` (create)

**Interfaces:**
- Consumes: Task 7's `process_format` parameter and `_ZARR_CAPABLE_LAYERS`.
- Produces: `resolve_process_format(layer, requested) -> ProcessFormat`, a
  module-level function in `_cli_process_only.py` so the rule has one home.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/cli/test_process_format_option.py`:

```python
"""The format default is layer-dependent, and two layers have no zarr form."""

from __future__ import annotations

import click
import pytest

from phenotypic._cli._cli_process_only import resolve_process_format


@pytest.mark.parametrize("layer", ["rgb", "gray"])
def test_primary_series_layers_default_to_zarr(layer: str) -> None:
    assert resolve_process_format(layer, None) == "zarr"


def test_detect_mat_defaults_to_tiff() -> None:
    """It has no store form at all; the default cannot be zarr."""
    assert resolve_process_format("detect_mat", None) == "tiff"


def test_objmap_defaults_to_tiff() -> None:
    """A bare `--mode process --layer objmap` must keep working."""
    assert resolve_process_format("objmap", None) == "tiff"


@pytest.mark.parametrize("layer", ["rgb", "gray", "detect_mat", "objmap"])
def test_an_explicit_tiff_request_is_always_honoured(layer: str) -> None:
    assert resolve_process_format(layer, "tiff") == "tiff"


@pytest.mark.parametrize("layer", ["rgb", "gray"])
def test_explicit_zarr_is_honoured_for_a_primary_series(layer: str) -> None:
    assert resolve_process_format(layer, "zarr") == "zarr"


def test_explicit_zarr_for_objmap_is_refused() -> None:
    with pytest.raises(click.UsageError, match="objmap"):
        resolve_process_format("objmap", "zarr")


def test_explicit_zarr_for_detect_mat_is_refused() -> None:
    with pytest.raises(click.UsageError, match="detect_mat"):
        resolve_process_format("detect_mat", "zarr")


def test_the_two_refusals_give_different_reasons() -> None:
    """One is an NGFF rule; the other is ours. A user deserves to know which.

    objmap is refused because NGFF 0.5 2.6 says a labels group is nested
    inside an image group and is not itself an image -- a format rule, and
    unfixable here. detect_mat is refused because PhenoTypic's own writer
    requires a primary series (`ngff_.primary_series`, ngff_.py:459-474) -- our
    rule, and changeable in its own design. Collapsing the two into one message
    would tell a detect_mat user that NGFF forbids something it does not.
    """
    with pytest.raises(click.UsageError) as objmap:
        resolve_process_format("objmap", "zarr")
    with pytest.raises(click.UsageError) as detect_mat:
        resolve_process_format("detect_mat", "zarr")

    assert "labels group" in str(objmap.value)
    assert "labels group" not in str(detect_mat.value)
    assert "primary series" in str(detect_mat.value)
    assert "primary series" not in str(objmap.value)
    # Each names a remedy the user can actually type.
    assert "--process-format tiff" in str(objmap.value)
    assert "--process-format tiff" in str(detect_mat.value)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/cli/test_process_format_option.py -v
```

Expected: `ImportError: cannot import name 'resolve_process_format'`.

- [ ] **Step 3: Implement the resolution rule**

Append to `src/phenotypic/_cli/_cli_process_only.py`, after
`_ZARR_CAPABLE_LAYERS`:

```python
#: Why each non-store layer is refused. Two entries, two reasons, deliberately
#: not collapsed: `objmap` is refused by the FORMAT, `detect_mat` by US.
_NO_ZARR_FORM: dict[str, str] = {
    "objmap": (
        "--layer objmap has no single-series OME-Zarr form (NGFF 0.5 §2.6: "
        "a labels group is nested inside an image group and is not itself an "
        "image). Use --process-format tiff for the 16-bit raw-label PNG, or "
        "--layer rgb."
    ),
    "detect_mat": (
        "--layer detect_mat has no single-series OME-Zarr form: PhenoTypic's "
        "store writer requires a primary series (rgb or gray) and detect_mat "
        "is neither. Use --process-format tiff for the float TIFF, or "
        "--layer gray."
    ),
}


def resolve_process_format(
    layer: ProcessOnlyLayer, requested: ProcessFormat | None
) -> ProcessFormat:
    """Resolve ``--process-format``, whose default depends on ``--layer``.

    The default is not a single constant: ``rgb`` and ``gray`` default to
    ``zarr`` and ``detect_mat``/``objmap`` to ``tiff``, so every bare command
    keeps working and each layer gets the format that suits it. The rule lives
    here rather than in the option declaration so it has exactly one home --
    the user-facing CLI and the per-image worker both call it.

    The two refusals carry different reasons on purpose. ``objmap`` is refused
    by NGFF: 0.5 §2.6 nests a label image inside an image group and states
    that the labels group is not itself an image, so a standalone objmap store
    has no conformant single-series form. ``detect_mat`` is refused by
    PhenoTypic: ``_write_store_part`` calls ``ngff_.primary_series``
    unconditionally and that function accepts only ``rgb`` or ``gray``, so
    ``_save_store(series=("detect_mat",))`` raises ``no primary series among
    ['detect_mat']``. The first is a format rule and unfixable here; the second
    is ours, and widening ``primary_series`` is a change that belongs in its own
    design. A user reading the message deserves to know which they are hitting.

    Args:
        layer: The layer being exported.
        requested: The user's explicit ``--process-format``, or ``None``.

    Returns:
        The resolved format.

    Raises:
        click.UsageError: On an explicit ``zarr`` for a layer with no store
            form, naming the reason and the remedy.
    """
    if requested is None:
        return "zarr" if layer in _ZARR_CAPABLE_LAYERS else "tiff"
    if requested == "zarr" and layer not in _ZARR_CAPABLE_LAYERS:
        raise click.UsageError(_NO_ZARR_FORM[layer])
    return requested
```

Add `import click` and extend the existing `phenotypic.sdk_.typing_` import
(`_cli_process_only.py:17`) to bring in `ProcessFormat` beside
`ImageTypeName` and `ProcessOnlyLayer`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/cli/test_process_format_option.py -v
```

Expected: PASS (13 tests — 8 functions, two parametrised over two layers and
one over four).

- [ ] **Step 5: Wire the option into the worker**

In `src/phenotypic/_cli/_cli_process_single.py`, beside the `--layer` option
(`:510-516`):

```python
@click.option(
    "--process-format",
    "process_format",
    type=click.Choice(["tiff", "zarr"]),
    default=None,
    help=(
        "Output format for --mode process. Default: zarr for rgb/gray (a "
        "single-series OME-Zarr store), tiff for detect_mat (float TIFF) and "
        "objmap (a 16-bit raw-label PNG)."
    ),
)
```

Add `process_format: Optional[str]` to `main`'s signature beside
`layer: Optional[str]` (`:561`).

In the validation block (`:588-596`), extend the existing `--layer` guard:

```python
        process_only_layer: Optional[ProcessOnlyLayer] = None
        resolved_process_format: ProcessFormat = "tiff"
        if cli_mode == "process":
            if layer is None:
                raise click.UsageError("--mode process requires --layer")
            process_only_layer = cast(ProcessOnlyLayer, layer)
            resolved_process_format = resolve_process_format(
                process_only_layer,
                cast("ProcessFormat | None", process_format),
            )
        else:
            if layer is not None:
                raise click.UsageError(
                    "--layer can only be used with --mode process"
                )
            if process_format is not None:
                raise click.UsageError(
                    "--process-format can only be used with --mode process"
                )
```

The `elif` becomes `else` with two checks inside — write it as shown rather than
adding a second `elif`, so the two options are validated together.

At the core call (`:681-692`), pass the resolved value:

```python
            process_single_apply_only_core(
                pipeline_path=pipeline,
                ...
                commit_guard=commit_guard,
                process_format=resolved_process_format,
            )
```

Import `resolve_process_format` alongside the existing
`process_single_apply_only_core` import (`:32-35`).

- [ ] **Step 6: Write the worker CLI-surface test**

Append to `tests/unit/cli/test_process_format_option.py`:

```python
from click.testing import CliRunner

# `main`, not `process_single`: that name does not exist. The command object
# is `main`, declared `@click.command()` at _cli_process_single.py:420.
from phenotypic._cli._cli_process_single import main as process_single_worker


def test_the_worker_advertises_the_option() -> None:
    result = CliRunner().invoke(process_single_worker, ["--help"])
    assert result.exit_code == 0
    assert "--process-format" in result.output
    assert "zarr for rgb/gray" in result.output   # the default is stated
```

Do **not** try to exercise the worker's validation by invoking it bare: it has
four `required=True` options (`--pipeline` `:424`, `--image` `:430`,
`--output-dir` `:436`, `--dataset-name` `:441`), two of which are
`click.Path(exists=True)`, so a bare `invoke` exits 2 on `Missing option
'--pipeline'` and never reaches the block under test. The user-facing
rejection is tested in Task 8b against the command a user actually runs.

- [ ] **Step 7: Run the CLI suite**

```bash
uv run pytest tests/unit/cli/ -q
uv run mypy src/phenotypic/_cli/_cli_process_single.py \
            src/phenotypic/_cli/_cli_process_only.py
uv run ruff check --fix src/phenotypic/_cli/_cli_process_single.py \
    src/phenotypic/_cli/_cli_process_only.py \
    tests/unit/cli/test_process_format_option.py
```

Expected: PASS (14 in this file).

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/_cli/_cli_process_single.py \
        src/phenotypic/_cli/_cli_process_only.py \
        tests/unit/cli/test_process_format_option.py
git commit -m "feat(cli): resolve_process_format, and the worker's --process-format

rgb and gray default to zarr; detect_mat and objmap default to tiff, so a bare
--mode process --layer objmap keeps working. An explicit zarr on either is a
UsageError, and the two carry different reasons: NGFF 2.6 nests a label image
inside an image group and states the labels group is not itself an image, while
detect_mat is refused by PhenoTypic's own writer, which requires a primary
series (rgb or gray). The first is a format rule, the second is ours, and a
user reading the message deserves to know which.

The rule lives in one function rather than in the option declaration, because
both the worker and the user-facing CLI need it. --ext is left untouched:
process mode already ignores it, so wiring it in would change tiff/png naming
as a side effect of a zarr change."
```

---

### Task 8b: `--process-format` on the user-facing CLI

**The option does not reach a user until this task lands.** `python -m
phenotypic` is `phenotypicCLI.py`: it declares its own `--layer` (`:1234-1243`),
validates it (`:1331-1339`), and builds an `ExecutionConfig` (`:1663`). Task 8a
wired only the per-image worker, so `--mode process --process-format zarr` is
still an unknown-option error, and every `process_only_output_path` call outside
`_cli_process_only.py` still computes a `.tiff` or `.png` path.

**Missing one call site is silent, not loud.** The parameter defaults to
`"tiff"`, so a site that does not pass the format simply computes the wrong
path: continuation hunts for a file that was never written, and every image
reprocesses forever. There are seven such sites, all verified by
`grep -rn process_only_output_path src/`:

| Site | Role | Layer |
|---|---|---|
| `phenotypicCLI.py:450` | legacy-marker promotion artifact | `config.process_only_layer` |
| `phenotypicCLI.py:954` | dry-run sample paths | `config.process_only_layer` |
| `_cli_execution_strategies.py:159` | local completion marker | `config.process_only_layer` |
| `_cli_process_single.py:721` | the worker's artifact publication | `process_only_layer` |
| `_cli_staged_strategy.py:128` | staged terminal check | literal `"objmap"` |
| `_cli_staged_strategy.py:402` | staged objmap export | literal `"objmap"` |
| `_cli_staged_resume.py:220` | staged resume terminal check | literal `"objmap"` |

The three staged sites pass `"objmap"` as a literal, which always resolves to
`tiff`, so they are safe today — but they pass `fmt="tiff"` **explicitly**
anyway, so that a future change to the default cannot silently move them.

**Files:**
- Modify: `src/phenotypic/phenotypicCLI.py` — `--layer` help (`:1239-1242`), new
  option beside it (`:1243`), `phenotypic_cli` signature (`:1245+`), validation
  (`:1331-1339`), `ExecutionConfig(...)` (`:1663`), and the two call sites
  (`:450-455`, `:954-959`)
- Modify: `src/phenotypic/_cli/_cli_types.py:185` (`ExecutionConfig`)
- Modify: `src/phenotypic/_cli/_cli_execution_strategies.py:159-164` and
  `:578-588`
- Modify: `src/phenotypic/_cli/_cli_staged_strategy.py:128-130`, `:402-404`
- Modify: `src/phenotypic/_cli/_cli_staged_resume.py:220-222`
- Modify: `src/phenotypic/_cli/_cli_slurm_array_scripts.py:297-304`
- Modify: `src/phenotypic/_cli/_cli_failure_tracker.py:101-156`
- Modify: `src/phenotypic/_cli/_cli_process_single.py:143-155`
  (`_worker_work_identity`'s digest call)
- Modify: `src/phenotypic/_cli/_cli_state_management.py:200-223`, `:309-316`
- Test: `tests/unit/cli/test_process_format_cli.py` (create)

**Interfaces:**
- Consumes: Task 8a's `resolve_process_format`.
- Produces: `ExecutionConfig.process_format: ProcessFormat = "tiff"`;
  `processing_configuration_digest_from_values(…, process_format: str)` as a
  **required** keyword.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/cli/test_process_format_cli.py`:

```python
"""--process-format reaches the command a user actually runs."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from phenotypic import Image, ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic._cli._cli_failure_tracker import (
    processing_configuration_digest_from_values,
)
from phenotypic._cli._cli_types import ExecutionConfig
from phenotypic.phenotypicCLI import phenotypic_cli


def _digest(**overrides) -> str:
    base = dict(
        image_type="Image",
        nrows=None,
        ncols=None,
        bit_depth=16,
        detect_mode="gray",
        process_only_layer="rgb",
        ext="tiff",
        process_format="tiff",
        include_dataset_column=True,
        overlay_alpha=0.3,
        save_overlays=True,
    )
    base.update(overrides)
    return processing_configuration_digest_from_values(**base)


def test_the_option_exists_and_states_its_default() -> None:
    result = CliRunner().invoke(phenotypic_cli, ["--help"])
    assert result.exit_code == 0
    assert "--process-format" in result.output
    assert "zarr for rgb/gray" in result.output


def test_the_layer_help_no_longer_claims_tiff_for_everything() -> None:
    """It said "TIFF for rgb/gray/detect_mat". That stops being true here."""
    result = CliRunner().invoke(phenotypic_cli, ["--help"])
    assert "TIFF for rgb/gray/detect_mat" not in result.output


@pytest.fixture
def run_inputs(tmp_path: Path) -> tuple[Path, Path]:
    pipeline = tmp_path / "p.json.pht-pipe"
    ImagePipeline().to_json(pipeline)
    images = tmp_path / "in"
    images.mkdir()
    Image(load_synth_yeast_plate()).rgb.imsave(filepath=images / "p01.tiff")
    return pipeline, images


def test_process_format_is_rejected_outside_process_mode(
    tmp_path: Path, run_inputs: tuple[Path, Path]
) -> None:
    """Mirrors how --layer already behaves (phenotypicCLI.py:1336-1339)."""
    pipeline, images = run_inputs
    result = CliRunner().invoke(
        phenotypic_cli,
        ["--pipeline", str(pipeline), "--input", str(images),
         "--output", str(tmp_path / "out"), "--mode", "full",
         "--process-format", "zarr"],
    )
    assert result.exit_code != 0
    assert "--process-format" in result.output


def test_an_impossible_layer_and_format_pair_is_refused(
    tmp_path: Path, run_inputs: tuple[Path, Path]
) -> None:
    pipeline, images = run_inputs
    result = CliRunner().invoke(
        phenotypic_cli,
        ["--pipeline", str(pipeline), "--input", str(images),
         "--output", str(tmp_path / "out"), "--mode", "process",
         "--layer", "objmap", "--process-format", "zarr"],
    )
    assert result.exit_code != 0
    assert "labels group" in result.output


def test_the_config_carries_a_resolved_format_not_none() -> None:
    """`ExecutionConfig` never holds the raw option; it holds the answer."""
    assert ExecutionConfig.__dataclass_fields__["process_format"].default == "tiff"


def test_the_format_joins_the_continuation_identity() -> None:
    """Switching format must invalidate continuation, not reuse the other kind."""
    assert _digest(process_format="tiff") != _digest(process_format="zarr")


def test_the_format_does_not_disturb_a_non_process_run(tmp_path: Path) -> None:
    """A full run's digest must not change, or every existing run resumes cold.

    `process_format` joins the payload only inside the process-only branch
    (_cli_failure_tracker.py:124-130), beside `ext`, exactly as
    `process_only_layer` does.
    """
    full = dict(
        image_type="Image", nrows=None, ncols=None, bit_depth=16,
        detect_mode="gray", process_only_layer=None, ext="tiff",
        include_dataset_column=True, overlay_alpha=0.3, save_overlays=True,
    )
    assert processing_configuration_digest_from_values(
        **full, process_format="tiff"
    ) == processing_configuration_digest_from_values(
        **full, process_format="zarr"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/cli/test_process_format_cli.py -v
```

Expected: `Error: No such option: --process-format` on the CLI tests,
`TypeError: … unexpected keyword argument 'process_format'` on the digest
tests, and `KeyError: 'process_format'` on the config test.

- [ ] **Step 3: Carry the format on `ExecutionConfig`**

In `src/phenotypic/_cli/_cli_types.py`, beside `process_only_layer` (`:183-185`):

```python
    # Process-only mode: run pipeline.apply() and export a single image layer
    # (no measurement / analysis output). None = normal forward/measure run.
    process_only_layer: Optional[ProcessOnlyLayer] = None

    # Resolved --process-format for a process run. Always the ANSWER, never
    # the raw option: `resolve_process_format` runs once at the CLI boundary,
    # so nothing downstream re-derives a default and drifts from it. Ignored
    # outside process mode, where the CLI rejects the option outright.
    process_format: ProcessFormat = "tiff"
```

Import `ProcessFormat` beside the existing `ProcessOnlyLayer` import.

- [ ] **Step 4: Declare, resolve, and store the option**

In `src/phenotypic/phenotypicCLI.py`, first correct the `--layer` help
(`:1239-1242`), which currently asserts a format this change falsifies:

```python
@click.option(
    "--layer",
    "layer",
    type=click.Choice(["rgb", "gray", "detect_mat", "objmap"]),
    default=None,
    help=(
        "Layer exported by --mode process. See --process-format for the "
        "output format and its per-layer default."
    ),
)
@click.option(
    "--process-format",
    "process_format",
    type=click.Choice(["tiff", "zarr"]),
    default=None,
    help=(
        "Output format for --mode process. Default: zarr for rgb/gray (a "
        "single-series OME-Zarr store carrying the pipeline that produced "
        "it), tiff for detect_mat (float TIFF) and objmap (a 16-bit "
        "raw-label PNG). --layer objmap and --layer detect_mat have no "
        "OME-Zarr form and refuse an explicit zarr."
    ),
)
```

Add `process_format: Optional[str]` to `phenotypic_cli`'s signature beside
`layer: Optional[str]`.

Extend the validation block (`:1331-1339`):

```python
        process_only_layer: Optional[ProcessOnlyLayer] = None
        resolved_process_format: ProcessFormat = "tiff"
        if cli_mode == "process":
            if layer is None:
                raise click.UsageError("--mode process requires --layer.")
            process_only_layer = cast(ProcessOnlyLayer, layer)
            resolved_process_format = resolve_process_format(
                process_only_layer,
                cast("ProcessFormat | None", process_format),
            )
        else:
            if layer is not None:
                raise click.UsageError(
                    "--layer can only be used with --mode process."
                )
            if process_format is not None:
                raise click.UsageError(
                    "--process-format can only be used with --mode process."
                )
```

Import `resolve_process_format` from `phenotypic._cli._cli_process_only`, and
`ProcessFormat` from `phenotypic.sdk_.typing_`, beside the existing
`ProcessOnlyLayer` import.

Pass it into the `ExecutionConfig(...)` construction, beside
`process_only_layer=process_only_layer` (`:1663`):

```python
            process_only_layer=process_only_layer,  # type: ignore[arg-type]
            process_format=resolved_process_format,
```

- [ ] **Step 5: Thread the format to all seven output-path sites**

Each site gains `fmt=`. The four that already read the layer from config read
the format from config too:

`phenotypicCLI.py:450-455`:

```python
                artifacts = {
                    "process_output": process_only_output_path(
                        output_dir,
                        image,
                        config.input_path,
                        config.process_only_layer,
                        fmt=config.process_format,
                    )
                }
```

`phenotypicCLI.py:954-959`:

```python
            sample_paths.append(
                process_only_output_path(
                    output_dir,
                    img,
                    config.input_path,
                    layer,  # type: ignore[arg-type]
                    fmt=config.process_format,
                )
            )
```

`_cli_execution_strategies.py:159-164`:

```python
        artifacts = {
            "process_output": process_only_output_path(
                output_dir,
                image_path,
                config.input_path,
                config.process_only_layer,
                fmt=config.process_format,
            )
        }
```

`_cli_process_single.py:721-727` — the worker has no `config`, so it uses the
value it resolved in Task 8a:

```python
                artifacts={
                    "process_output": process_only_output_path(
                        output_dir,
                        image,
                        input_root,
                        process_only_layer,
                        fmt=resolved_process_format,
                    )
                },
```

The three staged sites pass `fmt="tiff"` explicitly beside their `"objmap"`
literal (`_cli_staged_strategy.py:128-130` and `:402-404`,
`_cli_staged_resume.py:220-222`), e.g.:

```python
                return process_only_output_path(
                    output_dir, img, cfg.input_path, "objmap", fmt="tiff"
                ).is_file()
```

They are objmap-only, so `tiff` is the only format they can ever want; saying so
explicitly is what stops a future default change from silently relocating a
terminal-check path.

Finally, the local strategy's core call
(`_cli_execution_strategies.py:578-588`):

```python
            process_single_apply_only_core(
                pipeline_path=self.config.pipeline_json,
                image_path=image_path,
                input_root=self.config.input_path,
                output_dir=output_dir,
                image_type=self.config.image_type,
                layer=self.config.process_only_layer,  # type: ignore[arg-type]
                read_kwargs=read_kwargs,
                cli_nrows=self.config.nrows,
                cli_ncols=self.config.ncols,
                process_format=self.config.process_format,
            )
```

**Nothing here is a store-descriptor problem.** `_artifact_descriptor`
(`_cli_completion.py:61-94`) already branches on `resolved.is_dir()` and
describes a store by the SHA-256 of its root `zarr.json` rather than by size, so
`publish_image_success` handles a store `process_output` artifact with no
change. Verify this by reading it rather than assuming; do not add a second
descriptor path.

- [ ] **Step 6: Put the format on the worker command line**

In `_cli_slurm_array_scripts.py:297-304`:

```python
    if config.process_only_layer:
        _set_worker_mode(cmd_parts, "process")
        cmd_parts.extend(
            [
                "--layer",
                config.process_only_layer,
                "--process-format",
                config.process_format,
            ]
        )
```

Without this every SLURM worker resolves its own default, which is right today
by coincidence and wrong the moment a user passes `--process-format tiff` for
`rgb`.

- [ ] **Step 7: Join the continuation identity**

In `_cli_failure_tracker.py`, add `process_format: str` to
`processing_configuration_digest_from_values`'s keyword-only signature
(`:101-114`) — **required, with no default.** A default would let a missed
caller compute the old digest silently, which is the whole failure mode this
guards against; there are exactly two production callers and no test callers
(verified by grep), so a `TypeError` at the third is the correct outcome.

Add it to the payload **inside the process-only branch** (`:124-130`), beside
`ext`:

```python
    if process_only_layer is not None:
        payload.update(
            {
                "process_only_layer": process_only_layer,
                "ext": ext,
                # Beside `ext` and NOT in the base payload: a full or measure
                # run has no process format, and folding it into the base
                # would change every existing run's digest and cold-start
                # every continuation in flight.
                "process_format": process_format,
            }
        )
```

Update both callers:

- `processing_configuration_digest(config)` (`:142-156`) adds
  `process_format=config.process_format,`.
- `_worker_work_identity`'s call in `_cli_process_single.py:143-155` adds
  `process_format=resolved_process_format if mode == "process" else "tiff",`
  — mirroring the `process_only_layer=layer if mode == "process" else None`
  line directly above it, so the two stay consistent.

In `_cli_state_management.py`, record the format in the saved config (`:200-223`,
beside `"process_only_layer"` at `:210`):

```python
            "process_only_layer": config.process_only_layer,
            "process_format": config.process_format,
```

and add `"process_format"` to the compatibility loop's key tuple (`:309-316`):

```python
    for key in (
        "bit_depth",
        "detect_mode",
        "include_dataset_column",
        "overlay_alpha",
        "drop_originals",
        "save_overlays",
        "process_format",
    ):
```

That loop already does `if key not in state.config: continue` (`:317-318`), so a
state file written before this change stays compatible instead of failing to
resume — which is the right behaviour, since such a run was necessarily a TIFF
run and `process_format` defaults to `"tiff"`.

- [ ] **Step 8: Run tests to verify they pass**

```bash
uv run pytest tests/unit/cli/test_process_format_cli.py -v
```

Expected: PASS (7 tests).

- [ ] **Step 9: Run the whole CLI suite and type-check**

```bash
uv run pytest tests/unit/cli/ -q
uv run mypy src/phenotypic/_cli src/phenotypic/phenotypicCLI.py
uv run ruff check --fix src/phenotypic/phenotypicCLI.py \
    src/phenotypic/_cli/_cli_types.py \
    src/phenotypic/_cli/_cli_execution_strategies.py \
    src/phenotypic/_cli/_cli_staged_strategy.py \
    src/phenotypic/_cli/_cli_staged_resume.py \
    src/phenotypic/_cli/_cli_slurm_array_scripts.py \
    src/phenotypic/_cli/_cli_failure_tracker.py \
    src/phenotypic/_cli/_cli_process_single.py \
    src/phenotypic/_cli/_cli_state_management.py \
    tests/unit/cli/test_process_format_cli.py
```

Expected: PASS. A failing continuation test is the signal to check that
`process_format` landed in the process-only digest branch and **not** the base
payload.

- [ ] **Step 10: Commit**

```bash
git add src/phenotypic/phenotypicCLI.py src/phenotypic/_cli/ \
        tests/unit/cli/test_process_format_cli.py
git commit -m "feat(cli): --process-format reaches the user-facing command

_cli_process_single.py is the per-image SLURM worker; python -m phenotypic is
phenotypicCLI.py, which declares its own --layer and builds the ExecutionConfig.
Wiring only the worker left --mode process --process-format zarr an
unknown-option error.

phenotypicCLI declares the option, resolves it once beside the --layer guard,
and carries the ANSWER on ExecutionConfig so nothing downstream re-derives a
default. All seven process_only_output_path call sites now pass the format --
missing one is silent, because the parameter defaults to tiff and continuation
would simply hunt for a file that was never written. The three staged sites are
objmap-only and pass fmt=\"tiff\" explicitly so a future default change cannot
move them.

The format joins the SLURM worker command line and the continuation identity:
processing_configuration_digest_from_values takes it as a REQUIRED keyword and
folds it into the process-only branch beside ext, so switching format
invalidates continuation while a full run's digest is untouched."
```

---

### Task 9: consolidated metadata on a process-mode store

Task 1 added `consolidate` to `_save_store` / `_write_store_part`, applied to
the `.part` immediately before `promote_store`. This task switches it on for the
process writer and pins what it must not break.

Opening a store costs one GET per metadata file — 8 of the 12 — which is the
latency that matters once the destination is object storage and a viewer
enumerates many stores. `zarr.consolidate_metadata` collapses that to one.

**It is legal**, and the reason is precise. The Zarr v3 core spec: *"An
implementation MUST fail to open Zarr groups or arrays if any metadata fields
are present which (a) the implementation does not recognize and (b) are not
explicitly set to `"must_understand": false`."* zarr-python serialises
`consolidated_metadata` with `must_understand: false`, so a conformant reader
that does not recognise the key is **required to ignore it**.

Verified properties, all by execution against zarr 3.1.5: the key is a
**top-level sibling of `attributes`** (so `attributes.phenotypic` survives
untouched and `read_root_attributes` needs no change); it **adds no files**; and
per-node `zarr.json` documents all remain, so a reader that ignores it still
walks the tree. A process-mode store is written once into a `.part` and promoted
by rename, never mutated, so consolidation cannot go stale.

**Files:**
- Modify: `src/phenotypic/_cli/_cli_process_only.py` (`write_process_only_layer`
  — one keyword)
- Test: `tests/unit/cli/test_process_only_consolidated.py` (create)

**Interfaces:**
- Consumes: Task 1's `consolidate` parameter and Task 7's zarr branch.
- Produces: nothing new; the store gains a root-level `consolidated_metadata`
  key.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/cli/test_process_only_consolidated.py`:

```python
"""Consolidated metadata: one GET to open a store, and safely ignorable."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_ import ngff_
from phenotypic.sdk_.ngff_ import PhenotypicAttr
from phenotypic._cli._cli_process_only import write_process_only_layer


def _store(tmp_path: Path) -> Path:
    img = Image(load_synth_yeast_plate())
    out = tmp_path / "p01.ome.zarr"
    write_process_only_layer(img, "rgb", out, fmt="zarr")
    return out


def _root(store: Path) -> dict:
    return json.loads((store / "zarr.json").read_text(encoding="utf-8"))


def test_the_store_is_consolidated(tmp_path: Path) -> None:
    root = _root(_store(tmp_path))
    assert "consolidated_metadata" in root
    assert root["consolidated_metadata"]["metadata"]


def test_consolidation_is_marked_safely_ignorable(tmp_path: Path) -> None:
    """Zarr v3 requires readers to FAIL on an unknown key without this."""
    root = _root(_store(tmp_path))
    assert root["consolidated_metadata"]["must_understand"] is False


def test_the_phenotypic_block_survives_consolidation(tmp_path: Path) -> None:
    """It is a sibling of `attributes`, not nested inside it."""
    store = _store(tmp_path)
    root = _root(store)
    assert "consolidated_metadata" not in root["attributes"]
    block = root["attributes"][PhenotypicAttr.ROOT]
    assert block[PhenotypicAttr.STORE_SCHEMA_VERSION] == ngff_.STORE_SCHEMA_VERSION
    assert block[PhenotypicAttr.SERIES] == {"rgb": "rgb"}
    # And the existing reader still finds it unchanged.
    assert ngff_.read_root_attributes(store)[PhenotypicAttr.ROOT] == block


def test_per_node_metadata_still_exists(tmp_path: Path) -> None:
    """A reader that ignores the key must still be able to walk the tree."""
    store = _store(tmp_path)
    assert (store / "rgb" / "zarr.json").is_file()
    assert (store / "rgb" / "0" / "zarr.json").is_file()
    assert (store / "OME" / "zarr.json").is_file()


def test_consolidation_adds_no_files(tmp_path: Path) -> None:
    """The 12-file claim survives it. Same count, one fewer round trip."""
    img = Image(load_synth_yeast_plate())
    levels = ngff_.pyramid_level_count(*img.rgb[:].shape[:2])
    store = _store(tmp_path)
    assert len([p for p in store.rglob("*") if p.is_file()]) == 4 + 2 * levels


def test_a_consolidated_store_still_round_trips_through_imread(
    tmp_path: Path,
) -> None:
    img = Image(load_synth_yeast_plate())
    out = tmp_path / "p01.ome.zarr"
    write_process_only_layer(img, "rgb", out, fmt="zarr")
    assert np.array_equal(Image.imread(out).rgb[:], img.rgb[:])


def test_no_warning_escapes_the_writer(tmp_path: Path, recwarn) -> None:
    """TWO ZarrUserWarnings fire, not one, and both must be caught.

    zarr 3.1.5 emits `Consolidated metadata is currently not part in the Zarr
    format 3 specification` AND `Object at METADATA.ome.xml is not recognized
    as a component of a Zarr hierarchy` -- the latter once per image, which at
    AutoConvertRaw scale is tens of thousands of lines of log. A `message=`
    filter naming only consolidation catches one of them; filtering on the
    ZarrUserWarning class catches both.
    """
    from zarr.errors import ZarrUserWarning

    _store(tmp_path)
    assert [w for w in recwarn if issubclass(w.category, ZarrUserWarning)] == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/cli/test_process_only_consolidated.py -v
```

Expected: `KeyError: 'consolidated_metadata'` on the first two. The rest PASS
already — they are the regression half, pinning what consolidation must *not*
break.

- [ ] **Step 3: Switch it on in the zarr branch**

In `write_process_only_layer`, add one keyword to the `_save_store` call:

```python
        image._save_store(
            out_path,
            series=(layer,),
            write_objmap=False,
            levels=ngff_.pyramid_level_count(height, width),
            work_id=None,
            durable=None,
            commit_guard=commit_guard,
            write_image_class=False,
            # Consolidated INSIDE the .part, before the promote -- see
            # `_consolidate_store_part`. A process-mode store is written once
            # and never mutated, so the consolidated view cannot drift from
            # the tree it describes; do not lift this onto a store that is
            # rewritten in place.
            consolidate=True,
        )
```

There is no `_consolidate_published_store` helper and no second
`zarr.consolidate_metadata` call site: an earlier draft consolidated
`_save_store`'s **return value**, which rewrites the root `zarr.json` at the
final path and reintroduces the truncated-artifact failure the rename-commit
exists to prevent. Task 1 put the call in the only place it is safe.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/cli/test_process_only_consolidated.py -v
```

Expected: PASS (7 tests).

- [ ] **Step 5: Re-run the conformance gate**

```bash
uv run pytest tests/unit/sdk_/test_ngff_validity.py \
              tests/unit/test_ome_zarr_invariants.py \
              tests/unit/cli/ -q
```

Expected: PASS. If a schema validator rejects the root document, read the error:
the NGFF schemas validate `attributes.ome`, and `consolidated_metadata` is a
sibling of `attributes`, so it should be outside their scope. If the gate walks
the whole document strictly, that is worth reporting rather than working around.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_cli/_cli_process_only.py \
        tests/unit/cli/test_process_only_consolidated.py
git commit -m "feat(cli): consolidate metadata on a process-mode store

Opening a store costs one GET per metadata file -- 8 of 12 -- which is the
latency that matters on object storage. Consolidation collapses that to one and
adds no files.

Legal under the Zarr v3 extension mechanism, not in spite of it: the core spec
requires readers to fail on an unrecognised metadata field unless it carries
must_understand: false, and zarr-python writes exactly that. Per-node zarr.json
documents all remain, so a reader that ignores the key still walks the tree.
Safe here because a process-mode store is written once and promoted by rename,
so the consolidated view cannot go stale.

One keyword: the call itself lives in _write_store_part, applied to the .part
before promote_store, because consolidating the promoted path would rewrite its
root zarr.json in place."
```
