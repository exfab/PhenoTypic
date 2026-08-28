# Phase 3 — CLI process mode

The writer that produces the artifact, and the option that selects it.
**Depends on Tasks 1 and 4** (`write_image_class`, `basename_only`).

Read [`README.md`](README.md)'s **Global Constraints** first.

---

### Task 7: process-only zarr writer + provenance init

Two things land together because neither is testable without the other: the
zarr branch of the writer, and the `initialize_cli_provenance` call that gives
the store something to record. Verified today: `_cli_process_only.py` never
touches provenance, so a store written without this carries the empty journal
`new_provenance_journal()` returns.

**Files:**
- Modify: `src/phenotypic/_cli/_cli_process_only.py` (whole file, 120 lines)
- Test: `tests/unit/cli/test_process_only_zarr.py` (create)

**Interfaces:**
- Consumes: `Image._save_store(…, write_image_class=False)` (Task 1),
  `initialize_cli_provenance(…, basename_only=True)` (Task 4),
  `ngff_.pyramid_level_count`, `ngff_.STORE_SUFFIX`.
- Produces:
  - `ProcessFormat = Literal["tiff", "zarr"]` in `sdk_/typing_.py`
  - `process_only_output_path(output_dir, image_path, input_root, layer, fmt="tiff")`
  - `write_process_only_layer(image, layer, out_path, *, fmt="tiff", commit_guard=None)`
  - `process_single_apply_only_core(…, process_format: ProcessFormat = "tiff")`

  The two parameter names are deliberate and not a slip: the leaf helpers take
  `fmt`, the core takes `process_format` to mirror the CLI option it is fed
  from. Do not "unify" them.

- [ ] **Step 1: Add the format type**

In `src/phenotypic/sdk_/typing_.py`, beside `ProcessOnlyLayer` (`:97`):

```python
#: Output format for ``--mode process``. ``zarr`` writes a single-series
#: OME-Zarr store; ``tiff`` writes the flat image file (a 16-bit PNG for
#: ``objmap``). ``objmap`` has no OME-Zarr form -- see the CLI guard.
ProcessFormat = Literal["tiff", "zarr"]
```

- [ ] **Step 2: Write the failing tests**

Create `tests/unit/cli/test_process_only_zarr.py`:

```python
"""--mode process writes a single-series store carrying its own provenance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
    nested = tmp_path / "config" / "deep"
    nested.mkdir(parents=True)
    path = nested / "preprocess.json"
    ImagePipeline().to_json(path)
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


def test_a_single_series_rgb_store_is_twelve_files(tmp_path: Path) -> None:
    """Spec 1.1. Guards against an accidental extra series or level."""
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
    assert journal["pipeline"]["source_path"] == "preprocess.json"
    assert "/" not in journal["pipeline"]["source_path"]
    assert len(journal["pipeline"]["sha256"]) == 64


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
    """The default path for objmap, and the ACR contract, must not move."""
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

- [ ] **Step 4: Make the output path format-aware**

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
        image_path: The input image.
        input_root: The ``--input`` root the mirror is relative to.
        layer: The layer being exported.
        fmt: ``"zarr"`` names a ``<stem>.ome.zarr`` store directory;
            ``"tiff"`` names ``<stem>.png`` for ``objmap`` and
            ``<stem>.tiff`` otherwise.

    Returns:
        The output path. Bounded by the 1-level dataset scanner (D12).
    """
    from phenotypic.sdk_ import ngff_

    if fmt == "zarr":
        ext = ngff_.STORE_SUFFIX
    else:
        ext = ".png" if layer == "objmap" else ".tiff"
    try:
        rel = image_path.relative_to(input_root)
    except ValueError:
        rel = Path(image_path.name)
    return output_dir / rel.parent / f"{rel.stem}{ext}"
```

- [ ] **Step 5: Add the zarr branch to the writer**

```python
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
    writes the (all-zero) map; the run does not fail. ``objmap`` has no
    OME-Zarr form and the CLI refuses that combination before reaching here.
    """
    if fmt == "zarr":
        from phenotypic.sdk_ import ngff_

        if layer == "objmap":
            raise ValueError(
                "objmap has no single-series OME-Zarr form; write it as a "
                "16-bit PNG with fmt='tiff'"
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

- [ ] **Step 6: Initialise provenance in the core**

In `process_single_apply_only_core`, add the parameter and the call:

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

Inside the `try:`, immediately after `image = image_cls.imread(image_path, **read_kwargs)`
and the `set_detect_mode` call, before `pipeline.apply(...)`:

```python
        # Record the operations as they run. `basename_only` keeps the
        # publishing artifact free of cluster filesystem layout: a process-mode
        # store goes to a NAS and then to object storage, where an absolute
        # path would carry the username and project directory names. sha256
        # still pins the pipeline exactly.
        initialize_cli_provenance(
            image, pipeline_path, basename_only=True
        )
        pipeline.apply(image, inplace=True)
```

Import it at the top: `from .._core._provenance import initialize_cli_provenance`
— check how `_cli_process_single.py:26` imports it and match that path exactly.

Then thread the format into the two calls at the end:

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

Expected: PASS (8 tests). If the 12-file assertion fails, print the actual file
list before changing the assertion — an unexpected extra file means the writer
emitted a series or label it should not have, which is the bug the test is for.

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
artifact at the final path the way an in-place TIFF write can.

process_single_apply_only_core now calls initialize_cli_provenance before
pipeline.apply(), with basename_only=True: the store is published to a NAS and
then to object storage, where an absolute pipeline path would carry cluster
filesystem layout. Previously this path never touched provenance at all, so a
store would have carried an empty journal."
```

---

### Task 8: `--process-format`, layer-dependent default, objmap guard

**Files:**
- Modify: `src/phenotypic/_cli/_cli_process_single.py` — option block near
  `:473-517`, validation near `:588-596`, call site near `:681-691`
- Test: `tests/unit/cli/test_process_format_option.py` (create)

**Interfaces:**
- Consumes: Task 7's `process_format` parameter.
- Produces: `resolve_process_format(layer, requested) -> ProcessFormat`, a
  module-level function in `_cli_process_only.py` so the rule has one home.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/cli/test_process_format_option.py`:

```python
"""The format default is layer-dependent, and objmap has no zarr form."""

from __future__ import annotations

import click
import pytest

from phenotypic._cli._cli_process_only import resolve_process_format


@pytest.mark.parametrize("layer", ["rgb", "gray", "detect_mat"])
def test_image_layers_default_to_zarr(layer: str) -> None:
    assert resolve_process_format(layer, None) == "zarr"


def test_objmap_defaults_to_tiff(layer: str = "objmap") -> None:
    """A bare `--mode process --layer objmap` must keep working."""
    assert resolve_process_format(layer, None) == "tiff"


@pytest.mark.parametrize("layer", ["rgb", "gray", "detect_mat", "objmap"])
def test_an_explicit_tiff_request_is_always_honoured(layer: str) -> None:
    assert resolve_process_format(layer, "tiff") == "tiff"


def test_explicit_zarr_for_objmap_is_refused(layer: str = "objmap") -> None:
    with pytest.raises(click.UsageError, match="objmap"):
        resolve_process_format(layer, "zarr")


def test_the_refusal_names_the_remedy() -> None:
    with pytest.raises(click.UsageError, match="tiff"):
        resolve_process_format("objmap", "zarr")


@pytest.mark.parametrize("layer", ["rgb", "gray", "detect_mat"])
def test_explicit_zarr_is_honoured_elsewhere(layer: str) -> None:
    assert resolve_process_format(layer, "zarr") == "zarr"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/cli/test_process_format_option.py -v
```

Expected: `ImportError: cannot import name 'resolve_process_format'`.

- [ ] **Step 3: Implement the resolution rule**

Append to `src/phenotypic/_cli/_cli_process_only.py`:

```python
#: Layers whose natural process-mode output is an OME-Zarr store. ``objmap`` is
#: absent deliberately: NGFF 0.5 2.6 nests a label image inside an image group
#: and states that the labels group "is not itself an image", so a standalone
#: objmap store has no conformant single-series form.
_ZARR_DEFAULT_LAYERS: frozenset[str] = frozenset({"rgb", "gray", "detect_mat"})


def resolve_process_format(
    layer: ProcessOnlyLayer, requested: ProcessFormat | None
) -> ProcessFormat:
    """Resolve ``--process-format``, whose default depends on ``--layer``.

    The default is not a single constant: ``rgb``/``gray``/``detect_mat``
    default to ``zarr`` and ``objmap`` to ``tiff``, so every bare command keeps
    working and each layer gets the format that suits it. The rule lives here
    rather than in the option declaration so it has exactly one home.

    Args:
        layer: The layer being exported.
        requested: The user's explicit ``--process-format``, or ``None``.

    Returns:
        The resolved format.

    Raises:
        click.UsageError: On an explicit ``--layer objmap --process-format
            zarr``, naming the remedy.
    """
    if requested is None:
        return "zarr" if layer in _ZARR_DEFAULT_LAYERS else "tiff"
    if layer == "objmap" and requested == "zarr":
        raise click.UsageError(
            "--layer objmap has no single-series OME-Zarr form (NGFF 0.5 "
            "§2.6: a labels group is nested inside an image group and is "
            "not itself an image). Use --process-format tiff for the 16-bit "
            "raw-label PNG, or --layer rgb."
        )
    return requested
```

Add `import click` and the `ProcessFormat` import at the top of the module.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/cli/test_process_format_option.py -v
```

Expected: PASS (12 tests).

- [ ] **Step 5: Wire the option into the CLI**

In `src/phenotypic/_cli/_cli_process_single.py`, beside the `--layer` option
(`:511`):

```python
@click.option(
    "--process-format",
    "process_format",
    type=click.Choice(["tiff", "zarr"]),
    default=None,
    help=(
        "Output format for --mode process. Default: zarr for "
        "rgb/gray/detect_mat (a single-series OME-Zarr store), tiff for "
        "objmap (a 16-bit raw-label PNG)."
    ),
)
```

Add `process_format: Optional[str]` to the command function signature beside
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

Note the `elif` becomes `else` with two checks inside — write it as shown rather
than adding a second `elif`, so the two options are validated together.

At the call site (`:681`), pass the resolved value:

```python
            process_single_apply_only_core(
                pipeline_path=pipeline,
                ...
                commit_guard=commit_guard,
                process_format=resolved_process_format,
            )
```

Import `resolve_process_format` alongside the existing
`process_single_apply_only_core` import.

- [ ] **Step 6: Write the CLI-level tests**

Append to `tests/unit/cli/test_process_format_option.py`:

```python
from click.testing import CliRunner

from phenotypic._cli._cli_process_single import process_single  # confirm the name


def test_process_format_is_rejected_outside_process_mode(tmp_path) -> None:
    """Mirrors how --layer already behaves."""
    result = CliRunner().invoke(
        process_single,
        ["--mode", "full", "--process-format", "zarr"],
    )
    assert result.exit_code != 0
    assert "--process-format" in result.output


def test_process_format_appears_in_help() -> None:
    result = CliRunner().invoke(process_single, ["--help"])
    assert "--process-format" in result.output
    assert "zarr for" in result.output   # the layer-dependent default is stated
```

Open `_cli_process_single.py` and confirm the click command's actual object
name before writing the import; do not guess it.

- [ ] **Step 7: Run the CLI suite**

```bash
uv run pytest tests/unit/cli/ -q
uv run mypy src/phenotypic/_cli/_cli_process_single.py \
            src/phenotypic/_cli/_cli_process_only.py
uv run ruff check --fix src/phenotypic/_cli/_cli_process_single.py \
    src/phenotypic/_cli/_cli_process_only.py \
    tests/unit/cli/test_process_format_option.py
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/phenotypic/_cli/_cli_process_single.py \
        src/phenotypic/_cli/_cli_process_only.py \
        tests/unit/cli/test_process_format_option.py
git commit -m "feat(cli): add --process-format with a layer-dependent default

rgb/gray/detect_mat default to zarr; objmap defaults to tiff, so a bare
--mode process --layer objmap keeps working. An explicit
--layer objmap --process-format zarr is a UsageError naming the remedy, since
NGFF 2.6 nests a label image inside an image group and states the labels group
is not itself an image.

The rule lives in one function, resolve_process_format, rather than in the
option declaration. --ext is left untouched: process mode already ignores it,
so wiring it in would change tiff/png naming as a side effect."
```

---

### Task 9: consolidated metadata on a process-mode store

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
- Modify: `src/phenotypic/_cli/_cli_process_only.py` (`write_process_only_layer`)
- Test: `tests/unit/cli/test_process_only_consolidated.py` (create)

**Interfaces:**
- Consumes: Task 7's zarr branch.
- Produces: nothing new; the store gains a root-level `consolidated_metadata`
  key.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/cli/test_process_only_consolidated.py`:

```python
"""Consolidated metadata: one GET to open a store, and safely ignorable."""

from __future__ import annotations

import json
from pathlib import Path

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


def test_a_consolidated_store_still_round_trips_through_imread(
    tmp_path: Path,
) -> None:
    import numpy as np

    img = Image(load_synth_yeast_plate())
    out = tmp_path / "p01.ome.zarr"
    write_process_only_layer(img, "rgb", out, fmt="zarr")
    assert np.array_equal(Image.imread(out).rgb[:], img.rgb[:])
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/cli/test_process_only_consolidated.py -v
```

Expected: `KeyError: 'consolidated_metadata'` on the first two. The last three
PASS already — they are the regression half, pinning what consolidation must
*not* break.

- [ ] **Step 3: Consolidate inside the zarr branch**

In `write_process_only_layer`, after `image._save_store(...)` returns:

```python
        store = image._save_store(
            out_path,
            series=(layer,),
            write_objmap=False,
            levels=ngff_.pyramid_level_count(height, width),
            work_id=None,
            durable=None,
            commit_guard=commit_guard,
            write_image_class=False,
        )
        _consolidate_published_store(store)
        return
```

And add the helper beside it:

```python
def _consolidate_published_store(store: Path) -> None:
    """Collapse per-node metadata into the root, for cheap remote opens.

    Opening a store costs one GET per metadata file -- 8 of a single-series
    store's 12 -- which is the latency that matters once the destination is
    object storage and a viewer enumerates many stores.

    This is legal under the Zarr v3 extension mechanism rather than in spite of
    it: the spec requires a reader to fail on an unrecognised metadata field
    *unless* it carries ``"must_understand": false``, and zarr-python writes
    exactly that. A reader that does not know the key must ignore it, and the
    per-node ``zarr.json`` documents all remain, so it still walks the tree.

    Safe here specifically because a process-mode store is written once into a
    ``.part`` and promoted by rename. It is never mutated, so the consolidated
    view cannot drift from the tree it describes. Do not lift this onto a store
    that is rewritten in place.

    zarr-python warns that consolidated metadata "is currently not part in the
    Zarr format 3 specification". That is accurate but narrow -- it means *not
    core spec*, not *non-conformant* -- so the warning is suppressed here
    rather than globally.
    """
    import warnings as _warnings

    import zarr

    with _warnings.catch_warnings():
        _warnings.filterwarnings(
            "ignore",
            message=".*[Cc]onsolidated metadata.*",
            category=UserWarning,
        )
        zarr.consolidate_metadata(str(store))
```

Check the real warning category before pinning it — run
`uv run python -W error::UserWarning -c "import zarr; ..."` if the filter does
not catch it, and widen to the base `Warning` only if the specific class cannot
be named.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/cli/test_process_only_consolidated.py -v
```

Expected: PASS (5 tests).

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
so the consolidated view cannot go stale."
```
