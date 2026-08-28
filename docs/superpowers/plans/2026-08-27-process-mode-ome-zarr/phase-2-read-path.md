# Phase 2 — Read path

A pure NGFF resolver in `sdk_`, then the thin `imread` branch that consumes it.
This is where ecosystem compatibility actually lives: the resolver is what meets
a napari, QuPath, or `bioformats2raw` store.

Read [`README.md`](README.md)'s **Global Constraints** first. Two bind hardest
here: **never call `require_readable_store`** from this path, and **refuse
rather than silently project**.

---

### Task 5: `ngff_.read_ngff_image_spec` — the projection resolver

`Image` models a 2-D image, optionally with three colour channels. An arbitrary
NGFF store does not: it may be 5-D `tczyx`, carry many series, or be an HCS
plate. The mapping is explicit, ordered, and refuses rather than guesses.

The axis projection is split into its own pure function so the hard logic is
testable without building a store on disk.

**Files:**
- Modify: `src/phenotypic/sdk_/ngff_.py` (append a new section after the
  existing read helpers, around `:626`)
- Test: `tests/unit/sdk_/test_ngff_read_spec.py` (create)

**Interfaces:**
- Consumes: `ngff_.read_root_attributes` (`:589`), `ngff_.STORE_ROOT_JSON`,
  `sdk_.store_stem`.
- Produces:

```python
@dataclass(frozen=True)
class NgffImageSpec:
    array: np.ndarray        # projected to (H, W) or (H, W, 3)
    series: str              # resolved series path, relative to the store root
    level: int               # pyramid level actually read
    bit_depth: int | None    # from phenotypic.bit_depth, else inferred, else None
    phenotypic: dict         # the phenotypic block; {} when absent

def project_ngff_axes(
    axes: Sequence[Mapping[str, object]],
    shape: Sequence[int],
    *,
    t: int | None = None,
    z: int | None = None,
    c: int | None = None,
) -> tuple[tuple[object, ...], bool]:
    """Return (index tuple, is_rgb)."""

def read_ngff_image_spec(
    store_path: Path,
    *,
    series: str | None = None,
    level: int = 0,
    t: int | None = None,
    z: int | None = None,
    c: int | None = None,
) -> NgffImageSpec:
```

- [ ] **Step 1: Write the failing tests for the pure projector**

Create `tests/unit/sdk_/test_ngff_read_spec.py`:

```python
"""The imread projection rule: explicit, ordered, and it refuses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from phenotypic.sdk_ import ngff_


def _axes(*names: str) -> list[dict[str, str]]:
    kind = {"t": "time", "c": "channel", "z": "space", "y": "space", "x": "space"}
    return [{"name": n, "type": kind[n]} for n in names]


# --- the pure projector -----------------------------------------------------

def test_2d_passes_through_unprojected() -> None:
    index, is_rgb = ngff_.project_ngff_axes(_axes("y", "x"), (40, 30))
    assert index == (slice(None), slice(None))
    assert is_rgb is False


def test_three_channels_are_rgb() -> None:
    index, is_rgb = ngff_.project_ngff_axes(_axes("c", "y", "x"), (3, 40, 30))
    assert is_rgb is True
    assert index == (slice(None), slice(None), slice(None))


def test_singleton_axes_are_squeezed() -> None:
    index, is_rgb = ngff_.project_ngff_axes(
        _axes("t", "c", "z", "y", "x"), (1, 3, 1, 40, 30)
    )
    assert index == (0, slice(None), 0, slice(None), slice(None))
    assert is_rgb is True


def test_single_channel_squeezes_to_2d() -> None:
    index, is_rgb = ngff_.project_ngff_axes(_axes("c", "y", "x"), (1, 40, 30))
    assert index == (0, slice(None), slice(None))
    assert is_rgb is False


def test_a_real_time_axis_is_refused() -> None:
    with pytest.raises(ValueError, match="time"):
        ngff_.project_ngff_axes(_axes("t", "y", "x"), (10, 40, 30))


def test_a_real_time_axis_is_readable_with_an_explicit_index() -> None:
    index, _ = ngff_.project_ngff_axes(_axes("t", "y", "x"), (10, 40, 30), t=4)
    assert index == (4, slice(None), slice(None))


def test_a_real_z_axis_is_refused() -> None:
    with pytest.raises(ValueError, match="z|space"):
        ngff_.project_ngff_axes(_axes("z", "y", "x"), (12, 40, 30))


def test_five_channels_are_refused() -> None:
    with pytest.raises(ValueError, match="channel"):
        ngff_.project_ngff_axes(_axes("c", "y", "x"), (5, 40, 30))


def test_five_channels_are_readable_with_an_explicit_index() -> None:
    index, is_rgb = ngff_.project_ngff_axes(_axes("c", "y", "x"), (5, 40, 30), c=2)
    assert index == (2, slice(None), slice(None))
    assert is_rgb is False


def test_two_channels_are_refused_rather_than_guessed() -> None:
    """2 is neither a grayscale nor an RGB triple. Refuse."""
    with pytest.raises(ValueError, match="channel"):
        ngff_.project_ngff_axes(_axes("c", "y", "x"), (2, 40, 30))


def test_an_out_of_range_override_is_refused() -> None:
    with pytest.raises(ValueError):
        ngff_.project_ngff_axes(_axes("t", "y", "x"), (10, 40, 30), t=99)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/sdk_/test_ngff_read_spec.py -v
```

Expected: `AttributeError: module 'phenotypic.sdk_.ngff_' has no attribute
'project_ngff_axes'` on every test.

- [ ] **Step 3: Implement the pure projector**

Append to `src/phenotypic/sdk_/ngff_.py`:

```python
# ---------------------------------------------------------------------------
# Reading an arbitrary NGFF store as plain pixels (spec 4)
# ---------------------------------------------------------------------------


def project_ngff_axes(
    axes: Sequence[Mapping[str, object]],
    shape: Sequence[int],
    *,
    t: int | None = None,
    z: int | None = None,
    c: int | None = None,
) -> tuple[tuple[object, ...], bool]:
    """Map an NGFF array's axes onto PhenoTypic's 2-D image model.

    ``Image`` is 2-D, optionally with three colour channels. NGFF permits 2 to
    5 axes. This is the total mapping between them, and it **refuses rather
    than guesses**: silently reading ``t=0`` of a timelapse, or channel 0 of a
    five-channel acquisition, yields a plausible image and a wrong result that
    nothing downstream can detect.

    Args:
        axes: The ``multiscales[].axes`` list.
        shape: The level's array shape; same length and order as *axes*.
        t: Index to take on a ``time`` axis of size > 1. ``None`` refuses.
        z: Index to take on the third ``space`` axis when its size is > 1.
            ``None`` refuses.
        c: Index to take on a ``channel`` axis whose size is neither 1 nor 3.
            ``None`` refuses.

    Returns:
        ``(index, is_rgb)`` -- an index tuple to apply to the array, and whether
        the result carries three colour channels. When *is_rgb* is ``True`` the
        caller must still move the channel axis last; NGFF stores it first.

    Raises:
        ValueError: If *axes* and *shape* disagree in length, if an axis of
            size > 1 has no override, if a ``channel`` axis is neither 1 nor 3
            without an explicit *c*, or if an override is out of range.
    """
    if len(axes) != len(shape):
        raise ValueError(
            f"axes/shape mismatch: {len(axes)} axes for a {len(shape)}-D array"
        )

    def _pick(name: str, override: int | None, size: int, flag: str) -> object:
        if size == 1:
            return 0
        if override is None:
            raise ValueError(
                f"this store's {name!r} axis has size {size}; PhenoTypic's "
                f"Image is 2-D. Pass {flag}=<index> to choose one, or use "
                f"zarr directly to read the whole array."
            )
        if not 0 <= override < size:
            raise ValueError(
                f"{flag}={override} is out of range for a {name!r} axis of "
                f"size {size}"
            )
        return override

    index: list[object] = []
    is_rgb = False
    seen_space = 0
    n_space = sum(1 for a in axes if a.get("type") == "space")

    for axis, size in zip(axes, shape):
        kind = axis.get("type")
        name = str(axis.get("name", kind))
        if kind == "time":
            index.append(_pick(name, t, size, "t"))
        elif kind == "channel":
            if size == 3 and c is None:
                is_rgb = True
                index.append(slice(None))
            elif size == 1 and c is None:
                index.append(0)
            elif c is None:
                raise ValueError(
                    f"this store's channel axis has size {size}; PhenoTypic "
                    f"reads 1 (grayscale) or 3 (RGB). Pass c=<index> to "
                    f"choose one channel."
                )
            else:
                index.append(_pick(name, c, size, "c"))
        elif kind == "space":
            seen_space += 1
            # Three space axes means the first is the stacking (z) axis.
            if n_space == 3 and seen_space == 1:
                index.append(_pick(name, z, size, "z"))
            else:
                index.append(slice(None))
        else:
            # A custom or null axis type. NGFF permits it; we cannot map it.
            index.append(_pick(name, None, size, "(unsupported axis)"))

    return tuple(index), is_rgb
```

- [ ] **Step 4: Run the projector tests**

```bash
uv run pytest tests/unit/sdk_/test_ngff_read_spec.py -v
```

Expected: PASS (11 tests).

- [ ] **Step 5: Commit the projector**

```bash
git add src/phenotypic/sdk_/ngff_.py tests/unit/sdk_/test_ngff_read_spec.py
git commit -m "feat(ngff): add project_ngff_axes, the 2-D projection rule

Maps an arbitrary NGFF array's axes onto PhenoTypic's 2-D image model. Refuses
rather than guesses: an unsqueezable t, z, or channel axis raises and names the
override that would read it. Silently taking index 0 produces a plausible image
and an undetectable wrong result."
```

- [ ] **Step 6: Write the failing tests for the store resolver**

Append to `tests/unit/sdk_/test_ngff_read_spec.py`:

```python
# --- the store resolver -----------------------------------------------------

def _write_store(
    root: Path,
    *,
    series: dict[str, tuple[tuple[int, ...], list[dict[str, str]]]],
    series_list: list[str] | None = None,
    phenotypic: dict | None = None,
    extra_root_ome: dict | None = None,
) -> Path:
    """Build a minimal but conformant multi-series NGFF store."""
    group = zarr.create_group(store=str(root), zarr_format=3)
    root_ome: dict = {"version": "0.5", "bioformats2raw.layout": 3}
    root_ome.update(extra_root_ome or {})
    group.attrs["ome"] = root_ome
    if phenotypic is not None:
        group.attrs["phenotypic"] = phenotypic

    if series_list is not None:
        ome_group = group.create_group("OME")
        ome_group.attrs["ome"] = {"version": "0.5", "series": series_list}

    rng = np.random.default_rng(0)
    for name, (shape, axes) in series.items():
        sub = group.create_group(name)
        arr = sub.create_array(
            "0",
            shape=shape,
            chunks=shape,
            dtype="uint16",
            dimension_names=[a["name"] for a in axes],
        )
        arr[:] = rng.integers(1, 4096, size=shape, dtype=np.uint16)
        sub.attrs["ome"] = {
            "version": "0.5",
            "multiscales": [{
                "name": name,
                "axes": axes,
                "datasets": [{
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1.0] * len(shape)}
                    ],
                }],
            }],
        }
    return root


def test_resolver_reads_the_first_declared_series(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "s.ome.zarr",
        series={
            "rgb": ((3, 8, 6), _axes("c", "y", "x")),
            "gray": ((8, 6), _axes("y", "x")),
        },
        series_list=["rgb", "gray"],
    )
    spec = ngff_.read_ngff_image_spec(store)
    assert spec.series == "rgb"
    assert spec.array.shape == (8, 6, 3)     # transposed to HWC
    assert spec.bit_depth == 16              # inferred from uint16


def test_resolver_honours_an_explicit_series(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "s.ome.zarr",
        series={
            "rgb": ((3, 8, 6), _axes("c", "y", "x")),
            "gray": ((8, 6), _axes("y", "x")),
        },
        series_list=["rgb", "gray"],
    )
    spec = ngff_.read_ngff_image_spec(store, series="gray")
    assert spec.series == "gray"
    assert spec.array.shape == (8, 6)


def test_resolver_falls_back_to_group_zero_without_a_series_list(
    tmp_path: Path,
) -> None:
    """NGFF 2.2.3: no series attribute means consecutively numbered groups."""
    store = _write_store(
        tmp_path / "s.ome.zarr",
        series={"0": ((8, 6), _axes("y", "x"))},
    )
    assert ngff_.read_ngff_image_spec(store).series == "0"


def test_resolver_refuses_an_hcs_plate(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "p.ome.zarr",
        series={"A": ((8, 6), _axes("y", "x"))},
        extra_root_ome={"plate": {"name": "plate1", "wells": []}},
    )
    with pytest.raises(ValueError, match="plate"):
        ngff_.read_ngff_image_spec(store)


def test_resolver_reads_a_store_with_no_phenotypic_block(tmp_path: Path) -> None:
    """Case C. require_readable_store must never be reached from here."""
    store = _write_store(
        tmp_path / "foreign.ome.zarr",
        series={"0": ((8, 6), _axes("y", "x"))},
    )
    spec = ngff_.read_ngff_image_spec(store)
    assert spec.phenotypic == {}
    assert spec.array.shape == (8, 6)


def test_resolver_reads_a_future_store_version(tmp_path: Path) -> None:
    """A newer store's NGFF geometry is still NGFF (spec 4.6)."""
    store = _write_store(
        tmp_path / "future.ome.zarr",
        series={"0": ((8, 6), _axes("y", "x"))},
        phenotypic={"store_schema_version": 999},
    )
    assert ngff_.read_ngff_image_spec(store).array.shape == (8, 6)


def test_resolver_prefers_stored_bit_depth_over_dtype(tmp_path: Path) -> None:
    store = _write_store(
        tmp_path / "s.ome.zarr",
        series={"0": ((8, 6), _axes("y", "x"))},
        phenotypic={"store_schema_version": 3, "bit_depth": 12},
    )
    assert ngff_.read_ngff_image_spec(store).bit_depth == 12


def test_resolver_refuses_a_non_image_directory(tmp_path: Path) -> None:
    empty = tmp_path / "nothing.ome.zarr"
    empty.mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        ngff_.read_ngff_image_spec(empty)
```

- [ ] **Step 7: Run tests to verify they fail**

```bash
uv run pytest tests/unit/sdk_/test_ngff_read_spec.py -v -k resolver
```

Expected: `AttributeError: … has no attribute 'read_ngff_image_spec'`.

- [ ] **Step 8: Implement the resolver**

Append to `src/phenotypic/sdk_/ngff_.py`, after `project_ngff_axes`:

```python
@dataclass(frozen=True)
class NgffImageSpec:
    """One NGFF image, projected onto PhenoTypic's 2-D image model.

    Attributes:
        array: Level pixels as ``(H, W)`` or ``(H, W, 3)``.
        series: Resolved series path, relative to the store root.
        level: Pyramid level actually read.
        bit_depth: From ``phenotypic.bit_depth`` when present, else inferred
            from the dtype, else ``None``.
        phenotypic: The ``attributes.phenotypic`` block; ``{}`` when absent.
    """

    array: np.ndarray
    series: str
    level: int
    bit_depth: int | None
    phenotypic: dict


def _resolve_series_path(store_path: Path, attributes: dict) -> str:
    """Pick the series a generic reader should open. See spec 4.1."""
    ome = attributes.get("ome", {})
    if "plate" in ome:
        raise ValueError(
            f"{store_path} is an HCS plate, which is a collection of wells "
            f"rather than one image. Pass series=<row>/<col>/<field> to read "
            f"a single field."
        )

    ome_json = Path(store_path) / "OME" / STORE_ROOT_JSON
    if ome_json.is_file():
        payload = json.loads(ome_json.read_text(encoding="utf-8"))
        declared = payload.get("attributes", {}).get("ome", {}).get("series")
        if declared:
            return str(declared[0])

    if "multiscales" in ome:
        return ""  # the root group is itself the image

    if (Path(store_path) / "0" / STORE_ROOT_JSON).is_file():
        return "0"  # NGFF 2.2.3 consecutive-integer form

    raise ValueError(
        f"{store_path} declares no OME series, no multiscales at its root, "
        f"and no group '0'. It is not an OME-Zarr image."
    )


def read_ngff_image_spec(
    store_path: Path,
    *,
    series: str | None = None,
    level: int = 0,
    t: int | None = None,
    z: int | None = None,
    c: int | None = None,
) -> NgffImageSpec:
    """Read any OME-Zarr store as plain pixels.

    The read path behind :meth:`phenotypic.Image.imread` for a store. It reads
    NGFF **structure** only and treats ``attributes.phenotypic`` as optional
    enrichment, so a napari, QuPath, or ``bioformats2raw`` export works.

    It deliberately does **not** call :func:`require_readable_store`: that
    raises ``KeyError`` when the ``phenotypic`` block is absent, which is the
    normal condition for every third-party store -- the exact case this
    function exists to serve. A store written by a newer PhenoTypic is readable
    here, and correctly so: its NGFF geometry is still NGFF.

    Args:
        store_path: A ``*.ome.zarr`` directory.
        series: Series to read. ``None`` resolves it per spec 4.1.
        level: Pyramid level. ``0`` is the highest resolution; NGFF requires
            ``datasets`` to be ordered largest first.
        t: Index on a ``time`` axis of size > 1.
        z: Index on the stacking ``space`` axis when its size is > 1.
        c: Index on a ``channel`` axis that is neither 1 nor 3.

    Returns:
        An :class:`NgffImageSpec`.

    Raises:
        FileNotFoundError: If the store has no root ``zarr.json``.
        ValueError: If the store is an HCS plate, declares no readable image,
            or cannot be projected onto a 2-D image (see
            :func:`project_ngff_axes`).
    """
    import zarr

    store_path = Path(store_path)
    attributes = read_root_attributes(store_path)
    phenotypic = attributes.get(PhenotypicAttr.ROOT, {})

    resolved = _resolve_series_path(store_path, attributes) if series is None else series

    group_path = store_path / resolved if resolved else store_path
    payload = json.loads((group_path / STORE_ROOT_JSON).read_text(encoding="utf-8"))
    multiscales = payload["attributes"]["ome"]["multiscales"][0]
    axes = multiscales["axes"]
    datasets = multiscales["datasets"]
    if not 0 <= level < len(datasets):
        raise ValueError(
            f"level {level} is out of range; {store_path} has "
            f"{len(datasets)} pyramid level(s)"
        )

    array = zarr.open_array(store=str(group_path / datasets[level]["path"]), mode="r")
    index, is_rgb = project_ngff_axes(axes, array.shape, t=t, z=z, c=c)
    data = np.asarray(array[index])
    if is_rgb:
        data = np.moveaxis(data, 0, -1)  # NGFF stores channels first

    bit_depth = phenotypic.get("bit_depth")
    if bit_depth is None:
        bit_depth = {np.uint8: 8, np.uint16: 16}.get(data.dtype.type)

    return NgffImageSpec(
        array=data,
        series=resolved,
        level=level,
        bit_depth=int(bit_depth) if bit_depth is not None else None,
        phenotypic=dict(phenotypic),
    )
```

Add `from dataclasses import dataclass` and `from typing import Mapping` to the
module imports if they are not already present.

- [ ] **Step 9: Run tests to verify they pass**

```bash
uv run pytest tests/unit/sdk_/test_ngff_read_spec.py -v
```

Expected: PASS (19 tests).

- [ ] **Step 10: Lint, type-check, commit**

```bash
uv run mypy src/phenotypic/sdk_/ngff_.py
uv run ruff check --fix src/phenotypic/sdk_/ngff_.py \
    tests/unit/sdk_/test_ngff_read_spec.py
git add src/phenotypic/sdk_/ngff_.py tests/unit/sdk_/test_ngff_read_spec.py
git commit -m "feat(ngff): add read_ngff_image_spec, the plain-pixel store reader

Resolves a series per NGFF 2.2.3, reads one level, projects the axes onto
PhenoTypic's 2-D image model, and moves NGFF's leading channel axis last. Reads
structure only and treats the phenotypic block as optional enrichment, so a
napari, QuPath, or bioformats2raw export works. It never calls
require_readable_store, which raises on exactly the third-party stores this
exists to serve."
```

---

### Task 6: `Image.imread` store branch

`imread` currently dispatches purely on `filepath.suffix` against
`IO.ACCEPTED_FILE_EXTENSIONS` (`_image_io_handler.py:637`) and raises
`UnsupportedFileTypeError` for anything else, including a store directory.

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/_image_io_handler.py:602-700`
- Test: `tests/unit/sdk_/test_imread_store.py` (create)

**Interfaces:**
- Consumes: `ngff_.read_ngff_image_spec`, `ngff_.STORE_SUFFIX`,
  `sdk_.store_stem`.
- Produces: `Image.imread(path, …, series=None, level=0, t=None, z=None,
  c=None)` accepts a `*.ome.zarr` directory.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/sdk_/test_imread_store.py`:

```python
"""imread reads a store as plain pixels -- as if it were a TIFF."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.data import load_synth_yeast_plate
from phenotypic.sdk_ import ngff_
from phenotypic.sdk_.constants_ import IMAGE


def _processed_rgb_store(tmp_path: Path) -> tuple[Path, Image]:
    img = Image(load_synth_yeast_plate())
    store = img._save_store(
        tmp_path / "IMG_4471.ome.zarr",
        series=("rgb",),
        write_objmap=False,
        levels=ngff_.pyramid_level_count(*img.rgb[:].shape[:2]),
        work_id=None,
        durable=False,
        write_image_class=False,
    )
    return store, img


def test_imread_round_trips_rgb_bit_exactly(tmp_path: Path) -> None:
    store, img = _processed_rgb_store(tmp_path)
    loaded = Image.imread(store)
    assert np.array_equal(loaded.rgb[:], img.rgb[:])


def test_imread_uses_the_store_stem_not_path_stem(tmp_path: Path) -> None:
    """Path('IMG_4471.ome.zarr').stem is 'IMG_4471.ome' -- a wrong name."""
    store, _ = _processed_rgb_store(tmp_path)
    assert Image.imread(store).name == "IMG_4471"


def test_imread_records_the_store_suffix(tmp_path: Path) -> None:
    store, _ = _processed_rgb_store(tmp_path)
    assert Image.imread(store).metadata[IMAGE.SUFFIX] == ngff_.STORE_SUFFIX


def test_imread_carries_provenance_across(tmp_path: Path) -> None:
    """The operations that produced the pixels survive the round trip."""
    store, _ = _processed_rgb_store(tmp_path)
    loaded = Image.imread(store)
    assert loaded._metadata.provenance_journal is not None


def test_imread_yields_no_objects_from_a_processed_store(tmp_path: Path) -> None:
    """Pixels only. A process store has no objmap and imread invents none."""
    store, _ = _processed_rgb_store(tmp_path)
    assert Image.imread(store).num_objects == 0


def test_imread_reads_a_bundle_store_as_pixels(tmp_path: Path) -> None:
    """Documented behaviour: the verb decides, not the file (spec 3.2)."""
    img = Image(load_synth_yeast_plate())
    store = img.save2zarr(tmp_path / "bundle.ome.zarr")
    assert np.array_equal(Image.imread(store).rgb[:], img.rgb[:])


def test_a_non_store_directory_is_still_unsupported(tmp_path: Path) -> None:
    from phenotypic.sdk_.exceptions_ import UnsupportedFileTypeError

    plain = tmp_path / "just_a_folder"
    plain.mkdir()
    with pytest.raises((UnsupportedFileTypeError, ValueError, IsADirectoryError)):
        Image.imread(plain)
```

Check the real import path of `UnsupportedFileTypeError` before running — it is
imported at the top of `_image_io_handler.py`; use the same path.

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/sdk_/test_imread_store.py -v
```

Expected: FAIL with `UnsupportedFileTypeError: .zarr` — `Path("x.ome.zarr")`
has suffix `".zarr"`, which is not in `IO.ACCEPTED_FILE_EXTENSIONS`.

- [ ] **Step 3: Add the store branch**

In `imread`, immediately after `filepath: Path = Path(filepath)` and before the
suffix dispatch:

```python
        filepath = Path(filepath)
        if filepath.is_dir() and filepath.name.endswith(ngff_.STORE_SUFFIX):
            return cls._imread_store(
                filepath,
                series=series,
                level=level,
                t=t,
                z=z,
                c=c,
                **kwargs,
            )

        rawpy_params = rawpy_params or {}
```

Extend the signature with the five keyword-only overrides, defaulting to
`None`/`0`, and document them in `Args:` as applying only to a store. Import
`ngff_` at the top of the branch (the module already imports it lazily
elsewhere; follow the local convention).

Then add the helper beside it:

```python
    @classmethod
    def _imread_store(
        cls,
        store_path: Path,
        *,
        series: str | None = None,
        level: int = 0,
        t: int | None = None,
        z: int | None = None,
        c: int | None = None,
        **kwargs,
    ) -> Image:
        """Read an OME-Zarr store as plain pixels.

        The store analogue of the TIFF branch: pixels in, a fresh image out. It
        never restores PhenoTypic run state -- that is
        :meth:`load_zarr`'s job, and it refuses a store that is not a run
        bundle.

        Only what the file says about itself is carried across: the provenance
        journal and the ``imported`` metadata section. The ``protected`` and
        ``public`` sections are run state and are deliberately dropped; that is
        the line that keeps this from becoming a partial ``load_zarr``.
        """
        from phenotypic.sdk_ import ngff_
        from phenotypic.sdk_._io_constants import store_stem

        spec = ngff_.read_ngff_image_spec(
            store_path, series=series, level=level, t=t, z=z, c=c
        )
        name = store_stem(store_path)
        bit_depth = kwargs.pop("bit_depth", None) or spec.bit_depth
        image = cls(arr=spec.array, name=name, bit_depth=bit_depth, **kwargs)
        image.name = name
        image.metadata[IMAGE.SUFFIX] = ngff_.STORE_SUFFIX

        journal = spec.phenotypic.get(ngff_.PhenotypicAttr.PROVENANCE)
        if journal:
            image._metadata.provenance_journal = deepcopy(journal)

        imported = spec.phenotypic.get(ngff_.PhenotypicAttr.METADATA, {}).get(
            ngff_.PhenotypicAttr.IMPORTED, {}
        )
        for key, value in imported.items():
            image.metadata[key] = value
        return image
```

Read how the TIFF branch assigns imported metadata before writing that loop —
match it exactly rather than inventing a second idiom.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/sdk_/test_imread_store.py -v
```

Expected: PASS (7 tests).

- [ ] **Step 5: Add the doctest and run the full image suite**

Add to `imread`'s `Examples:` block:

```
            Read a process-mode store as plain pixels:

            >>> import tempfile
            >>> from pathlib import Path
            >>> from phenotypic import Image
            >>> from phenotypic.data import load_synth_yeast_plate
            >>> img = Image(load_synth_yeast_plate())
            >>> with tempfile.TemporaryDirectory() as tmp:
            ...     store = img.save2zarr(Path(tmp) / 'plate.ome.zarr')
            ...     Image.imread(store).rgb[:].shape == img.rgb[:].shape
            True
```

```bash
uv run pytest tests/unit/sdk_/ tests/unit/test_ome_zarr_invariants.py -q
uv run pytest --doctest-modules src/phenotypic/_core/_image_parts/_image_io_handler.py -q
uv run mypy src/phenotypic/_core/_image_parts/_image_io_handler.py
uv run ruff check --fix src/phenotypic/_core/_image_parts/_image_io_handler.py \
    tests/unit/sdk_/test_imread_store.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_core/_image_parts/_image_io_handler.py \
        tests/unit/sdk_/test_imread_store.py
git commit -m "feat(io): imread reads an OME-Zarr store as plain pixels

A directory ending in .ome.zarr routes to read_ngff_image_spec; everything else
keeps today's suffix dispatch. Pixels in, a fresh Image out -- never run state,
which stays load_zarr's job. Carries the provenance journal and the imported
metadata section, matching what imread already extracts from a TIFF; protected
and public are run state and are dropped. Names the image via store_stem, since
Path('img.ome.zarr').stem is 'img.ome'."
```
