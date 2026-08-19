# Phase 2 — Image/GridImage store I/O, path constants, conformance harness

> Global Constraints live in [`README.md`](README.md#global-constraints) and apply to
> every task here. Spec: [`design.md`](../../specs/2026-08-18-ome-zarr-image-store/design.md) §1, §2, §3.1, §3.3, §4.1, §7.

**Depends on:** Phase 1.
**Blocks:** Phases 3, 4, 5.

This phase makes an `Image` round-trip through a store. The HDF quartet is left in place
and still works — it is removed in Phase 6, after migration (Phase 5) has been built on top
of its readers.

---

### Task 2.1: Path constants and the store locator

**Files:**
- Modify: `src/phenotypic/sdk_/_io_constants.py` (line 656 `DIR_HDF`, line 1447
  `dataset_hdf_dir`, line 1886 `HdfAttr`, line 1936 `load_image_from_hdf`, line 2063
  `BundleLayout.hdf_path`)
- Test: `tests/unit/sdk_/test_io_constants.py` (extend), `tests/unit/sdk_/test_bundle_layout.py` (extend)

**Interfaces:**
- Consumes: `ngff_.STORE_SUFFIX`, `ngff_.PhenotypicAttr`, `ngff_.read_phenotypic_attributes`.
- Produces:
  ```python
  DIR_ZARR: Final[str] = "zarr"
  def dataset_zarr_dir(output_dir: Path, dataset: str) -> Path
  def zarr_store_path(output_dir: Path, dataset: str, stem: str) -> Path
  def load_image_from_store(store_path: Path, *, fallback: ImageTypeName = "Image") -> "_Image | _GridImage"
  BundleLayout.store_path(dataset: str, stem: str) -> Optional[Path]
  ```

**Constraints specific to this task:**
- `DIR_HDF` / `dataset_hdf_dir` / `HdfAttr` / `load_image_from_hdf` / `BundleLayout.hdf_path`
  are **kept** through Phase 5 — migration reads legacy trees and needs them. They are
  removed in Phase 6. Do not delete them here.
- `zarr_store_path` is the **only** place `STORE_SUFFIX` is joined to a stem. Nothing
  anywhere else may write `f"{stem}.ome.zarr"`. A grep gate in Phase 7 enforces this.
- `BundleLayout.store_path` returns the path only when it `is_dir()` — the store is a
  directory, so `is_file()` (as `hdf_path` uses) would always be `None`. This is the exact
  bug shape that makes a naive port silently disable every full-res GUI read.
- `load_image_from_store` dispatches on `phenotypic.image_class` — **not** on
  `Metadata_ImageType`, which is a different, user-visible field.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/sdk_/test_io_constants.py`:

```python
def test_zarr_store_path_is_the_only_suffix_joiner(tmp_path) -> None:
    from phenotypic.sdk_ import dataset_zarr_dir, zarr_store_path
    from phenotypic.sdk_.ngff_ import STORE_SUFFIX

    path = zarr_store_path(tmp_path, "plates", "plate_01")
    assert path == dataset_zarr_dir(tmp_path, "plates") / f"plate_01{STORE_SUFFIX}"
    assert path.parent.name == "zarr"


def test_dataset_zarr_dir_sits_beside_the_other_result_dirs(tmp_path) -> None:
    from phenotypic.sdk_ import dataset_results_dir, dataset_zarr_dir

    assert dataset_zarr_dir(tmp_path, "ds") == dataset_results_dir(tmp_path, "ds") / "zarr"
```

Append to `tests/unit/sdk_/test_bundle_layout.py`:

```python
def test_store_path_resolves_a_directory_not_a_file(tmp_path) -> None:
    """A store is a directory; an is_file() check would always return None."""
    from phenotypic.sdk_ import BundleLayout, zarr_store_path

    store = zarr_store_path(tmp_path, "ds", "img")
    store.mkdir(parents=True)
    (store / "zarr.json").write_text("{}", encoding="utf-8")
    (tmp_path / "deliverables").mkdir()
    (tmp_path / "deliverables" / "master_measurements.parquet").write_bytes(b"")
    layout = BundleLayout.detect(tmp_path)
    assert layout.store_path("ds", "img") == store


def test_store_path_returns_none_when_absent(tmp_path) -> None:
    from phenotypic.sdk_ import BundleLayout

    (tmp_path / "deliverables").mkdir()
    (tmp_path / "deliverables" / "master_measurements.parquet").write_bytes(b"")
    layout = BundleLayout.detect(tmp_path)
    assert layout.store_path("ds", "img") is None
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest tests/unit/sdk_/test_io_constants.py tests/unit/sdk_/test_bundle_layout.py -k "zarr or store_path" -v
```

Expected: `ImportError: cannot import name 'zarr_store_path'`.

- [ ] **Step 3: Add the constants and helpers**

In `_io_constants.py`, beside `DIR_HDF` (line 656):

```python
#: OME-Zarr image-state subdirectory: ``<output>/results/<ds>/zarr/``.
DIR_ZARR: Final[str] = "zarr"
```

Beside `dataset_hdf_dir` (line 1447):

```python
def dataset_zarr_dir(output_dir: Path, dataset: str) -> Path:
    """Return ``<output>/results/<dataset>/zarr/``."""
    return dataset_results_dir(output_dir, dataset) / DIR_ZARR


def zarr_store_path(output_dir: Path, dataset: str, stem: str) -> Path:
    """Return ``<output>/results/<dataset>/zarr/<stem>.ome.zarr/``.

    The single place ``.ome.zarr`` is joined to an image stem. Callers must
    never hand-join the suffix; a grep gate in the test suite enforces this.

    Args:
        output_dir: Run output root.
        dataset: Dataset name.
        stem: Image filename without extension.

    Returns:
        The per-image store path. Existence is not checked.
    """
    from phenotypic.sdk_.ngff_ import STORE_SUFFIX

    return dataset_zarr_dir(output_dir, dataset) / f"{stem}{STORE_SUFFIX}"
```

Beside `load_image_from_hdf` (line 1936):

```python
def load_image_from_store(
    store_path: Path,
    *,
    fallback: ImageTypeName = "Image",
) -> "_Image | _GridImage":
    """Read ``phenotypic.image_class`` from a store root and dispatch the loader.

    Dispatches on ``image_class`` (``Image`` / ``GridImage``), which is the
    loader-dispatch field. It is **not** ``Metadata_ImageType``, which is
    user-visible schema metadata and may be ``GridSection`` on a plain
    :class:`Image`.

    Args:
        store_path: Path to a ``*.ome.zarr`` directory.
        fallback: Class name used when the block carries no ``image_class``.

    Returns:
        An :class:`Image` or :class:`GridImage` loaded from the store.
    """
    from phenotypic import GridImage, Image
    from phenotypic.sdk_.constants_ import IMAGE_TYPES
    from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes

    block = read_phenotypic_attributes(store_path)
    class_name = block.get(PhenotypicAttr.IMAGE_CLASS, fallback)
    image_cls = GridImage if class_name == IMAGE_TYPES.GRID.value else Image
    return image_cls.load_zarr(store_path)
```

On `BundleLayout`, beside `hdf_path` (line 2063):

```python
    def store_path(self, dataset: str, stem: str) -> Optional[Path]:
        """Full-res per-image OME-Zarr store for ``(dataset, stem)``, or ``None``.

        Args:
            dataset: Dataset name (subdirectory under ``results/``).
            stem: Image stem (filename without extension).

        Returns:
            Resolved store path if the **directory** exists, otherwise ``None``.
            Note the ``is_dir`` check: a store is a directory, so the
            ``is_file`` test used by :meth:`hdf_path` would always return
            ``None`` here.
        """
        if self.output_root is None:
            return None
        candidate = zarr_store_path(self.output_root, dataset, stem)
        return candidate if candidate.is_dir() else None
```

Export `DIR_ZARR`, `dataset_zarr_dir`, `zarr_store_path`, and `load_image_from_store`
from `src/phenotypic/sdk_/__init__.py` beside their HDF counterparts.

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/sdk_/test_io_constants.py tests/unit/sdk_/test_bundle_layout.py -v
```

Expected: **all PASS.** No `xfail` is needed — an earlier draft told you to mark
`test_store_path_resolves_a_directory_not_a_file` as
`xfail(strict=True)` "because `Image.load_zarr` does not exist yet", but that test never
calls `load_zarr`: it creates the directory and asserts
`BundleLayout.store_path(...) == store`. It passes the moment Step 3 lands, and
`strict=True` would turn that XPASS into a failure.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/sdk_/_io_constants.py src/phenotypic/sdk_/__init__.py tests/unit/sdk_
git commit -m "feat(sdk): add zarr store path constants and locator

zarr_store_path is the only place .ome.zarr is joined to a stem.
BundleLayout.store_path checks is_dir, not is_file -- a store is a
directory, so a copy-paste of hdf_path would return None for every image
and silently disable full-res GUI reads. The HDF helpers stay until
Phase 6, because --mode migrate is built on their readers."
```

---

### Task 2.2: `save2zarr` and `load_zarr` on `ImageIOHandler`

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/_image_io_handler.py`
  (add beside `save2hdf5` at line 871 and `load_hdf5` at line 1155)
- Test: `tests/unit/core/test_image_zarr_roundtrip.py` (create)

**Interfaces:**
- Consumes: everything from Phase 1.
- Produces:
  ```python
  def save2zarr(self, path, *, work_id: str | None = None, durable: bool | None = None) -> Path
  @classmethod
  def load_zarr(cls, path, **kwargs) -> Image
  @classmethod
  def load_layer_zarr(cls, path, layer: str, level: int = 0) -> np.ndarray
  def _write_series(self, part: Path, series: str, array, levels: int) -> None
  def _series_names(self) -> list[str]
  ```

**Constraints specific to this task:**
- `save2zarr` builds into a `.part` via `ngff_.new_part_path`, writes **arrays first, then
  `OME/zarr.json`, then the root `zarr.json` last**, and promotes with
  `ngff_.promote_store(part, final, fsync=ngff_.durable_writes_enabled(durable))`.
- `rgb` is **omitted entirely** when `self.rgb.isempty()`.
- `objmap` is **always** written, zeros included.
- `work_id` goes into the attributes block at build time — no post-write patch.
- **`build_ome_xml` raises rather than returning `None`**, so there is no fallback branch
  here at all — the `OME/` group is always written (user ruling; ledger **PRE-G2** /
  **ALGO-3**). Dropping `series` while keeping named groups satisfied neither arm of §2.2.3
  and was strictly less conformant than either form. Pass the level-0 shapes and dtypes
  through so `<Pixels>` can carry its required attributes.
- `load_zarr` reads only level 0, restores the three metadata sections verbatim, and warns
  (without upcasting) when `Image.load_zarr` is called on a store whose `image_class` is
  `GridImage` — mirroring `load_hdf5`'s existing behaviour at line 1170.
- `load_layer_zarr(path, layer, level)` resolves `objmap` through
  `phenotypic.labels.objmap`, never by hard-coded path.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/core/test_image_zarr_roundtrip.py`:

```python
"""Image -> store -> Image must be bit-exact in layers and equal in metadata."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phenotypic import GridImage, Image
from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes
from phenotypic.data import load_synth_yeast_plate


@pytest.fixture
def plate() -> Image:
    return Image(load_synth_yeast_plate())


def test_layers_round_trip_bit_exact(plate: Image, tmp_path: Path) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    back = Image.load_zarr(store)
    np.testing.assert_array_equal(back.rgb[:], plate.rgb[:])
    np.testing.assert_array_equal(back.gray[:], plate.gray[:])
    np.testing.assert_array_equal(back.detect_mat[:], plate.detect_mat[:])
    np.testing.assert_array_equal(back.objmap[:], plate.objmap[:])


def test_metadata_sections_round_trip(plate: Image, tmp_path: Path) -> None:
    plate._metadata.public["Metadata_Strain"] = "BY4741"
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    back = Image.load_zarr(store)
    assert dict(back._metadata.public) == dict(plate._metadata.public)
    assert dict(back._metadata.protected) == dict(plate._metadata.protected)
    assert dict(back._metadata.imported) == dict(plate._metadata.imported)


def test_objmap_is_written_even_when_nothing_is_detected(
    plate: Image, tmp_path: Path
) -> None:
    """Stage 1 relies on this: valid_staged_store requires objmap to exist."""
    assert plate.num_objects == 0
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    block = read_phenotypic_attributes(store)
    objmap = block[PhenotypicAttr.LABELS]["objmap"]
    assert (store / objmap / "0").is_dir()
    assert (Image.load_zarr(store).objmap[:] == 0).all()


def test_rgb_is_omitted_entirely_when_empty(tmp_path: Path) -> None:
    # Image(<2-D ndarray>) yields rgb.isempty() is True -- verified by
    # execution. There is no rgb.clear(); the accessor has no such method.
    gray_only = Image(np.asarray(Image(load_synth_yeast_plate()).gray[:]))
    store = gray_only.save2zarr(tmp_path / "g.ome.zarr")
    assert not (store / "rgb").exists()
    block = read_phenotypic_attributes(store)
    assert "rgb" not in block[PhenotypicAttr.SERIES]
    assert block[PhenotypicAttr.LABELS]["objmap"] == "gray/labels/objmap"


def test_primary_series_is_first_in_the_ome_series_list(
    plate: Image, tmp_path: Path
) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    ome = json.loads((store / "OME" / "zarr.json").read_text(encoding="utf-8"))
    assert ome["attributes"]["ome"]["series"][0] == "rgb"


def test_root_zarr_json_is_written_last(plate: Image, tmp_path: Path, monkeypatch) -> None:
    """An interrupted store has no valid root and must read as absent."""
    from phenotypic.sdk_ import ngff_

    written: list[str] = []
    real_promote = ngff_.promote_store

    def _record(part: Path, final: Path, *, fsync: bool):
        for path in sorted(Path(part).rglob("zarr.json")):
            written.append(str(path.relative_to(part)))
        return real_promote(part, final, fsync=fsync)

    monkeypatch.setattr(ngff_, "promote_store", _record)
    plate.save2zarr(tmp_path / "p.ome.zarr")
    assert written[-1] == "zarr.json", written


def test_work_id_is_written_into_the_block_not_patched(
    plate: Image, tmp_path: Path
) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr", work_id="w-42")
    assert read_phenotypic_attributes(store)[PhenotypicAttr.WORK_ID] == "w-42"


def test_pyramid_depth_is_uniform_across_every_series(
    plate: Image, tmp_path: Path
) -> None:
    """NGFF requires a label image to carry its parent's level count."""
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    block = read_phenotypic_attributes(store)
    levels = block[PhenotypicAttr.PYRAMID]["levels"]
    members = [
        *block[PhenotypicAttr.SERIES].values(),
        *block[PhenotypicAttr.LABELS].values(),
    ]
    for member in members:
        found = sorted(p.name for p in (store / member).iterdir() if p.name.isdigit())
        assert found == [str(i) for i in range(levels)], member


def test_pyramid_depth_is_a_pure_function_of_shape(plate: Image, tmp_path: Path) -> None:
    """Fixed, not tunable: save2zarr takes no pyramid argument at all (P3)."""
    import inspect

    from phenotypic.sdk_ import ngff_

    assert "pyramid_levels" not in inspect.signature(Image.save2zarr).parameters
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    gray = plate.gray[:]
    assert read_phenotypic_attributes(store)[PhenotypicAttr.PYRAMID]["levels"] == (
        ngff_.pyramid_level_count(gray.shape[0], gray.shape[1])
    )


def test_load_layer_zarr_reads_one_layer_without_a_full_image(
    plate: Image, tmp_path: Path
) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    np.testing.assert_array_equal(Image.load_layer_zarr(store, "gray"), plate.gray[:])


def test_load_layer_zarr_resolves_objmap_via_the_labels_key(
    tmp_path: Path,
) -> None:
    """rgb-less stores put the label under gray; a hard-coded path would 404."""
    # Image(<2-D ndarray>) yields rgb.isempty() is True -- verified by
    # execution. There is no rgb.clear(); the accessor has no such method.
    gray_only = Image(np.asarray(Image(load_synth_yeast_plate()).gray[:]))
    store = gray_only.save2zarr(tmp_path / "g.ome.zarr")
    assert Image.load_layer_zarr(store, "objmap").shape == gray_only.gray[:].shape


def test_load_layer_zarr_can_read_a_pyramid_level(plate: Image, tmp_path: Path) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    full = Image.load_layer_zarr(store, "gray", level=0)
    half = Image.load_layer_zarr(store, "gray", level=1)
    assert half.shape == ((full.shape[0] + 1) // 2, (full.shape[1] + 1) // 2)


def test_load_layer_zarr_raises_keyerror_for_an_unknown_layer(
    plate: Image, tmp_path: Path
) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    with pytest.raises(KeyError):
        Image.load_layer_zarr(store, "not_a_layer")


def test_image_load_zarr_on_a_griddimage_store_warns_without_upcasting(
    tmp_path: Path,
) -> None:
    grid = GridImage(load_synth_yeast_plate(), nrows=8, ncols=12)
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    with pytest.warns(UserWarning, match="GridImage"):
        back = Image.load_zarr(store)
    assert type(back) is Image


def test_image_class_and_image_type_are_independent(tmp_path: Path) -> None:
    plain = Image(load_synth_yeast_plate())
    plain._metadata.protected["Metadata_ImageType"] = "GridSection"
    store = plain.save2zarr(tmp_path / "s.ome.zarr")
    block = read_phenotypic_attributes(store)
    assert block[PhenotypicAttr.IMAGE_CLASS] == "Image"
    assert (
        block[PhenotypicAttr.METADATA]["protected"]["Metadata_ImageType"]
        == "GridSection"
    )
    assert type(Image.load_zarr(store)) is Image


def test_detect_mode_illuminant_and_gamma_survive(plate: Image, tmp_path: Path) -> None:
    store = plate.save2zarr(tmp_path / "p.ome.zarr")
    back = Image.load_zarr(store)
    assert back.illuminant == plate.illuminant
    assert str(back.gamma) == str(plate.gamma)
    assert back._data.detect_mode == plate._data.detect_mode


def test_a_failed_ome_xml_build_aborts_the_write(
    plate: Image, tmp_path: Path, monkeypatch
) -> None:
    """Fatal, not degraded (ALGO-3). Dropping `series` while keeping named
    groups satisfied neither arm of §2.2.3, so the old fallback shipped a store
    LESS conformant than either option."""
    from phenotypic.sdk_ import ngff_

    def _boom(**kwargs):
        raise RuntimeError("synthetic OME-XML failure")

    monkeypatch.setattr(ngff_, "build_ome_xml", _boom)
    with pytest.raises(RuntimeError):
        plate.save2zarr(tmp_path / "p.ome.zarr")
    assert not (tmp_path / "p.ome.zarr").exists(), (
        "a failed write must leave no store -- the .part is never promoted"
    )
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/core/test_image_zarr_roundtrip.py -v
```

Expected: FAIL with `AttributeError: 'Image' object has no attribute 'save2zarr'`.

- [ ] **Step 3: Implement `save2zarr` / `load_zarr` / `load_layer_zarr`**

Add to `ImageIOHandler` (place `save2zarr` immediately after `save2hdf5` at line 871, and
`load_zarr` / `load_layer_zarr` after `load_layer_hdf5` at line 1194), so the HDF and store
paths sit side by side until Phase 6 removes the former.

```python
    def _series_names(self) -> list[str]:
        """Series this image will write, in canonical order.

        ``rgb`` is omitted entirely when empty; ``gray`` then becomes the
        primary series and the objmap label attaches under it.
        """
        from phenotypic.sdk_.ngff_ import SERIES_ORDER

        return [
            name
            for name in SERIES_ORDER
            if name != "rgb" or not self.rgb.isempty()
        ]

    def _write_series(
        self, part: Path, series: str, array: np.ndarray, levels: int
    ) -> None:
        """Write every pyramid level of one series (or label) into *part*.

        Args:
            part: The ``.part`` directory being built.
            series: Group path relative to *part*.
            array: Level-0 array.
            levels: Level count, uniform across the store.
        """
        import zarr

        from phenotypic.sdk_ import ngff_

        kind = "label" if series.endswith(ngff_.OBJMAP_LABEL) else "image"
        name = ngff_.OBJMAP_LABEL if kind == "label" else series
        for index, level in enumerate(ngff_.build_pyramid(array, levels, kind=kind)):
            handle = zarr.create_array(
                store=ngff_.long_path(part / series / str(index)),
                **ngff_.array_create_kwargs(level.shape, level.dtype, name),
            )
            handle[...] = level

    def save2zarr(
        self,
        path,
        *,
        work_id: str | None = None,
        durable: bool | None = None,
    ) -> Path:
        """Save the image as an OME-Zarr (NGFF 0.5 / Zarr v3) store.

        Builds a uuid-suffixed ``.part`` sibling, writes every array, then
        ``OME/zarr.json``, then the root ``zarr.json`` **last**, then promotes
        by directory rename. An interrupted write leaves no valid root, so the
        store reads as absent rather than as partial.

        Args:
            path: Target ``*.ome.zarr`` directory. Created or replaced.
            work_id: CLI work id, written into ``attributes.phenotypic`` at
                build time. Never patched in afterwards.
            durable: ``fsync`` before promoting. ``None`` auto-detects SLURM.

        Returns:
            The promoted store path.

        Examples:
            Save a synthetic plate and read it back:

            >>> from phenotypic.data import load_synth_yeast_plate
            >>> img = Image(load_synth_yeast_plate())
            >>> store = img.save2zarr('plate.ome.zarr')
            >>> reloaded = Image.load_zarr(store)
            >>> reloaded.gray[:].shape == img.gray[:].shape
            True
        """
        import json
        import logging
        from pathlib import Path as _Path

        from phenotypic.sdk_ import ngff_

        final = _Path(path)
        final.parent.mkdir(parents=True, exist_ok=True)
        part = ngff_.new_part_path(final)

        series_names = self._series_names()
        primary = ngff_.primary_series(series_names)
        gray = self.gray[:]
        levels = ngff_.pyramid_level_count(gray.shape[0], gray.shape[1])

        arrays: dict[str, np.ndarray] = {"gray": gray, "detect_mat": self.detect_mat[:]}
        if "rgb" in series_names:
            arrays["rgb"] = np.moveaxis(self.rgb[:], -1, 0)  # (H,W,C) -> (c,y,x)

        # 1. arrays and chunks
        for name in series_names:
            self._write_series(part, name, arrays[name], levels)
        objmap = self.objmap[:]
        self._write_series(part, ngff_.objmap_path(primary), objmap, levels)

        # 2. per-group ome metadata
        bit_depth = int(self.bit_depth or 8)
        name = self._metadata.protected.get("Metadata_ImageName")
        for series in series_names:
            shapes = ngff_.pyramid_level_shapes(arrays[series].shape, levels)
            block = {
                "version": ngff_.NGFF_VERSION,
                **ngff_.build_multiscales(
                    series=series, level_shapes=shapes, name=name
                ),
                **ngff_.build_omero(
                    series=series,
                    dtype=arrays[series].dtype,
                    bit_depth=bit_depth,
                    name=name,
                ),
            }
            self._write_group_json(part / series, {"ome": block})
        label_shapes = ngff_.pyramid_level_shapes(objmap.shape, levels)
        self._write_group_json(
            part / primary / ngff_.LABELS_GROUP,
            {"ome": {"version": ngff_.NGFF_VERSION, "labels": [ngff_.OBJMAP_LABEL]}},
        )
        self._write_group_json(
            part / ngff_.objmap_path(primary),
            {
                "ome": {
                    "version": ngff_.NGFF_VERSION,
                    # `name` here too: §2.4's "SHOULD contain the field 'name'"
                    # is not scoped to image series, and every sibling group
                    # honours it (ledger ALGO-R2B-16).
                    **ngff_.build_multiscales(
                        series=ngff_.OBJMAP_LABEL,
                        level_shapes=label_shapes,
                        name=f"{name}/{ngff_.OBJMAP_LABEL}" if name else None,
                    ),
                    **ngff_.build_image_label(),
                }
            },
        )

        # 3. OME/ group -- all or nothing
        sections = {
            "protected": dict(self._metadata.protected),
            "public": dict(self._metadata.public),
            "imported": dict(self._metadata.imported),
        }
        xml = ngff_.build_ome_xml(
            series_names=series_names,
            series_shapes={name: arrays[name].shape for name in series_names},
            series_dtypes={name: arrays[name].dtype for name in series_names},
            metadata_sections=sections,
        )
        ome_root: dict = {
            "version": ngff_.NGFF_VERSION,
            "bioformats2raw.layout": ngff_.BIOFORMATS2RAW_LAYOUT,
        }
        # Unconditional: `build_ome_xml` raises rather than returning None
        # (user ruling, ALGO-1), so there is no fallback branch. The branch an
        # earlier draft had here also did not do what it claimed -- it kept the
        # named rgb/gray/detect_mat groups, which is NOT the consecutive-integer
        # form NGFF 2.2.3 requires when `series` is absent.
        (part / ngff_.OME_GROUP).mkdir(parents=True, exist_ok=True)
        (part / ngff_.OME_GROUP / ngff_.OME_XML_NAME).write_text(xml, encoding="utf-8")
        self._write_group_json(
            part / ngff_.OME_GROUP,
            {"ome": {"version": ngff_.NGFF_VERSION, "series": series_names}},
        )

        # 4. root zarr.json LAST
        self._write_group_json(
            part,
            {
                "ome": ome_root,
                ngff_.PhenotypicAttr.ROOT: self._build_store_attributes(
                    series_names=series_names,
                    levels=levels,
                    sections=sections,
                    work_id=work_id,
                ),
            },
        )
        return ngff_.promote_store(
            part, final, fsync=ngff_.durable_writes_enabled(durable)
        )

    @staticmethod
    def _write_group_json(group_dir: Path, attributes: dict) -> None:
        """Write a Zarr v3 group ``zarr.json`` carrying *attributes*."""
        import json
        from pathlib import Path as _Path

        group_dir = _Path(group_dir)
        group_dir.mkdir(parents=True, exist_ok=True)
        (group_dir / "zarr.json").write_text(
            json.dumps(
                {"zarr_format": 3, "node_type": "group", "attributes": attributes},
                indent=2,
            ),
            encoding="utf-8",
        )

    def _build_store_attributes(
        self, *, series_names, levels, sections, work_id
    ) -> dict:
        """Assemble ``attributes.phenotypic``. Overridden by ``GridImage``."""
        from phenotypic.sdk_ import ngff_

        return ngff_.build_phenotypic_attributes(
            image_class=type(self).__name__,
            series_names=series_names,
            pyramid_levels=levels,
            metadata_sections=sections,
            detect_mode=self._data.detect_mode,
            illuminant=str(self.illuminant) if self.illuminant is not None else None,
            gamma=(
                self.gamma.name if hasattr(self.gamma, "name") else str(self.gamma)
            )
            if self.gamma is not None
            else None,
            work_id=work_id,
        )
```

`load_zarr` / `load_layer_zarr`:

```python
    @classmethod
    def load_zarr(cls, path, **kwargs) -> Image:
        """Load an image from an OME-Zarr store.

        Reads only level 0. ``attributes.phenotypic`` is the sole source of
        truth; the OME projection is never read back.

        Args:
            path: Path to a ``*.ome.zarr`` directory.
            **kwargs: Forwarded to the constructor, taking priority over
                anything recovered from the store.

        Returns:
            An :class:`Image` (or :class:`GridImage`, via the subclass).
        """
        import warnings

        from phenotypic.sdk_ import ngff_

        block = ngff_.read_phenotypic_attributes(path)
        saved_class = block.get(ngff_.PhenotypicAttr.IMAGE_CLASS)
        if saved_class == "GridImage" and cls.__name__ != "GridImage":
            warnings.warn(
                "Store was saved as GridImage; use GridImage.load_zarr to "
                "preserve grid state",
                UserWarning,
                stacklevel=2,
            )
        return cls._load_from_store(path, block, **kwargs)

    @classmethod
    def load_layer_zarr(cls, path, layer: str, level: int = 0) -> np.ndarray:
        """Read one layer at one pyramid level without building an Image.

        ``objmap`` is resolved through ``phenotypic.labels.objmap``, never by a
        hard-coded ``rgb/labels/objmap`` -- an rgb-less store puts the label
        under ``gray``.

        Args:
            path: Store path.
            layer: ``"rgb"``, ``"gray"``, ``"detect_mat"``, or ``"objmap"``.
            level: Pyramid level index; 0 is full resolution.

        Returns:
            The layer array. ``rgb`` is returned as ``(H, W, C)``.

        Raises:
            KeyError: If *layer* is not present in the store.
        """
        import zarr

        from phenotypic.sdk_ import ngff_

        block = ngff_.read_phenotypic_attributes(path)
        member = block[ngff_.PhenotypicAttr.SERIES].get(layer) or block[
            ngff_.PhenotypicAttr.LABELS
        ].get(layer)
        if member is None:
            raise KeyError(f"Layer {layer!r} not found in {path}")
        from pathlib import Path as _Path

        array = zarr.open_array(
            store=ngff_.long_path(_Path(path) / member / str(level)), mode="r"
        )[...]
        return np.moveaxis(array, 0, -1) if layer == "rgb" else array
```

`_load_from_store` follows the *shape* of `_load_v2_grouped` (line 984) — read each series'
level-0 array, restore the metadata sections, apply `detect_mode`, `illuminant`, `gamma`,
and `bit_depth` — but **must not copy its metadata merge**, which drops values. Write it
directly below `_load_v2_grouped` so the two read paths are visibly parallel.

> **Do not mirror `_image_io_handler.py:1071-1073`:**
>
> ```python
> for mapped, value in decoded.items():
>     if mapped in target and target[mapped] is not None:
>         continue          # <-- the constructor already set it; stored value is DROPPED
>     target[mapped] = value
> ```
>
> The constructor sets `Metadata_ImageType` before the loader runs
> (`_image_data_manager.py:104,156`; `_grid_image_handler.py:101`), so the stored value
> never lands. **Verified by execution** on the HDF path in this worktree:
> `before ImageType: GridSection` → `after ImageType: Image`.
>
> Spec §2.1 requires `image_class` and `Metadata_ImageType` to stay distinct and §7 mandates
> a test asserting both survive independently — so mirroring this bug makes
> `test_image_class_and_image_type_are_independent` (below) fail.
>
> `_load_from_store` therefore **assigns the three stored sections verbatim**, overwriting
> constructor defaults rather than deferring to them. This makes the store read path more
> correct than the HDF one; that divergence is deliberate and recorded as OPEN-QUESTIONS
> **D7**. The HDF loader is not fixed here — it is retired in Phase 6.
>
> `_metadata` also has a fourth section, `private`, which the HDF writer does not persist
> and which the store does not persist either. That is a deliberate carry-over, not an
> oversight; say so in the docstring so a later reader does not "complete" the set.

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/core/test_image_zarr_roundtrip.py -v
```

Expected: all PASS. Remove the `xfail` marker added in Task 2.1 Step 4 and re-run
`tests/unit/sdk_/test_bundle_layout.py`.

- [ ] **Step 5: Run the doctest for the new example**

```bash
uv run pytest --doctest-modules src/phenotypic/_core/_image_parts/_image_io_handler.py -q
```

Expected: PASS. The `save2zarr` docstring example must actually run — CLAUDE.md requires
runnable doctests using `load_synth_yeast_plate()`.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_core/_image_parts/_image_io_handler.py tests/unit/core/test_image_zarr_roundtrip.py tests/unit/sdk_/test_bundle_layout.py
git commit -m "feat(core): add save2zarr / load_zarr / load_layer_zarr

Arrays, then OME/zarr.json, then the root zarr.json last, then a rename
promote -- so an interrupted write reads as absent, not as partial. rgb is
omitted entirely when empty, which moves the primary series and the objmap
label to gray; readers resolve the label through phenotypic.labels.objmap
rather than a hard-coded path. objmap is always written, zeros included,
because valid_staged_store requires it after Stage 1. A failed OME-XML
build drops the whole OME/ group rather than emitting a partial one."
```

---

### Task 2.3: `GridImage` grid state

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/_grid_image_handler.py`
  (mirror `_save_image2hdfgroup` at line 464 and `_load_from_hdf5_group` at line 509)
- Test: `tests/unit/core/test_grid_image_zarr_roundtrip.py` (create)

**Interfaces:**
- Consumes: `Image._build_store_attributes`, `Image._load_from_store`.
- Produces: `GridImage._build_store_attributes` (override, adds `phenotypic.grid`) and
  `GridImage._load_from_store` (override, reads it back).

**Constraints specific to this task:**
- `phenotypic.grid` is `{"nrows": int, "ncols": int, "grid_finder": {"class": …, "params": …}}`,
  serialized with `SerializablePipeline._serialize_single_operation` exactly as the HDF
  path does at line 495.
- Deserialization failure warns and falls back to the default finder — mirroring line 543 —
  rather than raising. Grid state is recoverable; the image is not worth losing over it.
- `grid.nrows`/`ncols` are **not** projected as NGFF `plate` metadata (locked §2.5): HCS
  requires one image group per well, which would multiply group count by `nrows × ncols`
  for no reader benefit.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/core/test_grid_image_zarr_roundtrip.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from phenotypic import GridImage
from phenotypic.grid import AutoGridFinder
from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes
from phenotypic.data import load_synth_yeast_plate


@pytest.fixture
def grid() -> GridImage:
    return GridImage(load_synth_yeast_plate(), nrows=16, ncols=24)


def test_grid_dimensions_round_trip(grid: GridImage, tmp_path: Path) -> None:
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    back = GridImage.load_zarr(store)
    assert (back.nrows, back.ncols) == (16, 24)


def test_grid_finder_round_trips_by_class_and_params(tmp_path: Path) -> None:
    grid = GridImage(load_synth_yeast_plate(), grid_finder=AutoGridFinder(nrows=8, ncols=12))
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    back = GridImage.load_zarr(store)
    assert type(back.grid_finder).__name__ == "AutoGridFinder"
    assert (back.grid_finder.nrows, back.grid_finder.ncols) == (8, 12)


def test_grid_block_lives_under_phenotypic_not_ome(grid: GridImage, tmp_path: Path) -> None:
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    root = json.loads((store / "zarr.json").read_text(encoding="utf-8"))
    assert PhenotypicAttr.GRID in root["attributes"]["phenotypic"]
    assert "plate" not in json.dumps(root["attributes"]["ome"])


def test_no_hcs_plate_metadata_is_emitted(grid: GridImage, tmp_path: Path) -> None:
    """HCS would need one image group per well: 16x24 = 384 groups, no benefit."""
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    groups = [p.name for p in store.iterdir() if p.is_dir()]
    assert set(groups) <= {"OME", "rgb", "gray", "detect_mat"}


def test_corrupt_grid_finder_warns_and_falls_back(grid: GridImage, tmp_path: Path) -> None:
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    root_path = store / "zarr.json"
    payload = json.loads(root_path.read_text(encoding="utf-8"))
    payload["attributes"]["phenotypic"][PhenotypicAttr.GRID]["grid_finder"] = {
        "class": "NoSuchFinder",
        "params": {},
    }
    root_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.warns(UserWarning, match="GridFinder"):
        back = GridImage.load_zarr(store)
    assert back.grid_finder is not None


def test_explicit_kwargs_take_priority_over_stored_grid(
    grid: GridImage, tmp_path: Path
) -> None:
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    back = GridImage.load_zarr(store, nrows=8, ncols=12)
    assert (back.nrows, back.ncols) == (8, 12)


def test_image_class_records_gridimage(grid: GridImage, tmp_path: Path) -> None:
    store = grid.save2zarr(tmp_path / "g.ome.zarr")
    block = read_phenotypic_attributes(store)
    assert block[PhenotypicAttr.IMAGE_CLASS] == "GridImage"
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/core/test_grid_image_zarr_roundtrip.py -v
```

Expected: FAIL — `phenotypic.grid` is absent from the block.

- [ ] **Step 3: Implement the overrides**

```python
    def _build_store_attributes(self, *, series_names, levels, sections, work_id) -> dict:
        """Add ``phenotypic.grid`` to the base attributes block.

        ``nrows``/``ncols`` are deliberately **not** projected as NGFF ``plate``
        metadata: HCS requires each well to be a separate image group, while
        PhenotypicTypic's grid is a virtual partition of one array.
        """
        from phenotypic._core._pipeline_parts._serializable_pipeline import (
            SerializablePipeline,
        )
        from phenotypic.sdk_.ngff_ import PhenotypicAttr

        block = super()._build_store_attributes(
            series_names=series_names,
            levels=levels,
            sections=sections,
            work_id=work_id,
        )
        grid: dict = {"nrows": int(self.nrows), "ncols": int(self.ncols)}
        if self.grid_finder is not None:
            grid["grid_finder"] = {
                "class": type(self.grid_finder).__name__,
                "params": SerializablePipeline._serialize_single_operation(
                    self.grid_finder
                ),
            }
        block[PhenotypicAttr.GRID] = grid
        return block

    @classmethod
    def _load_from_store(cls, path, block, **kwargs):
        """Restore ``nrows``/``ncols``/``grid_finder`` before the base loader.

        Uses ``setdefault`` so explicit caller kwargs take priority, mirroring
        the HDF path. A deserialization failure warns and falls back to the
        default finder rather than raising: grid state is recoverable, the
        image is not worth losing over it.
        """
        import warnings

        from phenotypic.sdk_.ngff_ import PhenotypicAttr

        grid = block.get(PhenotypicAttr.GRID) or {}
        for key in ("nrows", "ncols"):
            if key in grid:
                try:
                    kwargs.setdefault(key, int(grid[key]))
                except (TypeError, ValueError):
                    pass
        payload = grid.get("grid_finder")
        if payload:
            from phenotypic._core._pipeline_parts._serializable_pipeline import (
                SerializablePipeline,
            )

            try:
                kwargs.setdefault(
                    "grid_finder",
                    SerializablePipeline._deserialize_operations({"__gf__": payload})[
                        "__gf__"
                    ],
                )
            except (KeyError, AttributeError, TypeError, ValueError) as exc:
                warnings.warn(
                    f"GridFinder deserialization failed "
                    f"({type(exc).__name__}: {exc}); falling back to the default "
                    "AutoGridFinder. Grid configuration may be incorrect.",
                    UserWarning,
                    stacklevel=2,
                )
        return super()._load_from_store(path, block, **kwargs)
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/core/test_grid_image_zarr_roundtrip.py tests/unit/core/test_image_zarr_roundtrip.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_core/_image_parts/_grid_image_handler.py tests/unit/core/test_grid_image_zarr_roundtrip.py
git commit -m "feat(core): persist grid state in attributes.phenotypic.grid

nrows/ncols/grid_finder round-trip through the phenotypic block, not
through NGFF plate metadata: HCS requires one image group per well, which
would multiply a 16x24 store's group count by 384 for no reader benefit.
A corrupt grid_finder warns and falls back, as the HDF path already does."
```

---

### Task 2.4: `save_intermediate_zarr` for builder node previews

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/_image_io_handler.py`
  (replace `save_intermediate_layers`, line 905)
- Modify: `src/phenotypic/_core/_pipeline_parts/_image_pipeline_core.py`
  (lines 1021, 1024, 1042, 1046, 1052 — **five** call sites, not three: two use
  `save2hdf5` and three use `save_intermediate_layers`)
- Modify: `src/phenotypic/gui/builder/_preview_cache.py` (lines 208, 212, 217, 284–286 —
  the DAG manifest's `"hdf"` key and the `base_00.h5` / `{i:02d}_{key}.h5` filenames)
- Test: `tests/unit/core/test_save_intermediate_zarr.py` (create),
  `tests/unit/gui/builder/test_preview_cache_manifest.py` (extend)

**Interfaces:**
- Consumes: `save2zarr` internals.
- Produces:
  ```python
  def save_intermediate_zarr(self, path, layers: tuple[str, ...]) -> Path
  ```
  and a builder DAG manifest whose per-node key is `"store"` (not `"hdf"`) holding
  `"<name>.ome.zarr"`.

**Constraints specific to this task:**
- `save_intermediate_zarr` writes a **single-level, no-pyramid** store —
  `levels=1` — a **private** argument on `_save_store`, not a CLI flag. Node previews are
  transient and small; pyramiding them would multiply builder-cache inodes for no gain.
- It still goes through the promote primitive: builder previews are written concurrently by
  Dash callbacks, so a half-written directory is a live failure mode there too.
- The manifest key rename `"hdf"` → `"store"` is a **GUI-visible contract change**.
  `_preview_cache.py:284-286` reads `parent_manifest["nodes"][pred_id]["hdf"]`, and
  `_preview_tiles.py:124` reads `node["hdf"]`. Both must change in the same commit, and any
  on-disk manifest from a previous session is invalidated — bump the manifest version
  constant so a stale cache is rebuilt rather than misread.
- Unknown layer names still raise `ValueError`, as `save_intermediate_layers` does at
  line 934.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/core/test_save_intermediate_zarr.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from phenotypic import Image
from phenotypic.sdk_.ngff_ import PhenotypicAttr, read_phenotypic_attributes
from phenotypic.data import load_synth_yeast_plate


@pytest.fixture
def plate() -> Image:
    return Image(load_synth_yeast_plate())


def test_writes_only_the_requested_layers(plate: Image, tmp_path: Path) -> None:
    store = plate.save_intermediate_zarr(tmp_path / "n.ome.zarr", layers=("gray",))
    block = read_phenotypic_attributes(store)
    assert set(block[PhenotypicAttr.SERIES]) == {"gray"}


def test_is_single_level_by_design(plate: Image, tmp_path: Path) -> None:
    """Node previews are transient; pyramiding them multiplies cache inodes."""
    store = plate.save_intermediate_zarr(
        tmp_path / "n.ome.zarr", layers=("gray", "detect_mat")
    )
    assert read_phenotypic_attributes(store)[PhenotypicAttr.PYRAMID]["levels"] == 1
    assert not (store / "gray" / "1").exists()


def test_round_trips_through_load_layer_zarr(plate: Image, tmp_path: Path) -> None:
    store = plate.save_intermediate_zarr(tmp_path / "n.ome.zarr", layers=("gray",))
    np.testing.assert_array_equal(Image.load_layer_zarr(store, "gray"), plate.gray[:])


def test_unknown_layer_names_raise(plate: Image, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown layer"):
        plate.save_intermediate_zarr(tmp_path / "n.ome.zarr", layers=("nope",))


def test_uses_the_promote_primitive(plate: Image, tmp_path: Path, monkeypatch) -> None:
    """Dash callbacks write these concurrently; a half-written dir is live risk."""
    from phenotypic.sdk_ import ngff_

    calls: list[str] = []
    real = ngff_.promote_store
    monkeypatch.setattr(
        ngff_,
        "promote_store",
        lambda part, final, *, fsync: (calls.append(final.name), real(part, final, fsync=fsync))[1],
    )
    plate.save_intermediate_zarr(tmp_path / "n.ome.zarr", layers=("gray",))
    assert calls == ["n.ome.zarr"]
```

Extend `tests/unit/gui/builder/test_preview_cache_manifest.py`:

```python
def test_manifest_node_key_is_store_not_hdf(tmp_path) -> None:
    """GUI-visible contract change; a stale 'hdf' key must not be read."""
    from phenotypic.gui.builder import _preview_cache

    manifest = _preview_cache.build_manifest_for_test(tmp_path)  # existing helper
    for node in manifest["nodes"].values():
        assert "hdf" not in node
        assert node["store"].endswith(".ome.zarr")


def test_manifest_version_bumped_so_stale_caches_rebuild() -> None:
    from phenotypic.gui.builder import _preview_cache

    assert _preview_cache.MANIFEST_VERSION >= 2
```

- [ ] **Step 2: Run them to verify they fail**

```bash
uv run pytest tests/unit/core/test_save_intermediate_zarr.py tests/unit/gui/builder/test_preview_cache_manifest.py -v
```

Expected: `AttributeError: 'Image' object has no attribute 'save_intermediate_zarr'` and a
manifest still carrying `"hdf"`.

- [ ] **Step 3: Implement**

Replace `save_intermediate_layers` with:

```python
    def save_intermediate_zarr(self, path, layers: tuple[str, ...]) -> Path:
        """Save only *layers* as a single-level OME-Zarr store.

        Used for GUI builder node previews. No pyramid: previews are transient
        and small, and pyramiding them would multiply builder-cache inodes for
        no reader benefit. The promote primitive is still used, because Dash
        callbacks write these concurrently.

        Args:
            path: Target ``*.ome.zarr`` directory.
            layers: Subset of ``("rgb", "gray", "detect_mat", "objmap")``.

        Returns:
            The promoted store path.

        Raises:
            ValueError: If *layers* contains unknown names.
        """
        valid = {"rgb", "gray", "detect_mat", "objmap"}
        unknown = set(layers) - valid
        if unknown:
            raise ValueError(f"Unknown layer names: {unknown}")
        return self._save_store(
            path, series=tuple(layers), levels=1, work_id=None, durable=False
        )
```

Refactor `save2zarr`'s body into `_save_store(path, *, series, levels, work_id, durable)`
so both entry points share one implementation. `save2zarr` calls it with
`series=tuple(self._series_names())` and `levels=ngff_.pyramid_level_count(...)`;
`save_intermediate_zarr` calls it with `levels=1`. **`levels` is private** — there is no
CLI or public-API lever, so two stores in one tree can never disagree (P3).

In `_image_pipeline_core.py`, change all five call sites. The two `save2hdf5(...)` calls at
lines 1021 and 1042 become `save2zarr(output_dir / "base_00.ome.zarr")` and
`save2zarr(output_dir / f"{i:02d}_{key}.ome.zarr")`; the three `save_intermediate_layers`
calls become `save_intermediate_zarr` with the same `layers=` argument and the
`.ome.zarr` name.

In `_preview_cache.py`, rename the manifest key and bump the version:

```python
#: Manifest schema version. Bumped when the per-node artifact moved from a
#: single ``.h5`` to an ``.ome.zarr`` store, so a manifest written by an older
#: session is rebuilt rather than misread.
MANIFEST_VERSION: Final[int] = 2
```

`_describe` writes `"store": filename` instead of `"hdf": filename`, and the predecessor
lookup at lines 284–286 reads `parent_manifest["nodes"][pred_id]["store"]`.

- [ ] **Step 4: Run the tests plus the builder GUI suite**

```bash
uv run pytest tests/unit/core/test_save_intermediate_zarr.py tests/unit/gui/builder -v
```

Expected: all PASS. `_preview_tiles.py:124`'s `node["hdf"]` will fail here if missed — fix
it in this task, not in Phase 4.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_core src/phenotypic/gui/builder tests/unit/core/test_save_intermediate_zarr.py tests/unit/gui/builder
git commit -m "feat(core): replace save_intermediate_layers with save_intermediate_zarr

Node previews become single-level stores -- transient and small, so a
pyramid would only multiply builder-cache inodes. All five pipeline call
sites move (three save_intermediate_layers plus two save2hdf5, which an
earlier count missed). The builder DAG manifest's per-node key becomes
'store'; MANIFEST_VERSION is bumped so a manifest from an older session is
rebuilt rather than misread through a key that no longer exists."
```

---

### Task 2.5: NGFF conformance harness

**Files:**
- **Extend**: `tests/_ngff_conformance.py` — **Phase 1 Task 1.4 creates it** with
  `_ome_xsd()` and `assert_ome_xml_valid`; this task adds the JSON-schema half
  (`_schema`, `_registry`, `_validate`, `_attributes`) plus `assert_store_conforms` and
  `_assert_reader_level_musts` on top.

  > **Ordering corrected (ledger ALGO-R2B-10).** An earlier draft created the whole module
  > here, while Task 1.4's `test_ome_xml_validates_against_the_vendored_xsd` already imported
  > `assert_ome_xml_valid` from it. Phase 1's exit criterion runs
  > `uv run pytest tests/unit/sdk_/test_ngff_*.py -q`, so Phase 1 would have failed with
  > `ModuleNotFoundError` on the two tests that import it (the imports are function-local, so
  > collection itself succeeds — ledger **GEN-50**). That matters more than an ordinary ordering
  > slip: this is the ALGO-1 remediation, so the executing agent meets a red gate at exactly
  > the moment deleting the assertion looks reasonable. The XSD half depends only on the
  > Phase-0 vendored fixture and `xmlschema`, both already in place at Phase 1, so it belongs
  > in the commit that introduces the builder.
- Test: `tests/unit/core/test_ngff_conformance.py`

**Interfaces:**
- Consumes: `tests/fixtures/ngff/0.5/*.schema` (Task 0.2).
- Produces:
  ```python
  def assert_store_conforms(store_path: Path) -> None
  def assert_ome_xml_valid(xml: str) -> None
  ```
  imported by every later phase that writes a store.

**Constraints specific to this task:**
- Validation failure **fails the suite**. It is never downgraded to a warning and never
  skipped on a missing dependency or fixture — a check that cannot run must fail.
- Validate three surfaces: every image series' group `zarr.json` against `image.schema`,
  the label group's against `label.schema`, and `OME/zarr.json` against `ome.schema`.
- **Validate `payload["attributes"]`, NOT `payload["attributes"]["ome"]`.** Every one of the
  three schemas is rooted at the *attributes* object — verified by download:

  ```json
  {"$id": ".../image.schema", "description": "The zarr.json attributes key",
   "type": "object", "properties": {"ome": {…}}, "required": ["ome"]}
  ```

  Passing the inner `ome` block fails with `'ome' is a required property` on **every**
  store, so all seven tests here plus every downstream `assert_store_conforms` call in
  Phases 3 and 5 would fail. Recorded as OPEN-QUESTIONS **B2**.
- **Resolve `_version.schema` from the vendored copy, never over the network.** All three
  schemas carry exactly one remote `$ref` to it. `jsonschema` >= 4.18 does not fetch remote
  refs — it raises `referencing.exceptions.Unresolvable`, which is **not** a
  `ValidationError`, so an unregistered ref makes the suite *error* rather than fail and
  leaves offline CI with no fallback. Build a `referencing.Registry` keyed on each file's
  `$id` and pass it to the validator.
- The published schemas are stricter than the prose in two places (`ome.schema` requires
  `["series", "version"]`; `label.schema` requires `["image-label", "version"]`) and
  **looser** in two others (`$defs/image-label` has no `required` list, so `colors` is
  optional; `$defs/omero` requires only `channels`, and the channel item has no `required`
  list at all). The harness reports **NGFF conformance, schema-encoded or not** — the
  vendored schemas where they encode a rule, plus `_assert_reader_level_musts` for the MUSTs
  they do not (`datasets[].path` resolving, `dimension_names` vs `axes`, label reachability
  and dtype, path-vs-`Image` order, uniform chunk keys). PhenoTypic policy that is *not* an
  NGFF rule is still asserted separately, in Phase 1 Task 1.4's unit tests.

  > **Amended (ledger SIMP-12).** An earlier draft drew this boundary at "schemas only" and
  > gave the reader-level MUSTs their own Phase 7 task. Two problems: the round-2 delta had
  > already crossed that line by putting `assert_ome_xml_valid` here, and these are properties
  > of a **written store**, not of a builder function, so Task 1.4 was never a place they
  > could live. Folding them in gates *every* store written anywhere in Phases 2–7 instead of
  > one store in one Phase-7 test — strictly stronger coverage, one fewer task. The ALGO-4
  > user ruling is preserved in substance: a reader-level gate still exists.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/core/test_ngff_conformance.py`:

```python
"""Every written store must validate against the vendored NGFF schemas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phenotypic import GridImage, Image
from phenotypic.data import load_synth_yeast_plate
from tests._ngff_conformance import assert_store_conforms


def test_a_written_image_store_conforms(tmp_path: Path) -> None:
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    assert_store_conforms(store)


def test_a_written_grid_store_conforms(tmp_path: Path) -> None:
    store = GridImage(load_synth_yeast_plate(), nrows=8, ncols=12).save2zarr(
        tmp_path / "g.ome.zarr"
    )
    assert_store_conforms(store)


def test_an_rgb_less_store_conforms(tmp_path: Path) -> None:
    image = Image(np.asarray(Image(load_synth_yeast_plate()).gray[:]))
    assert_store_conforms(image.save2zarr(tmp_path / "gray.ome.zarr"))


def test_a_single_level_node_preview_store_conforms(tmp_path: Path) -> None:
    """The levels=1 path (builder node previews) must conform too."""
    store = Image(load_synth_yeast_plate()).save_intermediate_zarr(
        tmp_path / "n.ome.zarr", layers=("gray", "detect_mat", "objmap")
    )
    assert_store_conforms(store)


def test_a_remote_ref_resolves_from_the_vendored_copy(tmp_path: Path) -> None:
    """An unregistered $ref raises Unresolvable, which is NOT a ValidationError,
    so the suite would error rather than fail. Verified: all three schemas
    reference _version.schema by absolute URL."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    assert_store_conforms(store)  # would raise Unresolvable without the registry


def test_a_wrong_version_string_is_rejected(tmp_path: Path) -> None:
    """Proves the resolved _version.schema enum is actually enforced."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    group = store / "gray" / "zarr.json"
    payload = json.loads(group.read_text(encoding="utf-8"))
    payload["attributes"]["ome"]["version"] = "0.4"
    group.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError):
        assert_store_conforms(store)


def test_partial_omero_is_rejected(tmp_path: Path) -> None:
    """Proves the gate has teeth: the exact defect an earlier draft would ship.

    Note what is and is not schema-enforced here. `$defs/omero` requires only
    `channels`, and the channel item has no `required` list -- so a channel
    missing `color` would validate. But `window`, *if present*, requires all
    four of start/min/end/max, which is what this truncation violates. Emitting
    the complete block is PhenoTypic policy (asserted in Phase 1 Task 1.4);
    this test covers the part the schema does enforce.
    """
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    group = store / "gray" / "zarr.json"
    payload = json.loads(group.read_text(encoding="utf-8"))
    payload["attributes"]["ome"]["omero"]["channels"][0]["window"] = {"max": 255}
    group.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError):
        assert_store_conforms(store)


def test_missing_image_label_is_rejected(tmp_path: Path) -> None:
    """label.schema requires image-label even though the prose says SHOULD."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    label_json = store / "rgb" / "labels" / "objmap" / "zarr.json"
    payload = json.loads(label_json.read_text(encoding="utf-8"))
    del payload["attributes"]["ome"]["image-label"]
    label_json.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError):
        assert_store_conforms(store)


def test_a_reordered_series_list_is_rejected(tmp_path: Path) -> None:
    """§2.2.3: path order MUST match the Image element order.

    Without this the path-order assertion is satisfied vacuously by the four
    positive tests -- which is exactly how a KeyError in it shipped green once
    already (ledger GEN-47 / GEN-33).
    """
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    ome_json = store / "OME" / "zarr.json"
    payload = json.loads(ome_json.read_text(encoding="utf-8"))
    payload["attributes"]["ome"]["series"].reverse()
    ome_json.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="series order"):
        assert_store_conforms(store)


def test_a_dangling_dataset_path_is_rejected(tmp_path: Path) -> None:
    """A reader follows datasets[].path; a dangling one is a broken store."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    group = store / "gray" / "zarr.json"
    payload = json.loads(group.read_text(encoding="utf-8"))
    payload["attributes"]["ome"]["multiscales"][0]["datasets"][0]["path"] = "9"
    group.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Exception):  # zarr raises before our assert
        assert_store_conforms(store)


def test_a_dimension_names_mismatch_is_rejected(tmp_path: Path) -> None:
    """§2.1 MUST -- and the only other assertion of it checks the builder."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    level = store / "gray" / "0" / "zarr.json"
    payload = json.loads(level.read_text(encoding="utf-8"))
    payload["dimension_names"] = list(reversed(payload["dimension_names"]))
    level.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError):
        assert_store_conforms(store)


def test_a_nested_chunk_key_separator_is_rejected(tmp_path: Path) -> None:
    """Design spec §1.4: the separator MUST be uniform store-wide."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    level = store / "gray" / "0" / "zarr.json"
    payload = json.loads(level.read_text(encoding="utf-8"))
    payload["chunk_key_encoding"] = {
        "name": "default",
        "configuration": {"separator": "/"},
    }
    level.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError, match="separator"):
        assert_store_conforms(store)


def test_a_label_less_store_still_conforms(tmp_path: Path) -> None:
    """save_intermediate_zarr(layers=("gray",)) writes one, and it is VALID.

    The reader-level fold turned this from tolerated into FileNotFoundError
    once already (ledger GEN-33).
    """
    image = Image(load_synth_yeast_plate())
    store = image.save_intermediate_zarr(tmp_path / "i.ome.zarr", layers=("gray",))
    assert_store_conforms(store)


def test_missing_series_is_rejected(tmp_path: Path) -> None:
    """ome.schema requires series."""
    store = Image(load_synth_yeast_plate()).save2zarr(tmp_path / "p.ome.zarr")
    ome_json = store / "OME" / "zarr.json"
    payload = json.loads(ome_json.read_text(encoding="utf-8"))
    del payload["attributes"]["ome"]["series"]
    ome_json.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AssertionError):
        assert_store_conforms(store)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
uv run pytest tests/unit/core/test_ngff_conformance.py -v
```

Expected: `ImportError: cannot import name 'assert_store_conforms' from 'tests._ngff_conformance'` — the module itself exists after Task 1.4 (ledger **GEN-49**).

- [ ] **Step 3: Write the harness**

Extend `tests/_ngff_conformance.py` (Task 1.4 created it with `_ome_xsd` and
`assert_ome_xml_valid`; both are shown again here so this block reads as the finished file):

```python
"""Validate a written store against the vendored NGFF 0.5 JSON schemas.

Conformance is checked with ``jsonschema`` rather than ``ome-zarr-models``,
which pins ``pydantic<2.13``; pydantic 2.13 has already shipped, so adopting it
would hold the project a release behind today. There is no ``[tool.uv]
conflicts`` block, so even a dev-group-only cap would bind the whole locked
environment.

A failure here fails the suite. It is never downgraded to a warning and never
skipped: a check that cannot run must fail.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import jsonschema
import referencing
import referencing.jsonschema

SCHEMA_DIR = Path(__file__).resolve().parent / "fixtures" / "ngff" / "0.5"


SCHEMA_NAMES = ("image.schema", "label.schema", "ome.schema", "_version.schema")


def _schema(name: str) -> dict:
    path = SCHEMA_DIR / name
    if not path.is_file():
        raise AssertionError(
            f"vendored NGFF schema missing: {path}. A conformance check that "
            "cannot run must fail, never skip."
        )
    return json.loads(path.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def _registry() -> referencing.Registry:
    """Resolve every ``$ref`` from the vendored copies, never over the network.

    All three schemas reference ``_version.schema`` by absolute URL. jsonschema
    >= 4.18 does not fetch remote refs; it raises
    ``referencing.exceptions.Unresolvable``, which is not a ``ValidationError``
    -- so without this the suite errors instead of failing, and offline CI has
    no fallback at all.
    """
    registry = referencing.Registry()
    for name in SCHEMA_NAMES:
        payload = _schema(name)
        resource = referencing.Resource.from_contents(
            payload, default_specification=referencing.jsonschema.DRAFT202012
        )
        registry = resource @ registry
    return registry


def _attributes(group_dir: Path) -> dict:
    """Return the group's whole ``attributes`` mapping.

    NOT ``attributes["ome"]``: each vendored schema is rooted at the attributes
    object itself (``"description": "The zarr.json attributes key"``,
    ``"required": ["ome"]``). Passing the inner block fails with
    ``'ome' is a required property`` on every store ever written.
    """
    payload = json.loads((group_dir / "zarr.json").read_text(encoding="utf-8"))
    return payload["attributes"]


def _validate(attributes: dict, schema_name: str, where: Path) -> None:
    schema = _schema(schema_name)
    validator = jsonschema.Draft202012Validator(schema, registry=_registry())
    try:
        validator.validate(attributes)
    except jsonschema.ValidationError as exc:
        raise AssertionError(
            f"{where} does not conform to {schema_name}: {exc.message} "
            f"(at {list(exc.absolute_path)})"
        ) from exc


@functools.lru_cache(maxsize=1)
def _ome_xsd() -> "xmlschema.XMLSchema":
    """Load the vendored OME-XML schema.

    NGFF §2.2.3 makes the document a conditional MUST, and JSON-schema
    validation says nothing about it -- an earlier draft emitted a `<Pixels />`
    with none of its eight required attributes and no `<MetadataOnly/>`, and the
    only test counted `"<Image "` occurrences.
    """
    import xmlschema

    path = Path(__file__).resolve().parent / "fixtures" / "ome" / "2016-06" / "ome.xsd"
    if not path.is_file():
        raise AssertionError(
            f"vendored OME schema missing: {path}. A conformance check that "
            "cannot run must fail, never skip."
        )
    return xmlschema.XMLSchema(str(path))


def assert_ome_xml_valid(xml: str) -> None:
    """Validate an OME-XML document against the vendored `ome.xsd`.

    Catches ``XMLSchemaException``, not ``XMLSchemaValidationError``. The
    narrower class does NOT cover a well-formedness failure, which raises
    ``XMLResourceParseError`` (MRO: ``XMLResourceError`` -> ``XMLSchemaException``
    -> ``ElementTree.ParseError`` -> ``SyntaxError``) -- and that is the *most
    likely* real failure, since a control character in an imported EXIF tag
    breaks well-formedness rather than schema conformance. With the narrow
    except, the documented contract below simply would not hold. Ledger
    **ALGO-R2B-11**.

    Args:
        xml: The document as a string.

    Raises:
        AssertionError: On any schema violation OR malformed document.
    """
    import xmlschema

    try:
        _ome_xsd().validate(xml)
    except xmlschema.XMLSchemaException as exc:
        # Name the exception type: the widened catch also takes
        # XMLSchemaOSError/XMLResourceOSError, and a disk error reported as
        # "not valid against ome.xsd" would be misread as a conformance bug.
        raise AssertionError(
            f"OME-XML failed validation against ome.xsd "
            f"[{type(exc).__name__}]: {exc}"
        ) from exc


def assert_store_conforms(store_path: Path) -> None:
    """Validate every conformance surface of one written store.

    Args:
        store_path: Path to a promoted ``*.ome.zarr`` directory.

    Raises:
        AssertionError: On any schema violation, missing schema fixture, or
            structurally absent group.
    """
    from phenotypic.sdk_.ngff_ import (
        OME_GROUP,
        OME_XML_NAME,
        PhenotypicAttr,
        read_phenotypic_attributes,
    )

    store = Path(store_path)
    block = read_phenotypic_attributes(store)

    for member in block[PhenotypicAttr.SERIES].values():
        _validate(_attributes(store / member), "image.schema", store / member)

    for member in block[PhenotypicAttr.LABELS].values():
        _validate(_attributes(store / member), "label.schema", store / member)

    ome_group = store / OME_GROUP
    # An ASSERTION, not a guard (ledger ALGO-3 / ALGO-13a). The group is
    # unconditional now, so `if ome_group.is_dir():` would let a regression that
    # stopped writing it pass every conformance test in the suite.
    # NOT "mandatory under layout 3" -- 2.2.3 makes the OME metadata a SHOULD
    # ("SHOULD have OME metadata ... in a file named OME/METADATA.ome.xml") and
    # the series attribute a MAY. What makes the group mandatory for THIS store
    # is the named-series layout: 2.2.3 says "If the 'series' attribute does not
    # exist and no 'plate' is present: separate 'multiscales' images MUST be
    # stored in consecutively numbered groups starting from 0". This writer
    # emits rgb/gray/detect_mat, not 0/1/2, so `series` is load-bearing -- and
    # OME/ is the only place 2.2.3 puts it. Ledger ALGO-16: an earlier message
    # cited layout 3, which a reader checking 2.2.3 would find overreaching, and
    # would then soften back to `if ome_group.is_dir():` -- reinstating the
    # silently-skipped surface ALGO-3/ALGO-13(a) closed.
    assert ome_group.is_dir(), (
        f"OME/ group is mandatory for a NAMED-series store: NGFF 2.2.3 requires "
        f"consecutively numbered groups when 'series' is absent, and 'series' "
        f"lives here: {store}"
    )
    _validate(_attributes(ome_group), "ome.schema", ome_group)
    # The XML is a separate conformance surface from the JSON: §2.2.3's MUST
    # is about ome.xsd, which no JSON schema covers.
    ome_xml = (ome_group / OME_XML_NAME).read_text(encoding="utf-8")
    assert_ome_xml_valid(ome_xml)

    _assert_reader_level_musts(store, block, ome_xml)


def _group_ome(group_dir: Path) -> dict:
    payload = json.loads((group_dir / "zarr.json").read_text(encoding="utf-8"))
    return payload["attributes"]["ome"]


def _assert_reader_level_musts(store: Path, block: dict, ome_xml: str) -> None:
    """Assert the NGFF MUSTs that no vendored JSON schema encodes.

    These are properties of a *written store* as a third-party reader resolves
    it -- not of a builder function -- so they cannot live in Phase 1's unit
    tests. Uses ``zarr`` only; the ``ome-zarr`` ban holds.

    Args:
        store: The promoted ``*.ome.zarr`` directory.
        block: The store's ``attributes.phenotypic`` block.
        ome_xml: The already-read ``OME/METADATA.ome.xml`` document.

    Raises:
        AssertionError: On any violation.
    """
    import re

    import numpy as np
    import zarr

    from phenotypic.sdk_.ngff_ import (
        LABELS_GROUP,
        OBJMAP_LABEL,
        OME_GROUP,
        PhenotypicAttr,
        objmap_path,
        primary_series,
    )

    series_names = list(block[PhenotypicAttr.SERIES].values())
    primary = primary_series(series_names)
    # `objmap_path`, not a hand-built f-string: Task 1.3 declares it precisely
    # so no caller re-encodes the layout (ledger GEN-53).
    # Read the path the STORE DECLARES, rather than re-deriving it: a
    # re-derived path cannot fail, whereas this turns the loop below into a real
    # check that the declared label path resolves (ledger ALGO-20).
    labels = block[PhenotypicAttr.LABELS]
    label_member = labels.get(OBJMAP_LABEL) if labels else None
    # The LABEL GROUP IS IN THIS LOOP, deliberately (ledger ALGO-R2B-14).
    # `label.schema` declares only `image-label` and `version` under
    # `properties.ome` and sets no `additionalProperties: false`, so it says
    # NOTHING about the label's multiscales block -- while 2.6 requires that
    # "the zarr.json file for the label image MUST implement the multiscales
    # specification". Validated by neither schema nor any other check unless it
    # is iterated here.

    # 2.2.3: "'series' MUST be a list of string objects, each of which is a
    # path to an image group. The order of the paths MUST match the order of the
    # 'Image' elements in 'OME/METADATA.ome.xml' IF PROVIDED." (The "if
    # provided" clause was elided in an earlier draft -- ledger ALGO-17.)
    #
    # TWO assertions, because the published rule has two halves and the name
    # comparison only covers one:
    #
    #  (a) COUNT. 2.2.3 also says "Every 'multiscales' group MUST represent
    #      exactly one OME-XML 'Image'". `Name` is use="optional" in ome.xsd, so
    #      a name-only scrape cannot see an unnamed <Image> -- a document with
    #      three named Images matching `declared` plus a fourth unnamed one
    #      would pass the order check while violating the 1:1 MUST.
    #  (b) ORDER. Comparing Image/@Name to the path list is STRONGER than the
    #      spec demands: the MUST is positional, and @Name has no spec-defined
    #      relationship to the group path. It works because `build_ome_xml` sets
    #      Name={quoteattr(series)}. Keep it -- it catches a reordering AND a
    #      naming drift -- but it is THIS WRITER'S CONVENTION, not the rule.
    #
    # Both quote styles are matched: `quoteattr` switches to single quotes when
    # the value contains a double quote. (The document is already parsed by
    # xmlschema one line above; we scrape rather than re-parse because a second
    # stdlib XML parse over user-derived EXIF text has no billion-laughs guard.)
    # The OME GROUP, not the root. Task 2.2 writes the root as
    # {"version", "bioformats2raw.layout"} and puts `series` on OME/ -- which is
    # also what ome.schema requires and what test_missing_series_is_rejected
    # deletes. Reading it off the root raised KeyError on EVERY call, i.e. every
    # store-writing test in Phases 2, 3 and 5. Ledger GEN-33.
    declared = _group_ome(store / OME_GROUP)["series"]
    n_images = len(re.findall(r"<Image\b", ome_xml))
    assert n_images == len(declared), (
        f"2.2.3: every multiscales group MUST represent exactly one Image -- "
        f"{n_images} Image elements vs {len(declared)} series"
    )
    xml_order = [
        double or single
        for double, single in re.findall(
            r"""<Image\b[^>]*?\bName=(?:"([^"]*)"|'([^']*)')""", ome_xml
        )
    ]
    assert xml_order == declared, f"series order != Image order: {xml_order} vs {declared}"

    level_counts: dict[str, int] = {}
    # A label-less store is VALID and must stay tolerated (ledger GEN-33): the
    # old harness iterated `block[LABELS].values()`, a no-op on an empty
    # mapping, and `save_intermediate_zarr(layers=("gray",))` (Task 2.4) writes
    # exactly such a store. Folding the label into this loop unguarded turned
    # "tolerated" into FileNotFoundError.
    members = [*series_names, label_member] if label_member else list(series_names)
    for name in members:
        multiscale = _group_ome(store / name)["multiscales"][0]
        expected_axes = [axis["name"] for axis in multiscale["axes"]]
        level_counts[name] = len(multiscale["datasets"])
        for dataset in multiscale["datasets"]:
            level = store / name / dataset["path"]
            # A reader follows datasets[].path; a dangling one is a broken store.
            array = zarr.open_array(store=str(level), mode="r")
            # 2.4: "MUST have the same number of dimensions ... The number of
            # dimensions and order MUST correspond to number and order of
            # 'axes'." `assert array.shape` was a tautology -- every zarr array
            # has a non-empty shape tuple (ledger ALGO-18). The real check that
            # `datasets[].path` resolves is that `open_array` did not raise.
            assert len(array.shape) == len(expected_axes), (
                name,
                dataset["path"],
                array.shape,
                expected_axes,
            )
            # 2.1 MUST -- dimension_names matches the declared axes. The only
            # other assertion of this anywhere checks the kwargs builder, not a
            # written array.
            meta = json.loads((level / "zarr.json").read_text(encoding="utf-8"))
            # `.get`, not `[]` (ledger ALGO-18). 2.1: "The 'dimension_names'
            # attribute MUST be included in the zarr.json of the Zarr array of a
            # multiscale level" -- but Zarr v3 lists it as OPTIONAL array
            # metadata, so an array missing it is a valid Zarr array and an NGFF
            # violation. That is exactly this assertion's case, and it must
            # surface as AssertionError, not KeyError, or the `Raises:` contract
            # is false for the failure it exists to catch.
            names = meta.get("dimension_names")
            assert names is not None and list(names) == expected_axes, (
                name,
                dataset["path"],
                names,
                expected_axes,
            )

    # 2.6 -- the label is reachable through the labels array, and its pixels are
    # an integer dtype. Skipped entirely for a label-less store.
    if label_member is not None:
        listed = _group_ome(store / primary / LABELS_GROUP)["labels"]
        assert listed == [OBJMAP_LABEL], listed
        label_array = zarr.open_array(
            store=str(store / label_member / "0"), mode="r"
        )
        assert np.issubdtype(label_array.dtype, np.integer), (
            "2.6: label pixels MUST be an integer dtype"
        )
        # 2.6 MUST -- the label's datasets array has the same number of levels
        # as the image it labels. The referent is the PRIMARY series: 2.6 says
        # "the original unlabeled image", and the label labels the primary.
        assert level_counts[label_member] == level_counts[primary], (
            "2.6: label level count MUST match the unlabeled image"
        )

    # DESIGN SPEC 1.4 (not NGFF -- neither NGFF nor Zarr v3 imposes cross-array
    # separator uniformity; this is a PhenoTypic rule, and the bare section
    # numbers elsewhere in this function are NGFF ones). ONE check, not two
    # (ledger SIMP-23): the declared separator is direct, authoritative, and
    # fires even on an array with no chunks written yet. An earlier draft paired
    # it with an `rglob("*/0")` no-subdirectory walk, which observes far less --
    # it only inspects level-0 directories, passes vacuously on an empty array,
    # and under the sharding codec the chunks live inside shard files. The
    # separator assertion subsumes it -- and it is COMPLETE, not a narrowing:
    # under Zarr v3's `sharding_indexed` codec the inner chunks are addressed by
    # BYTE OFFSETS in the shard index, not by keys, so the array's top-level
    # `chunk_key_encoding` is the only thing in the whole store that turns
    # coordinates into a path segment. There is no second separator to check
    # (ledger ALGO-19).
    #
    # Gated on `node_type == "array"`, so an ARRAY that omits
    # `chunk_key_encoding` -- mandatory array metadata in Zarr v3 -- fails here
    # instead of being skipped along with the group documents.
    for meta_path in store.rglob("zarr.json"):
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        if payload.get("node_type") != "array":
            continue
        encoding = payload.get("chunk_key_encoding")
        assert encoding is not None, f"{meta_path}: array has no chunk_key_encoding"
        separator = encoding.get("configuration", {}).get("separator")
        assert separator == ".", f"{meta_path}: separator is {separator!r}"
```

- [ ] **Step 4: Run the tests**

```bash
uv run pytest tests/unit/core/test_ngff_conformance.py -v
```

Expected: all PASS. If a real store fails validation, **fix the writer**, not the harness —
that is the harness doing its job.

- [ ] **Step 5: Commit**

```bash
git add tests/_ngff_conformance.py tests/unit/core/test_ngff_conformance.py
git commit -m "test: validate written stores against the vendored NGFF schemas

jsonschema, not ome-zarr-models, which pins pydantic<2.13. Three negative
tests prove the gate has teeth: a truncated omero window, a missing
image-label, and a missing series each fail, which are exactly the three
places the published schemas are stricter than the prose."
```

---

## Phase 2 exit criteria

- [ ] `uv run pytest tests/unit/core/test_image_zarr_roundtrip.py tests/unit/core/test_grid_image_zarr_roundtrip.py tests/unit/core/test_ngff_conformance.py tests/unit/core/test_save_intermediate_zarr.py -q` is green.
- [ ] `uv run pytest tests/unit/gui/builder -q` is green (manifest key rename).
- [ ] `uv run pytest --doctest-modules src/phenotypic/_core/_image_parts/_image_io_handler.py -q` is green.
- [ ] `grep -rn '\.ome\.zarr"' src/phenotypic --include='*.py' | grep -v 'ngff_.py\|_io_constants.py'` returns nothing.
- [ ] The HDF write path still works: `uv run pytest tests/unit/core/test_image_hdf_roundtrip.py -q` is green.
