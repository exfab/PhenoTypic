# `.draw()` — Interactive Label Editing in napari

**Date:** 2026-06-17
**Branch:** `image-draw-update`
**Status:** Approved design — ready for implementation plan

## Summary

Add a `.draw()` method that opens a PyQt napari viewer with the image's `rgb`,
`gray`, and `detect_mat` as image layers and the called accessor's data
(`objmap` or `objmask`) as an **editable** labels layer. The user edits the
labels with napari's built-in paint/fill/erase tools, then commits the edits
back to the `Image` via a "Save to Image" dock-widget button. The method blocks
until the viewer is closed and returns the (possibly mutated) root `Image`.

This is the labels-editing counterpart to the existing `PointPickerWidget`
(`tools_/napari_/_point_picker_widget.py`), which provides the same
open → edit → confirm → close → capture pattern for point coordinates.

## Decisions

| Question | Decision |
|----------|----------|
| Where does `.draw()` live? | `NapariLabelsMixin` — so only `image.objmap.draw()` and `image.objmask.draw()` exist; `gray`/`detect_mat` do **not** get a `.draw()`. |
| Viewer lifecycle | Blocking (`napari.run()`) + explicit **Save to Image** button. Mirrors `PointPickerWidget`. |
| Save target | Respects the accessor it was called on. `objmap` edits preserve integer label IDs; `objmask` edits save as a binary mask (relabels via skimage; original IDs lost — accepted). |
| `objmask` painted layer | **Strictly binary** — seeded as 0/1, `selected_label = 1`, and binarized (`> 0`) on save regardless of any stray label values. |
| Return value | The root `Image`. |
| Testing | Mirror `PointPickerWidget` tests — extract dock/save logic into a testable widget class; unit-test the save callback under the `qt-test` group with offscreen Qt. Do not drive the blocking loop. |
| Discard | Keep an explicit **Discard & Close** button in addition to plain window close (both leave the image untouched). |

## Architecture

Three pieces.

### 1. `tools_/napari_/_label_editor_widget.py` (new)

Mirrors `_point_picker_widget.py`. `qtpy` and `napari` are imported lazily
inside methods, never at module top level.

**`LabelEditorWidget`**

```
LabelEditorWidget.run(image, accessor_name, *, viewer=None) -> np.ndarray | None
```

- Guards `_HAS_NAPARI` (raise the same `ImportError` message used by
  `.napari()` / `PointPickerWidget`).
- Opens `napari.Viewer(title="Label Editor")` when `viewer is None`; otherwise
  reuses the supplied viewer.
- Adds image layers by reusing each accessor's existing
  `.napari(viewer=…, layer_name=…)`:
  - `rgb` only `if not image.rgb.isempty()`
  - `gray`
  - `detect_mat`
- Adds the **editable labels layer** seeded from the called accessor:
  - `accessor_name == "objmap"` → `image.objmap[:]` (integer labels).
  - `accessor_name == "objmask"` → `image.objmask[:].astype(np.uint8)`
    (binary 0/1); set `selected_label = 1`.
- Makes the labels layer the active layer with `mode = "paint"` so napari's
  paintbrush / fill / eraser / `selected_label` controls are immediately usable.
- Mounts `_LabelEditorPanel` as a right dock widget.
- Calls `napari.run()` (blocks).
- Returns `panel.saved_labels` (the array written back) or `None` if the user
  discarded / closed without saving.

**`_LabelEditorPanel`**

Mirrors `_PointPickerPanel`: subclasses `QWidget` at runtime via the `__new__`
trick to keep `qtpy` out of module import.

- Holds: `viewer`, `labels_layer`, `image`, `accessor_name`.
- `saved_labels: np.ndarray | None = None`.
- Buttons: **Save to Image**, **Discard & Close**.
- **Save** is the single testable unit — it writes back through the accessor
  setter, stashes the array, and closes:
  - `objmap`: `image.objmap[:] = labels_layer.data` (preserves IDs).
  - `objmask`: `image.objmask[:] = (labels_layer.data > 0)` (binarize →
    setter relabels).
  - then `self.saved_labels = <written array>`; `viewer.close()`.
- **Discard** just calls `viewer.close()`; image untouched.

### 2. `NapariLabelsMixin.draw(...)` (new method)

```python
def draw(self, *, viewer: napari.Viewer | None = None) -> Image:
```

- Guards `_HAS_NAPARI` with the standard `ImportError`.
- Calls `LabelEditorWidget().run(self._root_image, self._accessor_property_name)`.
- Returns `self._root_image`.
- Google-style docstring with a `load_synth_yeast_plate()`-based example;
  documents the `objmask`-relabels caveat and points to `.show()` / `.napari()`
  for previewing existing detections (echoing the `ManualRefine` note).
- The optional `viewer` kwarg parallels `.napari()`; when supplied,
  `LabelEditorWidget` uses it instead of creating one (still mounts the panel).
  Useful for tests and advanced reuse.

### 3. No changes to accessor setters

`ObjectMap.__setitem__` full-slice path and `ObjectMask.__setitem__` already
provide exactly the write-back semantics required (objmap preserves IDs; objmask
relabels). The `objmask` view follows the `objmap` backend automatically.

## Data Flow

```
image.objmap.draw()
  → NapariLabelsMixin.draw
    → LabelEditorWidget.run(image, "objmap")
      → viewer + image layers + editable labels layer
      → user paints; clicks "Save to Image"
        → panel: image.objmap[:] = layer.data   (sparse backend updated;
                                                  objmask view follows)
        → panel.saved_labels = data; viewer.close()
      → run returns array
  → draw returns image
```

## Error Handling / Edge Cases

- **napari not installed** → `ImportError` with the existing install hint.
- **No objects yet** → labels layer seeds from the empty/zero map; the user
  paints from scratch. Works for both accessors.
- **Discard / window close without saving** → no mutation; `draw` still returns
  the unchanged root `Image`.
- **Shape mismatch** → impossible: the layer is seeded from the same image, so
  the accessor setters' shape checks never trip.

## Testing

`tests/unit/tools_/test_label_editor_widget.py` (qt-test group,
`QT_QPA_PLATFORM=offscreen`), constructing `_LabelEditorPanel` against a real
`load_synth_yeast_plate()` image with a detector applied:

1. **objmap save preserves IDs** — set the labels-layer data to a modified
   integer array; call the save callback; assert `image.objmap[:]` equals it
   verbatim.
2. **objmask save relabels** — set the labels-layer data to a modified binary
   mask; call the save callback; assert the foreground matches and the map is
   sequentially relabeled.
3. **discard leaves image unchanged** — call the discard callback; assert
   `image.objmap[:]` is identical to before.
4. **ImportError without napari** — monkeypatch `_HAS_NAPARI = False`; assert
   `image.objmap.draw()` raises `ImportError`.

The blocking `napari.run()` loop itself is not driven in tests, matching the
`PointPickerWidget` test approach.

## Out of Scope (YAGNI)

- No undo/redo beyond napari's built-in layer history.
- No multi-label palette UI for `objmask` (strictly binary).
- No persistence to disk from within the editor (`imsave` already exists).
- No GUI/dashboard integration — `.draw()` is a notebook/REPL tool like
  `.napari()`.
