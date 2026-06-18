# `.draw()` Napari Label Editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `image.objmap.draw()` / `image.objmask.draw()` that open a blocking PyQt napari viewer for editing the labels layer and committing edits back to the `Image` via a "Save to Image" button.

**Architecture:** A new `LabelEditorWidget` (+ `_LabelEditorPanel` dock) in `tools_/napari_/` mirrors the existing `PointPickerWidget`: it opens a viewer with `rgb`/`gray`/`detect_mat` image layers plus one editable labels layer seeded from the called accessor, runs `napari.run()` (blocking), and a Save button writes the edited array back through the accessor setter. A new `draw()` method on the existing `NapariLabelsMixin` is the public entry point.

**Tech Stack:** Python, napari, qtpy (lazy-imported), scipy.sparse, pytest (`qt-test` group, `QT_QPA_PLATFORM=offscreen`).

## Global Constraints

- `uv` is the sole runner: `uv run pytest …`, `uv run mypy src/phenotypic`, `uv run ruff check --fix`.
- `napari` and `qtpy` MUST be lazy-imported (inside methods), never at module top level. Module-level `napari` only under `TYPE_CHECKING`.
- The napari install hint string is exactly: `"napari is required for interactive visualization. Install with: pip install phenotypic[napari]"`.
- Accessor write-back semantics (do not change): `image.objmap[:] = arr` preserves integer label IDs; `image.objmask[:] = mask` relabels via `skimage.measure.label` (original IDs lost — intended).
- Google-style docstrings; doctest examples must be runnable with `load_synth_yeast_plate()`.
- `.draw()` lives ONLY on `NapariLabelsMixin` (objmap/objmask) — not on `SingleChannelAccessor` (gray/detect_mat must not get it).
- Tests must not drive the blocking `napari.run()` loop; test panel callbacks directly via the MagicMock pattern from `tests/unit/tools_/test_point_picker_widget.py`.

---

### Task 1: `LabelEditorWidget` + `_LabelEditorPanel`

**Files:**
- Create: `src/phenotypic/tools_/napari_/_label_editor_widget.py`
- Modify: `src/phenotypic/tools_/napari_/__init__.py`
- Test: `tests/unit/tools_/test_label_editor_widget.py`

**Interfaces:**
- Consumes: `_HAS_NAPARI` from `phenotypic._core._image_parts.accessor_abstracts._image_accessor_base`; the accessor setters `image.objmap[:] = arr` / `image.objmask[:] = mask`; `image.rgb.isempty()`; `image.{rgb,gray,detect_mat}.napari(viewer=…, layer_name=…)`.
- Produces:
  - `LabelEditorWidget().run(image, accessor_name: str, *, viewer=None) -> np.ndarray | None` — returns the written-back array, or `None` if discarded/closed without saving.
  - `_LabelEditorPanel` with attributes `_viewer`, `_labels_layer`, `_image`, `_accessor_name`, `saved_labels: np.ndarray | None`, and methods `_save()` / `_discard()`.

- [ ] **Step 1: Write the failing test**

```python
"""Tests for LabelEditorWidget and _LabelEditorPanel logic."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# LabelEditorWidget public API
# ---------------------------------------------------------------------------


class TestLabelEditorWidget:
    def test_run_raises_import_error_without_napari(self):
        from phenotypic.tools_.napari_ import LabelEditorWidget

        w = LabelEditorWidget()
        with patch(
            "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
            False,
        ):
            with pytest.raises(ImportError, match="napari is required"):
                w.run(MagicMock(), "objmap")


# ---------------------------------------------------------------------------
# _LabelEditorPanel logic (tested without real Qt)
# ---------------------------------------------------------------------------


def _make_mock_panel(*, image, accessor_name: str, layer_data: np.ndarray) -> MagicMock:
    """Mimic a ``_LabelEditorPanel`` for logic testing without Qt.

    Mirrors ``tests/unit/tools_/test_point_picker_widget.py::_make_mock_panel``:
    build a ``MagicMock`` with the same attributes and bind the real class
    methods so ``_save``/``_discard`` run against a real Image.
    """
    from phenotypic.tools_.napari_._label_editor_widget import _LabelEditorPanel

    panel = MagicMock()

    labels_layer = MagicMock()
    labels_layer.data = layer_data.copy()
    panel._labels_layer = labels_layer

    panel._image = image
    panel._accessor_name = accessor_name
    panel._viewer = MagicMock()
    panel.saved_labels = None

    panel._save = lambda: _LabelEditorPanel._save(panel)
    panel._discard = lambda: _LabelEditorPanel._discard(panel)

    return panel


@pytest.fixture
def detected_image():
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector

    image = load_synth_yeast_plate()
    image = OtsuDetector().apply(image)
    return image


class TestSaveObjmap:
    """objmap save preserves the edited integer labels verbatim."""

    def test_save_writes_back_preserving_ids(self, detected_image):
        edited = detected_image.objmap[:].copy()
        # Stamp a small non-contiguous high label ID to prove IDs are preserved.
        edited[0:3, 0:3] = 777
        panel = _make_mock_panel(
            image=detected_image, accessor_name="objmap", layer_data=edited
        )

        panel._save()

        np.testing.assert_array_equal(detected_image.objmap[:], edited)
        assert 777 in np.unique(detected_image.objmap[:])
        assert panel.saved_labels is not None
        panel._viewer.close.assert_called_once()


class TestSaveObjmask:
    """objmask save binarizes then relabels (sequential IDs)."""

    def test_save_binarizes_and_relabels(self, detected_image):
        # Build a 2-blob binary layer with a non-binary stray value.
        mask = np.zeros(detected_image.objmask.shape, dtype=np.uint8)
        mask[5:10, 5:10] = 1
        mask[20:25, 20:25] = 5  # stray non-1 value -> must be binarized
        panel = _make_mock_panel(
            image=detected_image, accessor_name="objmask", layer_data=mask
        )

        panel._save()

        result_mask = detected_image.objmask[:]
        np.testing.assert_array_equal(result_mask, mask > 0)
        # Relabel produced sequential integer IDs on the objmap.
        labels = np.unique(detected_image.objmap[:])
        labels = labels[labels > 0]
        np.testing.assert_array_equal(labels, np.array([1, 2], dtype=labels.dtype))
        panel._viewer.close.assert_called_once()


class TestDiscard:
    """Discard closes the viewer and leaves the image untouched."""

    def test_discard_no_mutation(self, detected_image):
        before = detected_image.objmap[:].copy()
        edited = detected_image.objmap[:].copy()
        edited[0:3, 0:3] = 777
        panel = _make_mock_panel(
            image=detected_image, accessor_name="objmap", layer_data=edited
        )

        panel._discard()

        np.testing.assert_array_equal(detected_image.objmap[:], before)
        assert panel.saved_labels is None
        panel._viewer.close.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen uv run --group qt-test pytest tests/unit/tools_/test_label_editor_widget.py -v`
Expected: FAIL — `ImportError` / `ModuleNotFoundError: phenotypic.tools_.napari_._label_editor_widget` (and `LabelEditorWidget` not exported).

- [ ] **Step 3: Write minimal implementation**

Create `src/phenotypic/tools_/napari_/_label_editor_widget.py`:

```python
"""Blocking napari-based labels editor with save-back-to-image."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import napari


class LabelEditorWidget:
    """Open a napari viewer to edit an image's object labels and save them back.

    The viewer shows ``rgb`` (when present), ``gray``, and ``detect_mat`` as
    image layers plus one editable labels layer seeded from the called accessor
    (``objmap`` or ``objmask``). A dock panel provides "Save to Image" and
    "Discard & Close" buttons. ``run`` blocks until the viewer is closed.
    """

    def run(
        self,
        image,
        accessor_name: str,
        *,
        viewer: "napari.Viewer | None" = None,
    ) -> np.ndarray | None:
        """Open the editor and block until closed.

        Args:
            image: A PhenoTypic ``Image`` whose layers are displayed and whose
                ``objmap``/``objmask`` is edited.
            accessor_name: ``"objmap"`` or ``"objmask"`` — selects which accessor
                the editable layer is seeded from and saved back to.
            viewer: Optional existing napari viewer to reuse. When ``None`` a new
                ``napari.Viewer(title="Label Editor")`` is created.

        Returns:
            The array written back to the image on save, or ``None`` if the user
            discarded or closed the viewer without saving.

        Raises:
            ImportError: If napari is not installed.
        """
        from phenotypic._core._image_parts.accessor_abstracts._image_accessor_base import (
            _HAS_NAPARI,
        )

        if not _HAS_NAPARI:
            raise ImportError(
                "napari is required for interactive visualization. "
                "Install with: pip install phenotypic[napari]"
            )
        import napari

        active_viewer = viewer if viewer is not None else napari.Viewer(title="Label Editor")

        if not image.rgb.isempty():
            image.rgb.napari(viewer=active_viewer, layer_name="rgb")
        image.gray.napari(viewer=active_viewer, layer_name="gray")
        image.detect_mat.napari(viewer=active_viewer, layer_name="detect_mat")

        if accessor_name == "objmask":
            seed = image.objmask[:].astype(np.uint8)
        else:
            seed = np.asarray(image.objmap[:])
        labels_layer = active_viewer.add_labels(seed, name=f"{accessor_name}_edit")
        if accessor_name == "objmask":
            labels_layer.selected_label = 1
        labels_layer.mode = "paint"
        active_viewer.layers.selection.active = labels_layer

        panel = _LabelEditorPanel(active_viewer, labels_layer, image, accessor_name)
        active_viewer.window.add_dock_widget(panel, name="Label Editor", area="right")

        napari.run()

        return panel.saved_labels


class _LabelEditorPanel:
    """Dock widget with Save/Discard controls for the labels editor.

    Inherits from ``QWidget`` at runtime (via ``__new__``) so that ``qtpy`` is
    not imported at module level.

    Args:
        viewer: The napari viewer instance.
        labels_layer: The editable napari Labels layer.
        image: The PhenoTypic ``Image`` to write edits back to.
        accessor_name: ``"objmap"`` or ``"objmask"``.
    """

    def __new__(cls, *args, **kwargs):  # noqa: ARG003
        from qtpy.QtWidgets import QWidget

        if not issubclass(cls, QWidget):
            cls.__bases__ = (QWidget,)
        instance = QWidget.__new__(cls)
        return instance

    def __init__(self, viewer, labels_layer, image, accessor_name: str) -> None:
        from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

        QWidget.__init__(self)  # type: ignore[arg-type]

        self._viewer = viewer
        self._labels_layer = labels_layer
        self._image = image
        self._accessor_name = accessor_name
        self.saved_labels: np.ndarray | None = None

        layout = QVBoxLayout(self)  # type: ignore[call-overload]
        layout.setContentsMargins(4, 4, 4, 4)

        self._save_btn = QPushButton("Save to Image")
        self._discard_btn = QPushButton("Discard & Close")
        layout.addWidget(self._save_btn)
        layout.addWidget(self._discard_btn)

        self._save_btn.clicked.connect(self._save)
        self._discard_btn.clicked.connect(self._discard)

    def _save(self) -> None:
        """Write the edited labels back through the accessor, then close.

        ``objmap`` saves the integer array verbatim (label IDs preserved);
        ``objmask`` binarizes the layer (``> 0``) and saves it as a mask, which
        relabels the object map.
        """
        data = self._labels_layer.data
        if self._accessor_name == "objmask":
            self._image.objmask[:] = data > 0
            self.saved_labels = self._image.objmask[:]
        else:
            self._image.objmap[:] = np.asarray(data)
            self.saved_labels = self._image.objmap[:]
        self._viewer.close()

    def _discard(self) -> None:
        """Close the viewer without writing any edits back."""
        self._viewer.close()
```

Modify `src/phenotypic/tools_/napari_/__init__.py`:

```python
"""Napari-based interactive tools for PhenoTypic.

Developer utilities for visual point picking and coordinate selection
using napari viewers. These are dev-time tools, not user-facing GUI
components.
"""

from ._label_editor_widget import LabelEditorWidget
from ._point_picker_widget import PointPickerWidget

__all__ = [
    "LabelEditorWidget",
    "PointPickerWidget",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen uv run --group qt-test pytest tests/unit/tools_/test_label_editor_widget.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Lint + type-check the new module**

Run: `uv run ruff check --fix src/phenotypic/tools_/napari_/_label_editor_widget.py && uv run mypy src/phenotypic/tools_/napari_/_label_editor_widget.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/tools_/napari_/_label_editor_widget.py src/phenotypic/tools_/napari_/__init__.py tests/unit/tools_/test_label_editor_widget.py
git commit -m "feat(napari): add LabelEditorWidget for editing objmap/objmask labels

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `NapariLabelsMixin.draw()` public method

**Files:**
- Modify: `src/phenotypic/_core/_image_parts/accessor_abstracts/_napari_labels_mixin.py`
- Test: `tests/unit/tools_/test_label_editor_widget.py` (append `TestDrawMethod`)

**Interfaces:**
- Consumes: `LabelEditorWidget().run(image, accessor_name, *, viewer=None)` from Task 1; `self._root_image`; `self._accessor_property_name` (returns `"objmap"` / `"objmask"`); `_HAS_NAPARI`.
- Produces: `NapariLabelsMixin.draw(self, *, viewer=None) -> Image` — opens the editor and returns the (possibly mutated) root `Image`.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/tools_/test_label_editor_widget.py`:

```python
class TestDrawMethod:
    """Tests for NapariLabelsMixin.draw()."""

    def test_draw_raises_import_error_without_napari(self, detected_image):
        with patch(
            "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
            False,
        ):
            with pytest.raises(ImportError, match="napari is required"):
                detected_image.objmap.draw()

    def test_draw_delegates_and_returns_root_image(self, detected_image):
        with patch(
            "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
            True,
        ), patch(
            "phenotypic.tools_.napari_.LabelEditorWidget.run"
        ) as mock_run:
            mock_run.return_value = None

            result = detected_image.objmap.draw()

            mock_run.assert_called_once()
            # First positional arg is the root image, second is the accessor name.
            args, _ = mock_run.call_args
            assert args[0] is detected_image
            assert args[1] == "objmap"
            assert result is detected_image

    def test_draw_passes_objmask_accessor_name(self, detected_image):
        with patch(
            "phenotypic._core._image_parts.accessor_abstracts._image_accessor_base._HAS_NAPARI",
            True,
        ), patch(
            "phenotypic.tools_.napari_.LabelEditorWidget.run"
        ) as mock_run:
            mock_run.return_value = None

            detected_image.objmask.draw()

            args, _ = mock_run.call_args
            assert args[1] == "objmask"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `QT_QPA_PLATFORM=offscreen uv run --group qt-test pytest tests/unit/tools_/test_label_editor_widget.py::TestDrawMethod -v`
Expected: FAIL — `AttributeError: 'ObjectMap' object has no attribute 'draw'`.

- [ ] **Step 3: Write minimal implementation**

Add the `draw` method to `NapariLabelsMixin` in
`src/phenotypic/_core/_image_parts/accessor_abstracts/_napari_labels_mixin.py`
(immediately after the existing `napari` method, inside the class):

```python
    def draw(self, *, viewer: napari.Viewer | None = None) -> "Image":
        """Open a napari editor to hand-edit these labels and save them back.

        Launches a blocking PyQt napari viewer showing ``rgb`` (when present),
        ``gray``, and ``detect_mat`` as image layers plus this accessor's data
        as an **editable** labels layer. Use napari's built-in paintbrush, fill,
        and eraser tools to correct the segmentation, then click **Save to
        Image** in the dock panel to commit the edits to the parent image
        (or **Discard & Close** to abandon them).

        Editing ``objmap`` preserves the original integer label IDs. Editing
        ``objmask`` is strictly binary and **relabels** the object map on save
        (``skimage.measure.label``), so original IDs are not retained — call
        ``image.objmap.draw()`` instead when stable IDs matter. Preview the
        existing detections with :meth:`show` or :meth:`napari` before editing.

        Args:
            viewer: Optional existing napari viewer to reuse instead of opening
                a fresh one. Defaults to None.

        Returns:
            Image: The parent image, mutated in place if edits were saved.

        Raises:
            ImportError: If napari is not installed. Install with
                ``pip install phenotypic[napari]``.

        Examples:
            Hand-correct an auto-detected object map:

            >>> from phenotypic.data import load_synth_yeast_plate
            >>> from phenotypic.detect import OtsuDetector
            >>> image = OtsuDetector().apply(load_synth_yeast_plate())
            >>> image = image.objmap.draw()  # doctest: +SKIP
        """
        from phenotypic._core._image_parts.accessor_abstracts._image_accessor_base import (
            _HAS_NAPARI,
        )

        if not _HAS_NAPARI:
            raise ImportError(
                "napari is required for interactive visualization. "
                "Install with: pip install phenotypic[napari]"
            )

        from phenotypic.tools_.napari_ import LabelEditorWidget

        LabelEditorWidget().run(
            self._root_image, self._accessor_property_name, viewer=viewer
        )
        return self._root_image
```

Update the `TYPE_CHECKING` block at the top of the file so the `Image` return
annotation and the `viewer` annotation resolve. The existing block is:

```python
if TYPE_CHECKING:
    import napari
```

Replace it with:

```python
if TYPE_CHECKING:
    import napari

    from phenotypic._core._image import Image
```

- [ ] **Step 4: Run test to verify it passes**

Run: `QT_QPA_PLATFORM=offscreen uv run --group qt-test pytest tests/unit/tools_/test_label_editor_widget.py -v`
Expected: PASS (all tests including `TestDrawMethod`).

- [ ] **Step 5: Lint + type-check**

Run: `uv run ruff check --fix src/phenotypic/_core/_image_parts/accessor_abstracts/_napari_labels_mixin.py && uv run mypy src/phenotypic/_core/_image_parts/accessor_abstracts/_napari_labels_mixin.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add src/phenotypic/_core/_image_parts/accessor_abstracts/_napari_labels_mixin.py tests/unit/tools_/test_label_editor_widget.py
git commit -m "feat(core): add objmap/objmask .draw() napari label editor entry point

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Final subagent code review

**Files:** none (review only).

- [ ] **Step 1: Dispatch the code review**

Dispatch a `feature-dev:code-reviewer` subagent (via the Agent tool) to review the full diff of this branch against the spec at `docs/superpowers/specs/2026-06-17-objmap-objmask-draw-design.md`. Brief it to check:
- `napari`/`qtpy` are lazy-imported only (no top-level imports breaking non-GUI installs).
- Save-back semantics match the spec: objmap preserves IDs, objmask binarizes + relabels.
- `.draw()` exists only on objmap/objmask, not on gray/detect_mat.
- The install-hint string and docstring caveats are correct and runnable.
- Tests genuinely exercise the save/discard logic (not tautological) and follow the existing `_make_mock_panel` pattern.

- [ ] **Step 2: Triage and apply fixes**

Read the review with the `superpowers:receiving-code-review` skill. Apply only correct, in-scope fixes; for each, re-run the relevant test from Tasks 1–2 and commit.

- [ ] **Step 3: Full regression on the touched suite**

Run: `QT_QPA_PLATFORM=offscreen uv run --group qt-test pytest tests/unit/tools_/ -v`
Expected: PASS. Confirm `image.gray.draw` / `image.detect_mat.draw` do NOT exist:

Run: `uv run python -c "from phenotypic.data import load_synth_yeast_plate as L; i=L(); print(hasattr(i.gray,'draw'), hasattr(i.objmap,'draw'), hasattr(i.objmask,'draw'))"`
Expected: `False True True`

---

## Self-Review

**Spec coverage:**
- Home on `NapariLabelsMixin` only → Task 2 + Task 3 Step 3 assertion. ✓
- Blocking + Save button → Task 1 `run` (`napari.run()`) + `_LabelEditorPanel._save`. ✓
- Respect accessor (objmap preserves IDs; objmask relabels) → Task 1 `_save` + `TestSaveObjmap`/`TestSaveObjmask`. ✓
- objmask strictly binary (`> 0`, `selected_label = 1`) → Task 1 `run`/`_save` + stray-value test. ✓
- Returns root Image → Task 2 `draw` + `test_draw_delegates_and_returns_root_image`. ✓
- Image layers rgb(if not empty)/gray/detect_mat via reused `.napari()` → Task 1 `run`. ✓
- Tests mirror PointPickerWidget (MagicMock panel, no blocking loop) → Task 1 test module. ✓
- ImportError without napari → `TestLabelEditorWidget` + `TestDrawMethod`. ✓
- Discard & Close button → Task 1 `_discard` + `TestDiscard`. ✓
- Final subagent code reviewer → Task 3. ✓

**Placeholder scan:** none — all code and commands are concrete.

**Type consistency:** `run(image, accessor_name, *, viewer=None)`, `_save`/`_discard`, `saved_labels`, `_accessor_property_name`, and `draw(*, viewer=None) -> Image` are used identically across Tasks 1–3 and tests.
