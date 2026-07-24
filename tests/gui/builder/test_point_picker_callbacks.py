"""Structural callback ownership regressions for the Builder point picker."""

from __future__ import annotations

import dash

from phenotypic.gui.builder import _ids as ids
from phenotypic.gui.builder._point_picker import register_point_picker_callbacks


def test_point_picker_undo_only_owns_staged_modal_state() -> None:
    """Modal-local Undo cannot replace or clear the inspector preview.

    Builder has no pipeline-history Undo state transition. Its only Undo
    control removes the last point staged in the point-picker modal, so that
    callback must own only the staged modal store.
    """
    app = dash.Dash(__name__)
    register_point_picker_callbacks(app)

    undo_callbacks = [
        metadata
        for metadata in app.callback_map.values()
        if any(
            input_.get("id") == ids.BTN_PICKER_UNDO
            for input_ in metadata["inputs"]
        )
    ]

    assert len(undo_callbacks) == 1
    output = undo_callbacks[0]["output"]
    assert output.component_id == ids.PICKER_STAGED_STORE
    assert output.component_property == "data"
    assert output.component_id != ids.INSPECTOR_PREVIEW
