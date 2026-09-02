"""The hub still constructs, and no sub-app writes to an id it stopped mounting.

Spec section 7, check 1, reduced to its two load-bearing halves.

**The hub builds.** A deleted module that some module still imports fails
here at ``create_app`` time, which is the only place it *can* fail now that
the tests for the deleted surfaces are gone too. The results viewer's own
tab shape is NOT re-asserted: phase 1 created ``test_layout_tab_shape.py``
for exactly that and phase 5 edited it to the final two-tab list, so
restating it would ship a second ``dbc.Tabs`` walker beside that file's.

**Browse mounts no dangling Output.** Phase 5 shipped a follow-up because
unmounting the QC tab left ``colony_view`` writing to ``qc-review-dim-readout``,
an id the layout no longer carried; ``suppress_callback_exceptions`` relaxes
SERVER-side validation only, so dash-renderer threw and discarded the whole
response, freezing a working control. ``results_viewer/test_callback_output_ids.py``
now guards the viewer against that class of defect -- but phase 2 removed the
view-mode toggle, the timeline body and 64 ``BROWSE_TL_*`` ids from **browse**,
which that test cannot reach, and no other test builds browse's callback map
against its layout. This closes that half.
"""

from __future__ import annotations

from pathlib import Path

from phenotypic.gui.shell._sandbox import SandboxRoot

from tests._dash_layout import (
    DANGLING_OUTPUT_MESSAGE,
    dangling_callback_outputs,
)


def test_hub_app_constructs(tmp_path: Path) -> None:
    """The composed hub builds every eagerly-mounted sub-app."""
    from phenotypic.gui.shell._app import create_app

    app = create_app(SandboxRoot.from_path(tmp_path))

    assert app is not None
    assert app.server.wsgi_app.mounts, "the hub composed no sub-app mounts"


def test_browse_registers_no_callback_output_the_layout_dropped(
    tmp_path: Path,
) -> None:
    """No browse callback Output names an id the view-mode removal took away."""
    from phenotypic.gui.browse._app import create_app

    dangling = dangling_callback_outputs(create_app(SandboxRoot.from_path(tmp_path)))

    assert not dangling, DANGLING_OUTPUT_MESSAGE + f"{sorted(dangling)}"
