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


def _walk(node):
    """Yield every component in a built layout tree."""
    yield node
    children = getattr(node, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk(child)
    elif children is not None:
        yield from _walk(children)


def _dangling_outputs(app) -> set[tuple[str, str]]:
    """Callback Outputs naming a string id the built layout does not mount.

    Pattern-matching dict ids are excluded on purpose: a
    ``{"type": ..., "index": ALL}`` id legitimately matches zero components at
    layout time because the components it targets are created by other
    callbacks. Only string ids name a component the layout is expected to
    carry.
    """
    layout = app.layout() if callable(app.layout) else app.layout
    mounted = {
        node.id for node in _walk(layout) if isinstance(getattr(node, "id", None), str)
    }

    dangling: set[tuple[str, str]] = set()
    for key in app.callback_map:
        # The callback_map key encodes its outputs as
        # ``..<id>.<prop>...<id>.<prop>..`` with an optional ``@<hash>``
        # suffix on ``allow_duplicate`` outputs.
        for segment in key.strip(".").split("..."):
            segment = segment.strip(".").split("@", 1)[0]
            if "." not in segment or segment.startswith("{"):
                continue  # no property, or a pattern-matching dict id
            component_id, prop = segment.rsplit(".", 1)
            if component_id.startswith("{"):
                continue
            if component_id not in mounted:
                dangling.add((component_id, prop))
    return dangling


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

    app = create_app(SandboxRoot.from_path(tmp_path))

    assert not _dangling_outputs(app), (
        "these browse callback Outputs name components the layout does not "
        "mount, so dash-renderer will throw and DISCARD the whole callback "
        f"response (taking any co-outputs with it): "
        f"{sorted(_dangling_outputs(app))}"
    )
