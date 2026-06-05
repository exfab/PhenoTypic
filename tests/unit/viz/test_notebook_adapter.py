"""Unit tests for the ipywidgets adapter's pure render-loop helpers.

These exercise the value→kwarg mapping and the dependency-driven re-render set
without importing ipywidgets or a Jupyter kernel.
"""

from __future__ import annotations

import plotly.graph_objects as go

from phenotypic.abc_ import Control, FigureProvider, figure
from phenotypic.viz.notebook._adapter import (
    control_owners,
    initial_control_state,
    spec_control_kwargs,
    unique_controls,
)

SIGMA = Control(label="Sigma", kind="float", default=1.5, bounds=(0.0, 5.0))
METHOD = Control(label="Method", kind="select", default="meijering", options=("meijering", "sato"))


class _Provider(FigureProvider):
    @figure(title="Ridge", section="structure", controls={"sigma": SIGMA, "method": METHOD})
    def ridge(self, *, sigma, method) -> go.Figure:
        return go.Figure(go.Scatter(x=[sigma], y=[1], name=method))

    @figure(title="Smooth", section="structure", controls={"sigma": SIGMA})
    def smooth(self, *, sigma) -> go.Figure:
        return go.Figure(go.Scatter(x=[sigma], y=[2]))


def _specs():
    return _Provider().iter_figures()


def test_unique_controls_dedup_by_identity():
    # SIGMA is shared by both figures; METHOD only by ridge → 2 unique controls.
    controls = unique_controls(_specs())
    assert len(controls) == 2
    assert SIGMA in controls and METHOD in controls


def test_control_owners_maps_shared_control_to_both_specs():
    owners = control_owners(_specs())
    sigma_owners = {s.name for s in owners[id(SIGMA)]}
    method_owners = {s.name for s in owners[id(METHOD)]}
    assert sigma_owners == {"ridge", "smooth"}  # changing sigma re-renders both
    assert method_owners == {"ridge"}  # changing method re-renders only ridge


def test_initial_state_uses_defaults():
    state = initial_control_state(_specs())
    assert state[id(SIGMA)] == 1.5
    assert state[id(METHOD)] == "meijering"


def test_spec_control_kwargs_resolves_current_values():
    specs = {s.name: s for s in _specs()}
    state = initial_control_state(_specs())
    state[id(SIGMA)] = 3.0
    state[id(METHOD)] = "sato"
    assert spec_control_kwargs(specs["ridge"], state) == {"sigma": 3.0, "method": "sato"}
    assert spec_control_kwargs(specs["smooth"], state) == {"sigma": 3.0}
