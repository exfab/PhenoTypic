"""Unit tests for the centralized PhenoTypic Plotly theme."""

from __future__ import annotations

import importlib

import plotly.graph_objects as go
import plotly.io as pio

from phenotypic.tools_.viz.figures._theme import (
    BG,
    FONT_FAMILY,
    OKABE_ITO,
    PHENOTYPIC_TEMPLATE_NAME,
    apply_theme,
    register_phenotypic_template,
)


def test_template_name_constant() -> None:
    """The public template-name constant is the literal ``"phenotypic"``."""
    assert PHENOTYPIC_TEMPLATE_NAME == "phenotypic"


def test_register_adds_template_to_plotly_registry() -> None:
    """``register_phenotypic_template`` registers ``"phenotypic"``."""
    register_phenotypic_template()
    assert PHENOTYPIC_TEMPLATE_NAME in pio.templates


def test_import_auto_registers_template() -> None:
    """Importing the theme module is enough to register the template."""
    module = importlib.import_module("phenotypic.tools_.viz.figures._theme")
    importlib.reload(module)
    assert PHENOTYPIC_TEMPLATE_NAME in pio.templates


def test_apply_theme_sets_combined_template() -> None:
    """``apply_theme`` sets a template referencing the phenotypic layer."""
    fig = apply_theme(go.Figure())
    assert isinstance(fig, go.Figure)
    template = fig.layout.template
    assert template is not None
    # The combined "plotly+phenotypic" template carries our navy-anchored
    # colorway, proving the phenotypic layer is composed in.
    assert tuple(template.layout.colorway) == OKABE_ITO


def test_apply_theme_returns_same_object() -> None:
    """``apply_theme`` returns the very figure it was given."""
    fig = go.Figure()
    assert apply_theme(fig) is fig


def test_apply_theme_preserves_traces() -> None:
    """Theming must not read, delete, or mutate ``fig.data``."""
    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
    apply_theme(fig)
    assert len(fig.data) == 1
    assert tuple(fig.data[0].x) == (1, 2)
    assert tuple(fig.data[0].y) == (3, 4)


def test_apply_theme_is_idempotent() -> None:
    """Applying twice keeps one template and the traces intact."""
    fig = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
    apply_theme(fig)
    first_template = fig.layout.template
    apply_theme(fig)
    second_template = fig.layout.template
    # Same composed colorway both times; trace count unchanged.
    assert tuple(first_template.layout.colorway) == OKABE_ITO
    assert tuple(second_template.layout.colorway) == OKABE_ITO
    assert len(fig.data) == 1


def test_colorway_equals_okabe_ito() -> None:
    """The registered template's ``colorway`` matches ``OKABE_ITO``."""
    register_phenotypic_template()
    template = pio.templates[PHENOTYPIC_TEMPLATE_NAME]
    assert tuple(template.layout.colorway) == OKABE_ITO


def test_okabe_ito_matches_design_md_series_order() -> None:
    """``OKABE_ITO`` is the navy-anchored DESIGN.md series order + ink."""
    assert OKABE_ITO == (
        "#003660",
        "#E69F00",
        "#56B4E9",
        "#009E73",
        "#0072B2",
        "#CC79A7",
        "#D55E00",
        "#000000",
    )


def test_paper_bgcolor_matches_design_bg() -> None:
    """Figure background is BG (#f5f7fa, DESIGN.md figure.facecolor), not white."""
    register_phenotypic_template()
    template = pio.templates[PHENOTYPIC_TEMPLATE_NAME]
    assert template.layout.paper_bgcolor == BG


def test_chart_body_font_intentionally_differs_from_gui_chrome() -> None:
    """The chart body font is IBM Plex Sans, decoupled from the GUI chrome body.

    The GUI chrome moved to Comfortaa (``gui/_design.FONT_FAMILY_BODY``), but the
    chart subsystem deliberately stays on IBM Plex Sans for plot titles and legend
    names (DESIGN.md "06 -- Charts"). The two are expected to *differ* now -- this
    test pins that intent so neither side silently re-converges.
    """
    from phenotypic.gui._design import FONT_FAMILY_BODY

    assert FONT_FAMILY.startswith("'IBM Plex Sans'")
    assert FONT_FAMILY_BODY.startswith("'Comfortaa'")
    assert FONT_FAMILY != FONT_FAMILY_BODY


def test_chart_body_font_is_loaded_by_gui_import() -> None:
    """IBM Plex Sans must stay in the GUI ``@import`` so GUI-embedded charts render it.

    Plotly figures render inside Dash pages that load ``gui/_design`` tokens; if the
    webfont were dropped from the ``@import`` the chart body text would silently fall
    back to the system sans instead of IBM Plex Sans.
    """
    from phenotypic.gui._design import FONT_TOKENS_CSS

    assert "IBM+Plex+Sans" in FONT_TOKENS_CSS


def test_mono_font_does_not_drift_from_gui_design() -> None:
    """The theme mono stack mirrors ``gui/_design.FONT_FAMILY_MONO`` (no drift).

    Numeric axes / hover render in mono; keeping the two stacks identical means
    a chart's tick labels match the GUI's mono data text exactly.
    """
    from phenotypic.gui._design import FONT_FAMILY_MONO as GUI_FONT_FAMILY_MONO
    from phenotypic.tools_.viz.figures._theme import FONT_FAMILY_MONO

    assert FONT_FAMILY_MONO == GUI_FONT_FAMILY_MONO
