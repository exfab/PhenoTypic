"""Unit tests for the centralized PhenoTypic Plotly theme."""

from __future__ import annotations

import sys
import subprocess
import os
import json

import plotly.graph_objects as go
import plotly.io as pio

from phenotypic.sdk_.viz.figures._theme import (
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


def test_importing_the_theme_does_not_touch_plotly() -> None:
    """Importing the theme must NOT register the template — or import plotly.

    This replaces ``test_import_auto_registers_template``, which asserted the
    opposite. That contract was deliberately dropped: registering at import
    called into plotly from module scope, putting the whole library on the
    ``import phenotypic`` startup path for every run, most of which draw no
    figure. ``apply_theme`` registers on demand instead, which
    ``test_apply_theme_registers_on_demand`` below pins.

    The old test was also passing for the wrong reason. ``pio.templates`` is
    process-global and ``importlib.reload`` cannot un-register an entry, so it
    only passed because ``test_register_adds_template_to_plotly_registry`` ran
    first in the same process — in isolation it failed. A subprocess is the only
    honest way to ask this question.
    """
    script = (
        "import json, sys\n"
        "import phenotypic.sdk_.viz.figures._theme  # noqa: F401\n"
        "print(json.dumps({'plotly_loaded': 'plotly' in sys.modules}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert json.loads(completed.stdout.strip().splitlines()[-1])["plotly_loaded"] is False


def test_apply_theme_registers_on_demand() -> None:
    """The template must appear the first time a figure is themed, in a fresh process."""
    script = (
        "import json, sys\n"
        "from phenotypic.sdk_.viz.figures import apply_theme\n"
        "import plotly.graph_objects as go, plotly.io as pio\n"
        "before = 'phenotypic' in pio.templates\n"
        "apply_theme(go.Figure())\n"
        "print(json.dumps({'before': before, 'after': 'phenotypic' in pio.templates}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["before"] is False, "the template was registered before apply_theme ran"
    assert result["after"] is True, "apply_theme did not register the template"


def test_docs_build_renderer_is_configured_when_a_figure_is_themed() -> None:
    """Under PHENOTYPIC_DOCS_BUILD, theming a figure must set the nbsphinx renderer.

    This used to be an import side effect of the dash accessor, which sat on the
    ``import phenotypic`` path, so every figure in a docs build got
    ``notebook_connected`` whether or not anything called ``.dash()``. Deferring
    plotly removed that side effect and silently broke notebooks that render a
    figure as a terminal expression — ``assess_image_quality.ipynb`` contains no
    ``.dash()`` call, and its figures would have emitted a JSON MIME bundle that
    nbsphinx drops, vanishing from the built docs with nothing failing.
    """
    script = (
        "import json\n"
        "from phenotypic.sdk_.viz.figures import apply_theme\n"
        "import plotly.graph_objects as go, plotly.io as pio\n"
        "apply_theme(go.Figure())\n"
        "print(json.dumps({'renderer': pio.renderers.default}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PHENOTYPIC_DOCS_BUILD": "1"},
    )
    renderer = json.loads(completed.stdout.strip().splitlines()[-1])["renderer"]
    assert renderer == "notebook_connected", (
        f"docs build would render figures with {renderer!r}; nbsphinx drops that "
        "bundle and the figures disappear from the built site"
    )


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
    from phenotypic.sdk_.viz.figures._theme import FONT_FAMILY_MONO

    assert FONT_FAMILY_MONO == GUI_FONT_FAMILY_MONO
