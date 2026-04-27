"""Top-level test configuration.

Ensures that calling ``.show()`` on plotly or matplotlib figures during tests
does not spawn browser tabs or GUI windows.
"""

import matplotlib

matplotlib.use("Agg")

try:
    import plotly.io as pio

    pio.renderers.default = "json"
except ImportError:
    pass
