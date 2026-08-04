# Custom pipeline plot

Plot classes are ordinary serializable models that opt into a lifecycle from
`phenotypic.abc_.plotting`. Add the same configured object to its normal pipeline
slot and to `ImagePipeline(plots=[...])`; the serialized plot binding preserves that
shared identity.

```python
import plotly.graph_objects as go
from pydantic import BaseModel, ConfigDict

from phenotypic.abc_.plotting import PlotMeas


class PlotColonyArea(BaseModel, PlotMeas):
    model_config = ConfigDict(extra="forbid")

    area_column: str = "Size_Area"

    def inspect(self, subject=None, *, for_save=False, **overrides):
        del for_save, overrides
        if subject is None:
            raise TypeError("PlotColonyArea requires measurements")
        return go.Figure(
            go.Histogram(x=subject[self.area_column], name=self.area_column)
        )

    def report(self, subject=None, **overrides):
        return self.inspect(subject, **overrides)
```

Use `PlotImage` for per-image output, `PlotMeas` for the post-applied measurement
mirror, `PlotAnalysis` for a named analysis table, and `PlotQc` for QC-aware output.
`inspect()` returns the primary saveable figure. `report()` returns the complete
interactive report. Plotly and Matplotlib figures are both accepted by the CLI
publisher.

The CLI writes plots below `deliverables/plots/<ClassName>/`. A multi-page plot may
return `PlotOutput` with deterministically keyed `PlotPage` entries. Import both
output contracts from `phenotypic.abc_.plotting`; `phenotypic.plotting` contains
only the ready-to-use plot models.
