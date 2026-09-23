# Custom pipeline plot

Plot classes are ordinary serializable models that opt into a lifecycle from
`phenotypic.abc_.plotting`. Add the same configured object to its normal pipeline
slot and to `ImagePipeline(plots=[...])`; the serialized plot binding preserves that
shared identity.

## Declare figures with `@figure`

Write each figure as a method and mark it with the `@figure` decorator. The
histogram below shows how colony area is distributed across a plate:

```python
import plotly.graph_objects as go
from pydantic import BaseModel, ConfigDict

from phenotypic.abc_.plotting import PlotMeas, figure


class PlotColonyArea(BaseModel, PlotMeas):
    model_config = ConfigDict(extra="forbid")

    area_column: str = "Size_Area"

    @figure(title="Colony area", backend="plotly", primary=True)
    def area_histogram(self, measurements):
        return go.Figure(
            go.Histogram(x=measurements[self.area_column], name=self.area_column)
        )
```

Pass an instance to `plots=` and the pipeline publishes it every time the
measurement mirror is updated. In a notebook, call `inspect()` directly:

```python
from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureShape, MeasureSize

pipeline = ImagePipeline(
    ops={"detect": OtsuDetector()},
    meas={"size": MeasureSize(), "shape": MeasureShape()},
    plots=[PlotColonyArea()],
)

plate = load_synth_yeast_plate()
measurements = pipeline.apply_and_measure(plate, inplace=True)
fig = PlotColonyArea().inspect(measurements)
```

The decorator does three things for each figure method:

- **It applies the house theme.** A Plotly figure is themed after the method
  returns it. A matplotlib figure is built inside the PhenoTypic style context,
  because matplotlib styling has to be in effect while the figure is drawn.
- **It checks what the method returns.** If a method declared
  `backend="plotly"` returns a matplotlib figure, or the reverse, rendering
  raises `TypeError`. A method that returns `None` or an array raises the same
  error. The figure is never converted and the check never falls back to the
  other backend.
- **It declares the backend before any figure is drawn,** which is what lets
  the CLI check that backend before it processes a single plate (see
  Preflight, below).

`backend` is required and has no default. Leaving it out raises `TypeError`
when the class is defined. `inspect()` renders the figure marked
`primary=True`, or the only figure if the class declares just one. `report()`
composes every declared figure into one Plotly figure, or builds a notebook
dashboard when a figure declares `controls`.

Use `PlotImage` for per-image output, `PlotMeas` for the post-applied measurement
mirror, `PlotAnalysis` for a named analysis table, and `PlotQc` for QC-aware output.
The pipeline calls `inspect(subject, for_save=True)`. The decorated method
receives `for_save` only when it declares a parameter of that name, so declare
one if the saved figure should differ from the interactive one, for example by
showing every overlay trace instead of hiding some behind the legend.

### Matplotlib figures

Declare `backend="mpl"` and return a `matplotlib.figure.Figure`. Construct it
explicitly rather than through `pyplot`:

```python
from matplotlib.figure import Figure
from pydantic import BaseModel, ConfigDict

from phenotypic.abc_.plotting import PlotMeas, figure


class PlotColonyCircularity(BaseModel, PlotMeas):
    model_config = ConfigDict(extra="forbid")

    circularity_column: str = "Shape_Circularity"

    @figure(title="Colony circularity", backend="mpl", primary=True)
    def circularity_histogram(self, measurements):
        fig = Figure(figsize=(6, 4))
        ax = fig.subplots()
        ax.hist(measurements[self.circularity_column].dropna(), bins=30)
        ax.set_xlabel(self.circularity_column)
        ax.set_ylabel("Colonies")
        return fig
```

`report()` cannot compose matplotlib figures. It raises `TypeError` for any
class that declares an `mpl` figure, even a class whose only figure is
matplotlib. Call `inspect()`, which is unaffected, or override `report()` with a
plot-specific implementation.

### Decorating an `inspect()` override

A class that needs its own `inspect()` signature can decorate the override
itself. The override then declares its backend just as a figure method does,
and it is themed and checked in the same way. `MeasureSymZones` decorates its
`inspect()` this way. A smaller example plots colony area against circularity:

```python
import plotly.graph_objects as go
from pydantic import BaseModel, ConfigDict

from phenotypic.abc_.plotting import PlotMeas, figure


class PlotAreaVsCircularity(BaseModel, PlotMeas):
    model_config = ConfigDict(extra="forbid")

    @figure(title="Area against circularity", backend="plotly", primary=True)
    def inspect(self, measurements=None, *, for_save=False):
        if measurements is None:
            raise TypeError("PlotAreaVsCircularity requires measurements")
        return go.Figure(
            go.Scatter(
                x=measurements["Size_Area"],
                y=measurements["Shape_Circularity"],
                mode="markers",
            )
        )
```

## Escape hatch: override `inspect()` without the decorator

When `inspect()` must return several pages, or compose figures in a way no
single figure method can, override it and return a `PlotOutput` of
deterministically keyed `PlotPage` entries. Import both output contracts from
`phenotypic.abc_.plotting`. `phenotypic.plotting` contains only the
ready-to-use plot models.

An undecorated override bypasses what the decorator does: the figures it
returns are not themed, their backend is not checked, and the CLI preflight
classifies the plot as "undeclared". You can keep most of that by building each
page from decorated figure methods, as this two-page morphology report does:

```python
import plotly.graph_objects as go
from pydantic import BaseModel, ConfigDict

from phenotypic.abc_.plotting import PlotMeas, PlotOutput, PlotPage, figure


class PlotColonyMorphology(BaseModel, PlotMeas):
    model_config = ConfigDict(extra="forbid")

    @figure(title="Colony area", backend="plotly")
    def area(self, measurements):
        return go.Figure(go.Histogram(x=measurements["Size_Area"]))

    @figure(title="Colony circularity", backend="plotly")
    def circularity(self, measurements):
        return go.Figure(go.Histogram(x=measurements["Shape_Circularity"]))

    def inspect(self, subject=None, *, for_save=False, **overrides):
        del for_save, overrides
        if subject is None:
            raise TypeError("PlotColonyMorphology requires measurements")
        return PlotOutput(
            pages=(
                PlotPage(key="area", label="Colony area", figure=self.area(subject)),
                PlotPage(
                    key="circularity",
                    label="Colony circularity",
                    figure=self.circularity(subject),
                ),
            )
        )
```

Because the pages come from decorated methods, each one is themed and
backend-checked, and `report()` still composes them. A class that declares no
`@figure` methods at all must also override `report()`, because the inherited
one raises `RuntimeError` when there is nothing to compose.

## Where plots are published

The CLI publishes every plot below `deliverables/plots/<id>/`, where `<id>` is
the plot's binding id. The id depends on how the plot entered `plots=`:

| The plot object is… | Binding id | Directory |
|---|---|---|
| owned by a keyed slot (`ops`, `meas`, `post`, `filters`, `qc`) | the slot key | `meas={"sym": sym}` → `plots/sym/` |
| the pipeline's `model` | the class name | `model=LogGrowthModel(...)` → `plots/LogGrowthModel/` |
| not in any slot (inline) | the class name | `plots=[PlotColonyArea()]` → `plots/PlotColonyArea/` |

The pipeline matches a plot to a slot by identity, so pass the same object to
both:

```python
from phenotypic.measure import MeasureSymZones

sym = MeasureSymZones()
pipeline = ImagePipeline(
    ops={"detect": OtsuDetector()},
    meas={"sym": sym},
    plots=[sym],
)
```

The `model` slot is the exception among pipeline-owned objects. It holds one
object with no key, so its binding falls back to the class name, as an inline
plot's does.

Inside `plots/<id>/`, the layout depends on the lifecycle and on what
`inspect()` returned:

| Output | Published as |
|---|---|
| `PlotImage`, one figure | `<dataset>/<stem>-<hash>.<ext>`, one file per stored format, plus `.html` for a stored `plotly-json`; no manifest |
| `PlotImage`, a multi-page `PlotOutput` | `<dataset>/<stem>-<hash>/`, holding the pages and a `manifest.json` |
| `PlotMeas`, `PlotAnalysis`, `PlotQc` | the pages and a `manifest.json`, directly in `plots/<id>/` |

`<hash>` is derived from the original dataset and image name, so two plates
whose names differ only in characters that are unsafe in a filename never
overwrite each other. A page's filename comes from its `label`, or its `key`
when it has no label. A single figure returned from an aggregate plot is the
page `default`, so `PlotColonyArea` publishes `plots/PlotColonyArea/default.html`.

The tree also contains hidden `.lock` files, such as `.publication.lock` in
each manifest directory and `.plotlyjs.lock` and `.failures.lock` at the
plots root. They serialise concurrent writers, for example SLURM workers
publishing into the same run, and are not plot output.

## Storing figures with the image

A `PlotImage` figure is written **into the image's OME-Zarr store**, so a
dashboard can show a plate's figures from its `.ome.zarr` alone. The CLI does
this in `--mode full`, `--mode measure`, staged GPU runs, and
`--mode process --process-format zarr`. A flat `--process-format tiff` export
has no store, so it has no figures. `deliverables/plots/<id>/` is a **copy** of
the store's files, made once the store is in place. Aggregate plots
(`PlotMeas`, `PlotAnalysis`, `PlotQc`) have no per-image store and still
publish straight to `deliverables/plots/`, as described under **Two renderings**.

### Choosing the formats: `store=`

`@figure` takes a `store=` tuple naming the formats to keep. The set is closed:

| Format | File in the store | `media_type` | Backends |
|---|---|---|---|
| `plotly-json` | `<page>.plotly.json` | `application/vnd.plotly.v1+json` | `plotly` |
| `png` | `<page>.png` | `image/png` | `plotly` (needs Chrome), `mpl` |

Leave `store` out and each backend stores its default. For `backend="plotly"`
that is `("plotly-json",)`: lossless, interactive in any Plotly consumer, and
needing no Chrome. For `backend="mpl"` it is `("png",)`. Both formats are
byte-stable: the same figure serializes to the same bytes in every process.

A declaration the store cannot honour raises `TypeError` when the class is
defined. That covers a bare string (`store="png"`, write `store=("png",)`),
an empty tuple, an unknown or repeated name, and `"plotly-json"` on
`backend="mpl"`. An empty tuple is refused because `deliverables/plots/` is
copied from the store, so a figure that stores nothing would appear nowhere.

Two consequences are easy to miss:

- **A default Plotly figure has no PNG in `deliverables/`, even on a machine
  with Chrome.** The copy-out never renders; it copies what was stored and
  writes an `.html` page from each stored `plotly-json`. To get a PNG, declare
  `store=("plotly-json", "png")`. On a machine without Chrome, that declared PNG
  is then recorded as a failure, because you asked for it.
- **A Plotly figure built on the plate image is MB-sized as JSON.** Its size
  follows the figure, and a trace-only figure such as a histogram is KB-sized.
  But `px.imshow(..., binary_string=True)`, which the zone measurers use for
  their plate overview, embeds the whole image as a base64 PNG data URI, so the
  stored `plotly-json` is at least as large as a PNG of the plate.

The binding stores what the figure its `inspect()` renders declares. That is
the decorated `inspect()` override if there is one, or else the primary figure.
An **undecorated** `inspect()` override declares nothing. Each page it returns
is stored in the default for that page's own backend, and a page whose figure
type is neither Plotly nor matplotlib is recorded as a failed page.

The histogram of colony areas below keeps a PNG beside the Plotly JSON:

```python
import numpy as np
import plotly.graph_objects as go
from pydantic import BaseModel, ConfigDict

from phenotypic import ImagePipeline
from phenotypic.abc_.plotting import PlotImage, figure
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector


class PlotColonySizes(BaseModel, PlotImage):
    """Distribution of colony areas on one plate, in pixels."""

    model_config = ConfigDict(extra="forbid")

    @figure(
        title="Colony sizes",
        backend="plotly",
        store=("plotly-json", "png"),
        primary=True,
    )
    def colony_sizes(self, image):
        areas = np.bincount(image.objmap[:].ravel())[1:]
        return go.Figure(go.Histogram(x=areas[areas > 0], name="Area (px)"))


sizes = PlotColonySizes()
pipeline = ImagePipeline(ops={"detect": OtsuDetector()}, plots=[sizes])

plate = load_synth_yeast_plate()
pipeline.apply(plate, inplace=True)
fig = sizes.inspect(plate)
print([spec.store for spec in sizes.iter_figures()])
# [('plotly-json', 'png')]
```

A CLI run of this pipeline stores `PlotColonySizes/default.plotly.json` and
`PlotColonySizes/default.png` in each plate's run folder, and publishes both
under `deliverables/plots/PlotColonySizes/<dataset>/`, next to an `.html` page
generated from the JSON.

### Run folders

A store keeps one folder per run: `figures/<date>-<hash>/`. `<date>` is the
UTC date the run started, and `<hash>` is the first 12 hex characters of the
pipeline's sha256. The date is recorded once per run, so a run resumed on a
later day, or one that crosses midnight, still writes a single folder. Folders
are never deleted. A rerun with another pipeline or on another day adds a
folder beside the earlier ones, and a rerun with the same pipeline on the same
day replaces its own. `--mode measure` with the same pipeline as an earlier run
reuses that run's folder. The one exception is `--overwrite`, which deletes the
whole output folder, figures included. `deliverables/plots/` holds a copy of
**this run's** folder only. Reading a store's figures directly, including
choosing among its runs, is covered in
[Store Results in OME-Zarr](../../how_to/pages/zarr_storage.md).

### Figures that can only be drawn where the operation ran

Some figures cannot be redrawn from the finished image. `CalibrateColorRpcc`'s
tile overlay shows the as-shot checker pixels, and the colour correction then
overwrites them. A provider like that raises `FigureInputUnavailable` (from
`phenotypic.abc_.plotting`) from `inspect()` when it is given an image its own
`apply()` did not just process in this process. Keep the reference to that
image weak.

The exception is not a failure. The CLI keeps the figure already stored in
this run's folder, for example the overlay staged Stage 1 drew before Stage 3
rewrote the store. If this run's folder holds none, the CLI lists the binding
in the run's `unavailable` list, and any earlier run's folder keeps its copy.

## Two renderings

This section describes aggregate plots (`PlotMeas`, `PlotAnalysis`,
`PlotQc`). An image plot publishes what its store holds; see
**Storing figures with the image**.

A Plotly binding always publishes an interactive `.html` page. It publishes a
`.png` as well when Kaleido can drive Chrome to rasterise the figure. A
matplotlib binding publishes `.png` only, because matplotlib has no HTML export.

Chrome (or Chromium) is an external dependency that `uv sync` does not install.
Fetch one with `uv run plotly_get_chrome`, or point the `BROWSER_PATH`
environment variable at a browser you already have. Without it, Plotly plots
still publish their HTML pages and the run completes.

The HTML pages do not each embed Plotly's JavaScript. The run writes one
`plotly.min.js` at `deliverables/plots/plotly.min.js`, and every page loads it
through a relative path such as `../../plotly.min.js`. If you copy a plot
directory somewhere else, copy `plotly.min.js` too and keep it at the same
relative position, or the pages will open blank.

A rerun removes the other rendering of each page it publishes when that
rendering is left over from an earlier run. A PNG written on a node that had
Chrome is removed when the plot is rerun on a node without it, and an HTML page
is removed when a plot switches to matplotlib. Otherwise the leftover would sit
beside the new file as if this run had produced it. This applies to
single-figure image plots and to every page a new manifest lists.

The cleanup covers only those pages. If a page key from an earlier run no
longer appears, or a page fails on the rerun, its old files stay on disk.
In a manifest directory, trust `manifest.json` rather than a directory listing
for what the latest run published.

## Preflight

Before any image work begins, CLI validation checks what the configured plots
can render:

- **Chrome missing** is not an error. Validation logs one warning that names
  the Plotly bindings that will publish HTML without PNG, and suggests
  `plotly_get_chrome` to install it. An image plot is named only if it
  declares a Plotly `png`. Such a plot is listed as recording that PNG as
  failed, or, if `png` is its only format, as publishing nothing.
- **A declared backend whose library cannot be imported** fails validation
  with `Plot backend unavailable: …`, naming the bindings that declared it.
  Such a plot could not publish anything, so the run stops before it starts.

Each binding is classified by the figure its `inspect()` renders. If `inspect`
itself carries `@figure`, that is its backend. Otherwise, if the class uses the
inherited `inspect()`, its primary figure's backend is used. An undecorated
`inspect()` override is "undeclared": its backend is known only once it
renders, so the Chrome warning names it conditionally and the import check
never fails it.

## The plot manifest

Every directory holding a `manifest.json` records what the latest publication
wrote there. The manifest is at `schema_version: 2`:

```json
{
  "schema_version": 2,
  "plot_id": "PlotColonyMorphology",
  "class": "PlotColonyMorphology",
  "renderers": {"html": "available", "png": "unavailable: chrome not found"},
  "pages": [
    {"key": "area", "label": "Colony area", "backend": "plotly",
     "files": {"html": "Colony-area.html"}, "metadata": {}}
  ],
  "failed": [
    {"key": "circularity", "label": "Colony circularity",
     "error": "OSError: [Errno 28] No space left on device"}
  ]
}
```

- `renderers` records what this machine could render for the directory, not
  what landed on disk. A Plotly directory has `html: "available"` and a `png`
  verdict of `"available"` or `"unavailable: chrome not found"`. A matplotlib
  directory has only `png: "available"`. A directory that mixes both backends
  without Chrome reads `"available: matplotlib only; chrome not found"`. A
  manifest copied out for a multi-page image plot never probes Chrome. It
  records `html: "available"` when a page is Plotly and `png: "available"` when
  a page is matplotlib. Its `files` show which stored formats were copied, and
  `partial` lists the errors of the formats that failed for that page, whether
  the store recorded them or the copy hit them.
- `pages` lists each published page. `files` maps each format to the file it
  produced. It omits `png` when Chrome was unavailable and `html` for a
  matplotlib page. `backend` is `"plotly"` or `"matplotlib"`.
- `partial` appears on a page when one renderer failed but the other
  succeeded, for example when the HTML was written and the PNG export raised.
  It lists the errors.
- `failed` lists each page that produced no file at all, with its `key`,
  `label` and `error`. When every page fails, the manifest still says why.

## The failure record

Plots are best-effort: one broken figure does not stop a run that produced good
measurements. So that a missing plot is not also a silent one, every plot
failure the run swallows appends one JSON line to
`deliverables/plots/.failures.jsonl`:

```json
{"binding_id": "sym", "dataset": "plate_a", "error": "TypeError: ...",
 "image_stem": "A01", "lifecycle": "image", "plot_class": "MeasureSymZones",
 "ts": "2026-09-20T18:04:11Z"}
```

`lifecycle` is one of `image`, `measurements`, `analysis` and `qc`, the four
points at which a plot is refreshed. It is `page` when one page of a
multi-page output failed to render, whichever of those four was running.
`dataset` and `image_stem` appear only for image-lifecycle entries.
`plot_class` is `<unresolved>` when a QC plot failed before its class could be
determined.

An image-lifecycle entry may also carry `page` and `format`: the page key, and
the store format that failed, or `html` for the page generated from a stored
`plotly-json`. A failure the store recorded is copied with its `error` text
exactly as stored. A stored file that no longer matches its recorded sha256 is
recorded under its own binding, page and format, and is not copied. Two
binding ids are not plots. `<store>` means the store's figures descriptor could
not be read, for example because its `schema_version` is newer than this
release knows. `<run>` means the image's run folder could not be named, so the
image was written without figures.

Some refusals are not plot failures. When the GUI's snapshot check or a SLURM
worker's lifecycle fence refuses a publication, the process can no longer
confirm that it owns the output, so the whole refresh is abandoned and nothing
is written to `.failures.jsonl`.
