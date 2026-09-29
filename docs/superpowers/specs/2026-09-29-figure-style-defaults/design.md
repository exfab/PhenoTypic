# Figure style defaults for DESIGN.md

- **Date:** 2026-09-29
- **Branch:** `claude/magical-keller-kw5lu5`
- **Status:** design settled; awaiting implementation plan
- **Origin:** choices made in the artifact at
  `docs/superpowers/artifacts/2026-09-29-figure-typography/` (published at
  <https://claude.ai/artifact/9VfsXi3SKYRnneTMoja9gB>), recorded below as the settings
  block the author returned
- **Related:** `2026-09-29-chrome-font-nunito-sans` (lands first) and
  `2026-09-29-design-md-format` (the restructure this section is written for)

## Objective

Give agents a written house style for static figures, meaning figures drawn with
matplotlib for a manuscript, thesis, report or pipeline output, optimized for placement
on an A4 page and suitable for journal submission. The section guides; it does not
bind. Every rule in it is a default that an agent applies when the contributor has not
asked for something else, and a contributor may override any of it without giving a
reason.

## Non-goals

This spec does not restyle the interactive Plotly charts in the GUI, which keep
following the `DESIGN.md` Data Visualization rules; a pipeline figure declared
`@figure(backend="plotly")` is one of those. It adds no test that compares `DESIGN.md`
to code, a check the author declined on 2026-09-29. It also adds no per-journal preset
mechanism, which the author excluded in the artifact.

## Decisions taken in the artifact

```yaml
figures:
  font: dejavu
  ladder: balanced  # ticks 7 pt, labels 8 pt, panels 10 pt
  palette: canonical
  chrome: minimal
  panel_labels: paren
  include: [final-size, fonttype-42, vector-first, reproducible, colormaps, show-points,
            nomenclature, units, scale-bars, svg-text]
  exclude: [journal-override, redundant]
```

The author dropped `redundant` (pair color with marker or line style) in a follow-up
message after returning the block, so it appears under `exclude` here although the
pasted block listed it as included.

## Background

`DESIGN.md` today treats every chart as a screen chart. Its matplotlib `rcParams` block
(07, mirrored by `sdk_/viz/figures/_mpl_theme.py:21`) asks for IBM Plex Sans and
JetBrains Mono, neither of which matplotlib bundles. Under matplotlib 3.10.7 in this
repository's environment, `font_manager.findfont` resolves both to DejaVu Sans with a
"Font family not found" warning, while a machine that has Helvetica Neue installed
matches that entry of the same stack instead, so the same figure already renders in
different faces on different machines. The block also paints the figure background
`#FBFEF8`, which prints as a visible tinted box on white paper.

Three statements in `DESIGN.md` would contradict the chosen defaults if left as they
are, because they are written as absolute rules over all charts. The first is the
Absolute Constraint that no numeric data or axis label may render outside JetBrains
Mono, while figures will set numbers in DejaVu Sans. The second is the Absolute
Constraint and the matching Don't that fix the Okabe-Ito order as navy first, while
figures will use the published order with black first. The third is the Absolute
Constraint that sequential colorbars use the navy-to-sky ramp, while figures will use
cividis or viridis. The restructure in `2026-09-29-design-md-format` scopes all three to
GUI and on-screen charts, and the Figures section below says so from its side.

The size ladder and page geometry rest on these values. A4 is 210 x 297 mm (LaTeX
`classes.dtx`, `a4paper` option). Word's default 1 in side margins leave a text width of
210 - 2 x 25.4 = 159.2 mm and a text height of 297 - 2 x 25.4 = 246.2 mm; the margin
default itself was seen only in a search excerpt. LaTeX's article class on A4 sets a
narrower `\textwidth` of 345, 360 or 390 pt at 10, 11 or 12 pt body size, which is
121.3, 126.5 or 137.1 mm (`classes.dtx`, read directly; millimetres by conversion at
72.27 pt per inch). The journal text-size ranges shown in the artifact came from search
excerpts, because the proxy blocked every publisher's site, so this section does not
cite them as rules; the chosen 7/8/10 pt ladder stands as a house default on its own
footing.

## The section as it will read in DESIGN.md

The text below is the proposed section verbatim. It is written for the canonical
heading layout of `2026-09-29-design-md-format`, where it sits after Data
Visualization.

---

> ## Figures
>
> This section covers static figures drawn with matplotlib: pipeline figures declared
> `@figure(backend="mpl")`, analysis plots, and anything prepared for a manuscript,
> thesis, report or poster. Interactive Plotly charts in the GUI follow Data
> Visualization instead.
>
> ### How to apply these defaults
>
> Everything in this section is a default, not a constraint. When the contributor has
> not said otherwise, apply it. When the contributor asks for something different, such
> as an aggregate-only plot, a journal's own palette, a slide-sized figure or a
> gridded background, do what they asked; they do not need to justify it, and you
> should not argue for the default. When you depart from a default on your own
> judgment because the plot's purpose calls for it, for example plotting a
> distribution summary for 50,000 colonies where individual points would be noise, say
> so in one line in the figure's docstring or in your reply, so the contributor can
> reverse it.
>
> The Absolute Constraints govern GUI chrome and on-screen charts. They do not apply to
> static figures, and nothing in this section is absolute.
>
> ### Page and size
>
> Draw every figure at the size it will be printed, and place it at 100%. The point
> sizes below are only true when nothing rescales the figure afterwards.
>
> | Preset | Width | Use |
> |---|---|---|
> | `full` | 159.2 mm | Full text width of an A4 page with 1 in margins |
> | `half` | 77.1 mm | Two figures side by side with a 5 mm gutter |
>
> Choose the height to suit the content, up to the 246.2 mm text height of the same
> page. Set `figsize` in inches from these millimetre values, and lay out with
> `layout="constrained"` rather than `bbox_inches="tight"`, because the latter changes
> the saved dimensions and with them every point size.
>
> ### Typography
>
> Set all figure text in **DejaVu Sans**, matplotlib's bundled default, so a figure
> renders identically on macOS, Windows and Linux. Use `mathtext.fontset = "dejavusans"`
> so math matches the text, and DejaVu Sans Oblique for italics.
>
> | Text | Size at print | Weight |
> |---|---|---|
> | Tick labels, legend entries, annotations | 7 pt | regular |
> | Axis labels, colorbar labels, legend titles | 8 pt | regular |
> | Panel labels | 10 pt | bold |
>
> Lines follow the same scale: axes and ticks 0.6 pt, data lines 1.25 pt, markers
> 3 pt, tick length 3 pt.
>
> ### Color
>
> Use the Okabe-Ito palette in its published order, starting from black: `#000000`,
> `#E69F00`, `#56B4E9`, `#009E73`, `#0072B2`, `#D55E00`, `#CC79A7`. Skip yellow
> (`#F0E442`) for lines, points and text on white, where it is nearly invisible; it is
> usable as a large fill. Brand navy does not appear in figures. Beyond seven
> categorical series, group the remainder into an "other" category drawn in grey.
>
> For continuous data use a perceptually uniform map: `cividis` by default, `viridis`
> as the alternative. For data that diverge around a reference value, use a diverging
> map centred on that value, such as `RdBu_r`. Avoid `jet`, rainbow maps and red-green
> pairs.
>
> ### Layout and chrome
>
> Use a white figure and axes background, no gridlines, and only the bottom and left
> spines. Leave panel titles out; the caption carries the message. Draw legends
> without a frame.
>
> Label panels **(a)**, **(b)**, **(c)** in bold 10 pt at the top-left corner, just
> outside the axes.
>
> ### Showing data
>
> When the number of observations is small enough to read, plot the individual points
> over the summary, and state n and what error bars or bands show (s.d., s.e.m. or a
> confidence interval) in the caption. When the figure's purpose is the aggregate
> itself, plot the aggregate.
>
> Label axes as "Quantity (unit)", for example "Colony area (mm²)", and keep
> matplotlib's true minus sign.
>
> Italicize genus and species names and gene names, following the organism's
> nomenclature. For *Saccharomyces cerevisiae*, a mutant allele is lowercase italic
> (*ura3Δ*), the wild-type gene uppercase italic (*URA3*), the protein roman (Ura3),
> and strain identifiers roman.
>
> ### Image panels
>
> Burn a scale bar with a length label into every image panel of calibrated data; if
> the image is uncalibrated, label the bar in pixels. Show single channels in
> grayscale. Embed raster images at 300 dpi or more at their printed size, and at
> 600 dpi when they are combined with line art in the same file.
>
> ### Export
>
> | Output | Setting | Why |
> |---|---|---|
> | PDF (primary) | `pdf.fonttype = 42` | Text is embedded as TrueType, so it stays selectable and editable |
> | SVG | `svg.fonttype = "none"` | Text stays editable; it renders in a fallback face where DejaVu Sans is not installed, so treat the PDF as the reference |
> | PNG (preview) | `dpi=300` | For quick viewing only |
>
> Make exports reproducible: pass `metadata={"CreationDate": None}` for PDF and
> `metadata={"Date": None}` for SVG, and fix `svg.hashsalt`. matplotlib also honours
> `SOURCE_DATE_EPOCH`.

---

## Implementation outline

The section guides agents, but the code should produce these defaults when nobody
overrides them, or agents will fight the theme on every figure. The plan will cover
four changes.

`sdk_/viz/figures/_mpl_theme.py` switches `phenotypic_rc()` from the screen look to the
defaults above: DejaVu Sans, the 7/8/10 pt ladder and line widths, white backgrounds,
no grid, `pdf.fonttype` 42, `svg.fonttype` "none" and a fixed `svg.hashsalt`. Because
`@figure(backend="mpl")` wraps every matplotlib pipeline figure in
`phenotypic_mpl_context()` (`abc_/plotting/_pht_plot.py:259`), all of them pick up the
change.

`sdk_/_palette.py` gains the published Okabe-Ito order as a second, separately named
tuple. The existing `OKABE_ITO` (navy first) stays as it is, because the Plotly template
and the GUI read it.

The figures package gains named width presets and a helper that turns a preset and a
height in millimetres into a `figsize`, so a figure author never converts units by
hand. Today 62 `figsize=` call sites in `src/` use about a dozen ad hoc sizes; the plan
decides which of them are publication figures that should adopt a preset and which are
diagnostics that may keep their own size.

Existing tests that pin the screen look of the matplotlib theme change with it. The
plan identifies them by importer, not by directory.

One consequence needs a decision in the plan. Pipeline figures are stored inside the
per-image OME-Zarr stores (`figures/<run>/`), so a restyled theme changes those bytes
across versions. The layer semantics do not change, which suggests no bump of
`PROCESS_LAYER_SEMANTICS_REVISION`, so continued process runs would keep old-style
figures for images they already finished.

## Open question

Nothing in the section above is absolute, as the author asked. One item may deserve
different treatment: avoiding red-green colormaps is an Absolute Constraint for GUI
charts because it protects readers with deuteranopia, and the section above demotes it
to a default for figures. If the author wants that one item to stay absolute in
figures too, it moves back into the Absolute Constraints with its scope widened.

## References

- Rougier, N. P., Droettboom, M., & Bourne, P. E. (2014). Ten simple rules for better
  figures. *PLOS Computational Biology*, 10(9), e1003833.
  <https://doi.org/10.1371/journal.pcbi.1003833>
- Wong, B. (2011). Points of view: Color blindness. *Nature Methods*, 8, 441.
  <https://doi.org/10.1038/nmeth.1618>. The palette values and order were read from
  secondary sources, not from the article itself.
- Crameri, F., Shephard, G. E., & Heron, P. J. (2020). The misuse of colour in science
  communication. *Nature Communications*, 11, 5444.
  <https://doi.org/10.1038/s41467-020-19160-7>
- Weissgerber, T. L., Milic, N. M., Winham, S. J., & Garovic, V. D. (2015). Beyond bar
  and line graphs: time for a new data presentation paradigm. *PLOS Biology*, 13(4),
  e1002128. <https://doi.org/10.1371/journal.pbio.1002128>
- matplotlib 3.10.7 bundled fonts and `rcParamsDefault`, inspected in this repository's
  environment on 2026-09-29; matplotlib source at <https://github.com/matplotlib/matplotlib>
  for `pdf.fonttype`, `svg.hashsalt` and `SOURCE_DATE_EPOCH` behaviour.
- LaTeX `classes.dtx` for A4 dimensions and article-class text widths.
- Yeast genetic nomenclature follows the Saccharomyces Genome Database conventions;
  this session did not verify them against the SGD page, so the example above is
  [based on general knowledge of the field; no citation verified].
