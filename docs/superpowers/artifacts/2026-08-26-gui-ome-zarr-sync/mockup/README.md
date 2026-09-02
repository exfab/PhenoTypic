# Viewer rebuild mockups

Artboards for the rebuilt results viewer, backing
[`specs/2026-08-26-viewer-viv-rebuild`](../../../specs/2026-08-26-viewer-viv-rebuild/design.md) §6.

| File | Artboard |
|---|---|
| `Main.dc.html` | **Plate view** — Viv/deck.gl full-canvas deep zoom, floating vizarr-style controls |
| `ColonyGrid.dc.html` | **Colony view** — the `gui/_smart_grid/` port: N viewports, shared camera, curation radial retained |
| `canvas.json` | Canvas layout, frame sizes, and the two sticky notes |

These three files are the **source**. The published canvas
(https://claude.ai/code/artifact/7a8c50b6-042f-4948-9452-d6b6e557239f) is seeded from
them and is deliberately not committed — it carries ~2.5 MB of editor code and is
regenerable.

All chrome derives from the real design system rather than invented values:
`gui/_design.py` tokens (navy `#003660`, blue `#1b75bc`, gold `#febc11`, stage
`#0e1620`, Comfortaa + JetBrains Mono) and `results_viewer/_assets/results_viewer.css`
(colony cell borders `rgba(27,117,188,0.13)`, navy 2px selected outline, 16px checkbox).
The radial's six wedges are the actual `ERROR_CATEGORY_COLORS` map.

**Illustrative, not real data:** the plate is a synthetic 12×8 layout, and the filter
summary (`Size_Area > 120`) and the `halo` custom category are placeholders.
