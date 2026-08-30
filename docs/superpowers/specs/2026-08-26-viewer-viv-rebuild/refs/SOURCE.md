# Vendored upstream sources — viewer Viv rebuild

Read-only reference artifacts. Every `file:line` claim in this spec, its plan
and the facade's comments resolves against these copies. **Never lint,
format, "tidy" or bug-fix them** (root `CLAUDE.md`, "Porting a Reference
Algorithm"); `[tool.ruff] extend-exclude` covers `docs/superpowers/**/refs`,
but the rule binds regardless of tool.

## Provenance

`@vivjs/*` and `@zarrita/*` are the **exact npm artifacts the committed
bundle was built from**, not a branch snapshot: each tarball's SHA-512 was
checked against `tools/viv-bundle/package-lock.json`, and each extracted file
was diffed byte-for-byte against `tools/viv-bundle/node_modules/`. The viv
repository stops tagging at `v0.14.2` — releases since then ship through
changesets with no git ref — so **there is no upstream git ref for 0.22.1**
and the tarball is the only pinned artifact.

| Local file | Upstream | Retrieved | SHA-256 |
|---|---|---|---|
| `vivjs-loaders-0.22.1/index.mjs` | `@vivjs/loaders@0.22.1` tarball, `dist/index.mjs` | 2026-08-27 | `b4f26b12ea837f5345693e2b09f43ed5fb46ac99de965f9b60dddf9caf2dedf1` |
| `vivjs-loaders-0.22.1/LICENSE` | same tarball, `LICENSE` | 2026-08-27 | `674554cfd20dffadaa9bce34cb0e9928c287cee4953a190cf4243d9783dc64a7` |
| `vivjs-layers-0.22.1/index.mjs` | `@vivjs/layers@0.22.1` tarball, `dist/index.mjs` | 2026-08-27 | `e19cc35c784ad24d47e5163311581636cd81a24a8f5c4534c77a313a9b170691` |
| `vivjs-layers-0.22.1/LICENSE` | same tarball, `LICENSE` | 2026-08-27 | `674554cfd20dffadaa9bce34cb0e9928c287cee4953a190cf4243d9783dc64a7` |
| `zarrita-storage-0.1.4/fetch.ts` | `@zarrita/storage@0.1.4` tarball, `src/fetch.ts` | 2026-08-27 | `a0342e19b1c15ef5421c5e27d5a37431a4ae7a923c8e3d586c570c76f864fce8` |
| `zarrita-storage-0.1.4/types.ts` | same tarball, `src/types.ts` | 2026-08-27 | `cf5540db1b5635936091dc6a38b5b0a43ae61fa1ed98c9bab409af14e96d77ee` |
| `zarrita-storage-0.1.4/util.ts` | same tarball, `src/util.ts` | 2026-08-27 | `60c860c6f444c992c3e09f2d3ea1ce2dac4bee4009a48dc41b85be4cfe5cca4a` |
| `vizarr-be7ccc26/io.ts` | `hms-dbmi/vizarr@be7ccc26`, `src/io.ts` | 2026-08-27 | `f13c1459c29db4d821b43b5176531bc10142acadde7915a7f2137ef53bff8bce` |
| `vizarr-be7ccc26/ome.ts` | same commit, `src/ome.ts` | 2026-08-27 | `bb228cc5e442b1446ef8ec0e8aa8d6fe4557c6fb9d6a07935ad4b5748382cb86` |
| `vizarr-be7ccc26/utils.ts` | same commit, `src/utils.ts` | 2026-08-27 | `8bc32cfe032fa3b0a334992284496dd49b0cd7b30d892a315530f5d4a098bf70` |

Tarball integrity, as recorded in `tools/viv-bundle/package-lock.json` and
re-verified on download:

```text
@vivjs/loaders@0.22.1   sha512-DxdKcMbXLpWDMFUizMYAqk4+GyLozFaG+J3RlFY0U8RA9OpQQ7tlaPfwUjj1lRBvLruDN+RUfDxoQYuW/+74eg==
@vivjs/layers@0.22.1    sha512-gRNZSGsRsM+Gf+pirqWtSbtc9ecY5Epd58hmJPFD4XOC87HcZ5xhM5BCzxG+/jp4H88HG7v/Ax1FHk6zvuy2uQ==
@zarrita/storage@0.1.4  sha512-qURfJAQcQGRfDQ4J9HaCjGaj3jlJKc66bnRk6G/IeLUsM7WKyG7Bzsuf1EZurSXyc0I4LVcu6HaeQQ4d3kZ16g==
```

`vizarr` is a git-only project (no npm package for the app). The three files
were fetched from `raw.githubusercontent.com` at commit
`be7ccc260e848a2829873c8746f32b4f43599435` (2026-04-16), and each file's git
blob SHA-1 was checked against the GitHub contents API:

```text
io.ts     e05cd3ee37da693d55d1b436aa6a843b984f3f94
ome.ts    b17b65e86e4b9496ec3f54d0646cb33b62fe77c6
utils.ts  2ae11eecfe93d5cd19eaf4487f837cdfdd9734b7
```

## What each file is evidence for

### `vivjs-loaders-0.22.1/index.mjs` — why the resolver exists

`loadMultiscales(store, path = "")` (`:1075`) opens the group at `path`,
reads `attributes.ome.multiscales`, and when that key is absent falls back to
`let paths = ["0"]` (`:1082`) and opens `<root>/0` as an array. Our root group
carries only `{version, "bioformats2raw.layout": 3}`, and its children are
`rgb` / `gray` / `detect_mat` / `OME` — hence the spike's
`Node not found: v2 array`. **Nothing in this file reads `OME/zarr.json`.**

`load(store)` (`:1256`) is exported as `loadOmeZarrFromStore` and is what the
facade calls: it takes a store rather than a URL, which is what lets the
facade substitute its own status-contract-aware store. `loadOmeZarr(source)`
(`:1266`) builds a plain `FetchStore` and additionally **throws** unless
`options.type === "multiscales"` (`:1268-1270`).

### `vivjs-layers-0.22.1/index.mjs` — why `visible: false` is not enough

`MultiscaleImageLayer.renderLayers` builds its background sublayer as
`new ImageLayer(this.props, { id: "Background-Image-<id>", ..., visible:
!viewportId || this.context.viewport.id === viewportId, ... })`
(`:1009-1013`). The second argument overrides the inherited `visible`
unconditionally, so setting `visible: false` on the composite hides the
composite and its tiled sublayer but leaves the background one painting the
whole extent at low resolution. Measured; it is why the facade hides a layer
by removing it and rebuilds on the way back.

`ImageLayer.updateState` calls `loader.getRaster(...)` on a **single** pixel
source (`:776`), while `MultiscaleImageLayer` indexes `loader[resolution]`
(`:954`) — an array. Passing the level array to `ImageLayer` fails at layer
initialization with `e.getRaster is not a function` and paints nothing.

### `zarrita-storage-0.1.4/*.ts` — the status contract the facade replaces

`handle_response` (`fetch.ts:16-30`) returns `undefined` on **404** and
throws `Unexpected response status …` on every other non-2xx. That is why
the byte route answers Zarr v2 metadata probes with 404 rather than 400
(phase 1), and why the facade carries its own store: `FetchStore` would
flatten 409 and 422 into one opaque error. `fetch_suffix`
(`fetch.ts:30-48`) with the default `useSuffixRequest: false` does a HEAD
then an offset range, and passes the FULL content length as the range length
(`util.ts:26-42`), so the request overshoots EOF and the server clamps — the
facade replicates this deliberately.

### `vizarr-be7ccc26/*.ts` — and the thing it turns out NOT to do

The plan expected to vendor "whatever vizarr module resolves the
`bioformats2raw.layout` series list". **There is no such module.**
`classifySource` recognises the layout (`io.ts:102`, via
`utils.isBioformats2rawlayout`, `utils.ts:478-480`) and `createSourceData`
then *refuses* it: `throw new utils.RedirectError("Please open in
ome-ngff-validator", …)` (`io.ts:136-139`). So neither viv nor vizarr
resolves our root layout, and the resolver phase 3 writes is genuinely ours
rather than an adaptation.

What vizarr *does* do, and what phase 3 mirrors in shape, is label
resolution: `resolveOmeLabelsFromMultiscales` (`ome.ts:340-344`) opens
`<image>/labels` as a group and reads `attrs.labels`, then loads each named
child as its own multiscale source (`ome.ts:303`, `:317`). PhenoTypic writes
that same NGFF child (`rgb/labels/zarr.json` carries `ome.labels:
["objmap"]`) but resolves it through `attributes.phenotypic.labels.objmap`
instead — the key is optional, and an rgb-less store puts the label under
`gray`.
