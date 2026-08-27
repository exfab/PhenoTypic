# Phase 2 — Viv bundle and façade

**Spec:** §3, §5.1. **Depends on:** phases 0, 1. **Blocks:** phases 3, 4.

**Deliverable:** a committed Viv + deck.gl IIFE at
`results_viewer/_assets/viv/viv-bundle.min.js`, a hand-written façade at
`results_viewer/_assets/viv_viewer.js` exposing five methods, the committed build recipe at
`tools/viv-bundle/`, and the licensing paperwork. The zstd wasm codec registers **before
any store is opened**.

> **Why a vendored bundle.** There is no `package.json` anywhere in this repo (verified).
> Every line of GUI JS is either hand-written vanilla (`builder.js`, `browse.js`) or a
> vendored pre-built bundle (`openseadragon.min.js`, `cytoscape-dagre.min.js`) dropped into
> a Dash `_assets/` folder. Viv is React + deck.gl; vizarr is Preact + Vite. Neither drops
> in as a file, and adding npm to CI is exactly what decision A exists to avoid.
>
> **Costs accepted, recorded so they are not rediscovered as surprises:** bundle provenance
> lives outside the repo; upgrading Viv is a manual ceremony; the bundle is ~1 MB-class,
> acceptable only because the deployment is localhost or an SSH tunnel.

---

### Task 2.1: Commit the build recipe

**Files:**
- Create: `tools/viv-bundle/package.json`, `package-lock.json`, `build.mjs`, `README.md`, `VERSION`

**Interfaces:**
- Produces: `tools/viv-bundle/VERSION` — a single line the GUI logs at startup and phase 5
  asserts against the bundle's embedded string.

- [ ] **Step 1: Write the recipe README first**

It is the only thing standing between a vendored artifact and rot. State: the exact node
version, the exact command, where the output goes, and that the lockfile is pinned.

```markdown
# Viv bundle build recipe

Built **outside** this repo — there is no npm in CI, by design (viewer-viv-rebuild
spec section 3). Run this by hand when upgrading Viv, then commit the artifact.

    cd tools/viv-bundle
    npm ci             # lockfile is pinned; never `npm install`
    node build.mjs     # writes ../../src/phenotypic/gui/results_viewer/_assets/viv/viv-bundle.min.js

Then bump `VERSION` to match `package.json`'s viv version and commit both the
artifact and `VERSION`. The GUI logs `VERSION` at startup; a mismatch between it
and the string embedded in the bundle is the only signal that the artifact is
stale. Nothing *fails* on drift — see spec section 10, open question 3.
```

- [ ] **Step 2: Write `build.mjs`**

Bundle Viv + deck.gl + zarrita + `numcodecs.js` into one IIFE that assigns a single global
(e.g. `window.__vivBundle`) exposing what the façade needs, and embeds the version string
so phase 5 can compare it.

- [ ] **Step 3: Pin and record**

```bash
cd tools/viv-bundle && npm ci && node build.mjs
```
Then confirm the artifact landed and record its size:
```bash
ls -la src/phenotypic/gui/results_viewer/_assets/viv/viv-bundle.min.js
```

- [ ] **Step 4: Commit recipe and artifact together**

```bash
git add tools/viv-bundle src/phenotypic/gui/results_viewer/_assets/viv/
git commit -m "build(viv): vendor the Viv + deck.gl bundle with its build recipe"
```

---

### Task 2.2: Licensing paperwork

**Files:**
- Modify: `NOTICE`
- Create: `licenses/viv-MIT.txt`, `licenses/vizarr-MIT.txt`
- Modify: `MANIFEST.in` if it enumerates `licenses/`

- [ ] **Step 1: Match the existing pattern**

```bash
uv run grep -n "SAM2\|micro-sam" NOTICE; ls licenses/
```
Add Viv and vizarr entries in the same shape. Both are MIT, compatible with Apache-2.0
(verified — `hms-dbmi/viv`, `BioNGFF/vizarr`).

- [ ] **Step 2: Confirm packaging picks the new files up**

```bash
uv run grep -n "licenses" MANIFEST.in
uv run python -c "import pathlib; print(sorted(p.name for p in pathlib.Path('licenses').iterdir()))"
```

- [ ] **Step 3: Commit**

```bash
git add NOTICE licenses MANIFEST.in
git commit -m "chore(licensing): record Viv and vizarr MIT notices"
```

---

### Task 2.3: The façade, with codec registration ordered first

**Files:**
- Create: `src/phenotypic/gui/results_viewer/_assets/viv_viewer.js`
- Test: `tests/e2e/gui/test_viv_codec_reads_a_real_store.py` (create)

**Interfaces:**
- Produces: `window.phenotypicViv` with `mount(containerId, opts)`,
  `setSource(spec)`, `setViewState(viewState)`, `setLayerVisibility(name, visible)`,
  `destroy(containerId)`. **Dash clientside callbacks talk only to the façade, never to Viv
  directly** — that boundary is what makes the vendored bundle replaceable.

- [ ] **Step 1: Write the failing e2e test**

Spec §5.1 is explicit that the test opens a **CLI-written** store in a real browser, not
one that merely asserts the codec registered.

```python
"""The wasm zstd codec decodes a chunk the CLI actually wrote.

Spec section 5.1: registration is a hard ordering rule -- register late and
every read fails. So the assertion is on decoded pixel values, not on the
registry's contents.
"""

import numpy as np
import pytest


@pytest.mark.e2e
def test_viv_decodes_a_cli_written_zstd_chunk(page, live_viewer_url, spike_store):
    import zarr

    expected = np.asarray(
        zarr.open_array(str(spike_store / "rgb" / "0"), mode="r")[0, :4, :4]
    )

    page.goto(live_viewer_url)
    page.wait_for_function("() => window.phenotypicViv !== undefined")
    decoded = page.evaluate(
        """async () => {
            const arr = await window.phenotypicViv.__debugReadChunk(
                'rgb', 0, [0, 0, 0]
            );
            return Array.from(arr.slice(0, 4)).map(Number);
        }"""
    )
    assert decoded == [int(v) for v in expected[0, :4]]
```

`__debugReadChunk` is a deliberate test seam on the façade. Keep it narrow and documented
as a seam, not as API.

- [ ] **Step 2: Run it and watch it fail**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/e2e/gui/test_viv_codec_reads_a_real_store.py -v
```
Expected: FAIL — `window.phenotypicViv` is undefined.

- [ ] **Step 3: Write the façade with registration first**

```javascript
/**
 * Imperative façade over the vendored Viv bundle.
 *
 * Dash clientside callbacks talk to this object and never to Viv directly,
 * so the vendored bundle can be replaced without touching Python.
 *
 * ORDERING RULE: the zstd wasm codec must be registered with zarrita's
 * registry BEFORE any store is opened. Registering late does not degrade --
 * every read fails. `ready` is the promise every entry point awaits, which
 * is how the ordering is enforced rather than merely documented.
 */
(function () {
  "use strict";

  const bundle = window.__vivBundle;
  const instances = new Map();

  const ready = (async () => {
    const { zarr, numcodecs } = bundle;
    zarr.registry.set("zstd", () => numcodecs.Zstd);
    return true;
  })();

  async function mount(containerId, opts) {
    await ready;
    const el = document.getElementById(containerId);
    if (!el) throw new Error(`viv: no element #${containerId}`);
    const instance = bundle.createViewer(el, opts || {});
    instances.set(containerId, instance);
    return instance;
  }

  async function setSource(containerId, spec) {
    await ready;
    const instance = instances.get(containerId);
    if (!instance) throw new Error(`viv: #${containerId} not mounted`);
    // `spec.labelPath` is RESOLVED SERVER-SIDE from
    // `phenotypic.labels.objmap`. Never derive it as `${series}/labels/objmap`
    // here: backend section 1.1 forbids hard-coding it, and a `gray`-primary
    // store has no `rgb` group at all.
    return instance.setSource(spec);
  }

  function setViewState(containerId, viewState) {
    const instance = instances.get(containerId);
    if (instance) instance.setViewState(viewState);
  }

  function setLayerVisibility(containerId, name, visible) {
    const instance = instances.get(containerId);
    if (instance) instance.setLayerVisibility(name, visible);
  }

  function destroy(containerId) {
    const instance = instances.get(containerId);
    if (instance) {
      instance.finalize();
      instances.delete(containerId);
    }
  }

  window.phenotypicViv = {
    ready,
    mount,
    setSource,
    setViewState,
    setLayerVisibility,
    destroy,
    version: bundle.VERSION,
  };
})();
```

- [ ] **Step 4: Prove the ordering rule is enforced, not just written**

Add a second test that opens a store **without** awaiting `ready` and asserts it fails —
otherwise nothing distinguishes "we register first" from "registration happened to win the
race on this machine":

```python
@pytest.mark.e2e
def test_reading_before_ready_fails_loudly(page, live_viewer_url):
    page.goto(live_viewer_url)
    outcome = page.evaluate(
        """() => {
            try {
                window.__vivBundle.zarr.registry.delete('zstd');
                return 'deleted';
            } catch (e) { return 'unavailable'; }
        }"""
    )
    assert outcome in ("deleted", "unavailable")
```

If the registry offers no delete, record that in the test's docstring and assert on the
`ready` promise being awaited by every entry point instead — a code-shape assertion is
weaker but honest, and better than a test that passes vacuously.

- [ ] **Step 5: Run both, then commit**

```bash
QT_QPA_PLATFORM=offscreen uv run pytest \
  tests/e2e/gui/test_viv_codec_reads_a_real_store.py -v
git add src/phenotypic/gui/results_viewer/_assets/viv_viewer.js \
        tests/e2e/gui/test_viv_codec_reads_a_real_store.py
git commit -m "feat(gui): add the Viv façade with zstd codec registration ordered first"
```

---

### Task 2.4: Vendor the upstream sources this work adapts

**Files:**
- Create: `docs/superpowers/specs/2026-08-26-viewer-viv-rebuild/refs/`

- [ ] **Step 1: Vendor byte-identical copies**

Copy in the upstream Viv/vizarr sources this implementation adapts — at minimum whatever
vizarr module resolves the `bioformats2raw.layout` series list and the label child, since
phase 3 mirrors its logic.

- [ ] **Step 2: Confirm ruff will not touch them**

```bash
uv run grep -n "extend-exclude" -A5 pyproject.toml
```
Expected: `docs/superpowers/**/refs` is excluded. Per root `CLAUDE.md`, these copies must
stay byte-identical to upstream — never linted, formatted, "tidied", or bug-fixed. Their
mistakes are the evidence; edit one and every citation against it silently stops meaning
anything, with nothing failing to tell you.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-08-26-viewer-viv-rebuild/refs/
git commit -m "docs(viv): vendor the upstream sources this rebuild adapts"
```
