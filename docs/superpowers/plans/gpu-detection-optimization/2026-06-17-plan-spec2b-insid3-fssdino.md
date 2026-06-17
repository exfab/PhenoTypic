# Spec 2b — INSID3 + FSSDINO Detectors — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add the two **semantic, few-shot/in-context** GPU detectors from Spec 2 §4.3–§4.4 on the existing `GpuDetector` interface — `Insid3Detector` (one-shot in-context, reference image+mask) and `FssDinoDetector` (few-shot, support set) — plus the gated **DINOv3** checkpoint manager (deferred from Spec 2a), shared frozen-DINO prototype-matching helpers, semantic tiling, and licensing. Both emit `objmask` (`output_kind="semantic"`) and reuse the repo's downstream watershed (Spec 1 §8).

**Architecture:** Both subclass `GpuDetector`, `supports_batching=False`, `output_kind="semantic"`, `input_layer="rgb"`. They share a **frozen-DINO backbone + prototype-matching core** (`_dino_support.py`): extract DINO patch features → pool an exemplar prototype from the reference/support mask(s) → cosine-match query patches → upsample → threshold to a boolean `objmask`. `Insid3Detector` = 1 reference (one-shot); `FssDinoDetector` = a support set → k clustered prototypes + Gram-matrix refinement (clean-room from arXiv:2602.07550). Large plates **tile** (semantic tiling = union the per-tile masks). **No Spec 1 engine changes.**

**Tech Stack:** Python, pydantic v2, numpy, scikit-image, `transformers>=5.2.0` (DINOv2/DINOv3 `AutoModel`/`DINOv3ViTModel`, lazy), `huggingface_hub` (gated DINOv3), pytest. `uv` runner.

**Source of truth:** Spec 2 §4.3, §4.4, §6, §7, decisions O1/O3/O5/O6 + the **verified externals** (below). **Depends on Spec 2a** (the `foundation`/`gpu` extras, `require_license_acceptance`, `Sam3CheckpointManager` pattern, the `DinoVersion`/`DinoSize` aliases, the `_plan_tiles` tiling, the `dino_version=3` stub in `DinoSam2Detector` that this plan **completes**).

**Verified externals (research 2026-06-17):**
- **DINOv3** loads via `transformers.DINOv3ViTModel` (present in 5.12.1) or `AutoModel.from_pretrained("facebook/dinov3-vit{s|b|l}16-pretrain-lvd1689m")` + `AutoImageProcessor`. Gated repo, "DINOv3 License".
- **INSID3** (`github.com/visinf/INSID3`, **Apache-2.0**): training-free in-context segmentation on frozen DINOv3. API `build_insid3()` → `set_reference(ref_img, ref_mask)` → `set_target(img)` → `segment()`. Not pip-installable (research codebase).
- **FSSDINO** (paper arXiv:2602.07550, **CC BY-NC-SA**; code repo `hussni0997/fssdino` is **all-rights-reserved**): class-specific prototypes + Gram-matrix refinement on a frozen DINO backbone, few-shot, ~150 lines. **Clean-room from the paper only; do NOT vendor the repo.**

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `licenses/dinov3-License.txt`, `licenses/insid3-Apache-2.0.txt` | verbatim upstream licenses | Create |
| `NOTICE` | DINOv3 (gated) + INSID3 (Apache) + FSSDINO (paper CC BY-NC-SA, clean-room) entries | Modify |
| `src/phenotypic/detect/nn/_checkpoint_manager.py` | `Dinov3CheckpointManager` (gated, mirrors `Sam3CheckpointManager`) | Modify |
| `src/phenotypic/detect/nn/_cli.py` | `download --model-type dinov3 --dino-size …` | Modify |
| `src/phenotypic/detect/nn/_dino_support.py` | shared frozen-DINO backbone load + patch features + prototype pooling + cosine match | Create |
| `src/phenotypic/detect/nn/_tiling.py` | extract `_Tile`/`_plan_tiles` from `_sam3_detector.py` (shared) + `stitch_semantic_tiles` | Create |
| `src/phenotypic/detect/nn/_sam3_detector.py` | import tiling from `_tiling.py` (no behavior change) | Modify |
| `src/phenotypic/detect/nn/_dinosam2_detector.py` | wire `dino_version=3` to the real `Dinov3CheckpointManager` (remove the 2a `NotImplementedError` stub) | Modify |
| `src/phenotypic/detect/nn/_insid3_detector.py` | `Insid3Detector` | Create |
| `src/phenotypic/detect/nn/_fssdino_detector.py` | `FssDinoDetector` | Create |
| `src/phenotypic/detect/nn/__init__.py` | export both detectors | Modify |
| `tests/unit/detect/nn/test_insid3_detector.py`, `test_fssdino_detector.py` | construction/serialization/semantic-route/exemplar-validation/algorithm-math | Create |
| `tests/unit/detect/nn/test_dino_support.py` | the shared prototype-match core on synthetic features | Create |

**Convention note:** detectors construct **without** `transformers`/`torch`/weights (lazy load); only `.apply()` functional tests require `phenotypic[foundation]` + accepted gated DINOv3 + exemplar files, and **skip** otherwise (mirror `Sam3Detector`'s `FOUNDATION_AVAILABLE` + cache-probe skip). Every new `int`/`float` field carries a `TuneSpec`. `output_kind="semantic"` → the detector writes `image.objmask[:] = bool_mask` (Spec 1 §8 auto-labels into `objmap`); a shared assertion proves it round-trips through the staged HDF like a threshold detector.

---

### Task 1: Licensing + `Dinov3CheckpointManager` + CLI (+ complete the DinoSam2 v3 path)

**Files:** Create `licenses/dinov3-License.txt`, `licenses/insid3-Apache-2.0.txt`; Modify `NOTICE`, `_checkpoint_manager.py`, `_cli.py`, `_dinosam2_detector.py`; Test: extend `tests/unit/test_licensing_scaffolding.py`, `tests/unit/detect/nn/test_gated_checkpoint_managers.py`, `tests/unit/detect/nn/test_dinosam2_detector.py`.

- [ ] **Step 1: Failing tests** — licenses present + NOTICE names; a gated `Dinov3CheckpointManager` mirroring the SAM3 manager (acceptance gate → `snapshot_download`; 401/403 actionable); and `DinoSam2Detector(dino_version=3)._ensure_model_loaded()` no longer raises `NotImplementedError` (it now routes through `Dinov3CheckpointManager` — mock `snapshot_download` + the DINO load):

```python
def test_dinov3_license_and_insid3_present():
    assert (REPO / "licenses" / "dinov3-License.txt").is_file()
    assert (REPO / "licenses" / "insid3-Apache-2.0.txt").is_file()

def test_notice_names_dinov3_insid3_fssdino():
    n = (REPO / "NOTICE").read_text(encoding="utf-8")
    assert "DINOv3" in n and "INSID3" in n and "FSSDINO" in n

def test_dinov3_manager_requires_acceptance(monkeypatch):
    from phenotypic.detect.nn._checkpoint_manager import Dinov3CheckpointManager
    monkeypatch.delenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", raising=False)
    import pytest
    with pytest.raises(RuntimeError, match="license"):
        Dinov3CheckpointManager(size="base").download(interactive=False)
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.** `dinov3-License.txt`: **fetch the verbatim DINOv3 License** from `https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m` (the LICENSE blob; do NOT fabricate). `insid3-Apache-2.0.txt` = canonical Apache-2.0. `Dinov3CheckpointManager(size)` mirrors `Sam3CheckpointManager` (gated: `require_license_acceptance("dinov3", "DINOv3 License", url)` → `snapshot_download(repo_id=f"facebook/dinov3-vit{s|b|l}16-pretrain-lvd1689m")`; map `size→{s,b,l}`). NOTICE: add DINOv3 (gated), INSID3 (Apache-2.0, training-free in-context — clean-room/attribution), FSSDINO (paper arXiv:2602.07550 CC BY-NC-SA; **no code vendored, clean-room from the paper**). In `_dinosam2_detector.py`, replace the `dino_version=3 → NotImplementedError` stub with a real `Dinov3CheckpointManager` route (load DINOv3 via `AutoModel`). Extend `_cli.py` `--model-type` `click.Choice` with `dinov3`.
- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** `feat(foundation): gated Dinov3CheckpointManager + INSID3/DINOv3 licenses; complete DinoSam2 v3 path`.

---

### Task 2: Shared frozen-DINO prototype-match core (`_dino_support.py`)

**Files:** Create `src/phenotypic/detect/nn/_dino_support.py`; Test `tests/unit/detect/nn/test_dino_support.py`.

**Why:** INSID3 and FSSDINO share the same core — both pool an exemplar prototype from a mask and cosine-match it to query patch features. Centralize it (clean-room, generic), so the two detectors are thin specializations.

- [ ] **Step 1: Failing tests** (pure math on **synthetic** features/masks — no model):

```python
import numpy as np
from phenotypic.detect.nn._dino_support import pool_prototype, cosine_match_to_mask


def test_pool_prototype_is_masked_mean():
    feats = np.zeros((4, 4, 8), np.float32); feats[1:3, 1:3] = 1.0
    mask = np.zeros((4, 4), bool); mask[1:3, 1:3] = True
    proto = pool_prototype(feats, mask)
    assert np.allclose(proto, np.ones(8))           # mean over the masked patches


def test_cosine_match_recovers_prototype_region():
    feats = np.zeros((4, 4, 8), np.float32); feats[1:3, 1:3] = 1.0
    proto = np.ones(8, np.float32)
    out = cosine_match_to_mask(feats, proto, thresh=0.9, out_shape=(8, 8))
    assert out.dtype == bool and out.shape == (8, 8)
    assert out.any() and not out.all()              # foreground region, not everything
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.** `load_dino_backbone(dino_version, dino_size, device)` → `(model, processor)` via `AutoModel.from_pretrained(hf_id)` + `AutoImageProcessor` (lazy `transformers`; `dino_version=3` routes the gated download through `Dinov3CheckpointManager`; reuse the `_hf_dino_id` mapping from `DinoSam2Detector` — extract it here if cleaner). `extract_patch_features(model, processor, rgb_uint8) -> (Hp, Wp, D)` (run the ViT, drop CLS, reshape patch tokens to the grid). `pool_prototype(features, mask) -> (D,)` (resize `mask` to the patch grid, masked mean). `cosine_match_to_mask(features, prototype, thresh, out_shape) -> bool (out_shape)` (cosine sim per patch → reshape → `skimage.transform.resize` to `out_shape` → `> thresh`). Handle the degenerate zero-prototype/empty-mask case by returning an all-False mask (fail safe).
- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** `feat(detect): shared frozen-DINO prototype-match core (_dino_support)`.

---

### Task 3: Shared tiling (`_tiling.py`) + semantic stitch

**Files:** Create `src/phenotypic/detect/nn/_tiling.py`; Modify `_sam3_detector.py`; Test `tests/unit/detect/nn/test_tiling.py` (move/extend the SAM3 tiling tests).

- [ ] **Step 1:** Move `_Tile` + `_tile_starts` + `_plan_tiles` from `_sam3_detector.py` into `_tiling.py` (no behavior change — `_sam3_detector.py` imports them). Add `stitch_semantic_tiles(tiles, tile_masks, out_shape) -> bool` — OR the per-tile boolean masks back into a full-image mask (union; overlaps just OR, no NMS needed for semantic). Re-run the existing SAM3 tiling tests against the new module path (they must stay green) + add a stitch test (two overlapping tiles union correctly).
- [ ] **Step 2–4:** Run the SAM3 suite (unchanged behavior) + the new stitch test → green.
- [ ] **Step 5: Commit** `refactor(detect): extract shared tiling to _tiling.py + semantic stitch`.

---

### Task 4: `Insid3Detector` — one-shot in-context, semantic

**Files:** Create `src/phenotypic/detect/nn/_insid3_detector.py`; Modify `detect/nn/__init__.py`; Test `tests/unit/detect/nn/test_insid3_detector.py`.

- [ ] **Step 1: Failing tests** (construction / capability / semantic route / exemplar validation — no weights):

```python
from phenotypic.detect.nn import Insid3Detector
from phenotypic.data import load_synth_yeast_plate


class TestInsid3Construction:
    def test_capabilities(self):
        det = Insid3Detector()
        assert det.input_layer == "rgb"
        assert det.output_kind == "semantic"
        assert det.supports_batching is False

    def test_serialization_round_trip(self, tmp_path):
        det = Insid3Detector(reference_image=tmp_path / "r.tiff",
                             reference_mask=tmp_path / "m.png", similarity_thresh=0.7)
        r = Insid3Detector.from_json(det.to_json())
        assert r.similarity_thresh == 0.7

    def test_missing_reference_raises_on_load(self):
        import pytest
        with pytest.raises(ValueError, match="reference"):
            Insid3Detector()._ensure_model_loaded()   # no reference set
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.** Fields: `reference_image: Optional[Path] = None`, `reference_mask: Optional[Path] = None`, `dino_version: DinoVersion = 3` (INSID3 is DINOv3-native; gated), `dino_size: DinoSize = "base"`, `similarity_thresh: Annotated[float, TuneSpec(0.0, 1.0)] = 0.5`, tile fields (reuse `_tiling`), `device`. Capabilities: rgb / **semantic** / `supports_batching=False`. `_ensure_model_loaded()`: validate `reference_image`/`reference_mask` are set (else `ValueError`), load the DINO backbone (`_dino_support.load_dino_backbone`), extract the reference's features + `pool_prototype` over the reference mask, cache the prototype. `_infer_one(sample)`: `cosine_match_to_mask` of the query features to the cached prototype → boolean `objmask` (tile via `_tiling`/`stitch_semantic_tiles` for large plates). Returns a **bool** mask (the engine writes `objmask`). Re-point autodoc.
- [ ] **Step 4: Run → pass** (functional `.apply()` skips without `foundation` + gated DINOv3 + exemplars).
- [ ] **Step 5: Commit** `feat(detect): Insid3Detector — one-shot in-context semantic detector`.

---

### Task 5: `FssDinoDetector` — few-shot, semantic (clean-room)

**Files:** Create `src/phenotypic/detect/nn/_fssdino_detector.py`; Modify `detect/nn/__init__.py`; Test `tests/unit/detect/nn/test_fssdino_detector.py`.

**Recipe (clean-room from arXiv:2602.07550 — do NOT vendor `hussni0997/fssdino`):** load a frozen DINO backbone; for each `(support_image, support_mask)` extract masked patch features; **cluster** the pooled foreground features into `n_clusters` class-specific prototypes; at inference, score each query patch by max cosine similarity to any prototype, then apply **Gram-matrix refinement** (style/co-occurrence consistency from the paper); threshold → `objmask`.

- [ ] **Step 1: Failing tests** (construction / capability / serialization / support-set validation / a synthetic prototype-clustering+match unit test):

```python
from phenotypic.detect.nn import FssDinoDetector


class TestFssDinoConstruction:
    def test_capabilities(self):
        det = FssDinoDetector()
        assert det.output_kind == "semantic" and det.supports_batching is False

    def test_dino_version_defaults_to_2(self):
        assert FssDinoDetector().dino_version == 2   # DINOv2 default, ungated (Spec §4.4)

    def test_n_clusters_field(self):
        assert FssDinoDetector(n_clusters=8).n_clusters == 8

    def test_missing_support_raises_on_load(self):
        import pytest
        with pytest.raises(ValueError, match="support"):
            FssDinoDetector()._ensure_model_loaded()
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.** Fields: `support_images: List[Path] = []`, `support_masks: List[Path] = []`, `n_clusters: Annotated[int, TuneSpec(1, 20)] = 5`, `dino_version: DinoVersion = 2` (DINOv2 default — gate-free; DINOv3 opt-in), `dino_size: DinoSize = "base"`, `similarity_thresh: Annotated[float, TuneSpec(0.0, 1.0)] = 0.5`, `gram_weight: Annotated[float, TuneSpec(0.0, 1.0)] = 0.5`, tile fields, `device`. Capabilities: rgb / **semantic** / `supports_batching=False`. `_ensure_model_loaded()`: validate the support set (non-empty, equal lengths), load the backbone, build the `n_clusters` prototypes (k-means over pooled support foreground features) + the Gram statistics, cache. `_infer_one(sample)`: prototype-cosine + `gram_weight`-blended Gram refinement → threshold → boolean `objmask` (tiled). Add a synthetic-feature unit test for the prototype-clustering + match + threshold (like Task 2's, with k>1).
- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit** `feat(detect): FssDinoDetector — few-shot semantic detector (clean-room)`.

---

### Task 6: Exports + semantic-route round-trip + annotation gate + regression

- [ ] Export `Insid3Detector`, `FssDinoDetector` from `phenotypic.detect.nn` (+ `__all__`); pin tune-readiness in `test_foundation_detectors_exports.py`.
- [ ] **Semantic round-trip test:** a `FakeSemanticDetector`-style assertion is unnecessary — instead, a construction-level test that both set `output_kind="semantic"`, plus a functional-gated test (skips) that `.apply()` writes only `objmask` and `objmap[:] > 0 == objmask[:]` (Spec 1 §8 shared invariant).
- [ ] Annotation-coverage gate green (`n_clusters`, `similarity_thresh`, `gram_weight`, tile fields all carry `TuneSpec`).
- [ ] Targeted regression (sequential, no `-n`): `uv run pytest tests/unit/detect/nn -q` + `uv run mypy --follow-imports=silent` on the 4 new files + `uv run ruff check src/phenotypic/detect/nn tests/unit/detect/nn`. Reconcile any migration golden.
- [ ] **Commit** `test(detect): green regression for Insid3 + FssDino detectors`.

---

### Task 7: Docs — how-to + detector docstrings

- [ ] Extend `docs/source/how_to/pages/gpu_detection_setup.md`: the two semantic detectors, the **gated DINOv3** handshake (`uv run hf auth login`, `PHENOTYPIC_ACCEPT_MODEL_LICENSE=dinov3`), the **exemplar interface** (how to supply `reference_image`/`reference_mask` for INSID3, `support_images`/`support_masks` for FSSDINO), the per-model license table (INSID3 Apache, FSSDINO paper CC BY-NC-SA clean-room, DINOv3 gated), and the semantic→watershed downstream note. Each docstring documents `output_kind="semantic"`, the exemplar requirement, and the gated/ungated backbone. Add a content-check test.
- [ ] **Commit** `docs(foundation): INSID3 + FSSDINO install, gated DINOv3, exemplar interface`.

---

## Self-Review

**Spec coverage (Spec 2b):** gated DINOv3 manager (§7) → Task 1; INSID3 (§4.3) → Task 4; FSSDINO clean-room (§4.4) → Task 5; semantic→objmask route (§4, Spec 1 §8) → Tasks 4–6; licensing (§6) → Task 1; docs (§8) → Task 7. ✓
**Verified externals:** DINOv3 `AutoModel`/`DINOv3ViTModel`; INSID3 Apache + API; FSSDINO paper + CC BY-NC-SA. ✓
**Shared core:** `_dino_support.py` (prototype match) + `_tiling.py` (extracted from 2a) keep both detectors thin and DRY; the 2a `dino_version=3` stub is completed here.
**Out of scope:** none remaining in Spec 2 after this — Spec 2 (all four models) is complete once 2b lands.

## Plan-Review Resolutions (apply during implementation — supersedes the tasks where they conflict)

Plan-reviewer pass (2026-06-17) verified the DINOv3 load API, INSID3 Apache-2.0, FSSDINO
licensing, the semantic route, and the tiling extraction as **correct**, and found 1 real
bug + 3 fidelity gaps. The user resolved the design OQs. Apply all of the below.

**User decisions:**
- **D-faithful — FAITHFUL reproductions.** Implement the papers' actual methods (NOT naive
  prototype matching) and KEEP the `Insid3Detector` / `FssDinoDetector` names:
  - **INSID3 (C2):** implement the **positional-bias removal** that is INSID3's defining step.
    Read `github.com/visinf/INSID3` (Apache-2.0) — either **vendor its small debias module with
    attribution** OR clean-room the SVD/PCA-projection (find the positional component of DINO
    features on low-semantic inputs, project patch features onto its orthogonal complement)
    **before** prototype matching. Faithful, not naive cosine matching.
  - **FSSDINO (C3):** read the paper **arXiv:2602.07550 §3–4** (WebFetch the PDF / HTML) for the
    actual prototype + **Gram-matrix refinement** math and the **layer-selection** finding.
    Expose a `feature_layer: int` field (use `output_hidden_states=True` → `hidden_states[layer]`)
    and default it to the paper's recommended intermediate layer (not the last layer). Use the
    paper's Gram formulation — do not invent a `gram_weight` blend unless the paper supports it
    (document any deviation).
- **D-curated-example — SHIP a curated colony exemplar** under `src/phenotypic/_assets/exemplars/`
  (a reference colony RGB + its ground-truth mask, rendered once from `load_synth_yeast_plate()`).
  Use it as the **default** `reference_image`/`reference_mask` (INSID3) and default `support_*`
  (FSSDINO) so the detectors have a working default, and add it to `[tool.setuptools.package-data]`.
- **D-dino-version-for-testing — both detectors take `dino_version: DinoVersion`** so a **gate-free
  DINOv2** functional test can actually run in CI (DINOv2 + transformers are installed; no gate).
  INSID3 defaults to `dino_version=3` (its native backbone; the debias targets DINOv3's bias, and
  is a near-no-op on DINOv2); FSSDINO defaults per the paper but supports v2. **Add ONE real
  functional test per detector** that loads **DINOv2** + the bundled exemplar and asserts `.apply()`
  writes a non-empty `objmask` (skips only if `foundation` is absent — NOT gated, so it runs here).

**Blocker fix:**
- **C1 — DINOv3 register-token BUG (real, also in shipped 2a code).** DINOv3's `last_hidden_state`
  is `(B, 1 + num_register_tokens + Hp·Wp, D)` with **`config.num_register_tokens = 4`**. The
  planned `extract_patch_features` AND the existing `_dinosam2_detector._dino_dense_features`
  (lines ~365–394) slice `[:, 1:, :]` (drop CLS only) → 4 register tokens contaminate the patch
  grid for DINOv3 (the square-grid fallback silently mis-shapes it). **Fix:** slice
  `[:, 1 + model.config.num_register_tokens:, :]` (default 0 for v2), then reshape to `(Hp, Wp, D)`.
  Put the single correct implementation in `_dino_support.extract_patch_features` and **refactor
  `DinoSam2Detector` to call it** (delete the buggy private copy — don't leave the bug in two
  places). Add a synthetic test asserting an exact grid reshape for `num_register_tokens > 0`.

**Should-fixes:**
- **W1 — annotation readiness:** the coverage gate is scoped to `detect.__all__` (NOT `detect.nn`).
  Pin the new detectors' tune-readiness by **extending `tests/unit/detect/nn/test_foundation_detectors_exports.py`** (the dedicated test), not the coverage gate. (Every new numeric field still carries a `TuneSpec`: `n_clusters` `TuneSpec(1,20)`, `similarity_thresh`, `feature_layer` `TuneSpec(tunable=False)` or a bare `TuneSpec()`, tile fields.)
- **W2 — rewrite the v3-stub test:** `test_dinosam2_detector.py::test_dinov3_load_raises_not_implemented`
  must be **replaced** with a mocked-`snapshot_download` + mocked-`AutoModel` v3 success-path test
  (the stub is removed in Task 1).
- **W3 — `Dinov3CheckpointManager` = DINOv2-sized constructor + SAM3 gating:** `__init__(self, *, size)`
  + `_SIZE_TO_REPO` for the three `dinov3-vit{s|b|l}16-pretrain-lvd1689m` ids + `require_license_acceptance("dinov3", "DINOv3 License", url)`. (Not a literal "mirror SAM3" — it's a hybrid.)
- **W4 — exemplar mask ↔ patch-grid alignment:** resize the reference/support mask through the
  **same processed geometry** the image went through (`inputs.pixel_values.shape`), THEN downsample
  to the patch grid (`order=0`). Add a test with a **non-square** exemplar.
- **W5 (note):** real INSID3/FSSDINO accuracy still needs a manual gated-DINOv3 run; the bundled
  DINOv2 functional test validates plumbing + the v2 path, not paper-level accuracy.

## Execution Handoff
Plan + resolutions complete. Per the requested flow (same as 2a): implement with a **single Opus
agent** (applying this resolutions section — it must WebFetch the INSID3 method + the FSSDINO paper
§3–4 to implement faithfully), then a **code-review subagent**.
