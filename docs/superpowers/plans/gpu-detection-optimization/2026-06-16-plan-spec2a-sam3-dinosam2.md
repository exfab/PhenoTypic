# Spec 2a — SAM3 + DinoSam2 Detectors — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:
> subagent-driven-development (recommended) or superpowers:executing-plans. Steps use
> checkbox (`- [ ]`) syntax.

**Goal:** Add two instance-native GPU detectors on Spec 1's `GpuDetector` interface —
`Sam3` (text-prompted, true-batch, gated weights) and `DinoSam2Detector` (
DINOv2-default, training-free SAM2+DINO recipe) — plus the `foundation`/`gpu` packaging
extras, the gated-weights download flow, and the licensing files.

**Architecture:** Both subclass `GpuDetector`, mirror `Sam2` (
`src/phenotypic/detect/nn/_sam2_detector.py`), lazy-load in `_ensure_model_loaded()`,
and write a labeled `objmap` (`output_kind="instance"`). `Sam3` sets
`supports_batching=True` and overrides `infer_batch` with a true `(N,C,H,W)` forward via
the `transformers` SAM3 API (verified v5.2.0); `DinoSam2Detector` keeps
`supports_batching=False` and implements `_infer_one` (SAM2 AMG proposals → DINO feature
scoring/merge). **No Spec 1 engine changes** — these are pure interface implementations.

**Tech Stack:** Python, pydantic v2, numpy, scikit-image, `transformers>=5.2.0` (SAM3 +
DINO, lazy), `huggingface_hub` (gated download), `sam2` (existing), pytest. `uv` runner.

**Source of truth:**
`docs/superpowers/specs/gpu-detection-optimization/2026-06-16-gpu-detector-models-design.md`
§3–§9 (decisions O1, O2, O4) + the **verified** SAM3 transformers API (O7) and pin
`transformers>=5.2.0` (O3).

**Depends on:** Spec 1 (the `GpuDetector` interface + `_ensure_model_loaded`, the staged
engine, the licensing scaffolding `NOTICE`/`licenses/`/`require_license_acceptance`/
`PHENOTYPIC_ACCEPT_MODEL_LICENSE`). **Spec 2b (INSID3 + FSSDINO) is a separate follow-on
plan** (few-shot/semantic, clean-room, gated DINOv3).

**Verified SAM3 API (transformers v5.2.0, `model_doc/sam3`):**

```python
from transformers import Sam3Processor, Sam3Model
model = Sam3Model.from_pretrained("facebook/sam3")          # gated repo
processor = Sam3Processor.from_pretrained("facebook/sam3")
inputs = processor(images=[img1, img2], text=["colony", "colony"], return_tensors="pt")
outputs = model(**inputs)   # outputs.pred_masks (B, num_queries<=200, H, W), pred_boxes, pred_logits
results = processor.post_process_instance_segmentation(
    outputs, threshold=score_thresh, mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist())   # list[dict{masks, boxes, scores}], one per image
```

---

## File Structure

| File                                                                  | Responsibility                                                                                      | Action |
|-----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|--------|
| `pyproject.toml`                                                      | `foundation` + `gpu` extras; mypy `ignore_missing_imports` for `transformers.*`/`huggingface_hub.*` | Modify |
| `src/phenotypic/tools_/typing_.py`                                    | `DinoVersion = Literal[2, 3]`, `DinoSize = Literal["small","base","large"]` aliases                 | Modify |
| `licenses/sam3-SAM-License.txt`, `licenses/dinov2-Apache-2.0.txt`     | verbatim upstream licenses                                                                          | Create |
| `NOTICE`                                                              | add SAM3 + DINOv2 entries                                                                           | Modify |
| `src/phenotypic/detect/nn/_checkpoint_manager.py`                     | `Sam3CheckpointManager`, `Dinov2CheckpointManager` (gated/ungated HF pulls)                         | Modify |
| `src/phenotypic/detect/nn/__main__.py` (or the existing download CLI) | `download --model sam3/dinov2`, `list`, `clear` for the new managers                                | Modify |
| `src/phenotypic/detect/nn/_sam3_detector.py`                          | `Sam3` (batch + grid-aware tiling)                                                                  | Create |
| `src/phenotypic/detect/nn/_dinosam2_detector.py`                      | `DinoSam2Detector` (SAM2 AMG + DINO scoring)                                                        | Create |
| `src/phenotypic/detect/nn/__init__.py`                                | export `Sam3`, `DinoSam2Detector`                                                                   | Modify |
| `tests/unit/detect/nn/test_sam3_detector.py`                          | construction/serialization/capability/prompt/tiling-math                                            | Create |
| `tests/unit/detect/nn/test_dinosam2_detector.py`                      | construction/serialization/capability/dino_version routing                                          | Create |
| `tests/unit/detect/nn/test_gated_checkpoint_managers.py`              | acceptance gate, 401/403 message, offline                                                           | Create |

**Convention note:** detectors construct **without** `transformers`/`torch` (lazy import
inside `_ensure_model_loaded`/`infer_batch`/`_infer_one`); only `.apply()` functional
tests require `phenotypic[foundation]` + accepted gated weights and are **skipped**
otherwise (mirror `Sam2`'s `TestSam2DetectorFunctional` skip). New numeric fields on a
`detect/` op are pulled into the annotation-coverage gate (
`tests/unit/tune/test_annotation_coverage.py`) — every `int`/`float` field needs a
`TuneSpec` or `TuneSpec(tunable=False)`.

---

### Task 1: Packaging extras + DINO typing aliases

**Files:** Modify `pyproject.toml`, `src/phenotypic/tools_/typing_.py`; Test
`tests/unit/test_foundation_extra.py` (Create),
`tests/unit/tools_/test_typing_aliases.py` (extend or create).

- [ ] **Step 1: Failing test** — `tests/unit/test_foundation_extra.py`:

```python
import tomllib
from pathlib import Path
from typing import get_args

from phenotypic.tools_.typing_ import DinoVersion, DinoSize

REPO = Path(__file__).resolve().parents[2]


def test_foundation_and_gpu_extras_declared():
    data = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    assert "foundation" in extras and "gpu" in extras
    joined = " ".join(extras["foundation"])
    assert "transformers" in joined and "huggingface_hub" in joined
    # foundation pulls torch; gpu is the umbrella
    assert any("phenotypic[torch" in d for d in extras["foundation"])
    assert any("foundation" in d for d in extras["gpu"])


def test_dino_typing_aliases():
    assert set(get_args(DinoVersion)) == {2, 3}
    assert set(get_args(DinoSize)) == {"small", "base", "large"}
```

- [ ] **Step 2: Run → fail** (`uv run pytest tests/unit/test_foundation_extra.py -v`).

- [ ] **Step 3: Implement.** In `pyproject.toml` `[project.optional-dependencies]` add (
  keep the existing `torch` extra unchanged):

```toml
foundation = [
    "phenotypic[torch]",
    "transformers>=5.2.0",   # first release shipping Sam3Model + DINO AutoModel (O3, verified)
    "huggingface_hub>=0.26",
]
gpu = ["phenotypic[torch,foundation]"]
```

In the mypy config (`[tool.mypy]` overrides, beside `sam2.*`/`micro_sam.*`) add
`"transformers.*"` and `"huggingface_hub.*"` to `ignore_missing_imports`. In
`tools_/typing_.py`, next to the other `Literal` aliases:

```python
#: DINO backbone generation for DinoSam2Detector. 2 = DINOv2 (Apache, ungated,
#: default); 3 = DINOv3 (gated, opt-in — routes through require_license_acceptance).
DinoVersion = Literal[2, 3]
#: DINO backbone size.
DinoSize = Literal["small", "base", "large"]
```

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit**
  `feat(foundation): add foundation/gpu extras + DINO typing aliases`.

---

### Task 2: Licensing files (SAM3, DINOv2) + NOTICE

**Files:** Create `licenses/sam3-SAM-License.txt`, `licenses/dinov2-Apache-2.0.txt`;
Modify `NOTICE`; Test: extend `tests/unit/test_licensing_scaffolding.py`.

- [ ] **Step 1: Failing test** — append:

```python
def test_sam3_and_dinov2_licenses_present():
    assert (REPO / "licenses" / "sam3-SAM-License.txt").is_file()
    assert (REPO / "licenses" / "dinov2-Apache-2.0.txt").is_file()


def test_notice_names_sam3_and_dinov2():
    notice = (REPO / "NOTICE").read_text(encoding="utf-8")
    assert "SAM3" in notice and "DINOv2" in notice
    assert "does not redistribute" in notice.lower()
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.** `dinov2-Apache-2.0.txt` = the canonical Apache-2.0 text (
  copy the repo `LICENSE`, as Spec 1 did for SAM2). **`sam3-SAM-License.txt`** = the
  verbatim Meta SAM License — **the implementing agent must fetch the exact text
  from `https://huggingface.co/facebook/sam3` (LICENSE / license file) at build time**;
  do NOT fabricate license text. Add to `NOTICE` (after the SAM2/micro-sam entries):

```
- SAM3 (segment-anything-3), Meta — code via `transformers` (Apache-2.0);
  weights under the SAM License (commercial-OK, GATED on Hugging Face).
  Upstream:  https://huggingface.co/facebook/sam3
  License:   licenses/sam3-SAM-License.txt
- DINOv2, Meta — backbone for DinoSam2Detector; Apache-2.0, ungated.
  Upstream:  https://github.com/facebookresearch/dinov2
  License:   licenses/dinov2-Apache-2.0.txt
```

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit**
  `chore(licensing): SAM3 + DINOv2 license files + NOTICE entries`.

---

### Task 3: Gated checkpoint managers + download CLI

**Files:** Modify `src/phenotypic/detect/nn/_checkpoint_manager.py`, the `detect/nn`
download CLI; Test `tests/unit/detect/nn/test_gated_checkpoint_managers.py` (Create).

**Source of truth:** Spec 2 §7. Reuse `require_license_acceptance` (Spec 1, already in
`_checkpoint_manager.py`).

- [ ] **Step 1: Failing tests** — manager behavior, mocking `huggingface_hub`:

```python
import pytest


def test_sam3_manager_requires_acceptance(monkeypatch):
    from phenotypic.detect.nn._checkpoint_manager import Sam3CheckpointManager
    monkeypatch.delenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", raising=False)
    mgr = Sam3CheckpointManager()
    with pytest.raises(RuntimeError, match="license"):
        mgr.download(interactive=False)  # acceptance gate fires before any network


def test_sam3_manager_accepts_then_downloads(monkeypatch):
    from phenotypic.detect.nn import _checkpoint_manager as cm
    monkeypatch.setenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", "sam3")
    calls = {}
    monkeypatch.setattr(cm, "snapshot_download",
                        lambda **kw: calls.update(kw) or "/fake/cache/sam3")
    path = cm.Sam3CheckpointManager().download(interactive=False)
    assert calls["repo_id"] == "facebook/sam3"
    assert path == "/fake/cache/sam3"


def test_gated_download_401_is_actionable(monkeypatch):
    from phenotypic.detect.nn import _checkpoint_manager as cm
    monkeypatch.setenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", "sam3")

    def _raise(**kw):
        raise cm._GatedRepoError("403")  # or hf's GatedRepoError

    monkeypatch.setattr(cm, "snapshot_download", _raise)
    with pytest.raises(RuntimeError, match="Request access"):
        cm.Sam3CheckpointManager().download(interactive=False)


def test_dinov2_manager_is_ungated(monkeypatch):
    # DINOv2 needs no acceptance / token
    from phenotypic.detect.nn import _checkpoint_manager as cm
    monkeypatch.delenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", raising=False)
    monkeypatch.setattr(cm, "snapshot_download", lambda **kw: "/fake/dinov2")
    assert cm.Dinov2CheckpointManager(size="base").download() == "/fake/dinov2"
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.** Lazy-import `huggingface_hub` inside methods (module stays
  importable without it — mirror the torch-lazy pattern). Add a module-level
  `snapshot_download` indirection so tests patch one name:

```python
def _hf_snapshot_download(**kwargs):
    from huggingface_hub import snapshot_download as _dl
    return _dl(**kwargs)

snapshot_download = _hf_snapshot_download  # patch point for tests
_GATED_REPO_IDS = {"sam3": "facebook/sam3"}


class Sam3CheckpointManager:
    repo_id = "facebook/sam3"
    license_name = "SAM License"
    license_url = "https://huggingface.co/facebook/sam3"

    def download(self, *, interactive: bool = True) -> str:
        require_license_acceptance("sam3", self.license_name, self.license_url,
                                   interactive=interactive)
        try:
            return snapshot_download(repo_id=self.repo_id)
        except Exception as e:
            if _is_gated_or_auth_error(e):
                raise RuntimeError(
                    f"Cannot download {self.repo_id}: access not granted or no token. "
                    f"Request access at {self.license_url}, then run `uv run hf auth login` "
                    f"(or export HF_TOKEN)."
                ) from e
            raise
```

`Dinov2CheckpointManager(size=...)` maps size → HF id (
`facebook/dinov2-{small|base|large}` → e.g. `facebook/dinov2-base`), no acceptance gate,
plain `snapshot_download`. `_is_gated_or_auth_error` matches
`huggingface_hub.errors.GatedRepoError`/`HfHubHTTPError` 401/403 (import lazily; fall
back to string match on "401"/"403"/"gated"). Extend the existing
`python -m phenotypic.detect.nn download/list/clear` to route `--model sam3` /
`--model dinov2` to these managers (`--accept-license` sets the env var for the call).

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit**
  `feat(foundation): gated SAM3 + ungated DINOv2 checkpoint managers + CLI`.

---

### Task 4: `Sam3` — fields, lazy load, true-batch `infer_batch`

**Files:** Create `src/phenotypic/detect/nn/_sam3_detector.py`; Modify
`detect/nn/__init__.py`; Test `tests/unit/detect/nn/test_sam3_detector.py`.

- [ ] **Step 1: Failing tests** (construction / capability / serialization / prompt — no
  weights):

```python
from phenotypic.detect.nn import Sam3
from phenotypic import ImagePipeline


class TestSam3Construction:
    def test_capability_fields(self):
        det = Sam3()
        assert det.input_layer == "rgb"
        assert det.output_kind == "instance"
        assert det.supports_batching is True

    def test_prompt_defaults_and_overrides(self):
        assert Sam3().prompt == "colony"
        assert Sam3(prompt="yeast colony").prompt == "yeast colony"

    def test_serialization_round_trip(self):
        det = Sam3(prompt="bacterial colony", score_thresh=0.4)
        round = Sam3.from_json(det.to_json())
        assert round.prompt == "bacterial colony" and round.score_thresh == 0.4

    def test_constructs_without_transformers(self):
        # lazy import: building the op must not import transformers
        Sam3()  # no raise
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.** Fields: `prompt: str = "colony"` (plain str, no TuneSpec —
  parameterised free text), `score_thresh: Annotated[float, TuneSpec(0.0, 1.0)] = 0.5`,
  `mask_threshold: Annotated[float, TuneSpec(0.0, 1.0)] = 0.5`,
  `min_mask_region_area: Annotated[int, TuneSpec(0, 500)] = 0` (post-filter), tiling
  fields from Task 5, `device: Device = "auto"`. Capabilities: `input_layer="rgb"`,
  `output_kind="instance"`, `supports_batching=True`. `_model`/`_processor` are
  `PrivateAttr`. `_ensure_model_loaded()` lazy-imports `transformers`, builds
  `Sam3Model.from_pretrained("facebook/sam3").to(device)` +
  `Sam3Processor.from_pretrained(...)` (route the gated download via
  `Sam3CheckpointManager` / honour `HF_HUB_OFFLINE`). Override `infer_batch` (true
  batch, **before** tiling — tiling wraps it in Task 5):

```python
def infer_batch(self, batch):
    import numpy as np, torch

    self._ensure_model_loaded()
    images = [self._to_uint8(s) for s in batch]  # each (H,W,3) uint8
    inputs = self._processor(images=images, text=[self.prompt] * len(images),
                             return_tensors="pt").to(self._device)
    with torch.no_grad():
        outputs = self._model(**inputs)
    results = self._processor.post_process_instance_segmentation(
            outputs, threshold=self.score_thresh, mask_threshold=self.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist())
    return [self._paint_objmap(r) for r in results]  # uint16 objmap per image
```

`_paint_objmap(result)` sorts `result["masks"]` largest-first (by `mask.sum()`), paints
into a `uint16` objmap (same uint16-cap warning as `Sam2`), applies
`min_mask_region_area`. Re-point the autodoc line (
`Sam3.apply.__doc__ = Sam3.__doc__`).

- [ ] **Step 4: Run → pass** (functional `.apply()` tests skip without `foundation`
  +weights — add a `TestSam3Functional` skipif mirroring `Sam2`).
- [ ] **Step 5: Commit**
  `feat(detect): Sam3 — text-prompted true-batch instance detector`.

---

### Task 5: `Sam3` grid-aware tiling + cross-tile IoU-NMS (O4)

**Files:** Modify `_sam3_detector.py`; Test `test_sam3_detector.py`.

**Why:** SAM3 caps at `num_queries=200` per forward and runs at 1008 px internal — dense
plates (>200 colonies) must be tiled. O4: grid-aware for `GridImage` (per grid
section/band), fixed ~1008 px + ~15 % overlap otherwise; merge cross-tile instances by
IoU-NMS.

- [ ] **Step 1: Failing tests** (pure tiling math + merge — no model):

```python
import numpy as np
from phenotypic.detect.nn._sam3 import _plan_tiles, _merge_tiles_iou_nms


def test_plan_fixed_tiles_cover_with_overlap():
    tiles = _plan_tiles((3000, 3000), tile_px=1008, overlap=0.15, grid=None)
    assert all(t.h <= 1008 and t.w <= 1008 for t in tiles)
    # union of tiles covers the image
    covered = np.zeros((3000, 3000), bool)
    for t in tiles:
        covered[t.y0:t.y1, t.x0:t.x1] = True
    assert covered.all()


def test_merge_dedups_overlapping_instances():
    a = np.zeros((10, 10), np.uint16);
    a[2:6, 2:6] = 1
    b = np.zeros((10, 10), np.uint16);
    b[2:6, 2:6] = 1  # same blob from neighbour tile
    merged = _merge_tiles_iou_nms([a, b], iou_thresh=0.5)
    assert merged.max() == 1  # one instance, not two
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.** Tiling fields:
  `tile_px: Annotated[int, TuneSpec(512, 2048)] = 1008`,
  `tile_overlap: Annotated[float, TuneSpec(0.0, 0.4)] = 0.15`,
  `max_instances_per_tile: TuneSpec(tunable=False) = 200`.
  `_plan_tiles(shape, tile_px, overlap, grid)` returns tile rectangles: when `grid` is
  present (a `GridImage`'s grid sections/bands), one tile per section/band; else a fixed
  grid with overlap. `infer_batch` becomes: for each sample, if it fits one tile run the
  single-tile forward (Task 4); else crop tiles, batch them through the Task-4 forward,
  offset each tile's objmap back to full coordinates, and `_merge_tiles_iou_nms` (greedy
  NMS by IoU over the overlap regions, relabel to a contiguous uint16 objmap). Keep the
  un-tiled path as the default for small images.

- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit**
  `feat(detect): Sam3 grid-aware tiling + cross-tile IoU-NMS`.

---

### Task 6: `DinoSam2Detector` — SAM2 AMG proposals + DINO scoring (O1, O6)

**Files:** Create `src/phenotypic/detect/nn/_dinosam2_detector.py`; Modify
`detect/nn/__init__.py`; Test `tests/unit/detect/nn/test_dinosam2_detector.py`.

**Recipe (clean-room from "No time to train!" arXiv:2507.02798 — do NOT vendor):** SAM2
`SAM2AutomaticMaskGenerator` → class-agnostic proposals; pool DINO patch features inside
each proposal mask; score each proposal by cosine similarity of its pooled feature to
the foreground prototype (mean of high-confidence proposals' features); drop
low-similarity (background) proposals; merge near-duplicate proposals by IoU; paint
survivors largest-first into `objmap`.

- [ ] **Step 1: Failing tests** (construction / capability / dino routing — no weights):

```python
import pytest
from phenotypic.detect.nn import DinoSam2Detector


class TestDinoSam2Construction:
    def test_capability_fields(self):
        det = DinoSam2Detector()
        assert det.input_layer == "rgb"
        assert det.output_kind == "instance"
        assert det.supports_batching is False

    def test_dino_version_defaults_to_2(self):
        assert DinoSam2Detector().dino_version == 2     # DINOv2, ungated (O1)

    def test_dinov3_is_opt_in(self):
        det = DinoSam2Detector(dino_version=3, dino_size="base")
        assert det.dino_version == 3
        assert det._hf_dino_id() == "facebook/dinov3-vitb16-pretrain-lvd1689m"

    def test_dinov2_hf_id(self):
        assert DinoSam2Detector(dino_size="base")._hf_dino_id() == "facebook/dinov2-base"

    def test_serialization_round_trip(self):
        d = DinoSam2Detector(dino_version=3, similarity_thresh=0.6)
        r = DinoSam2Detector.from_json(d.to_json())
        assert r.dino_version == 3 and r.similarity_thresh == 0.6
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.** Fields: `dino_version: DinoVersion = 2`,
  `dino_size: DinoSize = "base"`, `sam2_model_size: ... = "tiny"` (reuse `Sam2`'s
  sizes), `similarity_thresh: Annotated[float, TuneSpec(0.0, 1.0)] = 0.5`,
  `merge_iou_thresh: Annotated[float, TuneSpec(0.0, 1.0)] = 0.7`,
  `min_proposal_area: Annotated[int, TuneSpec(0, 500)] = 0`, `device`. Capabilities:
  instance / rgb / `supports_batching=False` (default looped `infer_batch`).
  `_hf_dino_id()` maps `(dino_version, dino_size)` → HF id (`facebook/dinov2-{size}` or
  `facebook/dinov3-vit{b}16-pretrain-lvd1689m`); `dino_version=3` routes
  `_ensure_model_loaded` through `Dinov3CheckpointManager` (Spec 2b adds it — for 2a, a
  `dino_version=3` *load* may raise `NotImplementedError("DINOv3 lands in Spec 2b")`
  while construction/serialization still work, so the field + routing are testable now).
  `_ensure_model_loaded` builds the SAM2 AMG (reuse `Sam2` internals) + the DINOv2
  backbone (`AutoModel.from_pretrained(self._hf_dino_id())`). `_infer_one(sample)` runs
  the recipe above and returns a uint16 objmap.

- [ ] **Step 4: Run → pass** (functional `.apply()` skips without `foundation`+`sam2`
  +weights).
- [ ] **Step 5: Commit**
  `feat(detect): DinoSam2Detector — SAM2 proposals + DINOv2 scoring (DINOv2 default)`.

---

### Task 7: Exports + annotation-coverage gate + regression

**Files:** Modify `detect/nn/__init__.py` (if not already), verify the annotation gate,
regression.

- [ ] **Step 1:** Ensure `Sam3`, `DinoSam2Detector` are exported from
  `phenotypic.detect.nn` (and re-exported where `Sam2` is). Test:
  `from phenotypic.detect.nn import Sam3, DinoSam2Detector`.
- [ ] **Step 2: Annotation-coverage gate.** Run
  `uv run pytest tests/unit/tune/test_annotation_coverage.py -v`. Every new `int`/
  `float` field on the two detectors (`score_thresh`, `mask_threshold`,
  `min_mask_region_area`, `tile_px`, `tile_overlap`, `similarity_thresh`,
  `merge_iou_thresh`, `min_proposal_area`) must carry a `TuneSpec` (numeric window) or
  `TuneSpec(tunable=False)` (e.g. `max_instances_per_tile`). Fix any gap the gate
  reports.
- [ ] **Step 3: Targeted regression** (sequential, no `-n`):

```bash
uv run pytest tests/unit/detect/nn tests/unit/test_foundation_extra.py tests/unit/test_licensing_scaffolding.py tests/unit/tune/test_annotation_coverage.py -q
uv run mypy --follow-imports=silent src/phenotypic/detect/nn/_sam3.py src/phenotypic/detect/nn/_dinosam2_detector.py src/phenotypic/detect/nn/_checkpoint_manager.py src/phenotypic/tools_/typing_.py
uv run ruff check src/phenotypic/detect/nn tests/unit/detect/nn
```

Expected: PASS / SKIP (functional tests skip without `foundation`+weights). Reconcile
any migration golden touched by the new detectors.

- [ ] **Step 4: Commit** `test(detect): green regression for Sam3 + DinoSam2 detectors`.

---

### Task 8: Docs — how-to + detector docstrings

**Files:** Modify `docs/source/how_to/pages/gpu_detection_setup.md`, the detector
docstrings; Test: extend `tests/unit/test_docs_staged_cli.py` or a new
`test_docs_foundation.py`.

- [ ] **Step 1: Failing content check** — the how-to documents the new models + gated
  flow:

```python
def test_how_to_documents_sam3_and_gated_install():
    txt = (REPO / "docs/source/how_to/pages/gpu_detection_setup.md").read_text(
        encoding="utf-8")
    assert "Sam3" in txt and "DinoSam2Detector" in txt
    assert "hf auth login" in txt and "--extra foundation" in txt
```

- [ ] **Step 2: Run → fail.**
- [ ] **Step 3: Implement.** Add to the how-to: `uv sync --extra foundation` /
  `--extra gpu`; the per-model license table (SAM3 gated / DINOv2 Apache);
  `uv run hf auth login` + `PHENOTYPIC_ACCEPT_MODEL_LICENSE` for SAM3; HPCC pre-staging
  with `HF_HOME`/`HF_HUB_OFFLINE`; the env-var table (Spec 2 §7). Each detector
  docstring (Google style, runnable doctest via `load_synth_yeast_plate()`) documents
  `output_kind`, the gated-weight requirement (SAM3), the overrideable `prompt` (SAM3),
  and the DINOv2-default/DINOv3-opt-in (DinoSam2).
- [ ] **Step 4: Run → pass.**
- [ ] **Step 5: Commit**
  `docs(foundation): SAM3 + DinoSam2 install, gated weights, per-model licenses`.

---

## Self-Review

**Spec coverage (Spec 2a):** packaging extras (§5) → Task 1; licensing (§6) → Task 2;
gated download flow (§7) → Task 3; `Sam3` (§4.1, O4 tiling) → Tasks 4–5;
`DinoSam2Detector` (§4.2, O1) → Task 6; tests (§9) → Tasks 4–7; docs (§8) → Task 8. ✓
**O3/O7 resolved:** `transformers>=5.2.0`; SAM3 API verified (text prompt,
`post_process_instance_segmentation`, `original_sizes`). ✓
**Out of scope (Spec 2b — separate plan):** `Insid3Detector`, `FssDinoDetector` (
few-shot/semantic, clean-room, gated DINOv3, curated exemplar data), and the
`Dinov3CheckpointManager` (stubbed/NotImplemented for the `dino_version=3` *load* path
here).
**Type consistency:** `DinoVersion`/`DinoSize` aliases used by `DinoSam2Detector`;
`infer_batch(batch) -> List[np.ndarray]`, `_infer_one(sample) -> np.ndarray` match Spec
1's signatures.

## Plan-Review Resolutions (apply during implementation — supersedes the tasks where they conflict)

Plan-reviewer pass (2026-06-17) verified the SAM3 transformers API, the DINO ids, the HF
error hierarchy, and the AMG-reuse shape as **correct**. It found 2 blockers +
should-fixes;
the user resolved the design OQs. Apply all of the below.

**User decisions (resolve the surfaced OQs):**

- **D-tiling — FIXED GEOMETRIC TILES (drop grid-awareness).** O4's "per grid section" is
  architecturally infeasible: a `GpuDetector` runs *before* GridFinder and `_operate`
  passes
  only `getattr(image, input_layer)[:]` (a raw array) — the detector never sees the
  `GridImage`/grid. **Task 5:** `_plan_tiles(shape, tile_px, overlap)` has **no `grid`
  param**;
  always fixed ~`tile_px` tiles with `overlap`, IoU-NMS cross-tile merge. Drop the
  `grid`-aware branch and the grid test.
- **D-instance-only — NO semantic toggle.** Keep `Sam3` instance-only
  (`output_kind="instance"` fixed). SAM3's `semantic_seg` is left for a future 2b add.
- **D-foundation-install — INSTALL the `foundation` extra in the env** (
  `uv sync --extra foundation`)
  and smoke-test the **import path**: a new test asserts `transformers` + `Sam3Model`/
  `Sam3Processor`
  symbols resolve and the detector modules import. Functional `.apply()` (gated ~3.45 GB
  weights)
  still skips. Add a `FOUNDATION_AVAILABLE` flag to `detect/nn/__init__.py` (
  transformers
  importable) beside `SAM2_AVAILABLE` (S4) and gate the import-smoke + functional tests
  on it.

**Blocker fixes:**

- **B1 — match the EXISTING checkpoint-manager + CLI shapes.** Read
  `detect/nn/_checkpoint_manager.py` (`Sam2CheckpointManager.download` is a
  `@classmethod(model_size, *, force)`)
  and `detect/nn/_cli.py` (`--model-type {sam2,microsam}` `click.Choice` +
  `--model-size`, **no**
  `--model`/`--accept-license`). Task 3: make `Sam3CheckpointManager`/
  `Dinov2CheckpointManager`
  consistent with the existing manager style, and **extend `_cli.py`** — add `sam3`/
  `dinov2` to the
  `--model-type` `click.Choice` and add an `--accept-license` flag (sets
  `PHENOTYPIC_ACCEPT_MODEL_LICENSE` for the call). Keep the module-level
  `snapshot_download`
  indirection as the test patch point. The Task-3 test method names/signatures must
  match whatever
  manager shape you choose (instance vs classmethod) — pick one and make the tests
  follow it.
- **B2 — see D-tiling** (the O4 fix).

**Should-fixes:**

- **C1 — DinoSam2 AMG reuse:** there is NO public accessor for `Sam2._generator`. *
  *Rebuild**
  the `SAM2AutomaticMaskGenerator` in `DinoSam2Detector._ensure_model_loaded` (same
  `build_sam2` +
  generator construction as `Sam2`); if the duplication is ugly, extract a small shared
  `_build_sam2_generator(...)` helper. State which you did.
- **C2 — drive the recipe with a test:** add ≥1 algorithmic unit test on **synthetic**
  features/masks
  (e.g. prototype-cosine scoring + IoU merge on hand-built arrays, like the
  `_merge_tiles_iou_nms`
  test) so the clean-room recipe body isn't construction-only. Keep the concrete
  recipe (SAM2 AMG →
  per-proposal pooled DINO feature → cosine-to-prototype score → threshold filter → IoU
  merge).
- **C3 — `min_mask_region_area` default = 100** (match `Sam2`, not 0).
- **C4 — spell out tiling↔batch interaction:** in Task 5, regroup tiles by source image,
  run the
  Task-4 forward per tile-batch, set each tile's `target_sizes` to the **tile's own (
  H,W)** (not the
  full image), offset each tile objmap back to full coords, then IoU-NMS. Add a test
  that two images
  with different tile counts batch correctly.
- **C5 — device:** resolve `self.device` via `resolve_device(...)` and cache
  `self._device` in
  `_ensure_model_loaded` (mirror `Sam2`); the `infer_batch` snippet's `self._device`
  must be
  set there.
- **S1 — annotation-gate denominator:** verify the two detectors are discoverable by the
  gate
  (`detect/nn/__init__.__all__` / `iter_numeric_tunable_fields`). Make this an explicit
  Task-7 check,
  not an assumption. (All new numeric fields already carry a `TuneSpec` — verified by
  the reviewer.)
- **S2 — reuse `Sam2ModelSize`** for `DinoSam2Detector.sam2_model_size` (import the
  existing alias).
- **D-dino3-stub (kept):** ship `dino_version=3` field + `_hf_dino_id()` routing now;
  the DINOv3
  *load* raises `NotImplementedError("DINOv3 lands in Spec 2b")` (
  construction/serialization/`_hf_dino_id`
  still work). Reviewer judged this clean (a `dino_version=3` pipeline serializes but
  fails only at
  run time — deliberate, tested boundary).

## Execution Handoff

Plan + resolutions complete. Per the requested flow: implement with a **single Opus
agent**
(applying this resolutions section alongside the tasks), then a **code-review subagent
**.
