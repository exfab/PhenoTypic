# Spec 2 — GPU Detector Models (SAM3, DINO+SAM2, INSID3, FSSDINO)

- **Status:** Draft — plan-reviewer pass applied 2026-06-16 (S5/S6 → O6/O7); pending user review
- **Date:** 2026-06-16
- **Feature folder:** `docs/superpowers/specs/gpu-detection-optimization/`
- **Depends on:** Spec 1 — *Staged Batched GPU Detection Architecture*
  (`2026-06-16-staged-batched-gpu-detection-design.md`)

---

## 1. Scope

Add four new GPU detectors on top of Spec 1's `GpuDetector` interface — **no engine
changes**, only `infer_batch` / `output_kind` / `supports_batching` implementations —
plus the packaging, licensing, and gated-weights machinery they require.

All facts below were verified against primary sources during brainstorming research
(repos, HF model cards + LICENSE files, arXiv). Where a model is encumbered or a poor
fit, that is stated explicitly rather than smoothed over.

---

## 2. Model Roster — Feasibility & Licensing Matrix

| Model | Output route | Batching | Needs reference exemplars? | Weights | Code license | Weights license | Fit |
|---|---|---|---|---|---|---|---|
| **SAM3** | instance (`objmap`) | **true (N,C,H,W)** | no (1 text prompt `"colony"`) | gated HF (~3.45 GB) | SAM License (commercial-OK) | SAM License, **gated** | **Best new add** (tiling for >200 colonies/plate) |
| **DINO+SAM2** | instance (via SAM2 masks) | per-image | optional (reference-based variant) | SAM2 ungated; DINO backbone | SAM2 Apache-2.0; recipe = reimplement | SAM2 Apache; **DINOv2 Apache / DINOv3 gated** | **Good — default DINOv2** |
| **INSID3** | **semantic** (`objmask`) | per-image | **yes** (in-context img+mask) | ships none; gated DINOv3 backbone | Apache-2.0 | DINOv3 License (gated) | Niche — semantic + reference |
| **FSSDINO** | **semantic** (`objmask`) | per-image | **yes** (few-shot support set) | ships none; gated DINOv3 backbone | **none / all-rights-reserved** ⚠ | paper CC BY-NC-SA; DINOv3 gated | Weakest — **clean-room only** |

**Cross-cutting pattern:** SAM2/SAM3 are instance-native with usable licenses;
INSID3 + FSSDINO are DINOv3-based, **semantic-only**, **few-shot**, and license-
encumbered. The semantic pair emit `objmask` and reuse the user's downstream
watershed (Spec 1 §8) — they do **not** carry an in-detector instancer.

---

## 3. Strategic Steer — prefer **DINOv2** over DINOv3

Three of the four models can lean on a DINO backbone, and **DINOv3 is gated *and* under
a non-permissive custom license** (commercial-OK, but redistributable only *under the
DINOv3 License with the agreement attached* — incompatible with a permissive package;
each user must accept + `hf auth login`). **DINOv2 is Apache-2.0 and ungated** and
serves the same feature-matching role with minimal accuracy cost.

**Therefore:**
- **DINO+SAM2 → default to a `DinoSam2Detector` on DINOv2** (fully Apache/ungated:
  SAM2 Apache + DINOv2 Apache), with **DINOv3 as an explicit opt-in** for users who
  accept its gate. This makes the recommended config gate-free.
- **INSID3 / FSSDINO** (which are DINOv3-specific by construction) stay lower priority;
  if built, document the gated-DINOv3 requirement prominently.

**Open decision O1 (carry into the plan):** ship `DinoSam2Detector` **DINOv2-default
with DINOv3 opt-in**, vs. DINOv3-as-originally-named. Recommendation: DINOv2-default.

**Open decision O2:** whether to include the semantic/few-shot pair **INSID3 + FSSDINO
at all** in the first model cut, given they need curated exemplars, a gated backbone,
and (FSSDINO) a clean-room reimplementation. Recommendation: land SAM3 + DinoSam2 first;
treat INSID3/FSSDINO as a follow-on.

---

## 4. Per-Model Integration Design

All four subclass `GpuDetector`, lazy-load in `_ensure_model_loaded()`, read
`image.<input_layer>[:]`, and write back per their `output_kind` (Spec 1 §4, §8). They
mirror the existing `Sam2Detector` structure
(`src/phenotypic/detect/nn/_sam2_detector.py`).

### 4.1 `Sam3Detector` — instance, true-batch, gated

- **Load (transformers route, preferred over the git repo):**
  `Sam3Model.from_pretrained("facebook/sam3")` + `Sam3Processor.from_pretrained(...)`.
- **Fields:**
  - **`prompt: str = "colony"` — overrideable parameter.** SAM3 has **no prompt-free
    "segment everything" mode**; it requires a text/exemplar prompt. `prompt` is a
    free-text field the user can override per run (e.g. `"bacterial colony"`,
    `"yeast colony"`). Per project convention, a parameterised free-text string stays
    a plain `str` field (not an Enum/`Literal`), with no `TuneSpec` (it is not a
    numeric knob).
  - `score_thresh: float = 0.5` (Annotated with a `TuneSpec` per the annotation-
    coverage gate), `device` (reuse existing `Device` type). **Drop** SAM2-only fields
    (`model_size`, `points_per_side`, `pred_iou_thresh`, `stability_score_thresh`) —
    SAM3 has one checkpoint and no point grid. `min_mask_region_area` may be kept as a
    post-filter.
- **Capabilities:** `input_layer="rgb"`, `output_kind="instance"`,
  **`supports_batching=True`** → override `infer_batch` with a true `(N,C,H,W)`
  forward (`processor(images=[...], text=[prompt]*N)` → `model(**inputs)` →
  `post_process_instance_segmentation`), painting masks largest-first into `objmap`
  (same loop as `Sam2Detector`, sorting by `mask.sum()`).
- **Constraints to handle:** **200-instance cap** per forward (`num_queries=200`) →
  **tile dense plates** and merge; **gated weights** (§7); fixed **1008 px** internal
  resolution (model resizes; map masks back via `target_sizes`).

### 4.2 `DinoSam2Detector` — instance, DINOv2-default

- **Recipe (training-free, no Meta "DINOv3+SAM2" model exists — it's a composition):**
  run SAM2's `SAM2AutomaticMaskGenerator` to get class-agnostic mask proposals (already
  done by `Sam2Detector`), pool DINO features inside each proposal, and **score /
  filter / merge** proposals by feature similarity to drop background-like masks and
  fix over-segmentation. Closest published referent: *"No time to train!"*
  (arXiv:2507.02798) — **reimplement the recipe**; do not vendor that repo (license
  unverified).
- **Fields:** `dino_backbone: str = "facebook/dinov2-base"` (default DINOv2, ungated;
  DINOv3 ids accepted as opt-in), `sam2_model_size`, `device`, similarity/merge
  thresholds (numeric → `TuneSpec`). Optional `reference_images` / `reference_masks`
  fields enable the few-shot variant.
- **Capabilities:** `input_layer="rgb"`, `output_kind="instance"`,
  `supports_batching=False` (SAM2's per-image AMG bounds it) → use the default looped
  `infer_batch`; fill the GPU via `workers_per_gpu` packing (Spec 1 §7).

### 4.3 `Insid3Detector` — semantic, in-context, gated backbone

- **Source:** `visinf/INSID3` (Apache-2.0 code) — **vendor the small module with
  attribution** or clean-room; backbone is a frozen **gated DINOv3**.
- **Behavior:** one-shot in-context **semantic** segmentation — needs a **reference
  image + reference mask**, supplied as detector fields (`reference_image: Path`,
  `reference_mask: Path`), loaded in `_ensure_model_loaded()` and cached.
- **Capabilities:** `input_layer="rgb"`, **`output_kind="semantic"`** (emit `objmask`;
  user's downstream watershed instances it — Spec 1 §8), `supports_batching=False`,
  1024 px default → tiling for large plates.

### 4.4 `FssDinoDetector` — semantic, few-shot, **clean-room only**

- **Source:** `hussni0997/fssdino` has **no license (all rights reserved)** → **do not
  vendor**. The algorithm (~150 lines: prototype-cosine + Gram-matrix matching over a
  frozen DINO backbone) is simple and unpatented → **clean-room reimplement** from the
  paper (arXiv:2602.07550). Attribute the paper (CC BY-NC-SA).
- **Behavior:** few-shot **semantic** segmentation — needs a **support set**
  (`support_images: list[Path]`, `support_masks: list[Path]`, `n_clusters: int = 5`
  with `TuneSpec`); prototypes cached on the instance.
- **Capabilities:** `input_layer="rgb"`, **`output_kind="semantic"`**,
  `supports_batching=False`, 512 px default → tiling. Prefer a **DINOv2** backbone to
  avoid DINOv3 gating where the method permits.

---

## 5. Packaging — Dependency Extras

`transformers` / `huggingface_hub` are not currently declared. The installable code for
SAM3 + every DINO backbone collapses to those two, so a single `foundation` extra (not
per-model extras) is correct; per-model differentiation lives in the runtime download +
license gate (§7), not at install time.

```toml
[project.optional-dependencies]
# torch — UNCHANGED (preserves `phenotypic[torch]`): torch, torchvision, sam2

foundation = [
    "phenotypic[torch]",
    "transformers>=X.Y",       # pin to first release shipping Sam3Model + DINOv3
    "huggingface_hub>=A.B",
]
gpu = ["phenotypic[torch,foundation]"]   # umbrella: every GPU detector
```

- Self-referential extras (`phenotypic[torch]` in `foundation`, `foundation` in `gpu`)
  are valid PEP 621 / modern-pip.
- **Installing any extra never pulls encumbered material** — `transformers`,
  `huggingface_hub`, `sam2` are all permissive (Apache). Gated/non-permissive bits are
  **weights**, fetched at runtime behind the acceptance gate (§7).
- **No git/unlicensed deps in extras**: SAM3 native repo, INSID3 (git), FSSDINO
  (unlicensed) are not declared. SAM3 via `transformers`; INSID3 vendored/clean-room;
  FSSDINO clean-room — all need only `transformers`. `pip install git+…` stays a
  documented power-user option, never a declared dep.
- Windows markers stay on torch-y deps (`;sys_platform!='win32'`);
  `transformers`/`huggingface_hub` are pure-Python, unmarked.
- mypy: add `"transformers.*"` (and `"huggingface_hub.*"` if stubs missing) to the
  existing `ignore_missing_imports` module list (next to `sam2.*` / `micro_sam.*`).

### Install with uv

```bash
uv sync --extra gpu                  # everything (dev env)
uv sync --extra foundation           # SAM3 + DINO-based only
uv add 'phenotypic[gpu]'             # add as a dependency of a downstream project
uv pip install 'phenotypic[foundation]'
```

---

## 6. Licensing — Repo Additions (extends Spec 1 §12 scaffolding)

Populate the `licenses/` dir + `NOTICE` (created in Spec 1) with the **verified**
names:

- `licenses/sam2-Apache-2.0` (Spec 1)
- `licenses/sam3-SAM-License`
- `licenses/dinov3-License` (custom Meta)
- `licenses/dinov2-Apache-2.0` (if adopted)
- `licenses/insid3-Apache-2.0`
- **FSSDINO:** no license file — **do not vendor**; if reimplemented from the paper,
  attribute it (CC BY-NC-SA) in `NOTICE`.

`NOTICE` reiterates: **PhenoTypic does not redistribute model weights**; each weight is
downloaded by the user from upstream under that model's license, which the user accepts.

---

## 7. Gated Weights — Install Flow

**Scope:** only **SAM3** and **DINOv3** are gated. **SAM2** (existing `torch.hub`) and
**DINOv2** (plain `huggingface_hub` pull) are ungated and need no handshake.

**Principle:** the `foundation` extra installs only permissive code; **weights are never
in git or the wheel** and are downloaded at runtime under the **user's own** acceptance
of the model's license. PhenoTypic facilitates, never accepts-on-behalf or
redistributes.

**Step 1 — one-time human handshake (per user, per gated model; the binding step):**
1. Have a Hugging Face account.
2. Accept the gate on the model page (`huggingface.co/facebook/sam3`,
   `huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m`) — share contact info /
   request access. This *is* the license acceptance.
3. Authenticate locally once: `uv run hf auth login` (stores a token), or export
   `HF_TOKEN`.

**Step 2 — checkpoint manager does the authenticated pull.** Add gated managers
(`Sam3CheckpointManager`, `Dinov3CheckpointManager`) beside the existing
`Sam2CheckpointManager`, using `huggingface_hub.snapshot_download(repo_id=…, token=…)`
into the HF cache. Guardrails:
- **PhenoTypic's acceptance gate** (Spec 1 hook): before first download, print the
  license name + URL and require acknowledgement — interactive prompt, `--accept-license`
  flag, or `PHENOTYPIC_ACCEPT_MODEL_LICENSE=sam3,dinov3` for batch. (Informational layer
  on top of the binding HF gate.)
- **Actionable 401/403 handling:** if the gate isn't accepted or no token is present,
  catch it and print exactly what to do ("Request access at <url>, then run
  `uv run hf auth login`").

**Step 3 — download CLI (extended `phenotypic.detect.nn`):**
```bash
# with uv (project convention):
uv run python -m phenotypic.detect.nn download --model sam3 --accept-license
uv run python -m phenotypic.detect.nn download --model dinov3
uv run python -m phenotypic.detect.nn list
uv run python -m phenotypic.detect.nn clear --model-type sam3
# plain python -m equivalent (no uv):
python -m phenotypic.detect.nn download --model sam3 --accept-license
```

**Step 4 — HPCC pre-staging (offline compute nodes).** Never download inside a job:
1. On a **login node** (has internet), cache to shared storage and download once:
   ```bash
   export HF_HOME=/bigdata/exfab/<...>/hf_cache
   uv run hf auth login
   uv run python -m phenotypic.detect.nn download --model sam3 --accept-license
   uv run python -m phenotypic.detect.nn download --model dinov3
   ```
2. **Stage-2 SLURM jobs** set the same `HF_HOME` and run with `HF_HUB_OFFLINE=1`
   (+ `TRANSFORMERS_OFFLINE=1`), loading from the pre-staged cache, no network. (Mirrors
   the existing `micro_sam` pre-staging guidance.)

**Environment variables (documented in the GPU-setup how-to):**

| Var | Purpose |
|---|---|
| `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | auth for gated download (vs `hf auth login`) |
| `HF_HOME` (or `HF_HUB_CACHE`) | cache location → point at shared HPCC storage |
| `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` | force load-from-cache on offline nodes |
| `PHENOTYPIC_ACCEPT_MODEL_LICENSE` | non-interactive acknowledgement for batch jobs |

**Non-interactive acceptance:** SLURM jobs can't prompt — the binding acceptance (HF
gate + token) is done **once on the login node** during pre-staging; the compute job
carries only the already-accepted token + `PHENOTYPIC_ACCEPT_MODEL_LICENSE` and reads
the pre-staged cache. Acceptance never happens silently inside a batch job.

---

## 8. Docs

Update `docs/source/how_to/pages/gpu_detection_setup.md`: per-model install
(`uv sync --extra …`), license acceptance, `uv run hf auth login` for gated weights,
HPCC pre-staging, the env-var table, and a per-model license table. Each detector's
docstring documents its `output_kind`, the gated-weight requirement (where applicable),
and (SAM3) the overrideable `prompt`.

---

## 9. Testing

- Construction / serialization round-trips **without GPU or weights** (lazy load), for
  all four detectors (matches existing `Sam2Detector` doctest pattern).
- `output_kind="semantic"` detectors set only `objmask` and round-trip through the
  staged HDF identically to a threshold detector (shared assertion with Spec 1 §8).
- `Sam3Detector.prompt` override is honoured (mock the processor/model).
- Gated-download manager: 401/403 → actionable message; offline cache hit;
  acceptance-gate enforced.
- DINOv2-default `DinoSam2Detector` runs gate-free (no HF token required).

---

## 10. Open Questions

- **O1:** `DinoSam2Detector` DINOv2-default + DINOv3 opt-in (recommended) vs.
  DINOv3-as-named.
- **O2:** include INSID3 + FSSDINO in the first model cut, or land SAM3 + DinoSam2
  first and defer the semantic/few-shot pair (recommended).
- **O3:** exact `transformers` minimum version (pin against the changelog at
  implementation time — must include `Sam3Model` + DINOv3).
- **O4:** SAM3 dense-plate tiling strategy (tile size / overlap / instance merge) given
  the 200-query cap.
- **O5:** commercial-distribution posture for PhenoTypic — confirms whether the
  DINOv3-gated models are acceptable to document as first-class or as advanced opt-ins.
- **O6 (review S5):** verify the *"No time to train!"* (arXiv:2507.02798) recipe is
  freely clean-room reimplementable — the building blocks (SAM2, DINOv2) are Apache, but
  confirm the composition isn't patent-encumbered before building `DinoSam2Detector`.
- **O7 (review S6):** re-verify the SAM3 `transformers` API at implementation time — if
  an automatic-mask-generator ("segment everything") variant ships, the `prompt` field
  and the 200-query tiling (O4) may be partly unnecessary.

---

## 11. Dependency on Spec 1

Spec 2 requires Spec 1's `GpuDetector` interface (`input_layer`, `supports_batching`,
`output_kind`, `preprocess`/`collate`/`infer_batch`), the staged HDF engine, and the
licensing scaffolding. No Spec 1 engine changes are needed — the four models are pure
interface implementations.
