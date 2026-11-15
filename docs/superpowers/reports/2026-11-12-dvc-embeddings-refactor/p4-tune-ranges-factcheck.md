# Fact-check: `MeasureDinoEmbedding` tune ranges (P4 Task 3)

Reviewer: fresh `fact-checker` (Opus). It returned its report as messages, and
the orchestrator wrote them here. No repo file was edited by the reviewer.

## Verdicts

| Claim | Verdict | Action |
|---|---|---|
| DINOv2 S/B/L use patch 14; DINOv3 S/B/L use patch 16 | Correct | none |
| `crop_px` ∈ {112, 224, 336, 448} are "the multiples of both 14 and 16 in range" | Arithmetic correct; "in range" unjustified | plan wording fixed; range kept |
| HF models accept sizes other than 224 | Correct | none |
| `crop_pad_frac` over 0.0–1.0 | Unsupported, and per side vs total is undefined | **narrowed to 0.0–0.5, per side** |
| Register tokens "0 for DINOv2" | Needs nuance | "0 for the facebook/dinov2-* checkpoints" |
| CLS and mean pooling; normalize without resize; DINOv3 license gate | Correct | none |
| `test_every_tunable_crop_px_fits_both_patch_sizes` | Logic flag: checks a hard-coded list against itself | must read the TuneSpec |

## `crop_px`

- **Patch sizes.** HF configs for `facebook/dinov2-{small,base,large}` have
  `patch_size: 14` and `image_size: 518`. The DINOv3 model cards and README list
  every ViT as /16, and arXiv 2508.10104 §3.2 says "we use a patch size of 16
  pixels". The vitb16 config is gated (HTTP 401), so for base the evidence is
  the README and the shared naming.
- **Common multiples.** lcm(14, 16) = 112, so 112, 224, 336 and 448 are exactly
  the common multiples up to 448.
- **Lower bound.** 112 is at or above both models' smallest training crop:
  DINOv2 `ssl_default_config.yaml` has `local_crops_size: 96`, and DINOv3 §5.1
  gives local crops {112, 168, 224, 336}.
- **Upper bound.** 448 is below DINOv2's 518 high-resolution phase (arXiv
  2304.07193 §4: "we increase the resolution of images to 518×518 during a
  short period at the end of pretraining"), so position embeddings are only
  interpolated down. 560 is inside DINOv3's 768 range, so excluding it is a cost
  choice. Tokens per crop: 64, 256, 576, 1024 at patch 14 and 49, 196, 441, 784
  at patch 16; attention cost is quadratic.
- **Interpolation.** DINOv2 interpolates position embeddings unconditionally in
  `Dinov2Embeddings.forward` (`interpolate_pos_encoding`), and even 224 is
  interpolated from the 37×37 grid. DINOv3 uses RoPE ("can seamlessly process
  images at varying resolutions", §4.3). A non-multiple size does not raise: the
  stride-patch Conv2d floors it. The multiple-of-patch check is therefore a
  deliberate choice, not a model requirement.

## `crop_pad_frac`

- **Undefined.** Neither the plan, the spec nor the feasibility artifact said
  per side or in total. At f = 1.0 per side the crop is 3× the box side, and
  the colony fills about 1/9 of it.
- **Precedents:**
  - R-CNN (arXiv 1311.2524 §2.1): 16 px of context at a 227 px warp, about
    0.08 per side, which "outperformed the alternatives by a large margin
    (3-5 mAP points)".
  - ViLD (arXiv 2104.13921 §3.2) ensembles 1× and 1.5× crops; 1.5× is
    f = 0.25 per side.
  - DeepProfiler uses a fixed absolute `box_size`.
  - No source supports context up to 3×.
- **Arrayed-plate geometry.** A crop padded by f per side reaches the next
  colony once f > P/d − 1 (P is the grid pitch, d the colony diameter). On
  dense formats that is around f = 0.25, and at f = 1.0 a crop can take in up
  to 8 neighbours.
- **Overlap.** With `mask_background=True`, padding adds only zeroed pixels and
  overlaps with what `crop_px` controls.
- **Aspect ratio.** Resizing a non-square or edge-clipped box straight to
  `crop_px × crop_px` stretches irregular or filamentous colonies.

## Decisions (orchestrator, within the spec)

The spec pins only the default (0.25) and "objects only" (design.md:735).

- **Per side.** `crop_pad_frac` is per side, a fraction of each bbox dimension:
  crop side = (1 + 2f) × box side.
- **Range 0.0–0.5, default 0.25.** That is 1× to 2×, spanning the R-CNN and
  ViLD precedents; 0.25 equals ViLD's 1.5×.
- **Square the padded box** to its longer side, centred, before resizing.
  Clip to the image, and zero-pad a clipped box back to square rather than
  stretching it. A test pins that an elongated colony is not stretched. Grid
  cells keep the plan's rule.
- **`crop_px` stays {112, 224, 336, 448}**, with the reason in the plan
  reworded.
- **The `crop_px` test reads the TuneSpec**, and a second test pins the
  `crop_pad_frac` bounds.
- **`dtype`** (C18, separately): the plan's `Literal["float16", "float32"]`
  contradicted the spec (E3, design.md:737, and `FixedSizeList<float16>[D]` at
  :529/:565). It is now `Literal["float16"]`.

## Sources

- DINOv2: https://arxiv.org/abs/2304.07193 ;
  https://github.com/facebookresearch/dinov2/blob/main/dinov2/configs/ssl_default_config.yaml ;
  https://huggingface.co/facebook/dinov2-base/raw/main/config.json (also -small,
  -large and dinov2-with-registers-base, which has `num_register_tokens: 4`)
- DINOv3: https://arxiv.org/abs/2508.10104 ;
  https://github.com/facebookresearch/dinov3 ;
  https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
- transformers: `models/dinov2/modeling_dinov2.py`,
  `models/dinov3_vit/modeling_dinov3_vit.py`;
  https://huggingface.co/docs/transformers/main/en/model_doc/dinov3
- R-CNN https://arxiv.org/abs/1311.2524 ; ViLD https://arxiv.org/abs/2104.13921 ;
  DeepProfiler https://cytomining.github.io/DeepProfiler-handbook/docs/05-config.html
