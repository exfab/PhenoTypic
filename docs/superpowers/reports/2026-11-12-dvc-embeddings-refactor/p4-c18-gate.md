# C18 gate: P4 Task 3, `MeasureDinoEmbedding`

Reported by the C18 implementer; written here by the orchestrator. Logs and
scripts: `/bigdata/exfab/anguy344/gate-results/c18/`. The tune ranges were
corrected by `p4-tune-ranges-factcheck.md` before the code was committed.

## Commit

**`4d16b84d8`** adds:
- `MeasureDinoEmbedding` and the `DINO_EMBEDDING` schema enum (`EMBEDDING`,
  with `label` and `desc` only).
- In `_dino_support.py`: `load_dino_backbone(revision=)`, `resolved_revision`,
  `embed_crops` and `resolve_backbone_device`.
- Two ABC hooks: `_leading_fields`, which reorders fields in
  `__pydantic_init_subclass__`, and `_embed_object_boxes(image, boxes,
  labels)`. The second delegates to `_embed_boxes` by default.
- The `stub_backbone` fixture and the test file.

## Real-model smoke test

| Item | Value |
|---|---|
| Job | `32097385`, `intel`, 4 CPU, 16 GB |
| Model | DINOv2-small, CPU |
| Result | `1 passed in 148.91s`, COMPLETED 0:0 |
| Colonies, dim | 552, 384 |
| `measure()` | 147.2 s, including the model load |
| Wall time | 3:53 |
| Peak RSS | 1,689,436 kB (sacct), 1,644,000 kB (`time -v`) |
| Resolved revision | `ed25f3a31f01632728cabb09d1542f84ab7b0056` |

**First attempt (`32097264`): FAILED 1:0 with a Hugging Face 401.**
- The user's stored token (`~/.cache/huggingface/token`) has expired.
  `huggingface_hub` sends it even to the public, ungated DINOv2 repo, and the
  hub rejects the request.
- The request went to the network because transformers 5.12's
  `ImageProcessingMixin.get_image_processor_dict` probes `processor_config.json`
  before `preprocessor_config.json`. The cached snapshot has neither that file
  nor a no-exist marker for it.
- A 404 from that probe would be tolerated; a 401 is raised as
  `RepositoryNotFoundError`.
- The rerun set `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` in smoke.sbatch only
  (orchestrator-approved; the request is anonymous against a public repo). No
  token file was read or changed.
- The same load path, `_dino_support.py:245`, serves the existing detectors, so
  no code change was needed.
- **For the user:** refresh with `hf auth login` before using DINOv3, which is
  gated.
- **Follow-up for the final report:** an expired token surfaces as a misleading
  "repository not found".

## Decisions

Approved or accepted by the orchestrator:

- **`dtype` is `Literal["float16"]`.** The plan's `float32` option contradicted
  spec E3 (design.md:737) and the `FixedSizeList<float16>[D]` constraint.
- **Fact-check items applied:**
  - `crop_pad_frac` is per side, tuned over 0.0–0.5;
  - colony crops are squared to the longer side and zero-filled where they leave
    the image, never stretched; grid cells keep their exact rectangle;
  - the `crop_px` test reads the TuneSpec.
- **Backbone cache.** It is keyed by `(version, size, revision, device)`, which
  maps one-to-one onto `(hf_id, revision, device)`, with `maxsize=2`. The
  measurer module's `load_dino_backbone` is a lazy delegate, because importing
  `phenotypic.detect` costs about 1 s and pulls in matplotlib.
- **`device="auto"`** uses CUDA when it is available and otherwise falls back to
  the CPU without warning. CPU is the v1 target.
- **Grid embedders.** A grid refuses `mask_background=True` and a non-default
  `crop_pad_frac`; the default is accepted so the pipeline file round-trips.
- **`masked_dense`.** Tiles are `crop_px` wide and patch-aligned with no
  overlap. A region smaller than any patch centre falls back to the patches it
  touches, then to the nearest patches. It ignores `crop_pad_frac`,
  `mask_background` and `batch_size`, and the docstring says so.
- **Column name.** `output_header(DINO_EMBEDDING.EMBEDDING)` returns
  `"embedding"`, the column actually written. The generic
  `test_output_headers` test excludes `MeasureEmbedding` producers, because the
  defaults would load real DINOv2-base weights, about 350 MB. A stub test pins
  that the documented header is the written column.
- **Schema tier and unit.** The tier is `DiscriminativeFeature` and the unit is
  `NOT_A_QUANTITY`, with one line added to the unit snapshot.
- **Torch venv.** Torch tests run in an isolated venv,
  `/bigdata/exfab/anguy344/gate-results/c18/venv-foundation`, with the dev,
  foundation and tune extras. The shared `.venv` has no torch and is unchanged.

## Coverage gap: 21 tests skip without torch

The new test file has 21 tests that load the stub backbone and so need torch:
device resolution, `embed_crops`/`inference_mode`, and the `nn.Module` stub.
They skip in the shared env that the 24-shard gates use. Among them are the
crop-geometry tests (elongated colony, zero-fill at the edge), the pooling,
mask and pad tests, the grid and `masked_dense` tests, no objects, and the
payload's model entry.

They pass in the foundation venv. **The P4 phase gate (C19) runs the P4 files
in that venv as a second job.** C18 proposed replacing `embed_crops` with a
numpy fake for the geometry tests and kept the skips instead; that idea is
recorded as an alternative.

## Verification

- **Seen red first:**
  - a collection ImportError before the module existed;
  - the fact-check tests on a variant with the old crop and the old bound:
    `3 failed, 1 passed` (the module was restored byte for byte);
  - `test_output_headers[MeasureDinoEmbedding]`, which failed on torch and
    would also have failed on the header name.
- **Regression `32097325`:**
  - shared env: `1518 passed, 28 skipped`;
  - doctests: `2 passed`;
  - foundation venv: `1540 passed, 6 skipped`.
  - Scope: tests/unit/measure, the P4 files, the import guards, the tune
    annotation gate, tests/unit/schema, the util header/unit tests, the
    docs/readme measurement refs, and the detect/nn dino and preflight tests.
- **Foundation venv `32097387`, on the frozen final tree** (hashes checked
  before and after):
  - the new file: `51 passed`;
  - tests/unit/measure: `365 passed, 3 skipped`.
- **GUI registry and param-form tests (`32097268`):** `176 passed`. This ran on
  a slightly earlier tree; the registry is re-covered by the P4 phase gate.
- ruff: clean. mypy on the 7 touched source files: no issues.

## Security follow-up

An automated review flagged that `revision` accepts any ref, including
community PR refs, whose pickle weights `from_pretrained` would unpickle.

- **Fixed in `342479e63`:** `load_dino_backbone` (shared with the detectors)
  passes `use_safetensors=True`, and neither load passes `trust_remote_code`.
  - `test_load_dino_backbone_loads_safetensors_only_and_no_remote_code` runs
    with revision `None` and `refs/pr/1`. It uses a fake `transformers`, so it
    needs no torch and runs in the shared env.
  - On the old loader it failed 2 of 2 (`KeyError: 'use_safetensors'`).
  - Every checkpoint the measurer offers ships `model.safetensors`, per the
    hub's public file listing: `facebook/dinov2-{small,base,large}` and the
    gated `facebook/dinov3-vit{s,b,l}16-pretrain-lvd1689m`.
  - Pass lines:
    - foundation venv: test_dino_support `32 passed`, the new measurer file
      `51 passed`;
    - shared env: test_dino_support plus nn preflight `46 passed, 4 skipped`.
- `revision` keeps accepting any ref, as spec E3 says. The entry records the
  resolved commit, so provenance holds.
