# SLURM Per-Stage Chaining + Licensing Scaffolding — Implementation Plan (Spec 1, Plan 3 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Run the three staged GPU-detection stages on SLURM as a **3-link `afterany` dependency chain** with **per-stage resources** (CPU partition for Stages 1 & 3, GPU partition for Stage 2 as a small array of resident-model **shard-workers**), and establish the third-party **licensing scaffolding** (`NOTICE` + `licenses/` + a license-acceptance hook) for the SAM2/micro-sam components Spec 1 touches.

**Architecture:** A `StagedSlurmStrategy` generates three SBATCH script sets — Stage 1 = CPU array over images (preprocess → HDF), Stage 2 = GPU array over **shards** (resident model streams its shard of HDFs → sidecars), Stage 3 = CPU array over images (merge sidecar → measure → re-save HDF → delete sidecar) — wired `afterany:stage_{n-1}`. Work-lists are content-defined so a few per-image failures never block the next stage. Reuses Plan 2's stage workers and splitter; reuses the existing `#SBATCH`-directive + dispatcher machinery, extended from a single flat `slurm_args` to **per-stage** resource dicts.

**Tech Stack:** Python, SLURM (`sbatch`, array jobs, `--dependency=afterany`), the existing `tools_/slurm` helpers, pytest (script-generation tests only — **no live submission**). `uv` runner.

**Source of truth:** Spec 1 §7 (topology + the "real refactor" scope note S1), §9 (content-defined resume), §12 (licensing scaffolding); decisions D5, D6, D7, D13.

**Depends on:** Plan 1 (interface) + Plan 2 (splitter, sidecar, stage workers, `StagedGpuStrategy`).

**Grounded reuse points (verified):**
- `tools_/slurm/_sbatch.py::generate_sbatch_directives(...)` — builds `#SBATCH` lines from a `slurm_args` dict (keys `slurm_partition`, `slurm_cpus_per_task`, `slurm_gpus_per_node`, …).
- `tools_/slurm/_dispatcher.py::generate_dispatcher_chain(...)` (line 101) — the existing `afterok` chunk-chain dispatcher.
- `_cli/_cli_slurm_array_scripts.py::generate_array_job_script` (line 122) / `generate_all_array_job_scripts` (line 457) — array-script generation; the array task invokes `python -m phenotypic._cli._cli_process_single --mode …` per image.
- `_cli/_cli_slurm_submission.py::submit_slurm_script_chain` (line 33).
- `AutonomousSLURMStrategy` (`_cli/_cli_execution_strategies.py`) — the current single-partition SLURM strategy to supersede for GPU runs.
- CLI options in `phenotypicCLI.py`: `--slurm` (→ `slurm_args` via `parse_slurm_args`, line 648), `--njobs` (line 640).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `NOTICE` | Third-party component attribution + "weights not redistributed" statement | Create |
| `licenses/sam2-Apache-2.0.txt`, `licenses/micro-sam-*.txt` | Verbatim upstream licenses | Create |
| `pyproject.toml` | Include `NOTICE` + `licenses/` in sdist/wheel | Modify |
| `src/phenotypic/detect/nn/_checkpoint_manager.py` | `require_license_acceptance(model, license_name, url)` hook (no-op for ungated SAM2/micro-sam) | Modify |
| `src/phenotypic/phenotypicCLI.py` | `--gpu-slurm`, `--gpu-batch-size`, `--gpu-workers-per-gpu`, `--gpu-shards` options | Modify |
| `src/phenotypic/_cli/_cli_types.py` | `ExecutionConfig` fields for the above | Modify |
| `src/phenotypic/_cli/_cli_staged_slurm.py` | `partition_shards`, per-stage `slurm_args` resolution, `StagedSlurmStrategy` (3 script sets + `afterany` chain) | Create |
| `src/phenotypic/_cli/_cli_staged_slurm_worker.py` | Stage-2 shard-worker entrypoint (`python -m … <shard_index>`) | Create |
| `src/phenotypic/_cli/_cli_execution_strategies.py` | Route SLURM GPU runs to `StagedSlurmStrategy` | Modify |
| `tests/unit/cli/test_staged_slurm_scripts.py` | shard math + generated-script directives + `afterany` deps (no submission) | Create |
| `tests/unit/detect/nn/test_checkpoint_manager.py` | license-acceptance hook | Modify |

---

### Task 1: Licensing scaffolding (`NOTICE` + `licenses/` + packaging)

**Files:**
- Create: `NOTICE`, `licenses/sam2-Apache-2.0.txt`, `licenses/micro-sam-LICENSE.txt`
- Modify: `pyproject.toml`
- Test: `tests/unit/test_licensing_scaffolding.py` (Create)

- [ ] **Step 1: Write the failing test**

`tests/unit/test_licensing_scaffolding.py`:

```python
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_notice_exists_and_disclaims_weight_redistribution():
    notice = (REPO / "NOTICE").read_text(encoding="utf-8")
    assert "does not redistribute" in notice.lower()
    assert "SAM2" in notice


def test_license_files_present():
    assert (REPO / "licenses" / "sam2-Apache-2.0.txt").is_file()
    assert (REPO / "licenses" / "micro-sam-LICENSE.txt").is_file()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_licensing_scaffolding.py -v`
Expected: FAIL — files missing.

- [ ] **Step 3: Create the files**

`NOTICE`:

```
PhenoTypic — third-party component notices
==========================================

PhenoTypic is licensed under Apache-2.0 (see LICENSE). It depends on the
third-party components below. PhenoTypic does NOT redistribute any model
weights: model weights are downloaded by the user from the upstream source
under that model's own license, which the user must accept.

- SAM2 (segment-anything-2), Meta — code & weights Apache-2.0.
  Upstream: https://github.com/facebookresearch/sam2
  License: licenses/sam2-Apache-2.0.txt
- micro-sam (computational-cell-analytics) — conda-only; see its repository.
  Upstream: https://github.com/computational-cell-analytics/micro-sam
  License: licenses/micro-sam-LICENSE.txt
```

Create `licenses/sam2-Apache-2.0.txt` and `licenses/micro-sam-LICENSE.txt` with the verbatim upstream license text (copy from the upstream repos — SAM2 is the standard Apache-2.0 text; micro-sam ships its own LICENSE).

In `pyproject.toml`, add to the build include set so attribution travels in the sdist/wheel. For hatchling (confirm the build backend in `[build-system]`):

```toml
[tool.hatch.build]
include = ["src/phenotypic", "NOTICE", "licenses/**"]
```

(If the backend is setuptools, add `NOTICE` and `licenses/*` to `MANIFEST.in` instead.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/test_licensing_scaffolding.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add NOTICE licenses pyproject.toml tests/unit/test_licensing_scaffolding.py
git commit -m "chore(licensing): add NOTICE + licenses/ scaffolding (SAM2, micro-sam)"
```

---

### Task 2: License-acceptance hook in the checkpoint manager

**Files:**
- Modify: `src/phenotypic/detect/nn/_checkpoint_manager.py`
- Test: `tests/unit/detect/nn/test_checkpoint_manager.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_require_license_acceptance_noninteractive_env(monkeypatch):
    from phenotypic.detect.nn._checkpoint_manager import require_license_acceptance
    monkeypatch.setenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", "sam3,dinov3")
    # accepted via env -> returns without raising
    require_license_acceptance("dinov3", "DINOv3 License", "https://example/lic")


def test_require_license_acceptance_blocks_without_acceptance(monkeypatch):
    from phenotypic.detect.nn._checkpoint_manager import require_license_acceptance
    monkeypatch.delenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", raising=False)
    import pytest
    with pytest.raises(RuntimeError, match="license"):
        require_license_acceptance("dinov3", "DINOv3 License", "https://example/lic",
                                   interactive=False)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/detect/nn/test_checkpoint_manager.py -k license_acceptance -v`
Expected: FAIL — function missing.

- [ ] **Step 3: Implement the hook**

Add to `_checkpoint_manager.py`:

```python
import os


def require_license_acceptance(
    model: str, license_name: str, license_url: str, *, interactive: bool = True
) -> None:
    """Gate a gated-weights download on the user accepting the model's license.

    Acceptance is satisfied by ``PHENOTYPIC_ACCEPT_MODEL_LICENSE`` (comma list of
    model names) for non-interactive/batch use, or an interactive y/N prompt.
    No-op for ungated components that never call this. Raises RuntimeError if
    not accepted.
    """
    accepted = {
        m.strip().lower()
        for m in os.environ.get("PHENOTYPIC_ACCEPT_MODEL_LICENSE", "").split(",")
        if m.strip()
    }
    if model.lower() in accepted:
        return
    if interactive:
        print(f"\n{model} weights are under the {license_name}: {license_url}")
        resp = input(f"Accept the {license_name} to download {model}? [y/N] ")
        if resp.strip().lower() in ("y", "yes"):
            return
    raise RuntimeError(
        f"{model} weights require accepting the {license_name} ({license_url}). "
        f"Re-run with --accept-license or set "
        f"PHENOTYPIC_ACCEPT_MODEL_LICENSE={model} (and `hf auth login` for gated HF)."
    )
```

(SAM2/micro-sam are ungated and never call this — the hook exists for Spec 2's gated SAM3/DINOv3 managers.)

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/detect/nn/test_checkpoint_manager.py -k license_acceptance -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/detect/nn/_checkpoint_manager.py tests/unit/detect/nn/test_checkpoint_manager.py
git commit -m "feat(licensing): license-acceptance hook in checkpoint manager"
```

---

### Task 3: CLI flags for per-stage GPU resources + fill knobs

**Files:**
- Modify: `src/phenotypic/_cli/_cli_types.py` (new `ExecutionConfig` fields), `src/phenotypic/phenotypicCLI.py` (options + wiring)
- Test: `tests/integration/cli/test_gpu_cli_flags.py` (Create)

- [ ] **Step 1: Write the failing test**

`tests/integration/cli/test_gpu_cli_flags.py`:

```python
from click.testing import CliRunner
from phenotypic.phenotypicCLI import phenotypic_cli


def test_gpu_flags_parse(monkeypatch):
    runner = CliRunner()
    # --help must list the new options (no run needed)
    result = runner.invoke(phenotypic_cli, ["--help"])
    assert result.exit_code == 0
    for opt in ("--gpu-batch-size", "--gpu-workers-per-gpu", "--gpu-shards", "--gpu-slurm"):
        assert opt in result.output
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/integration/cli/test_gpu_cli_flags.py -v`
Expected: FAIL — options not present.

- [ ] **Step 3: Add the fields + options**

In `_cli_types.py`, add to `ExecutionConfig` (with defaults so existing constructions don't break):

```python
    gpu_batch_size: Union[int, str] = 1  # int, or "auto" (VRAM-probe; effective in Spec 2)
    gpu_workers_per_gpu: int = 1
    gpu_shards: int = 1  # parallel Stage-2 GPU tasks (SLURM-only; ignored locally)
    gpu_slurm_args: Dict[str, Any] = field(default_factory=dict)  # Stage-2 GPU resources (delta on slurm_args)
```

In `phenotypicCLI.py`, add the options near `--slurm`/`--njobs` and thread them into the `ExecutionConfig(...)` construction:

```python
def _parse_gpu_batch_size(ctx, param, value):
    if value == "auto":
        return "auto"
    try:
        return int(value)
    except (TypeError, ValueError):
        raise click.BadParameter("--gpu-batch-size must be an integer or 'auto'")

@click.option("--gpu-batch-size", "gpu_batch_size", default="1", show_default=True,
              callback=_parse_gpu_batch_size,
              help="Images per GPU forward pass (Stage 2). Integer, or 'auto' (VRAM-probe). "
                   "Effective only for batchable detectors; the 'auto' probe lands in Spec 2.")
@click.option("--gpu-workers-per-gpu", "gpu_workers_per_gpu", type=int, default=1, show_default=True,
              help="Model replicas packed per physical GPU (Stage 2) to fill a GPU for small models.")
@click.option("--gpu-shards", "gpu_shards", type=int, default=1, show_default=True,
              help="Parallel Stage-2 GPU tasks (one whole GPU each; SLURM-only, ignored locally). "
                   "Set to your concurrent-GPU count.")
@click.option("--gpu-slurm", "gpu_slurm_args", multiple=True,
              help="GPU-stage (Stage 2) SBATCH resources, e.g. --gpu-slurm slurm_partition=exfab "
                   "--gpu-slurm slurm_account=<acct>. Inherits/deltas over --slurm (the CPU profile "
                   "used by Stages 1&3); auto-adds slurm_gpus_per_node=1.")
```

And in the config build:

```python
            gpu_batch_size=gpu_batch_size,
            gpu_workers_per_gpu=gpu_workers_per_gpu,
            gpu_shards=gpu_shards,
            gpu_slurm_args=parse_slurm_args(gpu_slurm_args),
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/integration/cli/test_gpu_cli_flags.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_types.py src/phenotypic/phenotypicCLI.py tests/integration/cli/test_gpu_cli_flags.py
git commit -m "feat(staged): CLI flags for per-stage GPU resources + fill knobs"
```

---

### Task 4: Shard partitioning + per-stage `slurm_args` resolution

**Files:**
- Create: `src/phenotypic/_cli/_cli_staged_slurm.py` (helpers only, this task)
- Test: `tests/unit/cli/test_staged_slurm_scripts.py` (Create)

- [ ] **Step 1: Write the failing test**

`tests/unit/cli/test_staged_slurm_scripts.py`:

```python
from phenotypic._cli._cli_staged_slurm import partition_shards, resolve_stage_slurm_args


def test_partition_shards_even():
    items = list(range(10))
    shards = partition_shards(items, 3)
    assert [len(s) for s in shards] == [4, 3, 3]
    assert sorted(x for s in shards for x in s) == items  # no loss


def test_partition_shards_more_shards_than_items():
    shards = partition_shards([1, 2], 5)
    assert [len(s) for s in shards if s] == [1, 1]
    assert len([s for s in shards if s]) == 2


def test_gpu_stage_auto_requests_one_gpu():
    args = resolve_stage_slurm_args(gpu_slurm_args={"slurm_partition": "gpu"})
    assert args["slurm_gpus_per_node"] == 1  # auto-added when absent


def test_gpu_stage_respects_explicit_gpu_count():
    args = resolve_stage_slurm_args(
        gpu_slurm_args={"slurm_partition": "gpu", "slurm_gpus_per_node": 2}
    )
    assert args["slurm_gpus_per_node"] == 2


def test_gpu_stage_inherits_shared_keys_and_overrides_partition():
    args = resolve_stage_slurm_args(
        gpu_slurm_args={"slurm_partition": "exfab", "slurm_account": "exfab_acct"},
        cpu_slurm_args={"slurm_partition": "short", "slurm_qos": "normal"},
    )
    assert args["slurm_partition"] == "exfab"      # gpu overrides cpu
    assert args["slurm_account"] == "exfab_acct"   # added for the gpu partition
    assert args["slurm_qos"] == "normal"           # inherited from --slurm
    assert args["slurm_gpus_per_node"] == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_staged_slurm_scripts.py -k "partition_shards or gpu_stage" -v`
Expected: FAIL — module/functions missing.

- [ ] **Step 3: Implement the helpers**

`src/phenotypic/_cli/_cli_staged_slurm.py` (helpers section):

```python
"""SLURM 3-stage chaining for the staged GPU engine (Spec 1 §7)."""

from __future__ import annotations

from typing import Any, Dict, List, TypeVar

_T = TypeVar("_T")


def partition_shards(items: List[_T], n_shards: int) -> List[List[_T]]:
    """Split *items* into up to *n_shards* near-even contiguous shards (no loss)."""
    n = max(1, n_shards)
    k, r = divmod(len(items), n)
    shards: List[List[_T]] = []
    start = 0
    for i in range(n):
        size = k + (1 if i < r else 0)
        shards.append(items[start:start + size])
        start += size
    return shards


def resolve_stage_slurm_args(
    gpu_slurm_args: Dict[str, Any], cpu_slurm_args: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """GPU-stage SBATCH args: inherit/delta over the CPU profile + auto-1-GPU.

    Effective Stage-2 args = {**cpu_slurm_args, **gpu_slurm_args}, then auto-add
    slurm_gpus_per_node=1 if absent. Shared keys (account, qos) set in --slurm carry
    over; a separate GPU partition/account in --gpu-slurm overrides.
    """
    args = {**(cpu_slurm_args or {}), **gpu_slurm_args}
    if "slurm_gpus_per_node" not in args:
        args["slurm_gpus_per_node"] = 1
    return args
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_staged_slurm_scripts.py -k "partition_shards or gpu_stage" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_staged_slurm.py tests/unit/cli/test_staged_slurm_scripts.py
git commit -m "feat(staged): shard partitioning + per-stage GPU slurm_args resolution"
```

---

### Task 5: Stage-2 SLURM shard-worker entrypoint

**Files:**
- Create: `src/phenotypic/_cli/_cli_staged_slurm_worker.py`
- Test: `tests/integration/cli/test_staged_gpu_local.py` (add a shard-worker invocation test reusing the fake detector)

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/cli/test_staged_gpu_local.py`:

```python
def test_stage2_shard_worker_processes_its_shard(tmp_path):
    # Stage 1 first (reuse the core), then run the shard worker over shard 0.
    image_path = _write_image(tmp_path)
    out = tmp_path / "out"; out.mkdir()
    pipe = ImagePipeline(ops=[FakeGpuDetector(threshold=0.3)])
    pipe_path = out / "pipeline.json"
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    om = OutputManager.from_config(out, save_overlays=False)
    om.create_dataset_directories([Dataset("ds", [image_path], tmp_path, out)])
    stage1_preprocess_core(
        split_pipeline_at_gpu(ImagePipeline.from_json(pipe_path)),
        image_path, "ds", "img", out, om, image_type="Image",
    )

    from phenotypic._cli._cli_staged_slurm_worker import run_stage2_shard
    run_stage2_shard(
        pipeline_path=pipe_path, output_dir=out, image_type="Image",
        manifest=[("ds", "img")], shard_index=0, n_shards=1,
    )
    assert sidecar_exists(out, "ds", "img")
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/integration/cli/test_staged_gpu_local.py -k shard_worker -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the shard worker**

`src/phenotypic/_cli/_cli_staged_slurm_worker.py`:

```python
"""Stage-2 shard-worker: load the model once, stream a shard of HDFs to sidecars.

Invoked per SLURM array task as the Stage-2 body. Content-defined skip means a
requeued/duplicate task is idempotent.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from phenotypic import ImagePipeline
from phenotypic.tools_.typing_ import ImageTypeName
from ._cli_pipeline_split import split_pipeline_at_gpu
from ._cli_sidecar import sidecar_exists
from ._cli_staged_slurm import partition_shards
from ._cli_staged_workers import stage2_detect_core


def run_stage2_shard(
    pipeline_path: Path,
    output_dir: Path,
    image_type: ImageTypeName,
    manifest: List[Tuple[str, str]],   # ordered (dataset, image_stem) for ALL images
    shard_index: int,
    n_shards: int,
) -> None:
    """Process this task's shard: model loaded once, stream HDFs -> sidecars."""
    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipeline_path))
    plan.gpu_detector._ensure_model_loaded()  # ONCE per worker
    my_shard = partition_shards(manifest, n_shards)[shard_index]
    for dataset, stem in my_shard:
        if sidecar_exists(output_dir, dataset, stem):
            continue  # content-defined resume
        stage2_detect_core(plan.gpu_detector, output_dir, dataset, stem, image_type)
```

Add a `__main__`-style argv entry so the SBATCH script can call
`python -m phenotypic._cli._cli_staged_slurm_worker <pipeline> <out> <image_type> <manifest_json> <shard_index> <n_shards>` (parse argv, load the manifest JSON written by the strategy, call `run_stage2_shard`).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/integration/cli/test_staged_gpu_local.py -k shard_worker -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_staged_slurm_worker.py tests/integration/cli/test_staged_gpu_local.py
git commit -m "feat(staged): Stage-2 SLURM shard-worker (resident model streams a shard)"
```

---

### Task 6: `StagedSlurmStrategy` — 3 script sets + `afterany` chain

> **This is the "real surgery" the spec flags (S1).** Read `generate_array_job_script` (`_cli_slurm_array_scripts.py:122`) and `generate_sbatch_directives` (`tools_/slurm/_sbatch.py:27`) before implementing — you will mirror their structure to emit per-stage scripts with per-stage `slurm_args`.
>
> **Chain-of-chains decomposition (recommended structure — minimizes the surgery).** Do
> not rewrite the array generator; *wrap* it. `generate_all_array_job_scripts` already
> turns a work-list into a within-stage **chunk chain** (`afterok`-linked chunks when the
> list exceeds the SLURM array-size limit). Reuse it **once per stage**, with that stage's
> resources and array dimension:
>
> - **Stage 1** → chunked array over **images** (CPU `slurm_args`).
> - **Stage 2** → chunked array over **shards** (GPU `slurm_args`; usually a single array
>   since `n_shards` is small).
> - **Stage 3** → chunked array over **images** (CPU `slurm_args`).
>
> Then add a thin **stage layer** on top that submits the three (each itself possibly a
> chunk chain) and wires `--dependency=afterany:<last_job_of_prev_stage>` between stages.
> The result is a **chain of chains**: `afterok` between chunks *inside* a stage
> (array-limit overflow), `afterany` *between* stages (so per-image failures never block
> the next stage). New code = the stage layer + per-stage resource/array-dimension
> selection; the within-stage chunking is reused unchanged.

**Files:**
- Modify: `src/phenotypic/_cli/_cli_staged_slurm.py` (add `StagedSlurmStrategy` + script generation)
- Test: `tests/unit/cli/test_staged_slurm_scripts.py`

- [ ] **Step 1: Write the failing test (script content + deps, NO submission)**

Append:

```python
from phenotypic._cli._cli_staged_slurm import generate_staged_scripts


def test_generates_three_stage_scripts_with_correct_resources(tmp_path):
    scripts = generate_staged_scripts(
        pipeline_path=tmp_path / "p.json",
        datasets_manifest=[("ds", "a"), ("ds", "b"), ("ds", "c")],
        output_dir=tmp_path,
        image_type="Image",
        cpu_slurm_args={"slurm_partition": "batch", "slurm_cpus_per_task": 4},
        gpu_slurm_args={"slurm_partition": "gpu"},
        n_shards=2,
    )
    assert set(scripts) == {"stage1", "stage2", "stage3"}
    s1 = scripts["stage1"].read_text(encoding="utf-8")
    s2 = scripts["stage2"].read_text(encoding="utf-8")
    s3 = scripts["stage3"].read_text(encoding="utf-8")

    # Stage 1 & 3 on the CPU partition; Stage 2 on the GPU partition + 1 GPU
    assert "--partition=batch" in s1 and "--partition=batch" in s3
    assert "--partition=gpu" in s2 and "--gpus-per-node=1" in s2
    # Stage 1/3 = array over images (0-2); Stage 2 = array over shards (0-1)
    assert "--array=0-2" in s1 and "--array=0-2" in s3
    assert "--array=0-1" in s2
    # Stage 2 invokes the shard worker; Stage 3 invokes the merge worker
    assert "_cli_staged_slurm_worker" in s2


def test_chain_uses_afterany_dependencies():
    # generate_staged_scripts records the intended dependency type for the chain
    from phenotypic._cli._cli_staged_slurm import STAGE_DEPENDENCY
    assert STAGE_DEPENDENCY == "afterany"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_staged_slurm_scripts.py -k "three_stage or afterany" -v`
Expected: FAIL — `generate_staged_scripts` / `STAGE_DEPENDENCY` missing.

- [ ] **Step 3: Implement the generator + strategy**

In `_cli_staged_slurm.py` add `STAGE_DEPENDENCY = "afterany"` and `generate_staged_scripts(...)`. Each stage script is a standard `#SBATCH` header (built with `generate_sbatch_directives` from that stage's `slurm_args`) + an `#SBATCH --array=0-<N-1>` line + the per-task body:

- **Stage 1 / Stage 3 bodies** mirror the existing per-image worker call (`python -m phenotypic._cli._cli_process_single …`) but pointed at the staged Stage-1 / Stage-3 cores — add a `--stage {1|3}` argument to the single-image worker, or a thin `_cli_staged_step.py` entrypoint that dispatches to `stage1_preprocess_core` / `stage3_merge_measure_core` by array index into the image manifest.
- **Stage 2 body** invokes `python -m phenotypic._cli._cli_staged_slurm_worker <pipeline> <out> <image_type> <manifest.json> $SLURM_ARRAY_TASK_ID <n_shards>` (array size = `n_shards`).

`StagedSlurmStrategy.execute` then: writes the manifest JSON, calls `generate_staged_scripts`, and submits the three scripts with `--dependency=afterany:<prev_job_id>` between stages (CPU args for Stages 1&3 via `cpu_slurm_args = config.slurm_args`; GPU args for Stage 2 via `resolve_stage_slurm_args(config.gpu_slurm_args, config.slurm_args)` (inherit/delta over the CPU profile); `n_shards = config.gpu_shards` (default 1)). Reuse `submit_slurm_script_chain`'s `sbatch` invocation pattern, but submit exactly three links with explicit `afterany` deps.

Write the actual `sbatch` submission behind a small injectable runner (default `subprocess.run`) so the test in Step 1 exercises generation without submitting.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_staged_slurm_scripts.py -v`
Expected: PASS (generation + directives + deps; no live `sbatch`).

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_staged_slurm.py tests/unit/cli/test_staged_slurm_scripts.py
git commit -m "feat(staged): StagedSlurmStrategy — 3 per-stage script sets + afterany chain"
```

---

### Task 7: Stage-2 walltime auto-resume (SIGTERM handler + self-resubmit)

**Files:**
- Modify: `src/phenotypic/_cli/_cli_staged_slurm.py` (`--signal=B:TERM@<grace>` on the Stage-2 script; `build_stage2_continuation_script`), `src/phenotypic/_cli/_cli_staged_slurm_worker.py` (SIGTERM handler + resubmit-on-stop)
- Test: `tests/unit/cli/test_staged_slurm_scripts.py`, `tests/integration/cli/test_staged_gpu_local.py`

**Why:** SLURM does **not** auto-requeue a `TIMEOUT` job (only preemption/node-failure with `--requeue`). The durable half is already in place — each image's sidecar is written atomically and the worker skips on content — so no work is lost on a kill. This task adds the **trigger**: the Stage-2 worker catches the pre-walltime `SIGTERM` and `sbatch`-resubmits a continuation for *its* shard (afterany on itself). Content-defined skip means the continuation re-runs only the remaining (sidecar-less) images, and it repeats until the shard is complete.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/cli/test_staged_slurm_scripts.py
def test_stage2_script_carries_signal_directive(tmp_path):
    from phenotypic._cli._cli_staged_slurm import generate_staged_scripts
    scripts = generate_staged_scripts(
        pipeline_path=tmp_path / "p.json",
        datasets_manifest=[("ds", "a"), ("ds", "b")],
        output_dir=tmp_path, image_type="Image",
        cpu_slurm_args={"slurm_partition": "batch"},
        gpu_slurm_args={"slurm_partition": "gpu"},
        n_shards=1, signal_grace=120,
    )
    assert "--signal=B:TERM@120" in scripts["stage2"].read_text(encoding="utf-8")
```

```python
# tests/integration/cli/test_staged_gpu_local.py
def test_shard_worker_resubmits_remaining_on_sigterm(tmp_path, monkeypatch):
    import phenotypic._cli._cli_staged_slurm_worker as W
    image_path = _write_image(tmp_path)
    out = tmp_path / "out"; out.mkdir()
    pipe = ImagePipeline(ops=[FakeGpuDetector(threshold=0.3)])
    pipe_path = out / "pipeline.json"
    pipe_path.write_text(pipe.to_json(), encoding="utf-8")
    om = OutputManager.from_config(out, save_overlays=False)
    om.create_dataset_directories([Dataset("ds", [image_path], tmp_path, out)])
    stage1_preprocess_core(
        split_pipeline_at_gpu(ImagePipeline.from_json(pipe_path)),
        image_path, "ds", "img", out, om, image_type="Image",
    )
    submitted = []
    monkeypatch.setattr(W, "resubmit_stage2_continuation",
                        lambda **kw: submitted.append(kw))
    monkeypatch.setattr(W, "_should_stop", lambda: True)  # simulate SIGTERM before work
    W.run_stage2_shard(pipeline_path=pipe_path, output_dir=out, image_type="Image",
                       manifest=[("ds", "img")], shard_index=0, n_shards=1)
    assert submitted and submitted[0]["shard_index"] == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_staged_slurm_scripts.py -k signal_directive tests/integration/cli/test_staged_gpu_local.py -k resubmits_remaining -v`
Expected: FAIL — signal directive / resubmit hook missing.

- [ ] **Step 3: Implement the handler + resubmit**

In `_cli_staged_slurm_worker.py`:

```python
import os
import signal as _signal

_STOP = False


def _install_sigterm_handler() -> None:
    def _handler(signum, frame):
        global _STOP
        _STOP = True
    _signal.signal(_signal.SIGTERM, _handler)


def _should_stop() -> bool:
    return _STOP


def resubmit_stage2_continuation(*, pipeline_path, output_dir, image_type,
                                 manifest, shard_index, n_shards, runner=None) -> None:
    """Submit a 1-shard continuation (afterany on this job) for the remainder."""
    import subprocess
    from ._cli_staged_slurm import build_stage2_continuation_script
    script = build_stage2_continuation_script(
        pipeline_path, output_dir, image_type, manifest, shard_index, n_shards)
    runner = runner or (lambda s: subprocess.run(
        ["sbatch", f"--dependency=afterany:{os.environ.get('SLURM_JOB_ID', '')}", str(s)],
        check=True))
    runner(script)
```

Update `run_stage2_shard` to install the handler, check `_should_stop()` between images, and resubmit only when it stopped on the signal with remaining work:

```python
def run_stage2_shard(pipeline_path, output_dir, image_type, manifest,
                     shard_index, n_shards):
    _install_sigterm_handler()
    plan = split_pipeline_at_gpu(ImagePipeline.from_json(pipeline_path))
    plan.gpu_detector._ensure_model_loaded()  # ONCE per worker
    my_shard = partition_shards(manifest, n_shards)[shard_index]
    for dataset, stem in my_shard:
        if _should_stop():
            break
        if sidecar_exists(output_dir, dataset, stem):
            continue
        stage2_detect_core(plan.gpu_detector, output_dir, dataset, stem, image_type)
    if _should_stop():  # pre-walltime SIGTERM -> auto-resume the remainder
        remaining = [(d, s) for d, s in my_shard
                     if not sidecar_exists(output_dir, d, s)]
        if remaining:
            resubmit_stage2_continuation(
                pipeline_path=pipeline_path, output_dir=output_dir,
                image_type=image_type, manifest=manifest,
                shard_index=shard_index, n_shards=n_shards)
```

(Guarding on `_should_stop()` — not merely "remaining sidecars missing" — prevents an infinite resubmit loop when an image fails deterministically; per-image failures are logged to `failures.jsonl`, not retried forever.)

In `_cli_staged_slurm.py`: add `signal_grace: int = 120` to `generate_staged_scripts` so the Stage-2 directives include `#SBATCH --signal=B:TERM@{signal_grace}`, and add `build_stage2_continuation_script(...)` (a single-task Stage-2 script that runs `run_stage2_shard` for one shard index).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/cli/test_staged_slurm_scripts.py -k signal_directive tests/integration/cli/test_staged_gpu_local.py -k resubmits_remaining -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/phenotypic/_cli/_cli_staged_slurm.py src/phenotypic/_cli/_cli_staged_slurm_worker.py tests
git commit -m "feat(staged): Stage-2 walltime auto-resume (SIGTERM handler + shard re-submit)"
```

---

### Task 8: Route SLURM GPU runs to `StagedSlurmStrategy` + regression

**Files:**
- Modify: `src/phenotypic/_cli/_cli_execution_strategies.py`
- Test: `tests/unit/cli/test_staged_routing.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/cli/test_staged_routing.py`:

```python
def test_gpu_slurm_routes_to_staged_slurm(tmp_path):
    from phenotypic._cli._cli_staged_slurm import StagedSlurmStrategy
    pipe_path = _write_pipe(tmp_path, ImagePipeline(ops=[FakeGpuDetector()]))
    cfg = _minimal_slurm_config(pipe_path, tmp_path)  # slurm_args={"slurm_partition": "batch"}
    om = OutputManager.from_config(tmp_path, save_overlays=False)
    assert isinstance(create_execution_strategy(cfg, om), StagedSlurmStrategy)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/cli/test_staged_routing.py -k gpu_slurm -v`
Expected: FAIL — still returns `AutonomousSLURMStrategy`.

- [ ] **Step 3: Extend the factory**

In `create_execution_strategy`, before the existing SLURM branch:

```python
    if config.is_slurm_mode():
        from ._cli_validation import pipeline_requires_gpu
        from ._cli_staged_slurm import StagedSlurmStrategy
        if not config.measure_only and pipeline_requires_gpu(config.pipeline_json):
            return StagedSlurmStrategy(config, output_manager)
        return AutonomousSLURMStrategy(config, output_manager)
```

- [ ] **Step 4: Run to verify + full regression**

Run:
```bash
uv run pytest tests/unit/cli tests/integration/cli tests/unit/detect/nn tests/unit/test_licensing_scaffolding.py -v
uv run mypy src/phenotypic/_cli/_cli_staged_slurm.py src/phenotypic/_cli/_cli_staged_slurm_worker.py
uv run ruff check --fix src/phenotypic/_cli tests
```
Expected: PASS / clean. SLURM tests assert generated-script content only (no live submission).

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(staged): route SLURM GPU runs to StagedSlurmStrategy + regression" --allow-empty
```

---

### Task 9: Update CLAUDE.md + how-to + helper texts for SLURM staging + GPU flags

**Files:**
- Modify: `CLAUDE.md` (the `--gpu-*` flags, per-stage resources, walltime auto-resume)
- Modify: `docs/source/how_to/pages/gpu_detection_setup.md` (SLURM staged workflow + env vars + 3-level nesting + license acceptance)
- Test: `tests/unit/test_docs_staged_cli.py` (extend)

- [ ] **Step 1: Write the failing content check**

Append to `tests/unit/test_docs_staged_cli.py`:

```python
def test_claude_md_documents_gpu_flags():
    txt = (REPO / "CLAUDE.md").read_text(encoding="utf-8")
    for flag in ("--gpu-slurm", "--gpu-shards", "--gpu-workers-per-gpu", "--gpu-batch-size"):
        assert flag in txt
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_docs_staged_cli.py -k gpu_flags -v`
Expected: FAIL.

- [ ] **Step 3: Update the docs + verify helper texts**

In `CLAUDE.md`, document the GPU staging flags (mirroring Spec 1 §10):

> - `--gpu-slurm key=value` — GPU-stage SBATCH profile; **inherits/deltas over `--slurm`**
>   (set a separate GPU partition/account here). Stages 1 & 3 use `--slurm`.
> - `--gpu-shards N` (default 1) — parallel whole-GPU Stage-2 tasks (SLURM-only).
> - `--gpu-workers-per-gpu W` (default 1) — replicas packed per GPU (small-model fill).
> - `--gpu-batch-size N|auto` (default 1) — images/forward (batchable models; `auto` in Spec 2).
> Stage 2 survives walltime: each image's sidecar write is atomic and the worker
> SIGTERM-resubmits its shard, so a timeout never loses work.

In `docs/source/how_to/pages/gpu_detection_setup.md`, add a **"SLURM staged GPU
detection"** section: the 3-link `afterany` chain, per-stage CPU/GPU partitions, the
3-level nesting (`--gpu-shards` → `--gpu-workers-per-gpu` → `--gpu-batch-size`), the
`HF_HUB_OFFLINE` pre-staging env vars, and the walltime auto-resume.

Confirm the **CLI `--help`** strings added in Task 3 match this documentation (that
help text *is* the in-CLI helper text — keep them consistent).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/test_docs_staged_cli.py -k gpu_flags -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md docs/source/how_to/pages/gpu_detection_setup.md tests/unit/test_docs_staged_cli.py
git commit -m "docs(staged): document SLURM staged chaining + GPU flags (CLAUDE.md + how-to)"
```

---

## Self-Review

**Spec coverage:** per-stage CPU/GPU resources + 3-link chain (§7, D6) → Tasks 3–8; `afterany` content-defined chaining (§7, §9) → Task 6; **Stage-2 walltime auto-resume** (§9) → Task 7; shard-workers + auto-1-GPU (§7, D5/D7) → Tasks 4–6; licensing scaffolding (§12) → Tasks 1–2. ✓
**Sidecar synergy:** because Stage 2 writes a `.npy` (HDF read-only), there is **no HDF5 write-locking on the GPU nodes** — the `HDF5_USE_FILE_LOCKING` concern from earlier drafts is moot; Stages 1 & 3 write HDF atomically (temp+rename, already in `save_image_hdf`) on CPU nodes. Note this in the gpu-setup how-to.
**"Real surgery" honesty (Task 6):** the per-stage script generation mirrors existing `generate_array_job_script` / `generate_sbatch_directives` — the executing engineer must read those before implementing, and the tests assert generated content (not live `sbatch`).
**Cross-plan consistency:** reuses Plan 2's `split_pipeline_at_gpu`, `stage1/stage2/stage3` cores, `sidecar_exists`; `gpu_shards`/`gpu_slurm_args` fields added in Task 3 match their use in Tasks 4–7.

---

## Execution Handoff

All three Spec-1 plans drafted (`plans/2026-06-16-plan-{1,2,3}-*.md`). Implement in order (1 → 2 → 3). Spec 2 (the four models) gets its own plan after Spec 1 lands.
