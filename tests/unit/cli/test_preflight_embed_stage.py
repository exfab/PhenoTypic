"""The run preflight's view of a GPU Stage-4 run (#282 spec §4.7, plan P5 T3).

Two warnings, both about efficiency or resources and never correctness:

- ``PF-EMBED-SHARD-AMORTIZE``: too few images per GPU shard to pay off the
  model's cold load (O6, ``shard_amortization.py``);
- ``PF-EMBED-RESOURCES``: the Stage-4 profile's ``--mem`` is below the
  loader's modelled peak (``stage4_memory.py``).

And the cluster checks: a SLURM run that embeds in Stage 4 goes through the
staged controller with or without a ``GpuDetector``, so ``PF-GPU-PARTITION``
and ``PF-SLURM-LIMIT`` check its GPU profile (C14 note 7).

Shared env, no torch: the scheduler is a fake (``_run_scheduler_command``),
and input headers are seeded into the context's memo rather than written as
4012 x 6016 plates; the header reader's own shape fields are tested on small
real files below.
"""

from __future__ import annotations

import dataclasses
import json
import re
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from phenotypic import ImagePipeline
from phenotypic._cli import _cli_preflight
from phenotypic._cli._cli_input_headers import InputHeader, read_input_header
from phenotypic._cli._cli_preflight import PreflightContext, run_preflight
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureDinoEmbedding
from tests.unit.cli._preflight_support import make_config, make_datasets

REPO = Path(__file__).resolve().parents[3]
LOGIC_SCRIPTS = REPO / "docs/superpowers/logic_validation_scripts/2026-11-12-gpu-embedding-staging"

PINNED = "0123456789abcdef0123456789abcdef01234567"
GENEROUS = {"mem_gb": 256, "slurm_cpus_per_task": 8}


class _Scheduler:
    """``_run_scheduler_command`` stand-in: ``sinfo -p <name>`` prints *gres[name]*."""

    def __init__(self, gres: dict[str, str]):
        self.gres = gres

    def __call__(self, command, *, input=None, timeout, env=None):
        if command[:2] == ["sinfo", "-p"] and command[2] in self.gres:
            return subprocess.CompletedProcess(command, 0, self.gres[command[2]], "")
        return subprocess.CompletedProcess(command, 0, "", "")


def check_embed_shard_amortization(context: PreflightContext) -> list:
    return _cli_preflight.check_embed_shard_amortization(context)


def check_embed_loader_memory(context: PreflightContext) -> list:
    return _cli_preflight.check_embed_loader_memory(context)


def _embedder(**fields: Any) -> MeasureDinoEmbedding:
    return MeasureDinoEmbedding(**{"revision": PINNED, **fields})


def _context(
    tmp_path: Path,
    *,
    n_images: int,
    embed_stage: str,
    gpu_shards: int = 1,
    gpu_slurm: dict[str, Any] | None = None,
    header_hw: tuple[int, int] = (4012, 6016),
    bit_depth: int | None = 8,
    embedder: MeasureDinoEmbedding | None = None,
    meas: dict[str, MeasureDinoEmbedding] | None = None,
    slurm: bool = True,
) -> PreflightContext:
    pipeline = ImagePipeline(
        ops={"det": OtsuDetector()}, meas=meas or {"dino": embedder or _embedder()}
    )
    images = [tmp_path / "in" / f"plate{i:04d}.tif" for i in range(n_images)]
    config = make_config(
        output_dir=tmp_path / "out",
        force_local=not slurm,
        slurm_args={"slurm_partition": "short", "slurm_time": "01:00:00"} if slurm else {},
        gpu_slurm_args=dict(GENEROUS if gpu_slurm is None else gpu_slurm),
        gpu_shards=gpu_shards,
        embed_stage_request=embed_stage,
        embed_stage=embed_stage,
    )
    context = PreflightContext(
        config=config, pipeline=pipeline, datasets=make_datasets(*images), mode="full"
    )
    height, width = header_hw
    context.scratch["headers"] = [
        InputHeader(str(path), channels=3, bits=bit_depth, height=height, width=width)
        for path in images
    ]
    return context


def preflight_for(tmp_path: Path, *, sinfo_gres: dict[str, str] | None = None, **kwargs: Any) -> list:
    """Every finding of the whole run preflight over a SLURM Stage-4-capable run.

    Asserts that no check crashed, so a check that dereferences a field the
    context lacks cannot hide as a missing finding.
    """
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(_cli_preflight, "_run_scheduler_command", _Scheduler(sinfo_gres or {}))
        monkeypatch.setattr(_cli_preflight, "_scheduler_available", lambda name: True)
        findings = list(run_preflight(_context(tmp_path, **kwargs)).findings)
    finally:
        monkeypatch.undo()
    crashed = [f.message for f in findings if f.code == "PF-CHECK-CRASHED"]
    assert not crashed, crashed
    return findings


def codes(findings: list) -> set[str]:
    return {f.code for f in findings}


def needed_gb(findings: list) -> float:
    (finding,) = [f for f in findings if f.code == "PF-EMBED-RESOURCES"]
    return float(re.search(r"needs about ([\d.]+) GiB", finding.message).group(1))


def _load_script(name: str):
    spec = spec_from_file_location(name.removesuffix(".py"), LOGIC_SCRIPTS / name)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- PF-EMBED-SHARD-AMORTIZE (O6) --------------------------------------------------------


def test_too_few_images_per_shard_warns(tmp_path):
    findings = preflight_for(tmp_path, n_images=40, gpu_shards=4, embed_stage="gpu")
    (f,) = [f for f in findings if f.code == "PF-EMBED-SHARD-AMORTIZE"]
    assert f.severity == "warning" and "10 per shard" in f.message and "correct either way" in f.message
    assert "439" in f.message and "750 s" in f.message
    assert "measured for base/large; small and giant use the nearest measured size" in f.message


def test_enough_images_per_shard_is_silent(tmp_path):
    assert "PF-EMBED-SHARD-AMORTIZE" not in codes(preflight_for(tmp_path, n_images=1000, gpu_shards=2,
                                                                embed_stage="gpu"))      # 500 >= 439


def test_the_threshold_is_the_largest_backbones(tmp_path):
    """300 per shard is below base's 439 and above large's 297: the large backbone decides."""
    base_only = _context(tmp_path, n_images=600, gpu_shards=2, embed_stage="gpu")
    assert [f.code for f in check_embed_shard_amortization(base_only)] == ["PF-EMBED-SHARD-AMORTIZE"]

    with_large = _context(tmp_path, n_images=600, gpu_shards=2, embed_stage="gpu",
                          meas={"dino": _embedder(), "dino_l": _embedder(size="large")})
    assert check_embed_shard_amortization(with_large) == []


@pytest.mark.parametrize(("size", "threshold"), [("small", 439), ("base", 439), ("large", 297), ("giant", 297)])
def test_each_size_takes_its_nearest_measured_threshold(tmp_path, size, threshold):
    def fires(n_images: int) -> bool:
        context = _context(tmp_path, n_images=n_images, embed_stage="gpu", embedder=_embedder(size=size))
        return bool(check_embed_shard_amortization(context))

    assert fires(threshold - 1) and not fires(threshold)


def test_a_local_gpu_run_gets_no_shard_warning(tmp_path):
    """``--gpu-shards`` is SLURM-only; a local Stage 4 loads the model once."""
    context = _context(tmp_path, n_images=4, embed_stage="gpu", slurm=False)
    assert check_embed_shard_amortization(context) == []


def test_the_amortisation_constants_are_the_scripts():
    script = _load_script("shard_amortization.py")
    assert _cli_preflight._COLD_LOAD_S == script.COLD_LOAD_S
    assert _cli_preflight._WARM_GPU_S_PER_IMAGE == script.WARM_GPU_S_PER_IMAGE
    for size in script.WARM_GPU_S_PER_IMAGE:
        assert _cli_preflight._amortize_min_images_per_shard(size) == script.amortize_min_images_per_shard(size)


# --- PF-EMBED-RESOURCES ----------------------------------------------------------------


def test_low_mem_for_the_loader_warns(tmp_path):
    findings = preflight_for(tmp_path, n_images=1000, gpu_slurm={"mem_gb": 8, "slurm_cpus_per_task": 8},
                             header_hw=(4012, 6016), embed_stage="gpu")
    assert "PF-EMBED-RESOURCES" in codes(findings)
    (f,) = [f for f in findings if f.code == "PF-EMBED-RESOURCES"]
    assert f.severity == "warning" and "8.0 GiB" in f.message


def test_enough_mem_for_the_loader_is_silent(tmp_path):
    findings = preflight_for(tmp_path, n_images=1000, gpu_slurm={"mem_gb": 48, "slurm_cpus_per_task": 8},
                             embed_stage="gpu")
    assert "PF-EMBED-RESOURCES" not in codes(findings)


def test_a_uint16_plate_needs_more(tmp_path):
    low = {"mem_gb": 1, "slurm_cpus_per_task": 8}
    need8 = needed_gb(preflight_for(tmp_path, n_images=1000, gpu_slurm=low, header_hw=(4012, 6016), bit_depth=8,
                                    embed_stage="gpu"))
    need16 = needed_gb(preflight_for(tmp_path, n_images=1000, gpu_slurm=low, header_hw=(4012, 6016), bit_depth=16,
                                     embed_stage="gpu"))
    assert need16 > need8


def test_the_needed_bytes_are_the_memory_scripts(tmp_path):
    """The script's own case: 8 CPUs, 4012 x 6016 uint8, its crop bytes."""
    script = _load_script("stage4_memory.py")
    assert _cli_preflight._LOADER_BASE_BYTES == script.LOADER_BASE_BYTES
    assert _cli_preflight._MAX_IN_FLIGHT_OUTPUT_BYTES == script.MAX_IN_FLIGHT_OUTPUT_BYTES
    crops = 66 * 3 * 224 * 224 * 4
    for sample_bytes in (1, 2):
        assert _cli_preflight._loader_per_worker(4012, 6016, sample_bytes, crops) == script.loader_per_worker(
            4012, 6016, sample_bytes, crops)


def test_masked_dense_adds_the_in_flight_bound(tmp_path):
    low = {"mem_gb": 1, "slurm_cpus_per_task": 1}
    crop = needed_gb(preflight_for(tmp_path, n_images=10, gpu_slurm=low, embed_stage="gpu"))
    dense = needed_gb(preflight_for(tmp_path, n_images=10, gpu_slurm=low, embed_stage="gpu",
                                    embedder=_embedder(pooling="masked_dense")))
    assert dense - crop == pytest.approx(_cli_preflight._MAX_IN_FLIGHT_OUTPUT_BYTES / 2**30, abs=0.1)


def test_fewer_loader_workers_need_less(tmp_path):
    def need(cpus: int) -> float:
        return needed_gb(preflight_for(tmp_path, n_images=10, gpu_slurm={"mem_gb": 1, "slurm_cpus_per_task": cpus},
                                       embed_stage="gpu"))

    assert need(2) < need(8)


def test_mem_per_cpu_counts_for_every_cpu(tmp_path):
    enough = {"slurm_mem_per_cpu": "8G", "slurm_cpus_per_task": 8}      # 64 GiB in all
    assert check_embed_loader_memory(_context(tmp_path, n_images=10, embed_stage="gpu", gpu_slurm=enough)) == []
    short = {"slurm_mem_per_cpu": "1G", "slurm_cpus_per_task": 8}       # 8 GiB in all
    assert check_embed_loader_memory(_context(tmp_path, n_images=10, embed_stage="gpu", gpu_slurm=short))


def test_no_mem_request_or_no_shape_is_skipped_visibly(tmp_path, caplog):
    """DefMemPerCPU is not read, and a header without a shape is not assumed; each skip logs why."""
    caplog.set_level("DEBUG", logger=_cli_preflight.logger.name)
    unset = {"slurm_cpus_per_task": 8}
    assert check_embed_loader_memory(_context(tmp_path, n_images=10, embed_stage="gpu", gpu_slurm=unset)) == []
    assert "neither --mem nor --mem-per-cpu" in caplog.text
    context = _context(tmp_path, n_images=10, embed_stage="gpu", gpu_slurm={"mem_gb": 1})
    context.scratch["headers"] = [InputHeader(h.path, channels=3, bits=8) for h in context.scratch["headers"]]
    assert check_embed_loader_memory(context) == []
    assert "no input header gives a plate shape" in caplog.text


def test_a_given_grid_shape_bounds_the_crops(tmp_path):
    """``--nrows/--ncols`` replace the 1536-crop bound: a 96-format plate needs less."""
    low = {"mem_gb": 1, "slurm_cpus_per_task": 8}
    bound = check_embed_loader_memory(_context(tmp_path, n_images=10, embed_stage="gpu", gpu_slurm=low))
    plate96 = _context(tmp_path, n_images=10, embed_stage="gpu", gpu_slurm=low)
    plate96 = PreflightContext(
        config=dataclasses.replace(plate96.config, nrows=8, ncols=12), pipeline=plate96.pipeline,
        datasets=plate96.datasets, mode="full", scratch=plate96.scratch,
    )
    assert needed_gb(check_embed_loader_memory(plate96)) < needed_gb(bound)


def test_inline_runs_get_neither(tmp_path):
    assert not codes(preflight_for(tmp_path, n_images=4, gpu_shards=4, embed_stage="inline",
                                   gpu_slurm={"mem_gb": 1, "slurm_cpus_per_task": 8})) & {
        "PF-EMBED-SHARD-AMORTIZE", "PF-EMBED-RESOURCES"}


# --- the staged SLURM checks reach a no-detector Stage-4 run (C14 note 7) ---------------


def test_a_no_detector_gpu_embed_run_checks_the_gpu_partition(tmp_path):
    findings = preflight_for(tmp_path, n_images=1000, gpu_slurm={"slurm_partition": "intel"},
                             embed_stage="gpu", sinfo_gres={"intel": "(null)\n"})
    assert "PF-GPU-PARTITION" in codes(findings)


def test_an_inline_no_detector_run_checks_no_gpu_partition(tmp_path):
    findings = preflight_for(tmp_path, n_images=1000, gpu_slurm={"slurm_partition": "intel"},
                             embed_stage="inline", sinfo_gres={"intel": "(null)\n"})
    assert "PF-GPU-PARTITION" not in codes(findings)


def test_a_no_detector_gpu_embed_run_checks_the_staged_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(_cli_preflight, "_slurm_submission_limits", lambda: (2, 1000))
    findings = preflight_for(tmp_path, n_images=1000, embed_stage="gpu")
    assert "PF-SLURM-LIMIT" in codes(findings)


def test_a_dinov3_embedder_is_refused_at_preflight_without_the_licence(tmp_path, monkeypatch):
    monkeypatch.delenv("PHENOTYPIC_ACCEPT_MODEL_LICENSE", raising=False)
    findings = preflight_for(tmp_path, n_images=1000, embed_stage="gpu",
                             embedder=_embedder(version=3, size="base"))
    assert "PF-LICENSE" in codes(findings)       # the existing check reaches a Stage-4 run (spec §4.5; U11: no 7b)


# --- torch-free, and the header's shape ----------------------------------------------------

_SUBPROCESS_PROBE = """
import json, sys
from pathlib import Path
import tests.unit.cli.test_preflight_embed_stage as t
tmp = Path(sys.argv[1])
before = set(sys.modules)
t.preflight_for(tmp, n_images=40, embed_stage="gpu", gpu_slurm={"mem_gb": 1, "slurm_cpus_per_task": 8})
print(json.dumps(sorted(set(sys.modules) - before)))
"""


def modules_loaded_by(tmp_path: Path) -> set[str]:
    """Modules a fresh interpreter imports while the preflight runs (not while it is built)."""
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_PROBE, str(tmp_path)],
        cwd=REPO, capture_output=True, text=True, timeout=300, check=True,
    )
    return set(json.loads(result.stdout.strip().splitlines()[-1]))


def test_the_checks_import_no_torch(tmp_path):
    loaded = modules_loaded_by(tmp_path)
    assert loaded, "the probe saw no import at all, so it could not see torch either"
    assert not {name for name in loaded if name == "torch" or name.startswith("torch.")}


def test_headers_carry_the_plate_shape(tmp_path):
    import tifffile
    from PIL import Image as PILImage

    tifffile.imwrite(tmp_path / "rgb16.tif", np.zeros((40, 60, 3), dtype=np.uint16))
    PILImage.fromarray(np.zeros((30, 50, 3), dtype=np.uint8)).save(tmp_path / "rgb8.png")

    tiff, png = read_input_header(tmp_path / "rgb16.tif"), read_input_header(tmp_path / "rgb8.png")
    assert (tiff.height, tiff.width, tiff.bits) == (40, 60, 16)
    assert (png.height, png.width, png.bits) == (30, 50, 8)


def test_a_channel_first_tiff_reports_its_plane_shape(tmp_path):
    import tifffile

    tifffile.imwrite(tmp_path / "planar.tif", np.zeros((3, 40, 60), dtype=np.uint8), photometric="rgb",
                     planarconfig="separate")
    header = read_input_header(tmp_path / "planar.tif")
    assert (header.channels, header.height, header.width) == (3, 40, 60)
