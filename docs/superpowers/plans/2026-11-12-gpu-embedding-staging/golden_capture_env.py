"""Record the inline golden's capture environment and its float16 vectors (#282 P1).

The inline golden (``tests/unit/measure/golden/inline_embeddings_281.json``)
stores hashes only. Its CPU bytes depend on the CPU family and on torch's
thread count (gate-results/c3-282/diag, 2026-11-20): the five stub cases with
a learned layer fail on Cascade Lake and Milan at either commit, and the real
cases need 4 threads taken from ``OMP_NUM_THREADS=4``: an explicit
``torch.set_num_threads(4)`` changes their bytes (probes 32278806 and
32279288 against 32279287), so capture with ``--threads 0``. This script runs **at the capture commit** (faef903f3)
on a Broadwell node, recomputes every golden row exactly as the golden test
does, refuses unless each row's ``vectors_sha256`` equals the committed
golden's, and then writes:

- ``environment.json``: the CPU model name, Slurm node and feature, torch
  version, ``OMP_NUM_THREADS``, ``torch.get_num_threads()`` and CPU capability
  it reproduced the golden in;
- ``vectors.npz``: each row's float16 vectors, keyed ``<case>|<payload key
  with / as |>``, for the golden test's always-on cosine check.

It drives the shipped code, so it lives beside the plan, not under
``logic_validation_scripts/``. Run from the capture tree's root::

    PYTHONPATH=src:. HF_HUB_DISABLE_IMPLICIT_TOKEN=1 OMP_NUM_THREADS=4 \\
        <foundation python> <this file> --threads 0 --rows all --out <dir>

Exits non-zero if any row does not reproduce.

``--cosine`` instead measures, on any CPU, each row's worst ``1 - cosine``
against the committed ``inline_embeddings_281.vectors.npz``: the evidence for
the golden test's always-on ``COSINE_FLOOR``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def cpu_model_name() -> str | None:
    """The first ``model name`` of ``/proc/cpuinfo``, or ``None`` off Linux."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return None
    return None


def slurm_features(node: str) -> str | None:
    """The node's ``AvailableFeatures`` from ``scontrol``, or ``None``."""
    try:
        out = subprocess.run(
            ["scontrol", "show", "node", node], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    for token in out.split():
        if token.startswith("AvailableFeatures="):
            return token.split("=", 1)[1]
    return None


def recompute_rows(which: str) -> dict[str, dict[str, object]]:
    """The golden rows (*which*: ``stub``, ``real`` or ``all``) as ``{case: {payload_key: payload}}``.

    Each group runs as the golden test runs it; ``all`` runs the stub rows first.
    """
    import pytest

    from phenotypic import ImagePipeline
    from phenotypic.data import load_synth_yeast_plate
    from phenotypic.detect import OtsuDetector
    from phenotypic.measure import MeasureDinoEmbedding
    from phenotypic.measure import _measure_dino_embedding as module
    from tests._tiny_run import inline_golden_cases, install_stub_backbone

    plate = OtsuDetector().apply(load_synth_yeast_plate())
    rows: dict[str, dict[str, object]] = {}
    for case in sorted(inline_golden_cases()) if which in ("stub", "all") else ():
        with pytest.MonkeyPatch.context() as monkeypatch:
            install_stub_backbone(
                monkeypatch,
                parametric=True,
                num_register_tokens=4 if case == "registers_mean" else 0,
            )
            rows[case] = inline_golden_cases()[case].embed(plate)
        module._cached_backbone.cache_clear()
    real = {
        "objects_cls": {},
        "objects_mean_masked": {
            "pooling": "crop_mean", "mask_background": True, "crop_pad_frac": 0.5,
        },
        "objects_dense": {"pooling": "masked_dense", "crop_px": 112},
    }
    for case, kwargs in real.items() if which in ("real", "all") else ():
        pipe = ImagePipeline(
            meas={"emb": MeasureDinoEmbedding(size="small", device="cpu", **kwargs)}
        )
        rows[f"real_small_{case}"] = pipe.embed(plate)
    module._cached_backbone.cache_clear()
    return rows


def capture_environment(threads: int, out: Path, golden: Path, which: str) -> int:
    """Reproduce every golden row; write the environment and vectors if all match."""
    import numpy as np
    import torch

    if threads > 0:
        torch.set_num_threads(threads)
    expected = json.loads(golden.read_text(encoding="utf-8"))
    vectors: dict[str, np.ndarray] = {}
    mismatched: list[str] = []
    for case, payloads in recompute_rows(which).items():
        for key, payload in payloads.items():
            stored = np.stack(payload.frame["embedding"].to_list()).astype(np.float16)
            digest = hashlib.sha256(np.ascontiguousarray(stored).tobytes()).hexdigest()
            want = expected[case][key]["vectors_sha256"]
            status = "ok" if digest == want else "MISMATCH"
            print(f"{status} {case} {key} {digest}")
            if digest != want:
                mismatched.append(f"{case}|{key}")
            vectors[f"{case}|{key.replace('/', '|')}"] = stored
    node = os.environ.get("SLURMD_NODENAME") or platform.node().split(".")[0]
    environment = {
        "cpu_model": cpu_model_name(),
        "slurm_node": node,
        "slurm_features": slurm_features(node),
        "torch_version": torch.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch_num_threads": torch.get_num_threads(),
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    print(json.dumps(environment, indent=2))
    if mismatched:
        print(f"refusing to write: {len(mismatched)} rows do not reproduce: {mismatched}")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    (out / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    np.savez_compressed(out / "vectors.npz", **vectors)
    print(f"wrote {out}/environment.json and {out}/vectors.npz ({len(vectors)} rows)")
    return 0


def measure_cosine_distance(golden: Path, which: str) -> int:
    """Print each row's worst per-vector ``1 - cosine`` to the committed vectors."""
    import numpy as np

    stored = np.load(golden.with_name("inline_embeddings_281.vectors.npz"))
    worst_overall = 0.0
    for case, payloads in recompute_rows(which).items():
        for key, payload in payloads.items():
            got = np.stack(payload.frame["embedding"].to_list()).astype(np.float16).astype(np.float64)
            want = stored[f"{case}|{key.replace('/', '|')}"].astype(np.float64)
            norms = np.linalg.norm(got, axis=1) * np.linalg.norm(want, axis=1)
            dots = np.einsum("ij,ij->i", got, want)
            cosine = np.where(norms > 0, dots / np.where(norms > 0, norms, 1), 1.0)
            distance = float(1 - cosine.min())
            identical = bool(np.array_equal(got, want))
            worst_overall = max(worst_overall, distance)
            print(f"cosine {case} {key} worst_1_minus_cos={distance:.3e} identical={identical}")
    print(f"cosine worst overall 1 - cos = {worst_overall:.3e}")
    print(json.dumps({"cpu_model": cpu_model_name(), "omp": os.environ.get("OMP_NUM_THREADS")}))
    return 0


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--threads", type=int, required=True,
        help="torch.set_num_threads value; 0 leaves torch's default (OMP_NUM_THREADS)",
    )
    parser.add_argument("--out", type=Path, default=Path("."))
    parser.add_argument("--rows", choices=("stub", "real", "all"), default="all")
    parser.add_argument("--cosine", action="store_true", help="measure, write nothing")
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("tests/unit/measure/golden/inline_embeddings_281.json"),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_arguments(sys.argv[1:])
    if args.cosine:
        sys.exit(measure_cosine_distance(args.golden, args.rows))
    sys.exit(capture_environment(args.threads, args.out, args.golden, args.rows))
