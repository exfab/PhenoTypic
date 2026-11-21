"""The Stage-4 worker: every ``MeasureDinoEmbedding`` of a staged run, on a GPU (#282).

This module starts with the device half (#282 P4 T1), which the CLI also
calls at startup for a local ``--embed-stage gpu``: :func:`stage4_device`
and :func:`check_cuda_usable`. Inside Stage 4 an embedder's device is
``cuda``; a missing or unusable CUDA fails the stage and never falls back to
the CPU (spec §4.4, §4.8). The one escape is the test-only
``PHENOTYPIC_STAGE4_DEVICE=cpu``.

torch is imported inside functions only.
"""

from __future__ import annotations

import os
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from pathlib import Path

    from ._cli_output_manager import OutputManager
    from ._cli_types import Dataset, ExecutionConfig

#: Test-only: ``cpu`` runs Stage 4 on the CPU, so a CPU node can exercise it.
STAGE4_DEVICE_ENV: Final[str] = "PHENOTYPIC_STAGE4_DEVICE"

_ARCH = re.compile(r"^(sm|compute)_(\d+)(\d)[a-z]?$")


class Stage4DeviceError(RuntimeError):
    """Stage 4 has no usable CUDA device: none visible, or no kernels for it."""


def stage4_device() -> str:
    """The device Stage 4 runs on: ``"cuda"``, or ``"cpu"`` under the test escape.

    With ``"cuda"``, :func:`check_cuda_usable` runs first.

    Raises:
        Stage4DeviceError: As :func:`check_cuda_usable`.
    """
    if os.environ.get(STAGE4_DEVICE_ENV) == "cpu":
        return "cpu"
    check_cuda_usable()
    return "cuda"


def check_cuda_usable() -> None:
    """Refuse a process whose torch cannot run on GPU 0.

    Usable means torch is installed, ``torch.cuda.is_available()``, and this
    torch build carries kernels for device 0's compute capability
    (:func:`has_kernels_for`). The last is the P100 case (spec §4.8): such a
    node passes ``is_available()``, and the job would otherwise die at the
    first forward pass, after all its CPU work, with "no kernel image is
    available".

    Raises:
        Stage4DeviceError: Naming what is missing: torch, a CUDA device, or
            kernels for the device (with its name and the build's arch list).
    """
    try:
        import torch
    except ImportError as exc:
        raise Stage4DeviceError(
            "torch is not installed, so no CUDA device is usable (install the "
            "foundation extra)"
        ) from exc
    if not torch.cuda.is_available():
        raise Stage4DeviceError(
            "no CUDA device is usable (torch.cuda.is_available() is False)"
        )
    major, minor = torch.cuda.get_device_capability(0)
    arches = list(torch.cuda.get_arch_list())
    if not has_kernels_for(major, minor, arches):
        name = torch.cuda.get_device_name(0)
        raise Stage4DeviceError(
            f"CUDA device 0 ({name}, sm_{major}{minor}) has no kernels in this "
            f"torch build (built for {' '.join(arches) or 'no architecture'})"
        )


def has_kernels_for(major: int, minor: int, arches: Sequence[str]) -> bool:
    """Whether a torch build for *arches* runs on a ``sm_<major><minor>`` device.

    A ``sm_XY`` binary runs on any device of the same major version and a
    minor version at least ``Y`` (CUDA binary compatibility), so an Ada GPU
    (``sm_89``) runs a build that lists ``sm_86`` and no ``sm_89``, which
    is how the cluster's torch 2.11 + CUDA 13 wheel is built. A
    ``compute_XY`` entry is PTX, which the driver compiles for any device of
    capability ``XY`` or newer. An exact ``sm_<major><minor> in arches`` test
    would refuse every Ada GPU, the ``exfab`` partition's included.

    Args:
        major: Device compute capability, major.
        minor: Device compute capability, minor.
        arches: ``torch.cuda.get_arch_list()``, e.g. ``["sm_86", "sm_90"]``.
    """
    for arch in arches:
        match = _ARCH.match(arch)
        if match is None:
            continue
        kind, arch_major, arch_minor = match[1], int(match[2]), int(match[3])
        if kind == "sm" and arch_major == major and arch_minor <= minor:
            return True
        if kind == "compute" and (arch_major, arch_minor) <= (major, minor):
            return True
    return False


def run_local_stage4(
    config: "ExecutionConfig",
    output_manager: "OutputManager",
    output_dir: "Path",
    datasets: "Sequence[Dataset]",
) -> dict[str, tuple[int, int]]:
    """Run Stage 4 in this process over *datasets*: ``{dataset: (published, failed)}``.

    A placeholder until #282 P4 T3 builds the loop; nothing calls it yet.
    The deferral tests patch it out (``no_stage4``), so their meaning holds
    once T3 wires it in after the Stage-3 loop.

    Raises:
        NotImplementedError: Always, until P4 T3.
    """
    raise NotImplementedError("Stage 4 is built in #282 P4 T3")
