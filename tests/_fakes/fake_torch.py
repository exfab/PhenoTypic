"""A stand-in ``torch`` module exposing only the accelerator probes.

``resolve_device`` imports ``torch`` inside the function and asks it three
questions (``cuda``, ``mps``, ``xpu``); HPU is probed through
``habana_frameworks``, which is absent here and reads as unavailable. Putting
this module in ``sys.modules`` lets a test decide which accelerators "exist"
without PyTorch installed and independently of the host's real hardware.
"""

from __future__ import annotations

import sys
import types

import pytest


def install_fake_torch(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda: bool = False,
    mps: bool = False,
    xpu: bool = False,
) -> types.ModuleType:
    """Install a fake ``torch`` whose accelerator probes return the given flags.

    Args:
        monkeypatch: The test's monkeypatch; the real module (if any) is
            restored when the test ends.
        cuda: What ``torch.cuda.is_available()`` returns.
        mps: What ``torch.backends.mps.is_available()`` returns.
        xpu: What ``torch.xpu.is_available()`` returns.

    Returns:
        The installed fake module.
    """
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: cuda)
    fake.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: mps)
    )
    fake.xpu = types.SimpleNamespace(is_available=lambda: xpu)
    monkeypatch.setitem(sys.modules, "torch", fake)
    # A real habana install would answer the HPU probe; force it absent.
    monkeypatch.setitem(sys.modules, "habana_frameworks", None)
    monkeypatch.setitem(sys.modules, "habana_frameworks.torch", None)
    return fake
