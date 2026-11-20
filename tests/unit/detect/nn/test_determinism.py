"""``configure_determinism``: the settings Stage 4 runs under (#282 P2 T2). [torch]

Each test restores torch's global flags afterwards, so no later test in the
process runs deterministic by accident. No GPU: the flags are settable on the
CPU, and ``device="cuda"`` exercises the cuBLAS check without one.
"""

from __future__ import annotations

import os

import pytest

from phenotypic.detect.nn._helper._batch_encoder import configure_determinism

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _restore_torch_flags():
    saved = (
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    yield
    (deterministic, benchmark, precision, matmul_tf32, cudnn_tf32, algorithms, warn_only) = saved
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    torch.set_float32_matmul_precision(precision)
    torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
    torch.backends.cudnn.allow_tf32 = cudnn_tf32
    torch.use_deterministic_algorithms(algorithms, warn_only=warn_only)


def _assert_deterministic():
    assert torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark
    assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
    assert torch.get_float32_matmul_precision() == "highest"
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.is_deterministic_algorithms_warn_only_enabled()


def test_determinism_settings(monkeypatch):
    # Set, so the test passes on a GPU node too, where the check applies.
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.allow_tf32 = True                    # torch's default
    applied = configure_determinism()
    _assert_deterministic()
    assert applied == {
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": True,
        "cublas_workspace_config": ":4096:8",
    }


def test_it_is_idempotent(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    assert configure_determinism() == configure_determinism()
    _assert_deterministic()


def test_a_users_16_8_is_kept_on_cuda(monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    applied = configure_determinism(device="cuda")
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert applied["cublas_workspace_config"] == ":16:8"
    _assert_deterministic()


@pytest.mark.parametrize("value", [None, ":0:0"])
def test_cuda_without_a_deterministic_workspace_refuses_before_any_flag(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", value)
    torch.backends.cudnn.deterministic = False
    with pytest.raises(RuntimeError, match=r"CUBLAS_WORKSPACE_CONFIG.*:4096:8.*:16:8"):
        configure_determinism(device="cuda")
    assert not torch.backends.cudnn.deterministic             # nothing was applied
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == value  # and nothing was set for the user


def test_the_cpu_needs_no_cublas_workspace(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    applied = configure_determinism(device="cpu")
    assert applied["cublas_workspace_config"] is None
    _assert_deterministic()
