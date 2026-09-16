"""Every ``_infer_batch`` override must call ``_ensure_model_loaded()``.

WHY THIS IS A TEST AND NOT ONLY A DOC. The base ``_infer_batch``
(``abc_/_gpu_detector.py:207``) is where the ``_ensure_model_loaded()`` call
lives, and the contrib guide tells batchable detectors to override that method.
An override that drops the call **passes every CLI test in this repo**: the
staged engine loads the model up front, once per worker
(``_cli_staged_strategy.py``, ``_cli_staged_slurm_worker.py``), so the omission
is invisible there. It fails only in the notebook path, where
``_operate -> _collate -> _infer_batch`` has no other caller.

A bug invisible in the engine and visible only in ``op.apply(image)`` is the
awkward shape: the engine is where the large runs happen and the notebook is
where people develop, so it would survive a full green suite and surface the
first time someone tried their own detector interactively.

Nothing in the tree is broken today -- ``Sam3`` is the only override and it
calls it correctly. This guard passes now and catches the next one, which is
the point: an obligation a reader has to infer from a caller list is an
obligation that gets dropped.

AST rather than runtime: the call cannot be observed without loading a real
model, and a runtime probe that stubs the model out would be asserting about
the stub.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[3] / "src" / "phenotypic"
_REQUIRED = "_ensure_model_loaded"
_METHOD = "_infer_batch"


def _overrides(path: Path) -> list[tuple[str, ast.FunctionDef]]:
    """Every ``(class_name, node)`` defining ``_infer_batch`` in *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        for item in cls.body:
            if (
                isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == _METHOD
            ):
                found.append((cls.name, item))
    return found


def _calls_ensure_model_loaded(node: ast.AST) -> bool:
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == _REQUIRED:
            return True
        if isinstance(func, ast.Name) and func.id == _REQUIRED:
            return True
    return False


def _all_overrides():
    out = []
    for path in sorted(_SRC.rglob("*.py")):
        for cls_name, node in _overrides(path):
            out.append((path, cls_name, node))
    return out


def test_the_scan_finds_the_known_overrides():
    """Premise: the scan is looking at something.

    Without this, a glob that silently matched nothing would make the guard
    below vacuously true -- a check whose success message is identical to its
    no-op message, which is the defect pattern this change kept producing.
    """
    found = {cls for _, cls, _ in _all_overrides()}
    assert "GpuDetector" in found, "the ABC's own definition was not found"
    assert "Sam3" in found, (
        "Sam3's override was not found; either it was removed or the scan is "
        f"broken. Found: {sorted(found)}"
    )


@pytest.mark.parametrize(
    "path,cls_name,node",
    _all_overrides(),
    ids=[f"{cls}" for _, cls, _ in _all_overrides()],
)
def test_every_infer_batch_calls_ensure_model_loaded(path, cls_name, node):
    assert _calls_ensure_model_loaded(node), (
        f"{cls_name}._infer_batch ({path.name}:{node.lineno}) does not call "
        f"{_REQUIRED}(). The base implementation does, and overriding it takes "
        f"that call with it. The staged CLI engine loads the model up front, so "
        f"this omission is INVISIBLE in every CLI test -- it breaks only "
        f"op.apply(image), where _operate -> _collate -> _infer_batch is the "
        f"only path to the model."
    )
