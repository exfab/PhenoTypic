"""Tests for Run Console form discovery and staged-GPU controls."""

from __future__ import annotations

from typing import Any

from phenotypic.gui import _config
from phenotypic.gui.run_console import _form, _ids
from phenotypic.gui.shell._sandbox import SandboxRoot


def test_input_tree_uses_canonical_image_extensions(
    tmp_path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def fake_directory_tree(*args: Any, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(_form, "directory_tree", fake_directory_tree)
    sandbox = SandboxRoot.from_path(tmp_path)

    result = _form.render_input_tree(sandbox)

    assert result is not None
    assert captured["extensions"] is _config.IMAGE_EXTS
    assert captured["select_files"] is False
    assert ".cr3" in _config.IMAGE_EXTS
    assert ".raw" not in _config.IMAGE_EXTS


def test_staged_gpu_form_controls_are_mounted_hidden(tmp_path) -> None:
    form = _form.build_form(SandboxRoot.from_path(tmp_path))
    components = list(_walk_components(form))
    by_id = {
        component.id: component
        for component in components
        if getattr(component, "id", None) is not None
    }

    assert by_id[_ids.RC_STAGED_GPU_SECTION].style == {"display": "none"}
    assert by_id[_ids.RC_INPUT_GPU_SHARDS].value == 1
    assert _ids.RC_INPUT_GPU_SLURM in by_id
    cpu_gpu_labels = [
        component
        for component in components
        if getattr(component, "html_for", None)
        == _ids.RC_INPUT_SLURM_GPUS
    ]
    assert len(cpu_gpu_labels) == 1
    assert cpu_gpu_labels[0].children == "CPU-stage GPUs"


def test_staged_gpu_refusal_alert_is_mounted_closed_outside_the_gpu_section(
    tmp_path,
) -> None:
    """The refusal applies in Local mode too, where the GPU section is hidden.

    Mounted inside that section, the alert would be invisible for exactly the
    Local-mode refusals it exists to explain.
    """
    form = _form.build_form(SandboxRoot.from_path(tmp_path))
    by_id = {
        component.id: component
        for component in _walk_components(form)
        if getattr(component, "id", None) is not None
    }

    alert = by_id[_ids.RC_STAGED_GPU_REFUSAL]
    assert alert.is_open is False
    inside_section = {
        getattr(component, "id", None)
        for component in _walk_components(by_id[_ids.RC_STAGED_GPU_SECTION])
    }
    assert _ids.RC_STAGED_GPU_REFUSAL not in inside_section


def test_metadata_preflight_is_visible_and_defaults_to_omit(tmp_path) -> None:
    form = _form.build_form(SandboxRoot.from_path(tmp_path))
    by_id = {
        component.id: component
        for component in _walk_components(form)
        if getattr(component, "id", None) is not None
    }

    assert _ids.RC_METADATA_PREFLIGHT in by_id
    assert by_id[_ids.RC_METADATA_CHOICE].value == "omit"
    assert by_id[_ids.RC_METADATA_ACKNOWLEDGEMENT].value == []


def _walk_components(component: Any):
    yield component
    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk_components(child)
    elif children is not None:
        yield from _walk_components(children)
