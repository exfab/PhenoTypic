"""Tier 3: the composed hub serves before the builder or any heavy library is loaded."""

from __future__ import annotations

from pathlib import Path

from tests._startup_probe import run_startup_probe

#: What the composed hub loads before any request, with the chain that loads it. Measured
#: during the plan review; a shell-side module-level chain may be listed here with its
#: justification (spec, tier 3). None of these is a deferral target.
HUB_ALLOWED_BEFORE_FIRST_REQUEST: dict[str, str] = {
    "plotly": "dash -> plotly, dash's own import; third-party, cannot be cut",
    "pandas": "_gui/analysis/_callbacks.py:25, via compose_hub's eager analysis import",
    "polars": "_gui/analysis/_callbacks.py:26 and _gui/run_console/_request_safety.py:15",
    "pyarrow": "the pandas/polars parquet stack",
    "scipy": "_gui/_operation_registry.py:18 `from phenotypic import ImagePipeline` -> the image core",
    "skimage": "the same chain as scipy",
    "matplotlib": "matplotlib core, not pyplot; the same chain as scipy",
}

#: The spec's minimum: none of these may load before the first request.
HUB_WATCHED_MODULES: tuple[str, ...] = (
    "bm3d", "colour", "cv2", "h5py", "mahotas", "matplotlib.pyplot", "numba",
)


def test_composed_hub_builds_the_builder_on_its_first_request(tmp_path: Path) -> None:
    report = run_startup_probe(
        "from phenotypic._gui.shell._app import create_app\n"
        "from phenotypic._gui.shell._sandbox import SandboxRoot\n"
        f"sandbox = SandboxRoot.from_path({str(tmp_path)!r})\n"
        "app = create_app(sandbox, start_idle_thread=False, start_slurm_observer=False)\n"
        f"watched = {list(HUB_WATCHED_MODULES)!r}\n"
        "loaded_before = [m for m in watched if m in sys.modules]\n"
        "detect_before = 'phenotypic.detect' in sys.modules\n"
        "response = app.server.test_client().get('/builder/')\n"
        "report = {'dash': 'dash' in sys.modules, 'loaded_before': loaded_before,\n"
        "          'detect_before': detect_before, 'status': response.status_code,\n"
        "          'detect_after': 'phenotypic.detect' in sys.modules}\n"
    )
    assert report["dash"] is True
    assert sorted(set(report["loaded_before"]) - set(HUB_ALLOWED_BEFORE_FIRST_REQUEST)) == []
    assert report["detect_before"] is False
    assert report["status"] == 200
    assert report["detect_after"] is True
