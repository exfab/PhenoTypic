"""The CLI writes UTF-8 even when its stdout's locale encoding cannot.

On Windows a redirected stdout (a pipe or a file, as the GUI run console
uses) takes the locale code page, typically cp1252, and the first ``✓``
status line raised ``UnicodeEncodeError`` before any image ran. Setting
``PYTHONIOENCODING=cp1252`` reproduces that stream on any host, so this test
runs a real subprocess with it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile

from phenotypic import ImagePipeline
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSize

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_a_cp1252_stdout_does_not_stop_the_cli(tmp_path: Path) -> None:
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_text(
        ImagePipeline(ops={"d": OtsuDetector()}, meas={"s": MeasureSize()}).to_json(),
        encoding="utf-8",
    )
    (tmp_path / "in" / "plate1").mkdir(parents=True)
    tifffile.imwrite(tmp_path / "in" / "plate1" / "img001.tiff", np.zeros((16, 16, 3), np.uint8))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), *filter(None, [env.get("PYTHONPATH")])])
    env["PYTHONIOENCODING"] = "cp1252"
    env.pop("PYTHONUTF8", None)

    done = subprocess.run(
        [sys.executable, "-m", "phenotypic", "--pipeline", str(pipeline),
         "--input", str(tmp_path / "in"), "--output", str(tmp_path / "out"),
         "--image-type", "Image", "--dry-run"],
        capture_output=True, env=env, cwd=REPO_ROOT, timeout=300,
    )

    stdout = done.stdout.decode("utf-8")
    assert done.returncode == 0, stdout + done.stderr.decode("utf-8", errors="replace")
    assert "✓ Pipeline loaded successfully" in stdout
