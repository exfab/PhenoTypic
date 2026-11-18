"""Build one real tables-v2 store with the CLI (no ngio import here)."""
import subprocess, sys
from pathlib import Path
import numpy as np
from phenotypic import ImagePipeline
from phenotypic.data import load_synth_yeast_plate
from phenotypic.detect import OtsuDetector
from phenotypic.measure import MeasureSize

root = Path(sys.argv[1]); inp = root / "in" / "ds"; out = root / "out"
inp.mkdir(parents=True, exist_ok=True)
from PIL import Image as PILImage
img = load_synth_yeast_plate()
PILImage.fromarray(img.rgb[:].astype("uint8")).save(inp / "plate_1.tif")
pipe = root / "pipe.pht-pipe.json"
pipe.write_text(ImagePipeline(ops=[OtsuDetector()], meas={"size": MeasureSize()}).to_json(), encoding="utf-8")
r = subprocess.run([sys.executable, "-m", "phenotypic", "-i", str(root / "in"), "-o", str(out),
                    "--pipeline", str(pipe), "--njobs", "1"], capture_output=True, text=True)
print("cli exit", r.returncode); print(r.stdout[-1500:]); print(r.stderr[-1500:])
for s in out.rglob("*.ome.zarr"):
    print("STORE", s)
