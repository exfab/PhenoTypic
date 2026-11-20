"""Open a PhenoTypic tables-v2 store with ngio only (never imports phenotypic)."""
import json, sys, traceback
from pathlib import Path
import ngio
print("ngio", getattr(ngio, "__version__", "?"))
store = Path(sys.argv[1])
print("tables/zarr.json:", (store / "tables/zarr.json").read_text())
for run in sorted(p for p in (store / "tables").iterdir() if p.is_dir()):
    print("run", run.name, "->", sorted(q.name for q in run.iterdir()))
    m = run / "measurements" / "zarr.json"
    if m.exists():
        print("  measurements attrs:", json.loads(m.read_text()).get("attributes"))
def attempt(label, fn):
    try:
        print(label, "->", fn())
    except Exception as e:
        print(label, "-> RAISED", type(e).__name__, str(e)[:300])
for target in (store, store / "rgb"):
    def opened(t=target):
        c = ngio.open_ome_zarr_container(str(t))
        out = {"tables": c.list_tables()}
        try: out["labels"] = c.list_labels()
        except Exception as e: out["labels"] = f"RAISED {type(e).__name__}"
        return out
    attempt(f"open_ome_zarr_container({target.name})", opened)
for name in ("open_table", "open_tables_container"):
    fn = getattr(ngio, name, None) or getattr(getattr(ngio, "tables", None), name, None)
    print(name, "available:", fn is not None)
    if fn is None: continue
    for g in sorted((store / "tables").glob("*/measurements")) + sorted((store / "tables").glob("*")):
        attempt(f"{name}({g.relative_to(store)})", lambda g=g: repr(fn(str(g)))[:300])
