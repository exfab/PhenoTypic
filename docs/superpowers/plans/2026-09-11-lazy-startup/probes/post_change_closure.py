"""Static import closure under the PLANNED changes (no repo edits). Reports forbidden libs reached per root and the chain."""
import ast, sys
from collections import deque
from pathlib import Path
SRC = Path("src")
FORBIDDEN = {"scipy", "skimage", "pandas", "pyarrow", "polars", "colour", "numba", "h5py", "matplotlib", "plotly", "mahotas", "cv2", "bm3d", "dash"}
DEFERRED_LIBS = {"colour", "numba", "h5py", "mahotas", "cv2", "bm3d", "plotly", "matplotlib"}
SDK_HEAVY_SUBMODULES = {"colourspace", "hdf_", "_measurement_tables", "_metadata_migration", "mixin"}
UNTOUCHED_DEFERRAL = {"phenotypic.sdk_.hdf_", "phenotypic.sdk_.viz.figures._theme", "phenotypic._cli._cli_process_single",
                      "phenotypic.correction._color_correction._color_checker_profile", "phenotypic.correction._color_correction._helpers",
                      "phenotypic.sdk_.colourspace", "phenotypic.sdk_.reconnect._tensor_voting", "phenotypic.sdk_.branch_pathfinding._dijkstra_kernels"}
def mfile(name):
    b = SRC.joinpath(*name.split("."))
    if (b / "__init__.py").is_file(): return b / "__init__.py", True
    if b.with_suffix(".py").is_file(): return b.with_suffix(".py"), False
    return None, False
def tc(t): return (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING")
def planned_drop(name, mod, alias_names):
    """True when the planned change removes this module-level import edge."""
    top = mod.split(".")[0]
    if name == "phenotypic":
        return True  # lazy top-level __init__
    if name == "phenotypic.sdk_" and (mod.split(".")[-1] in SDK_HEAVY_SUBMODULES or any(a in SDK_HEAVY_SUBMODULES for a in alias_names)):
        return True
    if name == "phenotypic.abc_" and (mod.endswith("._prefab_pipeline") or mod.startswith("phenotypic._core._image_parts.detection_modes")):
        return True
    if name == "phenotypic._core._image_parts._grid_image_handler" and mod in ("phenotypic.grid", "phenotypic.measure"):
        return True
    if name in ("phenotypic.detect._filamentous_fungi_detector", "phenotypic.detect._two_k_filamentous_detector") and mod.startswith("phenotypic.sdk_.reconnect"):
        return True
    if name in ("phenotypic._core._image_parts.color_space_accessors._xyz_conversion",) and "colourspace" in mod:
        return True
    if top in DEFERRED_LIBS and name not in UNTOUCHED_DEFERRAL:
        return True
    return False
def imports(name):
    p, is_pkg = mfile(name)
    if p is None: return []
    tree = ast.parse(p.read_text(encoding="utf-8")); pkg = name if is_pkg else name.rpartition(".")[0]; out = []
    def walk(body):
        for n in body:
            if isinstance(n, ast.Import):
                for a in n.names:
                    if not planned_drop(name, a.name, []): out.append(a.name)
            elif isinstance(n, ast.ImportFrom):
                if n.level:
                    parts = pkg.split("."); base = ".".join(parts[: len(parts) - (n.level - 1)])
                    mod = f"{base}.{n.module}" if n.module else base
                else: mod = n.module or ""
                if mod == "__future__": continue
                names = [a.name for a in n.names]
                if planned_drop(name, mod, names): continue
                out.append(mod)
                for a in names:
                    if mod.startswith("phenotypic") and mfile(f"{mod}.{a}")[0] is not None:
                        if not planned_drop(name, f"{mod}.{a}", []): out.append(f"{mod}.{a}")
            elif isinstance(n, ast.If):
                if not tc(n.test): walk(n.body)
                walk(n.orelse)
            elif isinstance(n, ast.Try): walk(n.body); [walk(h.body) for h in n.handlers]; walk(n.orelse)
    walk(tree.body); return out
def closure(root_imports, label):
    parent, libs, q = {}, {}, deque()
    for r in root_imports:
        parts = r.split(".")
        for i in range(2, len(parts) + 1):
            t = ".".join(parts[:i])
            if t not in parent and mfile(t)[0] is not None: parent[t] = label; q.append(t)
    while q:
        cur = q.popleft()
        for imp in imports(cur):
            if imp.split(".")[0] == "phenotypic":
                parts = imp.split(".")
                for i in range(2, len(parts) + 1):
                    t = ".".join(parts[:i])
                    if t not in parent and mfile(t)[0] is not None: parent[t] = cur; q.append(t)
            elif imp.split(".")[0] in FORBIDDEN or imp in ("matplotlib.pyplot",):
                libs.setdefault(imp.split(".")[0] if imp != "matplotlib.pyplot" else "matplotlib.pyplot", cur)
    return parent, libs
def chain(parent, mod, label):
    out = []
    while mod != label and mod in parent:
        out.append(mod.removeprefix("phenotypic.")); mod = parent[mod]
    return " <- ".join(out[:6]) + (" <- …" if len(out) > 6 else "")
mode = sys.argv[1]
if mode == "cli":
    tree = ast.parse(Path("src/phenotypic/phenotypicCLI.py").read_text(encoding="utf-8"))
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("phenotypic"):
            roots = [n.module] if n.module != "phenotypic" else [f"phenotypic._core._image_pipeline"]
            parent, libs = closure(roots, "ROOT")
            shown = "; ".join(f"{k} via {chain(parent, v, 'ROOT')}" for k, v in sorted(libs.items())[:4])
            print(f"L{n.lineno:<4d} {n.module:48s} {'HEAVY: ' + ', '.join(sorted(libs)) if libs else 'light'}")
            if libs: print(f"        e.g. {shown[:300]}")
else:
    roots = sys.argv[2:]
    parent, libs = closure(roots, "ROOT")
    print(f"== roots {roots}: {len(parent)} phenotypic modules; forbidden reached: {sorted(libs) or 'none'}")
    for k, v in sorted(libs.items()):
        print(f"   {k:18s} via {chain(parent, v, 'ROOT')}")
