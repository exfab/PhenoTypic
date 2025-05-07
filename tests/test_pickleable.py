import inspect, pickle, importlib, pkgutil, phenotypic, pytest

def walk_package(pkg):
    """Yield (qualified_name, obj) for every public, top‑level object in *pkg*
    and all of its sub‑modules, skipping module objects themselves."""
    modules = [pkg]                           # start with the root
    if hasattr(pkg, "__path__"):              # add all sub‑modules
        modules += [
            importlib.import_module(name)
            for _, name, _ in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + ".")
        ]

    seen = set()
    for mod in modules:
        for attr in dir(mod):
            if attr.startswith("_"):
                continue
            obj = getattr(mod, attr)
            if inspect.ismodule(obj):
                continue
            qualname = f"{mod.__name__}.{attr}"
            if qualname not in seen:
                seen.add(qualname)
                yield qualname, obj

_public = list(walk_package(phenotypic))

@pytest.mark.parametrize("qualname,obj", _public)
def test_picklable(qualname, obj):
    pickle.dumps(obj)            # will fail fast on the first bad object